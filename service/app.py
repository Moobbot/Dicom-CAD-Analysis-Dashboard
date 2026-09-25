"""CAD-COVID model service for the diagnosis app (result contract v1).

    uvicorn service.app:app --host 0.0.0.0 --port 5557      (ONE worker: see inference_gate)

GET  /health       — answers while a case runs; 503 while the model is not usable
GET  /info         — model identity and version
POST /api_predict  — {session_id, output_dir}: scores uploads/<session_id>/ and writes
                     results/<session_id>/<output_dir>/ (see service/case.py)

RESEARCH MODEL: the pipeline has not been validated for clinical use (see README, "Validation").
The diagnosis app keeps it disabled until the operator enables it.

Strict start (no silent fallback): the U-Net weights must be a real HDF5 file (not a Git LFS
pointer), TensorFlow, SimpleITK, PyRadiomics and the feature settings must be present — otherwise
/health says why and every case is refused. The classifier is trained in memory at every start
(deterministic, a few seconds) and identified by a fingerprint of its predictions.
"""
import hashlib
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from service import case as C
from service.inference_gate import serialized, start_watchdog, status as gate_status
from service.model_info import build_info, source_digest

REPO = Path(__file__).resolve().parent.parent
UPLOADS = os.environ.get("UPLOAD_FOLDER", "/app/uploads")
RESULTS = os.environ.get("RESULTS_FOLDER", "/app/results")
HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"

log = logging.getLogger("cad-service")
STATE = {"loaded": False, "reason": "the model is starting", "info": None}


def _load() -> dict:
    import numpy as np
    import tensorflow as tf

    # CPU only in this release: TensorFlow would take most of the GPU memory that Sybil and CVD
    # share. (The image installs tensorflow-cpu; this is a second guard.)
    tf.config.set_visible_devices([], "GPU")

    from modules import inference

    logging.getLogger("CADInference").setLevel(logging.WARNING)  # no per-slice chatter
    params = REPO / "Analysis Dashboard" / "params.yaml"
    if not params.is_file():
        raise RuntimeError("the feature settings (Analysis Dashboard/params.yaml) are missing")
    try:
        import SimpleITK  # noqa: F401
        import radiomics  # noqa: F401
    except ImportError as e:
        raise RuntimeError(f"a feature library is missing ({e.name})") from e

    unet = inference.get_unet_model_path()
    if not unet.is_file():
        raise RuntimeError("the U-Net weights are missing")
    with open(unet, "rb") as f:
        if f.read(8) != HDF5_SIGNATURE:
            raise RuntimeError("the U-Net weights are not an HDF5 file (a Git LFS pointer? run `git lfs pull`)")
    if inference.load_segmentation_model() is None:
        raise RuntimeError("the U-Net could not be loaded")

    pipeline = inference.train_classifier()
    inference._CLASSIFIER_PIPELINE = pipeline  # diagnose_image uses it; nothing cached on disk
    X, _ = inference.training_features()
    fingerprint = hashlib.sha256(np.round(pipeline.predict_proba(X), 8).tobytes()).hexdigest()[:12]

    code = source_digest([str(REPO / "modules" / "inference.py"), str(REPO / "service" / "app.py"), str(REPO / "service" / "case.py"), str(params)])
    csvs = [str(REPO / "Analysis Dashboard" / n) for n in ("extracted_features_Covid.csv", "extracted_features_normal.csv")]
    info = build_info("cad", f"src.{code[:8]}" if code else "src.unknown", [str(unet), *csvs], [f"clf.{fingerprint}"], "cpu")
    info["result_contract"] = C.RESULT_CONTRACT
    info["research_model"] = True
    return info


@asynccontextmanager
async def lifespan(_app: FastAPI):
    try:
        STATE["info"] = await run_in_threadpool(_load)
        STATE.update(loaded=True, reason=None)
        log.info("[model_info] %s", STATE["info"]["version"])
    except Exception as e:  # the service stays up to report WHY on /health
        STATE.update(loaded=False, reason=str(e))
        log.error("CAD model not loaded: %s", e)
    start_watchdog()
    yield


app = FastAPI(title="CAD-COVID model service", lifespan=lifespan)


@app.get("/health")
def health():
    st = gate_status()
    healthy = STATE["loaded"] and not st["stuck"]
    body = {"status": "ok" if healthy else "unhealthy", "model_loaded": STATE["loaded"], **st}
    if not STATE["loaded"]:
        body["reason"] = STATE["reason"]
    return JSONResponse(body, status_code=200 if healthy else 503)


@app.get("/info")
def info():
    if not STATE["loaded"]:
        return JSONResponse({"model": "cad", "loaded": False, "version": None, "reason": STATE["reason"]}, status_code=503)
    return STATE["info"]


@serialized
def predict_case(session_id: str, output_dir: str) -> dict:
    import pydicom

    from modules import inference

    outdir = os.path.join(RESULTS, session_id, output_dir)
    os.makedirs(outdir, exist_ok=True)  # created at once: the app sees the case as active
    with C.Heartbeat(outdir):
        slices = C.discover_slices(
            os.path.join(UPLOADS, session_id),
            lambda p: pydicom.dcmread(p, stop_before_pixels=True),
        )
        frame_at = set(C.gif_frame_indices(len(slices)))
        outcomes, overlays, labels = [], [], []
        for i, s in enumerate(slices):
            try:
                # A neutral name: diagnose_image logs it and returns it.
                r = inference.diagnose_image(s.path, filename=f"slice_{i + 1}.dcm")
            except Exception as e:
                log.warning("slice %d/%d could not be scored (%s)", i + 1, len(slices), type(e).__name__)
                r = None
            outcome = C.slice_outcome(r)
            outcomes.append(outcome)
            if i in frame_at and r is not None:
                overlays.append(r["overlay"])
                labels.append(f"slice {i + 1}/{len(slices)}" + ("" if outcome.covid_probability is not None else " (not analysed)"))
        summary = C.summarize(outcomes)
        gif = None
        if overlays:
            C.write_gif(os.path.join(outdir, "results.gif"), overlays, labels)
            gif = "results.gif"
        return C.contract_response(summary, STATE["info"]["version"], gif)


@app.post("/api_predict")
async def api_predict(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "The body must be JSON"}, status_code=400)
    session_id = body.get("session_id") if isinstance(body, dict) else None
    output_dir = body.get("output_dir", "cad") if isinstance(body, dict) else None
    if not isinstance(session_id, str) or not C.SESSION_ID.match(session_id):
        return JSONResponse({"error": "Invalid session_id"}, status_code=400)
    if not isinstance(output_dir, str) or not C.OUTPUT_DIR.match(output_dir):
        return JSONResponse({"error": "Invalid output_dir"}, status_code=400)
    if not STATE["loaded"]:
        return JSONResponse({"error": f"The CAD model is not loaded: {STATE['reason']}"}, status_code=503)
    if not os.path.isdir(os.path.join(UPLOADS, session_id)):
        return JSONResponse({"error": "Session folder not found"}, status_code=404)
    try:
        return await run_in_threadpool(predict_case, session_id, output_dir)
    except C.CaseError as e:
        return JSONResponse({"error": str(e)}, status_code=e.status)
