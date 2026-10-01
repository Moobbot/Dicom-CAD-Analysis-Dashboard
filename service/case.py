"""One CT case through the CAD model, as the diagnosis app's result contract v1 expects.

No TensorFlow here (unit-tested without it): reading and ordering the slices, turning the
per-slice results of `modules.inference.diagnose_image` into ONE case result, the GIF, the
heartbeat that tells the app the case is still running, and the contract response.

The model itself scores 2D slices. The case rule is the original one (predict.py): the case is
COVID-19 if any slice is. What this layer adds:
- a slice where the pipeline fell back (heuristic mask, fake features, missing features) is NOT
  analysed — its score would not come from the model — and the case says how many were left out;
- with slices left out and no positive slice, the case is "Indeterminate", never "Normal":
  a negative for the whole case cannot be concluded from partial coverage;
- nothing identifying leaves the service (no PatientID, no file names in outputs).
"""
import math
import os
import re
import threading
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

# Use with .fullmatch(): `$` would also accept a trailing newline.
SESSION_ID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
OUTPUT_DIR = re.compile(r"[a-z][a-z0-9_]{0,31}")
# DICOM objects that are not images: skipped, like non-DICOM files.
NON_IMAGE_SOP_CLASSES = {"1.2.840.10008.1.3.10"}  # Media Storage Directory (DICOMDIR)
NON_IMAGE_MODALITIES = {"SR", "KO", "PR", "REG", "SEG", "RTSTRUCT", "DOC"}
MAX_GIF_FRAMES = 60
HEARTBEAT_SECONDS = 30

RESULT_CONTRACT = 1


class CaseError(Exception):
    """The case cannot be scored; `status` is the HTTP status to answer with."""

    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


# ---------------------------------------------------------------- slices


@dataclass(frozen=True)
class Slice:
    path: str
    z: float
    instance: Optional[int]


@dataclass
class CaseSlices:
    slices: List[Slice]
    unreadable: int = 0  # .dcm files whose header could not be read (counted as not analysed)


def _int_or_none(value) -> Optional[int]:
    try:
        return int(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _is_image(h) -> bool:
    sop = str(getattr(h, "SOPClassUID", "") or getattr(getattr(h, "file_meta", None), "MediaStorageSOPClassUID", "") or "")
    modality = str(getattr(h, "Modality", "") or "")
    return sop not in NON_IMAGE_SOP_CLASSES and modality not in NON_IMAGE_MODALITIES and getattr(h, "Rows", None) is not None


def discover_slices(upload_dir: str, read_header: Callable[[str], object]) -> CaseSlices:
    """The CT slices of the case in z order.

    Files are recognised by CONTENT (`read_header` raises on a non-DICOM file), not by suffix:
    the app keeps `.dcm` names but an upload may hold `.DCM` or extensionless DICOM files.
    Skipped: files that are not DICOM and DICOM objects that are not images (DICOMDIR, reports).
    A `.dcm` file that cannot be read is counted as a slice that could not be analysed — it may be
    part of the lungs. Refused, with the reason: no image, a non-CT modality, a multi-frame file,
    several series, two slices at the same position and number.
    """
    names = sorted(n for n in os.listdir(upload_dir) if not n.startswith("."))
    slices: List[Slice] = []
    unreadable = 0
    series = set()
    for name in names:
        path = os.path.join(upload_dir, name)
        if not os.path.isfile(path) or name.lower().endswith((".zip", ".png")):
            continue
        try:
            h = read_header(path)
        except Exception:
            if name.lower().endswith(".dcm"):
                unreadable += 1
            continue  # not DICOM (README, residue...)
        if not _is_image(h):
            continue
        modality = str(getattr(h, "Modality", "") or "")
        if modality and modality != "CT":
            raise CaseError(f"The case holds {modality} images; this model reads CT only.")
        frames = _int_or_none(getattr(h, "NumberOfFrames", None))
        if frames and frames > 1:
            raise CaseError("The case holds a multi-frame DICOM file; upload one image per slice.")
        position = getattr(h, "ImagePositionPatient", None)
        if not position or len(position) < 3:
            raise CaseError("A DICOM slice has no ImagePositionPatient; the slices cannot be ordered.")
        uid = getattr(h, "SeriesInstanceUID", None)
        if uid:
            series.add(str(uid))
        slices.append(Slice(path, float(position[2]), _int_or_none(getattr(h, "InstanceNumber", None))))
    if not slices:
        raise CaseError("No DICOM image found in the case.")
    if len(series) > 1:
        raise CaseError(f"The case holds {len(series)} DICOM series; upload one series at a time.")
    ordered = sorted(slices, key=lambda s: (s.z, s.instance if s.instance is not None else math.inf))
    for a, b in zip(ordered, ordered[1:]):
        if a.z == b.z and a.instance == b.instance:
            raise CaseError("Two slices have the same position and number (duplicate files?).")
    return CaseSlices(ordered, unreadable)


# ---------------------------------------------------------------- aggregation

FALLBACK_TEXT = {
    "segmentation_unavailable": "the segmentation model was not available",
    "segmentation_error": "lung segmentation failed",
    "empty_mask": "no lung was found",
    "params_missing": "the feature settings were missing",
    "feature_missing": "some features could not be computed",
    "radiomics_unavailable": "the feature library was not available",
    "radiomics_error": "feature extraction failed",
    "nonfinite_features": "features were not finite",
    "slice_error": "the slice could not be read",
}


@dataclass
class SliceOutcome:
    covid_probability: Optional[float]  # None = not analysed
    reasons: List[str] = field(default_factory=list)


def slice_outcome(result: Optional[dict]) -> SliceOutcome:
    """A diagnose_image() result, or None when it raised."""
    if result is None:
        return SliceOutcome(None, ["slice_error"])
    reasons = list(dict.fromkeys(result.get("fallbacks") or []))
    features = result.get("features") or {}
    if any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in features.values()):
        reasons.append("nonfinite_features")
    p = result.get("probability_covid")
    if reasons or not isinstance(p, (int, float)) or not math.isfinite(p):
        return SliceOutcome(None, reasons or ["slice_error"])
    return SliceOutcome(float(p))


@dataclass
class CaseSummary:
    total: int
    analysed: int
    covid: int
    label: str
    mean_score: float
    left_out: Dict[str, int]


def summarize(outcomes: Sequence[SliceOutcome], threshold: float = 0.5) -> CaseSummary:
    """The case result. A slice counts as COVID-19 when its COVID-19 probability wins (> 0.5),
    as `diagnose_image` labels it (argmax of the two classes)."""
    analysed = [o.covid_probability for o in outcomes if o.covid_probability is not None]
    left_out: Dict[str, int] = {}
    for o in outcomes:
        for r in o.reasons:
            left_out[r] = left_out.get(r, 0) + 1
    if not analysed:
        why = "; ".join(f"{FALLBACK_TEXT.get(k, k)} ({n})" for k, n in sorted(left_out.items()))
        raise CaseError(f"No slice could be analysed without a fallback: {why}."[:300], 422)
    covid = sum(1 for p in analysed if p > threshold)
    if covid > 0:
        label = "COVID-19"
    elif len(analysed) < len(outcomes):
        label = "Indeterminate"
    else:
        label = "Normal"
    return CaseSummary(len(outcomes), len(analysed), covid, label, sum(analysed) / len(analysed), left_out)


def contract_response(summary: CaseSummary, version: str, gif_name: Optional[str]) -> dict:
    """The result contract v1 body (see the diagnosis app, modules/diagnosis/contract-result.ts)."""
    warnings = []
    not_analysed = summary.total - summary.analysed
    if not_analysed:
        why = "; ".join(f"{FALLBACK_TEXT.get(k, k)} ({n})" for k, n in sorted(summary.left_out.items()))
        warnings.append(f"{not_analysed} of {summary.total} slices were not analysed: {why}."[:300])
    if summary.label == "Indeterminate":
        warnings.append("No analysed slice was classified COVID-19, but not every slice could be analysed: no conclusion for the case.")
    return {
        "result_contract": RESULT_CONTRACT,
        "model_version": version,
        "results": [
            {
                "key": "predicted_class",
                "label": "Predicted class",
                "kind": "label",
                "value": summary.label,
                "hint": "COVID-19 if any analysed slice is classified COVID-19 (rule of the original model).",
            },
            {
                "key": "mean_covid_score",
                "label": "Mean COVID-19 score",
                "kind": "score",
                "value": round(summary.mean_score, 6),
                "hint": "Mean random-forest vote share for COVID-19 across analysed slices; not a calibrated probability.",
            },
            {"key": "covid_slices", "label": "Slices classified COVID-19", "kind": "measurement", "value": summary.covid, "unit": "slices"},
            {"key": "analysed_slices", "label": "Analysed slices", "kind": "measurement", "value": summary.analysed, "unit": "slices"},
        ],
        "artifacts": [{"kind": "gif", "path": gif_name, "label": "Lung segmentation overlay"}] if gif_name else [],
        "warnings": warnings,
        "notes": [f"{summary.covid} of {summary.analysed} analysed slices classified as COVID-19."],
    }


# ---------------------------------------------------------------- outputs


def gif_frame_indices(n: int, limit: int = MAX_GIF_FRAMES) -> List[int]:
    """At most `limit` evenly spread slice indices."""
    if n <= limit:
        return list(range(n))
    return sorted({round(i * (n - 1) / (limit - 1)) for i in range(limit)})


def write_gif(path: str, overlays: Sequence, labels: Sequence[str], duration_ms: int = 200) -> None:
    """Animated GIF of overlay frames, each labelled with its position only (no file name)."""
    from PIL import Image, ImageDraw

    frames = []
    for overlay, label in zip(overlays, labels):
        im = Image.fromarray(overlay).convert("RGB").resize((384, 384))
        ImageDraw.Draw(im).text((8, 8), label, fill=(255, 255, 255))
        frames.append(im)
    if not frames:
        raise CaseError("No slice to draw.", 422)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=duration_ms, loop=0)


class Heartbeat:
    """Touches <outdir>/.heartbeat while the case runs: the app's clean-up job looks at file
    activity in results/<session>/ and must not delete a case that is still being scored."""

    def __init__(self, outdir: str, every: float = HEARTBEAT_SECONDS):
        self.path = os.path.join(outdir, ".heartbeat")
        self.every = every
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _beat(self):
        with open(self.path, "a"):
            pass
        os.utime(self.path)

    def _run(self):
        while not self._stop.wait(self.every):
            self._beat()

    def __enter__(self):
        self._beat()
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=5)
        try:
            os.remove(self.path)
        except OSError:
            pass
        return False
