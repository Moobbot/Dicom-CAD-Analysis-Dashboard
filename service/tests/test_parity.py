"""Parity and determinism of the served model — run inside the service image.

    docker run --rm -v <case dir>:/case:ro -v "$PWD/service/tests:/app/service/tests:ro" \
        -e CAD_PARITY_CASE_DIR=/case cad-service \
        sh -c "pip install -q pytest && python -m pytest service/tests"

(the image does not contain the tests: .dockerignore leaves them out)

- the classifier trained twice gives identical predictions (it is re-trained at every start);
- the service's case result equals a direct loop of `modules.inference.diagnose_image` over the
  same slices in the same order (the wrapper adds ordering and aggregation, not a new model).

The case (a DICOM series) is not distributed with the code. Skipped without TensorFlow.
"""
import math
import os
import shutil
import sys
import uuid

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

pytest.importorskip("tensorflow")
pytest.importorskip("radiomics")

CASE = os.environ.get("CAD_PARITY_CASE_DIR")


def test_radiomics_stays_quiet_after_its_first_use(monkeypatch, caplog):
    # `import radiomics` sets its logger back to INFO; the first case imports it lazily
    # (modules/inference.py), after the service had set WARNING: ~3000 INFO lines per case.
    import logging

    # The service logs at INFO (modules/inference.py: basicConfig); pytest's own handlers make that
    # basicConfig a no-op here, so set the root level as the service has it (restored after the test).
    caplog.set_level(logging.INFO)
    from service import app

    # As in a fresh service process: radiomics not imported yet (importorskip above imported it;
    # monkeypatch puts the modules back afterwards).
    for name in [n for n in sys.modules if n == "radiomics" or n.startswith("radiomics.")]:
        monkeypatch.delitem(sys.modules, name)
    app._quiet_logs()  # what the service does at start (_load)
    from radiomics import featureextractor  # noqa: F401  (the lazy import of the first case)

    assert logging.getLogger("radiomics").getEffectiveLevel() >= logging.WARNING
    # pykwalify validates the feature settings once per slice: "validation.valid" at INFO.
    assert logging.getLogger("pykwalify.core").getEffectiveLevel() >= logging.WARNING


def test_classifier_training_is_deterministic():
    import numpy as np

    from modules import inference

    X, _ = inference.training_features()
    a = inference.train_classifier().predict_proba(X)
    b = inference.train_classifier().predict_proba(X)
    assert np.array_equal(a, b)


@pytest.mark.skipif(not CASE, reason="CAD_PARITY_CASE_DIR not set")
def test_service_result_equals_a_direct_diagnose_image_loop(tmp_path, monkeypatch):
    import pydicom

    from modules import inference
    from service import app as service_app
    from service import case as C

    sid = str(uuid.uuid4())
    uploads, results = tmp_path / "uploads", tmp_path / "results"
    shutil.copytree(CASE, uploads / sid)
    monkeypatch.setattr(service_app, "UPLOADS", str(uploads))
    monkeypatch.setattr(service_app, "RESULTS", str(results))
    service_app.STATE["info"] = service_app._load()

    body = service_app.predict_case(sid, "cad")

    found = C.discover_slices(str(uploads / sid), lambda p: pydicom.dcmread(p, stop_before_pixels=True))
    direct = [C.slice_outcome(inference.diagnose_image(s.path, filename="x.dcm")) for s in found.slices]
    expected = C.summarize(direct)
    values = {r["key"]: r["value"] for r in body["results"]}
    assert values["predicted_class"] == expected.label
    assert values["covid_slices"] == expected.covid
    assert values["analysed_slices"] == expected.analysed
    assert math.isclose(values["mean_covid_score"], round(expected.mean_score, 6), rel_tol=0, abs_tol=1e-9)
    assert (results / sid / "cad" / "results.gif").is_file()
    assert not (results / sid / "cad" / ".heartbeat").exists()
