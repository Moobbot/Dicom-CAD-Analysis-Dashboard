"""Case layer of the CAD service (service/case.py) — no TensorFlow, no pydicom needed."""
import os
import sys
import time
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from service import case as C  # noqa: E402


def header(z, inst=None, series="1.2.3", modality="CT", frames=None):
    return SimpleNamespace(ImagePositionPatient=[0, 0, z], InstanceNumber=inst, SeriesInstanceUID=series, Modality=modality, NumberOfFrames=frames)


def make_case(tmp_path, headers):
    """Files named so that name order != z order; `headers` maps file name -> header or None (not DICOM)."""
    for name in headers:
        (tmp_path / name).write_bytes(b"x")
    return lambda p: _read(headers[os.path.basename(p)])


def _read(h):
    if h is None:
        raise ValueError("not DICOM")
    return h


# ---------------------------------------------------------------- discovery


def test_slices_in_z_order_recognised_by_content_not_suffix(tmp_path):
    read = make_case(tmp_path, {"img_00001.dcm": header(3.0), "IMG_00002.DCM": header(1.0), "img_00003": header(2.0), "README": None, ".hidden": header(9.0)})
    got = [os.path.basename(s.path) for s in C.discover_slices(str(tmp_path), read)]
    assert got == ["IMG_00002.DCM", "img_00003", "img_00001.dcm"]


@pytest.mark.parametrize(
    "headers,message",
    [
        ({"a.dcm": header(1.0, modality="MR")}, "CT only"),
        ({"a.dcm": header(1.0, frames="12")}, "multi-frame"),
        ({"a.dcm": header(1.0, series="1"), "b.dcm": header(2.0, series="2")}, "2 DICOM series"),
        ({"a.dcm": header(1.0, 4), "b.dcm": header(1.0, 4)}, "duplicate"),
        ({"README": None}, "No DICOM image"),
        ({"a.dcm": SimpleNamespace(ImagePositionPatient=None, Modality="CT")}, "cannot be ordered"),
    ],
)
def test_ambiguous_or_unsupported_cases_are_refused_with_a_reason(tmp_path, headers, message):
    read = make_case(tmp_path, headers)
    with pytest.raises(C.CaseError, match=message) as e:
        C.discover_slices(str(tmp_path), read)
    assert e.value.status == 400


def test_same_z_ordered_by_instance_number(tmp_path):
    read = make_case(tmp_path, {"b.dcm": header(1.0, 2), "a.dcm": header(1.0, 1), "c.dcm": header(1.0, "x")})
    assert [os.path.basename(s.path) for s in C.discover_slices(str(tmp_path), read)] == ["a.dcm", "b.dcm", "c.dcm"]


# ---------------------------------------------------------------- slices -> case


def result(p, fallbacks=(), features=None):
    return {"probability_covid": p, "fallbacks": list(fallbacks), "features": features or {"f": 1.0}}


def test_a_slice_with_any_fallback_or_non_finite_feature_is_not_analysed():
    assert C.slice_outcome(result(0.7)).covid_probability == 0.7
    assert C.slice_outcome(result(0.7, ["empty_mask"])).reasons == ["empty_mask"]
    assert C.slice_outcome(result(0.7, features={"f": float("nan")})).reasons == ["nonfinite_features"]
    assert C.slice_outcome(None).reasons == ["slice_error"]
    assert C.slice_outcome(result(float("nan"))).covid_probability is None


def test_case_rule_any_positive_slice_means_covid():
    s = C.summarize([C.slice_outcome(result(p)) for p in (0.2, 0.9, 0.1)])
    assert (s.label, s.covid, s.analysed, s.total) == ("COVID-19", 1, 3, 3)
    assert s.mean_score == pytest.approx(0.4)


def test_all_analysed_and_none_positive_is_normal():
    assert C.summarize([C.slice_outcome(result(p)) for p in (0.2, 0.5, 0.1)]).label == "Normal"  # 0.5 is not > 0.5


def test_regression_review_x3_partial_coverage_never_concludes_normal():
    # Codex review X3: excluding failed slices could manufacture a "Normal" case.
    outcomes = [C.slice_outcome(result(0.1)), C.slice_outcome(result(0.95, ["segmentation_error"])), C.slice_outcome(None)]
    s = C.summarize(outcomes)
    assert s.label == "Indeterminate"
    assert s.left_out == {"segmentation_error": 1, "slice_error": 1}
    # ...but a positive analysed slice is still COVID-19 under the original rule
    assert C.summarize(outcomes + [C.slice_outcome(result(0.8))]).label == "COVID-19"


def test_no_analysable_slice_is_an_error_not_a_result():
    with pytest.raises(C.CaseError) as e:
        C.summarize([C.slice_outcome(result(0.9, ["radiomics_error"]))])
    assert e.value.status == 422


# ---------------------------------------------------------------- contract body


def test_contract_response_shape_and_limits():
    s = C.summarize([C.slice_outcome(result(0.1)), C.slice_outcome(result(0.9, ["empty_mask"]))])
    body = C.contract_response(s, "cad@src.1+w.2+clf.3", "results.gif")
    assert body["result_contract"] == 1
    kinds = {r["key"]: r["kind"] for r in body["results"]}
    assert kinds == {"predicted_class": "label", "mean_covid_score": "score", "covid_slices": "measurement", "analysed_slices": "measurement"}
    assert all(r.get("unit") for r in body["results"] if r["kind"] == "measurement")
    assert body["artifacts"] == [{"kind": "gif", "path": "results.gif", "label": "Lung segmentation overlay"}]
    assert body["warnings"][0].startswith("1 of 2 slices were not analysed: no lung was found (1).")
    assert "no conclusion" in body["warnings"][1]
    assert all(len(w) <= 300 for w in body["warnings"] + body["notes"])
    assert all(len(r.get("hint", "")) <= 300 and len(r["label"]) <= 120 for r in body["results"])


def test_no_gif_means_no_artifact():
    s = C.summarize([C.slice_outcome(result(0.1))])
    assert C.contract_response(s, "v", None)["artifacts"] == []


# ---------------------------------------------------------------- outputs


def test_gif_frames_are_capped_and_spread():
    assert C.gif_frame_indices(10) == list(range(10))
    idx = C.gif_frame_indices(291)
    assert len(idx) == 60 and idx[0] == 0 and idx[-1] == 290


def test_gif_has_no_file_names(tmp_path):
    np = pytest.importorskip("numpy")
    pytest.importorskip("PIL")
    overlays = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(3)]
    path = tmp_path / "results.gif"
    C.write_gif(str(path), overlays, ["slice 1/3", "slice 2/3", "slice 3/3"])
    assert path.read_bytes()[:6] == b"GIF89a"


def test_heartbeat_is_refreshed_while_running_and_removed_after(tmp_path):
    with C.Heartbeat(str(tmp_path), every=0.05) as hb:
        first = os.stat(hb.path).st_mtime_ns
        time.sleep(0.25)
        assert os.stat(hb.path).st_mtime_ns > first
    assert not os.path.exists(os.path.join(tmp_path, ".heartbeat"))


@pytest.mark.parametrize("sid,ok", [("3f2b8c1e-4a5d-4e6f-8a9b-0c1d2e3f4a5b", True), ("../etc", False), ("3F2B8C1E-4A5D-4E6F-8A9B-0C1D2E3F4A5B", False)])
def test_session_id_validation(sid, ok):
    assert bool(C.SESSION_ID.match(sid)) is ok


@pytest.mark.parametrize("out,ok", [("cad", True), ("cad_covid", True), ("../sybil", False), ("Cad", False), ("", False)])
def test_output_dir_validation(out, ok):
    assert bool(C.OUTPUT_DIR.match(out)) is ok
