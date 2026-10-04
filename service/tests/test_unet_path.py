"""Which U-Net file the model loads (modules/inference.get_unet_model_path).

The weights are tracked under a name with a space ("lung_segmentation_unet .h5", Git LFS); a
checkout without git-lfs has a ~130-byte pointer there, and the real file may sit next to it under
the name without a space. The pointer must not hide the real file.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

inference = pytest.importorskip("modules.inference", reason="the model's libraries (numpy, pandas, Pillow) are not installed")

HDF5 = b"\x89HDF\r\n\x1a\n" + b"weights"
POINTER = b"version https://git-lfs.github.com/spec/v1\noid sha256:0000\nsize 372567864\n"
TRACKED = "lung_segmentation_unet .h5"
SECOND = "lung_segmentation_unet.h5"


@pytest.fixture
def folder(tmp_path, monkeypatch):
    monkeypatch.setattr(inference, "BASE_DIR", tmp_path)
    weights = tmp_path / "ML" / "UNET Training"
    weights.mkdir(parents=True)
    return weights


def test_a_pointer_under_the_first_name_does_not_hide_the_real_file(folder):
    (folder / TRACKED).write_bytes(POINTER)
    (folder / SECOND).write_bytes(HDF5)
    assert inference.get_unet_model_path() == folder / SECOND


def test_the_tracked_name_is_still_preferred_when_both_are_real(folder):
    (folder / TRACKED).write_bytes(HDF5)
    (folder / SECOND).write_bytes(HDF5)
    assert inference.get_unet_model_path() == folder / TRACKED


def test_only_a_pointer_is_returned_so_that_the_caller_can_say_what_is_wrong(folder):
    (folder / TRACKED).write_bytes(POINTER)
    assert inference.get_unet_model_path() == folder / TRACKED


def test_a_real_file_next_to_the_code_is_found_behind_two_pointers(folder, tmp_path):
    (folder / TRACKED).write_bytes(POINTER)
    (folder / SECOND).write_bytes(POINTER)
    (tmp_path / SECOND).write_bytes(HDF5)
    assert inference.get_unet_model_path() == tmp_path / SECOND


def test_no_file_at_all_names_the_tracked_file(folder):
    assert inference.get_unet_model_path() == folder / TRACKED


def test_the_service_and_the_model_agree_on_the_two_names(folder):
    """service/weights.py decides whether weights are there from the names in UNET_NAMES; the model
    must look for the same names, in the same order."""
    from service import weights as W

    assert W.UNET_NAMES == (TRACKED, SECOND)
    for name in reversed(W.UNET_NAMES):
        (folder / name).write_bytes(HDF5)
        assert inference.get_unet_model_path() == folder / name
