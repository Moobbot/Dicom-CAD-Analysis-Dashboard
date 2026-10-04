"""Weight files of the CAD service (service/weights.py) — standard library only, no TensorFlow.

The folder is a folder of the checkout mounted into the container: whatever is in it belongs to
the installation. The rules under test: a downloaded file is installed only if it is the expected
file (SHA-256), and nothing that is already in the folder is ever replaced or removed — a Git LFS
pointer included.
"""
import hashlib
import http.server
import os
import re
import stat
import sys
import threading

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from service import weights as W  # noqa: E402

REAL = W.HDF5_SIGNATURE + b"the validated U-Net"
OTHER_REAL = W.HDF5_SIGNATURE + b"a U-Net the hospital trained itself"
POINTER = b"version https://git-lfs.github.com/spec/v1\noid sha256:0000\nsize 372567864\n"
MASKS = b"\x93NUMPY the masks"
ERROR_PAGE = b"<html>Access denied by the proxy</html>"

TRACKED = "lung_segmentation_unet .h5"  # the name in git (Git LFS), with a space
DOWNLOADED = "lung_segmentation_unet.h5"
UNET_URL = "https://example.test/unet"
MASKS_URL = "https://example.test/masks"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def specs(unet_urls=(UNET_URL,)):
    return dict(
        unet=W.Weight(name=DOWNLOADED, sha256=sha(REAL), urls=tuple(unet_urls)),
        predictions=W.Weight(name="predictions.npy", sha256=sha(MASKS), urls=(MASKS_URL,)),
    )


class Site:
    """A download site: `pages` maps an address to what it sends (bytes) or to an error to raise."""

    def __init__(self, pages=None):
        self.pages = {UNET_URL: REAL, MASKS_URL: MASKS} if pages is None else pages
        self.calls = []

    def __call__(self, url, dest):
        self.calls.append(url)
        page = self.pages[url]
        if isinstance(page, Exception):
            raise page
        with open(dest, "wb") as f:
            f.write(page)


def put(folder, **files):
    for name, data in files.items():
        (folder / name).write_bytes(data)


def snapshot(folder):
    """Name -> (content, inode, mtime): equal snapshots mean no file was replaced, rewritten or removed."""
    out = {}
    for name in sorted(os.listdir(folder)):
        st = os.stat(folder / name)
        out[name] = ((folder / name).read_bytes(), st.st_ino, st.st_mtime_ns)
    return out


def ensure(folder, site, **overrides):
    return W.ensure_weights(str(folder), fetch=site, **{**specs(), **overrides})


# --- nothing to do ---------------------------------------------------------------------------


def test_a_complete_folder_is_left_alone_and_nothing_is_downloaded(tmp_path):
    put(tmp_path, **{TRACKED: REAL, "predictions.npy": MASKS})
    before, site = snapshot(tmp_path), Site()

    status = ensure(tmp_path, site)

    assert status == {DOWNLOADED: "present", "predictions.npy": "present"}
    assert site.calls == []
    assert snapshot(tmp_path) == before


def test_weights_that_are_not_the_validated_file_are_used_as_they_are_and_reported(tmp_path):
    """Another real HDF5 file is the installation's choice (the version string records its digest):
    it is neither replaced nor downloaded over, but the log must say that it is not the validated file."""
    put(tmp_path, **{TRACKED: OTHER_REAL, "predictions.npy": MASKS})
    before, site = snapshot(tmp_path), Site()

    assert ensure(tmp_path, site)[DOWNLOADED] == "unexpected"
    assert site.calls == []
    assert snapshot(tmp_path) == before


def test_a_truncated_file_with_the_right_signature_is_reported(tmp_path, caplog):
    """An interrupted copy keeps the HDF5 signature and cannot be loaded: it stays (never replaced),
    and the log names the file and says what to do — "present" would hide the cause."""
    put(tmp_path, **{TRACKED: POINTER, DOWNLOADED: REAL[:12], "predictions.npy": MASKS})
    before, site = snapshot(tmp_path), Site()

    with caplog.at_level("WARNING"):
        status = ensure(tmp_path, site)

    assert status[DOWNLOADED] == "unexpected"
    assert site.calls == []
    assert snapshot(tmp_path) == before
    assert "lung_segmentation_unet.h5" in caplog.text and "move it away" in caplog.text


def test_the_file_the_model_will_load_is_the_one_that_is_checked(tmp_path):
    """Both names hold a real file: the model takes the tracked one, so that one decides."""
    put(tmp_path, **{TRACKED: REAL, DOWNLOADED: OTHER_REAL, "predictions.npy": MASKS})
    assert ensure(tmp_path, Site())[DOWNLOADED] == "present"
    put(tmp_path, **{TRACKED: OTHER_REAL, DOWNLOADED: REAL})
    assert ensure(tmp_path, Site())[DOWNLOADED] == "unexpected"


def test_real_weights_under_the_second_name_make_the_download_unnecessary(tmp_path):
    put(tmp_path, **{TRACKED: POINTER, DOWNLOADED: REAL, "predictions.npy": MASKS})
    before, site = snapshot(tmp_path), Site()

    assert ensure(tmp_path, site)[DOWNLOADED] == "present"
    assert site.calls == []
    assert snapshot(tmp_path) == before


# --- the Git LFS pointer ---------------------------------------------------------------------


def test_a_git_lfs_pointer_is_neither_replaced_nor_removed(tmp_path):
    """A checkout without git-lfs: the tracked name holds a pointer. The real file is downloaded
    next to it; the pointer stays as git wrote it (the checkout must not look modified)."""
    put(tmp_path, **{TRACKED: POINTER, "predictions.npy": MASKS})
    before, site = snapshot(tmp_path), Site()

    status = ensure(tmp_path, site)

    assert status[DOWNLOADED] == "downloaded"
    assert site.calls == [UNET_URL]
    after = snapshot(tmp_path)
    assert after[TRACKED] == before[TRACKED]
    assert after[DOWNLOADED][0] == REAL
    assert sorted(after) == sorted([TRACKED, DOWNLOADED, "predictions.npy"])


def test_an_empty_folder_gets_both_files(tmp_path):
    site = Site()

    status = ensure(tmp_path, site)

    assert status == {DOWNLOADED: "downloaded", "predictions.npy": "downloaded"}
    assert (tmp_path / DOWNLOADED).read_bytes() == REAL
    assert (tmp_path / "predictions.npy").read_bytes() == MASKS
    assert sorted(os.listdir(tmp_path)) == [DOWNLOADED, "predictions.npy"]


def test_the_folder_is_created_when_it_does_not_exist(tmp_path):
    folder = tmp_path / "ML" / "UNET Training"

    status = ensure(folder, Site())

    assert status[DOWNLOADED] == "downloaded"
    assert (folder / DOWNLOADED).read_bytes() == REAL


def test_an_installed_file_can_be_read_by_other_users(tmp_path):
    """The temporary file is owner-only; the installed file must be readable from the host."""
    ensure(tmp_path, Site())
    assert stat.S_IMODE(os.stat(tmp_path / DOWNLOADED).st_mode) == 0o644


# --- the checksum ----------------------------------------------------------------------------


def test_a_download_that_is_not_the_expected_file_is_not_installed(tmp_path):
    """A proxy's error page, a download cut short, a changed release asset: nothing is installed."""
    put(tmp_path, **{TRACKED: POINTER, "predictions.npy": MASKS})
    before, site = snapshot(tmp_path), Site({UNET_URL: ERROR_PAGE})

    status = ensure(tmp_path, site)

    assert status[DOWNLOADED] == "missing"
    assert snapshot(tmp_path) == before  # no file under the real name, no temporary file left


def test_a_real_hdf5_file_that_is_not_the_expected_one_is_not_installed(tmp_path):
    """The signature alone is not the check: the download must be the file the release was validated with."""
    site = Site({UNET_URL: OTHER_REAL, MASKS_URL: MASKS})

    status = ensure(tmp_path, site)

    assert status[DOWNLOADED] == "missing"
    assert not (tmp_path / DOWNLOADED).exists()


def test_the_next_address_is_tried_when_the_first_sends_something_else(tmp_path):
    second = "https://mirror.test/unet"
    site = Site({UNET_URL: ERROR_PAGE, second: REAL, MASKS_URL: MASKS})

    status = W.ensure_weights(str(tmp_path), fetch=site, **specs(unet_urls=(UNET_URL, second)))

    assert status[DOWNLOADED] == "downloaded"
    assert site.calls[:2] == [UNET_URL, second]
    assert (tmp_path / DOWNLOADED).read_bytes() == REAL


# --- never replace ---------------------------------------------------------------------------


def test_a_file_under_the_download_name_is_not_replaced(tmp_path):
    """No real weights anywhere, and the name the download would take is in use: the file stays,
    and nothing is downloaded (it could not be installed anyway)."""
    put(tmp_path, **{TRACKED: POINTER, DOWNLOADED: b"half a download from an older version", "predictions.npy": MASKS})
    before, site = snapshot(tmp_path), Site()

    status = ensure(tmp_path, site)

    assert status[DOWNLOADED] == "missing"
    assert site.calls == []
    assert snapshot(tmp_path) == before


def test_a_file_that_appears_during_the_download_is_kept(tmp_path):
    """An operator copies the weights in while the service downloads them: the operator's file wins."""

    def slow_site(url, dest):
        with open(dest, "wb") as f:
            f.write(REAL if url == UNET_URL else MASKS)
        if url == UNET_URL:
            (tmp_path / DOWNLOADED).write_bytes(OTHER_REAL)

    status = ensure(tmp_path, slow_site)

    assert status[DOWNLOADED] == "unexpected"  # kept, and reported as not the validated file
    assert (tmp_path / DOWNLOADED).read_bytes() == OTHER_REAL
    assert sorted(os.listdir(tmp_path)) == [DOWNLOADED, "predictions.npy"]


def test_existing_predictions_are_never_replaced_whatever_they_hold(tmp_path):
    put(tmp_path, **{TRACKED: REAL, "predictions.npy": b"tiny"})
    before, site = snapshot(tmp_path), Site()

    status = ensure(tmp_path, site)

    assert status["predictions.npy"] == "present"
    assert site.calls == []
    assert snapshot(tmp_path) == before


def test_a_broken_link_under_a_name_is_left_as_it_is(tmp_path):
    os.symlink(str(tmp_path / "gone"), str(tmp_path / DOWNLOADED))
    site = Site()

    status = ensure(tmp_path, site)

    assert status[DOWNLOADED] == "missing"
    assert UNET_URL not in site.calls
    assert os.path.islink(tmp_path / DOWNLOADED)


def test_a_file_system_without_hard_links_does_not_replace_a_file_that_is_there(tmp_path, monkeypatch):
    """Without hard links the install is a rename after a last look (a file created in the instant
    between the look and the rename is not covered: see _install)."""

    def no_hard_links(src, dst):
        raise OSError(95, "Operation not supported")

    monkeypatch.setattr(W.os, "link", no_hard_links)

    assert ensure(tmp_path, Site())[DOWNLOADED] == "downloaded"
    assert (tmp_path / DOWNLOADED).read_bytes() == REAL

    # and with the name taken meanwhile, the rename does not happen
    os.remove(tmp_path / DOWNLOADED)

    def racing_site(url, dest):
        with open(dest, "wb") as f:
            f.write(REAL if url == UNET_URL else MASKS)
        if url == UNET_URL:
            (tmp_path / DOWNLOADED).write_bytes(OTHER_REAL)

    assert ensure(tmp_path, racing_site)[DOWNLOADED] == "unexpected"
    assert (tmp_path / DOWNLOADED).read_bytes() == OTHER_REAL


# --- failures --------------------------------------------------------------------------------


def test_an_unreachable_site_leaves_the_folder_as_it_was(tmp_path):
    put(tmp_path, **{TRACKED: POINTER})
    before = snapshot(tmp_path)
    site = Site({UNET_URL: OSError("Network is unreachable"), MASKS_URL: OSError("Network is unreachable")})

    status = ensure(tmp_path, site)

    assert status == {DOWNLOADED: "missing", "predictions.npy": "missing"}
    assert snapshot(tmp_path) == before


def test_one_file_that_cannot_be_provided_does_not_stop_the_other(tmp_path):
    site = Site({UNET_URL: OSError("timed out"), MASKS_URL: MASKS})

    status = ensure(tmp_path, site)

    assert status == {DOWNLOADED: "missing", "predictions.npy": "downloaded"}
    assert (tmp_path / "predictions.npy").read_bytes() == MASKS


OURS = DOWNLOADED + ".k3j2_x9a.partial"  # the shape of the service's own temporary files (8 characters)
NOT_OURS = [DOWNLOADED + ".backup.partial", DOWNLOADED + ".k3j2.partial", "notes.partial"]


def test_only_the_services_own_temporary_file_of_a_killed_download_is_removed(tmp_path):
    """The one exception to "nothing is removed": the file a killed download of this service left.
    Anything else that merely looks similar belongs to the installation and stays."""
    put(tmp_path, **{OURS: b"half", **{name: b"an operator's file" for name in NOT_OURS}})

    ensure(tmp_path, Site())

    assert sorted(os.listdir(tmp_path)) == sorted([DOWNLOADED, "predictions.npy", *NOT_OURS])


def test_the_temporary_file_is_removed_even_when_the_weights_arrived_another_way(tmp_path):
    """A download was killed, then the operator ran `git lfs pull`: 373 MB must not stay forever."""
    put(tmp_path, **{TRACKED: REAL, "predictions.npy": MASKS, OURS: b"half"})
    site = Site()

    assert ensure(tmp_path, site) == {DOWNLOADED: "present", "predictions.npy": "present"}
    assert site.calls == []
    assert sorted(os.listdir(tmp_path)) == sorted([TRACKED, "predictions.npy"])


@pytest.mark.parametrize("step", ["chmod", "fsync"])
def test_a_best_effort_step_that_fails_does_not_undo_a_verified_download(tmp_path, monkeypatch, step):
    """Some mounts refuse chmod or fsync (network shares): the file is verified all the same."""

    def refuse(*args, **kwargs):
        raise OSError(95, "Operation not supported")

    monkeypatch.setattr(W.os, step, refuse)

    assert ensure(tmp_path, Site()) == {DOWNLOADED: "downloaded", "predictions.npy": "downloaded"}
    assert (tmp_path / DOWNLOADED).read_bytes() == REAL


def test_a_read_only_folder_is_reported_not_raised(tmp_path):
    if os.geteuid() == 0:
        pytest.skip("root writes into a read-only folder")
    os.chmod(tmp_path, 0o555)
    try:
        status = ensure(tmp_path, Site())
    finally:
        os.chmod(tmp_path, 0o755)
    assert status == {DOWNLOADED: "missing", "predictions.npy": "missing"}


def test_a_folder_that_cannot_be_created_is_reported_not_raised(tmp_path):
    """The service must still start and say on /health what is wrong."""
    (tmp_path / "ML").write_bytes(b"a file where the folder should be")

    status = ensure(tmp_path / "ML" / "UNET Training", Site())

    assert status == {DOWNLOADED: "missing", "predictions.npy": "missing"}


# --- what the service says -------------------------------------------------------------------


def test_summary_says_what_happened_in_one_line():
    line = W.summary({DOWNLOADED: "downloaded", "predictions.npy": "present"})
    assert line == "Weights: lung_segmentation_unet.h5 downloaded and verified; predictions.npy present."


def test_summary_says_when_a_file_is_not_the_validated_one():
    line = W.summary({DOWNLOADED: "unexpected", "predictions.npy": "present"})
    assert line == "Weights: lung_segmentation_unet.h5 present but NOT the validated file (used as it is); predictions.npy present."


def test_summary_of_a_missing_file_says_what_to_do():
    line = W.summary({DOWNLOADED: "missing", "predictions.npy": "present"})
    assert "lung_segmentation_unet.h5 MISSING" in line
    assert "git lfs pull" in line and "restart" in line


# --- download() over real HTTP ---------------------------------------------------------------


@pytest.fixture
def http_server(tmp_path):
    served = tmp_path / "served"
    served.mkdir()
    (served / "unet").write_bytes(REAL)

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(served), **kw)

        def log_message(self, *a):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def test_download_saves_the_file_and_raises_on_an_http_error(tmp_path, http_server):
    dest = tmp_path / "out"
    W.download(f"{http_server}/unet", str(dest))
    assert dest.read_bytes() == REAL
    with pytest.raises(Exception):
        W.download(f"{http_server}/gone", str(tmp_path / "out2"))


def test_ensure_weights_through_the_real_downloader(tmp_path, http_server):
    folder = tmp_path / "weights"
    spec = specs(unet_urls=(f"{http_server}/gone", f"{http_server}/unet"))

    status = W.ensure_weights(str(folder), unet=spec["unet"], predictions=W.Weight("predictions.npy", sha(MASKS), (f"{http_server}/gone",)))

    assert status == {DOWNLOADED: "downloaded", "predictions.npy": "missing"}
    assert (folder / DOWNLOADED).read_bytes() == REAL


# --- the real list, and the files that must agree with it ------------------------------------


def test_the_real_list_is_the_release_of_this_repository():
    assert W.UNET.name == DOWNLOADED and W.PREDICTIONS.name == "predictions.npy"
    assert W.UNET_NAMES == (TRACKED, DOWNLOADED)
    for w in (W.UNET, W.PREDICTIONS):
        assert re.fullmatch(r"[0-9a-f]{64}", w.sha256), w.name
        assert w.urls == (f"https://github.com/Moobbot/Dicom-CAD-Analysis-Dashboard/releases/download/v-0.1/{w.name}",)


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_the_tracked_weights_are_the_file_the_service_would_download():
    """The Git LFS object of the checkout and the release asset must be the same file."""
    lfs = os.path.join(ROOT, "ML", "UNET Training", TRACKED)
    if not os.path.isfile(lfs):
        pytest.skip("the weights folder is not in the image; run this test in the repository")
    with open(lfs, "rb") as f:
        head = f.read(200)
    if head.startswith(W.HDF5_SIGNATURE):
        assert W.file_sha256(lfs) == W.UNET.sha256
    else:  # a checkout without git-lfs: the pointer names the object
        assert f"oid sha256:{W.UNET.sha256}".encode() in head


def test_a_downloaded_file_does_not_make_the_checkout_look_modified():
    path = os.path.join(ROOT, ".gitignore")
    if not os.path.isfile(path):
        pytest.skip(".gitignore is not in the image; run this test in the repository")
    with open(path, encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f]
    assert "/ML/UNET Training/lung_segmentation_unet.h5" in lines
    assert "/ML/UNET Training/*.partial" in lines
