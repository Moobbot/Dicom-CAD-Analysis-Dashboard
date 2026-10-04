"""The weight files of the CAD service, and how the service gets them.

The model reads its U-Net from ML/UNET Training/. In the diagnosis app that folder is a folder of
the checkout mounted into the container: whatever is in it belongs to the installation. git
tracks the U-Net there as "lung_segmentation_unet .h5" (with a space) through Git LFS, so a
checkout without git-lfs has a ~130-byte pointer under that name instead of the 373 MB file.

At start-up ensure_weights() provides what is missing, under two rules:

- a downloaded file is installed only if it is the file the service was validated with (SHA-256):
  a proxy's error page, a download cut short or a changed release asset is not installed;
- nothing that is already in the folder is ever replaced or removed. The U-Net is downloaded
  under the name WITHOUT a space, next to the pointer, which stays as git wrote it (the checkout
  does not look modified; the downloaded name is in .gitignore). The model picks the first of
  the two names that holds a real HDF5 file (modules/inference.get_unet_model_path). The one
  exception: the temporary file a killed download of this service left behind
  (`<name>.<8 characters>.partial`) is removed at the next start.

Standard library only: the tests run without TensorFlow. Kept out of the files the version string
is computed from (see app._load): how a file got there does not change the model.
"""
import hashlib
import logging
import os
import re
import shutil
import tempfile
import urllib.request
from typing import Callable, Dict, NamedTuple, Optional, Tuple

HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"
_CHUNK = 1024 * 1024
_TIMEOUT_SECONDS = 60


class Weight(NamedTuple):
    name: str  # the name a download is installed under
    sha256: str  # of the file the service was validated with
    urls: Tuple[str, ...]  # tried in order


_RELEASE = "https://github.com/Moobbot/Dicom-CAD-Analysis-Dashboard/releases/download/v-0.1/"

UNET = Weight(
    name="lung_segmentation_unet.h5",
    sha256="8d32d0c430d1077c82d1612eb332d3a34687f1aff8aa842e5a704050c41c0748",  # = the Git LFS object
    urls=(_RELEASE + "lung_segmentation_unet.h5",),
)
# The masks the U-Net training notebook saved. The service does not read them; they are provided
# so that a folder filled by the service holds what the folder of a full checkout holds.
PREDICTIONS = Weight(
    name="predictions.npy",
    sha256="d36c26b5c0adeff7532940607b75bbed8984fda2c73f4b68cf8e5f3ac271302f",
    urls=(_RELEASE + "predictions.npy",),
)
# The names the model looks for, in its order: the tracked one (with a space), then the download's.
UNET_NAMES = ("lung_segmentation_unet .h5", UNET.name)

Fetch = Callable[[str, str], None]
_log = logging.getLogger("cad-service")


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(_CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def is_hdf5(path: str) -> bool:
    """A real HDF5 file, as opposed to a Git LFS pointer, an error page or an empty file."""
    try:
        with open(path, "rb") as f:
            return f.read(len(HDF5_SIGNATURE)) == HDF5_SIGNATURE
    except OSError:
        return False


def download(url: str, dest: str) -> None:
    """Save `url` to `dest`. Raises on any network or HTTP error."""
    request = urllib.request.Request(url, headers={"User-Agent": "cad-covid-service"})
    with urllib.request.urlopen(request, timeout=_TIMEOUT_SECONDS) as response, open(dest, "wb") as out:
        shutil.copyfileobj(response, out, _CHUNK)


def _remove(path: str) -> None:
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


def _remove_stale_partials(dest: str) -> None:
    """The temporary file of a download that was killed half-way. Ours only: exactly the shape
    tempfile.mkstemp gives them below (`<name>.<8 characters>.partial`), so that a file an operator
    named `<name>.backup.partial` is not taken for one."""
    folder, name = os.path.split(dest)
    ours = re.compile(re.escape(name) + r"\.[a-z0-9_]{8}\.partial")
    try:
        entries = os.listdir(folder)
    except OSError:
        return
    for entry in entries:
        if ours.fullmatch(entry):
            try:
                _remove(os.path.join(folder, entry))
            except OSError:
                pass


def _sync(path: str) -> None:
    """Flush a file (or a folder entry) to disk: a power cut must not leave an empty file under the
    real name. Best effort: some mounts (network shares) refuse it, and the file is verified anyway."""
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


def _install(write: Callable[[str], None], dest: str, sha256: str) -> str:
    """Write to a temporary file of its own next to `dest`, and give it the real name only if it is
    the expected file AND the name is still free. Returns:

    installed  the file is now under the real name
    different  what was written is not the expected file: nothing installed
    exists     something took the real name meanwhile (e.g. an operator copied a file in): left alone

    Never replaces or removes anything under the real name. Raises what `write` raises.
    """
    folder, name = os.path.split(dest)
    fd, partial = tempfile.mkstemp(prefix=name + ".", suffix=".partial", dir=folder)
    os.close(fd)
    try:
        write(partial)
        if file_sha256(partial) != sha256:
            return "different"
        try:
            os.chmod(partial, 0o644)  # mkstemp makes it owner-only
        except OSError:
            pass  # a mount that has no modes: the file is verified all the same
        _sync(partial)
        try:
            os.link(partial, dest)  # atomic, and fails when the name is taken
        except FileExistsError:
            return "exists"
        except OSError:
            # A file system without hard links: rename after a last look (not atomic there).
            if os.path.lexists(dest):
                return "exists"
            os.rename(partial, dest)
        _sync(folder)
        return "installed"
    finally:
        _remove(partial)


def _download(spec: Weight, dest: str, fetch: Fetch, log: logging.Logger) -> str:
    """'downloaded', 'exists' (the name was taken meanwhile), or 'missing'."""
    for url in spec.urls:
        host = url.split("/")[2]
        try:
            log.warning("%s is not in the weights folder: downloading it from %s ...", spec.name, host)
            result = _install(lambda partial, url=url: fetch(url, partial), dest, spec.sha256)
            if result == "installed":
                log.warning("%s: downloaded and verified (SHA-256).", spec.name)
                return "downloaded"
            if result == "exists":
                return "exists"
            log.error("What %s sent is not %s (a download cut short, or an error page): not installed.", host, spec.name)
        except Exception as e:  # network, HTTP, disk: try the next address
            log.warning("Could not download %s from %s: %s", spec.name, host, e)
    return "missing"


def _unet_in_use(folder: str) -> Optional[str]:
    """The file the model will load: the first of its two names that holds a real HDF5 file."""
    for name in UNET_NAMES:
        path = os.path.join(folder, name)
        if is_hdf5(path):
            return path
    return None


def _existing_unet(path: str, spec: Weight, log: logging.Logger) -> str:
    """A real HDF5 file is used as it is, whatever it holds — but the log says when it is not the
    validated file: a copy that was cut short keeps the HDF5 signature and cannot be loaded."""
    if file_sha256(path) == spec.sha256:
        return "present"
    log.warning(
        '"%s" in the weights folder is not the file this service was validated with (another U-Net, '
        "or a copy that was cut short). It is used as it is; if the model cannot load it, move it away "
        "and restart the service: the validated file is then downloaded.",
        os.path.basename(path),
    )
    return "unexpected"


def _ensure_unet(folder: str, spec: Weight, fetch: Fetch, log: logging.Logger) -> str:
    dest = os.path.join(folder, spec.name)
    _remove_stale_partials(dest)
    in_use = _unet_in_use(folder)
    if in_use:
        return _existing_unet(in_use, spec, log)
    if os.path.lexists(dest):
        log.error(
            "%s in the weights folder is not an HDF5 file, and no other file there is. It is not "
            "replaced: move it away and restart the service.",
            spec.name,
        )
        return "missing"
    outcome = _download(spec, dest, fetch, log)
    if outcome == "exists":  # a file appeared under the name while we were downloading: it is kept
        in_use = _unet_in_use(folder)
        return _existing_unet(in_use, spec, log) if in_use else "missing"
    return outcome


def _ensure_plain(folder: str, spec: Weight, fetch: Fetch, log: logging.Logger) -> str:
    dest = os.path.join(folder, spec.name)
    _remove_stale_partials(dest)
    if os.path.isfile(dest):
        return "present"
    if os.path.lexists(dest):
        log.error("%s in the weights folder is not a readable file (a broken link?): left as it is.", spec.name)
        return "missing"
    outcome = _download(spec, dest, fetch, log)
    return "present" if outcome == "exists" else outcome


def ensure_weights(
    folder: str,
    unet: Weight = UNET,
    predictions: Weight = PREDICTIONS,
    fetch: Fetch = download,
    log: logging.Logger = _log,
) -> Dict[str, str]:
    """Make sure the weight files are in `folder`. Returns, per file name:

    present     already there, used as it is (the U-Net: the validated file, under one of its two names)
    unexpected  the U-Net is a real HDF5 file but not the validated one: used as it is, and a warning is logged
    downloaded  was missing (the U-Net: or only a Git LFS pointer was there), downloaded and verified
    missing     could not be provided (logged); the service's start-up checks report the consequence
    """
    status: Dict[str, str] = {}
    for spec, provide in ((unet, _ensure_unet), (predictions, _ensure_plain)):
        try:
            os.makedirs(folder, exist_ok=True)
            status[spec.name] = provide(folder, spec, fetch, log)
        except Exception as e:  # e.g. the folder is read-only: one file must not stop the other
            log.error("Could not provide %s: %s", spec.name, e)
            status[spec.name] = "missing"
    return status


_WORDING = {
    "present": "present",
    "unexpected": "present but NOT the validated file (used as it is)",
    "downloaded": "downloaded and verified",
    "missing": "MISSING",
}


def summary(status: Dict[str, str]) -> str:
    """One line for the log: what ensure_weights() found or did, and what to do about a missing file."""
    line = "Weights: " + "; ".join(f"{name} {_WORDING.get(state, state)}" for name, state in status.items()) + "."
    if "missing" in status.values():
        line += (
            " Put the missing file(s) into the folder mounted at /app/ML/UNET Training"
            " (`git lfs pull` in the CAD-COVID checkout fetches the U-Net) and restart the service."
        )
    return line
