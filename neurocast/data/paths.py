"""Where the data lives -- resolved from this file, never from the working directory.

The raw corpus is 0.5-2 TB and lives on a portable USB hard disk, reached through
the ``data`` junction in the repository root (``data -> D:\\neurocast-data``).
Five things can go wrong with that arrangement without any error, and each one
either fills the system drive or damages a download:

1. **Relative paths.** ``data_path="./data/..."`` resolves against the shell's
   working directory; launched from ``C:\\Windows\\system32`` it writes there.
   Every path here is absolute, derived from this file's location.
2. **The drive is unplugged, or returns under another letter.** The junction
   then dangles -- or, worse, now points at whatever disk took that letter. A
   marker file written once on the real data root is checked before any write.
3. **Caches default to the system drive.** ``huggingface_hub`` keeps its Xet
   cache under ``~/.cache/huggingface``. :func:`configure_downloads` moves it onto
   the data drive. Those variables are read when ``huggingface_hub`` is
   *imported*, so this must run first -- and it refuses if it is too late.
4. **The data root is on the system drive.** A 500 GB download would fill C:.
   Refused unless the marker explicitly allows it (clusters, Linux workstations).
5. **A spinning disk.** ``hf-xet`` writes in parallel, which suits SSDs and
   thrashes a portable HDD. The marker records the disk type and
   :func:`configure_downloads` switches ``hf-xet`` to sequential writes.

``NEUROCAST_DATA`` overrides the location (e.g. on a cluster). ``scripts/data_setup.py``
creates the marker, checks all of the above, and fetches data through this module.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

__all__ = [
    "REPO_ROOT",
    "MARKER_NAME",
    "DataRootError",
    "link_path",
    "data_root",
    "dataset_dir",
    "read_marker",
    "write_marker",
    "configure_downloads",
    "free_bytes",
    "filesystem_name",
    "replace_file",
]

#: Repository root: two levels above ``neurocast/data/``.
REPO_ROOT = Path(__file__).resolve().parents[2]

#: Written once on the real data root; its absence means "not the right disk".
MARKER_NAME = ".neurocast-data-root.json"


class DataRootError(RuntimeError):
    """The data location is missing, wrong, or unsafe to write to."""


def link_path() -> Path:
    """The configured data location, before resolving any link."""
    env = os.environ.get("NEUROCAST_DATA")
    return Path(env).expanduser() if env else REPO_ROOT / "data"


def _system_drive() -> str:
    return os.environ.get("SystemDrive", "C:").upper() if os.name == "nt" else ""


def read_marker(root: Path) -> dict:
    p = Path(root) / MARKER_NAME
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as exc:
        raise DataRootError(f"marker {p} is not valid JSON: {exc}") from exc


def write_marker(root: Path, *, disk: str, allow_system_drive: bool = False) -> Path:
    """Mark ``root`` as the data root. ``disk`` is ``"hdd"`` or ``"ssd"``."""
    if disk not in ("hdd", "ssd"):
        raise ValueError(f"disk must be 'hdd' or 'ssd', got {disk!r}")
    root = Path(root)
    if not root.is_dir():
        raise DataRootError(f"{root} is not an existing directory")
    p = root / MARKER_NAME
    p.write_text(json.dumps({
        "purpose": "NeuroCast data root. Its presence tells neurocast.data.paths "
                   "this is the right disk. Do not delete.",
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "disk": disk,
        "allow_system_drive": bool(allow_system_drive),
    }, indent=2), encoding="utf-8")
    return p


def data_root(*, require_marker: bool = True) -> Path:
    """Resolved, verified data root. Raises :class:`DataRootError` if unsafe.

    Every check here guards against a failure that would otherwise be silent:
    see the module docstring.
    """
    link = link_path()
    if not link.exists():
        hint = ""
        is_junction = getattr(link, "is_junction", lambda: False)()  # 3.12+
        if is_junction or link.is_symlink():
            try:
                target = os.readlink(link)
            except OSError:
                target = "?"
            hint = (
                f"\n  It is a link to {target!s}, which is not there. If that is a USB "
                "drive, it is unplugged or mounted under a different letter. Plug it "
                "in; if the letter changed, re-point the link (see the README, "
                "'Data on an external drive')."
            )
        raise DataRootError(f"data location {link} does not resolve.{hint}")

    root = link.resolve()
    marker = read_marker(root)
    if require_marker and not marker:
        raise DataRootError(
            f"{root} has no {MARKER_NAME}. Either this is not the data disk (another "
            "drive took its letter?) or it was never initialised -- run\n"
            f"  python scripts/data_setup.py init {root}"
        )
    sysdrive = _system_drive()
    if sysdrive and root.drive.upper() == sysdrive and not marker.get("allow_system_drive"):
        raise DataRootError(
            f"data root {root} is on the system drive {sysdrive}; a corpus download "
            "would fill it. Point the 'data' link at the external drive, or re-run "
            "`data_setup.py init --allow-system-drive` if this is really intended."
        )
    return root


def dataset_dir(name: str) -> Path:
    """Absolute directory for one dataset under the verified root, created if needed."""
    if not name or Path(name).is_absolute() or ".." in Path(name).parts:
        raise ValueError(f"dataset name must be a plain relative name, got {name!r}")
    d = data_root() / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def configure_downloads(root: Path | None = None) -> dict[str, str]:
    """Point download caches at the data drive. Call before ``huggingface_hub`` is imported.

    Sets (without overriding anything the user already exported):

    * ``HF_HOME`` -> ``<root>/.cache/huggingface``, which moves the token, the hub
      cache and the Xet cache off the system drive;
    * ``HF_XET_RECONSTRUCT_WRITE_SEQUENTIALLY=1`` when the marker says ``hdd``.

    Raises if the cache location is already pinned elsewhere. ``huggingface_hub``
    reads these variables once, when its ``constants`` module loads -- *not* on a
    bare ``import huggingface_hub``, which is lazy. Once ``constants`` is loaded,
    setting them would look like it worked and change nothing.
    """
    root = data_root() if root is None else Path(root)
    marker = read_marker(root)
    wanted = {"HF_HOME": str(root / ".cache" / "huggingface")}
    if marker.get("disk", "hdd") == "hdd":
        wanted["HF_XET_RECONSTRUCT_WRITE_SEQUENTIALLY"] = "1"
    for k, v in wanted.items():
        os.environ.setdefault(k, v)

    constants = sys.modules.get("huggingface_hub.constants")
    if constants is not None:
        actual = Path(constants.HF_HOME).resolve()
        if actual != Path(os.environ["HF_HOME"]).resolve():
            raise DataRootError(
                f"huggingface_hub was imported before configure_downloads(); its cache "
                f"is pinned to {actual}. Call configure_downloads() first."
            )
    return {k: os.environ[k] for k in wanted}


def free_bytes(path: Path) -> int:
    return shutil.disk_usage(path).free


def filesystem_name(path: Path) -> str | None:
    """``"NTFS"``, ``"exFAT"``, ``"FAT32"``... on Windows; ``None`` elsewhere."""
    if os.name != "nt":
        return None
    import ctypes

    root = Path(path).resolve().anchor            # e.g. "D:\\"
    buf = ctypes.create_unicode_buffer(64)
    ok = ctypes.windll.kernel32.GetVolumeInformationW(
        ctypes.c_wchar_p(root), None, 0, None, None, None, buf, len(buf)
    )
    return buf.value if ok else None


def replace_file(src: Path, dst: Path, *, attempts: int = 20, delay: float = 0.5) -> None:
    """``src.replace(dst)``, retried while another process holds either file.

    On Windows a file written a moment ago is often opened by an antivirus or
    indexer scan, and the rename fails with ``PermissionError`` [WinError 32]. It
    stopped MEG-MASC preprocessing an hour in. The scan lets go within seconds.
    """
    import time

    for i in range(attempts):
        try:
            Path(src).replace(dst)
            return
        except PermissionError:
            if i == attempts - 1:
                raise
            time.sleep(delay)
