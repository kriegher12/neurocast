"""Prove the data-location guards fire. Runs in throwaway temp folders; never touches the data drive.

``neurocast.data.paths`` exists to stop five silent failures of keeping the
corpus on a USB drive behind a junction (see its docstring). Each one is
provoked here and must be refused, and the safe configuration must pass:

1. a data location that does not exist
2. a directory that is not the initialised data disk (no marker) -- what you
   get when another drive takes the USB drive's letter
3. a data root on the system drive (Windows), unless explicitly allowed
4. a junction whose target drive has vanished -- the unplugged-drive case
5. paths are identical from any working directory; names cannot escape the root
6. download caches land on the data drive, in sequential-write mode for an HDD
7. configuring after ``huggingface_hub`` was imported is refused, not ignored

Run:  .venv/Scripts/python.exe scripts/validate_paths.py
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from neurocast.data import paths  # noqa: E402


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
    return ok


def raises(fn, exc=paths.DataRootError, contains: str = "") -> tuple[bool, str]:
    try:
        fn()
    except exc as e:
        return (contains.lower() in str(e).lower()), str(e).splitlines()[0][:90]
    except Exception as e:  # noqa: BLE001
        return False, f"wrong exception {type(e).__name__}: {e}"
    return False, "no exception -- the guard did not fire"


def in_subprocess(code: str, env: dict) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", code], cwd=str(ROOT), env=env,
                          capture_output=True, text=True, encoding="utf-8", errors="replace")


def main() -> int:
    ok = True
    saved = os.environ.get("NEUROCAST_DATA")
    tmp = Path(tempfile.mkdtemp(prefix="neurocast-paths-"))
    windows = os.name == "nt"
    try:
        print("=" * 78)
        print("1-3. Locations that must be refused")
        print("=" * 78)
        os.environ["NEUROCAST_DATA"] = str(tmp / "nowhere")
        good, msg = raises(paths.data_root, contains="does not resolve")
        ok &= check("a missing data location is refused", good, msg)

        bare = tmp / "bare"
        bare.mkdir()
        os.environ["NEUROCAST_DATA"] = str(bare)
        good, msg = raises(paths.data_root, contains="no .neurocast-data-root")
        ok &= check("a directory without the marker is refused (wrong disk)", good, msg)

        strict = tmp / "strict"
        strict.mkdir()
        paths.write_marker(strict, disk="hdd")
        os.environ["NEUROCAST_DATA"] = str(strict)
        if windows and strict.resolve().drive.upper() == os.environ.get("SystemDrive", "C:").upper():
            good, msg = raises(paths.data_root, contains="system drive")
            ok &= check("a data root on the system drive is refused", good, msg)
        else:
            print("[SKIP] system-drive refusal (temp dir is not on the Windows system drive)")

        print()
        print("=" * 78)
        print("4. The unplugged-drive case: a junction whose target is gone")
        print("=" * 78)
        if windows:
            target, link = tmp / "usb-drive", tmp / "data-link"
            target.mkdir()
            paths.write_marker(target, disk="hdd", allow_system_drive=True)
            made = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)],
                                  capture_output=True, text=True)
            if made.returncode != 0:
                print(f"[SKIP] could not create a junction: {made.stderr.strip()}")
            else:
                os.environ["NEUROCAST_DATA"] = str(link)
                ok &= check("while the drive is present, the junction resolves",
                            paths.data_root() == target.resolve())
                (target / paths.MARKER_NAME).unlink()   # by the real path, never via the link
                target.rmdir()                          # the drive "is unplugged"
                good, msg = raises(paths.data_root, contains="unplugged")
                ok &= check("a dangling junction is refused, with the unplugged-drive hint",
                            good, msg)
                subprocess.run(["cmd", "/c", "rmdir", str(link)], capture_output=True)
                ok &= check("the junction is removed with rmdir (link only)",
                            not os.path.lexists(link))
        else:
            print("[SKIP] junctions are Windows-only")

        print()
        print("=" * 78)
        print("5. Absolute, working-directory independent, and confined")
        print("=" * 78)
        allowed = tmp / "allowed"
        allowed.mkdir()
        paths.write_marker(allowed, disk="hdd", allow_system_drive=True)
        os.environ["NEUROCAST_DATA"] = str(allowed)
        here = os.getcwd()
        try:
            os.chdir(tmp)
            a = paths.dataset_dir("LibriBrain100")
            os.chdir(Path(tmp).anchor)
            b = paths.dataset_dir("LibriBrain100")
        finally:
            os.chdir(here)
        ok &= check("same absolute dataset path from two different working directories",
                    a == b and a.is_absolute() and a.parent == allowed.resolve(), str(a))
        good, msg = raises(lambda: paths.dataset_dir("../escape"), exc=ValueError)
        ok &= check("a dataset name cannot escape the data root", good or "plain" in msg, msg)

        print()
        print("=" * 78)
        print("6-7. Download caches")
        print("=" * 78)
        env = {k: v for k, v in os.environ.items() if not k.startswith("HF_")}
        env["NEUROCAST_DATA"] = str(allowed)
        p = in_subprocess(
            "from neurocast.data import paths; paths.configure_downloads()\n"
            "import os, json; from huggingface_hub import constants as c\n"
            "print(json.dumps({'home': c.HF_HOME, 'xet': getattr(c, 'HF_XET_CACHE', c.HF_HOME),"
            " 'seq': os.environ.get('HF_XET_RECONSTRUCT_WRITE_SEQUENTIALLY')}))", env)
        try:
            got = json.loads(p.stdout.strip().splitlines()[-1])
            under = all(Path(got[k]).resolve().is_relative_to(allowed.resolve())
                        for k in ("home", "xet"))
            ok &= check("HF home and Xet cache are moved onto the data root", under,
                        got["home"])
            ok &= check("an HDD marker switches hf-xet to sequential writes", got["seq"] == "1")
        except (IndexError, json.JSONDecodeError):
            ok &= check("cache redirection subprocess ran", False, p.stderr.strip()[-200:])

        ssd = tmp / "ssd"
        ssd.mkdir()
        paths.write_marker(ssd, disk="ssd", allow_system_drive=True)
        env_ssd = dict(env, NEUROCAST_DATA=str(ssd))
        p = in_subprocess("from neurocast.data import paths; import os; paths.configure_downloads();"
                          "print(os.environ.get('HF_XET_RECONSTRUCT_WRITE_SEQUENTIALLY'))", env_ssd)
        ok &= check("an SSD marker leaves hf-xet in parallel mode",
                    p.stdout.strip().endswith("None"), p.stdout.strip()[-20:])

        # The library pins its cache when huggingface_hub.constants loads. A bare
        # `import huggingface_hub` is lazy and does NOT pin it -- an earlier draft
        # of this test assumed it did, and this validator caught the assumption.
        p = in_subprocess("import huggingface_hub\n"
                          "from neurocast.data import paths; paths.configure_downloads()\n"
                          "from huggingface_hub import constants as c; print(c.HF_HOME)", env)
        ok &= check("after a bare (lazy) import, configuring still takes effect",
                    p.stdout.strip().startswith(str(allowed.resolve())), p.stdout.strip()[-60:])
        p = in_subprocess("from huggingface_hub import constants\n"
                          "from neurocast.data import paths\n"
                          "try:\n    paths.configure_downloads(); print('NOT REFUSED')\n"
                          "except paths.DataRootError as e:\n    print('REFUSED', e)", env)
        ok &= check("once the cache is pinned elsewhere, configuring is refused, not ignored",
                    p.stdout.startswith("REFUSED"), p.stdout.strip()[:90])
    finally:
        if saved is None:
            os.environ.pop("NEUROCAST_DATA", None)
        else:
            os.environ["NEUROCAST_DATA"] = saved
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    print("=" * 78)
    if ok:
        print("PATH GUARDS VALIDATED: wrong disk, unplugged drive, system drive and late")
        print("cache configuration are all refused; paths ignore the working directory.")
        return 0
    print("PATH GUARDS FAILED validation.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
