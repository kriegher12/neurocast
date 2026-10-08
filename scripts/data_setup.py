"""Set up, check and fill the data drive -- from any working directory.

    python scripts/data_setup.py init D:\\neurocast-data --disk hdd   # once per data disk
    python scripts/data_setup.py check                                # before each download session
    python scripts/data_setup.py fetch --dry-run                      # what would come, and how big
    python scripts/data_setup.py fetch                                # one LibriBrain100 run (default)
    python scripts/data_setup.py fetch --subjects 0 --corpus sherlock --limit 12

The script moves into the project folder itself and uses only absolute paths,
so it does the same thing launched from the repository, from
``C:\\Windows\\system32``, or from an IDE. Use the venv's interpreter
(``.venv\\Scripts\\python.exe``) or it will not find the packages.

Why ``fetch`` instead of constructing a pnpl dataset
----------------------------------------------------
pnpl's ``LibriBrain100`` downloads **every selected run inside its constructor**
(``ensure_file`` per record, whatever ``preload_files`` says), and its defaults
select every subject and corpus -- about half a terabyte, blocking, the moment
the object is created. ``fetch`` selects runs through pnpl's own manifest,
sizes them on the Hub *before* downloading, refuses if they will not fit, and
writes them into exactly the layout pnpl reads. Afterwards, construct pnpl
datasets with the same selection and they find the files already there.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)  # "change into the project folder" -- done here, not left to the shell

from neurocast.data import paths  # noqa: E402  (must not import huggingface_hub)

GB = 1024**3
MIN_FREE_DATA = 50 * GB       # refuse below this on the data drive
WARN_FREE_DATA = 600 * GB     # LibriBrain100 alone is ~0.5 TB
WARN_FREE_SYSTEM = 20 * GB
FETCH_MARGIN = 20 * GB        # never fill the data drive to the last byte


class Report:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str]] = []

    def add(self, status: str, name: str, detail: str = "") -> None:
        self.rows.append((status, name, detail))
        print(f"[{status:<4}] {name}{('  ' + detail) if detail else ''}", flush=True)

    @property
    def failed(self) -> bool:
        return any(s == "FAIL" for s, _, _ in self.rows)

    @property
    def warned(self) -> bool:
        return any(s == "WARN" for s, _, _ in self.rows)


def _gb(n: int) -> str:
    return f"{n / GB:,.1f} GB"


# ----------------------------------------------------------------------------- check

def run_checks(verbose: bool = True) -> tuple[Report, Path | None]:
    r = Report()
    if not verbose:
        r.add = lambda s, n, d="": r.rows.append((s, n, d))  # type: ignore[method-assign]

    link = paths.link_path()
    try:
        root = paths.data_root()
    except paths.DataRootError as exc:
        r.add("FAIL", "data location", str(exc))
        return r, None
    via = f"{link} -> {root}" if link.resolve() != link.absolute() else str(root)
    r.add("OK", "data location resolves", via)

    marker = paths.read_marker(root)
    r.add("OK", "marker present: this is the initialised data disk",
          f"created {marker.get('created', '?')}, disk={marker.get('disk', '?')}")
    if os.name == "nt":
        r.add("OK", "data root is not on the system drive",
              f"{root.drive} (system drive is {os.environ.get('SystemDrive', 'C:')})")

    fs = paths.filesystem_name(root)
    if fs:
        if fs.upper().startswith("FAT"):
            r.add("FAIL", f"filesystem {fs}", "4 GB file-size limit; reformat as NTFS or exFAT")
        else:
            r.add("OK", f"filesystem {fs}", "no 4 GB file limit")

    probe = root / ".write-probe-delete-me"
    try:
        probe.write_bytes(b"ok")
        probe.unlink()
        r.add("OK", "data root is writable")
    except OSError as exc:
        r.add("FAIL", "data root is writable", str(exc))

    free = paths.free_bytes(root)
    if free < MIN_FREE_DATA:
        r.add("FAIL", "free space on the data drive", _gb(free))
    elif free < WARN_FREE_DATA:
        r.add("WARN", "free space on the data drive", f"{_gb(free)} -- less than LibriBrain100 needs")
    else:
        r.add("OK", "free space on the data drive",
              f"{_gb(free)} (Tier 1 in full needs ~2 TB; plan subsets)")

    sys_root = Path((os.environ.get("SystemDrive", "C:") + "\\") if os.name == "nt" else "/")
    sys_free = paths.free_bytes(sys_root)
    r.add("WARN" if sys_free < WARN_FREE_SYSTEM else "OK",
          "free space on the system drive", f"{_gb(sys_free)} (caches are redirected off it)")

    try:
        env = paths.configure_downloads(root)
        from huggingface_hub import constants as c

        targets = {"HF_HOME": c.HF_HOME, "HF_HUB_CACHE": c.HF_HUB_CACHE,
                   "HF_XET_CACHE": getattr(c, "HF_XET_CACHE", c.HF_HOME)}
        off = {k: v for k, v in targets.items() if not Path(v).resolve().is_relative_to(root)}
        if off:
            r.add("FAIL", "Hugging Face caches on the data drive",
                  "; ".join(f"{k}={v}" for k, v in off.items())
                  + " -- an HF_* variable set elsewhere is overriding this")
        else:
            r.add("OK", "Hugging Face caches on the data drive", targets["HF_HOME"])
        seq = env.get("HF_XET_RECONSTRUCT_WRITE_SEQUENTIALLY")
        r.add("OK", "hf-xet write mode",
              "sequential (spinning disk)" if seq == "1" else "parallel (SSD)")
    except paths.DataRootError as exc:
        r.add("FAIL", "Hugging Face caches", str(exc))

    for mod in ("h5py", "hf_xet", "pnpl"):
        try:
            __import__(mod)
            r.add("OK", f"{mod} imports")
        except Exception as exc:  # noqa: BLE001 -- report whatever the loader says
            r.add("FAIL", f"{mod} imports", f"{type(exc).__name__}: {exc}")

    if os.name == "nt":
        try:
            import winreg

            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                r"SYSTEM\CurrentControlSet\Control\FileSystem") as k:
                on = winreg.QueryValueEx(k, "LongPathsEnabled")[0] == 1
        except OSError:
            on = False
        r.add("OK" if on else "WARN", "Windows long paths",
              "enabled" if on else "disabled; deep BIDS paths over 260 chars will fail")

    gi = (ROOT / ".gitignore").read_text(encoding="utf-8") if (ROOT / ".gitignore").exists() else ""
    ig = (ROOT / ".ignore").read_text(encoding="utf-8") if (ROOT / ".ignore").exists() else ""
    missing = [p for p in ("data/", ".env") if p not in gi.split()]
    if missing:
        r.add("WARN", ".gitignore", f"missing {missing}: data or credentials could be committed")
    else:
        r.add("OK", ".gitignore covers data/ and .env")
    r.add("OK" if "data/" in ig.split() else "WARN", ".ignore keeps search tools out of data/",
          "" if "data/" in ig.split() else "code search would crawl the whole corpus")
    return r, root


def cmd_check(_args) -> int:
    print(f"project folder: {ROOT}\n")
    r, _ = run_checks()
    print()
    if r.failed:
        print("NOT READY: fix every FAIL above before downloading.")
        return 1
    print("READY" + (" (read the WARN lines)" if r.warned else "") + ".")
    print("Before a long download: plug the drive straight into the PC (not a hub), and")
    print("keep the PC from sleeping. An interrupted download resumes on the next run.")
    return 0


# ------------------------------------------------------------------------------ init

def cmd_init(args) -> int:
    target = Path(args.path).resolve()
    if not target.is_dir():
        print(f"{target} does not exist; create it first.")
        return 1
    existing = paths.read_marker(target)
    if existing and not args.force:
        print(f"{target} is already initialised ({existing.get('created')}). --force to rewrite.")
        return 0
    m = paths.write_marker(target, disk=args.disk, allow_system_drive=args.allow_system_drive)
    print(f"wrote {m}")
    link = paths.link_path()
    if link.exists() and link.resolve() == target:
        print(f"{link} -> {target}: linked.")
    else:
        print(f"note: {link} does not point here. Link it (PowerShell):")
        print(f"  New-Item -ItemType Junction -Path {ROOT / 'data'} -Target {target}")
    return 0


# ----------------------------------------------------------------------------- fetch

def _libribrain_files(records, preprocessing: str):
    """(relative path, primary repo, fallback repo) for each file pnpl will read."""
    from pnpl.datasets.libribrain100.constants import REPO_KEY_TO_ID

    out = []
    for rec in records:
        stem = f"sub-{rec.subject}_ses-{rec.session}_task-{rec.task}_run-{rec.run}"
        primary = REPO_KEY_TO_ID[rec.repo]
        fallback = [v for v in REPO_KEY_TO_ID.values() if v != primary]
        out.append((f"{rec.task}/derivatives/serialised/{stem}_proc-{preprocessing}_meg.h5",
                    primary, fallback))
        out.append((f"{rec.task}/derivatives/events/{stem}_events.tsv", primary, fallback))
    return out


def _remote_sizes(files):
    """rel path -> (repo, bytes) for files found on the Hub; metadata only, no data."""
    from huggingface_hub import HfApi

    api = HfApi()
    found: dict[str, tuple[str, int]] = {}
    pending = list(files)
    for attempt in range(2):                      # primary repo, then fallback
        by_repo: dict[str, list[str]] = {}
        for rel, primary, fallback in pending:
            repo = primary if attempt == 0 else (fallback[0] if fallback else None)
            if repo:
                by_repo.setdefault(repo, []).append(rel)
        for repo, rels in by_repo.items():
            for i in range(0, len(rels), 50):
                for info in api.get_paths_info(repo, rels[i:i + 50], repo_type="dataset"):
                    size = getattr(info, "size", None)
                    if size is not None:
                        found[info.path] = (repo, int(size))
        pending = [f for f in pending if f[0] not in found]
    return found, [f[0] for f in pending]


def cmd_fetch(args) -> int:
    report, root = run_checks(verbose=False)
    if report.failed or root is None:
        for s, n, d in report.rows:
            if s == "FAIL":
                print(f"[FAIL] {n}  {d}")
        print("\nRefusing to download: run `scripts/data_setup.py check` and fix the FAILs.")
        return 1
    paths.configure_downloads(root)               # before huggingface_hub is imported

    from pnpl.datasets.libribrain100 import manifest, selectors
    from pnpl.datasets.libribrain100.constants import DEFAULT_PREPROCESSING_STR

    include = [tuple(k.split(",")) for k in args.run_key] or None
    records = manifest.select_records(
        subjects=selectors.normalize_subjects(args.subjects),
        corpus=selectors.normalize_corpus(args.corpus),
        partition=selectors.normalize_partition(args.partition),
        include_run_keys=include,
    )
    if not records:
        print("No LibriBrain100 runs match that selection.")
        return 1
    total_runs = len(records)
    if args.limit != "all":
        records = records[: int(args.limit)]

    ds_dir = paths.dataset_dir("LibriBrain100")
    files = _libribrain_files(records, DEFAULT_PREPROCESSING_STR)
    print(f"LibriBrain100 -> {ds_dir}")
    print(f"selection: {len(records)} of {total_runs} matching runs "
          f"({'--limit ' + args.limit if args.limit != 'all' else 'all'}); "
          f"querying sizes on the Hub (metadata only) ...")
    sizes, missing = _remote_sizes(files)

    todo, have = [], 0
    for rel, _, _ in files:
        if rel not in sizes:
            continue
        repo, size = sizes[rel]
        local = ds_dir / rel
        if local.exists() and local.stat().st_size == size:
            have += size
        else:
            todo.append((rel, repo, size))
    need = sum(s for _, _, s in todo)
    free = paths.free_bytes(root)

    print(f"  on the Hub: {len(sizes)} files; already here: {_gb(have)}; to download: "
          f"{len(todo)} files, {_gb(need)}; free on {root.drive or root}: {_gb(free)}")
    for rel in missing:
        print(f"  not on the Hub (skipped; LibriBrain2 may still be uploading): {rel}")
    if args.dry_run:
        for rel, repo, size in todo[:20]:
            print(f"    {size / GB:8.2f} GB  {repo}  {rel}")
        if len(todo) > 20:
            print(f"    ... and {len(todo) - 20} more")
        print("dry run: nothing downloaded.")
        return 0
    if need + FETCH_MARGIN > free:
        print(f"Refusing: {_gb(need)} plus a {_gb(FETCH_MARGIN)} margin does not fit in "
              f"{_gb(free)}. Narrow the selection (--limit, --subjects, --corpus).")
        return 1
    if not todo:
        print("Everything selected is already downloaded.")
        return 0

    from huggingface_hub import hf_hub_download

    for i, (rel, repo, size) in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {rel}  ({size / GB:.2f} GB)", flush=True)
        got = Path(hf_hub_download(repo_id=repo, repo_type="dataset", filename=rel,
                                   local_dir=str(ds_dir)))
        if got.stat().st_size != size:
            print(f"  size mismatch: {got.stat().st_size} vs {size}; delete it and re-run")
            return 1

    import h5py

    first_h5 = next(ds_dir / rel for rel, _, _ in files if rel.endswith(".h5") and rel in sizes)
    with h5py.File(first_h5, "r") as f:
        desc = ", ".join(f"{k}{tuple(f[k].shape) if hasattr(f[k], 'shape') else ''}"
                         for k in list(f.keys())[:4])
    print(f"\nverified {first_h5.name}: {desc}")
    keys = [r.run_key for r in records]
    print("\nLoad it with pnpl using the SAME selection (otherwise its constructor fetches")
    print("everything it selects):")
    print(f"  LibriBrain100(data_path=r'{ds_dir}', task=..., include_run_keys={keys[:3]}"
          f"{'...' if len(keys) > 3 else ''})")
    return 0


def cmd_fetch_files(args) -> int:
    """Download every file in a Hub dataset repo matching glob patterns.

    For files pnpl's manifest does not know about -- e.g. the 20 broad listeners
    published only as partial sessions (``derivatives/serialised_competition/
    ..._desc-firsthalf_meg.h5``), whose names pnpl's loader never constructs.
    Same guards as ``fetch``: checks first, sized on the Hub, refused if it will
    not fit, written into the repo's own layout under the dataset folder.
    """
    import fnmatch

    report, root = run_checks(verbose=False)
    if report.failed or root is None:
        print("Refusing to download: run `scripts/data_setup.py check` and fix the FAILs.")
        return 1
    paths.configure_downloads(root)

    from huggingface_hub import HfApi, hf_hub_download

    ds_dir = paths.dataset_dir(args.dest)
    listing = [f for f in HfApi().list_repo_tree(args.repo, repo_type="dataset", recursive=True)
               if getattr(f, "size", None) is not None]
    chosen = [f for f in listing if any(fnmatch.fnmatch(f.path, pat) for pat in args.include)]
    todo = [f for f in chosen
            if not ((ds_dir / f.path).exists() and (ds_dir / f.path).stat().st_size == f.size)]
    need, free = sum(f.size for f in todo), paths.free_bytes(root)
    print(f"{args.repo} -> {ds_dir}")
    print(f"  {len(chosen)} files match {args.include}; already here {len(chosen) - len(todo)}; "
          f"to download {len(todo)} files, {_gb(need)}; free {_gb(free)}", flush=True)
    if args.dry_run:
        print("dry run: nothing downloaded.")
        return 0
    if need + FETCH_MARGIN > free:
        print(f"Refusing: {_gb(need)} plus a {_gb(FETCH_MARGIN)} margin does not fit.")
        return 1

    import time

    t0, done = time.time(), 0
    for i, f in enumerate(sorted(todo, key=lambda f: f.path), 1):
        got = Path(hf_hub_download(repo_id=args.repo, repo_type="dataset", filename=f.path,
                                   local_dir=str(ds_dir)))
        if got.stat().st_size != f.size:
            print(f"  size mismatch on {f.path}: {got.stat().st_size} vs {f.size}")
            return 1
        done += f.size
        rate = done / max(time.time() - t0, 1e-9) / 1e6
        print(f"[{i}/{len(todo)}] {_gb(done)} of {_gb(need)}  {rate:5.1f} MB/s  {f.path}",
              flush=True)
    print(f"done: {len(todo)} files, {_gb(done)} in {(time.time() - t0) / 60:.1f} min")
    return 0


#: OSF datasets pnpl knows, by the folder name they get under the data root.
OSF_DATASETS = {"MEG-MASC": ("pnpl.datasets.gwilliams2022", "Gwilliams2022")}


def cmd_fetch_osf(args) -> int:
    """Download files of an OSF-hosted dataset matching glob patterns.

    MEG-MASC (Gwilliams et al.) lives on OSF, split over four components; pnpl's
    ``Gwilliams2022`` class walks all four into one manifest. Same guards as the
    other fetches: checks first, sized from the manifest, refused if it will not
    fit, written in the dataset's BIDS layout under the data root.
    """
    import fnmatch
    import importlib
    import time

    report, root = run_checks(verbose=False)
    if report.failed or root is None:
        print("Refusing to download: run `scripts/data_setup.py check` and fix the FAILs.")
        return 1
    module, name = OSF_DATASETS[args.dataset]
    cls = getattr(importlib.import_module(module), name)
    ds_dir = paths.dataset_dir(args.dataset)
    print(f"{args.dataset}: listing OSF storage (metadata only) ...", flush=True)
    manifest = cls.get_dataset_manifest()
    total = sum(int(e.get("size") or 0) for e in manifest.values())
    chosen = {k: e for k, e in manifest.items() if any(fnmatch.fnmatch(k, pat) for pat in args.include)}
    todo = {k: e for k, e in chosen.items()
            if not ((ds_dir / k).exists() and (ds_dir / k).stat().st_size == int(e.get("size") or -1))}
    need, free = sum(int(e.get("size") or 0) for e in todo.values()), paths.free_bytes(root)
    print(f"  {len(manifest)} files, {_gb(total)} in all; {len(chosen)} match {args.include}; "
          f"to download {len(todo)} files, {_gb(need)}; free {_gb(free)}", flush=True)
    if args.dry_run:
        for k in sorted(todo)[:10]:
            print(f"    {int(todo[k].get('size') or 0) / GB:8.3f} GB  {k}")
        print("dry run: nothing downloaded.")
        return 0
    if need + FETCH_MARGIN > free:
        print(f"Refusing: {_gb(need)} plus a {_gb(FETCH_MARGIN)} margin does not fit.")
        return 1
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def one(k):
        dest = ds_dir / k
        dest.parent.mkdir(parents=True, exist_ok=True)
        cls._download_with_retry_static(fpath=str(dest), rel_path=k, entry=todo[k],
                                        files_base=cls.OSF_FILES_BASE.rstrip("/"),
                                        token=cls._osf_token())
        return int(todo[k].get("size") or 0)

    # A few connections at once: OSF can serve one connection at ~0.1 MB/s while
    # the line does 4-5 MB/s, and one slow file then holds up the whole queue.
    t0, done = time.time(), 0
    with ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(one, k) for k in sorted(todo)]
        for i, fut in enumerate(as_completed(futures), 1):
            done += fut.result()
            if i % 10 == 0 or i == len(todo):
                print(f"[{i}/{len(todo)}] {_gb(done)} of {_gb(need)}  "
                      f"{done / max(time.time() - t0, 1e-9) / 1e6:5.1f} MB/s", flush=True)
    print(f"done: {len(todo)} files in {(time.time() - t0) / 60:.1f} min")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd")

    sub.add_parser("check", help="verify the data drive, caches and tooling")

    p = sub.add_parser("init", help="mark a directory as the data root")
    p.add_argument("path")
    p.add_argument("--disk", choices=("hdd", "ssd"), default="hdd",
                   help="hdd (default) makes hf-xet write sequentially")
    p.add_argument("--allow-system-drive", action="store_true")
    p.add_argument("--force", action="store_true")

    p = sub.add_parser("fetch", help="download LibriBrain100 runs, sized and checked first")
    p.add_argument("--subjects", default="0", help="'0', 'deep', 'broad', 'all', or ids")
    p.add_argument("--corpus", default="sherlock")
    p.add_argument("--partition", default=None, help="train / validation / test")
    p.add_argument("--run-key", action="append", default=[], metavar="SUB,SES,TASK,RUN",
                   help="explicit run, repeatable (e.g. 0,1,Sherlock1,1)")
    p.add_argument("--limit", default="1", help="max runs to fetch, or 'all' (default 1)")
    p.add_argument("--dry-run", action="store_true", help="size it, download nothing")

    p = sub.add_parser("fetch-files", help="download Hub files matching glob patterns")
    p.add_argument("--repo", required=True, help="e.g. pnpl/LibriBrain2")
    p.add_argument("--include", action="append", required=True, metavar="GLOB",
                   help="repeatable, e.g. 'Sherlock1/derivatives/events/*'")
    p.add_argument("--dest", default="LibriBrain100", help="dataset folder under the data root")
    p.add_argument("--dry-run", action="store_true")

    p = sub.add_parser("fetch-osf", help="download OSF dataset files matching glob patterns")
    p.add_argument("--dataset", choices=tuple(OSF_DATASETS), default="MEG-MASC")
    p.add_argument("--workers", type=int, default=4, help="parallel connections (default 4)")
    p.add_argument("--include", action="append", required=True, metavar="GLOB",
                   help="repeatable, e.g. 'sub-*/ses-*/meg/*_events.tsv'")
    p.add_argument("--dry-run", action="store_true")

    args = ap.parse_args()
    return {"init": cmd_init, "fetch": cmd_fetch, "fetch-files": cmd_fetch_files,
            "fetch-osf": cmd_fetch_osf}.get(args.cmd, cmd_check)(args)


if __name__ == "__main__":
    raise SystemExit(main())
