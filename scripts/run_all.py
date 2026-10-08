"""Run every validator and report one table. Nothing downstream is trusted until
this exits 0.

    .venv/Scripts/python.exe scripts/run_all.py                  # validators, ~3 min
    .venv/Scripts/python.exe scripts/run_all.py --rehearsals     # + bake-off, transfer, demo
    .venv/Scripts/python.exe scripts/run_all.py --only atlas eval

Each script runs in its own process with this interpreter, so one failure cannot
leave state behind for the next. A failing script's last lines are printed.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parent

#: Order matters only for reading: the harness that audits everything else first.
VALIDATORS = (
    "controls", "canonical", "splits", "paths", "tokenizer", "backbone", "flops", "adapt",
    "decode", "libribrain", "atlas", "eval", "conditioning", "generative", "pretrain", "viz",
)
REHEARSALS = ("run_bakeoff", "run_transfer", "run_demo")


def discover() -> list[Path]:
    """Validator scripts in run order, plus any new ``validate_*.py`` not yet listed."""
    listed = [SCRIPTS / f"validate_{n}.py" for n in VALIDATORS]
    extra = sorted(p for p in SCRIPTS.glob("validate_*.py") if p not in listed)
    return [p for p in listed if p.exists()] + extra


def run(path: Path) -> tuple[int, float, str]:
    t0 = time.time()
    proc = subprocess.run([sys.executable, str(path)], cwd=ROOT, capture_output=True,
                          text=True, encoding="utf-8", errors="replace")
    return proc.returncode, time.time() - t0, proc.stdout + proc.stderr


def verdict(output: str) -> str:
    for line in reversed(output.strip().splitlines()):
        if any(k in line for k in ("VALIDATED", "FAILED", "Traceback", "Error")):
            return line.strip()[:70]
    return ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rehearsals", action="store_true",
                    help="also run the bake-off, transfer and demo rehearsals (~7 min)")
    ap.add_argument("--only", nargs="+", metavar="NAME",
                    help="run only these, e.g. --only atlas eval run_demo")
    args = ap.parse_args()

    paths = discover()
    if args.rehearsals:
        paths += [SCRIPTS / f"{n}.py" for n in REHEARSALS]
    if args.only:
        want = set(args.only)
        paths = [p for p in discover() + [SCRIPTS / f"{n}.py" for n in REHEARSALS]
                 if p.stem in want or p.stem.removeprefix("validate_") in want]
        if not paths:
            print(f"no script matches {sorted(want)}")
            return 2

    print(f"{'script':<26} {'result':<6} {'secs':>6}  verdict")
    print("-" * 96)
    failed = []
    total = 0.0
    for p in paths:
        rc, secs, out = run(p)
        total += secs
        mark = "PASS" if rc == 0 else "FAIL"
        print(f"{p.stem:<26} {mark:<6} {secs:6.0f}  {verdict(out)}", flush=True)
        if rc != 0:
            failed.append((p, out))
    print("-" * 96)

    for p, out in failed:
        print(f"\n--- {p.name} (last 25 lines) ---")
        print("\n".join(out.strip().splitlines()[-25:]))

    print(f"\n{len(paths) - len(failed)}/{len(paths)} passed in {total:.0f}s")
    if failed:
        print("Do not trust any downstream number until every script passes.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
