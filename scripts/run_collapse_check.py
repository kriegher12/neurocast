"""The masked-collapse diagnostic on any pre-training checkpoint, mid-run or final.

``run_pretrain_scale.py --evaluate`` diagnoses the best checkpoints after a whole
queue; this answers "is it collapsed?" for one checkpoint in a minute, without
waiting for the identity and word-probe pass over 64 recordings.

    .venv-gpu/Scripts/python.exe scripts/run_collapse_check.py runs/pretrain_scale/masked_s0.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from run_pretrain_pilot import collapse  # noqa: E402
from run_pretrain_scale import ARMS, build  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    s = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = s["cfg"]
    arm = next(k for k, v in ARMS.items() if v["arm"] == cfg["arm"]
               and v.get("visible_weight", 0.0) == cfg.get("visible_weight", 0.0))
    del s
    tr = build(arm, cfg["seed"], cfg["steps"], args.device, prefetch=0)
    tr.load(args.checkpoint)
    rep = collapse(tr)
    held = [e["heldout"] for e in tr.log if e.get("heldout") is not None]
    out = {"checkpoint": str(args.checkpoint), "step": tr.step, "heldout_last": held[-1] if held else None,
           "spread": rep.spread, "sensitivity": rep.sensitivity, "model_r2": rep.model_r2,
           "reference_r2": rep.reference_r2, "isolated_r2": rep.isolated_r2, "collapsed": rep.collapsed}
    print(f"{args.checkpoint.name} at step {tr.step}: {rep.summary()}")
    dest = args.checkpoint.with_name(args.checkpoint.stem + f".collapse_step{tr.step}.json")
    dest.write_text(json.dumps(out, indent=2, default=float))
    print(f"saved {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
