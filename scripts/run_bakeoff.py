"""Phase-1 rehearsal: all five arms, matched measured compute, one table.

This is the experimental apparatus H1 and H2 run on, exercised end to end on
synthetic data with known ground truth. It is a **rehearsal, not evidence**: a
few dozen steps on a toy corpus cannot settle which pretraining objective is
better. What it does establish is that the machinery produces the right table,
with the right columns, under a correctly enforced budget -- so that pointing it
at LibriBrain100 is a change of data source and nothing else.

The synthetic corpus is built so the right answer is known in advance: subject
identity is strong and easy (per-subject 1/f exponents and channel gains -- the
exact carriers FMScope names), while the task label is weak and shared across
subjects. A representation scoring high on subject identity and low on label has
taken precisely the shortcut the Identity Trap describes.

Every arm gets the same measured FLOP budget, so arms that cost more per step
simply take fewer steps. That is the point: the EMA teacher alone is +33%, and a
fixed step count would silently gift the latent arms extra compute.

Usage
-----
    .venv/Scripts/python.exe scripts/run_bakeoff.py            # ~2 min, CPU
    .venv/Scripts/python.exe scripts/run_bakeoff.py --budget 2e13
    .venv/Scripts/python.exe scripts/run_bakeoff.py --rung 25m --sites 102
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from neurocast.audit.fmscope import linear_probe_accuracy, subject_variance  # noqa: E402
from neurocast.data.synthetic import SyntheticCorpus  # noqa: E402
from neurocast.objectives.arms import OBJECTIVES  # noqa: E402
from neurocast.train.flops import compare_arms  # noqa: E402
from neurocast.train.loop import (  # noqa: E402
    TrainConfig,
    extract_representations,
    train,
)
from validate_tokenizer import megin_montage  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=4e12,
                    help="measured FLOPs per arm")
    ap.add_argument("--rung", default="6m")
    ap.add_argument("--sites", type=int, default=32,
                    help="MEG sites; 3 channels each plus 3 peripherals")
    ap.add_argument("--n-patch", type=int, default=16)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--subjects", type=int, default=6)
    ap.add_argument("--labels", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    montage = megin_montage(n_sites=args.sites)
    corpus = SyntheticCorpus(
        montage, n_subjects=args.subjects, n_labels=args.labels, seed=args.seed
    )

    print("=" * 84)
    print("PHASE-1 REHEARSAL -- five objectives, matched measured compute")
    print("=" * 84)
    print(f"  montage      {len(montage)} channels")
    print("  " + corpus.describe().replace("\n", "\n  "))
    print(f"  budget       {args.budget:.1e} measured FLOPs per arm")
    print()
    print("Ground truth: subject identity is STRONG and easy, label is WEAK and")
    print("shared. An arm whose frozen representation scores high on subject and")
    print("low on label has taken the Identity Trap shortcut.")
    print()

    rows = []
    ledgers = []
    for arm in OBJECTIVES:
        t0 = time.time()
        cfg = TrainConfig(
            arm=arm, rung=args.rung, budget=args.budget,
            batch_size=args.batch, n_patch=args.n_patch, seed=args.seed,
        )
        model, res = train(cfg, corpus, montage, verbose=True)
        ledgers.append(res.ledger)

        z, subj, lab = extract_representations(
            model, corpus, n=192, n_patch=args.n_patch, seed=args.seed + 99
        )
        # Standardise before probing so scale differences between arms do not
        # masquerade as information differences.
        z = (z - z.mean(0)) / (z.std(0) + 1e-8)

        ident = subject_variance(z, subj, labels=lab, n_permutations=200,
                                 seed=args.seed)
        label_acc = linear_probe_accuracy(z, lab, seed=args.seed)

        rows.append(
            dict(
                arm=arm,
                steps=res.steps,
                gflop=res.flops_per_step / 1e9,
                loss0=res.initial_loss,
                loss1=res.final_loss,
                label_acc=label_acc,
                label_chance=1.0 / args.labels,
                ident=ident,
                collapsed=res.collapsed,
                secs=time.time() - t0,
            )
        )

    print()
    print("=" * 84)
    print("RESULTS")
    print("=" * 84)
    hdr = (
        f"{'arm':<11} {'steps':>6} {'GF/step':>8} {'loss':>16} "
        f"{'label':>13} {'subject id':>22}"
    )
    print(hdr)
    print(f"{'':<11} {'':>6} {'':>8} {'start -> end':>16} "
          f"{'acc (chance)':>13} {'eta2 ratio  verdict':>22}")
    print("-" * 84)
    for r in rows:
        ident = r["ident"]
        flag = " COLLAPSED" if r["collapsed"] else ""
        print(
            f"{r['arm']:<11} {r['steps']:>6} {r['gflop']:>8.2f} "
            f"{r['loss0']:>7.3f} ->{r['loss1']:>7.3f} "
            f"{r['label_acc']:>6.3f} ({r['label_chance']:.2f}) "
            f"{ident.ratio_to_null:>9.1f}x  {ident.verdict:<16}{flag}"
        )
    print("-" * 84)

    print()
    print("Identity detail (frozen, pooled -- the Identity Trap protocol):")
    for r in rows:
        print(f"  {r['arm']:<11} {r['ident'].summary()}")

    print()
    matched, report = compare_arms(ledgers, tol=0.05)
    print("Compute audit:")
    print("  " + report.replace("\n", "\n  "))

    print()
    print("=" * 84)
    print("READING THIS TABLE")
    print("=" * 84)
    print("  LOSS IS NOT COMPARABLE ACROSS ROWS. The arms optimise different")
    print("  quantities -- NLL in nats, MSE, smooth-L1 -- on different targets.")
    print("  Read loss within a row (start -> end) only. The cross-arm columns are")
    print("  label accuracy and the subject-identity ratio, which share a scale.")
    print()
    print("  Do NOT read H1 or H2 off these numbers. A few dozen steps on toy data")
    print("  settles nothing about which objective is better. What is established:")
    print("   * every arm trains, under an identical MEASURED compute budget")
    print("   * arms that cost more per step take fewer steps, automatically")
    print("   * the identity column is produced alongside accuracy, not after it")
    print("   * pointing this at LibriBrain100 is a data-source change, nothing more")
    print()
    if not matched:
        print("  WARNING: arms were not compute-matched. Fix before interpreting.")
    total = sum(r["secs"] for r in rows)
    print(f"  wall clock: {total:.0f}s for {len(rows)} arms")
    return 0 if matched else 1


if __name__ == "__main__":
    raise SystemExit(main())
