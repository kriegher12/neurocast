"""Quantify risk R1 -- and test the design's proposed mitigations against it.

R1, the claim this program would bet against: ``log p(X | w)`` sums over ~30,600
dimensions while the information separating one word from another is plausibly
under a bit. This script builds a synthetic decoding problem with that shape and
measures what actually happens.

It overturned two of the design's three R1 mitigations. Both results are asserted
below, so they cannot quietly regress:

* **CFG-style subtraction is a no-op for ranking.** ``log p(X|empty)`` is constant
  across candidates within a trial; subtracting it at any weight leaves every
  ranking identical.
* **A discriminative subspace fitted on the labels does not help.** At full rank
  it is mathematically identical to full-D scoring; at reduced rank its directions
  inherit the labels' noise.

What does work: a subspace supplied from **unlabelled** data. With the true
subspace, 16 labelled trials per word suffice at any dimension -- so the
bottleneck is estimating the subspace, which the labels cannot do. That is a
quantitative argument for the forecaster, whose job is to learn exactly that
subspace from hours of unlabelled MEG.

Run:  .venv/Scripts/python.exe scripts/validate_decode.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neurocast.decode.metrics import (  # noqa: E402
    accuracy_at_k,
    balanced_accuracy_at_k,
    chance_bacc_at_k,
)
from neurocast.decode.scorer import (  # noqa: E402
    DiscriminativeSubspace,
    SubspaceDecoder,
    calibrate_lambda,
    gaussian_loglik,
    lm_pmi,
    neural_pmi,
    unlabelled_basis,
)

K = 50            # PNPL vocabulary size
D_SIG = 16        # dimensions carrying word information
AMP = 0.45        # word-signal amplitude per signal dimension
N_TRAIN = 16      # labelled trials per word -- deliberately scarce
N_TEST = 12
N_UNLAB = 32_000  # unlabelled samples for the subspace


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
    return ok


class Problem:
    """Word signal in a random D_SIG-dim rotated subspace of a D-dim observation."""

    def __init__(self, rng: np.random.Generator, d: int):
        self.rng = rng
        self.d = d
        self.basis, _ = np.linalg.qr(rng.standard_normal((d, D_SIG)))
        self.means = (rng.standard_normal((K, D_SIG)) * AMP) @ self.basis.T

    def labelled(self, n_per: int):
        y = np.repeat(np.arange(K), n_per)
        return self.means[y] + self.rng.standard_normal((len(y), self.d)), y

    def unlabelled(self, n: int) -> np.ndarray:
        y = self.rng.integers(0, K, n)   # generated, then discarded: never used
        return self.means[y] + self.rng.standard_normal((n, self.d))


def main() -> int:
    rng = np.random.default_rng(0)
    ok = True
    chance = chance_bacc_at_k(K, 10)

    print("=" * 84)
    print("1. Metric sanity")
    print("=" * 84)
    y = np.repeat(np.arange(K), 40)
    accs = [balanced_accuracy_at_k(rng.standard_normal((len(y), K)), y, 10)
            for _ in range(20)]
    print(f"  random scores, {K} classes: BAcc@10 = {np.mean(accs):.3f} "
          f"+- {np.std(accs):.3f}  (expected {chance:.3f})")
    ok &= check("random BAcc@10 matches k/K", abs(np.mean(accs) - chance) < 0.01)

    print()
    print("=" * 84)
    print("2. THE R1 CURVE, and which mitigations survive it")
    print("=" * 84)
    print(f"  {K} words, signal in {D_SIG} rotated dims, {N_TRAIN} labelled trials/word,")
    print(f"  {N_UNLAB:,} unlabelled samples. Real window is 30,600 dims.")
    print()
    print(f"  {'D':>5} {'oracle':>7} {'full-D':>7} {'LDA r=K-1':>10} {'LDA r=16':>9} "
          f"{'true subsp':>11} {'unlab subsp':>12}")
    print("  " + "-" * 66)

    rows = []
    for d in (128, 512, 1024):
        pb = Problem(rng, d)
        xtr, ytr = pb.labelled(N_TRAIN)
        xte, yte = pb.labelled(N_TEST)

        oracle = balanced_accuracy_at_k(gaussian_loglik(xte, pb.means, 1.0), yte, 10)
        mhat = np.stack([xtr[ytr == c].mean(0) for c in range(K)])
        full = balanced_accuracy_at_k(gaussian_loglik(xte, mhat, 1.0), yte, 10)

        lda_full = balanced_accuracy_at_k(
            DiscriminativeSubspace(shrinkage=0.5).fit(xtr, ytr).loglik(xte), yte, 10)
        lda_r16 = balanced_accuracy_at_k(
            DiscriminativeSubspace(rank=D_SIG, shrinkage=0.5).fit(xtr, ytr).loglik(xte),
            yte, 10)

        true_sub = balanced_accuracy_at_k(
            SubspaceDecoder(pb.basis).fit(xtr, ytr).loglik(xte), yte, 10)
        ub = unlabelled_basis(pb.unlabelled(N_UNLAB), D_SIG)
        unlab_sub = balanced_accuracy_at_k(
            SubspaceDecoder(ub).fit(xtr, ytr).loglik(xte), yte, 10)

        rows.append(dict(d=d, oracle=oracle, full=full, lda_full=lda_full,
                         lda_r16=lda_r16, true_sub=true_sub, unlab_sub=unlab_sub))
        print(f"  {d:>5} {oracle:>7.3f} {full:>7.3f} {lda_full:>10.3f} {lda_r16:>9.3f} "
              f"{true_sub:>11.3f} {unlab_sub:>12.3f}")
    print("  " + "-" * 66)
    print(f"  chance {chance:.3f}.  'oracle' = true class means; 'true subsp' = true")
    print("  subspace with means from the labels; 'unlab subsp' = PCA on unlabelled data.")
    print()

    lo, hi = rows[0], rows[-1]
    ok &= check("R1 is real: full-D likelihood degrades with dimension",
                lo["full"] - hi["full"] > 0.10,
                f"{lo['full']:.3f} -> {hi['full']:.3f}")
    ok &= check("full-rank LDA is equivalent to full-D (design mitigation #1 fails)",
                all(abs(r["lda_full"] - r["full"]) < 0.03 for r in rows),
                "max gap " + f"{max(abs(r['lda_full'] - r['full']) for r in rows):.3f}")
    ok &= check("rank-16 LDA does not rescue high D (labels can't find the subspace)",
                hi["lda_r16"] < hi["true_sub"] - 0.20,
                f"{hi['lda_r16']:.3f} vs true subspace {hi['true_sub']:.3f}")
    ok &= check("with the TRUE subspace, 16 labels/word suffice at every D",
                all(r["true_sub"] > 0.65 for r in rows),
                "min " + f"{min(r['true_sub'] for r in rows):.3f}")
    ok &= check("a subspace from UNLABELLED data beats full-D at high D",
                hi["unlab_sub"] > hi["full"] + 0.05,
                f"{hi['unlab_sub']:.3f} vs {hi['full']:.3f}")
    print("  -> the bottleneck is estimating the subspace, not the labels. It has to")
    print("     come from unlabelled data -- which is the forecaster's job.")

    print()
    print("=" * 84)
    print("3. CFG-style subtraction cannot change a ranking (design mitigation #2 fails)")
    print("=" * 84)
    pb = Problem(rng, 128)
    xtr, ytr = pb.labelled(N_TRAIN)
    xte, yte = pb.labelled(N_TEST)
    mhat = np.stack([xtr[ytr == c].mean(0) for c in range(K)])
    ll = gaussian_loglik(xte, mhat, 1.0)
    ll_null = 50.0 * rng.standard_normal(len(yte))
    base = balanced_accuracy_at_k(ll, yte, 10)
    same = True
    for w in (0.0, 0.5, 1.0, 2.0, 10.0):
        b = balanced_accuracy_at_k(neural_pmi(ll, ll_null, w), yte, 10)
        same &= abs(b - base) < 1e-12
        print(f"  w_cfg = {w:5.1f}   BAcc@10 = {b:.4f}")
    ok &= check("BAcc@10 identical at every w_cfg", same,
                "a per-trial constant cannot reorder candidates")

    print()
    print("=" * 84)
    print("4. LM pointwise mutual information vs. a raw frequency-skewed prior")
    print("=" * 84)
    freq = 1.0 / np.arange(1, K + 1) ** 1.1
    freq /= freq.sum()
    log_freq = np.log(freq)
    n = len(yte)
    logits = 1.2 * np.eye(K)[yte] + log_freq[None, :] + 0.6 * rng.standard_normal((n, K))
    lm_ctx = logits - np.log(np.exp(logits).sum(1, keepdims=True))

    raw_b = balanced_accuracy_at_k(lm_ctx, yte, 10)
    pmi_b = balanced_accuracy_at_k(lm_pmi(lm_ctx, log_freq), yte, 10)
    rare = yte >= K // 2

    def rare_recall(s):
        return float((np.argsort(-s, 1)[:, :10] == yte[:, None]).any(1)[rare].mean())

    rr, pr = rare_recall(lm_ctx), rare_recall(lm_pmi(lm_ctx, log_freq))
    print(f"  raw LM prior   BAcc@10 {raw_b:.3f}   rare-word recall {rr:.3f}")
    print(f"  LM PMI         BAcc@10 {pmi_b:.3f}   rare-word recall {pr:.3f}")
    ok &= check("PMI beats the raw prior on balanced accuracy", pmi_b > raw_b,
                f"{pmi_b - raw_b:+.3f}")
    ok &= check("the gain comes from rare words", pr > rr, f"{rr:.3f} -> {pr:.3f}")

    print()
    print("=" * 84)
    print("5. The lambda constraint, with a per-lambda ceiling")
    print("=" * 84)
    ub = unlabelled_basis(pb.unlabelled(N_UNLAB), D_SIG)
    dec = SubspaceDecoder(ub).fit(xtr, ytr)
    neural = dec.loglik(xte)
    neural_blind = dec.loglik(rng.standard_normal(xte.shape))
    lm = lm_pmi(lm_ctx, log_freq)
    cu = chance + 0.03

    honest = calibrate_lambda(neural, neural_blind, lm, yte, chance_upper=cu)
    print(f"  honest  {honest.summary()}")
    ok &= check("honest decoder: brain adds over blind input, LM held fixed",
                honest.brain_gain > 0.05, f"{honest.brain_gain:+.3f}")

    leaky_blind = neural_blind + 6.0 * np.eye(K)[yte]
    leaky = calibrate_lambda(neural, leaky_blind, lm, yte, chance_upper=cu)
    b0, c0 = leaky.curve[0][2], leaky.ceilings[0]
    print(f"  leaky   blind channel at lambda=0: {b0:.3f}  ceiling {c0:.3f}")
    ok &= check("leaky blind channel is caught at lambda=0", b0 > c0,
                "per-lambda ceiling tracks chance when no LM is in the score")
    n_ok_h = sum(b <= c for (_, _, b), c in zip(honest.curve, honest.ceilings))
    n_ok_l = sum(b <= c for (_, _, b), c in zip(leaky.curve, leaky.ceilings))
    print(f"  eligible lambdas: honest {n_ok_h}/{len(honest.curve)}   "
          f"leaky {n_ok_l}/{len(leaky.curve)}")
    ok &= check("constraint rejects more lambdas for the leaky channel", n_ok_l < n_ok_h)

    print()
    print("=" * 84)
    if ok:
        print("DECODE VALIDATED. R1 is real. Of the design's three mitigations, CFG and")
        print("a label-fitted subspace both fail; a subspace from unlabelled data works.")
        return 0
    print("DECODE FAILED validation.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
