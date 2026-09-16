"""Subject-identity leakage diagnostics. The H2 metric.

*The Identity Trap* (arXiv 2606.06647) audited LaBraM, CBraMod and REVE and
found frozen subject-identity variance at **13-89x a random null in 12/12
model-dataset pairs**, rising a further +10 to +63 percentage points under
fine-tuning. Much of what reads as clinical accuracy in that literature is
really *who the subject is*.

That paper does not test whether the **pretraining objective** causes the
leakage. This module exists to run that test, which is H2:

    forecasting objectives starve the identity shortcut, because a subject's
    nuisance structure is visible in the context window and therefore need not
    be stored in the weights

The headline statistic is a variance ratio against a permuted-subject null.
Ratio near 1 means the representation carries no more subject information than
chance; 13-89x is what the audited models scored.

Reported for **every checkpoint, next to accuracy**. Not an appendix.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["eta_squared", "IdentityReport", "subject_variance", "linear_probe_accuracy"]


def eta_squared(z: np.ndarray, groups: np.ndarray) -> float:
    """Fraction of representation variance explained by group membership.

    ``SS_between / SS_total`` over the multivariate representation. Bounded in
    [0, 1]; 0 means group membership explains nothing.
    """
    z = np.asarray(z, dtype=float)
    groups = np.asarray(groups)
    if z.ndim != 2:
        raise ValueError(f"expected (n_samples, d), got {z.shape}")
    if len(groups) != z.shape[0]:
        raise ValueError(f"{z.shape[0]} samples but {len(groups)} group labels")

    mu = z.mean(axis=0, keepdims=True)
    ss_total = float(((z - mu) ** 2).sum())
    if ss_total <= 0:
        return 0.0
    ss_between = 0.0
    for g in np.unique(groups):
        m = groups == g
        ss_between += int(m.sum()) * float(((z[m].mean(axis=0) - mu[0]) ** 2).sum())
    return ss_between / ss_total


def linear_probe_accuracy(
    z: np.ndarray, labels: np.ndarray, *, n_folds: int = 4, seed: int = 0
) -> float:
    """Cross-validated nearest-centroid accuracy. A cheap linear probe.

    Deliberately weak: if even a nearest-centroid probe recovers subject identity
    from a frozen representation, the leakage is not subtle.
    """
    z = np.asarray(z, dtype=float)
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(labels))
    folds = np.array_split(order, n_folds)
    classes = np.unique(labels)

    accs = []
    for i in range(n_folds):
        te = folds[i]
        tr = np.concatenate([folds[j] for j in range(n_folds) if j != i])
        cents, present = [], []
        for c in classes:
            m = labels[tr] == c
            if m.any():
                cents.append(z[tr][m].mean(axis=0))
                present.append(c)
        if len(present) < 2:
            continue
        cents = np.stack(cents)
        d = ((z[te][:, None, :] - cents[None]) ** 2).sum(-1)
        pred = np.array(present)[np.argmin(d, axis=1)]
        accs.append(float((pred == labels[te]).mean()))
    return float(np.mean(accs)) if accs else float("nan")


@dataclass(frozen=True)
class IdentityReport:
    """Subject-leakage measurement for one checkpoint."""

    eta2_subject: float
    eta2_null_mean: float
    eta2_null_p95: float
    ratio_to_null: float
    probe_accuracy: float
    chance_accuracy: float
    eta2_label: float | None = None
    n_permutations: int = 200

    @property
    def leaking(self) -> bool:
        """True if subject variance exceeds the permuted-subject null."""
        return self.eta2_subject > self.eta2_null_p95

    @property
    def verdict(self) -> str:
        r = self.ratio_to_null
        if r < 2:
            return "clean"
        if r < 5:
            return "mild"
        if r < 13:
            return "substantial"
        return "identity-trapped"

    def summary(self) -> str:
        lab = (
            f"  label eta2 {self.eta2_label:.4f}" if self.eta2_label is not None else ""
        )
        return (
            f"subject eta2 {self.eta2_subject:.4f}  "
            f"null {self.eta2_null_mean:.4f}  "
            f"ratio {self.ratio_to_null:6.1f}x  "
            f"probe {self.probe_accuracy:.3f} (chance {self.chance_accuracy:.3f})  "
            f"[{self.verdict}]{lab}"
        )


def subject_variance(
    z: np.ndarray,
    subjects: np.ndarray,
    *,
    labels: np.ndarray | None = None,
    n_permutations: int = 200,
    seed: int = 0,
) -> IdentityReport:
    """Measure how much of a frozen representation is subject identity.

    Parameters
    ----------
    z
        ``(n_samples, d)`` frozen representations -- pooled encoder outputs, not
        fine-tuned task features.
    subjects
        ``(n_samples,)`` subject ids.
    labels
        Optional task labels. When given, the report also carries label eta^2 so
        the two can be compared directly: a representation where subject
        variance dwarfs label variance is not a task representation.

    Notes
    -----
    The null permutes subject assignment, which preserves the representation's
    own covariance structure and the group sizes. That matters -- a null built by
    resampling the representation would be far too permissive, and would make
    almost any model look clean.
    """
    z = np.asarray(z, dtype=float)
    subjects = np.asarray(subjects)
    rng = np.random.default_rng(seed)

    observed = eta_squared(z, subjects)
    null = np.array(
        [eta_squared(z, subjects[rng.permutation(len(subjects))])
         for _ in range(n_permutations)]
    )
    null_mean = float(null.mean())
    null_p95 = float(np.percentile(null, 95))

    probe = linear_probe_accuracy(z, subjects, seed=seed)
    chance = 1.0 / len(np.unique(subjects))

    return IdentityReport(
        eta2_subject=observed,
        eta2_null_mean=null_mean,
        eta2_null_p95=null_p95,
        ratio_to_null=observed / max(null_mean, 1e-12),
        probe_accuracy=probe,
        chance_accuracy=chance,
        eta2_label=eta_squared(z, np.asarray(labels)) if labels is not None else None,
        n_permutations=n_permutations,
    )
