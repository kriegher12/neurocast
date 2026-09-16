"""Verify episodic sampling cannot leak, and that nuisance descriptors are real.

The guard band is the headline. Sampling a prompt from ten seconds before the
query would let the model "adapt to the subject" by reading an autocorrelated
neighbour -- slow drift, head position, alpha state -- and produce a spectacular
H3 result that means nothing. So the guard is checked here **independently**,
with raw index arithmetic, rather than by calling the same overlap helper the
sampler uses. A guard that polices itself is not a guard.

Also checked:
  * prompt and query never share a sample
  * prompt-free, cross-session and mismatched episodes occur at their target rates
  * K respects the half-of-available cap
  * the aperiodic fit recovers a known 1/f exponent
  * prompt swapping exchanges prompts while keeping queries
  * counterfactual nuisance injection moves the descriptors it should

Run:  .venv/Scripts/python.exe scripts/validate_adapt.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neurocast.adapt import nuisance  # noqa: E402
from neurocast.adapt.episodes import EpisodeSampler  # noqa: E402
from neurocast.data.splits import Segment  # noqa: E402

FS = 250.0
GUARD_S = 60.0
N_EPISODES = 3000


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
    return ok


def build_corpus() -> list[Segment]:
    """Three subjects: one deep and multi-session, two shallow."""
    hour = int(3600 * FS)
    segs = [
        Segment("sub-0", "ses-1", "run-1", 0, 3 * hour, "sherlock-01"),
        Segment("sub-0", "ses-2", "run-1", 0, 2 * hour, "sherlock-02"),
        Segment("sub-1", "ses-1", "run-1", 0, hour // 2, "sherlock-01"),
        Segment("sub-2", "ses-1", "run-1", 0, hour // 3, "sherlock-03"),
    ]
    return segs


def pink(rng: np.random.Generator, n_ch: int, n_t: int, exponent: float) -> np.ndarray:
    """Noise with PSD proportional to f^-exponent."""
    spec = np.fft.rfft(rng.standard_normal((n_ch, n_t)), axis=-1)
    f = np.fft.rfftfreq(n_t, d=1.0 / FS)
    scale = np.ones_like(f)
    scale[1:] = f[1:] ** (-exponent / 2.0)
    return np.fft.irfft(spec * scale, n=n_t, axis=-1)


def main() -> int:
    rng = np.random.default_rng(0)
    ok = True

    segs = build_corpus()
    sampler = EpisodeSampler(segs, fs=FS, query_samples=4096, guard_seconds=GUARD_S)
    guard = int(GUARD_S * FS)

    print("=" * 78)
    print("Corpus")
    print("=" * 78)
    for s in sampler.subjects:
        print(f"  {s}: {sampler.available_minutes(s):7.1f} min available")

    episodes = sampler.sample_many(N_EPISODES, rng)

    print()
    print("=" * 78)
    print(f"1. GUARD BAND -- independently verified over {N_EPISODES:,} episodes")
    print("=" * 78)

    violations = []
    overlaps = []
    closest = np.inf
    for ep in episodes:
        q = ep.query
        for c in ep.prompt:
            if (c.subject, c.session, c.run) != (q.subject, q.session, q.run):
                continue  # different recording: guard does not apply
            # Raw arithmetic, not Segment.overlaps -- deliberately independent.
            gap = c.start - q.stop if c.start >= q.stop else q.start - c.stop
            closest = min(closest, gap)
            if gap < 0:
                overlaps.append((ep, c))
            elif gap < guard:
                violations.append((ep, c, gap))

    same_rec = sum(
        1
        for ep in episodes
        for c in ep.prompt
        if (c.subject, c.session, c.run) == (ep.query.subject, ep.query.session, ep.query.run)
    )
    print(f"  same-recording prompt chunks examined: {same_rec:,}")
    print(f"  closest approach to a query: {closest / FS:.1f} s "
          f"(guard is {GUARD_S:.0f} s)")

    ok &= check("no prompt chunk overlaps its query", not overlaps,
                f"{len(overlaps)} overlaps")
    ok &= check("no prompt chunk sits inside the guard band", not violations,
                f"{len(violations)} violations")
    ok &= check("the guard is actually binding (some chunk lands near it)",
                closest < 3 * guard, f"closest {closest / FS:.1f}s")

    print()
    print("=" * 78)
    print("2. Episode composition matches the intended rates")
    print("=" * 78)

    n = len(episodes)
    free = sum(e.is_prompt_free for e in episodes) / n
    mism = sum(e.mismatched for e in episodes) / n
    cross = sum(e.cross_session for e in episodes if not e.is_prompt_free)
    cross /= max(sum(not e.is_prompt_free for e in episodes), 1)
    print(f"  prompt-free   {free:.3f}  (target 0.10)")
    print(f"  mismatched    {mism:.3f}  (target 0.15)")
    print(f"  cross-session {cross:.3f}  (target 0.30, capped by availability)")

    ok &= check("prompt-free rate near target", 0.07 < free < 0.14, f"{free:.3f}")
    ok &= check("mismatch rate near target", 0.11 < mism < 0.20, f"{mism:.3f}")
    ok &= check("mismatched episodes really are cross-subject",
                all(e.query_subject != e.prompt_subject
                    for e in episodes if e.mismatched))
    ok &= check("matched episodes really are same-subject",
                all(e.query_subject == e.prompt_subject
                    for e in episodes if not e.mismatched))

    truly_cross = [
        e for e in episodes if e.cross_session and e.prompt and not e.mismatched
    ]
    ok &= check("cross-session prompts use a different session",
                all(c.session != e.query.session for e in truly_cross for c in e.prompt),
                f"{len(truly_cross)} checked")

    print()
    print("=" * 78)
    print("3. K respects the half-of-available cap")
    print("=" * 78)

    worst = 0.0
    for e in episodes:
        if e.prompt:
            frac = e.prompt_minutes / sampler.available_minutes(e.prompt_subject)
            worst = max(worst, frac)
    ks = np.array([e.prompt_minutes for e in episodes if e.prompt])
    print(f"  K range {ks.min():.2f} - {ks.max():.2f} min, median {np.median(ks):.2f}")
    print(f"  worst prompt share of a subject's data: {worst:.3f}")
    ok &= check("no prompt exceeds half the subject's material", worst <= 0.5 + 1e-9,
                f"{worst:.3f}")
    ok &= check("K spans at least a decade (log-uniform schedule)",
                ks.max() / max(ks.min(), 1e-9) > 10, f"{ks.max() / ks.min():.0f}x")

    print()
    print("=" * 78)
    print("4. Aperiodic fit recovers a known 1/f exponent")
    print("=" * 78)

    n_t = int(20 * FS)
    errs = []
    for target in (0.5, 1.0, 1.5, 2.0):
        x = pink(rng, 16, n_t, target)
        desc = nuisance.compute(x, FS)
        est = float(desc.exponent.mean())
        errs.append(abs(est - target))
        print(f"  target {target:.1f}   estimated {est:5.2f}   error {abs(est - target):.3f}")
    ok &= check("1/f exponent recovered across the range", max(errs) < 0.15,
                f"max error {max(errs):.3f}")

    desc = nuisance.compute(pink(rng, 8, n_t, 1.0), FS)
    print(f"  descriptors: {desc.summary()}")
    ok &= check("descriptor matrix has the expected shape",
                desc.to_matrix().shape == (8, 3 + len(nuisance.BANDS) + 1),
                f"{desc.to_matrix().shape}")

    print()
    print("=" * 78)
    print("5. Counterfactual nuisance injection")
    print("=" * 78)

    bumped = desc.perturb(d_exponent=0.3, gain_factor=1.2)
    d_exp = float((bumped.exponent - desc.exponent).mean())
    d_var = float((bumped.log_variance - desc.log_variance).mean())
    expected_var = 2 * np.log10(1.2)
    print(f"  exponent shift {d_exp:+.3f} (requested +0.300)")
    print(f"  log-variance shift {d_var:+.4f} (expected {expected_var:+.4f} for x1.2 gain)")
    ok &= check("exponent perturbation applies exactly", abs(d_exp - 0.3) < 1e-9)
    ok &= check("gain perturbation applies exactly", abs(d_var - expected_var) < 1e-9)

    print()
    print("=" * 78)
    print("6. Prompt swap keeps queries, exchanges prompts")
    print("=" * 78)

    a, b = next(
        (x, y)
        for x, y in zip(episodes, episodes[1:])
        if x.prompt and y.prompt and x.prompt_subject != y.prompt_subject
    )
    sa, sb = sampler.swap_prompts(a, b)
    ok &= check("queries are unchanged",
                sa.query == a.query and sb.query == b.query)
    ok &= check("prompts are exchanged",
                sa.prompt == b.prompt and sb.prompt == a.prompt)
    ok &= check("swapped episodes are flagged mismatched",
                sa.mismatched and sb.mismatched,
                "identity probe should follow the prompt, not the query")

    print()
    print("=" * 78)
    if ok:
        print("EPISODIC SAMPLING VALIDATED: guard bands hold, rates on target,")
        print("nuisance descriptors recover ground truth.")
        return 0
    print("EPISODIC SAMPLING FAILED validation.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
