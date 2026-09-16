"""Prove the split machinery refuses leaky splits.

Same philosophy as the control harness: a guard that never fires is not a guard.
Each check below constructs a split that IS leaky and asserts the module
catches it, then constructs the clean equivalent and asserts it passes.

Run:  .venv/Scripts/python.exe scripts/validate_splits.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neurocast.data.canonical import CANONICAL_FS  # noqa: E402
from neurocast.data.splits import (  # noqa: E402
    DEFAULT_GUARD_SECONDS,
    Segment,
    Split,
    assert_disjoint,
    assert_fixed_duration,
    blocked_time_split,
    stimulus_split,
    subject_split,
)

GUARD = int(DEFAULT_GUARD_SECONDS * CANONICAL_FS)  # 60 s -> 15000 samples


def expect_raises(name: str, fn) -> bool:
    try:
        fn()
    except AssertionError as exc:
        print(f"[PASS] {name}\n         caught: {str(exc)[:96]}")
        return True
    except Exception as exc:  # pragma: no cover
        print(f"[FAIL] {name} -- wrong exception type {type(exc).__name__}: {exc}")
        return False
    print(f"[FAIL] {name} -- leak was NOT caught")
    return False


def expect_ok(name: str, fn) -> bool:
    try:
        fn()
    except Exception as exc:
        print(f"[FAIL] {name} -- clean split rejected: {exc}")
        return False
    print(f"[PASS] {name}")
    return True


def main() -> int:
    ok = True
    hour = int(3600 * CANONICAL_FS)

    print("=" * 78)
    print("1. Temporally-blocked split with guard bands")
    print("=" * 78)

    segs = [
        Segment("sub-0", "ses-1", "run-1", 0, 4 * hour, stimulus_id="sherlock-01"),
        Segment("sub-1", "ses-1", "run-1", 0, 2 * hour, stimulus_id="sherlock-02"),
    ]
    splits = blocked_time_split(
        segs, {"train": 0.8, "val": 0.1, "test": 0.1}, guard_samples=GUARD
    )
    for s in splits.values():
        print("  " + s.summary())

    ok &= expect_ok(
        "guarded blocked split is disjoint at full guard width",
        lambda: assert_disjoint(splits, guard_samples=GUARD),
    )

    leaky = blocked_time_split(
        segs, {"train": 0.8, "val": 0.1, "test": 0.1}, guard_samples=0
    )
    ok &= expect_raises(
        "unguarded blocked split is rejected when a guard is required",
        lambda: assert_disjoint(leaky, guard_samples=GUARD),
    )

    print()
    print("=" * 78)
    print("2. Subject-disjoint split")
    print("=" * 78)

    multi = [
        Segment(f"sub-{i}", "ses-1", "run-1", 0, hour, stimulus_id="sherlock-01")
        for i in range(4)
    ]
    clean = subject_split(
        multi, {"sub-0": "train", "sub-1": "train", "sub-2": "test", "sub-3": "test"}
    )
    for s in clean.values():
        print("  " + s.summary())
    ok &= expect_ok(
        "subject-disjoint split passes",
        lambda: assert_disjoint(clean, subject_disjoint=True),
    )

    shared = {
        "train": Split("train", (multi[0], multi[1], multi[2])),
        "test": Split("test", (multi[2], multi[3])),
    }
    ok &= expect_raises(
        "shared subject across splits is rejected",
        lambda: assert_disjoint(shared, subject_disjoint=True),
    )

    print()
    print("=" * 78)
    print("3. Stimulus-disjoint split (the one everyone forgets)")
    print("=" * 78)

    # Every subject hears the same two books -- the LibriBrain situation.
    both_books = [
        Segment(f"sub-{i}", "ses-1", f"run-{b}", b * hour, (b + 1) * hour,
                stimulus_id=f"sherlock-{b:02d}")
        for i in range(3)
        for b in range(2)
    ]
    by_book = stimulus_split(
        both_books, {"sherlock-00": "train", "sherlock-01": "test"}
    )
    for s in by_book.values():
        print("  " + s.summary())
    ok &= expect_ok(
        "book-disjoint split passes",
        lambda: assert_disjoint(by_book, stimulus_disjoint=True),
    )

    # Subject-disjoint but stimulus-SHARED: passes a naive check, leaks text.
    subject_only = subject_split(
        both_books, {"sub-0": "train", "sub-1": "train", "sub-2": "test"}
    )
    ok &= expect_ok(
        "subject-disjoint-only split passes a subject check (as expected)",
        lambda: assert_disjoint(subject_only, subject_disjoint=True),
    )
    ok &= expect_raises(
        "...but is rejected once stimulus disjointness is required",
        lambda: assert_disjoint(subject_only, stimulus_disjoint=True),
    )
    print("         ^ this is the pair that matters: a split can be perfectly")
    print("           subject-disjoint and still let the model memorise the text.")

    print()
    print("=" * 78)
    print("4. Fixed-duration evaluation windows")
    print("=" * 78)

    win = int(0.4 * CANONICAL_FS)  # PNPL word window: 0.2-0.6 s = 100 samples
    good = [Segment("sub-0", "ses-1", "run-1", i * 1000, i * 1000 + win) for i in range(20)]
    ok &= expect_ok(
        f"uniform {win}-sample windows pass",
        lambda: assert_fixed_duration(good, win),
    )
    bad = good + [Segment("sub-0", "ses-1", "run-1", 90_000, 90_000 + win + 17)]
    ok &= expect_raises(
        "a single variable-length window is rejected",
        lambda: assert_fixed_duration(bad, win),
    )

    print()
    print("=" * 78)
    print("5. Fingerprint stability")
    print("=" * 78)
    a = Split("x", tuple(good))
    b = Split("x", tuple(reversed(good)))
    same = a.fingerprint() == b.fingerprint()
    print(f"  fingerprint (ordered)   {a.fingerprint()}")
    print(f"  fingerprint (reversed)  {b.fingerprint()}")
    ok &= same
    print(f"[{'PASS' if same else 'FAIL'}] fingerprint is order-independent")
    differs = Split("x", tuple(bad)).fingerprint() != a.fingerprint()
    ok &= differs
    print(f"[{'PASS' if differs else 'FAIL'}] fingerprint changes when content changes")

    print()
    print("=" * 78)
    if ok:
        print("SPLIT MACHINERY VALIDATED: every leak above was caught.")
        return 0
    print("SPLIT MACHINERY FAILED validation.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
