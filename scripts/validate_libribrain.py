"""Validate the LibriBrain audit pipeline on synthetic data with known answers.

The real-data audit (``scripts/run_libribrain_audit.py``) re-measures the lost
version's findings. Before trusting it on MEG, every piece is checked here on
synthetic files laid out exactly like the Hugging Face release:

1. Runs are found wherever they were published -- including the partial
   ``serialised_competition/..._desc-firsthalf`` files pnpl's loader misses.
2. Windows: only trials fully inside the recording are kept (partial releases
   end mid-chapter); the session normalisation is an invertible affine; the
   125 Hz decimation gives exact 64 ms patch means.
3. Timing features never reveal a word that starts after the window ends.
4. The vocabulary counts each spoken word once, not once per listener.
5. Occurrence shuffling moves all listeners' trials of a word together.
6. THE TIMING CONTROL, falsified both ways: when the label lives in word
   timing, the onset-only signal decodes it; when it lives in the word's own
   waveform, the onset-only signal is at chance.
7. The timing classifier finds timing information and nothing in its absence.
8. LM PMI earns nothing from word frequency.
9. The timing-matched test and information beyond covariates, both ways.
10. MEG-MASC events: word rows only, pseudo-word rows dropped, curly
    apostrophes straightened, one key per word occurrence, and the same timing
    features as LibriBrain's.

Run:  .venv/Scripts/python.exe scripts/validate_libribrain.py
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from neurocast.audit.fake_signals import onset_only  # noqa: E402
from neurocast.data.libribrain import extract_windows, find_runs, read_words  # noqa: E402
from neurocast.decode.baselines import BigramPMI, TimingClassifier  # noqa: E402
from neurocast.decode.decoders import LinearPooled, patch_features  # noqa: E402
from neurocast.decode.metrics import balanced_accuracy_at_k  # noqa: E402
from neurocast.decode.words import label_trials, timing_features, vocabulary, word_table  # noqa: E402

FS = 250.0


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
    return ok


HEADER = "idx\twavile\tkind\tsegment\tsentenceidx\twordidx\tphonemeidx\ttimemeg\ttimeds\ttimechapter\ttimesentence\tduration\n"


def write_events(path: Path, words, onsets, durations) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [HEADER]
    for i, (w, o, d) in enumerate(zip(words, onsets, durations)):
        lines.append(f"{i}\tsegments/\tword\t{w}\t{i // 10}.0\t{i % 10}.0\t\t{o}\t{o}\t{o}\t{o}\t{d}\n")
        lines.append(f"{i}\tsegments/\tphoneme\tx_B\t{i // 10}.0\t{i % 10}.0\t0.0\t{o}\t{o}\t{o}\t{o}\t{d}\n")
    path.write_text("".join(lines), encoding="utf-8")


def write_h5(path: Path, x: np.ndarray) -> None:
    import h5py

    path.parent.mkdir(parents=True, exist_ok=True)
    n = x.shape[0]
    with h5py.File(path, "w") as f:
        f.create_dataset("data", data=x.astype(np.float32))
        f.create_dataset("times", data=np.arange(x.shape[1]) / FS)
        f.attrs["sample_frequency"] = FS
        f.attrs["channel_names"] = ", ".join(f"MEG{i:04d}" for i in range(n))
        f.attrs["channel_types"] = ", ".join("mag" if i % 3 == 0 else "grad" for i in range(n))


def main() -> int:
    ok = True
    rng = np.random.default_rng(0)
    tmp = Path(tempfile.mkdtemp(prefix="neurocast-libribrain-"))
    try:
        base = tmp / "LibriBrain100" / "Sherlock1" / "derivatives"
        vocab_words = ["the", "a", "holmes", "watson", "said"]
        n_words, gap = 300, 0.6
        words = [vocab_words[i % 5] for i in range(n_words)]
        onsets = 1.0 + gap * np.arange(n_words)
        durs = np.full(n_words, 0.3)
        proc = "proc-bads+headpos+sss+notch+bp+ds"
        n_t_full = int((onsets[-1] + 2.0) * FS)
        for sub, desc, frac in (("1", None, 1.0), ("13", "firsthalf", 0.5)):
            write_events(base / "events" / f"sub-{sub}_ses-12_task-Sherlock1_run-1_events.tsv",
                         words, onsets, durs)
            folder = "serialised" if desc is None else "serialised_competition"
            tag = "" if desc is None else f"_desc-{desc}"
            x = 1e-12 * rng.standard_normal((6, int(n_t_full * frac)))
            write_h5(base / folder / f"sub-{sub}_ses-12_task-Sherlock1_run-1_{proc}{tag}_meg.h5", x)

        print("=" * 78)
        print("1. Finding runs, including the partial releases")
        print("=" * 78)
        runs = find_runs(root=tmp / "LibriBrain100", require_meg=True)
        descs = {r.subject: r.desc for r in runs}
        ok &= check("both listeners found, the partial one under serialised_competition",
                    descs == {"1": "full", "13": "firsthalf"}, str(descs))

        print()
        print("=" * 78)
        print("2. Windows: coverage, normalisation, decimation")
        print("=" * 78)
        r1, r13 = sorted(runs, key=lambda r: int(r.subject))
        w_on = read_words(r1.events)["onset"].to_numpy()
        w1, k1 = extract_windows(r1, w_on)
        w13, k13 = extract_windows(r13, w_on)
        ok &= check("full release keeps every window", bool(k1.all()), f"{k1.sum()}/{len(k1)}")
        ok &= check("first-half release keeps only windows inside its recording",
                    0.4 < k13.mean() < 0.6 and not k13[-1] and k13[0],
                    f"{k13.sum()}/{len(k13)} kept")
        ok &= check("windows are (n, C, 64) float16 at 125 Hz",
                    w1.shape == (len(w_on), 6, 64) and w1.dtype == np.float16)
        z = w1.astype(np.float32)
        ok &= check("session-scalar output is centred and O(1)",
                    abs(float(np.median(z))) < 0.1 and 0.5 < float(z.std()) < 2.0,
                    f"median {np.median(z):+.3f}, sd {z.std():.3f}")
        win = np.arange(128, dtype=float)[None, None]
        dec = win.reshape(1, 1, 64, 2).mean(-1)
        ok &= check("125 Hz patch means equal the 250 Hz 64 ms means",
                    np.allclose(patch_features(dec), win.reshape(1, 1, 8, 16).mean(-1).reshape(1, -1)))

        print()
        print("=" * 78)
        print("3-5. Timing features, vocabulary, occurrence shuffle")
        print("=" * 78)
        tf = timing_features(read_words(r1.events))
        ok &= check("a next word after the window is absent, never a number",
                    bool(np.isnan(tf["next1"]).all()), "words are 0.6 s apart; window 0.512 s")
        tf_close = timing_features(read_words(r1.events), window_s=0.7)
        ok &= check("...and present when it starts inside the window",
                    bool(np.allclose(tf_close["next1"].dropna(), gap)))
        two = word_table(find_runs(root=tmp / "LibriBrain100"))
        counts = two.drop_duplicates("occ")["word"].value_counts()
        ok &= check("the vocabulary counts each spoken word once, not per listener",
                    int(counts.max()) == n_words // 5 and vocabulary(two, 3)[0] in vocab_words)
        from run_libribrain_audit import occurrence_shuffle  # noqa: E402
        t = label_trials(two, vocab_words)
        ys = occurrence_shuffle(t["label"].to_numpy(), t["occ"].to_numpy(), rng)
        grouped = t.assign(ys=ys).groupby("occ")["ys"].nunique().max()
        ok &= check("occurrence shuffle moves both listeners' trials together",
                    grouped == 1 and sorted(ys) == sorted(t["label"]), "one label per occurrence")

        print()
        print("=" * 78)
        print("6. The timing control, falsified both ways")
        print("=" * 78)
        sf, n, c, tlen = 125.0, 1200, 8, 64
        k_cls = 4
        erf = np.sin(np.linspace(0, 3 * np.pi, tlen))[None] * rng.standard_normal((c, 1))
        templates = rng.standard_normal((k_cls, c, tlen))

        def make(label_in_timing: bool, seed: int):
            g = np.random.default_rng(seed)
            y = g.integers(0, k_cls, n)
            on = np.cumsum(np.full(n, 2.0))                        # trials far apart
            nxt = 0.08 + 0.08 * y if label_in_timing else np.full(n, 0.2)
            all_on = np.sort(np.r_[on, on + nxt])
            w = np.empty((n, c, tlen), np.float32)
            for i in range(n):
                s = int(round(nxt[i] * sf))
                w[i] = erf + 0.6 * g.standard_normal((c, tlen))
                w[i, :, s:] += erf[:, : tlen - s]                   # next word's response
                if not label_in_timing:
                    w[i] += 0.8 * templates[y[i]]                   # word-specific waveform
            return w.astype(np.float16), y, on, {"r": all_on}

        for label_in_timing, want in ((True, "decodable"), (False, "chance")):
            wf, yf, onf, allf = make(label_in_timing, 1)
            wt, yt, ont, allt = make(label_in_timing, 2)
            occ = np.arange(n).astype(str)
            run = np.full(n, "r")
            real = LinearPooled(n_components=32).fit(patch_features(wf), yf, occ)
            fake_f = onset_only(wf, onf, run, allf, sf, rng)
            fake_t = onset_only(wt, ont, run, allt, sf, rng)
            fake = LinearPooled(n_components=32).fit(patch_features(fake_f), yf, occ)
            a_real = balanced_accuracy_at_k(real.scores(patch_features(wt), k_cls), yt, 1)
            a_fake = balanced_accuracy_at_k(fake.scores(patch_features(fake_t), k_cls), yt, 1)
            where = "timing" if label_in_timing else "the word's own waveform"
            print(f"  label in {where:<24} real {a_real:.3f}   onset-only {a_fake:.3f}  (chance 0.25)")
            if label_in_timing:
                ok &= check("label in timing: onset-only input decodes it", a_fake > 0.6)
            else:
                ok &= check("label in the waveform: real input decodes it", a_real > 0.6)
                ok &= check("...and onset-only input is at chance", a_fake < 0.33)

        print()
        print("=" * 78)
        print("7-8. No-brain baselines")
        print("=" * 78)
        import pandas as pd

        def timing_df(y, informative, g):
            nxt = 0.1 + 0.08 * y + 0.01 * g.standard_normal(len(y)) if informative else g.uniform(0.1, 0.4, len(y))
            d = {c: np.full(len(y), np.nan) for c in ("next2", "next3", "next4")}
            return pd.DataFrame({"next1": nxt, **d, "gap_prev": g.uniform(0, 0.1, len(y)),
                                 "pos_in_sentence": g.integers(0, 10, len(y)).astype(float)})

        for informative in (True, False):
            g = np.random.default_rng(3)
            ytr, yte = g.integers(0, 4, 4000), g.integers(0, 4, 2000)
            clf = TimingClassifier().fit(timing_df(ytr, informative, g), ytr, 4)
            acc = balanced_accuracy_at_k(clf.log_proba(timing_df(yte, informative, g)), yte, 1)
            if informative:
                ok &= check("timing classifier recovers timing information", acc > 0.9, f"{acc:.3f}")
            else:
                ok &= check("...and finds nothing when timing is uninformative", acc < 0.3, f"{acc:.3f}")

        freq = ["the"] * 500 + ["a"] * 300 + ["holmes"] * 20 + ["watson"] * 20
        seq = list(np.random.default_rng(4).permutation(freq))
        lm = BigramPMI().fit([seq])
        vocab = ["the", "a", "holmes", "watson"]
        y = np.array([vocab.index(w) for w in seq[1:]])
        s = lm.pmi(seq[:-1], vocab)
        b = balanced_accuracy_at_k(s, y, 1)
        # A raw frequency prior names "the" every time (60% plain accuracy for free);
        # in a sequence with no context, PMI must not systematically prefer it.
        share_the = float((s.argmax(1) == 0).mean())
        ok &= check("LM PMI does not reward frequency in a context-free sequence",
                    b < 0.35 and share_the < 0.5,
                    f"BAcc {b:.3f} (chance 0.25); picks 'the' {100 * share_the:.0f}% of the "
                    "time vs 100% for a frequency prior")
        ok &= matched_tests()
        ok &= megmasc_tests(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    print("=" * 78)
    if ok:
        print("LIBRIBRAIN PIPELINE VALIDATED: partial releases found, coverage enforced,")
        print("the onset-only control carries timing and nothing else, and MEG-MASC's")
        print("events parse into the same word table.")
        return 0
    print("LIBRIBRAIN PIPELINE FAILED validation.")
    return 1


def megmasc_tests(tmp: Path) -> bool:
    """Section 10: MEG-MASC's events.tsv, parsed into the audit's word table."""
    from neurocast.data import megmasc

    print()
    print("=" * 78)
    print("10. MEG-MASC events")
    print("=" * 78)
    rows = []

    def ev(onset, dur, kind, **kw):
        meta = {"story": "lw1", "kind": kind, "sound": "stimuli/audio/lw1_0.wav", **kw}
        rows.append(f"{onset}\t{dur}\t{meta!r}\t1\t{int(onset * 1000)}")

    ev(10.0, 0.0, "sound", start=0.0)
    for i, (w, on, d) in enumerate([("Tara", 10.0, 0.3), ("it\u2019s", 10.31, 0.2), ("a", 10.51, 0.1),
                                    ("dog", 10.7, 0.4)]):
        ev(on, 0.08, "phoneme", phoneme="t_B", sequence_id=0.0, word_index=float(i),
           condition="sentence")
        ev(on, d, "word", word=w, start=on - 10.0, sequence_id=0.0, word_index=float(i),
           condition="sentence")
    ev(10.75, 0.2, "word", word="ro", start=0.75, sequence_id=0.0, word_index=3.0,
       condition="pseudo_words")
    ev(12.0, 0.3, "word", word="tiny", start=2.0, sequence_id=1.0, word_index=0.0,
       condition="word_list")
    f = megmasc.events_path("01", "0", "0", tmp / "MEG-MASC")
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text("\ufeffonset\tduration\ttrial_type\tvalue\tsample\n" + "\n".join(rows) + "\n",
                 encoding="utf-8")
    w = megmasc.word_table("01", "0", ["0"], window_s=0.512, root=tmp / "MEG-MASC")
    ok = True
    ok &= check("word rows only, pseudo-words dropped, in order",
                w["word"].tolist() == ["tara", "it's", "a", "dog", "tiny"], str(w["word"].tolist()))
    ok &= check("one key per occurrence (task, condition, sequence, index)", w["occ"].is_unique)
    ok &= check("timing features as LibriBrain's: next onset, NaN beyond the window",
                abs(w["next1"].iloc[0] - 0.31) < 1e-9 and np.isnan(w["next3"].iloc[0])
                and w["condition"].tolist()[-1] == "word_list",
                f"next1 {w['next1'].iloc[0]:.2f}, next3 {w['next3'].iloc[0]}")
    ok &= check("each word's start in its stimulus file is kept, for loudness",
                np.allclose(w["t_sound"].to_numpy()[:4], [0.0, 0.31, 0.51, 0.7]))
    return ok


def matched_tests() -> bool:
    """Section 9: the timing-matched test and information beyond covariates."""
    import pandas as pd

    from neurocast.audit.timing_matched import info_beyond_covariates, matched_gain, strata

    print()
    print("=" * 78)
    print("9. Timing-matched test and information beyond covariates")
    print("=" * 78)
    ok = True
    g = np.random.default_rng(7)
    k, n_occ, listeners = 10, 600, 3
    y_occ = g.integers(0, k, n_occ)
    pair = y_occ // 2                          # words 2j, 2j+1 share timing and length
    df = pd.DataFrame({
        "next1": np.repeat(0.12 + 0.06 * pair + 0.004 * g.standard_normal(n_occ), listeners),
        "next2": np.nan,
        "gap_prev": np.repeat(g.uniform(0, 0.02, n_occ), listeners),
        "duration": np.repeat(0.2 + 0.05 * pair, listeners),
        "loudness_db": np.repeat(g.normal(-30, 3, n_occ), listeners),
    })
    y = np.repeat(y_occ, listeners)
    occ = np.repeat(np.arange(n_occ), listeners).astype(str)
    groups = strata(df)
    mu = 0.12 + 0.06 * (np.arange(k) // 2)
    timing_scores = -((df["next1"].to_numpy()[:, None] - mu[None]) ** 2) / 0.004**2
    brain_scores = 1.5 * np.eye(k)[y] + g.standard_normal((len(y), k))
    for name, s, want_low in (("timing-only scorer", timing_scores, True),
                              ("word-information scorer", brain_scores, False)):
        r = matched_gain(s, y, occ, groups, n_perm=100, seed=1)
        print(f"  {name:<24} unmatched gain {r.unmatched_gain:+.3f}  matched {r.gain:+.3f}  "
              f"surviving {100 * r.surviving:4.0f}%  [{r.verdict}]")
        if want_low:
            ok &= check("timing-only information does not survive matching",
                        r.surviving < 1 / 3 and r.verdict != "survives matching")
        else:
            ok &= check("word information independent of timing survives matching",
                        r.surviving > 2 / 3 and r.verdict == "survives matching")

    cov = df[["next1", "gap_prev", "duration", "loudness_db"]].to_numpy()
    noise = info_beyond_covariates(cov, g.standard_normal((len(y), k)), y, occ, n_boot=200)
    real = info_beyond_covariates(cov, brain_scores, y, occ, n_boot=200)
    freq = info_beyond_covariates(cov, np.tile(np.log(np.bincount(y, minlength=k) + 1.0),
                                               (len(y), 1)), y, occ, n_boot=200)
    print(f"  beyond covariates: noise {noise}   word information {real}   frequency {freq}")
    ok &= check("noise scores add nothing beyond covariates", noise.lo <= 0.0 <= noise.hi + 1e-3)
    ok &= check("frequency-only scores add exactly nothing", abs(freq.nats) < 1e-9)
    ok &= check("word information adds positive nats", real.lo > 0.05)

    # The MEG-MASC leak: noise scores, except that one group of trials -- a held-out
    # story -- carries the decoder's "never trained" marker on two words that occur
    # only in that group. The marker pattern identifies the group, the group
    # predicts those words, and a naive fit turned that into positive nats.
    story = np.repeat(g.integers(0, 4, n_occ), listeners)
    y_leak = np.where((story == 1) & (g.random(len(y)) < 0.3), k - 1 - (np.arange(len(y)) % 2),
                      y % (k - 2))
    marked = g.standard_normal((len(y), k))
    marked[np.ix_(story == 1, [k - 2, k - 1])] = -1e6
    leak = info_beyond_covariates(cov, marked, y_leak, occ, n_boot=200)
    print(f"  noise with a story-specific 'never trained' marker: {leak}")
    ok &= check("an untrained-class marker that identifies a group adds nothing",
                leak.lo <= 0.0 <= leak.hi + 1e-3 and leak.dropped == (k - 2, k - 1),
                f"classes left out {leak.dropped}")
    return ok


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
