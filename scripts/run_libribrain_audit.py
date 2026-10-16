"""Re-run the lost version's LibriBrain100 audit and compare with the pitch.

The only record of the earlier, lost version of this project is a faculty pitch
(``docs/recovered/neurocast-pitch.html``) quoting real-data results on 32
listeners. This script re-measures them on freshly downloaded data and prints
each number next to the one the pitch claims. It never edits the claims: a
mismatch is reported, not tuned away.

    .venv/Scripts/python.exe scripts/run_libribrain_audit.py --stage nobrain
    .venv/Scripts/python.exe scripts/run_libribrain_audit.py --stage brain

Protocol (pitch, "ML configuration reference"): 32 broad listeners; fit on
Sherlock1 session 11 (chapter 11), test on session 12 (chapter 12); the 50 most
frequent words; windows of 0.512 s from word onset; balanced accuracy top-1
(competition metric, chance 0.02) and top-10 (chance 0.20).

The 2026 competition itself scores 1.0 s windows over a fixed vocabulary
(``pnpl.competition``: ``WINDOW_SECONDS = 1.0``, ``PRIMARY_VOCAB``), and its
sentence test data ships each word's onset. ``--window 1.024 --vocab
competition`` re-runs every stage under that protocol (1.024 s = 16 whole 64 ms
patches) into ``runs/libribrain_audit_w1024_competition/``; the default
protocol's results are never overwritten.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from neurocast.data.libribrain import find_runs  # noqa: E402
from neurocast.data.paths import dataset_dir  # noqa: E402
from neurocast.decode.baselines import (  # noqa: E402
    BigramPMI,
    TimingClassifier,
    combine_scores,
    tune_weight,
)
from neurocast.decode.metrics import balanced_accuracy_at_k  # noqa: E402
from neurocast.decode.words import WINDOW_S, label_trials, vocabulary, word_table  # noqa: E402

BROAD = {str(i) for i in range(1, 33)}
TEST_CHAPTERS = {("Sherlock1", "11"), ("Sherlock1", "12")}
OUT = ROOT / "runs" / "libribrain_audit"


def bacc(scores, y):
    return balanced_accuracy_at_k(scores, y, 1), balanced_accuracy_at_k(scores, y, 10)


def row(name, got, claim, note=""):
    g1, g10 = got
    print(f"  {name:<34} {g1:7.3f} {g10:8.3f}   {claim:<22} {note}")


def competition_vocabulary() -> list[str]:
    """The 2026 competition's primary 50 words, normalised like the events."""
    from pnpl.competition import PRIMARY_VOCAB

    from neurocast.data.libribrain import normalize_word

    return [normalize_word(w) for w in PRIMARY_VOCAB]


def deep_listener_words(exclude=TEST_CHAPTERS, tasks_prefix=None, window_s=WINDOW_S):
    """Word tables of sub-0's sessions, minus the test chapters' audio."""
    base = dataset_dir("LibriBrain100")
    tasks = sorted(p.name for p in base.iterdir() if (p / "derivatives" / "events").is_dir())
    if tasks_prefix:
        tasks = [t for t in tasks if t.startswith(tasks_prefix)]
    runs = [r for t in tasks for r in find_runs(task=t, subjects={"0"})
            if (r.task, r.session) not in exclude]
    return runs, (word_table(runs, window_s=window_s) if runs else None)


def stage_nobrain(args) -> dict:
    t0 = time.time()
    fit_runs = find_runs(task="Sherlock1", sessions={"11"}, subjects=BROAD)
    test_runs = find_runs(task="Sherlock1", sessions={"12"}, subjects=BROAD)
    print(f"listeners with onsets: fit {len(fit_runs)}, test {len(test_runs)}")
    words_fit = word_table(fit_runs, window_s=args.window)
    words_test = word_table(test_runs, window_s=args.window)
    if args.vocab == "competition":
        vocab = competition_vocabulary()
        absent = [w for w in vocab if w not in set(words_fit["word"])]
        print(f"vocabulary: the 2026 competition's 50 words; absent from chapter 11: {absent}")
    else:
        vocab = vocabulary(words_fit, 50)
        print(f"vocabulary: 50 most frequent words of chapter 11, e.g. {vocab[:12]}")
    print(f"window: {args.window:.3f} s from each onset")
    fit, test = label_trials(words_fit, vocab), label_trials(words_test, vocab)
    print(f"word trials: fit {len(fit):,}   test {len(test):,}   (pitch: 32,198 / 32,251)")

    # Timing classifier: trained on the deep listener's other Sherlock sessions.
    tr_runs, tr_words = deep_listener_words(tasks_prefix="Sherlock", window_s=args.window)
    tr = label_trials(tr_words, vocab)
    n_all = len(tr)
    if args.train_words and len(tr) > args.train_words:
        tr = tr.sample(args.train_words, random_state=args.seed)
    print(f"timing training: {len(tr):,} word trials from {len(tr_runs)} sub-0 Sherlock sessions "
          f"({n_all:,} available; pitch: 60,000)")
    timing = TimingClassifier().fit(tr, tr["label"].to_numpy(), n_classes=len(vocab))
    s_time_fit, s_time = timing.log_proba(fit), timing.log_proba(test)

    # Bigram LM: sub-0's Sherlock sessions except the test chapters' audio. That
    # is 112 sessions -- the pitch's "112 training sessions" exactly.
    lm_runs, lm_words = deep_listener_words(tasks_prefix="Sherlock", window_s=args.window)
    seqs = [g["word"].tolist() for _, g in lm_words.groupby(["task", "session", "subject"], sort=False)]
    lm = BigramPMI().fit(seqs)
    print(f"language model: {lm.n:,} words from {len(lm_runs)} sub-0 sessions "
          f"(pitch: 650,000 from 112)")
    s_lm_fit = lm.pmi(fit["prev_word"], vocab)
    s_lm = lm.pmi(test["prev_word"], vocab)

    lam = tune_weight(s_time_fit, s_lm_fit, fit["label"].to_numpy())
    s_both = combine_scores((s_time, 1.0), (s_lm, lam))
    y = test["label"].to_numpy()

    res = {
        "n_fit": len(fit), "n_test": len(test), "vocab": vocab,
        "timing_train_words": len(tr), "lm_words": lm.n, "lm_sessions": len(lm_runs),
        "timing": bacc(s_time, y), "lm": bacc(s_lm, y), "lm_timing": bacc(s_both, y),
        "lm_timing_weight": lam, "window_s": args.window, "vocab_kind": args.vocab,
    }
    print()
    print(f"  {'held-out chapter 12, 32 listeners':<34} {'BAcc@1':>7} {'BAcc@10':>8}   {'pitch claims':<22}")
    print("  " + "-" * 86)
    row("chance", (0.02, 0.20), "0.02 / 0.20")
    row("speech timing only (no brain)", res["timing"], "0.15-0.18 / 0.61-0.67")
    row("language model, true prev. word", res["lm"], "~0.15 / ~0.62 (chart)")
    row("language model + timing", res["lm_timing"], "~0.35 / 0.80", f"weight {lam:.2f}, tuned on ch. 11")
    print(f"\n  {time.time() - t0:.0f}s")
    return res


def load_split(session: str, vocab: list[str], listeners: set[str], regime: str,
               window_s: float = WINDOW_S):
    """Windows + trial table for one chapter, MEG-covered trials only, cached on D:."""
    import pandas as pd

    from neurocast.data.libribrain import extract_windows

    n_samples = int(round(window_s * 250))
    if n_samples % 16:
        raise ValueError(f"window {window_s} s is not a whole number of 64 ms patches")
    cache = dataset_dir("derived") / "libribrain_audit" / (
        regime if n_samples == 128 else f"{regime}_{n_samples}")
    cache.mkdir(parents=True, exist_ok=True)
    runs = find_runs(task="Sherlock1", sessions={session}, subjects=listeners, require_meg=True)
    wins, tabs, all_onsets = [], [], {}
    for r in runs:
        words = word_table([r], window_s=window_s)
        trials = label_trials(words, vocab)
        rid = f"sub-{r.subject}_ses-{r.session}"
        all_onsets[rid] = words["onset"].to_numpy()
        f = cache / f"{rid}.npz"
        if f.exists():
            z = np.load(f)
            w, kept = z["windows"], z["kept"]
        else:
            w, kept = extract_windows(r, trials["onset"].to_numpy(), n_samples=n_samples,
                                      regime=regime)
            np.savez(f, windows=w, kept=kept)
        t = trials[kept].copy()
        t["run_id"], t["desc"] = rid, r.desc
        wins.append(w)
        tabs.append(t)
        print(f"  {rid:<14} {r.desc:<12} {kept.sum():5d}/{len(kept)} trials covered", flush=True)
    return np.concatenate(wins), pd.concat(tabs, ignore_index=True), all_onsets


def occurrence_shuffle(y: np.ndarray, occ: np.ndarray, rng) -> np.ndarray:
    """Permute labels across word occurrences; every listener's trial of an occurrence moves together."""
    u, inv = np.unique(occ, return_inverse=True)
    first = np.zeros(len(u), dtype=int)
    first[inv[::-1]] = np.arange(len(occ))[::-1]
    lab = y[first]
    return lab[rng.permutation(len(u))][inv]


def stage_brain(args) -> dict:
    from neurocast.audit.fake_signals import onset_only
    from neurocast.audit.surrogates import phase_randomized_surrogate
    from neurocast.decode.decoders import (
        CNNDecoder, LinearPerPerson, LinearPooled, MLPDecoder, patch_features)

    t0 = time.time()
    listeners = set(args.listeners.split(",")) if args.listeners else BROAD
    nb = json.loads((OUT / "nobrain.json").read_text())
    vocab, K = nb["vocab"], len(nb["vocab"])
    print(f"loading chapter 11 (fit) and 12 (test), {len(listeners)} listeners, {args.regime}")
    w_fit, fit, on_fit = load_split("11", vocab, listeners, args.regime, args.window)
    w_te, te, on_te = load_split("12", vocab, listeners, args.regime, args.window)
    yf, yt = fit["label"].to_numpy(), te["label"].to_numpy()
    print(f"MEG-covered word trials: fit {len(fit):,}   test {len(te):,}   (pitch: 32,198 / 32,251)")

    # Timing baseline re-scored on exactly these (MEG-covered) test trials.
    tr_runs, tr_words = deep_listener_words(tasks_prefix="Sherlock", window_s=args.window)
    tr = label_trials(tr_words, vocab)
    if len(tr) > args.train_words > 0:
        tr = tr.sample(args.train_words, random_state=args.seed)
    timing = TimingClassifier().fit(tr, tr["label"].to_numpy(), n_classes=K)
    res = {"n_fit": len(fit), "n_test": len(te), "listeners": len(listeners),
           "timing": bacc(timing.log_proba(te), yt)}

    xf, xt = patch_features(w_fit), patch_features(w_te)
    occ_f, sub_f, sub_t = fit["occ"].to_numpy(), fit["subject"].to_numpy(), te["subject"].to_numpy()

    def linear_on(wf, wt):
        m = LinearPooled(seed=args.seed).fit(patch_features(wf), yf, occ_f)
        return bacc(m.scores(patch_features(wt), K), yt)

    print("decoders ...", flush=True)
    saved = {"timing": timing.log_proba(te)}
    lin = LinearPooled(seed=args.seed).fit(xf, yf, occ_f)
    saved["linear"] = lin.scores(xt, K)
    res["linear"] = bacc(saved["linear"], yt)
    res["linear_alpha"] = float(lin.clf.alpha_)
    print(f"  linear pooled      {res['linear']}", flush=True)
    saved["per_person"] = LinearPerPerson(seed=args.seed).fit(xf, yf, occ_f, sub_f).scores(xt, K, sub_t)
    res["per_person"] = bacc(saved["per_person"], yt)
    print(f"  linear per person  {res['per_person']}", flush=True)
    if not args.skip_nets:
        mlp = MLPDecoder(seed=args.seed).fit(xf, yf, occ_f, K)
        saved["mlp"] = mlp.scores(xt)
        res["mlp"] = bacc(saved["mlp"], yt)
        print(f"  MLP                {res['mlp']}  ({mlp.epochs} epochs)", flush=True)
        cnn = CNNDecoder(seed=args.seed).fit(w_fit, yf, occ_f, K)
        saved["cnn"] = cnn.scores(w_te)
        res["cnn"] = bacc(saved["cnn"], yt)
        print(f"  CNN                {res['cnn']}  ({cnn.epochs} epochs)", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    te.to_csv(OUT / "test_trials.csv", index=False)
    np.savez_compressed(OUT / "test_scores.npz", y=yt, **saved)

    print("controls (linear decoder, refitted on each fake signal) ...", flush=True)
    rng = np.random.default_rng(args.seed)
    sf = 125.0
    if not args.skip_controls:
        fake_f = onset_only(w_fit, fit["onset"].to_numpy(), fit["run_id"].to_numpy(), on_fit, sf, rng)
        fake_t = onset_only(w_te, te["onset"].to_numpy(), te["run_id"].to_numpy(), on_te, sf, rng)
        res["onset_only"] = linear_on(fake_f, fake_t)
        del fake_f, fake_t
    def scrambled(w, chunk=2000):
        # Trial by trial is identical to all at once (each trial draws its own
        # phases); chunking keeps the FFT from needing ~8 GB of RAM.
        # The surrogate keeps each window's channel means (the DC bin), and those
        # are onset-locked: this control reads 0.236, not chance. With the means
        # removed it reads 0.201; the means alone, 0.240 (run_phase_control.py).
        return np.concatenate([
            phase_randomized_surrogate(w[i:i + chunk].astype(np.float32), rng).astype(np.float16)
            for i in range(0, len(w), chunk)])

    if not args.skip_controls:
        pr_f, pr_t = scrambled(w_fit), scrambled(w_te)
        res["phase_scrambled"] = linear_on(pr_f, pr_t)
        del pr_f, pr_t
    null = []
    for i in range(args.n_null):
        ys = occurrence_shuffle(yf, occ_f, np.random.default_rng(1000 + i))
        m = LinearPooled(seed=args.seed).fit(xf, ys, occ_f)
        null.append(bacc(m.scores(xt, K), yt))
    null = np.array(null)
    res["shuffled_label_null"] = {"mean": null.mean(0).tolist(),
                                  "p95": np.percentile(null, 95, axis=0).tolist()}

    # Identity audit of the raw features (pitch: "raw signal: 3x" person information,
    # against 29x for forecasting and 10x for masked pre-trained features).
    from neurocast.audit.fmscope import subject_variance

    sub_idx = np.random.default_rng(args.seed).choice(len(xt), min(len(xt), 8000), replace=False)
    zf = (xt[sub_idx] - xt[sub_idx].mean(0)) / (xt[sub_idx].std(0) + 1e-6)
    ident = subject_variance(zf, sub_t[sub_idx], labels=yt[sub_idx], n_permutations=50,
                             seed=args.seed)
    res["identity_raw"] = {"ratio": ident.ratio_to_null, "eta2": ident.eta2_subject,
                           "probe": ident.probe_accuracy, "chance": ident.chance_accuracy,
                           "eta2_label": ident.eta2_label}

    gain = res["linear"][1] - 0.20
    share = ((res["onset_only"][1] - 0.20) / gain if gain > 0 and "onset_only" in res
             else float("nan"))
    print()
    print(f"  {'held-out chapter 12':<34} {'BAcc@1':>7} {'BAcc@10':>8}   {'pitch claims':<22}")
    print("  " + "-" * 86)
    row("speech timing only (no brain)", res["timing"], "0.15-0.18 / 0.61-0.67")
    row("linear pooled", res["linear"], "0.027 / 0.257")
    row("linear, one per person", res["per_person"], "0.028 / 0.245")
    if "mlp" in res:
        row("MLP", res["mlp"], "0.036 / 0.278")
        row("CNN, raw 125 Hz", res["cnn"], "0.044 / 0.298")
    if "onset_only" in res:
        row("linear on onset-only input", res["onset_only"], "~0.02 / ~0.215 (chart)",
            f"{100 * share:.0f}% of the linear gain (pitch: 21-53%)")
        row("linear on phase-scrambled input", res["phase_scrambled"], "~0.02 / ~0.217 (chart)")
    row(f"shuffled labels, mean of {args.n_null}", tuple(res["shuffled_label_null"]["mean"]),
        "chance", f"p95 {res['shuffled_label_null']['p95'][1]:.3f}")
    res["onset_only_share_of_linear_gain"] = share
    print(f"\n  identity audit, raw 64 ms features of 8,000 test trials: {ident.summary()}")
    print("  pitch: raw signal 3x (forecasting features 29x, masked 10x)")
    print(f"\n  {time.time() - t0:.0f}s")
    return res


#: Covariates the "information beyond" test conditions on: timing, length, loudness.
COVARIATES = ("next1", "next2", "next3", "next4", "gap_prev", "pos_in_sentence",
              "duration", "loudness_db")


def matched_setup(te):
    """Loudness from the chapter's audio, the strata, and the covariate matrix for ``te``."""
    from neurocast.audit.timing_matched import strata
    from neurocast.decode.covariates import chapter_audio_path, read_wav, word_loudness_db

    audio, rate = read_wav(chapter_audio_path(dataset_dir("LibriBrain100"), "Sherlock1", "12"))
    te["loudness_db"] = word_loudness_db(audio, rate, te["t_chapter"].to_numpy(),
                                         te["duration"].to_numpy())
    cov = np.column_stack([np.nan_to_num(te[c].to_numpy(float), nan=-1.0) for c in COVARIATES]
                          + [np.isnan(te[c].to_numpy(float)) for c in ("next1", "next2", "next3")])
    return strata(te), cov


def stage_matched(args) -> dict:
    """Timing-matched test and information beyond covariates, on saved test scores."""
    import pandas as pd

    from neurocast.audit.timing_matched import info_beyond_covariates, matched_gain

    t0 = time.time()
    te = pd.read_csv(OUT / "test_trials.csv")
    z = np.load(OUT / "test_scores.npz")
    y, occ = z["y"], te["occ"].to_numpy()
    groups, cov = matched_setup(te)
    sizes = pd.Series(groups).value_counts()
    print(f"{len(te):,} test trials in {len(sizes):,} strata "
          f"(median {int(sizes.median())} trials); loudness {te['loudness_db'].min():.0f} to "
          f"{te['loudness_db'].max():.0f} dB")

    scorers = [k for k in ("timing", "linear", "per_person", "mlp", "cnn") if k in z]
    res = {"n_strata": int(len(sizes))}
    print(f"\n  {'scorer':<12} {'unmatched gain':>15} {'matched gain':>13} {'surviving':>10}  verdict")
    for k in scorers:
        r = matched_gain(z[k], y, occ, groups, metric="rank", n_perm=args.n_perm, seed=args.seed)
        res[f"matched_{k}"] = {"gain": r.gain, "unmatched_gain": r.unmatched_gain,
                               "surviving": r.surviving, "p95": r.null_p95, "score": r.score,
                               "verdict": r.verdict}
        print(f"  {k:<12} {r.unmatched_gain:>+15.4f} {r.gain:>+13.4f} {100 * r.surviving:>9.0f}%  "
              f"{r.verdict}", flush=True)

    # Information beyond covariates: timing features, duration, loudness.
    print(f"\n  information beyond timing, duration and loudness (held out, every word equal):")
    for k in [s for s in scorers if s != "timing"]:
        b = info_beyond_covariates(cov, z[k], y, occ, n_boot=args.n_boot, seed=args.seed)
        res[f"beyond_{k}"] = {"nats": b.nats, "lo": b.lo, "hi": b.hi}
        print(f"  {k:<12} {b}", flush=True)
    # Falsification: scores carrying only word frequency must add nothing.
    freq = np.log(np.bincount(y, minlength=z["linear"].shape[1]) + 1.0)
    b = info_beyond_covariates(cov, np.tile(freq, (len(y), 1)), y, occ, n_boot=200)
    res["beyond_frequency_only"] = {"nats": b.nats, "lo": b.lo, "hi": b.hi}
    print(f"  {'freq only':<12} {b}   <- must include zero")
    print(f"\n  {time.time() - t0:.0f}s")
    return res


#: Stimulus-only rungs (README section 2.4) by their extra columns; rung A is timing alone.
RUNGS = ("A", "B", "C", "D")


def rung_columns(n_bins: int, n_bands: int) -> dict:
    env = tuple(f"env{j:02d}" for j in range(n_bins))
    bands = tuple(f"b{b}_{j:02d}" for b in range(n_bands) for j in range(n_bins))
    base = ("duration", "loudness_rel")
    return {"A": (), "B": base, "C": base + env, "D": base + env + bands}


def add_acoustics(trials, run, window_s: float, cache: dict):
    """Envelope columns for one run's trials, or ``None`` if its audio fails the alignment check.

    Levels are dB relative to the session's median word level, floored 60 dB below.
    ``cache`` keeps the last chapter's audio and band power (32 listeners share one).
    """
    import pandas as pd

    from neurocast.decode.covariates import (
        alignment_check, band_power, chapter_audio_path, event_times, read_wav, window_levels,
        word_loudness_db)

    key = f"{run.task} ses-{run.session}"
    try:
        path = chapter_audio_path(dataset_dir("LibriBrain100"), run.task, run.session)
    except FileNotFoundError:
        return None, {"session": key, "status": "no audio"}
    if cache.get("path") != path:
        audio, rate = read_wav(path)
        p, centres = band_power(audio, rate)
        cache.clear()
        cache.update(path=path, audio=audio, rate=rate, p=p, centres=centres)
    audio, rate, p, centres = cache["audio"], cache["rate"], cache["p"], cache["centres"]
    ev = event_times(run.events)
    chk = {"session": key, "audio": path.name, **alignment_check(audio, rate, ev)}
    if not (chk["contrast_db"] >= 10.0 and chk["overrun_s"] <= 0.05):
        return None, {**chk, "status": "failed alignment"}
    ref = chk["word_level_db"]
    lv = window_levels(p, centres, trials["onset"].to_numpy(), window_s,
                       ev["meg"].to_numpy(), ev["chapter"].to_numpy())
    lv = np.maximum(lv - ref, -60.0)
    n_bins, n_cols = lv.shape[1], lv.shape[2]
    cols = {"loudness_rel": np.maximum(word_loudness_db(
        audio, rate, trials["t_chapter"].to_numpy(), trials["duration"].to_numpy()) - ref, -60.0)}
    cols.update({f"env{j:02d}": lv[:, j, -1] for j in range(n_bins)})
    cols.update({f"b{b}_{j:02d}": lv[:, j, b] for b in range(n_cols - 1) for j in range(n_bins)})
    out = pd.concat([trials.reset_index(drop=True), pd.DataFrame(cols)], axis=1)
    out.index = trials.index
    return out, {**chk, "status": "ok"}


def stage_acoustic(args) -> dict:
    """Stimulus-only rungs A-D, the matched test on them, and information beyond them."""
    import pandas as pd

    from neurocast.audit.timing_matched import info_beyond_model, matched_gain
    from neurocast.decode.covariates import BAND_EDGES, BIN_S

    t0 = time.time()
    nb = json.loads((OUT / "nobrain.json").read_text())
    vocab, K = nb["vocab"], len(nb["vocab"])
    n_bins, n_bands = int(round(args.window / BIN_S)), len(BAND_EDGES) - 1
    cols = rung_columns(n_bins, n_bands)
    print(f"window {args.window:.3f} s = {n_bins} bins of {BIN_S * 1000:.0f} ms; {K} words; "
          f"{n_bands} bands {BAND_EDGES[0]:.0f}-{BAND_EDGES[-1]:.0f} Hz")

    # Training trials: the deep listener's Sherlock sessions with audio, minus the
    # test chapters and Sherlock 4 chapter 8 (annotation errors, README section 9).
    exclude = TEST_CHAPTERS | {("Sherlock4", "8")}
    runs = [r for r in deep_listener_words(exclude, tasks_prefix="Sherlock", window_s=args.window)[0]]
    print(f"training: {len(runs)} sub-0 sessions; reading each chapter's audio ...", flush=True)
    cache, checks, parts = {}, [], []
    for r in runs:
        t = label_trials(word_table([r], window_s=args.window), vocab)
        feat, c = add_acoustics(t, r, args.window, cache)
        checks.append(c)
        if feat is not None:                    # envelope columns as float32: ~190k rows x 150
            env_cols = [k for k in feat.columns if re.fullmatch(r"env\d+|b\d+_\d+", k)]
            parts.append(feat.astype(dict.fromkeys(env_cols, np.float32)))
        mark = "" if c["status"] == "ok" else f"   <- {c['status']}"
        if "contrast_db" in c:
            print(f"  {c['session']:<20} {c['audio']:<44} words {c['contrast_db']:+5.1f} dB over "
                  f"silence, overrun {c['overrun_s']:+6.2f} s{mark}", flush=True)
        else:
            print(f"  {c['session']:<20} {'-':<44} {mark}", flush=True)
    passing = pd.concat(parts, ignore_index=True)
    train = (passing.sample(args.train_words, random_state=args.seed)
             if 0 < args.train_words < len(passing) else passing)
    y_tr = train["label"].to_numpy()
    logfreq = np.log(np.bincount(y_tr, minlength=K) + 1.0)
    print(f"  {len(parts)} sessions pass the alignment check; training on {len(train):,} of their "
          f"{len(passing):,} word trials", flush=True)

    # Test trials: exactly the saved brain scores' rows (chapter 12, MEG-covered).
    te = pd.read_csv(OUT / "test_trials.csv")
    z = np.load(OUT / "test_scores.npz")
    y, occ = z["y"], te["occ"].to_numpy()
    by_id = {f"sub-{r.subject}_ses-{r.session}": r
             for r in find_runs(task="Sherlock1", sessions={"12"}, subjects=BROAD)}
    feats, test_checks = [], []
    for rid, g in te.groupby("run_id", sort=False):
        f, chk = add_acoustics(g, by_id[rid], args.window, cache)
        test_checks.append(chk)
        if f is None:
            raise SystemExit(f"test run {rid} fails the alignment check: {chk}")
        feats.append(f)
    te_f = pd.concat(feats).loc[te.index]
    c12 = test_checks[0]
    print(f"test: {len(te_f):,} trials, {len(test_checks)} listeners; chapter 12 words "
          f"{min(c['contrast_db'] for c in test_checks):+.1f} dB over silence or more")

    res = {"window_s": args.window, "n_bins": n_bins, "n_train": len(train),
           "train_sessions": len(parts), "alignment": checks, "test_alignment": c12, "rungs": {}}
    logp = {}
    print(f"\n  {'stimulus-only rung':<44} {'BAcc@1':>7} {'BAcc@10':>8}")
    print("  " + "-" * 62)
    print(f"  {'saved timing classifier (all sessions)':<44} {bacc(z['timing'], y)[0]:7.3f} "
          f"{bacc(z['timing'], y)[1]:8.3f}")
    names = {"A": "A  timing", "B": "B  + duration, loudness", "C": "C  + broadband envelope",
             "D": "D  + 8 band envelopes"}
    for k in RUNGS:
        clf = TimingClassifier(extra=cols[k]).fit(train, y_tr, n_classes=K)
        logp[k] = clf.log_proba(te_f)
        res["rungs"][k] = bacc(logp[k], y)
        print(f"  {names[k]:<44} {res['rungs'][k][0]:7.3f} {res['rungs'][k][1]:8.3f}", flush=True)
    d_minus_a = res["rungs"]["D"][1] - res["rungs"]["A"][1]
    res["acoustics_add"] = bool(d_minus_a >= 0.02)
    print(f"  D - A: {d_minus_a:+.3f} BAcc@10 -> acoustics {'add' if res['acoustics_add'] else 'do not add'}"
          " (rule: >= 0.02)")

    groups, _ = matched_setup(te.copy())
    print(f"\n  timing-matched test on the stimulus-only scores ({args.n_perm} permutations):")
    for k in ("A", "C", "D"):
        r = matched_gain(logp[k], y, occ, groups, metric="rank", n_perm=args.n_perm, seed=args.seed)
        res[f"matched_{k}"] = {"gain": r.gain, "unmatched_gain": r.unmatched_gain,
                               "surviving": r.surviving, "verdict": r.verdict}
        print(f"  {names[k]:<28} unmatched {r.unmatched_gain:+.4f}  matched {r.gain:+.4f}  "
              f"surviving {100 * r.surviving:4.0f}%  [{r.verdict}]", flush=True)
    res["matched_controls_acoustics"] = not any(
        res[f"matched_{k}"]["verdict"] == "survives matching" for k in ("C", "D"))

    scores = {k: z[k] for k in ("timing", "linear", "per_person", "mlp", "cnn") if k in z}
    sweep = OUT / "test_scores_sweep.npz"
    if sweep.exists():
        zs = np.load(sweep)
        scores.update({f"sweep:{k}": zs[k] for k in zs.files if k != "y" and zs[k].shape == z["linear"].shape})
    old = json.loads((OUT / "matched.json").read_text()) if (OUT / "matched.json").exists() else {}
    print(f"\n  information beyond the stimulus-only model, nats/word (95% interval):")
    print(f"  {'scorer':<22} {'beyond D (timing + envelopes)':<40} {'beyond C':<40} "
          "beyond timing, duration, loudness (section 2.1)")
    for k, s in scores.items():
        bd = info_beyond_model(logp["D"], s, y, occ, prior=logfreq, n_boot=args.n_boot, seed=args.seed)
        bc = info_beyond_model(logp["C"], s, y, occ, prior=logfreq, n_boot=args.n_boot, seed=args.seed)
        prev = old.get(f"beyond_{k}")
        prev_s = f"{prev['nats']:+.4f} [{prev['lo']:+.4f}, {prev['hi']:+.4f}]" if prev else "-"
        res[f"beyond_D_{k}"] = {"nats": bd.nats, "lo": bd.lo, "hi": bd.hi, "dropped": list(bd.dropped)}
        res[f"beyond_C_{k}"] = {"nats": bc.nats, "lo": bc.lo, "hi": bc.hi}
        print(f"  {k:<22} {str(bd):<40} {str(bc):<40} {prev_s}", flush=True)
    print(f"\n  {time.time() - t0:.0f}s")
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("nobrain", "brain", "matched", "acoustic"), default="nobrain")
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--train-words", type=int, default=60_000,
                    help="cap on timing-classifier training trials (pitch: 60,000); 0 = all")
    ap.add_argument("--listeners", default="", help="comma-separated subset, e.g. 1,2,3")
    ap.add_argument("--regime", default="session-scalar",
                    choices=("type-const", "session-scalar", "session-robust"))
    ap.add_argument("--skip-nets", action="store_true", help="linear decoders and controls only")
    ap.add_argument("--n-null", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--window", type=float, default=WINDOW_S,
                    help="seconds from onset (the competition scores 1.0; 1.024 = 16 patches)")
    ap.add_argument("--vocab", choices=("top50", "competition"), default="top50",
                    help="50 most frequent words of chapter 11, or the 2026 competition's")
    ap.add_argument("--skip-controls", action="store_true",
                    help="brain stage without the onset-only and phase-scrambled refits")
    args = ap.parse_args()
    global OUT
    if abs(args.window - WINDOW_S) > 1e-9 or args.vocab != "top50":
        OUT = ROOT / "runs" / f"libribrain_audit_w{int(round(args.window * 1000))}_{args.vocab}"

    res = {"nobrain": stage_nobrain, "brain": stage_brain, "matched": stage_matched,
           "acoustic": stage_acoustic}[args.stage](args)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{args.stage}.json").write_text(json.dumps(res, indent=2, default=float))
    print(f"  saved {OUT.relative_to(ROOT) / (args.stage + '.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
