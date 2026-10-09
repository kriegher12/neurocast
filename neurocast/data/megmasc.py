"""MEG-MASC (Gwilliams et al.) word events, in the shape the LibriBrain code uses.

MEG-MASC: 27 listeners, two identical sessions each, four synthesised stories
(MASC corpus) interleaved with random word lists -- BIDS on OSF, fetched with
``scripts/data_setup.py fetch-osf``. Every event row's ``trial_type`` is a
Python-literal dict: ``kind`` (sound / phoneme / word), ``word``, ``story``,
``sequence_id``, ``word_index``, and ``condition`` -- ``"sentence"`` inside a
story, ``"word_list"`` inside a random list.

That second condition is why this dataset is the timing test's natural
replication. In running speech, when the next word starts says something about
this one; in a random word list it should say much less. :func:`read_words`
returns the same columns as :func:`neurocast.data.libribrain.read_words`, so
:func:`neurocast.decode.words.timing_features` and the timing classifier apply
unchanged.
"""

from __future__ import annotations

import ast
from pathlib import Path

from .libribrain import normalize_word
from .paths import dataset_dir, replace_file

__all__ = ["TASK_STORIES", "events_path", "raw_path", "read_words", "word_table", "preprocess",
           "load_preprocessed", "head_transform", "kit_montage", "cache_for_pretraining"]

TASK_STORIES = {"0": "lw1", "1": "cable_spool_fort", "2": "easy_money", "3": "The_Black_Widow"}


def events_path(subject: str, session: str, task: str, root: Path | None = None) -> Path:
    root = Path(root) if root is not None else dataset_dir("MEG-MASC")
    return root / f"sub-{subject}" / f"ses-{session}" / "meg" / f"sub-{subject}_ses-{session}_task-{task}_events.tsv"


def raw_path(subject: str, session: str, task: str, root: Path | None = None) -> Path:
    return events_path(subject, session, task, root).with_name(
        f"sub-{subject}_ses-{session}_task-{task}_meg.con")


def read_words(events: Path):
    """Word rows of one MEG-MASC events file, in order.

    Columns: ``onset``, ``duration`` (s, recording time), ``word`` (normalised),
    ``sentence`` (the sequence id), ``widx`` (word index in the sequence),
    ``condition``, ``story``, and ``sound`` / ``t_sound`` -- the stimulus file
    and the word's start inside it, for measuring loudness.
    """
    import pandas as pd

    ev = pd.read_csv(events, sep="\t", encoding="utf-8-sig")
    meta = ev["trial_type"].map(ast.literal_eval)
    keep = meta.map(lambda d: d.get("kind") == "word")
    ev, meta = ev[keep], meta[keep]
    out = pd.DataFrame({
        "onset": ev["onset"].astype(float).to_numpy(),
        "duration": ev["duration"].astype(float).to_numpy(),
        "word": meta.map(lambda d: normalize_word(d.get("word", ""))).to_numpy(),
        "sentence": meta.map(lambda d: int(d.get("sequence_id", -1))).to_numpy(),
        "widx": meta.map(lambda d: int(d.get("word_index", -1))).to_numpy(),
        "condition": meta.map(lambda d: d.get("condition", "")).to_numpy(),
        "story": meta.map(lambda d: d.get("story", "")).to_numpy(),
        "sound": meta.map(lambda d: d.get("sound", "")).to_numpy(),
        "t_sound": meta.map(lambda d: float(d.get("start", float("nan")))).to_numpy(),
    })
    # A handful of "pseudo_words" rows share sequence and index with a real word.
    out = out[(out["word"] != "") & (out["condition"] != "pseudo_words")]
    return out.sort_values("onset", kind="stable").reset_index(drop=True)


def word_table(subject: str, session: str, tasks, *, window_s: float, root: Path | None = None):
    """Words of one listener's session for the given tasks, with timing features.

    ``occ`` -- (task, condition, sequence, word index) -- identifies a word
    occurrence; every listener heard the same audio (relative timing agrees
    across listeners to 1e-13 s), so it is what splits must group by.
    """
    import pandas as pd

    from ..decode.words import timing_features

    parts = []
    for task in tasks:
        f = events_path(subject, session, task, root)
        if not f.exists():
            continue
        w = timing_features(read_words(f), window_s)
        w.insert(0, "subject", subject)
        w.insert(1, "session", session)
        w.insert(2, "task", task)
        w["occ"] = (task + "/" + w["condition"] + "/" + w["sentence"].astype(str) + "/"
                    + w["widx"].astype(str))
        parts.append(w)
    return pd.concat(parts, ignore_index=True) if parts else None


def preprocess(subject: str, session: str, task: str, cache_dir: Path, *, sfreq: float = 250.0,
               l_freq: float = 0.1, h_freq: float = 100.0, line: float = 60.0,
               root: Path | None = None) -> Path:
    """One KIT recording -> ``(208, T)`` float16 in canonical units, cached; returns the path.

    Close to LibriBrain's released pipeline (notch, band-pass, downsample to
    250 Hz), without its bad-channel, head-position and SSS steps, which need
    Elekta-specific information. MEG-MASC was recorded in New York: mains is
    60 Hz. MNE types KIT axial gradiometers as ``mag`` (tesla), so the frozen
    canonical scale is the same picotesla as LibriBrain's magnetometers. The
    16 reference sensors are dropped. Sample 0 is recording time 0, the clock of
    the events' ``onset`` column.
    """
    import json

    import mne
    import numpy as np

    from .canonical import to_canonical

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / f"sub-{subject}_ses-{session}_task-{task}.npy"
    if out.exists():
        return out
    raw = mne.io.read_raw_kit(raw_path(subject, session, task, root), preload=True, verbose="WARNING")
    if raw.first_samp != 0:
        raise ValueError(f"{out.name}: first sample {raw.first_samp}, expected 0")
    raw.pick("mag")
    raw.notch_filter(np.arange(line, h_freq, line), verbose="WARNING")
    raw.filter(l_freq, h_freq, verbose="WARNING")
    raw.resample(sfreq, verbose="WARNING")
    x, _ = to_canonical(raw.get_data().astype(np.float32), ["mag"] * len(raw.ch_names))
    tmp = out.with_suffix(".tmp.npy")
    np.save(tmp, x.astype(np.float16))
    replace_file(tmp, out)
    out.with_suffix(".json").write_text(json.dumps(
        {"sfreq": sfreq, "l_freq": l_freq, "h_freq": h_freq, "line": line,
         "channel_names": list(raw.ch_names)}))
    return out


def load_preprocessed(path: Path, regime: str = "session-scalar"):
    """A cached recording, normalised under ``regime``: ``(x (C, T) float32, sfreq)``."""
    import json

    import numpy as np

    from .canonical import NormalizationRegime, fit_normalizer

    path = Path(path)
    meta = json.loads(path.with_suffix(".json").read_text())
    x = np.load(path).astype(np.float32)
    types = ["mag"] * x.shape[0]
    x = fit_normalizer(x[:, ::4], NormalizationRegime(regime), ch_types=types).apply(x)
    return x.astype(np.float32), float(meta["sfreq"])


def _read_points(path: Path):
    """FastSCAN ASCII (MEG-MASC's ``.pos``): ``x y z`` per line in mm, ``%`` comments -> metres."""
    import numpy as np

    pts = []
    for line in Path(path).read_text().splitlines():
        f = line.strip().split()
        if len(f) >= 3 and not line.strip().startswith("%"):
            try:
                pts.append([float(f[0]), float(f[1]), float(f[2])])
            except ValueError:
                continue
    if not pts:
        raise ValueError(f"no points in {path}")
    return np.asarray(pts) / 1000.0


def head_transform(subject: str, session: str = "0", task: str = "0", root: Path | None = None):
    """Device-to-head transform (4 x 4) from the marker coils and digitised head shape.

    MNE's KIT reader computes it from ``*_markers.mrk`` and the ELP / HSP point
    files; MEG-MASC ships those as FastSCAN ``.pos``, which MNE does not read
    by extension, so they are parsed here (as pnpl's loader does). Also returns
    the MEG channels' names and device-frame ``loc`` arrays.
    """
    import mne
    import numpy as np

    con = raw_path(subject, session, task, root)
    d = con.parent
    raw = mne.io.read_raw_kit(
        con, mrk=str(d / f"sub-{subject}_ses-{session}_task-{task}_markers.mrk"),
        elp=_read_points(d / f"sub-{subject}_ses-{session}_acq-ELP_headshape.pos"),
        hsp=_read_points(d / f"sub-{subject}_ses-{session}_acq-HSP_headshape.pos"),
        preload=False, verbose="ERROR")
    picks = mne.pick_types(raw.info, meg="mag")
    names = [raw.ch_names[i] for i in picks]
    loc = np.array([raw.info["chs"][i]["loc"][:12] for i in picks])
    return np.asarray(raw.info["dev_head_t"]["trans"]), names, loc


def kit_montage(subjects, session: str = "0", task: str = "0", root: Path | None = None,
                name: str = "kit-208-megmasc"):
    """One NeuroCast montage for MEG-MASC's 208 KIT axial gradiometers, in head coordinates.

    NeuroCast groups sensors into left/right x anterior/posterior quadrants of a
    head-centred RAS frame, so device coordinates must be moved into the head.
    Every listener's head sits a little differently in the helmet (translations
    differ by ~2 cm), but one model has one montage: the transform used is the
    average over ``subjects`` (mean translation; mean rotation projected back
    onto a rotation). Without LibriBrain's head-position correction, the
    remaining per-listener differences are part of the data.

    KIT axial gradiometers are typed as the magnetometer family: MNE reports them
    in tesla, they share the picotesla canonical scale, and an axial
    gradiometer has no in-plane baseline to encode.
    """
    import numpy as np

    from ..tokenizer.descriptor import Montage, SensorDescriptor
    from .canonical import ChannelType

    trans, names, loc = [], None, None
    for sub in subjects:
        t, n, l = head_transform(sub, session, task, root)
        if names is None:
            names, loc = n, l
        elif n != names or not np.allclose(l, loc):
            raise ValueError(f"sub-{sub}: KIT channel geometry differs from sub-{subjects[0]}")
        trans.append(t)
    trans = np.mean(trans, axis=0)
    u, _, vt = np.linalg.svd(trans[:3, :3])
    rot, shift = u @ vt, trans[:3, 3]
    pos = loc[:, :3] @ rot.T + shift
    ez = loc[:, 9:12] @ rot.T
    sensors = tuple(SensorDescriptor(n, p, e, ChannelType.MAG) for n, p, e in zip(names, pos, ez))
    return Montage(name, sensors)


def cache_for_pretraining(subjects, tasks, out_dir: Path, *, session: str = "0",
                          regime: str = "session-scalar", verbose: bool = True) -> list[str]:
    """Time-major, normalised story spans of preprocessed recordings, for
    :func:`neurocast.train.pretrain.open_cached`. Keys ``sub-XX_task-T``.

    Same layout as :func:`neurocast.train.pretrain.cache_recordings`: float16
    ``(T, C)`` plus a JSON sidecar; the story span runs from the first word's
    onset to the last word's end; ``regime`` is fitted on the whole recording.
    """
    import json

    import numpy as np

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    src = dataset_dir("derived") / "megmasc"
    keys = []
    for sub in subjects:
        for task in tasks:
            key = f"sub-{sub}_task-{task}"
            f, meta = out_dir / f"{key}.npy", out_dir / f"{key}.json"
            pre = src / f"sub-{sub}_ses-{session}_task-{task}.npy"
            if not pre.exists():
                continue
            keys.append(key)
            if f.exists() and meta.exists():
                continue
            x, sfreq = load_preprocessed(pre, regime)
            w = read_words(events_path(sub, session, task))
            a = max(0, int(w["onset"].iloc[0] * sfreq))
            b = min(int((w["onset"].iloc[-1] + w["duration"].iloc[-1]) * sfreq), x.shape[1])
            tmp = out_dir / f"{key}.tmp.npy"
            np.save(tmp, np.ascontiguousarray(x[:, a:b].astype(np.float16).T))
            replace_file(tmp, f)
            names = json.loads(pre.with_suffix(".json").read_text())["channel_names"]
            meta.write_text(json.dumps({"story": [a, b], "sfreq": sfreq, "regime": regime,
                                        "channel_names": names}))
            if verbose:
                print(f"  cached {key}: {(b - a) / sfreq / 60:.1f} min", flush=True)
    return keys
