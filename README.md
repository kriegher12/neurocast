# NeuroCast

**Forecasting the brain as a foundation for decoding it.**

A stimulus-conditioned, subject-in-context predictive model of brain dynamics —
built so that the same artifact serves as a pretraining objective, a measurement
instrument, and the forward model of a decoder.

---

## 1. The idea in one page

Non-invasive brain decoding is stuck on one problem and quietly compromised by a
second.

**Stuck.** Decoders work *within* a subject and collapse *across* subjects. On
identical 50-word decoding from MEG, a within-subject supervised model reaches
**73.23% BAcc@10** while the best cross-subject self-supervised model reaches
**42.96%**. Every brain foundation model retrains or fine-tunes per subject.

**Compromised.** A wave of 2026 audits shows much of the reported progress is
shortcut learning:

- *The Identity Trap* ([arXiv 2606.06647]) — frozen subject-identity variance is
  **13–89× a random null in 12/12 model-dataset pairs**, and fine-tuning makes it
  **worse (+10 to +63pp)**. Removing the aperiodic 1/f component drops the
  subject probe by 9–19pp: 1/f is a subject fingerprint.
- Brain-to-language retrieval — signal-blind **Gaussian noise scores 66.3%
  Rank@1** under the field's standard variable-length protocol
  ([arXiv 2605.24524]).
- MOABB's large reproducibility study — deep learning still hasn't clearly beaten
  2010s Riemannian methods ([arXiv 2404.15319]).

**The bet.** Both problems have one cause: the field trains **classifiers**, and
a classifier is free to solve its task however it can. Identity is the cheapest
shortcut available. Train a **forecaster** instead — a model of *what this brain
will do next* — and the shortcut stops paying, because the target changes every
millisecond and depends on the stimulus.

This is not a hunch. The Sept-2026 *Roadmap for MEG Foundation Models*
([arXiv 2609.04461]) states verbatim that for pretraining objectives **"the
decisive comparison at matched data, compute and architecture is still
missing"**, that JEPA-style latent prediction for MEG is **"completely
unexplored"**, and that MEG scaling laws are **"not yet done"**.

### Falsifiable claims

| | Claim | Kill condition |
|---|---|---|
| **H1** | At matched data/compute/architecture, forecasting transfers better than masked reconstruction | masking wins or ties on both axes across ≥2 of 3 compute tiers, 3 seeds |
| **H2** | Forecasting starves the identity shortcut — lower FMScope subject-variance than masked models | forecasting arms leak as much as masked arms |
| **H3** | The subject is a *prompt*, not a fine-tune: K minutes in-context ≥ fine-tuning on the same K minutes | prompting underperforms LoRA by >3pp BAcc@10 at K ∈ {10,20,40} min |
| **H4** | The forecaster inverts into a decoder via noisy-channel inference with an LLM prior | see §9, risk R1 — this is the claim most likely to fail |
| **H5** | Geometry conditioning transfers MEG→EEG/OPM/iEEG without retraining | — |

Plus one deliverable that is **architecture-independent and survives all five
failing**: the **Neural Predictability Atlas** — how many bits of a human brain's
future are knowable, by horizon, band, region, and conditioning set. The
information-theoretic framework exists ([arXiv 2603.27074]) and has never been
applied to human M/EEG.

**Mechanism for H2.** A subject's nuisance structure — 1/f slope, head position,
sensor coupling — is *visible in the context window*. A model that can read it
from the prompt has no incentive to store it in weights. And the corpus is too
small to memorise 800 subjects anyway (§6).

---

## 2. Status

Phase 0 infrastructure is built and **validated against ground truth**. No model
code yet — deliberately. The plan's rule is that the leakage harness ships before
anything it might have to audit.

| Module | State | Validated by |
|---|---|---|
| `audit/surrogates.py` | ✅ working | `scripts/validate_controls.py` |
| `audit/controls.py` | ✅ working | `scripts/validate_controls.py` |
| `data/canonical.py` | ✅ working | `scripts/validate_canonical.py` |
| `data/splits.py` | ✅ working | `scripts/validate_splits.py` |
| `tokenizer/descriptor.py` | ✅ working | `scripts/validate_tokenizer.py` |
| `tokenizer/embedding.py` | ✅ working | `scripts/validate_tokenizer.py` |
| `tokenizer/patch.py` | ✅ working | `scripts/validate_tokenizer.py` |
| `tokenizer/perceiver.py` | ✅ working | `scripts/validate_tokenizer.py` |
| `models/backbone.py` | ✅ working | `scripts/validate_backbone.py` |
| `models/heads.py` | ✅ working | `scripts/validate_backbone.py` |
| `objectives/arms.py` | ✅ working | `scripts/validate_backbone.py` |
| `train/flops.py` | ✅ working | `scripts/validate_flops.py` |
| `adapt/episodes.py` | ✅ working | `scripts/validate_adapt.py` |
| `adapt/nuisance.py` | ✅ working | `scripts/validate_adapt.py` |
| `audit/fmscope.py` | ✅ working | `scripts/run_bakeoff.py` |
| `models/neurocast.py` | ✅ working | `scripts/run_bakeoff.py` |
| `data/synthetic.py` | ✅ working | `scripts/run_bakeoff.py` |
| `train/loop.py` | ✅ working | `scripts/run_bakeoff.py` |
| `decode/`, `atlas/`, `viz/` | ⬜ not started | — |

Every validation script is built to be **falsified first**: each constructs a
case that *is* leaky and asserts the harness catches it. A guard that never fires
is not a guard.

---

## 3. Install

Requires Python ≥ 3.11 (developed on 3.14.7) and ~3 GB for dependencies.

```bash
git clone <your-fork> neurocast && cd neurocast
python -m venv .venv
```

```bash
.venv/Scripts/python.exe -m pip install numpy scipy mne pnpl
```

On Linux/macOS use `.venv/bin/python` throughout. `pnpl` pulls in `torch`
automatically. For GPU training install the CUDA build explicitly:

```bash
.venv/Scripts/python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu124
```

Verify:

```bash
.venv/Scripts/python.exe -c "import torch, pnpl, mne; print(torch.__version__, torch.cuda.is_available())"
```

---

## 4. Run the validation suite

No data download needed — all three run on synthetic data in under a minute.

```bash
.venv/Scripts/python.exe scripts/validate_controls.py
```

```bash
.venv/Scripts/python.exe scripts/validate_canonical.py
```

```bash
.venv/Scripts/python.exe scripts/validate_splits.py
```

```bash
.venv/Scripts/python.exe scripts/validate_tokenizer.py
```

```bash
.venv/Scripts/python.exe scripts/validate_backbone.py
```

```bash
.venv/Scripts/python.exe scripts/validate_flops.py
```

```bash
.venv/Scripts/python.exe scripts/validate_adapt.py
```

All seven must exit 0 before you trust any downstream number.

### Then run the bake-off itself

```bash
.venv/Scripts/python.exe scripts/run_bakeoff.py
```

~3 minutes on CPU. Trains all five objectives on synthetic data under an
identical **measured** FLOP budget and prints the Phase-1 comparison table.
Pointing it at LibriBrain100 is a data-source change, nothing more.

### What they prove

**`validate_controls.py`** — three cases with known ground truth:

| Case | Truth | Required verdict |
|---|---|---|
| A — classes differ in alpha power | decodable, but *only* spectral | `phase_randomized` must **FAIL** |
| B — sign-flipped evoked response | genuine temporal dynamics | all controls must **PASS** |
| C — constant DC offset | not neural at all | `dc_only` must **FAIL** |

Case A is the one that matters. Phase randomisation preserves the power spectrum
*exactly*, including the 1/f slope. Any model whose accuracy survives it is
reading spectral identity, not neural dynamics.

**`validate_canonical.py`** — enforces INV-1 and shows the failure mode it
prevents. Without the Jacobian correction, four arms modelling *identical data*
report likelihoods spanning **84,065 nats**, decided purely by how hard each one
normalised. With the correction they agree to **1.8e-12 nats**.

**`validate_splits.py`** — constructs each leak and asserts it's caught:
unguarded temporal blocks, shared subjects, shared stimuli, variable-length
windows. The important pair is the last: a split can be *perfectly
subject-disjoint* and still let the model memorise the audiobook text.

**`validate_tokenizer.py`** — four structural guarantees, on a synthetic
306-channel MEGIN array (102 mag + 204 planar grad + EOG/ECG) and a 64-channel
EEG cap:

| Claim | Why it matters | Result |
|---|---|---|
| quadrants partition the channels | otherwise `log p(x)` is not a likelihood | exact |
| **group confinement** | makes the factorisation an identity, not a bound | leak **0.0e+00** |
| zero new parameters for a new montage | H5 rests on it | 518,336 params either way |
| deterministic geometry encoding | a checkpoint must deploy on unseen hardware | reproduces across seeds |

Group confinement is the one to watch. Perturbing quadrant 1's raw signal
changes *only* quadrant 1's token, to machine precision — every other group is
bit-identical. That is what licenses
`p(x_t) = prod_g p(x_t^(g) | x_t^(<g), x_<t)` as an exact chain rule.

**`validate_backbone.py`** — causality first, because a backbone that leaks one
step of future information manufactures exactly the result this project exists
to measure honestly. It would appear to forecast beautifully while reading the
answer, and it does not announce itself.

| Check | Result |
|---|---|
| perturbing position *j* leaves every earlier position untouched | **0.0e+00** at j = 5, 12, 23 |
| one window-4 layer does not propagate beyond its window | 0.0e+00 past the window |
| `ar_shift` pairs output *i* with target *i+1* | exact |
| mixture head integrates to 1 | 1.000000 |
| NLL of standard-normal draws vs Gaussian entropy | 1.4125 vs 1.4189 |
| collapse monitor flags a collapsed representation | stable-rank 0.016 vs 0.457 |
| all five arms: finite loss, gradient reaches a shared trunk | ✅ |

---

## 5. Where to get the training data

MEG-first, generalising outward. Most of Tier 1 comes through **one loader**.

### 5.1 The easy path — `pnpl`

`pnpl` (v0.2.0) handles download, preprocessing and PyTorch collation for five
MEG corpora:

```python
from pnpl.datasets import LibriBrain100Word

ds = LibriBrain100Word(
    data_path="./data/LibriBrain100",
    partition="train",
    tmin=0.2, tmax=0.6,      # the PNPL word window: 100 samples at 250 Hz
    standardize=True,
    download=True,
)
x, y = ds[0]                  # x: (306, 100) MEG, y: word label
```

Also exposed: `LibriBrain100`, `LibriBrain100Speech`, `LibriBrain100Phoneme`,
`LibriBrainSentence`, `Gwilliams2022`, `Armeni2022`, `Schoffelen2019`,
`Pallier2025`.

Default preprocessing string is
`bads+headpos+sss+notch+bp+ds` — bad-channel handling, head-position correction,
Maxwell/SSS filtering, 50/100 Hz notch, bandpass, downsample to 250 Hz.

### 5.2 Tier 1 — core MEG corpus

| Dataset | Subjects | Hours | Sensors | Size | Access |
|---|---|---|---|---|---|
| **LibriBrain100** | 33 (1 deep ≈80h + 32 ≈40min) | ~102 | 306ch MEGIN | ~0.5 TB | CC-BY-4.0, via `pnpl` / [HF `pnpl/LibriBrain`] |
| **MEG-MASC** (Gwilliams) | 27 | 56 | 208ch | ~100 GB | CC0, OpenNeuro **ds004633** |
| **Armeni 2022** | 3 | 30 | 275ch CTF | ~200 GB | Radboud Data Repository |
| **MOUS** (Schoffelen) | 204 | — | 275ch CTF | ~1 TB | Radboud RDR, some DUA components |
| **THINGS-MEG** | 4 | — | 272ch | 377 GB | CC0, OpenNeuro **ds004212** — **hold out** |
| **Cam-CAN** | 612 | ~350 rest | 306ch | ~1 TB | **Gated**, apply at [cam-can.org] |

⚠️ **Disk.** LibriBrain alone is 264 GB; LibriBrain100 more than doubles it.
Budget **~2 TB** for Tier 1. Start with a single subject to validate shapes.

OpenNeuro datasets download without credentials:

```bash
aws s3 sync --no-sign-request s3://openneuro.org/ds004633 data/ds004633/
```

or, if you prefer the Python client:

```bash
.venv/Scripts/python.exe -m pip install openneuro-py && openneuro-py download --dataset=ds004633 --target-dir=data/ds004633
```

### 5.3 ⚠️ The peripheral-channel problem — read before Phase 1

LibriBrain **recorded** bipolar EOG (outer canthi; above/below left eye) and
bipolar ECG (clavicle/hip), but **the preprocessed HDF5 release is 306-channel
MEG only.**

The peripheral pretraining objective and the artifact-only control both need
them, so they must be re-derived from the **raw FIF files on HuggingFace**. This
is a hard Phase 0 dependency — resolve it before building the peripheral arm.
If the FIF release also lacks them, fall back to ICA-derived ocular/cardiac
component time-courses, which is scientifically weaker and **must be labelled as
derived** wherever reported.

### 5.4 Tier 2 — cross-modal and peripheral

| Dataset | What it adds | Access |
|---|---|---|
| **CNeuroMod** | 6 subjects, >80h each, fMRI **+ MEG + EDA + oculometry** — the only corpus with brain *and* peripheral *and* naturalistic stimulus | [github.com/courtois-neuromod], DataLad |
| **1000-h Japanese EEG-EMG-audio** | 3 subjects, 1,020h, EEG + facial EMG + audio | [arXiv 2606.01264] |
| **ZuCo 1.0/2.0** | EEG + eye-tracking + reading, 30 subjects | OSF |
| **Digit-span EEG+pupil+ECG** | 86 subjects | OpenNeuro **ds003838** |
| **EEGDash** | 791 datasets, ~86,000h, BIDS-first | `pip install eegdash` |
| **AJILE12** | 1,280h human iEEG + pose | DANDI |

### 5.5 Baseline to beat

**MEG-XL** (Jayalath & Parker Jones, ICML 2026) — the PNPL Broad-track baseline,
with **public code and checkpoints**:

```bash
git clone https://github.com/neural-processing-lab/MEG-XL
```

It is *almost this model built the other way*, which makes it the ideal control
arm:

| | MEG-XL | NeuroCast |
|---|---|---|
| Context | 2.5 min (~191k tokens) | same — long context is settled, not our contribution |
| Tokenizer | frozen BioCodec RVQ, 6 levels, vocab 256, ×12 compression | same, initially |
| Backbone | 8-layer, 512-dim, criss-cross attention, ~20M params | same, initially |
| Pretraining data | ~300h — CamCAN, MOUS, SMN4Lang, 800+ subjects | same |
| **Objective** | **masked token prediction** (40%, 3s blocks) | **causal forecasting** |
| **New-subject adaptation** | **end-to-end fine-tuning** (1–2h vs 50h supervised) | **in-context, zero gradient updates** |
| Identity leakage | **never measured** | headline metric |

Hold everything fixed at their published recipe; change *only* the objective and
the adaptation mechanism. Someone else already built and validated the control.

Note their fine-tuning step is exactly the operation the Identity Trap paper
shows inflates leakage by +10 to +63pp — H2's prediction, testable on a public
checkpoint in days.

---

## 5a. Measured compute — three findings that change the bake-off

`6ND` (six FLOPs per parameter per token) is how this literature reports compute.
For NeuroCast it is wrong by a factor of **3.9×**, and the error is
*arm-dependent* — which means an objective could "win" H1 on accounting alone.
All figures measured with `FlopCounterMode` over a complete step
(front-end → backbone → head → loss → backward), 309 channels, 24.9M params:

| Finding | Measurement | Why it matters |
|---|---|---|
| measured ÷ 6ND | **3.91×** | 6ND assumes cost is parameter-weighted matmuls; false for any channel-dimension front-end |
| SensorPerceiver share | **81.2% → 61.2%** after retuning | scales with *channel count*, which does not appear in 6ND at all |
| EMA teacher overhead | **+33.2%** | arms J and A-lat pay it, M and A-lik do not — excluding it silently gifts the latent arms compute |

**The front-end was eating the budget.** At the original defaults, 81.2% of every
FLOP went to pooling channels rather than modelling dynamics — stable from 1 s to
16 s windows, so structural rather than an artifact of window length. A poor
allocation for a project whose scientific claim is about temporal prediction. A
measured ablation (309 ch, d=384, 4.1 s):

| latents/group | cross layers | GFLOP | rel | params |
|---|---|---|---|---|
| 16 | 2 | 141.8 | 1.00× | 5,940,096 |
| 8 | 2 | 94.2 | 0.66× | 4,745,088 |
| 16 | 1 | 73.2 | 0.52× | 4,165,632 |
| **8** | **1** | **48.2** | **0.34×** | **2,970,624** ← default |
| 4 | 1 | 35.8 | 0.25× | 2,373,120 |

Dropping the second cross layer is the single biggest win: one layer already
lets every latent see every channel in its quadrant, and the backbone supplies
the depth. Defaults are now 8 latents / 1 layer, taking the front-end to 61.2%.

This is **provisional**. Perceiver capacity could plausibly move the H1 result,
so `latents_per_group` and `n_cross_layers` join `n_groups` in the Phase-1
ablation, and the chosen point gets reported with the bake-off rather than
assumed.

**HPO is charged to the arm that spent it.** A 16-trial search buys ~8M fewer
training steps than a 2-trial search at the same tier. That is what separates
"objective X is better" from "we tuned X harder", and essentially nobody does it.
`compare_arms()` refuses a >2% spread before any comparison table is written.

---

## 6. Hardware — and why not to scale the model

Run the token-budget arithmetic before reaching for a bigger model:

| Quantity | Value |
|---|---|
| Realistic MEG corpus | ~600 h |
| Raw scalars | 600h × 3600s × 250Hz × 306ch ≈ **1.65 × 10¹¹** |
| Sequence positions @ 62.5 tok/s | **1.35 × 10⁸** |
| Chinchilla-optimal N at 4 epochs | **~34M parameters** (G=5; ~27M at G=4) |

MEG-XL is 20M and is SOTA. That is not a coincidence. The ladder is
**6M / 25M / 90M / 320M**, governed by *repeated-data* scaling laws (~4 epochs
near-free, decaying past ~16). **A 1B-parameter brain foundation model is, on
current public data, a category error.**

This is also the strongest argument for H3: there isn't enough data to memorise
800 subjects in weights, but there is enough to learn to *read* a subject from
context.

| Phase | Scope | GPU-hours (A100) |
|---|---|---|
| 0 | harness, data layer, reproduce baseline, control table | ~200 |
| 1 | 5-arm objective bake-off, 2 tiers × 3 seeds + charged HPO | ~1,500 |
| 2 | subject-in-context + stimulus conditioning | ~3,000 |
| 3 | scaling ladder + Atlas production run | ~6,000 |
| 4 | noisy-channel decoder, EEG/iEEG transfer | ~3,000 |
| | **total** | **~14,000** |

≈ 2 months on one 8×A100 node. The validation suite in §4 runs on a laptop CPU.

---

## 7. Repository layout

```
neurocast/
├── audit/              ★ built first — a number without controls is not a number
│   ├── surrogates.py     gaussian · AR(16) · phase-randomised (uni/multivariate) · dc_only
│   └── controls.py       permutation null, pass/fail report, artifact-only row
├── data/
│   ├── canonical.py    ★ INV-1: frozen observation space + log|det J| bookkeeping
│   └── splits.py       ★ temporal/subject/stimulus disjointness + fingerprints
├── tokenizer/          ★ montage-agnostic front-end (~3.2M params)
│   ├── descriptor.py     SensorDescriptor, quadrants, partition assertion
│   ├── embedding.py      frozen Gaussian Fourier features + sensor MLP
│   ├── patch.py          64 ms non-overlapping patches, one projection per family
│   └── perceiver.py      quadrant-confined pooling, exact AR factorisation
├── train/
│   └── flops.py        ★ measured FLOP accounting; HPO charged per arm
├── models/             ★ causal stack + prediction heads
│   ├── backbone.py       sliding-window + global causal attention, prompt x-attn
│   └── heads.py          gaussian mixture (flagship) · latent · masked · peripheral
├── objectives/         ★ the bake-off registry — Phase 1 loops over this
│   └── arms.py           masked | jepa | ar_latent | ar_lik | peripheral
├── adapt/              ★ subject-as-prompt
│   ├── episodes.py       episodic sampler, guard bands, mismatch negatives
│   └── nuisance.py       1/f + band power + gain: the ablatable identity pathway
├── decode/               ⬜ noisy-channel beam search + LLM prior
├── atlas/                ⬜ forecastability estimators
└── viz/                  ⬜ topographic movie rendering (real vs forecast)
scripts/                 validation entry points
```

★ = modules where a bug silently fabricates results rather than crashing. These
are written and tested before anything that could depend on them.

`objectives/` being a swappable module is the whole design: Phase 1 is a loop
over that directory with everything else frozen.

---

## 8. Invariants

Enforced in code, not by discipline.

**INV-1 — likelihoods live in a frozen canonical space.** All NLL reported in
nats·s⁻¹·sensor⁻¹. Any normalisation is an explicit invertible affine whose
`log|det J|` is added back. Without this the arm that normalises hardest "wins"
by shrinking the support. See `data/canonical.py`.

**Clipping is never in the likelihood path.** Clipping isn't invertible, so it
breaks change-of-variables silently. `fit_normalizer` accepts a `clip` value for
recording only — it is deliberately excluded from the returned transform.

**The primary null is a Whittle PSD-matched stationary Gaussian**, not white
noise. It already knows 1/f and the alpha peak, so excess predictability is
structure *beyond spectral stationarity*. Get this wrong and the Atlas is just a
paper about 1/f. Always report NLL alongside (a) Δ vs. Whittle and (b) Δ vs.
per-session AR(16).

**Conditioning-side filters must be causal.** Zero-phase `filtfilt` smears future
into past and fabricates spectacular forecastability.

**Guard bands are mandatory and explicit.** 60 s default between train/eval
blocks and between a subject prompt and its query. `guard_samples` has no default
— passing 0 must be a visible choice.

**Every result carries its control rows inline.** Not an appendix.

---

## 9. Known traps

Things that already bit us, or are expected to.

**A DC offset survives every spectral surrogate.** Discovered while validating
the harness: a Gaussian-windowed sine is not zero-mean, so sign-flipping it by
condition puts the class signal in the DC bin — which phase randomisation
legitimately preserves, because DC *is* part of the power spectrum. This is the
shape of the block-design confound (drift, impedance, head movement correlating
with condition). Hence `dc_only` as a first-class control.

**The time-shift control doesn't do what it claims.** Shifting every trial
identically lets a scorer that refits internally simply relearn the shifted
template — it scores unchanged. It is demoted to a diagnostic here. The real
control must be applied in **continuous time before epoching**, where this suite
cannot reach; pass it via `extra=`.

**Permutation thresholds are themselves noisy.** At 200 permutations we saw a
control pass at 0.5833 against a 0.5834 threshold. Use ≥500, more for
publication.

**R1 — the highest-risk item in the whole program.** `log p(X | w)` sums over
306×100 ≈ 30,600 dimensions while the word-discriminative information is
plausibly **under 1 bit**. The between-candidate score difference is a tiny
signal riding on an enormous common-mode term. Mitigation is built in from day
one, not as a rescue: a discriminatively-trained low-rank likelihood subspace,
plus CFG-style nuisance cancellation `log p(X|w) − log p(X|∅)` — which at weight
1 is exactly pointwise mutual information and cancels the subject-specific
nuisance common to all candidates. If it still fails, the forecaster survives as
a pretraining objective and Atlas instrument, and decoding goes discriminative.

**A zero-initialised density head starves the trunk.** The mixture head was
initialised fully at zero so it would start as a standard normal — which makes
`d(log p)/d(h)` identically zero, so the head trains and the encoder receives no
gradient at all on the first step. Caught by the "gradient reaches the trunk"
assertion, not by inspection. Weights are now small-but-nonzero (std 1e-3).

**Compute the LM-only baseline in week 1.** With a 50-word vocabulary and real
sentence context, a language model alone may already reach 40–55% BAcc@10. If
that is near the published 42.96% cross-subject baseline, the benchmark has far
less headroom than it appears, and effort should move to the Atlas.

---

## 7a. The Phase-1 apparatus, exercised end to end

`scripts/run_bakeoff.py` runs all five arms on a synthetic corpus whose ground
truth is known by construction: subject identity is **strong and easy**
(per-subject 1/f exponents and channel gains — the carriers FMScope names),
while the task label is **weak and shared across subjects**. An arm scoring high
on subject and low on label has taken exactly the Identity Trap shortcut.

Observed run (32 sites / 99 channels, 6M rung, 4e12 FLOPs per arm, 202 s CPU):

| arm | steps | GF/step | label acc (chance 0.25) | subject η² ratio | verdict |
|---|---|---|---|---|---|
| masked | 105 | 38.05 | 0.562 | 3.3× | mild |
| jepa | 78 | 50.93 | 0.552 | 5.8× | substantial |
| ar_latent | 78 | 50.92 | 0.552 | 2.4× | mild |
| ar_lik | 105 | 37.91 | 0.359 | **1.9×** | clean |
| peripheral | 105 | 37.88 | 0.417 | 3.0× | mild |

Compute spread across arms: **0.57%**. Note the EMA arms cost +34% per step and
automatically took fewer steps (78 vs 105) — a fixed step count would have
silently gifted them extra compute.

**This is a rehearsal, not evidence.** A few dozen steps on toy data settles
nothing about which objective is better, and the identity ordering here is at
noise level. What it establishes is that the apparatus produces the right table,
with the identity column alongside accuracy rather than after it, under a
correctly enforced budget.

⚠️ **Loss is not comparable across rows** — the arms optimise NLL (nats), MSE and
smooth-L1 on different targets. Read loss within a row only. The cross-arm
columns are label accuracy and the subject ratio, which share a scale.

---

## 8a. Episodic sampling — how H3 is trained for

A new subject enters as a **prompt**, never a gradient step. Four choices, each
closing a specific failure:

| Choice | Closes |
|---|---|
| **guard band ≥ 60 s** | a prompt 10 s before the query shares slow drift, head position and alpha state — the model "adapts" by reading an autocorrelated neighbour |
| **K = 0 with p = 0.1** | gives an unconditional model free, and makes `Δ log p = log p(x\|C_K) − log p(x\|∅)` — i.e. `I(x; C_K)` — an Atlas axis by-product |
| **cross-session 30%** | otherwise the model learns "match this session's noise floor" and collapses when a subject returns on another day |
| **mismatched negatives 15%** | without them the model ignores a decorative prompt and H3 is untested; the margin doubles as a subject-ID score, which is itself a leakage measurement |

K is sampled log-uniformly over 30 s – 40 min so the model sees the whole
adaptation curve — the exact axis H3 is evaluated on.

Verified over 3,000 episodes, with the guard checked by **independent raw index
arithmetic** rather than the sampler's own overlap helper:

| Check | Result |
|---|---|
| same-recording prompt chunks examined | 84,163 |
| prompt/query overlaps | **0** |
| guard-band violations | **0** |
| closest approach to a query | exactly 60.0 s (binding, not vacuous) |
| 1/f exponent recovery vs ground truth (0.5–2.0) | max error **0.009** |

⚠️ **Known limitation.** Cross-session prompts reached only 10.8% against a 30%
target, because most synthetic subjects are single-session. This will bite on the
real corpus: LibriBrain100's 32 broad subjects have ~40 min each and are likely
single-session, so H3's session-drift robustness can only be tested on the deep
subject and on multi-session corpora (Armeni, MOUS). Worth confirming before
claiming drift robustness.

---

## 9a. One substitution from the design, stated plainly

The design called for Mamba-2 selective state-space blocks, for linear-time
scaling to the full 2.5-minute context. `models/backbone.py` implements those
slots with **full causal attention** instead, and exposes `SSMBlock` as the
drop-in interface.

The reason is verifiability. A hand-rolled chunked associative scan is easy to
get subtly wrong in ways that leak future information — and a leak in *that*
direction fabricates the exact result the project is trying to measure.
Attention is O(N²) but its causality is trivially testable, and
`validate_backbone.py` does test it by perturbation. Wire in `mamba-ssm` once
that harness can be pointed at it.

At Phase-1 scale (16 s windows, ~1,600 positions) quadratic attention is
entirely affordable. The substitution only bites at the Phase-2 long-context
stage.

---

## 10. Roadmap

| Phase | Deliverable | Gate |
|---|---|---|
| **0** | Harness ✅, data layer ✅, reproduce MEG-XL, FMScope its checkpoint, the control table | published number reproduces; null controls at chance |
| **1** | 5-arm matched-compute bake-off | forecasting shows lower identity leakage at equal-or-better transfer |
| **2** | Subject-in-context vs `emb`/`lora`/`full-ft` at matched minutes | in-context ≥ fine-tuning, with lower leakage |
| **3** | Scaling ladder + cross-sensor transfer | zero-shot transfer beats from-scratch on an unseen array |
| **4** | Neural Predictability Atlas | — (architecture-independent; ships regardless) |
| **5** | Noisy-channel decoder + the demo | — |

**Phase 0 may produce a standalone result in days:** nobody has measured identity
leakage in a cross-subject MEG model, and the Identity Trap paper predicts
fine-tuning makes it sharply worse. Either answer is worth writing up.

### The demo

Split screen. Left: the subject's real MEG as a topographic movie. Right:
NeuroCast's forecast of the next second, rendered identically — the model
dreaming the brain forward. Below: decoded text with a calibrated confidence
trace, and a null-control panel that stays flat at chance while the real one
tracks. New subject, ten minutes of context, no training.

---

## 11. References

**The gap**
- [A Roadmap for MEG Foundation Models][arXiv 2609.04461] — Sept 2026
- [The Identity Trap in EEG Foundation Models][arXiv 2606.06647]
- [Source Attribution for Non-Invasive Brain-to-Language Retrieval][arXiv 2605.24524]
- [Scaling laws for decoding images from brain activity][arXiv 2501.15322] — per-subject data scales; subject count doesn't
- [Benchmarking Probabilistic TS Forecasting on Neural Activity][arXiv 2510.18037] — the ~1.5s ceiling, stated open
- [Forecastability as an Information-Theoretic Limit][arXiv 2603.27074] — never applied to neural data
- [Temporal autocorrelation leakage in EEG decoding][arXiv 2405.17024]

**Baselines and data**
- [MEG-XL: Data-Efficient Brain-to-Text via Long-Context Pre-Training][arXiv 2602.02494] · [code](https://github.com/neural-processing-lab/MEG-XL)
- [LibriBrain100][arXiv 2608.25204] · [2026 PNPL Competition][arXiv 2609.03231]
- [Brain2Qwerty][arXiv 2502.17480]
- [MOABB reproducibility study][arXiv 2404.15319]

[arXiv 2609.04461]: https://arxiv.org/abs/2609.04461
[arXiv 2606.06647]: https://arxiv.org/abs/2606.06647
[arXiv 2605.24524]: https://arxiv.org/abs/2605.24524
[arXiv 2501.15322]: https://arxiv.org/abs/2501.15322
[arXiv 2510.18037]: https://arxiv.org/abs/2510.18037
[arXiv 2603.27074]: https://arxiv.org/abs/2603.27074
[arXiv 2405.17024]: https://arxiv.org/abs/2405.17024
[arXiv 2602.02494]: https://arxiv.org/abs/2602.02494
[arXiv 2608.25204]: https://arxiv.org/abs/2608.25204
[arXiv 2609.03231]: https://arxiv.org/abs/2609.03231
[arXiv 2502.17480]: https://arxiv.org/abs/2502.17480
[arXiv 2404.15319]: https://arxiv.org/abs/2404.15319
[arXiv 2606.01264]: https://arxiv.org/abs/2606.01264
[HF `pnpl/LibriBrain`]: https://huggingface.co/datasets/pnpl/LibriBrain
[cam-can.org]: https://cam-can.mrc-cbu.cam.ac.uk/dataset/
[github.com/courtois-neuromod]: https://github.com/courtois-neuromod

---

## License

Code: MIT. Datasets carry their own licences — check each before redistribution.
Cam-CAN, NSRR and parts of MOUS require an application or DUA.
