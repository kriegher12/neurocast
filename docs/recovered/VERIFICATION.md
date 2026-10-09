# Verifying the lost version against its pitch

An earlier, further-along NeuroCast was built on another machine (RTX 4050 6 GB
laptop) and lost after 29 September 2026. The only surviving record is a faculty
pitch speaker script, preserved here as [`neurocast-pitch.html`](neurocast-pitch.html)
with its two figures ([score chart](pitch-score-chart.png),
[predictability](pitch-predictability.png)). It quotes real-data results from a
README section that no longer exists.

This ledger checks every checkable claim in that pitch against the rebuilt code
and freshly downloaded LibriBrain100 data. Claims are never edited to fit: a
mismatch is recorded as one.

**Status key** — ✅ reproduced / matches · ≈ close, with a stated difference ·
❌ does not match · ⏳ not yet re-run · — not checkable without the lost code

Re-run everything here with:

```powershell
.venv\Scripts\python.exe scripts\run_libribrain_audit.py --stage nobrain
.venv\Scripts\python.exe scripts\run_libribrain_audit.py --stage brain
.venv\Scripts\python.exe scripts\run_libribrain_audit.py --stage matched
.venv\Scripts\python.exe scripts\run_phase_control.py
.venv\Scripts\python.exe scripts\run_libribrain_predictability.py
.venv-gpu\Scripts\python.exe scripts\run_pretrain_pilot.py --arm ar_lik
.venv-gpu\Scripts\python.exe scripts\run_pretrain_pilot.py --arm masked
.venv-gpu\Scripts\python.exe scripts\run_pretrain_pilot.py --arm masked --visible-weight 1.0
.venv-gpu\Scripts\python.exe scripts\run_pretrain_pilot.py --evaluate
.venv-gpu\Scripts\python.exe scripts\run_pretrain_pilot.py --name masked_lowlr
.venv-gpu\Scripts\python.exe scripts\run_pretrain_pilot.py --evaluate --collapse-only
```

Results land in `runs/libribrain_audit/*.json` and `runs/pretrain/pilot_eval.json`.

## 1. Data and protocol

| Pitch claim | Rebuilt / measured | Status |
|---|---|---|
| 32 broad listeners: sub-1–12 whole sessions, sub-13–22 first halves, sub-23–32 first quarters | Hub listing: exactly that; partial files carry `_desc-firsthalf` / `_desc-firstquarter` | ✅ |
| 20 listeners "published only as partial sessions that the official loader cannot find" | they sit in `derivatives/serialised_competition/` with a `desc-` suffix pnpl's filename builder never generates | ✅ |
| about 17 GB for all 32 listeners | 17.4 GB (64 sessions) | ✅ |
| licence CC-BY-NC-4.0 (non-commercial) | both dataset cards say `cc-by-nc-4.0` (the old README said CC-BY-4.0 — corrected) | ✅ |
| fit chapter 11, test chapter 12 (stimulus-disjoint) | same | ✅ |
| 32,198 fit / 32,251 test trials | **32,198 / 32,251** — identical, once trials are limited to windows inside each recording | ✅ |
| timing classifier trained on 60,000 words from the deep listener's other sessions | 60,000 sampled from 285,693 word trials in 112 sub-0 Sherlock sessions | ✅ (setup) |
| language model trained on 650,000 words from 112 training sessions | **650,138 words from 112 sub-0 Sherlock sessions** | ✅ |
| pre-training pilot data: deep listener, "3 sessions, 47.5 min" | sub-0 Sherlock2 sessions 1–3: **47.5 min** exactly | ✅ |
| "37.6 min of story-only data" | first-to-last word: 45.0 min; minus a 16% held-out tail for loss tracking: **37.8 min** | ≈ (definition inferred) |
| sensor geometry for NeuroCast on LibriBrain | the release's `metadata/neo/sss_cal.dat`: 306 sensors, types match the H5 exactly, positions match `sensor_xyz.json` to 7e-9 m | ✅ (rebuilt) |

## 2. Findings that need no brain data

Held-out chapter 12, 32 listeners, 50 words. BAcc@1 / BAcc@10; chance 0.02 / 0.20.
Scored on all 53,018 test-chapter word onsets (the brain stage re-scores timing
on the 32,251 MEG-covered ones: 0.160 / 0.669).

| Pitch claim | Rebuilt | Status |
|---|---|---|
| speech timing alone: 0.15–0.18 / 0.61–0.67 | **0.166 / 0.666** | ✅ |
| language model, true previous word (chart): ~0.15 / ~0.62 | **0.221 / 0.639** | ≈ top-10 close; top-1 higher (bigram smoothing differs) |
| language model + timing: ~0.35 / 0.80 | **0.372 / 0.818** (weight 0.56, tuned on chapter 11) | ✅ |

## 3. Brain decoders and controls

| Pitch claim | Rebuilt | Status |
|---|---|---|
32,198 fit / 32,251 test MEG-covered trials, session-scalar normalisation,
64 ms patch means (306 × 8 features) for the linear decoders and the MLP.

| Pitch claim | Rebuilt | Status |
|---|---|---|
| linear (pooled): 0.027 / 0.257 | **0.031 / 0.259** (ridge α 10⁴) | ≈ top-10 matches; top-1 +0.004 |
| linear, one per person: 0.028 / 0.245 | **0.027 / 0.245** | ✅ |
| MLP: 0.036 / 0.278 | **0.034 / 0.268** | ≈ top-10 0.010 lower |
| CNN on raw 125 Hz: 0.044 / 0.298 | **0.042 / 0.301** | ✅ |
| onset-only fake signal reproduces 21–53% of the linear gain (chart ~0.215) | **0.024 / 0.216 — 28%** of the linear gain | ✅ |
| phase-scrambled input near chance (chart ~0.217) | as run: **0.024 / 0.236** — above the shuffled p95 (0.209). With each window's mean also removed: **0.023 / 0.201** — chance | ✅ once the window mean goes (decomposition below) |
| shuffled-label null at chance | 10 occurrence-level shuffles: mean **0.020 / 0.203**, p95 0.023 / 0.209 | ✅ |
| identity audit, raw signal: 3× | **1.8×** subject η² over its permutation null (8,000 test trials, 64 ms features); subject probe 7.5% vs 3.1% chance | ❌ (lower; feature set unstated in the pitch) |

**Why the phase-scrambled control first sat above chance** (`scripts/run_phase_control.py`,
same linear decoder refitted on each input). The rebuilt surrogate randomises
Fourier phases but, by design, leaves the DC bin — each window's channel means —
untouched. Those means are onset-locked: the slow evoked field shifts them.

| Input to the linear decoder | BAcc@1 / BAcc@10 | Share of the linear gain |
|---|---|---|
| real signal | 0.031 / 0.259 | 100% |
| phase-scrambled, window means kept (the audit's control as run) | 0.027 / 0.234 | 58% |
| phase-scrambled, window means removed (spectrum only) | 0.023 / **0.201** | 2% |
| window means only, constant in time | 0.032 / **0.240** | **67%** |

The per-trial spectrum carries nothing; the 0.512 s window mean carries two
thirds of what the linear decoder finds. The pitch lists "window mean only" among
its fake signals but its surviving text gives no number for it.

## 4. The decisive tests

32,251 test trials in 417 strata (next-onset, gap and second-onset bins;
duration and loudness tertiles, loudness from the chapter's audio). Gain = class-
balanced normalised rank of the true word among the stratum's words, minus the
mean of 200 occurrence-level permutations within strata. The rule, fixed before
the run: under a third surviving = "mostly timing".

| Scorer | Unmatched gain | Matched gain | Surviving | Nats/word beyond covariates [95% CI] |
|---|---|---|---|---|
| timing only (no brain) | +0.323 | +0.069 | 21% — mostly timing | — (it *is* the covariates) |
| linear | +0.058 | +0.030 | **52%** | +0.0076 [+0.0046, +0.0102] |
| linear, one per person | +0.038 | +0.028 | 72% | +0.0012 [−0.0000, +0.0026] |
| MLP | +0.066 | +0.032 | 49% | +0.0184 [+0.0144, +0.0221] |
| CNN | +0.093 | +0.056 | 60% | **+0.0309** [+0.0229, +0.0386] |
| frequency-only scores | | | | +0.0000 [+0.0000, +0.0000] |

| Pitch claim | Rebuilt | Status |
|---|---|---|
| about half of the gain survives matching on timing, length, loudness | linear **52%** (MLP 49%, CNN 60%) | ✅ |
| a timing-only scorer does not survive (implied by the test's design) | 21% → "mostly timing" | ✅ |
| the stronger the decoder, the more genuine word information it finds; the CNN five times the linear | nats beyond covariates rise linear → MLP → CNN; CNN / linear = **4.1×** (matched rank gain: 1.8×) | ≈ |
| best decoder adds ~0.01 nats/word beyond timing, length, loudness | CNN **0.031**; linear 0.008 | ❌ about 3× more here (the pitch's own 5× ratio implies its linear added ~0.002) |
| frequency-only scores give zero | exactly 0 | ✅ |

### Stronger decoders, the pitch's next step

"Stronger decoders through the timing-matched test (CNN tuned on fit data only,
EEGNet)" — `scripts/run_decoder_sweep.py`, GPU, same trials and test. The CNN's
width, kernel, dropout and learning rate were chosen from 36 settings by
validation inside chapter 11 only. The choice was width 32, kernel 9, dropout
0.5, lr 2e-3: the pitch's CNN with more dropout.

| Decoder | BAcc@1 / BAcc@10 | Surviving matching | Nats/word beyond covariates [95% CI] |
|---|---|---|---|
| linear (§4) | 0.031 / 0.259 | 52% | +0.008 [+0.005, +0.010] |
| subject CNN (one spatial layer per listener) | 0.031 / 0.262 | 33% | +0.006 [+0.002, +0.009] |
| EEGNet | 0.038 / 0.296 | 44% | +0.033 [+0.026, +0.038] |
| CNN, the pitch's (§4) | 0.042 / 0.301 | 60% | +0.031 [+0.023, +0.039] |
| CNN, tuned on chapter 11 | 0.045 / 0.310 | 57% | +0.046 [+0.038, +0.054] |
| CNN, tuned, 5 seeds averaged | **0.049 / 0.322** | 60% | **+0.048** [+0.040, +0.056] |

The pitch's trend continues: a stronger decoder finds more word information
beyond timing, length and loudness. The best here finds 6× the linear decoder's,
and the share of its gain that survives matching stays near 50–60%. A per-listener
spatial layer hurts, probably by overfitting ~1,000 trials per person.

## 4b. Under the competition's own protocol

The pitch: "In the competition's test data, word onsets within each sentence are
provided, so this information is available to everyone." pnpl's competition
module (`pnpl.competition`, the 2026 holdout loader) confirms it. Each listener's
`holdout01_sentence.npz` carries `word_onsets_s` (the onset of every word,
seconds from sentence start) next to the sentence MEG. Words are classified from
**1.0 s** windows (`WINDOW_SECONDS = 1.0`; the pitch said half a second) over a
fixed 50-word list (`PRIMARY_VOCAB`). A second source, `holdout2_word`, holds
isolated word epochs, and those carry no onsets of neighbouring words.

The same audit, re-run with 1.024 s windows (16 whole patches) and the
competition's 50 words (`--window 1.024 --vocab competition`; results in
`runs/libribrain_audit_w1024_competition/`). Chapters 11/12 contain 47 and 45 of
the 50 words. "always" occurs in chapter 12 only, so the brain decoders cannot
learn it; the right-hand column re-scores everything on the 44 words present in
both chapters.

| | BAcc@1 / BAcc@10, 50 words | 44 shared words (chance 0.023 / 0.227) |
|---|---|---|
| speech timing only — no brain data | **0.123 / 0.687** | **0.144 / 0.704** |
| language model + timing — no brain data | 0.313 / 0.839 (all onsets) | — |
| linear | 0.020 / 0.236 | 0.029 / 0.267 |
| linear, one per person | 0.025 / 0.236 | 0.026 / 0.258 |
| MLP | 0.017 / 0.202 | 0.022 / 0.243 |
| CNN | 0.021 / 0.224 | 0.024 / 0.269 |
| shuffled labels (linear, 10×) | 0.017 / 0.196, p95 0.205 | — |

The decoders are the pitch's, sized for 0.5 s windows. Re-tuning the CNN for
1.024 s windows on chapter 11 (the same 36-setting sweep as §4) chose width
128, kernel 9, dropout 0.3, lr 1e-3. Its best validation BAcc@10 inside chapter
11 was 0.236, against 0.349 for the 0.512 s task, so tuning finds no CNN that
decodes these windows well. Scored on chapter 12 (`decoder_sweep.json` in the
same folder):

| Decoder, 1.024 s windows | BAcc@1 / BAcc@10 | Surviving matching | Nats/word beyond covariates [95% CI] |
|---|---|---|---|
| CNN, tuned | 0.030 / 0.226 | 35% | −0.017 [−0.026, −0.001] |
| CNN, tuned, 5 seeds | 0.026 / 0.264 | 46% | −0.028 [−0.049, +0.016] |
| subject CNN | 0.024 / 0.214 | 27% | −0.013 [−0.021, −0.001] |
| EEGNet | 0.024 / 0.223 | 35% | +0.012 [−0.004, +0.036] |

Stronger and re-tuned decoders do not change the picture: under the
competition's protocol no decoder here adds detectable word information beyond
timing, length and loudness, while timing alone scores 0.69.

| Timing-matched test (400 strata) | Unmatched gain | Matched gain | Surviving | Nats/word beyond covariates [95% CI] |
|---|---|---|---|---|
| timing only | +0.310 | +0.087 | 28% — mostly timing | — |
| linear | +0.032 | +0.011 | 35% — below the null's p95: **no word information beyond timing** | +0.001 [−0.003, +0.005] (3 never-trained words left out) |
| linear, one per person | +0.025 | +0.015 | 59% (just above p95) | −0.001 [−0.003, +0.000] (9 left out) |
| MLP | +0.028 | +0.010 | 35% — below p95 | +0.001 [−0.013, +0.023] |
| CNN | +0.048 | +0.023 | 48% | −0.008 [−0.026, +0.027] |

Under the competition's protocol, on LibriBrain chapters 11–12, speech timing
alone puts the heard word in the top ten ~70% of the time. No brain decoder here
reaches 0.27, and none adds detectable information beyond timing, length and
loudness. Two limits: the competition's own test stimuli are probably not these
chapters ("really" and "new" occur in neither), and its isolated-word epochs
carry no timing. The pitch planned to report "one possible issue with the
competition data … privately to the organisers rather than publish" — this
section is that issue, measured.

## 4c. A second dataset: MEG-MASC, timing only

The pitch planned "MEG-MASC as a second dataset for the timing test"
(`scripts/run_megmasc_timing.py`; events fetched with `data_setup.py fetch-osf`).
MEG-MASC (Gwilliams et al.) is synthesised speech: four stories with random word
lists interleaved. Relative word timing is identical across its 27 listeners to
1e-13 s, so one listener's events stand for all. Same 50-most-frequent-words
task and timing features; leave one story out, pooled.

| | BAcc@1 / BAcc@10 |
|---|---|
| timing only, 0.512 s windows (3,348 words) | **0.095 / 0.416** |
| … words in running sentences (3,281) | 0.097 / 0.418 |
| … words in random word lists (67 — too few to read) | 0.014 / 0.345 |
| timing only, 1.0 s windows | 0.093 / 0.419 |
| shuffled labels, 20× (0.512 s) | 0.021 / 0.203, p95 0.226 |
| LibriBrain for comparison, trained on 2,500 words (MEG-MASC's fit size) | 0.121 / 0.549 |
| LibriBrain, trained on 60,000 words (the audit) | 0.176 / 0.662 (one trial per occurrence) |

The confound replicates: on a second dataset, speech timing alone predicts the
heard word far above chance. It is weaker than on LibriBrain even at matched
training size (0.416 vs 0.549). The likely reason is synthesised rather than
natural reading, but that is not tested here. In MEG-MASC's word lists the
words run back to back, so the next onset still gives away the word's own
duration; the lists are not a timing-free control.

**The brain half** (`scripts/run_megmasc_audit.py`). All 27 listeners,
session 0, 208 KIT sensors, notch 60 Hz, 0.1–100 Hz, 250 Hz, 0.512 s windows,
90,396 trials, leave one story out:

| | BAcc@1 / BAcc@10 | Nats/word beyond covariates [95% CI] |
|---|---|---|
| timing only | 0.095 / 0.412 | — |
| linear decoder | 0.031 / 0.231 | **+0.0035** [+0.0023, +0.0047] (3 single-story words left out) |
| frequency-only scores | — | +0.0000 |

The linear decoder carries a little word information beyond timing, length and
loudness: about half of LibriBrain's linear decoder (0.0076), from listeners with
an hour each rather than chapters 11–12 heard by 32 people. The
**timing-matched test is not usable here.** MEG-MASC's four short stories give
686 strata with ~5 word occurrences each, and even the timing-only scorer reads
a *negative* matched gain (−0.032). A stratum that small holds too few distinct
words for a within-stratum ranking to mean anything, so it is reported as
underpowered, not as a verdict. (Five listeners first: linear 0.030 / 0.230,
+0.0013 nats/word.)

A first run read **+0.029** nats/word here. It was an artefact, found because it
contradicted the decoder's near-zero rank gain. "roy" and "chad" occur only in
story 1 and "allan" only in story 3, so each fold's decoder had never seen them
and scored them −1e6. That marker told the one-weight model which story a trial
came from, and the story predicts the names. Within every story the decoder
added 0.000–0.004. `info_beyond_covariates` now leaves out classes some trials
have no score for. `validate_libribrain.py` §9 pins it: pure noise with such a
marker read +0.25 nats/word before the fix and exactly 0 after. The pitch-protocol
results above have no unscored classes and are unchanged. On the competition
protocol, the linear decoders' "beyond" values moved from −0.003/−0.007 to
+0.001/−0.001, with the same conclusion.

## 5. Predictability

| Pitch claim | Rebuilt | Status |
|---|---|---|
| almost all predictability comes from the spectrum | sub-1, 8 sensors: from the spectrum 0.648 / 0.186 / 0.124 / 0.026 nats/sample at 4 / 16 / 64 / 256 ms; beyond it ≤0 / ≤0 / 0.018 / 0.020 — 0%, 0%, 12%, 43% of the total | ✅ (beyond only matters where little is predictable) |
| beyond-spectrum predictability is small: ~0.005–0.028 nats/sample at 4–256 ms (figure) | across sessions 0.016–0.034 at 64–256 ms in broadband/theta/alpha; **negative at 4 ms** in several bands where the figure is positive | ≈ magnitude; ❌ at 4 ms |
| across sessions > within one session for broadband, theta, alpha | yes, at 16–256 ms | ✅ |
| 40–60% of that excess is the session changing | broadband **48%** ✅; theta 79%, alpha 77% (higher) | ≈ |
| beta and gamma similar across and within | beta: share 32%, both small — plausibly "similar"; gamma: share 91% — **not** similar | ≈ beta; ❌ gamma |

Setup differences that could explain the residual: 180 s train/test segments per
condition (the pitch does not state its length), 8 magnetometers spread evenly
(the pitch does not say which 8 sensors), and the heteroscedastic model at the
null's 64-lag mean order (a lower order fabricates large negative values on
band-limited signals — `validate_atlas.py` section 11).

## 6. Model and training code

| Pitch claim | Rebuilt / measured | Status |
|---|---|---|
| sensor embedding from position, orientation, type; new layout adds no parameters | `tokenizer/`, validated | ✅ |
| 16-sample (64 ms) patches; 8 latents per group, 5 groups, 1 cross layer → 5 tokens per patch | defaults in `tokenizer/perceiver.py` | ✅ |
| transformer with rotary encoding, sliding window 256, causal for forecasting, **bidirectional for masked** | matches — the bidirectional masked trunk was re-derived independently in this rebuild (README §9) | ✅ |
| rungs: tiny 1.3M · 6m 24.9M · 25m 62.7M ("names are historical") | **1.3M · 24.9M · 62.7M** (90m 182.8M, 320m 475.5M) | ✅ |
| seven objectives: masked, masked spectral, JEPA, latent forecasting, likelihood forecasting, CPC, eye/heart | five arms; masked-spectral exists only as an option; **CPC missing** | ❌ partial |
| masked target: hidden 1-s blocks (40%) of whitened per-region PCA, rank 32; visible-position term (weight 1.0) | rebuilt in `train/pretrain.py` and `MaskedReconstruction(visible_weight=…)` | ✅ (rebuilt) |
| trainer: AdamW 1e-3, betas (0.9, 0.95), wd 0.01, clip 1.0 | same | ✅ |
| 100 warm-up steps, cosine to 10%, mixed precision, bit-exact resume | rebuilt; resume verified bit-exact on CPU for both arms (`validate_pretrain.py`); bf16 on the RTX 3050 | ✅ (rebuilt) |
| multi-GPU via torchrun ("written, not yet tested") | not rebuilt | — |
| 6m rung pilot at 0.18 s/step on an RTX 4050 | **0.21–0.25 s/step** on this machine's RTX 3050 4 GB (bf16); 15–18 min per arm | ≈ (slower GPU) |
| masked model collapsed on real data: "its output ignored its input" | plain masked arm at step 4,000, 48 held-out windows: prediction spread **0.000**, sensitivity to swapped context **0.000**, R² 0.000 — **COLLAPSED** | ✅ — and on 61 h too: collapsed at steps 2,000 and 10,000, held-out flat at 0.978 (README §2.2) |
| fixed by also scoring visible positions (weight 1.0) | spread 0.076, sensitivity **0.95**, R² **+0.069** — reads its input | ✅ |
| a diagnostic script checks any masked checkpoint | `audit/collapse.py`, validated on AR(1) targets (`validate_pretrain.py` §5) | ✅ (rebuilt) |
| "a linear predictor recovers 34–48% of a hidden patch from its neighbours" | one patch hidden, ±2 patches of the same region visible: **14%**; under the pitch's own 1-s block masks: −4.5% (the neighbours are hidden too) | ❌ lower (target definition — whitened rank-32 components, every component weighted equally — may differ) |
| "lower learning rates and other targets still collapse" | masked at lr 3e-4 (else identical): held-out 0.906, spread 0.000, sensitivity 0.000 — **COLLAPSED**; other targets not re-run | ✅ lower LR; — other targets |
| forecasting features carry ~3× more person information than masked (29× vs 10×) | at N = 2,048 windows (64 per listener, ceiling 66×): forecasting **11.8×**, fixed masked **4.9×** — **2.4×** more; at N = 384 (ceiling 12.4×): 3.1× vs 1.8× | ≈ direction and ratio; absolute values depend on N, which the pitch does not give. **At scale** (61 h, 3 seeds, best checkpoints, README §2.2): forecasting **33.2×**, fixed masked **4.8×**, 6–8× more in every seed. The direction holds and the gap widens; forecasting's 33× is close to the pitch's 29× |
| raw signal: 3× | per-channel log amplitude of the same 4-s windows: **32.9×** (N 2,048); 64 ms patch means of word windows: 1.8× (§3) | ❌ — neither raw feature gives 3×; the pitch's raw feature is unstated |
| "the two approaches decode about equally well" | frozen features, same linear probe and split as §3: forecasting **0.027 / 0.243**, fixed masked **0.028 / 0.242**; collapsed masked 0.020 / 0.202 (chance) | ✅ — both below the raw-signal linear decoder (0.259); at scale, 0.245 vs 0.246 (3 seeds each) |
| held-out loss tracking | forecasting overfits: held-out best 0.753 (step 2,000), 0.827 at step 4,000, train 0.41; the evaluation uses the final checkpoint, as the pitch's 4,000-step recipe implies | — (the pitch gives no held-out numbers) |
| 18 validation scripts, `scripts/run_all_checks.py --fast` | 16 validators, `scripts/run_all.py` | ≈ (different set) |
