# NeuroCast — working notes for Claude

Research code testing whether forecasting the brain (MEG) is a better pre-training
objective than masking, with the leakage controls built first. The README is the
source of truth: §2 status and results, §9 known traps, §10 invariants.
`docs/recovered/VERIFICATION.md` is the claim-by-claim ledger against the lost
version's pitch (`docs/recovered/neurocast-pitch.html`).

## Environment

- Windows 11; Git Bash or PowerShell. Run scripts from the repo root; they put the
  root on `sys.path` themselves.
- `.venv\Scripts\python.exe`: CPU torch. Validators, audits, data scripts.
- `.venv-gpu\Scripts\python.exe`: CUDA torch (RTX 3050, 4 GB). Training and GPU
  sweeps. The Windows driver spills into shared RAM instead of failing (about 20×
  slower), so cap it with `torch.cuda.set_per_process_memory_fraction` and keep
  eval batches small.
- 16 GB RAM, often only 1–5 GB free. Stream recordings; never load a corpus whole.
- An Application Control policy can block a newly installed compiled extension
  for a few minutes (seen with scipy and scikit-learn). Re-test the import before
  rewriting around it. Never change security settings.
- The `neurocast/` package imports only NumPy and torch, never scipy (README §3).

## Data

- `data/` is a junction to `D:\neurocast-data`, a USB spinning disk. **Never
  `Remove-Item -Recurse` it**; remove the link alone with `cmd /c rmdir data`.
- Every download goes through `scripts/data_setup.py` (`check`, `fetch`,
  `fetch-files`, `fetch-osf`). Nothing large on C:.
- The disk reads at ~40–65 MB/s and occasionally drops a read. Read each recording
  once per pass and feed every model from that pass, use the retrying reads
  (`neurocast/train/pretrain.py` `_read_retrying`), and run disk-heavy jobs one at
  a time.
- Credentials (`HF_TOKEN`, Radboud WebDAV, Kaggle) live in `.env`, which is
  gitignored. Never print or commit them.

## Before trusting any number

- `.venv\Scripts\python.exe scripts/run_all.py`: 16 validators, ~3 min. All must
  pass after any change under `neurocast/`. `--only <name>` runs a subset.
- New behaviour gets a provoking case in the matching `scripts/validate_*.py`:
  show the check fails on the broken version and passes on the fixed one.

## Research rules

- Write the decision rules (which result supports or kills a hypothesis) into the
  README before running the experiment. Report the outcome against them, including
  when it goes against the hypothesis.
- Every result carries its controls inline: timing-only and onset-only baselines,
  the timing-matched test, information beyond covariates in nats, and the
  reference NLL rows (README §10).
- Person-identity ratios are capped at (N−1)/(S−1). Compare arms only at equal N
  and report the ceiling.
- Masked arms keep the visible-position term; plain masked collapses on real data
  (README §2.1–2.2).
- Split by session, story or listener, with guard bands. Held-out sessions never
  reach training (`validate_pretrain.py` section 6).
- Drop classes or trials a decoder never scored (`UNSCORED` in
  `neurocast/audit/timing_matched.py`); scoring the −1e6 marker leaked story
  identity on MEG-MASC.
- Pitch numbers are claims to re-verify. Record each status in the ledger.

## Long runs

- Training evaluates and checkpoints every `eval_every` steps (500 in the scale
  and multi-listener scripts), and resume is bit-exact. Pause disk-heavy work only
  right after a checkpoint.
- **Launch every long-running job detached, through WMI:**
  `powershell -ExecutionPolicy Bypass -File scripts\launch_detached.ps1 runs\<dir>\chain.cmd`.
  Never use `Start-Process` or a background tool call for a run that must outlive
  the session: those are children of the app's shell, and on 16 October 2026 every
  chain started that way died silently at 00:15 when the session ended. The
  launcher refuses to start a chain that is already running.
- Before relaunching a background chain, stop the old one and check for duplicate
  Python processes.
- Finished runs' full checkpoints were slimmed on 16 October 2026: the optimiser,
  scheduler and EMA-teacher state were dropped, with the originals moved to
  `D:\neurocast-data\trash\runs-2026-10-16`. They load for evaluation and keep
  their logs, but cannot resume training.
- Outputs go to `runs/` (gitignored). Results that matter are written into the
  README and ledger; figures go to `docs/figures/`.

## Git

- `origin` is github.com/kriegher12/neurocast, a **public** repo. Ask before
  pushing to `main`. No AI attribution lines in commits or PRs.
- `core.autocrlf=true` and `main` mixes line endings: `README.md`,
  `docs/recovered/VERIFICATION.md` and `scripts/data_setup.py` are CRLF, the rest
  LF. Keep each file's style so diffs show only real changes.
- `__pycache__/*.pyc` files are still tracked on `main`; never stage them.
- The competition-protocol finding (ledger §4b) was meant to reach the LibriBrain
  organisers privately. Ask before publishing anything further about it.
