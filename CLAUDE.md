# CLAUDE.md — project guide for automated assistants

Working notes for anyone (human or AI) picking up this project. Keep it terse,
accurate, and up to date. If a rule here goes stale, fix it.

---

## 1. What this project is

FIT VUT **SUR 2025/2026** semester project (25 points). Build a detector for
one specific target person from:

- face images (PNG), and
- voice recordings (WAV).

Three systems must be submitted:

1. image-only classifier,
2. audio-only classifier,
3. multimodal fusion of both.

Plus `dokumentace.pdf` (~3 pages) and `SRC/`. Full brief: [`assignment.txt`](assignment.txt).

### Deadlines

| Date           | Event                                      |
| -------------- | ------------------------------------------ |
| 2026-05-03     | Evaluation data released (morning)         |
| 2026-05-04 23:59 | Submission deadline (IS upload)          |
| 2026-05-05     | Answer key published                       |
| 2026-05-06     | Results analysis lecture                   |
| Exam period    | Oral defense (mandatory for full grade)    |

### Output format (per eval sample, one line)

```
<stem> <score> <hard_decision>
```

- `stem` = filename without `.wav` / `.png`
- `score` = real number, **higher = more confident target**
- `hard_decision` = `1` if target, `0` otherwise, **assuming prior 0.5**

---

## 2. Data facts (derived from `notebooks/data.ipynb`)

- **Target identity:** `m431` (one person, 3 sessions total)
  - train sessions: `01`, `02`
  - dev session: `03` (disjoint from train)
- **Non-target train:** 13 speakers (`f401–f406`, `m414`, `m416`, `m417`, `m419–m422`)
- **Non-target dev:** 6 speakers (`f407–f409`, `m423–m425`) — **identity-disjoint** from train
- Filename schema: `<identity>_<session>_<prompt>_i<inst>_<take>.{png,wav}`
- Every PNG has a paired WAV of the same stem → multimodal fusion is trivial
  at the sample level.

### Why the split matters

The organizers already respect the right structure:

- target split is **session-disjoint** → measures generalization to a new
  recording sitting of the same person;
- non-target split is **identity-disjoint** → measures rejection of unseen
  strangers.

We must preserve this when we re-carve cross-validation folds on the combined
`train + dev` pool (see §4).

---

## 3. Hard rules (from `assignment.txt`)

- **No external data.** No pretrained embeddings (face or voice). Trainable
  only on provided data. Augmentation of provided data is allowed.
- Evaluation data will contain distribution shift (noise, quality changes).
  Systems must **generalize**, not memorize.
- Reproducing only the lecture baselines (MFCC+GMM, linear classifier on
  images) caps the grade. Own approach required.
- At most 6 results files per group will be scored. Name them meaningfully
  (e.g. `audio_gmm.txt`, `image_cnn.txt`, `fusion_logreg.txt`).
- Submission = one ZIP: `login1_login2.zip` with result files, `SRC/`, and
  `dokumentace.pdf`. **Never include the evaluation data.**

---

## 4. Methodology — non-negotiable

### Validation strategy: session/speaker-aware group K-Fold

A random K-Fold leaks. Use grouped folds:

- **target** rows: group by `session_id` (identity × session).
- **non-target** rows: group by `identity`.

Implementation note: both groupings feed one `GroupKFold` via a virtual
`group` column (target gets session ids, non-target gets identity ids; name
spaces are disjoint). Authoritative splitter lives in `src/data/splits.py`.

All three systems (audio, image, fusion) share the **same fold assignment**.
This is load-bearing for stacking fusion via OOF scores.

Target only has 3 sessions → consider 3-fold Leave-One-Session-Out on target
with matching non-target speaker groups, rather than forcing 5-fold.

### Metrics

Primary:
- **EER** (Equal Error Rate)
- **min-DCF** at prior 0.5 (matches assignment)

Secondary: ROC-AUC for sanity, log-loss for calibration.

Report `mean ± std` across folds. A large std is information — include it.

### Threshold for hard decision

Calibrate OOF scores (Platt or isotonic) and pick the threshold that
implements the Bayes rule at prior 0.5. Do not eyeball.

### Guard rails against leakage

1. Fit scaler / PCA / UBM / CMN statistics **only on train fold**, apply on val.
2. **Augment only the train fold** of a given split — never produce augmented
   copies of val samples.
3. One experiment changes one axis. If you move three knobs and it improves,
   you haven't learned anything.

---

## 5. Repository layout

```
assignment.txt          — course brief (Czech, do not edit)
README.md               — user-facing intro
CLAUDE.md               — this file
pyproject.toml / uv.lock — uv-managed deps; Python 3.12
main.py                 — entrypoint placeholder
data/                   — provided corpus (gitignored)
notebooks/              — exploratory / visual analysis
  data.ipynb            — manifest, per-session / per-modality stats
src/
  data/                 — manifests, split logic, fold generation
  features/             — MFCC / filterbank / image feature extractors
  models/               — per-modality classifiers
  fusion/               — score / feature / decision fusion
  eval/                 — metrics, thresholding, OOF helpers
experiments/            — one .md per experiment (see experiments/README.md)
results/                — result files for IS upload (gitignored for binaries)
logs/                   — runtime logs (gitignored)
```

---

## 6. Developer workflow

### Running things

- Everything runs via `uv run ...`. Never invoke a bare `python` — it misses
  the project's pinned env.
- Notebooks: `uv run jupyter lab` (or VS Code with the uv-managed kernel).
- Notebook mirrors: we use `jupytext` so notebooks diff cleanly in git.

### Experiment discipline

- Every experiment gets its own `experiments/EXXX_slug.md` — see
  `experiments/README.md` for the workflow and `experiments/_template.md`
  for the template.
- Write the hypothesis **before** running. Fill in the result, interpretation,
  and next step after.
- Append a one-line summary to `experiments/index.md`.

### Commits

- **Semantic, one-line commit messages.** Imperative mood, lowercase prefix,
  scope optional: `feat: ...`, `fix: ...`, `docs: ...`, `chore: ...`,
  `refactor: ...`, `test: ...`, `exp: ...`.
- **No co-author trailer.** Never add `Co-authored-by: Claude` or similar.
- Commit the minimal change that makes sense — don't bundle unrelated changes.
- Never commit data (`data/`) or heavy artifacts (see `.gitignore`).

### Before submission

- Systems must run from a clean clone with `uv sync && uv run <entrypoint>`.
- Result filenames are self-describing (e.g. `audio_gmm.txt`).
- `dokumentace.pdf` explains *why* each design choice was made, not just what.
- ZIP excludes `data/` and any eval data.

---

## 7. What to build next (living TODO)

Keep this list short. Rework as reality changes.

- [x] Project skeleton + deps (`uv add` done: scikit-learn, scipy, librosa,
      matplotlib, tqdm, pandas, jupytext).
- [x] Data exploration notebook.
- [ ] `src/data/splits.py` — single source of truth for fold assignment.
- [ ] `src/eval/metrics.py` — EER, min-DCF, OOF collection helpers.
- [ ] Audio baseline (MFCC + GMM) → first experiment.
- [ ] Image baseline (PCA + logreg) → first experiment.
- [ ] Augmentation ablations per modality.
- [ ] Fusion (score-level first, then feature-level).
- [ ] Calibration + threshold selection on OOF.
- [ ] Draft `dokumentace.pdf` section by section as experiments land.
