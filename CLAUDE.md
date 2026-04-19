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

## 7. Empirical findings (updated as experiments land)

| Exp | Modality | Model | EER mean±std |
|-----|----------|-------|--------------|
| E001 | audio | MFCC 13 + GMM | 17.92 ± 7.81 % |
| E002 | audio | MFCC 13+Δ+ΔΔ + GMM | 10.09 ± 1.81 % |
| E003 | audio | UBM 32 + MAP r=16, MFCC+Δ+ΔΔ | **7.45 ± 5.04 %** ← audio flagship |
| E004 | image | PCA 50 + LogReg | **4.49 ± 4.26 %** ← image flagship |
| E005 | image | LBP 4×4 + LogReg | 17.78 ± 23.58 % ❌ fold 2 collapse |
| E006 | image | PCA 100 + LDA shrinkage=auto | 18.24 ± 1.53 % ❌ 1D bottleneck |
| E007 | image | PCA 50 + LogReg + aug (flip+brightness+noise) | **0.97 ± 0.86 %** ← image flagship |
| E008 | audio | UBM+MAP + aug (noise+speed) | **4.21 ± 3.11 %** ← audio flagship |

**Key findings so far:**
- Deltas (+Δ+ΔΔ) gave −7.83% EER on audio and cut variance dramatically — always use them.
- Image beats audio: faces are very distinctive in PCA space for this person.
- LBP failed: session pose/lighting shift kills texture features. PCA global structure more stable.
- LDA collapses to 1D for binary — with 20 target samples this loses badly vs logreg in 50D.
- UBM+MAP threshold ≈ 0 (calibrated). Image logreg threshold ≈ −5 (needs calibration before fusion).
- Large fold std is expected with 3 LOSO folds — report it, never hide it.

---

## 8. What to build next (living TODO)

- [x] Project skeleton + deps
- [x] Data exploration notebook
- [x] `src/data/splits.py` — LOSO splitter, group-aware
- [x] `src/eval/metrics.py` — EER, min-DCF, hard decisions
- [x] Audio E001–E008 (UBM+MAP+aug, flagship=E008, EER 4.21%)
- [x] Image E004–E007 (PCA+logreg+aug, flagship=E007, EER 0.97%)
- [ ] Score calibration (Platt on OOF) — image threshold ≈ −5, needed before fusion
- [ ] E009: Score-level fusion (calibrated E008 audio + E007 image OOF → logreg)
- [ ] Production scripts: `predict_audio.py`, `predict_image.py`, `predict_fusion.py`
- [ ] Self-test mini-eval set (mandatory before submission)
- [ ] `dokumentace.pdf` — write section by section as experiments land
