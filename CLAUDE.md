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

## 7. Empirical findings

### Experiment progression

| Exp | Modality | Model | CV EER mean±std | Notes |
|-----|----------|-------|-----------------|-------|
| E001 | audio | MFCC 13 + GMM | 17.92 ± 7.81 % | anchor |
| E002 | audio | MFCC 13+Δ+ΔΔ + GMM | 10.09 ± 1.81 % | deltas: −7.83% |
| E003 | audio | UBM 32 + MAP r=16, MFCC+Δ+ΔΔ | 7.45 ± 5.04 % | |
| E004 | image | PCA 50 + LogReg | 4.49 ± 4.26 % | anchor |
| E005 | image | LBP 4×4 + LogReg | 17.78 ± 23.58 % | ❌ fold 2 collapse |
| E006 | image | PCA 100 + LDA shrinkage=auto | 18.24 ± 1.53 % | ❌ 1D bottleneck |
| E007 | image | PCA 50 + LogReg + aug (+All) | **0.97 ± 0.86 %** | ← image flagship |
| E008 | audio | UBM+MAP + aug (+All) | **4.21 ± 3.11 %** | ← audio flagship |
| E009 | fusion | Platt calib + MFCC+image grid w=0.28 | 3.75 % OOF overall | superseded by E023 |
| E010 | audio | UBM 64 + MAP + aug | 6.39 ± 3.93 % | ❌ over-param; 32 confirmed |
| E011 | image | PCA sweep n∈{20…150} + aug | 0.97 ± 0.86 % (n=50 wins) | n=50 confirmed optimal |
| E012 | audio | UBM+MAP + CMVN + aug | 6.16 ± 3.78 % | ❌ CMVN removes discriminative variance |
| E013 | audio | MAP r∈{4,8,16,32,64} + aug | 4.21 ± 3.11 % (r≤16 plateau) | r=16 confirmed |
| E014 | audio | +All + codec/lownoise/pitch/clip | 4.21 % (+All still wins) | ❌ all 4 new augs hurt |
| E015 | image | +All + jpeg/blur/rotate/contrast | 0.97 % (+All still wins) | ❌ all 4 new augs hurt |
| E016 | audio | FBank 40+Δ+ΔΔ (120d) + UBM+MAP | 9.95 ± 1.36 % | ❌ DCT compression is beneficial regularization for GMM |
| E017 | audio | GMM supervector (1248d) + LinearSVM | 9.07 ± 7.45 % | ❌ n_features >> n_samples; LLR inductive bias wins |
| E018 | audio | VTLP α∈[0.9,1.1] (replace speed) | 3.94 ± 3.28 % | ↔ marginal mean gain, min-DCF regresses |
| E019 | audio | SDC N=7,d=1,P=3 (104d) + UBM+MAP | 12.96 ± 4.79 % | ❌ too high-dim for GMM-32 on small data |
| E020 | audio | LPCC 13+Δ+ΔΔ + UBM+MAP | 3.33 ± 4.14 % | ✓ best audio mean EER + min-DCF → new audio flagship |
| E021 | audio | PLP 13+Δ+ΔΔ (Bark+EL+cbrt) + UBM+MAP | 5.56 ± 2.58 % | ❌ Bark front-end too lossy; cube-root hurts GMM fitting |
| E022 | audio | MFCC+LPCC score fusion | 3.33 ± 4.14 % | ↔ collapsed to LPCC alone (w=0.07); calibration asymmetry |
| E023 | fusion | Platt calib + LPCC+image grid w=0.36 | 0.52 % OOF overall | superseded by E026/E027 |
| E024 | audio | LPC order ablation ∈{8,10,12,14,16,20} | 3.33% @ order=12 | order=12 confirmed (≥2.5pp moat) |
| E025 | audio | LPCC+Pitch augmentation | **1.94 ± 1.57 %** | ← new audio flagship; pitch uniquely helps LPCC (not MFCC) |
| E026 | fusion | MFCC+LPCC+image grid (trimodal) | **0.26 % OOF overall** | halves E023 via residual MFCC complementarity |
| E027 | fusion | MFCC+LPCC(+Pitch)+image grid | **0.26 % OOF overall** | ← fusion flagship; matches E026 but more robust audio backbone |

### EER: per-fold mean vs OOF overall (important distinction)

Per-fold mean and OOF overall EER are different numbers and mean different things:

| System | Per-fold mean | OOF overall |
|--------|--------------|-------------|
| Image E007 | 0.97% | ~10% |
| Audio E008 | 4.21% | ~9% |
| Fusion E009 | — | 3.75% |

**Per-fold mean** = average of 3 separate evaluations, each on its own session context.
**OOF overall** = all 222 samples evaluated at once, mixing 3 different fold models.
The production script retrains on all data (a 4th model) — eval performance will be
somewhere between these. Report both; do not cherry-pick the better number.

### Key findings
- Deltas (+Δ+ΔΔ): −7.83% EER, variance collapse — always use them for audio.
- Image beats audio in CV: PCA captures Ondra's global appearance well.
- **Risk**: 0.97% image EER may be optimistic — eval has new sessions + Burget's degradations.
- LBP ❌: session-to-session appearance shift killed texture features.
- LDA ❌: 1D bottleneck with 20 target samples loses to logreg in 50D.
- Brightness jitter was the key image augmentation (session lighting is the main variation).
- Speed perturbation was the key audio augmentation (speaking rate varies across sessions).
- Fusion (w=0.28 audio, w=0.72 image) beats both modalities — complementary signal.
- Calibration: image threshold after Platt = −0.07 (good), audio = −0.41 (imperfect).
- UBM 64 ❌: over-parameterized for dataset size (~2700 frames/component vs ~5400 for 32).
- CMVN ❌: UBM diagonal covariance already learns scale; per-utterance std is noisy on short clips.
- MAP r∈{4,8,16} plateau: r=16 is confirmed robust, not arbitrary.
- More aggressive audio augs ❌ (codec/pitch/clip/10dB noise): all hurt — E008 +All is optimal.
- More aggressive image augs ❌ (jpeg/blur/rotate/contrast): all hurt — E007 +All is optimal.
- FBank/SDC/supervector ❌: higher-dim features over-parameterize GMM-32 on our dataset. DCT compression and Δ+ΔΔ are beneficial regularizers, not limitations.
- VTLP ↔: marginal mean gain but min-DCF regresses — E008 stays.
- LPCC ✓: 3.33 ± 4.14% mean EER, min-DCF 0.0333 — best on both metrics. Fold errors complementary to E008 (fold0 LPCC fails, E008 wins; fold1 LPCC wins, E008 fails).
- PLP ❌: Bark front-end too coarse for 20-band LPC; cube-root compression hurts GMM fitting.
- LPCC+MFCC audio fusion (E022) ↔: global Platt calibration gave LPCC 2× dynamic range → grid search converged to w=0.07 (essentially LPCC alone). Fusion = LPCC alone result.
- E023 LPCC+image fusion: 0.52% OOF EER (−3.23pp vs E009). Superseded by trimodal.
- E024: LPC order=12 confirmed (≥2.5pp moat over neighbors).
- E025: **+Pitch augmentation uniquely helps LPCC** (hurt MFCC in E014). Because LPCC encodes formants directly, pitch shift is the right inductive bias. Std collapses 4.14→1.57.
- E026/E027: trimodal fusion MFCC + LPCC + image halves E023 OOF EER: **0.26%**.
- **All experiments complete: 27 total. Flagships: audio=E025 LPCC+Pitch, image=E007, fusion=E027 trimodal.**

---

## 8. Submission strategy (6 result files)

Burget scores up to 6 files. Use the slots to show ablation progression:

| File | Experiment | Purpose |
|------|-----------|---------|
| `audio_gmm_baseline.txt` | E001 | Audio anchor (lecture baseline) |
| `audio_ubm_map_aug.txt` | E008 | Audio flagship |
| `image_pca_logreg_baseline.txt` | E004 | Image anchor (no aug) |
| `image_pca_logreg_aug.txt` | E007 | Image flagship |
| `fusion_score.txt` | E009 | Fusion flagship |
| `audio_ubm_map_noaug.txt` | E003 | Audio ablation (shows aug contribution) |

This tells the story: baseline → UBM+MAP → augmentation → fusion.

---

## 9. What to build next (living TODO)

- [x] Project skeleton + deps
- [x] Data exploration notebook
- [x] `src/data/splits.py` — LOSO splitter, group-aware
- [x] `src/eval/metrics.py` — EER, min-DCF, hard decisions
- [x] Audio E001–E008 (UBM+MAP+aug, flagship=E008)
- [x] Image E004–E007 (PCA+logreg+aug, flagship=E007)
- [x] Score calibration (Platt + class_weight='balanced' on OOF)
- [x] E009 score-level fusion (grid search w=0.28)
- [x] Production scripts (`predict_audio.py`, `predict_image.py`, `predict_fusion.py`)
- [ ] Self-test mini-eval set — **mandatory before submission** (Burget: "people send files with scores scrambled")
- [ ] Generate the 6 result files on eval data (2026-05-03 morning)
- [ ] `dokumentace.pdf` — explain WHY for every design choice, ablation tables required

### ✅ 27 experiments complete. Flagships locked: audio=LPCC+Pitch (E025, 1.94% per-fold mean), image=PCA+aug (E007, 0.97%), fusion=trimodal (E027, 0.26% OOF EER).
