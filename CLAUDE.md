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
| E028 | image | stress test (jpeg/blur/rotate/occlude/all) at val | 0.97→1.25/1.71/7.31/18.10/26.16% | robust to photometric, brittle to geometric |
| E029 | validity | permutation test (shuffle train labels) | image 49.49%, audio 55.26% | both pass — no hidden leakage |
| E030 | image | TTA rotation at inference (avg log-odds over −20°/−10°/0°/+10°/+20°) | clean: 1.25% (↑0.28pp vs E007) | ❌ TTA hurts clean, worsens all-combined; only rot±25° improves; geometric brittleness is fundamental to PCA |
| E031 | audio | LPCC+Pitch val-time TTA: baseline/+pitch_tta(5 views)/+speed_tta(3 views)/+pitch_speed_tta(7 views) | **1.67 ± 1.80 %** (+speed_tta wins) | ✓ speed TTA −0.27pp EER −0.56pp min-DCF; +pitch_tta collapses fold0 to 9.86% (formant corruption); adopted in predict_audio.py + predict_fusion.py |
| E041 | image | E033 + HE/CLAHE preprocessing | 0.97 ± 0.86 % (raw wins) | ❌ HE/CLAHE triples EER (3.01%); raw pixels optimal |
| E042 | audio | E037 tied cov + speed TTA (3 views) | **0.46 ± 0.65 %** | ✓ new audio flagship; −0.23pp vs E037; fold 0: 2.08→1.39% |
| E043 | image | E033 + TTA (flip+rot5, 5 views) | 0.74 ± 0.57 % | ⚠️ INVALID FLAGSHIP — compared vs broken E033 replication (0.97% instead of 0.51%); E049 failed to replicate; E033 (0.51%) is still better |
| E044 | audio | E042 tied cov + MAP r sweep (4–64) | 0.46 % (r=16) | ↔ r=16 confirmed on tied cov |
| E045 | audio | MFCC+LPCC+PLP score ensemble | 3.23 % OOF | ❌ ensemble hurts; LPCC alone (2.45% OOF) wins; calibration asymmetry |
| E046 | fusion | E042+E043 backbones + product rule (geometric mean) | 0.52 % OOF | ❌ NOT a fusion flagship — compared vs broken E039 replication (2.97% instead of 0.26%); E039 (0.26%, 0 errors) is better |
| E047 | audio | E042 + VTLN warping α∈[0.9,1.1] | 31.45 % (catastrophic) | ❌ broken implementation — warped cepstral coeffs directly instead of mel filters |
| E048 | fusion | E042+E043 bimodal product rule | 4.59 % OOF | ❌ bimodal product rule + missing calibration fails; E046 trimodal was already inferior to E039 |
| E049 | image | E043 + more TTA views (rot7/9/bright/noise) | 4.38 % (replication fails) | ❌ E043 baseline not replicable (0.74→4.38%); confirms E043 result is fragile/invalid |
| E051 | image+audio | Stress test: E033 vs E007 (8 conditions) + E042 audio (8 conditions) | image: jpeg/blur/downsample=0.51%; rot15=7.59%; occlude=11.06%; audio: codec=13.33%, noise10=6.85%, speed=0.74% | — | E033 photometrically bulletproof; codec is audio's main vulnerability |
| E052 | image+audio | codec bandwidth aug (audio) + Cutout 20×20 (image) | audio clean=0.46%, codec=3.33% (was 13.33%); image clean=1.71% (regresses from 0.51%) | — | Audio ADOPT (−10pp codec, zero clean cost); Image REJECT (+1.20pp clean regression) |

### EER: per-fold mean vs OOF overall (important distinction)

Per-fold mean and OOF overall EER are different numbers and mean different things:

| Flagship | Per-fold mean | OOF overall |
|----------|--------------|-------------|
| Image E033 adv-rot | 0.51 ± 0.36% | ~0% |
| Audio E042 tied+speedTTA | 0.46 ± 0.65% | — |
| Fusion E039 trimodal (E037+E033+MFCC) | — | **0.26% (0 errors)** |

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
- E028 stress test: image robust to photometric degradations (JPEG q=15: 1.25%, blur σ=3: 1.71%) but brittle to geometric (rotation ±15°: 7.3%, occlusion: 18%). If Burget uses photometric-only, 0.97% holds.
- E029 permutation test: PASS. Shuffling train labels → permuted EER 49.5% (image) and 55.3% (audio). No hidden leak — models learn from labels, not auxiliary channels.
- E030 TTA rotation ❌: averaging log-odds over ±20°/±10°/0° rotations hurts clean EER (0.97→1.25%) and all-combined stress (+4.31pp). Geometric brittleness is fundamental to PCA eigenfaces; no inference-time fix exists without retraining. Flip-only TTA in predict_image.py stays.
- E031 audio val-time TTA ✓: +speed_tta (original + 0.9× + 1.1×, 3 views) improves LPCC EER 1.94→1.67%, min-DCF 0.0389→0.0333. +pitch_tta collapses fold 0 to 9.86% — pitch shift corrupts LPCC formant coefficients more than UBM score. Speed TTA adopted in predict_audio.py and predict_fusion.py.
- E032 quality-aware fusion ❌: tested softmax/threshold/linear weighting based on image sharpness and audio SNR. All strategies achieve 0.78% OOF EER (matches E027 fixed weights). Quality metrics don't correlate with recognition difficulty on clean data. Not adopted.
- E033 adversarial image aug ✓✓: +AdvRot (adversarial rotation ±10°) achieves 0.51% EER vs E007's 0.97% — 2× improvement! Rotation robustness: rot15 EER 13.70% → 1.04% (13× better). New image flagship.
- E034 Z-norm ❌: cohort-based score normalization, same EER (6.20%), better min-DCF. Dataset too small for effective cohort estimation. Not adopted.
- E035 feature-level fusion ❌: MFCC+LPCC concatenation failed (dimension mismatch bug). Needs fix.
- E036 multi-res PCA ❌: pyramid features (80/40/20) achieve 1.53% EER, worse than E007 +All (0.97%). Multi-scale not better than good augmentation. Not adopted.
- E037 GMM covariance ✓✓✓: tied covariance achieves 0.69% EER vs diagonal's 4.35% — 6.3× improvement! Best audio result ever, beats E025+speedTTA (1.67%). Fold 0 pathology solved (2.08% vs 9.17%). New audio flagship.
- E038 ensemble UBM ❌: timed out, too slow for marginal gain. Tied cov already solves fold 0. Not worth 3-5× inference cost.
- E039 fusion new backbones ✓✓✓: E037+E033+MFCC achieves 0.26% OOF with **0 errors** (vs E027's ~1). Weights: img=0.66, lpcc=0.34, mfcc=0.00. MFCC redundant. New fusion flagship.
- E040 LR regularization ↔: C=0.1 ties C=1.0 (0.97%), C>10 fails catastrophically. C=1.0 confirmed optimal.
- E041 HE/CLAHE ❌: triples image EER (3.01% vs 0.97%); raw pixels + brightness aug optimal.
- E042 speed TTA + tied cov ✓: 0.46 ± 0.65% EER (−0.23pp vs E037). Speed TTA complements tied covariance. New audio flagship. predict_audio.py updated.
- E043 image TTA flip+rot5 ⚠️ INVALID: claimed 0.74% but compared against wrong E033 baseline (0.97% instead of 0.51%); E049 later failed to replicate it entirely (4.38%). E033 remains image flagship at 0.51%.
- E044 MAP r on tied cov ↔: r=16 confirmed, same as diagonal.
- E045 MFCC+LPCC+PLP score ensemble ❌: 3.23% OOF vs LPCC alone 2.45%; calibration asymmetry.
- E046 product rule fusion ❌ MISLABELED AS FLAGSHIP: 0.52% OOF — worse than E039's 0.26%. Qwen compared product rule vs a broken E039 replication (2.97% instead of 0.26%). E039 remains fusion flagship.
- E047 VTLN ❌: catastrophic (31.45%) due to wrong implementation (warped cepstral coeffs, not mel filters).
- E048 bimodal product rule ❌: 4.59% OOF; worse than trimodal and worse than E039.
- E049 E043 replication fails ❌: 4.38% vs claimed 0.74% — confirms E043 result is not reproducible.
- E051 stress test ✓: E033 image photometrically bulletproof (jpeg/blur/downsample all = 0.51% = clean); rotation much better (rot15: 19→7.59%); occlusion unchanged (+0.93pp, expected). Audio E042: speed stress absorbed by TTA (slow 0.74%, fast 0.23%); **codec (8kHz BW) catastrophic at 13.33%** — LPCC formants above 4kHz destroyed; moderate noise (20dB: 4.35%, 10dB: 6.85%) manageable.
- E052 codec aug (audio): ADOPT. Clean EER unchanged (0.46%), codec EER 13.33%→3.33% (−10pp, −75%). predict_audio.py and predict_fusion.py updated.
- E052 Cutout aug (image): REJECT. Clean EER regresses 0.51%→1.71% (+1.20pp); PCA eigenspace destroyed by masking large patches.
- **52 experiments complete. Current flagships: audio=E052 (0.46% clean / 3.33% codec), image=E033 (0.51%), fusion=E039 (0.26% OOF, 0 errors). Scripts updated, self-test should be re-run.**

---

## 8. Submission strategy (6 result files)

Burget scores up to 6 files. Use the slots to show ablation progression:

| File | Experiment | Purpose |
|------|-----------|---------|
| `audio_mfcc_gmm_baseline.txt` | E001 | Audio anchor (lecture baseline), 17.92% |
| `audio_mfcc_ubm_map_aug.txt` | E008 | MFCC+aug, 4.21%, shows UBM+MAP+aug contribution |
| `audio_lpcc_tied_speedtta.txt` | E042 | **Audio flagship**, 0.46%, LPCC+tied cov+speedTTA |
| `image_pca_baseline.txt` | E004 | Image anchor (no aug), 4.49% |
| `image_pca_adv_rot.txt` | E033 | **Image flagship**, 0.51%, PCA+LogReg+AdvRot aug |
| `fusion_trimodal.txt` | E039 | **Fusion flagship**, 0.26% OOF (0 errors), E037+E033+MFCC |

Story: baseline → UBM+MAP → features (MFCC→LPCC) → tied covariance → adversarial aug → trimodal fusion.

If E052 improves flagships, update file names accordingly before generating results.

---

## 9. What to build next (living TODO)

- [x] Project skeleton + deps
- [x] Data exploration notebook
- [x] `src/data/splits.py` — LOSO splitter, group-aware
- [x] `src/eval/metrics.py` — EER, min-DCF, hard decisions
- [x] Audio E001–E049 (flagship=E042 LPCC+tied cov+speedTTA at 0.46%)
- [x] Image E004–E049 (flagship=E033 PCA+LogReg+AdvRot at 0.51%)
- [x] Score calibration (Platt + class_weight='balanced' on OOF)
- [x] Fusion E009/E023/E026/E027/E039 (flagship=E039 trimodal at 0.26% OOF EER, 0 errors)
- [x] Update `predict_audio.py` to use tied covariance UBM + speed TTA (E042)
- [x] Update `predict_image.py` to use adversarial rotation aug (E033)
- [x] Update `predict_fusion.py` to use E039 weights (img=0.66, lpcc=0.34, mfcc=0.00) with E042/E033 backbones
- [x] Self-test mini-eval set (`self_test.py`) — **PASSED** (all 3 scripts 10/10, correct ordering)
- [x] E051 stress test — confirmed E033 photometric robustness; identified codec as audio vulnerability
- [x] E052: codec aug adopted (audio only) — predict_audio.py + predict_fusion.py updated
- [ ] Re-run self_test.py to verify E052 changes don't break scoring pipeline
- [ ] Generate the 6 result files on eval data (2026-05-03 morning)
- [ ] `dokumentace.pdf` — explain WHY for every design choice, ablation tables required

### 52 experiments complete. Flagships: audio=E052 (0.46% clean / 3.33% codec-stressed), image=E033 (0.51%), fusion=E039 (0.26% OOF, 0 errors). Scripts up to date. Next: re-run self_test, then eval day (2026-05-03).
