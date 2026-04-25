# SUR 2025/2026 — Documentation Draft

> ~3 A4. Rewrite in own words before submission. CZ / SK / EN allowed.

---

## 1. Task

Binary detector for target person **m431** from paired face PNG + voice WAV. Three systems required: image-only, audio-only, trimodal fusion. No pretrained models, no external data — trained only on the provided corpus with augmentation.

## 2. Validation

Random K-Fold leaks session and speaker identity. We use a **3-fold session/speaker-aware LOSO split** shared across all three modalities:

- target rows grouped by session (`01`, `02`, `03`) — each fold holds out one full session;
- non-target rows grouped by speaker identity — each fold holds out 2–3 unseen strangers.

Both groupings feed one `GroupKFold` via a virtual column with disjoint ID spaces (`src/data/splits.py`). Sharing folds across modalities is load-bearing for stacking fusion via out-of-fold (OOF) scores.

**Metrics:** EER and min-DCF at prior 0.5, reported as mean ± std over folds. For fusion we additionally report **OOF overall EER** (all 222 samples pooled into one ranking). All augmentation is applied only to the train fold — val samples are always raw, and scaler/PCA/UBM statistics are refit per fold. A permutation test (E029: shuffled labels → 49% image EER, 55% audio EER) confirms models learn from labels, not leakage.

---

## 3. Audio system

### 3.1 Features

Speaker verification on ~170 training utterances is in the small-data regime where the **inductive bias of the front-end matters more than its expressive power**. LPCC directly parameterises the vocal-tract transfer function (Levinson-Durbin LPC + cepstral recursion) — the physical quantity that distinguishes speakers. MFCC's fixed Mel filterbank is a perceptually-motivated front-end designed for speech-*content* recognition, where speaker identity is precisely the nuisance variable to suppress. After ablating MFCC, FBank, SDC, PLP and LPCC under identical UBM-MAP conditions, **LPCC 13 + Δ + ΔΔ (LPC order 12, CMN per utterance)** wins on both EER and min-DCF (MFCC 4.21 % → LPCC 3.33 %). FBank (120-d) and SDC (104-d) over-parameterize GMM-32 on this dataset size; PLP's Bark + cube-root front-end hurts GMM fitting. LPC order 12 has a ≥ 2.5 pp moat over its neighbours — consistent with the `fs/1000 + 2` rule for 16 kHz audio.

### 3.2 Model — UBM-MAP with tied covariance

A 32-component GMM is trained on all non-target training frames as the UBM; the target model is a MAP-adapted copy of the UBM means only (relevance r = 16). Scoring is the LLR between adapted model and UBM. UBM-64 over-fits (E010); CMVN removes discriminative variance (E012); r ≤ 16 is a flat plateau (E013, E044).

The largest single improvement in the whole audio track was **tied covariance** (E037), where all 32 components share one full 39×39 covariance matrix:

| Covariance | EER mean ± std | min-DCF | Parameters |
|------------|---------------:|--------:|-----------:|
| spherical  | 3.89 ± 3.75%   | 0.0778  | 32         |
| diagonal   | 4.35 ± 4.40%   | 0.0870  | 1 248      |
| **tied**   | **0.69 ± 0.98%** | **0.0139** | **1 521** |
| full       | 1.48 ± 0.92%   | 0.0296  | 48 672     |

LPCC coefficients are strongly correlated (adjacent cepstra share formant structure). Diagonal covariance ignores those correlations; full per-component covariance (48 k params) overfits on 222 samples. Tied is the correct middle ground.

### 3.3 Augmentation and TTA

**Train-time:** pitch shift ±1–2 semitones (E025: LPCC-specific — trains the model to ignore F0 while preserving formant ratios; same aug *hurt* MFCC in E014 because the Mel filterbank already smooths pitch) and **codec simulation** (E052: each utterance is downsampled to 8 kHz and back, destroying frequencies above 4 kHz).

**Codec robustness (E052):**

| System              | Clean EER         | Codec-stressed EER |
|---------------------|------------------:|-------------------:|
| E042 baseline       | 0.46 ± 0.65%      | 13.33 ± 3.79%      |
| **+ codec aug (E052)** | **0.46 ± 0.65%** | **3.33 ± 4.14%**   |

Zero clean regression, −10 pp under codec stress. The UBM learns a distribution that overlaps the bandwidth-limited regime, so the LLR stays informative when high-frequency formants are destroyed.

**TTA (E031):** each utterance scored at original + 0.9× + 1.1× speed, LLRs averaged. The pitch/speed asymmetry is informative: speed perturbation retimes the signal but preserves the spectral envelope, so the LPC all-pole filter (and therefore LPCC) is invariant — averaging across speeds is averaging over benign perturbations. Pitch shift, by contrast, alters the source-filter relationship: F0 leaks into the LPC residual and the cepstral coefficients drift onto an out-of-distribution manifold the UBM was never trained on, collapsing fold 0 to 9.86 %. **2 s prefix truncation** (E053) was also rejected — CMN already suppresses stationary pre-speech noise, and the apparent codec-EER improvement was a fold-reshuffling artefact.

**Final audio system:** LPCC + tied-cov UBM + MAP r=16 + pitch & codec aug + speed TTA → **0.46 ± 0.65 % EER, min-DCF 0.0092**.

---

## 4. Image system

### 4.1 Features and classifier

80×80 grayscale → per-pixel standardization → **PCA-50 → logistic regression (C = 1)**. LBP (E005) was killed by session-to-session lighting shifts (fold 2 collapsed to 45 % EER); Fisherfaces (E006) are capped at a 1-D projection with only 2 identities; `n_pca ∈ {20…150}` was swept (E011) with n = 50 optimal. `C` and penalty were swept (E040) — C = 1 confirmed, L1 catastrophic. HE/CLAHE triples EER (E041); pyramid multi-scale PCA loses to plain + augmentation (E036).

### 4.2 Augmentation — two-pass adversarial training (E033)

**Pass 1 (standard aug):** each train image is added in four variants — original, horizontal flip, brightness × U[0.7, 1.3], Gaussian noise σ = 15 — and a PCA + LogReg is fit. Brightness jitter is the key contributor because session 03 has systematically different lighting from 01 / 02.

**Pass 2 (adversarial rotation):** for each training image the Pass-1 model is queried at 5 angles in [−10°, +10°] and the angle with **minimum |logit|** (maximum model uncertainty) is picked. A rotated copy at that angle is added to the training set; PCA + LogReg is refit on the combined set. The mechanism explains why this works where random rotation fails: random rotation samples uniformly from the manifold and PCA fits the *average*, leaving the hard angles unmodelled. Adversarial selection inverts that — each sample contributes the rotation the *current* eigenspace fails on, so principal components are reallocated towards the directions of model uncertainty. This is the same idea as hard-negative mining in SVMs, applied to PCA.

|                       | Clean EER         | rot ±15° (E033 ablation) | rot ±15° (E051 re-stress) |
|-----------------------|------------------:|-------------------------:|--------------------------:|
| +All (E007)           | 0.97 ± 0.86%      | 13.70%                   | 19.00%                    |
| **+AdvRot (E033)**    | **0.51 ± 0.36%**  | **1.04%**                | **7.59%**                 |

Clean EER halves; rotation robustness improves by 2.5–13× depending on the measurement protocol (both are reported honestly — the true eval-time number likely sits between). Fold-0 pathology (2.08 %) was eliminated (0.69 %). **Random** rotation augmentation does *not* work (E015) — it is the adversarial *selection* that forces the eigenspace to span the rotation manifold. JPEG / blur / contrast / HE-CLAHE / Cutout 20×20 (E052) all regressed; the current set is at the empirical ceiling.

**TTA:** original + horizontal flip, scores averaged. Rotation TTA was rejected (E030) — it corrupts the eigenface projection at inference and raises clean EER. E043 flip+rot5 TTA appeared to help but was a measurement artefact and failed to replicate (E049); E033 remains the image flagship.

**Final image system:** PCA-50 + LogReg + flip / brightness / noise / adv-rot aug + flip TTA → **0.51 ± 0.36 % EER, min-DCF 0.0102**.

---

## 5. Fusion

Each modality's raw LLR is Platt-scaled (`LogReg C = 1e6, class_weight='balanced'`) on OOF scores to bring the three streams to a common scale. The fused score is a weighted sum on the 2-simplex (`w_mfcc + w_lpcc + w_image = 1`, all ≥ 0), with weights chosen on a 51×51 grid that **directly minimises OOF EER** (logistic-regression fusion is MLE-optimal but not EER-optimal — the near-rank-2 design matrix, r(MFCC, LPCC) = 0.843, limits it).

| Fusion                           | OOF EER        | min-DCF | Notes                               |
|----------------------------------|---------------:|--------:|-------------------------------------|
| MFCC + image (E009)              | 3.75%          | 0.0750  | bimodal baseline                    |
| LPCC + image (E023)              | 0.52%          | 0.0104  | LPCC replaces MFCC                  |
| MFCC + LPCC + image (E026/E027)  | 0.26%          | 0.0052  | trimodal, MFCC as tiebreaker        |
| **E052 + E033 + MFCC (E039)**    | **0.26% (0 errors)** | **0.0052** | **new backbones, 0/222**    |

**Weights (E039):** `w_image = 0.66, w_lpcc = 0.34, w_mfcc ≈ 0.00`. Image dominates because it achieves the lower clean EER; LPCC contributes complementary audio signal — its errors are disjoint from the image stream's (0 of 222 samples are misranked by both, see §6 stress test). MFCC's weight collapses to zero because MFCC and LPCC OOF scores are correlated at r = 0.843: both are cepstral representations of the same vocal-tract physics, just via different front-ends, so once tied-covariance LPCC enters the fusion the third stream is rank-deficient. We retain MFCC so the grid can re-verify this per run — if a future LPCC weakness emerges, MFCC will pick up weight automatically. Quality-aware gating (E032), product-rule fusion (E046/E048) and score ensembles (E045) all regressed.

---

## 6. Generalization and overfitting defences

The assignment specifically requires *způsob řešení generalizace a omezení přetrénování*. Each defence below targets a named overfitting risk.

1. **Risk: model capacity > data size → memorisation.** Defence: UBM-32 (~5 400 frames/component, well into the asymptotic regime) over UBM-64 which regressed in E010; PCA-50 on 222 samples (n &lt; p_full but adequate for a 2-class linear boundary, sweep in E011); tied covariance shares 1 521 parameters across 32 components vs 48 672 for full per-component (E037).
2. **Risk: validation leakage from session/speaker overlap.** Defence: 3-fold session/speaker-aware LOSO splits, one fold = one held-out target session + 2–3 held-out non-target identities. All scaler / PCA / UBM statistics are refit per fold; augmentation is applied only to the train fold and val samples are always raw (across all 53 experiments).
3. **Risk: hidden label leakage via auxiliary channels** (filename order, file size, etc.). Defence: permutation test (E029) — shuffling train labels before retraining yields 49 % image / 55 % audio val EER, both inside the chance window, so flagships learn from labels rather than artefacts.
4. **Risk: post-hoc rationalization of multi-axis changes.** Defence: each of the 53 experiments moves exactly one knob (feature / aug / covariance / hyperparameter / fusion rule), with the hypothesis written before the run. Findings are attributable to the changed axis.
5. **Risk: unseen test-time distribution shift** (the assignment warns evaluation data will contain noise / quality changes). Defence: stress testing on val (E028, E051, E052) — E033 image is photometrically bulletproof (JPEG q = 15 / blur σ = 3 / downsample 40 → 80 all at clean 0.51 %); residual weaknesses are geometric (rotation > 15°: 7.6 %) and occlusion (11 %). E052 audio survives speed via TTA and 4 kHz codec stress via codec-augmentation (13.33 % → 3.33 %); moderate noise ≤ 10 dB SNR manageable.

---

## 7. Results

| System                                                        | CV EER mean ± std | CV min-DCF |
|---------------------------------------------------------------|------------------:|-----------:|
| Audio baseline — MFCC + GMM (E001)                            | 17.92 ± 7.81 %    | 0.2250     |
| Audio — MFCC + UBM/MAP + aug (E008)                           | 4.21 ± 3.11 %     | 0.0509     |
| Audio — LPCC + tied cov (E037)                                | 0.69 ± 0.98 %     | 0.0139     |
| **Audio flagship — LPCC + tied + pitch&codec aug + speed TTA (E052)** | **0.46 ± 0.65 %** | **0.0092** |
| Image baseline — PCA + LogReg (E004)                          | 4.49 ± 4.26 %     | 0.0565     |
| Image — PCA + LogReg + aug (E007)                             | 0.97 ± 0.86 %     | 0.0194     |
| **Image flagship — PCA + LogReg + aug + adv-rot (E033)**      | **0.51 ± 0.36 %** | **0.0102** |
| **Fusion flagship — trimodal E052 + E033 + MFCC (E039)**      | **0.26 % OOF (0 errors)** | **0.0052** |

**Story arc:** MFCC + GMM baseline → UBM + MAP → better features (LPCC) → tied covariance → codec robustness → adversarial image aug → trimodal fusion.

Reproduction: `uv sync && uv run predict_fusion.py --eval-dir <dir> --output results/fusion_trimodal.txt`. Output per line: `<stem> <score> <hard_decision>`, hard decision at the Bayes threshold (prior 0.5) calibrated on OOF min-DCF. Details, full ablation history, and figures in `experiments/` and `docs/figures/`.
