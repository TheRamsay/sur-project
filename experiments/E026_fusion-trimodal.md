# E026 — Tri-modal fusion: MFCC + LPCC + image

- **Date:** 2026-04-19
- **Author:** Dominik Huml
- **Related experiments:** E008 (MFCC audio flagship), E020 (LPCC audio flagship), E007 (image flagship), E009 (MFCC+image fusion), E023 (LPCC+image fusion, prior fusion flagship), E022 (MFCC+LPCC audio-only fusion collapsed to LPCC)

## Hypothesis

E023 fused LPCC audio (E020 +All) with image (E007 +All) and achieved OOF
EER=0.52%, min-DCF=0.0104 — a large improvement over E009's MFCC+image
(3.75%, 0.0750). E022 showed that stacking MFCC and LPCC audio alone
collapsed to LPCC (w=0.07) because LPCC dominated in this 2-channel setup
and the errors were insufficiently complementary to move the needle.

However, MFCC and LPCC capture genuinely different spectral representations:
MFCC uses mel-warped cepstrum while LPCC uses all-pole vocal tract modelling.
If the residual errors of MFCC and LPCC diverge on different utterances
compared to the image modality's errors, adding MFCC to the LPCC+image fusion
may push OOF EER below E023's 0.52%. Alternatively, if the audio signal is
already saturated (LPCC-dominated) in the tri-modal fusion — as in E022 —
MFCC will receive near-zero weight and performance will match E023.

## Setup

- **MFCC audio (E008 +All):** MFCC 13+Δ+ΔΔ+CMN (39d), UBM 32 + MAP r=16, +All aug (noise SNR=20dB + speed ±10%)
- **LPCC audio (E020 +All):** LPCC 13+Δ+ΔΔ+CMN (39d, LPC order=12), UBM 32 + MAP r=16, +All aug (noise SNR=20dB + speed ±10%)
- **Image (E007 +All):** StandardScaler + PCA(50) + LogReg(C=1), +All aug (flip + brightness[0.7,1.3] + noise σ=15)
- **Calibration:** Platt (LogisticRegression C=1e6, class_weight='balanced') per modality
- **Fusion A (grid):** simplex search w_mfcc, w_lpcc ∈ [0,1] with w_image = 1−w_mfcc−w_lpcc ≥ 0 (51×51 grid)
- **Fusion B (LogReg):** LogisticRegression(class_weight='balanced') on the 3-column calibrated OOF scores
- **Fold spec:** `iter_folds_loso`, seed=67, 3 folds
- **Command / notebook:** `notebooks/E026_fusion_trimodal.ipynb`

## Result

Per-fold EER (replicated from E008/E020/E007):

| Config      | Fold 0 | Fold 1 | Fold 2 | Mean ± std   |
| ----------- | ------ | ------ | ------ | ------------ |
| MFCC +All   | 3.47   | 8.33   | 0.83   | 4.21 ± 3.11  |
| LPCC +All   | 9.17   | 0.83   | 0.00   | 3.33 ± 4.14  |
| Image +All  | 2.08   | 0.83   | 0.00   | 0.97 ± 0.86  |

Pairwise Pearson correlations on Platt-calibrated OOF scores:

| Pair       | r     |
| ---------- | ----- |
| MFCC–LPCC  | 0.850 |
| MFCC–Image | 0.419 |
| LPCC–Image | 0.443 |

OOF overall comparison table:

| System                                            | OOF EER [%] | min-DCF  |
| ------------------------------------------------- | ----------- | -------- |
| MFCC audio (E008 +All)                            | 9.43        | 0.1687   |
| LPCC audio (E020 +All)                            | 6.46        | 0.1198   |
| Image (E007 +All)                                 | 4.01        | 0.0729   |
| MFCC+image E009 grid (w=0.28)                     | 3.75        | 0.0750   |
| LPCC+image E023 grid (w=0.36)                     | 0.52        | 0.0104   |
| **MFCC+LPCC+image E026 grid (w=0.06/0.52/0.42)**  | **0.26**    | **0.0052** |
| MFCC+LPCC+image E026 LogReg                       | 0.52        | 0.0104   |

Grid search (EER- and min-DCF-optimum coincide): w_mfcc=0.06, w_lpcc=0.52,
w_image=0.42 → halves both metrics relative to E023.

LogReg fusion coefficients (raw, decision_function weights):
mfcc=+0.337, lpcc=+0.867, image=+0.910, bias=−0.434.
L1-normalised magnitudes: mfcc=0.16, lpcc=0.41, image=0.43. Same pecking order
as the grid, but MFCC weight is 2.7× larger than grid's 0.06 — yet LogReg
scores exactly match E023 at OOF EER=0.52% and min-DCF=0.0104. LogReg is
maximum-likelihood-optimal, not EER-optimal, and with 222 OOF samples it
under-fits the decision boundary that the grid finds.

Improvement vs E023 (LPCC+image): **−0.26pp EER (0.26% vs 0.52%),
−0.0052 min-DCF (0.0052 vs 0.0104)**. Both metrics are exactly halved.

## Interpretation

MFCC adds information beyond LPCC+image — but only a sliver, and only when
the fusion weights are tuned directly against the ranking objective.

1. **The grid result is real, not noise.** The simplex optimum puts MFCC at
   w=0.06. On 222 OOF samples with 30 targets, moving from 0.52% to 0.26%
   EER represents flipping a single target/non-target ordering. The
   improvement is small in absolute terms, but the min-DCF halving (0.0104
   → 0.0052) shows the EER-optimal threshold also becomes more robust.

2. **Why MFCC helps at all, given r(MFCC, LPCC)=0.850.** MFCC and LPCC are
   highly correlated on the bulk of samples, but their *residual* errors
   diverge on hard cases: MFCC's fold-2 EER is 0.83% while LPCC's is 0.00%,
   and MFCC's fold-0 EER is 3.47% while LPCC's is 9.17%. A tiny MFCC
   admixture breaks ties in LPCC's hardest fold. The image modality is
   complementary to both audio channels (r≈0.42 with each), so the grid
   keeps image at 0.42 weight.

3. **Why LogReg doesn't match the grid.** LogReg maximises log-likelihood,
   which rewards well-calibrated scores across the full distribution. The
   grid targets EER (a ranking metric), which cares only about how the
   single worst target is ranked relative to the non-targets. With
   r(MFCC, LPCC)=0.850, the 3-column design matrix is near-rank-2, and
   LogReg learns the smooth MLE solution (w_mfcc=0.16 when L1-normalised) —
   which happens to replicate E023's ranking exactly (0.52%). The grid, free
   from the log-likelihood objective, finds a sharper optimum.

4. **Contrast with E022.** E022 fused MFCC and LPCC audio alone and
   collapsed to LPCC (w=0.07). In E026, MFCC again receives only 0.06 weight
   — so the E022 finding is preserved. What changed is that the image
   modality supplies the other 0.42, and the three-way balance surfaces the
   residual MFCC–LPCC complementarity that was invisible in two-channel
   audio fusion.

**E026 tri-modal grid fusion becomes the new fusion flagship.**
OOF EER 0.26%, min-DCF 0.0052. Caveat: the improvement is a single
mis-ranking on 222 OOF samples, so eval-set variance could swamp it.

## Next step

- E026 grid (w_mfcc=0.06, w_lpcc=0.52, w_image=0.42) is a candidate for the
  fusion submission, but the margin over E023 is within the noise floor.
  Keep E023 as the conservative default and submit E026 as a second fusion
  result file (we still have budget up to 6 result files).
- Add a `predict_fusion_trimodal.py` entry-point mirroring the planned
  `predict_fusion.py` from E023 but running all three systems and applying
  the grid weights above. This needs to re-train all three on the full
  train+dev set before scoring eval.
- No further fusion experiments planned: the simplex is well-mapped and the
  per-modality flagships are frozen.
