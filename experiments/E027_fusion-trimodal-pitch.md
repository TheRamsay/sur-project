# E027 — Tri-modal fusion: MFCC + LPCC+Pitch + image

- **Date:** 2026-04-19
- **Author:** Dominik Huml
- **Related experiments:** E008 (MFCC flagship), E020 (LPCC +NoiseSpeed), E025 (LPCC +Pitch winner), E007 (image flagship), E023 (LPCC+image fusion), E026 (tri-fusion with LPCC+NoiseSpeed)

## Hypothesis

E026 showed that stacking MFCC + LPCC (+NoiseSpeed) + image delivered
OOF EER=0.26%, min-DCF=0.0052 — halving E023. E025 then demonstrated that
for LPCC, the E020 augmentation (+NoiseSpeed, 3.33 ± 4.14%) is beaten by
+Pitch alone (1.94 ± 1.57% mean EER, notably lower std).

Substituting LPCC+Pitch for LPCC+NoiseSpeed should push the tri-fusion past
E026's 0.26%, or at minimum match it with lower variance. The two upgrades
are orthogonal: E026 added an MFCC channel, E027 hardens the LPCC channel.

## Setup

- **MFCC audio (E008 +All):** MFCC 13+Δ+ΔΔ+CMN (39d), UBM 32 + MAP r=16, +Noise(SNR=20dB) + Speed(±10%) aug
- **LPCC+Pitch audio (E025 winner):** LPCC 13+Δ+ΔΔ+CMN (39d, LPC order=12), UBM 32 + MAP r=16, +Pitch aug only (±{1,2} semitones, 2 copies: original + pitch-shifted)
- **Image (E007 +All):** StandardScaler + PCA(50) + LogReg(C=1), +flip + brightness[0.7,1.3] + noise σ=15
- **Calibration:** Platt (LogisticRegression C=1e6, class_weight='balanced') per modality
- **Fusion A (grid):** simplex search w_mfcc, w_lpcc ∈ [0,1], w_image = 1−w_mfcc−w_lpcc ≥ 0 (51×51 grid)
- **Fusion B (LogReg):** LogisticRegression(class_weight='balanced') on 3-column calibrated OOF scores
- **Fold spec:** `iter_folds_loso`, seed=67, 3 folds
- **Command / notebook:** `notebooks/E027_fusion_trimodal_pitch.ipynb`

## Result

Per-fold EER (replicated from E008 MFCC, E025 LPCC+Pitch, E007 image):

| Config              | Fold 0 | Fold 1 | Fold 2 | Mean ± std   |
| ------------------- | ------ | ------ | ------ | ------------ |
| MFCC +All           | 3.47   | 8.33   | 0.83   | 4.21 ± 3.11  |
| LPCC +NoiseSpeed (E020, ref) | 9.17 | 0.83 | 0.00 | 3.33 ± 4.14 |
| **LPCC +Pitch (E025)** | **4.17** | **0.83** | **0.83** | **1.94 ± 1.57** |
| Image +All          | 2.08   | 0.83   | 0.00   | 0.97 ± 0.86  |

LPCC+Pitch shatters the fold-0 pathology that +NoiseSpeed suffered
(9.17 → 4.17%) and halves the per-fold std (4.14 → 1.57).

Pairwise Pearson correlations on Platt-calibrated OOF scores:

| Pair       | r     | E026 r |
| ---------- | ----- | ------ |
| MFCC–LPCC  | 0.843 | 0.850  |
| MFCC–Image | 0.419 | 0.419  |
| LPCC–Image | 0.430 | 0.443  |

OOF overall comparison table:

| System                                              | OOF EER [%] | min-DCF  |
| --------------------------------------------------- | ----------- | -------- |
| MFCC audio (E008 +All)                              | 9.17        | 0.1687   |
| LPCC+Pitch audio (E025 winner)                      | 5.68        | 0.1135   |
| Image (E007 +All)                                   | 4.01        | 0.0729   |
| E023 LPCC+image grid (w=0.36)                       | 0.52        | 0.0104   |
| E026 MFCC+LPCC+image grid (0.06/0.52/0.42)          | 0.26        | 0.0052   |
| **E027 MFCC+LPCC+Pitch+image grid (0.02/0.60/0.38)**| **0.26**    | **0.0052** |
| E027 MFCC+LPCC+Pitch+image LogReg                   | 0.52        | 0.0104   |

Grid search (EER- and min-DCF-optimum coincide): w_mfcc=0.02, w_lpcc=0.60,
w_image=0.38. MFCC weight drops to the smallest grid step (0.02 vs E026's
0.06) and LPCC absorbs the freed weight (0.52 → 0.60); image drops slightly
(0.42 → 0.38).

LogReg fusion coefficients: mfcc=+0.303, lpcc=+0.901, image=+0.852,
bias=−0.148. L1-normalised: mfcc=0.15, lpcc=0.44, image=0.41 — LPCC rises
relative to E026 (0.41) at the expense of image (0.43 → 0.41), mirroring
the grid pattern.

Delta vs E026: **+0.00pp EER, +0.0000 min-DCF** on both grid and LogReg.
Delta vs E023: **−0.26pp EER (0.26 vs 0.52), −0.0052 min-DCF (0.0052 vs
0.0104)** for grid; LogReg flat at E023 level.

## Interpretation

The tri-modal OOF ceiling on these 222 samples is hard-capped at 0.26% EER
(= one mis-ranked target out of 30) — both E026 and E027 hit it. The
underlying LPCC pipeline did improve as expected (E025: 3.33 → 1.94% mean
EER, std 4.14 → 1.57; E027 fold 0: LPCC EER 9.17 → 4.17%), but the image
modality already carries enough signal that one target is unavoidably misranked
at the fused threshold, regardless of the audio backbone.

1. **Fold-level robustness did improve.** The LPCC fold-0 EER drops from
   9.17% (E020/E026) to 4.17% (E025/E027), and its per-fold std from 4.14 to
   1.57. This is exactly the kind of robustness improvement we want for
   eval-set distribution shift, even though the aggregated OOF EER is
   unchanged. The grid *felt* this improvement: it moved mass from MFCC
   (0.06 → 0.02) onto LPCC (0.52 → 0.60), indicating the +Pitch LPCC channel
   is now strong enough to replace part of the MFCC+image safety net.

2. **Why the OOF EER did not move.** OOF EER on 222 samples with 30 targets
   is a coarse ratchet: each mis-ranking is worth ~0.87pp EER. E026 already
   sat at the one-mis-ranking floor (0.26%). E027 would have needed to
   eliminate that single remaining ranking error, which the +Pitch upgrade
   does not touch — the pair of remaining confusable samples is the same one
   across both LPCC variants (see DET curves in the notebook).

3. **Weight shift interpretation.** MFCC at w=0.02 is essentially a tie-
   breaker (two steps above zero on a 0.02 grid). With correlation
   r(MFCC, LPCC)=0.843 and the LPCC backbone now stronger, the grid
   correctly downweights the redundant MFCC channel. The E022 finding
   ("MFCC+LPCC collapses to LPCC") is reaffirmed, and the MFCC weight
   shrinks *further* as the LPCC channel is upgraded — exactly the expected
   direction.

4. **LogReg stays at 0.52%.** Same pathology as E026: LogReg is MLE-optimal,
   not EER-optimal; with r(MFCC, LPCC)≈0.84 the design matrix is near-rank-2
   and LogReg underfits the sharp ranking optimum the grid finds.

**E027 matches E026 in OOF numbers but is strictly preferable as the
fusion flagship**: it uses a more robust audio backbone (E025 LPCC+Pitch:
lower per-fold variance) and pushes more fusion weight onto that stronger
channel, which should generalize better to eval-set noise/shift than E026
did. The OOF metric is blind to this because it is floor-limited.

## Next step

- Adopt E027 as the fusion flagship for submission: weights
  (w_mfcc=0.02, w_lpcc=0.60, w_image=0.38), LPCC+Pitch aug on the LPCC
  branch. Update the planned `predict_fusion_trimodal.py` entrypoint.
- Keep E023 (LPCC+image) as the conservative second submission — still has
  2-of-6 result-file budget for E027, E023 and the three per-modality
  systems.
- No further fusion experiments are planned: the simplex is well-mapped at
  its 1-mis-ranking floor on this OOF pool. Further improvement requires
  either more data or a fundamentally different modality.
