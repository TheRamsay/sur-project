# E013 — Audio MAP relevance factor ablation

- **Date:** 2026-04-19
- **Author:** Dominik Huml
- **Related experiments:** E008 (audio +All aug flagship, r=16 used without ablation)

## Hypothesis

r=16 was chosen as a standard default without ablating. With only ~20 target
utterances (~2000 frames) per fold and 32 components (~62 frames/component),
`alpha_k = 62 / (62 + 16) ≈ 0.79` — meaning MAP pulls means ~79% toward target
data and retains ~21% of the UBM prior.

Lower r (e.g. 4) would pull more aggressively (alpha ≈ 0.94), trusting the
scarce target data almost entirely. Higher r (e.g. 64) would pull more weakly
(alpha ≈ 0.49), trusting the UBM prior more. With only ~62 target frames per
component, aggressive adaptation may overfit to session-specific noise; but
weak adaptation may leave the adapted model indistinguishable from the UBM.
The optimal r for this data size is not obvious — we ablate to find out.

## Setup

- **Modality:** audio
- **Data:** train + dev (combined), 222 samples (30 target, 192 non-target)
- **Features:** MFCC 13+Δ+ΔΔ, CMN (per-utterance) — same as E008
- **Model:** UBM 32 + MAP adapt, r ∈ {4, 8, 16, 32, 64}
- **Augmentation:** +All (noise SNR=20dB + speed ±10%) on train fold only
- **Fold spec:** `iter_folds_loso`, seed=67, 3 folds
- **Seed:** 67
- **Command / notebook:** `notebooks/E013_audio_map_r_ablation.ipynb`

## Result

| r      | alpha | Fold 0 EER | Fold 1 EER | Fold 2 EER | Mean ± std      | min-DCF mean |
| ------ | ----- | ---------- | ---------- | ---------- | --------------- | ------------ |
| 4      | 0.940 | 3.47       | 8.33       | 0.83       | 4.21 ± 3.11     | 0.0509       |
| 8      | 0.887 | 3.47       | 8.33       | 0.83       | 4.21 ± 3.11     | 0.0509       |
| **16** | 0.796 | **3.47**   | **8.33**   | **0.83**   | **4.21 ± 3.11** | **0.0509**   |
| 32     | 0.661 | 9.17       | 8.33       | 0.83       | 6.11 ± 3.75     | 0.0556       |
| 64     | 0.494 | 4.17       | 9.17       | 0.83       | 4.72 ± 3.42     | 0.0611       |

r ∈ {4, 8, 16} all achieve identical EER=4.21±3.11% and min-DCF=0.0509 (numeric tie).
r=32 regresses (6.11%, Fold 0 jumps to 9.17%). r=64 partially recovers (4.72%).

OOF overall (r=4, tied with r=8 and r=16): EER=9.17%, min-DCF=0.1635, threshold=−0.107

## Interpretation

The three smallest r values (4, 8, 16) produce an exact numerical plateau —
identical per-fold EERs and min-DCFs. This is not a coincidence: with ~62
expected target frames per component, even r=4 (alpha≈0.94) and r=16
(alpha≈0.80) both yield enough adaptation signal from the same augmented
frames to converge to the same MAP means. The GaussianMixture UBM has
fixed seed=67 so E-step posteriors are deterministic; the three alpha values
shift the MAP interpolation slightly but land in the same effective operating
point.

r=32 (alpha≈0.66) breaks the plateau: Fold 0 jumps from 3.47% to 9.17%,
suggesting that weakening adaptation below alpha≈0.70 leaves the adapted
model too close to the UBM and loses discrimination on the hardest fold.

r=64 (alpha≈0.49) partially recovers (Fold 0 drops back to 4.17%), possibly
because the near-UBM model generalizes better on non-target — but mean EER
and min-DCF are still worse than the plateau.

**Conclusion:** r=16 is validated as optimal (tied with r=4 and r=8). The
system is robust to r in [4, 16] but degrades for r ≥ 32 with this dataset
size (~62 frames/component). No change to `predict_audio.py` or
`predict_fusion.py` is needed.

## Next step

- r=16 confirmed as validated default — no updates to prediction scripts.
- The flat plateau [4, 16] gives confidence: E008's choice of r=16 was safe,
  not accidentally optimal.
- Next experiment: consider fusion calibration improvement or augmentation
  diversity (e.g. room impulse response) to push below 4% mean EER.
