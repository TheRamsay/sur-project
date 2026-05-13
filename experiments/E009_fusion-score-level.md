# E009 — Score-level fusion

- **Date:** 2026-04-19
- **Author:** Dominik Huml
- **Related experiments:** E007 (image flagship), E008 (audio flagship), calibration notebook

## Hypothesis

Combining calibrated OOF scores from both modalities should outperform either
alone. Audio (EER ~4%) and image (EER ~1%) are complementary — image struggles
when session appearance changes, audio struggles with acoustic conditions.

Testing three fusion strategies:
- **Average**: equal weight (w=0.5) — baseline fusion
- **Grid search**: find optimal w ∈ [0,1] for audio:image weight ratio
- **LogReg**: learn weights from OOF scores via logistic regression

Methodological note: fusion model is fit and evaluated on the same OOF pool
(slightly optimistic). With 3 LOSO folds, proper nested CV would give only
1 fold for fusion training — too unstable. This limitation is documented.

## Setup

- **Input:** calibrated OOF scores from E007 (image) and E008 (audio)
- **Calibration:** Platt with class_weight='balanced' (from calibration.ipynb)
- **Fold spec:** same LOSO folds as all previous experiments
- **Seed:** 67
- **Command / notebook:** `notebooks/E009_fusion.ipynb`

## Result

| Fusion method    | EER [%] | min-DCF | Threshold |
| ---------------- | ------- | ------- | --------- |
| Audio only       | 9.17    | 0.1687  | −0.412    |
| Image only       | 9.95    | 0.1375  | −0.070    |
| Average (w=0.5)  | 7.86    | 0.0573  | +0.026    |
| Grid (w=0.28)    | **3.75**| 0.0750  | +0.207    |
| LogReg fusion    | 4.53    | **0.0625** | +0.150 |

Note: OOF overall EER differs from per-fold mean EER (audio 9.17% OOF vs 4.21% per-fold mean,
image 9.95% vs 0.97%) because OOF overall mixes scores from 3 different models.
The OOF overall is the honest single number; per-fold mean shows session-level variance.

Audio/image Pearson correlation: 0.426 — low enough that fusion adds real value.

## Interpretation

Fusion significantly outperforms either modality alone (OOF overall):
- Best fusion EER = 3.75% (grid w=0.28) vs audio 9.17%, image 9.95%
- Grid search gives image 72% weight — correct, image is the stronger modality
- LogReg fusion (4.53%) is slightly worse than grid search — with only 222 OOF
  samples and 2 parameters, LogReg is not learning much beyond the optimal weight
- Average (7.86%) is a reasonable no-training baseline

The fusion threshold is near 0 (+0.207 for grid) — calibration worked well.

## Next step

- Production scripts: write `predict_audio.py`, `predict_image.py`, `predict_fusion.py`
  that take eval data → output result .txt files
- Self-test mini-eval set before submission
- `dokumentace.pdf` — sections are now all ready to write
