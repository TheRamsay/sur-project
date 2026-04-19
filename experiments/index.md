# Experiment index

One row per experiment. Keep mean ± std to 2 decimals. The authoritative
numbers live in each `EXXX_*.md` — this table is just for scanning.

| ID   | Slug | Modality | Model                  | CV EER [%]    | CV min-DCF    | Notes |
| ---- | ---- | -------- | ---------------------- | ------------- | ------------- | ----- |
| E001 | audio-mfcc-gmm-baseline | audio | GMM 8/32 components, MFCC 13, CMN | 17.92 ± 7.81 | 0.2250 ± 0.0722 | anchor, threshold uncalibrated |
| E002 | audio-mfcc-deltas | audio | GMM 8/32 components, MFCC 13+Δ+ΔΔ, CMN | 10.09 ± 1.81 | 0.1796 ± 0.0540 | −7.83% EER vs E001, std collapsed |
| E003 | audio-gmm-ubm-map | audio | UBM 32 + MAP adapt r=16, MFCC 13+Δ+ΔΔ | 7.45 ± 5.04 | 0.1019 ± 0.0594 | flagship audio, threshold≈0 (well calibrated) |
| E004 | image-pca-logreg-baseline | image | PCA 50 + LogReg C=1, 80×80 grayscale | 4.49 ± 4.26 | 0.0565 ± 0.0352 | beats audio flagship, strong anchor |
| E005 | image-lbp-logreg | image | LBP 4×4 grid 256-bin + LogReg C=1 | 17.78 ± 23.58 | 0.2278 ± 0.2502 | fold 2 collapsed (45% EER), session shift kills LBP |
| E006 | image-fisherfaces | image | PCA 100 + LDA shrinkage=auto | 18.24 ± 1.53 | 0.2657 ± 0.0667 | LDA 1D bottleneck loses to logreg in 50D, E004 stays flagship |
| E007 | image-augmentation-ablation | image | PCA 50 + LogReg + aug ablation (flip/bright/noise/all) | 0.97 ± 0.86 (+All) | 0.0194 | brightness key contributor; +All = new image flagship |
| E008 | audio-augmentation-ablation | audio | UBM+MAP + aug ablation (noise/speed/all) | 4.21 ± 3.11 (+All) | 0.0509 | speed key contributor; +All = new audio flagship |
| E009 | fusion-score-level | fusion | Platt calib + grid search w=0.28 (72% image) | 3.75 (OOF overall) | 0.0750 | beats both modalities alone; image gets more weight |
| E010 | audio-ubm64 | audio | UBM 64 + MAP adapt r=16, +All aug | 6.39 ± 3.93 | 0.0611 | regression vs E008 (4.21%); 64 comp over-parameterized for dataset size |
| E011 | image-pca-sweep | image | PCA sweep n∈{20,30,50,75,100,150} + LogReg + +All aug | 0.97 ± 0.86 (n=50) | 0.0194 | n=50 confirmed optimal; <30 underfit, ≥75 plateau at 1.25% EER |
| E012 | audio-cmvn | audio | UBM 32 + MAP r=16, MFCC 13+Δ+ΔΔ, CMVN (+All aug) | 6.16 ± 3.78 | 0.0796 | regression vs E008 CMN (4.21%); CMVN removes speaker-discriminative variance |
