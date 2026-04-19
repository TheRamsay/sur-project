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
| E013 | audio-map-r-ablation | audio | UBM 32 + MAP r∈{4,8,16,32,64}, +All aug | 4.21 ± 3.11 (r≤16) | 0.0509 | flat plateau r∈{4,8,16}; r=32 regresses; r=16 confirmed as default |
| E014 | audio-new-aug-ablation | audio | UBM 32 + MAP r=16, +All + new augs (codec/lownoise/pitch/clip/allnew) | 4.21 ± 3.11 (+All E008 wins) | 0.0509 | none of the 4 new augs beat E008 +All; +Codec closest (4.81%); +LowNoise worst (7.59%) |
| E015 | image-new-aug-ablation | image | PCA 50 + LogReg + E007 +All + new augs (jpeg/blur/rotate/contrast/allnew) | 0.97 ± 0.86 (E007 +All wins) | 0.0194 | none of the 4 new augs beat E007 +All; +Blur closest (1.53%); +Contrast worst (5.88%) |
| E016 | audio-fbank | audio | UBM 32 + MAP r=16, FBank 40+Δ+ΔΔ (120d), +All aug | 9.95 ± 1.36 | 0.1324 | regression vs E008 (4.21%); 120d over-parameterizes GMM; DCT compression is beneficial regularization for GMM |
| E017 | audio-gmm-svm | audio | UBM 32 + MAP r=16, deviation supervector (1248d) + LinearSVC C=1.0, +All aug | 9.07 ± 7.45 | 0.1370 | regression vs E008 (4.21%); n_features>>n_samples kills SVM; LLR inductive bias superior for small data |
| E018 | audio-vtlp | audio | UBM 32 + MAP r=16, VTLP α∈[0.9,1.1] (replace speed: 3.94±3.28%; add to +All: 6.34±3.95%) | 3.94 ± 3.28 (+VTLP_replace_speed) | 0.0685 | marginal mean gain vs E008 (−0.27pp) but min-DCF regresses; +All+VTLP hurts via UBM dilution; E008 +All stays flagship |
| E019 | audio-sdc | audio | UBM 32 + MAP r=16, MFCC 13 + SDC N=7 d=1 P=3 (104d), +All aug | 12.96 ± 4.79 | 0.1778 | regression vs E008 (4.21%); 104d over-parameterizes GMM-32 on small data; Δ+ΔΔ stays optimal |
| E020 | audio-lpcc | audio | UBM 32 + MAP r=16, LPCC 13+Δ+ΔΔ (39d, LPC order=12 cep via FFT), +All aug | 3.33 ± 4.14 | 0.0333 | marginal mean improvement vs E008 (−0.88pp); higher variance (fold 0 regresses to 9.17%); min-DCF clearly better (0.0333 vs 0.0509); LPCC candidate for fusion |
| E021 | audio-plp | audio | UBM 32 + MAP r=16, PLP 13+Δ+ΔΔ (39d, Bark 20 bands, EL, cube-root, LPC order=12), +All aug | 5.56 ± 2.58 | 0.0944 | regression vs E008 (+1.35pp) and E020 (+2.23pp); fold 2 catastrophic (9.17%); equal loudness + coarse Bark filterbank hurt more than cube-root compression helps; PLP not viable in this UBM-32 regime |
| E022 | audio-mfcc-lpcc-fusion | audio | UBM 32 + MAP r=16, MFCC + LPCC score-level fusion, Platt calib, grid w=0.07 | 3.33 ± 4.14 | 0.0333 | fusion collapses to LPCC (w=0.07); global OOF Platt asymmetric (LPCC slope 10.1 vs MFCC 5.6); per-fold complementarity lost; E020 LPCC remains audio flagship |
| E023 | fusion-lpcc-image | fusion | LPCC audio (E020) + image (E007) +All, Platt calib, grid w=0.36 | 0.52 (OOF overall) | 0.0104 | −3.23pp vs E009 MFCC+image (3.75%); min-DCF 0.0104 vs 0.0750; new fusion flagship |
| E024 | audio-lpc-order-ablation | audio | UBM 32 + MAP r=16, LPCC 13+Δ+ΔΔ, LPC order∈{8,10,12,14,16,20}, +All aug | 3.33 ± 4.14 (order=12) | 0.0333 | order=12 confirmed optimal; ≥2.5pp moat over neighbors (8→6.90, 10→6.62, 14→5.88, 16→5.83, 20→6.57); fs/1000+2 rule holds; keep predict_audio.py unchanged |
