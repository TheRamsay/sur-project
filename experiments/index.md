# Experiment index

One row per experiment. Keep mean ± std to 2 decimals. The authoritative
numbers live in each `EXXX_*.md` — this table is just for scanning.

| ID   | Slug | Modality | Model                  | CV EER [%]    | CV min-DCF    | Notes |
| ---- | ---- | -------- | ---------------------- | ------------- | ------------- | ----- |
| E001 | audio-mfcc-gmm-baseline | audio | GMM 8/32 components, MFCC 13, CMN | 17.92 ± 7.81 | 0.2250 ± 0.0722 | anchor, threshold uncalibrated |
| E002 | audio-mfcc-deltas | audio | GMM 8/32 components, MFCC 13+Δ+ΔΔ, CMN | 10.09 ± 1.81 | 0.1796 ± 0.0540 | −7.83% EER vs E001, std collapsed |
| E003 | audio-gmm-ubm-map | audio | UBM 32 + MAP adapt r=16, MFCC 13+Δ+ΔΔ | 7.45 ± 5.04 | 0.1019 ± 0.0594 | flagship audio, threshold≈0 (well calibrated) |
