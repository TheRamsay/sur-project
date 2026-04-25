# `src/`: source tree

Single source of truth for the production code behind the three submitted systems.

| Submodule     | Purpose                                                                 |
| ------------- | ----------------------------------------------------------------------- |
| `data/`       | Manifest loading, train/dev splits, group-aware CV fold generation.     |
| `features/`   | Feature extraction (MFCC, filterbanks, image features). Cache-friendly. |
| `models/`     | Per-modality classifiers (audio, image). One module per approach.       |
| `fusion/`     | Score- / feature- / decision-level fusion + calibration.                |
| `eval/`       | Metrics (EER, min-DCF), thresholding, OOF collection, report helpers.   |

## Conventions

- No code in a submodule depends on another except via explicit imports.
- Functions take paths/arrays in, return arrays/metrics out. No globals.
- Every model exposes `fit(X, y, groups) -> self` and `score(X) -> float[n]`
  returning a higher-is-target score, mirroring the evaluation format.
- Fold assignment comes from **one** function in `data/`: all systems share it
  so OOF scores align across modalities for stacking fusion.
