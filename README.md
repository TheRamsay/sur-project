# SUR 2025/2026 target-person detector

Detector for target speaker `m431` from paired PNG + WAV samples.
The repository contains image-only, audio-only, and multimodal fusion systems.

This project implements a biometric verification system for a single enrolled
person. Each test sample contains one image and one audio recording. The system
extracts image and speech evidence, scores whether the sample belongs to the
target person, and reports a binary decision.

The image model uses grayscale pixels with standardization, PCA, and logistic
regression. The audio model uses cepstral speech features with GMM-UBM and MAP
adaptation. The final system applies score calibration and weighted score-level
fusion. Model selection and reporting use EER and normalized min-DCF, with
accuracy treated only as a secondary diagnostic because the data are imbalanced.

## Setup

The code expects Python 3.12 and uses `uv` for dependency management.

```bash
uv sync
```

The course dataset is not versioned in this repository. Place it under `data/`
with the original train/dev layout.

```text
data/
  target_train/
  target_dev/
  non_target_train/
  non_target_dev/
```

## Run

Each prediction script trains the required models from `data/` at call time,
cross-validates internally where needed, then scores the supplied evaluation
directory.

```bash
uv run predict_audio.py  --eval-dir <dir> --output results/audio.txt
uv run predict_image.py  --eval-dir <dir> --output results/image.txt
uv run predict_fusion.py --eval-dir <dir> --output results/fusion.txt
```

Output format

```text
<stem> <score> <hard_decision>
```

Higher scores mean stronger evidence for the target person. The hard decision
is `1` for target and `0` for non-target.

## Systems

- Audio uses LPCC/MFCC-style frame features, GMM-UBM background modelling, MAP mean
  adaptation for the target model, and log-likelihood-ratio scoring.
- Image uses grayscale image vectors, standardization, PCA dimensionality reduction,
  and logistic-regression scoring.
- Fusion uses score-level calibration and weighted fusion of audio/image streams.

The detailed design, validation protocol, and final numbers are in
[`docs/dokumentace.md`](docs/dokumentace.md) and
[`docs/dokumentace.pdf`](docs/dokumentace.pdf).

## Sanity check

Run the self-test before packaging or publishing changes. It builds a small
temporary eval set from the local data and verifies output format, score
ordering, and decisions for all three entry points.

```bash
uv run self_test.py
```

## Layout

- `predict_*.py`, `self_test.py` entry points
- `src/` features, augmentation, models, fusion, metrics
- `experiments/` one md log per experiment
- `docs/` `dokumentace.md` and figures
- `data/` corpus, gitignored
- `results/`, `cache/`, `logs/` generated artifacts, gitignored
