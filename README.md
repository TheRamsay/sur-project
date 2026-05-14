# SUR 2025/2026: target-person detector

Detector for target speaker `m431` from paired PNG + WAV samples.
Three systems: image-only, audio-only, multimodal fusion.

The project was built for a small biometric verification task: given an
evaluation directory with one image and one audio file per sample, produce a
score and a hard target/non-target decision. The submitted fusion system
combines complementary evidence from face-like image features and speaker audio
features.

## Setup

The code expects Python 3.12 and uses [`uv`](https://github.com/astral-sh/uv)
for dependency management.

```bash
uv sync
```

The course dataset is not versioned in this repository. Place it under `data/`
with the original train/dev layout:

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

Output format:

```text
<stem> <score> <hard_decision>
```

Higher scores mean stronger evidence for the target person. The hard decision
is `1` for target and `0` for non-target.

## Systems

- Audio: LPCC/MFCC-style frame features, GMM-UBM background modelling, MAP mean
  adaptation for the target model, and log-likelihood-ratio scoring.
- Image: grayscale image vectors, standardization, PCA dimensionality reduction,
  and logistic-regression scoring.
- Fusion: score-level calibration and weighted fusion of audio/image streams.

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

- `predict_*.py`, `self_test.py`: entry points
- `src/`: features, augmentation, models, fusion, metrics
- `experiments/`: one md log per experiment
- `docs/`: `dokumentace.md` and figures
- `data/`: corpus, gitignored
- `results/`, `cache/`, `logs/`: generated artifacts, gitignored
