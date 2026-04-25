# SUR 2025/2026 — target-person detector

Semester project for **FIT VUT SUR 2025/2026**: build a detector for one
specific target person (`m431`) from a face image (PNG) and a voice recording
(WAV). Three systems are required: image-only, audio-only, and multimodal
fusion. See [`assignment.txt`](assignment.txt) for the full brief.

## Setup

```bash
uv sync
```

Python 3.12, all dependencies pinned in `pyproject.toml` / `uv.lock`.

## Running the systems

Each script reads a directory of evaluation samples and writes one line per
sample to the output file (`<stem> <score> <hard_decision>`):

```bash
# audio (E052: LPCC + tied-cov UBM-MAP + pitch & codec aug + speed TTA)
uv run predict_audio.py  --eval-dir path/to/eval --output results/audio.txt

# image (E033: PCA + LogReg + adversarial rotation aug + flip TTA)
uv run predict_image.py  --eval-dir path/to/eval --output results/image.txt

# trimodal fusion (E039: MFCC + LPCC + image, simplex grid weights)
uv run predict_fusion.py --eval-dir path/to/eval --output results/fusion.txt
```

Each script trains end-to-end on `data/` at call time (a few minutes per run),
calibrates with a per-fold OOF Platt, and applies the calibrated threshold to
the eval set.

## Sanity check before submission

```bash
uv run self_test.py
```

Builds a 10-sample eval directory from dev data, runs all three predict
scripts, and verifies output format plus score ordering. Should report
`✓ ALL CHECKS PASSED`.

## Layout

- `predict_*.py` — three entry points, one per system
- `self_test.py` — pre-submission sanity check
- `src/` — feature extraction, augmentation, models, fusion, metrics
- `experiments/` — one markdown log per experiment
- `notebooks/` — exploratory analysis
- `docs/` — `dokumentace.md` (submission documentation) and figures
- `data/` — provided corpus, gitignored
