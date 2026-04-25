# SUR 2025/2026 — target-person detector

Detector for target speaker `m431` from paired PNG + WAV samples.
Three systems: image-only, audio-only, multimodal fusion.
See [`assignment.txt`](assignment.txt) for the brief.

## Setup

```bash
uv sync
```

## Run

```bash
uv run predict_audio.py  --eval-dir <dir> --output results/audio.txt
uv run predict_image.py  --eval-dir <dir> --output results/image.txt
uv run predict_fusion.py --eval-dir <dir> --output results/fusion.txt
```

Each script trains on `data/` at call time and writes
`<stem> <score> <hard_decision>` per line.

## Sanity check

```bash
uv run self_test.py
```

## Layout

- `predict_*.py`, `self_test.py` — entry points
- `src/` — features, augmentation, models, fusion, metrics
- `experiments/` — one md log per experiment
- `notebooks/` — exploratory analysis
- `docs/` — `dokumentace.md` and figures
- `data/` — corpus, gitignored
