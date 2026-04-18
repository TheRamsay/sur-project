# SUR 2025/2026 — target-person detector

Semester project for **FIT VUT SUR 2025/2026**: build a detector for one
specific target person from a face image (PNG) and a voice recording (WAV),
producing three systems — image-only, audio-only, and multimodal fusion.

See [`CLAUDE.md`](CLAUDE.md) for conventions, methodology, and the current
TODO. See [`assignment.txt`](assignment.txt) for the full course brief.

## Quick start

```bash
uv sync
uv run jupyter lab   # explore notebooks/data.ipynb
```

## Layout

- `src/` — production code for the three systems
- `notebooks/` — exploratory analysis
- `experiments/` — one markdown log per experiment
- `data/` — provided corpus (not versioned)
