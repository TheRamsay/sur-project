#!/usr/bin/env python3
"""E052 audio system: LPCC + tied-covariance UBM + MAP + pitch & codec aug + speed TTA.

Flagship CV EER: 0.46 +/- 0.65 % (clean), 3.33 +/- 4.14 % (codec-stressed).

Usage:
    uv run predict_audio.py --eval-dir /path/to/eval --output results/audio_lpcc_tied_codecaug.txt
"""
import argparse
from pathlib import Path

import numpy as np

from src.data.manifest import find_wav
from src.data.splits import iter_folds_loso, load_manifest
from src.eval.metrics import compute_min_dcf
from src.fusion.calibration import apply_platt, fit_platt
from src.models.ubm_map import score_lpcc_speed_tta, train_lpcc_pipeline

SEED = 67


def collect_oof(manifest, data_dir: Path) -> np.ndarray:
    oof = np.full(len(manifest), np.nan)
    for fold_id, train_idx, val_idx in iter_folds_loso(manifest, seed=SEED):
        ubm, adapted = train_lpcc_pipeline(
            manifest.loc[train_idx], data_dir, augment=True, seed=SEED + fold_id
        )
        for idx, row in manifest.loc[val_idx].iterrows():
            oof[idx] = score_lpcc_speed_tta(find_wav(row["stem"], data_dir), adapted, ubm)
    return oof


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data", type=Path)
    parser.add_argument("--eval-dir", required=True, type=Path)
    parser.add_argument("--output", default="results/audio_lpcc_tied_codecaug.txt", type=Path)
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    eval_dir = args.eval_dir.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(data_dir)
    y_all = manifest["label"].to_numpy()

    print("Collecting OOF scores for calibration...")
    oof_raw = collect_oof(manifest, data_dir)

    print("Fitting Platt calibrator...")
    cal = fit_platt(oof_raw, y_all)
    oof_cal = apply_platt(cal, oof_raw)
    _, threshold = compute_min_dcf(oof_cal[y_all == 1], oof_cal[y_all == 0])
    print(f"  Threshold (min-DCF on OOF): {threshold:.4f}")

    print("Retraining on all data...")
    ubm, adapted = train_lpcc_pipeline(manifest, data_dir, augment=True, seed=SEED)

    print(f"Scoring eval data in {eval_dir} ...")
    wavs = sorted(eval_dir.glob("*.wav"))
    if not wavs:
        raise RuntimeError(f"No .wav files found in {eval_dir}")

    with open(args.output, "w") as f:
        for wav in wavs:
            raw = score_lpcc_speed_tta(wav, adapted, ubm)
            score = float(cal.decision_function([[raw]])[0])
            hard = 1 if score >= threshold else 0
            f.write(f"{wav.stem} {score:.6f} {hard}\n")

    print(f"Written {len(wavs)} lines -> {args.output}")


if __name__ == "__main__":
    main()
