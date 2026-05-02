#!/usr/bin/env python3
"""E033 image system: PCA-50 + LogReg + flip/brightness/noise/adv-rot aug + flip TTA.

Flagship CV EER: 0.51 +/- 0.36 %.

Usage:
    uv run predict_image.py --eval-dir /path/to/eval --output results/image_pca_adv_rot.txt
"""
import argparse
from pathlib import Path

import numpy as np

from src.data.manifest import find_png
from src.data.splits import iter_folds_loso, load_manifest
from src.eval.metrics import compute_min_dcf
from src.features.image import load_image
from src.fusion.calibration import apply_platt, fit_platt
from src.models.pca_logreg import score_image_flip_tta, train_image_pipeline

SEED = 67


def collect_oof(manifest, data_dir: Path) -> np.ndarray:
    oof = np.full(len(manifest), np.nan)
    for fold_id, train_idx, val_idx in iter_folds_loso(manifest, seed=SEED):
        scaler, pca, clf = train_image_pipeline(
            manifest.loc[train_idx], data_dir, augment=True, seed=SEED + fold_id
        )
        X_val = np.stack(
            [load_image(find_png(row["stem"], data_dir)) for _, row in manifest.loc[val_idx].iterrows()]
        )
        X_pca = pca.transform(scaler.transform(X_val))
        oof[val_idx] = clf.decision_function(X_pca)
    return oof


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data", type=Path)
    parser.add_argument("--eval-dir", required=True, type=Path)
    parser.add_argument("--output", default="results/image_pca_adv_rot.txt", type=Path)
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
    scaler, pca, clf = train_image_pipeline(manifest, data_dir, augment=True, seed=SEED)

    print(f"Scoring eval data in {eval_dir} ...")
    pngs = sorted(eval_dir.glob("*.png"))
    if not pngs:
        raise RuntimeError(f"No .png files found in {eval_dir}")

    with open(args.output, "w") as f:
        for png in pngs:
            raw = score_image_flip_tta(png, scaler, pca, clf)
            score = float(cal.decision_function([[raw]])[0])
            hard = 1 if score >= threshold else 0
            f.write(f"{png.stem} {score:.6f} {hard}\n")

    print(f"Written {len(pngs)} lines -> {args.output}")


if __name__ == "__main__":
    main()
