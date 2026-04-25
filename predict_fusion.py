#!/usr/bin/env python3
"""E039 trimodal fusion: MFCC + LPCC (E052) + Image (E033).

Per-stream Platt calibration, then a 51×51 simplex grid search over weights
that directly minimises OOF EER. Flagship: 0.26 % OOF EER (0/222 errors).

Usage:
    uv run predict_fusion.py --eval-dir /path/to/eval --output results/fusion_trimodal.txt
"""
import argparse
from pathlib import Path

import numpy as np

from src.data.manifest import find_png, find_wav
from src.data.splits import iter_folds_loso, load_manifest
from src.eval.metrics import compute_min_dcf
from src.fusion.calibration import apply_platt, fit_platt
from src.fusion.weights import grid_search_simplex
from src.models.pca_logreg import score_image, train_image_pipeline
from src.models.ubm_map import (
    score_lpcc_speed_tta,
    score_mfcc,
    train_lpcc_pipeline,
    train_mfcc_pipeline,
)

SEED = 67


def collect_oof_streams(manifest, data_dir: Path):
    """Return (oof_mfcc, oof_lpcc, oof_image) — three Platt-pre-calibration streams."""
    n = len(manifest)
    oof_mfcc, oof_lpcc, oof_image = (np.full(n, np.nan) for _ in range(3))

    for fold_id, train_idx, val_idx in iter_folds_loso(manifest, seed=SEED):
        seed_f = SEED + fold_id
        train_df = manifest.loc[train_idx]
        val_df = manifest.loc[val_idx]
        print(f"  fold {fold_id}…")

        ubm_m, ad_m = train_mfcc_pipeline(train_df, data_dir, augment=True, seed=seed_f)
        ubm_l, ad_l = train_lpcc_pipeline(train_df, data_dir, augment=True, seed=seed_f)
        scaler, pca, clf = train_image_pipeline(train_df, data_dir, augment=True, seed=seed_f)

        for idx, row in val_df.iterrows():
            oof_mfcc[idx] = score_mfcc(find_wav(row["stem"], data_dir), ad_m, ubm_m)
            oof_lpcc[idx] = score_lpcc_speed_tta(find_wav(row["stem"], data_dir), ad_l, ubm_l)
            oof_image[idx] = score_image(find_png(row["stem"], data_dir), scaler, pca, clf)

    return oof_mfcc, oof_lpcc, oof_image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data", type=Path)
    parser.add_argument("--eval-dir", required=True, type=Path)
    parser.add_argument("--output", default="results/fusion_trimodal.txt", type=Path)
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    eval_dir = args.eval_dir.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(data_dir)
    y_all = manifest["label"].to_numpy()

    print("Collecting OOF scores (3 streams × 3 folds)…")
    oof_mfcc, oof_lpcc, oof_image = collect_oof_streams(manifest, data_dir)

    print("Fitting per-stream Platt calibrators…")
    cal_m = fit_platt(oof_mfcc, y_all)
    cal_l = fit_platt(oof_lpcc, y_all)
    cal_i = fit_platt(oof_image, y_all)
    cal_mo = apply_platt(cal_m, oof_mfcc)
    cal_lo = apply_platt(cal_l, oof_lpcc)
    cal_io = apply_platt(cal_i, oof_image)

    print("Grid search on the 2-simplex (51 steps)…")
    eer_best, (W_M, W_L, W_I) = grid_search_simplex([cal_mo, cal_lo, cal_io], y_all)
    fused_oof = W_M * cal_mo + W_L * cal_lo + W_I * cal_io
    _, threshold = compute_min_dcf(fused_oof[y_all == 1], fused_oof[y_all == 0])
    print(f"  weights: mfcc={W_M:.2f}  lpcc={W_L:.2f}  image={W_I:.2f}")
    print(f"  OOF EER={eer_best * 100:.2f}%  threshold={threshold:.4f}")

    print("Retraining all 3 models on full data…")
    ubm_m, ad_m = train_mfcc_pipeline(manifest, data_dir, augment=True, seed=SEED)
    ubm_l, ad_l = train_lpcc_pipeline(manifest, data_dir, augment=True, seed=SEED)
    scaler, pca, clf = train_image_pipeline(manifest, data_dir, augment=True, seed=SEED)

    print(f"Scoring eval dir {eval_dir}…")
    wavs = sorted(eval_dir.glob("*.wav"))
    if not wavs:
        raise RuntimeError(f"No .wav files in {eval_dir}")

    with open(args.output, "w") as f:
        for wav in wavs:
            stem = wav.stem
            png = eval_dir / (stem + ".png")

            raw_m = score_mfcc(wav, ad_m, ubm_m)
            raw_l = score_lpcc_speed_tta(wav, ad_l, ubm_l)
            s_m = float(cal_m.decision_function([[raw_m]])[0])
            s_l = float(cal_l.decision_function([[raw_l]])[0])

            if png.exists():
                raw_i = score_image(png, scaler, pca, clf)
                s_i = float(cal_i.decision_function([[raw_i]])[0])
                score = W_M * s_m + W_L * s_l + W_I * s_i
            else:
                # audio-only fallback if image missing
                wa = W_M + W_L
                score = (W_M / wa) * s_m + (W_L / wa) * s_l

            hard = 1 if score >= threshold else 0
            f.write(f"{stem} {score:.6f} {hard}\n")

    print(f"Written {len(wavs)} lines → {args.output}")


if __name__ == "__main__":
    main()
