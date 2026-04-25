"""E052 audio stress test: re-run E051's audio stress matrix with the codec-aug
flagship (E052) so we have a paired before/after table & figure for §3.3.
"""
from pathlib import Path
import numpy as np
import librosa
import pickle

from src.data.manifest import find_wav
from src.data.splits import iter_folds_loso, load_manifest
from src.eval.metrics import compute_eer
from src.features.audio import extract_lpcc
from src.models.ubm_map import (
    train_lpcc_pipeline,
    score_lpcc_speed_tta,
    llr_score,
)

SEED = 67
DATA = Path("data").resolve()
RNG = np.random.default_rng(42)


def stress_noise(y, snr_db):
    p = np.mean(y**2) + 1e-10
    noise = RNG.normal(0, np.sqrt(p / 10 ** (snr_db / 10)), len(y))
    return (y + noise.astype(y.dtype)).clip(-1, 1)


def stress_speed(y, rate):
    return librosa.effects.time_stretch(y, rate=rate)


def stress_codec(y, sr):
    y_down = librosa.resample(y, orig_sr=sr, target_sr=8000)
    return librosa.resample(y_down, orig_sr=8000, target_sr=sr)


def apply_stress(y, sr, name):
    if name == "clean":   return y
    if name == "noise20": return stress_noise(y, 20)
    if name == "noise10": return stress_noise(y, 10)
    if name == "noise5":  return stress_noise(y, 5)
    if name == "slow":    return stress_speed(y, 0.8)
    if name == "fast":    return stress_speed(y, 1.2)
    if name == "codec":   return stress_codec(y, sr)
    if name == "all":
        y = stress_noise(y, 10)
        y = stress_codec(y, sr)
        return stress_speed(y, 0.85)
    raise ValueError(name)


def score_with_tta_on_array(y, sr, adapted, ubm):
    """Speed TTA over original-array (already stressed). Avoids re-loading."""
    scores = []
    for rate in (0.9, 1.0, 1.1):
        ys = y if rate == 1.0 else librosa.effects.time_stretch(y, rate=rate)
        feat = extract_lpcc(ys, sr)
        scores.append(llr_score(feat, adapted, ubm))
    return float(np.mean(scores))


STRESSES = ["clean", "noise20", "noise10", "noise5", "slow", "fast", "codec", "all"]


def main():
    manifest = load_manifest(DATA)
    print(f"{len(manifest)} samples")

    results = {s: [] for s in STRESSES}
    print("\n=== E052 (LPCC + tied cov + pitch&codec aug + speed TTA) ===")
    for fold_id, train_idx, val_idx in iter_folds_loso(manifest, seed=SEED):
        print(f"  fold {fold_id}…", end=" ", flush=True)
        train_df = manifest.loc[train_idx]
        ubm, adapted = train_lpcc_pipeline(
            train_df, DATA, augment=True, seed=SEED + fold_id
        )
        val_df = manifest.loc[val_idx]

        for stress in STRESSES:
            scores, labels = [], []
            for _, row in val_df.iterrows():
                y, sr = librosa.load(find_wav(row["stem"], DATA), sr=None, mono=True)
                y_s = apply_stress(y, sr, stress)
                scores.append(score_with_tta_on_array(y_s, sr, adapted, ubm))
                labels.append(row["label"])
            scores = np.array(scores); labels = np.array(labels)
            eer, _ = compute_eer(scores[labels == 1], scores[labels == 0])
            results[stress].append(eer * 100)
        print("done")

    print("\n=== AUDIO STRESS TEST RESULTS (E052) ===\n")
    print(f'{"Stress":<14} {"F0":>6} {"F1":>6} {"F2":>6} {"Mean±Std":>18}')
    print("-" * 50)
    for s in STRESSES:
        f = results[s]
        print(f"{s:<14} {f[0]:>6.2f} {f[1]:>6.2f} {f[2]:>6.2f}  {np.mean(f):>5.2f} ± {np.std(f):.2f}%")

    out = Path("cache/E052_stress.pkl")
    out.parent.mkdir(exist_ok=True)
    pickle.dump(results, open(out, "wb"))
    print(f"\nsaved → {out}")


if __name__ == "__main__":
    main()
