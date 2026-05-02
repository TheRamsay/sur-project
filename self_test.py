#!/usr/bin/env python3
"""Self-test: run the three predict scripts on a 10-sample mini eval set
built from dev data (known ground truth) and check output format plus
score ordering. A common failure mode is stems correct but scores
shuffled due to dict-iteration order; this catches that.

Usage:
    uv run self_test.py
"""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Ground truth: files we'll use for the self-test
# Taken from dev splits, we know exactly which are target (m431) and which aren't
TARGET_STEMS = [
    "m431_03_p01_i0_0",
    "m431_03_p02_i0_0",
    "m431_03_p03_i0_0",
    "m431_03_p04_i0_0",
    "m431_03_p05_i0_0",
]
NONTARGET_STEMS = [
    "f407_01_f16_i0_0",
    "f407_01_r08_i0_0",
    "f408_01_f16_i0_0",
    "f408_01_r08_i0_0",
    "m423_01_f16_i0_0",
]

DATA_DIR = Path("data")
SCRIPTS = {
    "audio": ("predict_audio.py", "audio_ubm_map_aug.txt", ".wav"),
    "image": ("predict_image.py", "image_pca_logreg_aug.txt", ".png"),
    "fusion": ("predict_fusion.py", "fusion_score.txt", ".wav"),  # fusion finds .png itself
}


def find_file(stem: str, ext: str) -> Path:
    for sf in ("target_train", "target_dev", "non_target_train", "non_target_dev"):
        p = DATA_DIR / sf / (stem + ext)
        if p.exists():
            return p
    raise FileNotFoundError(f"{stem}{ext}")


def parse_results(result_file: Path) -> list[tuple[str, float, int]]:
    rows = []
    for lineno, line in enumerate(result_file.read_text().splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 3:
            raise ValueError(f"Line {lineno}: expected 3 fields, got {len(parts)}: {line!r}")
        stem, score_str, dec_str = parts
        try:
            score = float(score_str)
        except ValueError:
            raise ValueError(f"Line {lineno}: score is not a float: {score_str!r}")
        if dec_str not in ("0", "1"):
            raise ValueError(f"Line {lineno}: hard_decision must be 0 or 1, got {dec_str!r}")
        rows.append((stem, score, int(dec_str)))
    return rows


def check_results(
    rows: list,
    expected_stems: set[str],
    gt: dict[str, int],
    script_name: str,
) -> list[str]:
    problems = []

    # 1. Count
    if len(rows) != len(expected_stems):
        problems.append(f"Expected {len(expected_stems)} lines, got {len(rows)}")

    # 2. No duplicate stems
    seen_stems = [r[0] for r in rows]
    if len(seen_stems) != len(set(seen_stems)):
        problems.append("Duplicate stems in output")

    # 3. All expected stems present
    output_stems = set(seen_stems)
    missing = expected_stems - output_stems
    extra   = output_stems - expected_stems
    if missing:
        problems.append(f"Missing stems: {sorted(missing)}")
    if extra:
        problems.append(f"Unexpected stems: {sorted(extra)}")

    # 4. No file extension in stem
    with_ext = [s for s in seen_stems if s.endswith((".wav", ".png", ".mp3"))]
    if with_ext:
        problems.append(f"Stems with extensions: {with_ext}")

    # 5. Score-decision consistency (decision = 1 iff score >= threshold)
    # We infer threshold from the boundary between 0 and 1 decisions
    ones  = [r[1] for r in rows if r[2] == 1]
    zeros = [r[1] for r in rows if r[2] == 0]
    if ones and zeros:
        min_one = min(ones)
        max_zero = max(zeros)
        if max_zero >= min_one:
            problems.append(
                f"Score/decision inconsistency: "
                f"non-target score {max_zero:.4f} >= target decision boundary {min_one:.4f}"
            )

    # 6. Target samples score higher than non-target (sanity)
    target_scores    = [r[1] for r in rows if gt.get(r[0]) == 1]
    nontarget_scores = [r[1] for r in rows if gt.get(r[0]) == 0]
    if target_scores and nontarget_scores:
        mean_t  = sum(target_scores)  / len(target_scores)
        mean_nt = sum(nontarget_scores) / len(nontarget_scores)
        if mean_t <= mean_nt:
            problems.append(
                f"Score ordering WRONG: mean target score ({mean_t:.3f}) "
                f"<= mean non-target score ({mean_nt:.3f})"
            )

    return problems


def main():
    print("=" * 60)
    print("Self-test: output format + score ordering verification")
    print("=" * 60)

    # Ground truth dict
    gt = {s: 1 for s in TARGET_STEMS} | {s: 0 for s in NONTARGET_STEMS}
    all_stems = set(gt.keys())

    # Build temporary eval directory
    eval_dir = Path(tempfile.mkdtemp(prefix="sur_selftest_"))
    print(f"\nBuilding mini eval directory ({len(all_stems)} samples)...")
    try:
        for stem in all_stems:
            for ext in (".wav", ".png"):
                try:
                    src = find_file(stem, ext)
                    shutil.copy(src, eval_dir / (stem + ext))
                except FileNotFoundError:
                    pass  # some stems may only have one modality

        print(f"  {len(list(eval_dir.iterdir()))} files copied to {eval_dir}")

        overall_ok = True

        for modality, (script, out_name, _) in SCRIPTS.items():
            print(f"\n{'-'*60}")
            print(f"Running {script} ...")
            out_file = eval_dir / out_name

            result = subprocess.run(
                ["uv", "run", script,
                 "--eval-dir", str(eval_dir),
                 "--output",   str(out_file)],
                capture_output=True, text=True,
            )

            if result.returncode != 0:
                print(f"  FAIL SCRIPT FAILED (exit {result.returncode})")
                print(result.stderr[-1000:])
                overall_ok = False
                continue

            # Parse
            try:
                rows = parse_results(out_file)
            except ValueError as e:
                print(f"  FAIL PARSE ERROR: {e}")
                overall_ok = False
                continue

            # Check
            # For fusion, expected stems are .wav stems
            problems = check_results(rows, all_stems, gt, script)

            if problems:
                overall_ok = False
                print(f"  FAIL FAIL:")
                for p in problems:
                    print(f"      - {p}")
            else:
                target_scores    = [r[1] for r in rows if gt.get(r[0]) == 1]
                nontarget_scores = [r[1] for r in rows if gt.get(r[0]) == 0]
                mean_t  = sum(target_scores)  / len(target_scores)
                mean_nt = sum(nontarget_scores) / len(nontarget_scores)
                decisions_correct = sum(
                    1 for r in rows if r[2] == gt.get(r[0], -1)
                )
                print(f"  OK OK")
                print(f"      Scores: target mean: {mean_t:+.3f},  non-target mean: {mean_nt:+.3f}")
                print(f"      Decisions correct: {decisions_correct}/{len(rows)}")
                print(f"      Sample lines:")
                for r in sorted(rows, key=lambda x: -x[1])[:3]:
                    label = "target" if gt.get(r[0]) == 1 else "non-target"
                    print(f"        {r[0]}  {r[1]:+.4f}  {r[2]}  ({label})")

        print(f"\n{'=' * 60}")
        if overall_ok:
            print("OK ALL CHECKS PASSED, safe to submit")
        else:
            print("FAIL SOME CHECKS FAILED, fix before submission")
            sys.exit(1)

    finally:
        shutil.rmtree(eval_dir)


if __name__ == "__main__":
    main()
