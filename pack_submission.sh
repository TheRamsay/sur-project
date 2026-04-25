#!/usr/bin/env bash
# Build the SUR 2025/2026 submission ZIP.
#
# Layout produced (per assignment.txt):
#   <login>.zip
#   ├── dokumentace.pdf
#   ├── <result_files>.txt        (≤ 6, meaningful names)
#   └── SRC/
#       ├── predict_audio.py
#       ├── predict_image.py
#       ├── predict_fusion.py
#       ├── self_test.py
#       ├── pyproject.toml
#       ├── uv.lock
#       ├── README.md
#       └── src/                  (the Python package)
#
# Usage:
#   ./pack_submission.sh                       # uses defaults
#   ./pack_submission.sh --login xhumld00 --pdf docs/dokumentace.pdf

set -euo pipefail

LOGIN="xhumld00"
PDF="docs/dokumentace.pdf"
RESULTS_DIR="results"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --login) LOGIN="$2"; shift 2 ;;
    --pdf) PDF="$2"; shift 2 ;;
    --results-dir) RESULTS_DIR="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

ZIP="${LOGIN}.zip"
STAGE="$(mktemp -d -t sur_pack_XXXXXX)"
trap 'rm -rf "$STAGE"' EXIT

echo "Staging in $STAGE"

# --- documentation ---------------------------------------------------------
if [[ ! -f "$PDF" ]]; then
  echo "ERROR: $PDF not found. Render it first:" >&2
  echo "  pandoc docs/dokumentace.md -o docs/dokumentace.pdf --pdf-engine=xelatex -V geometry:margin=2cm" >&2
  exit 1
fi
cp "$PDF" "$STAGE/dokumentace.pdf"

# --- result files ----------------------------------------------------------
shopt -s nullglob
RESULTS=("$RESULTS_DIR"/*.txt)
if [[ ${#RESULTS[@]} -eq 0 ]]; then
  echo "ERROR: no result files found in $RESULTS_DIR/" >&2
  echo "Run predict_*.py on the eval data first." >&2
  exit 1
fi
if [[ ${#RESULTS[@]} -gt 6 ]]; then
  echo "ERROR: ${#RESULTS[@]} result files in $RESULTS_DIR/, but assignment limits to 6." >&2
  echo "Found: ${RESULTS[*]}" >&2
  exit 1
fi
for f in "${RESULTS[@]}"; do cp "$f" "$STAGE/$(basename "$f")"; done

# Sanity-check format: each line "stem score hard_decision"
for f in "${RESULTS[@]}"; do
  awk 'NF != 3 { print FILENAME":"NR": expected 3 fields, got "NF; exit 1 }
       $3 != "0" && $3 != "1" { print FILENAME":"NR": hard decision not 0/1: "$3; exit 1 }' "$f" \
       || { echo "ERROR: malformed result file $f" >&2; exit 1; }
done

# --- SRC/ ------------------------------------------------------------------
mkdir -p "$STAGE/SRC"
cp predict_audio.py predict_image.py predict_fusion.py self_test.py \
   pyproject.toml uv.lock README.md "$STAGE/SRC/"
cp -R src "$STAGE/SRC/src"
find "$STAGE/SRC/src" -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true

# --- build ZIP -------------------------------------------------------------
rm -f "$ZIP"
( cd "$STAGE" && zip -r -q "$OLDPWD/$ZIP" . )

echo
echo "✓ Built $ZIP ($(du -h "$ZIP" | cut -f1))"
echo "  Contents:"
unzip -l "$ZIP" | tail -n +4 | head -n -2 | awk '{print "    "$NF}'
echo
echo "Next: upload $ZIP to IS by 2026-05-04 23:59."
