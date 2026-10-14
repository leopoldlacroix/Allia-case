#!/usr/bin/env bash
# Classify the sample with rules + every profile in runs/envs/, then compare.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="$(command -v python3)"
fi

EMAILS="${EMAILS:-data/sample_accountant_emails.json}"
LABELS="${LABELS:-data/eval_labels.json}"
RESULTS="${RESULTS:-runs/results}"
REVIEWS="${REVIEWS:-runs/reviews}"
COMPARE_OUT="${COMPARE_OUT:-runs/compare.html}"

mkdir -p "$RESULTS" "$REVIEWS"

echo "==> rules"
"$PYTHON" -m classifier.review "$EMAILS" \
  --method rules \
  --labels "$LABELS" \
  --tag rules \
  --outputs-dir "$RESULTS" \
  --out "$REVIEWS/rules.html"

shopt -s nullglob
profiles=(runs/envs/*.env)
if [[ ${#profiles[@]} -eq 0 ]]; then
  echo "error: no env profiles under runs/envs/" >&2
  exit 1
fi

for env_file in "${profiles[@]}"; do
  tag="$(basename "$env_file" .env)"
  echo "==> llm profile $tag ($env_file)"
  "$PYTHON" -m classifier.review "$EMAILS" \
    --method llm \
    --env-file "$env_file" \
    --tag "$tag" \
    --labels "$LABELS" \
    --outputs-dir "$RESULTS" \
    --out "$REVIEWS/${tag}.html"
done

echo "==> compare → $COMPARE_OUT"
"$PYTHON" -m classifier.compare \
  --emails "$EMAILS" \
  --labels "$LABELS" \
  --outputs-dir "$RESULTS" \
  --out "$COMPARE_OUT"

echo "done. open $COMPARE_OUT"
