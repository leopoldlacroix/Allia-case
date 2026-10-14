#!/usr/bin/env bash
# Classify the sample with rules + every profile in runs/envs/, then compare.
#
# Env knobs:
#   RESULTS=…     where JSON artifacts go (default runs/results)
#   WORKERS=4     parallel emails per model (Ollama may still GPU-serialize)
#   LIMIT=5       only first N emails (faster smoke)
#   EMAILS=… LABELS=… REVIEWS=… COMPARE_OUT=…
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
WORKERS="${WORKERS:-4}"
LIMIT="${LIMIT:-}"

mkdir -p "$RESULTS" "$REVIEWS"

extra_args=(--workers "$WORKERS" --outputs-dir "$RESULTS")
if [[ -n "$LIMIT" ]]; then
  extra_args+=(--limit "$LIMIT")
fi

echo "==> rules (workers=$WORKERS${LIMIT:+ limit=$LIMIT} → $RESULTS)"
"$PYTHON" -m classifier.review "$EMAILS" \
  --method rules \
  --labels "$LABELS" \
  --tag rules \
  --out "$REVIEWS/rules.html" \
  "${extra_args[@]}"

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
    --out "$REVIEWS/${tag}.html" \
    "${extra_args[@]}"
done

echo "==> compare → $COMPARE_OUT"
"$PYTHON" -m classifier.compare \
  --emails "$EMAILS" \
  --labels "$LABELS" \
  --outputs-dir "$RESULTS" \
  --out "$COMPARE_OUT"

echo "done. open $COMPARE_OUT"
