# Local experiment runs

```text
runs/
  envs/          # one profile env per model (safe to commit for local Ollama)
  results/       # saved classification JSON (gitignored)
  reviews/       # optional per-run HTML (gitignored)
  compare.html   # regenerated from results/ (gitignored)
```

## Profiles (`runs/envs/`)

| File | Mode | Model |
| --- | --- | --- |
| `tev1-0.8b.env` | systemone | `tev1:0.8b` |
| `tev1-4b.env` | systemone | `tev1:4b` |
| `qwen3.5-0.8b.env` | chat | `qwen3.5:0.8b` |
| `llama3.2-3b.env` | chat | `Llama3.2:3B` |
| `gemma4-e4b.env` | chat | `gemma4:e4b` |

Pull models if needed:

```bash
for m in tev1:0.8b tev1:4b qwen3.5:0.8b Llama3.2:3B gemma4:e4b; do
  ollama pull "$m"
done
```

## Full suite (rules + every env profile + compare)

From the repo root (uses `.venv` when present):

```bash
# all 20 emails, 4 parallel workers (default WORKERS=4 in the script)
WORKERS=4 ./scripts/run_ollama_suite.sh

# smoke: first N emails only
LIMIT=3 WORKERS=4 ./scripts/run_ollama_suite.sh

# write artifacts somewhere else
RESULTS=/tmp/allia-runs COMPARE_OUT=/tmp/compare.html ./scripts/run_ollama_suite.sh
```

Suite env knobs:

| Variable | Default | Meaning |
| --- | --- | --- |
| `WORKERS` | `4` | Parallel emails per run |
| `LIMIT` | (unset) | Only first N emails; omit for the full sample |
| `RESULTS` | `runs/results` | Where run JSON is saved |
| `REVIEWS` | `runs/reviews` | Per-run HTML |
| `COMPARE_OUT` | `runs/compare.html` | Compare dashboard path |
| `EMAILS` | `data/sample_accountant_emails.json` | Input emails |
| `LABELS` | `data/eval_labels.json` | Provisional labels (table row in compare) |

## Single run (one method / one profile)

```bash
# rules
.venv/bin/python -m classifier.review data/sample_accountant_emails.json \
  --method rules --tag rules \
  --labels data/eval_labels.json \
  --outputs-dir runs/results \
  --out runs/reviews/rules.html \
  --workers 4

# one LLM profile
.venv/bin/python -m classifier.review data/sample_accountant_emails.json \
  --method llm --env-file runs/envs/tev1-4b.env --tag tev1-4b \
  --labels data/eval_labels.json \
  --outputs-dir runs/results \
  --out runs/reviews/tev1-4b.html \
  --workers 4

# JSON only on stdout (also saves under --outputs-dir unless --no-save)
.venv/bin/python -m classifier data/sample_accountant_emails.json \
  --method llm --env-file runs/envs/gemma4-e4b.env --tag gemma4-e4b \
  --outputs-dir runs/results --workers 4
```

Useful flags: `--limit N`, `--no-save`, `--model` / `--api-mode` / `--base-url` / `--api-key` (override the env file).

## Regenerate compare HTML

After deleting old JSON under `runs/results/`, or after the suite finishes, rebuild the dashboard from whatever files remain:

```bash
.venv/bin/python -m classifier.compare \
  --emails data/sample_accountant_emails.json \
  --labels data/eval_labels.json \
  --outputs-dir runs/results \
  --out runs/compare.html
```

Compare a subset of artifacts:

```bash
.venv/bin/python -m classifier.compare \
  --emails data/sample_accountant_emails.json \
  --labels data/eval_labels.json \
  --inputs runs/results/20261014T151331Z_llm_tev1-4b_tev1-4b.json \
           runs/results/20261014T144855Z_rules_rules.json \
  --out runs/compare.html
```

Omit provisional labels entirely: add `--no-labels`.

Open `runs/compare.html` in a browser. Provisional labels appear as one row among the runs (source / topic columns), not as ground truth.

## Housekeeping

```bash
# clear saved runs (then regenerate compare, or re-run the suite)
rm -f runs/results/*.json runs/reviews/*.html runs/compare.html
```

**Speed note:** `--workers` / `WORKERS` overlaps requests. On a single local Ollama GPU they often still queue, so wall-clock may not drop linearly; systemone and remote APIs usually benefit more. Use `LIMIT` only for quick smoke loops.
