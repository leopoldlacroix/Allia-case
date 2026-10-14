# Local experiment runs

```text
runs/
  envs/       # one profile env per model (safe to commit for local Ollama)
  results/    # saved classification JSON (gitignored)
  reviews/    # optional per-run HTML (gitignored)
  compare.html
```

Profiles currently match your Ollama pulls:

| File | Mode | Model |
| --- | --- | --- |
| `tev1-0.8b.env` | systemone | `tev1:0.8b` |
| `tev1-4b.env` | systemone | `tev1:4b` |
| `qwen3.5-0.8b.env` | chat | `qwen3.5:0.8b` |
| `llama3.2-3b.env` | chat | `Llama3.2:3B` |
| `gemma4-e4b.env` | chat | `gemma4:e4b` |

Run all of them (plus rules) and build the compare page:

```bash
./scripts/run_ollama_suite.sh

# faster smoke (first 3 emails, 4 parallel workers)
LIMIT=3 WORKERS=4 ./scripts/run_ollama_suite.sh

# custom results folder
RESULTS=/tmp/allia-runs ./scripts/run_ollama_suite.sh
```

Results can live anywhere — pass `--outputs-dir` to review/classifier, and the same path to `classifier.compare`. Default is `runs/results`.

**Speed note:** `--workers N` overlaps email requests. On a single local Ollama GPU that often still queues, so wall-clock may not drop linearly; systemone / remote APIs benefit more. Use `LIMIT` for quick label-check loops.
