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
```
