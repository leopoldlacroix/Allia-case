# Classifying an accountant's inbox

Accountants mix client mail, colleague mail, and everything else in one inbox. This take-home builds a **classifier** that, given one email in the sample JSON shape, returns:

- **source:** `client` | `internal` | `external` (marketing, newsletter, spam, non-person mail)
- **topic:** `accounting` | `taxes` | `payroll` | `legal` | `other`

**In scope:** classification methodology, readable code, low false positives (especially not calling a real person `external`).  
**Out of scope:** draft replies (no reply gold data; the brief grades classification).

Brief: [`docs/assignement.md`](docs/assignement.md). Sample: [`data/sample_accountant_emails.json`](data/sample_accountant_emails.json) (20 emails, no official answer key).

## Approach

1. **Start simple.** A deterministic **rules** method is the default baseline: transparent, fast, no API key, easy to audit via `reason` strings.
2. **Keep methods swappable.** Every method implements the same `Classifier` contract (`Email` → `Classification`). The registry + executor never care how labels were produced.
3. **Add LLMs without rewriting the pipeline.** `--method llm` talks to any OpenAI-compatible chat API **or** a TypeSafe/Jev/Tev1 **systemone** decision endpoint, selected by config.
4. **Treat labels as provisional.** [`data/eval_labels.json`](data/eval_labels.json) was hand-built (largely aligned with rules) so we can compare runs. It is **not** ground truth. The compare dashboard shows it as one more row next to model outputs.
5. **Optimize for inspection.** Save each run under `runs/results/`, then regenerate a side-by-side HTML page to spot model consensus vs provisional labels.

## Solution shape

```text
emails JSON  →  executor  →  Classifier.classify(email)  →  Classification
                               ↑
                    registry: "rules" | "llm"
```

Output type ([`classifier/types.py`](classifier/types.py)):

| Field | Role |
| --- | --- |
| `id` | Email id |
| `source` / `topic` | Closed label sets |
| `reason` | Short audit trail (which rule fired, or model rationale) |
| `method` | Registry name (`rules` / `llm`) |

Adding a method: implement `Classifier` under `classifier/methods/`, register it in [`classifier/registry.py`](classifier/registry.py).

## Methods

### rules (default)

Standard library only. Source and topic are independent.

**Source** ([`classifier/methods/rules/source.py`](classifier/methods/rules/source.py)) — From-address, first match wins; firm domain `accountingfirm.com` (`FIRM_DOMAIN`):

1. Firm domain → `internal`
2. Bulk local-part / spam phrase → `external`
3. Else → `client` (named person at another company, including prospects)

**Topic** ([`classifier/methods/rules/topic.py`](classifier/methods/rules/topic.py)) — Keyword scores (subject weight 2, body/attachments weight 1). Need score ≥ 2 or fall back to `other`. Ties prefer the earlier topic in the list.

### llm

Optional deps in [`requirements.txt`](requirements.txt). Shared email fields go through [`classifier/methods/llm/payload.py`](classifier/methods/llm/payload.py). Prompts live in [`classifier/methods/llm/prompts.py`](classifier/methods/llm/prompts.py) (chat) and choice criteria in [`systemone.py`](classifier/methods/llm/systemone.py).

| `LLM_API_MODE` | When | Mechanism |
| --- | --- | --- |
| `chat` (default) | OpenRouter, Ollama gemma/llama/qwen, … | LangChain `ChatOpenAI` + agent + `ToolStrategy` `ClassificationResult` (no helper tools). Needs `LLM_RECURSION_LIMIT` ≥ ~3 for model + structured tool. |
| `systemone` | Tev1 / Jev / TypeSafe decision APIs | One HTTP POST to `{LLM_BASE_URL}/systemone` with two `choice` questions (`source`, `topic`). Aliases: `tev`, `tev1`, `jev`, `typesafe`. |

**Config load order:** process env → repo `.env` → `--env-file` → CLI `--model` / `--api-mode` / `--base-url` / `--api-key`.

**Fail-safes:** missing key/model/packages, timeouts, rate limits, connection errors → fall back to **rules** per email (reason prefixed). Disable with `LLM_FALLBACK=0` or `--no-fallback`. `LLM_STRICT=1` fails on bad config instead of falling back.

On Python 3.14, a harmless upstream Pydantic/LangChain warning is filtered when loading chat mode.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U -r requirements.txt
cp .env.example .env        # optional default LLM profile
```

Local Ollama profiles for this machine live under [`runs/envs/`](runs/envs/) (see [`runs/README.md`](runs/README.md) for pull + suite commands).

## How to run

```bash
# baseline (JSON on stdout, progress on stderr)
python3 -m classifier data/sample_accountant_emails.json
python3 -m classifier --method rules data/sample_accountant_emails.json
python3 -m classifier --method llm --env-file runs/envs/tev1-4b.env --tag tev1-4b

# HTML review for one method
python3 -m classifier.review data/sample_accountant_emails.json \
  --method rules --labels data/eval_labels.json \
  --tag rules --out runs/reviews/rules.html --workers 4

# full local suite: rules + every runs/envs profile + compare page
WORKERS=4 ./scripts/run_ollama_suite.sh
LIMIT=3 WORKERS=4 ./scripts/run_ollama_suite.sh   # smoke (first N emails)

# regenerate compare after deleting/changing runs/results/*.json
python3 -m classifier.compare \
  --emails data/sample_accountant_emails.json \
  --labels data/eval_labels.json \
  --outputs-dir runs/results \
  --out runs/compare.html

# tests
python3 -m unittest discover -s tests
```

Useful knobs: `--workers N` (parallel emails), `--limit N`, `--outputs-dir PATH`, `--tag NAME`, `--no-save`.  
Full experiment CLI (suite env vars, single-profile examples, compare `--inputs`): **[`runs/README.md`](runs/README.md)**.

Each saved run is `runs/results/<timestamp>_<method>_<model>_<tag>.json`. The compare page shows agreement with provisional labels (not a grade), per-email **source** / **topic** columns, and a **check provisional?** badge when ≥2 runs agree against that row.

## Debug

1. Run and Debug → **Debug LLM: one email** ([`scripts/debug_one_email.py`](scripts/debug_one_email.py)), or `python scripts/debug_one_email.py <index-or-id>`.
2. Breakpoints: [`classifier/methods/llm/__init__.py`](classifier/methods/llm/__init__.py) (`_invoke_once`), chat [`agent.py`](classifier/methods/llm/agent.py), or systemone [`systemone.py`](classifier/methods/llm/systemone.py).
3. The debug script sets `LLM_FALLBACK=0` so failures surface.

## Layout

```text
classifier/
  types.py, registry.py, executor.py, cli_options.py
  review.py, compare.py, outputs.py, log.py
  methods/rules/          # source + topic heuristics
  methods/llm/            # chat agent, systemone, config, prompts, payload
runs/
  envs/                   # per-model .env profiles
  results/ reviews/       # artifacts (gitignored)
  README.md               # experiment commands
data/
  sample_accountant_emails.json
  eval_labels.json        # provisional only
scripts/
  run_ollama_suite.sh
  debug_one_email.py
tests/
docs/assignement.md
```
