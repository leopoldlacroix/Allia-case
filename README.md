# Classifying an accountant's inbox

Accountants receive client mail, mail from colleagues, and everything else in the same inbox. This project reads one email in the shape of [`data/sample_accountant_emails.json`](data/sample_accountant_emails.json) and returns two labels:

- **source:** `client`, `internal`, or `external` (marketing, newsletters, spam, and other mail that is not from a person)
- **topic:** `accounting`, `taxes`, `payroll`, `legal`, or `other`

The assignment asks for classification only. Draft replies are out of scope here (no reply gold data, and the brief grades classification methodology).

The sample has 20 emails and no answer key. Provisional labels live in [`data/eval_labels.json`](data/eval_labels.json) so the original sample stays intact. Edit those labels freely; do not rewrite the sample to fit a method.

## Layout

```text
classifier/
  types.py                 # Email in, Classification out, Classifier protocol
  registry.py              # method name → implementation
  executor.py              # load any emails JSON + run a method
  review.py                # static HTML review page for eyeballing results
  outputs.py               # save run JSON under runs/results/
  compare.py               # side-by-side dashboard from saved runs
  methods/
    rules/                 # address + keyword rules
    llm/                   # chat or systemone (optional deps)
runs/
  envs/                    # one .env profile per model
  results/                 # saved run JSON (gitignored)
  reviews/                 # optional per-run HTML (gitignored)
  compare.html
data/
  sample_accountant_emails.json
  eval_labels.json
tests/
docs/assignement.md
```

## Architecture

```text
emails JSON  →  executor  →  Classifier.classify(email)  →  Classification
                               ↑
                    registry: "rules" | "llm"
```

`--method llm` supports two API modes via `LLM_API_MODE`:

| Mode | When to use | How it works |
| --- | --- | --- |
| `chat` (default) | OpenAI-compatible chat (OpenRouter, Ollama gemma/llama, …) | LangChain `ChatOpenAI` + `ToolStrategy` `ClassificationResult` |
| `systemone` | TypeSafe / Jev / Tev1 decision models | One POST to `{LLM_BASE_URL}/systemone` with two `choice` questions (source, topic) |

| Variable | Meaning |
| --- | --- |
| `LLM_API_MODE` | `chat` or `systemone` (aliases: `tev1`, `jev`, `typesafe`) |
| `LLM_API_KEY` | Provider API key (`ollama` is fine for local Ollama) |
| `LLM_MODEL` | Model id |
| `LLM_BASE_URL` | API root (default `https://openrouter.ai/api/v1`) |

For Tev1 on Ollama: `LLM_API_MODE=systemone`, `LLM_MODEL=tev1`, `LLM_BASE_URL=http://localhost:11434/v1`.

### Profiles and CLI overrides

Prefer a profile under [`runs/envs/`](runs/envs/) instead of editing `.env` between runs. Load order: process env → repo `.env` → `--env-file` → `--model` / `--api-mode` / `--base-url` / `--api-key`.

```bash
python3 -m classifier.review data/sample_accountant_emails.json \
  --method llm --env-file runs/envs/tev1-0.8b.env --tag tev1-0.8b \
  --labels data/eval_labels.json --out runs/reviews/tev1-0.8b.html

# or run rules + every Ollama profile, then compare:
./scripts/run_ollama_suite.sh
LIMIT=3 WORKERS=4 ./scripts/run_ollama_suite.sh   # smoke
RESULTS=/tmp/my-runs ./scripts/run_ollama_suite.sh  # custom folder
```

Each run writes `<outputs-dir>/<timestamp>_<method>_<model>_<tag>.json` (default `runs/results/`; any `--outputs-dir` works). Use `--no-save` to skip. `--workers N` overlaps emails; `--limit N` classifies only the first N. See [`runs/README.md`](runs/README.md).

## Methods

### rules (default)

Deterministic, standard library only. See source/topic sections below.

### llm

Requires [`requirements.txt`](requirements.txt) and a `.env` (see [`.env.example`](.env.example)):

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U -r requirements.txt
# set LLM_API_KEY, LLM_MODEL, LLM_BASE_URL in .env
```

On Python 3.14, LangChain may still ship a harmless Pydantic v1 compatibility warning upstream; this project filters it when loading the llm method.

**Fail-safes:** missing key/model/packages, timeouts, rate limits, and connection errors fall back to **rules** per email (reason starts with `llm unavailable …; used rules:`). Set `LLM_FALLBACK=0` or pass `--no-fallback` to fail fast while debugging. `LLM_STRICT=1` raises on bad config instead of falling back.

## Source rules

The sender is the participant with `type` `from`. The firm domain is `accountingfirm.com` (override with `FIRM_DOMAIN`). First match wins.

1. Sender domain is the firm → `internal`.
2. Bulk local-part or spam phrase → `external`.
3. Anyone else → `client`.

## Topic rules

Topic follows content, separately from source. Keyword score must reach 2 (subject weight 2, body/attachments weight 1); otherwise `other`.

## Run

```bash
python3 -m classifier data/sample_accountant_emails.json
python3 -m classifier --method rules data/sample_accountant_emails.json
python3 -m classifier --method llm data/sample_accountant_emails.json
python3 -m classifier --method llm --no-fallback data/sample_accountant_emails.json
python3 -m unittest discover -s tests
```

Progress lines go to stderr (`[classifier] 3/20 …`, `[llm] invoke …`). JSON results stay on stdout. Free models can take many seconds per email — that is normal, not a freeze.

## Debug in Cursor / VS Code

1. Open the Run and Debug view (sidebar play icon, or `Ctrl+Shift+D` / `Cmd+Shift+D`).
2. Choose **Debug LLM: one email** (uses [scripts/debug_one_email.py](scripts/debug_one_email.py) on the first sample email).
3. Set breakpoints in [`classifier/methods/llm/__init__.py`](classifier/methods/llm/__init__.py) (`_invoke_once`) or [`classifier/methods/llm/agent.py`](classifier/methods/llm/agent.py) (`extract_structured_result`).
4. Press F5. Inspect `state`, `structured_response`, and `messages` in the Variables panel.

Change the launch `args` (or run `python scripts/debug_one_email.py <index-or-id>`) to pick another email. That script sets `LLM_FALLBACK=0` so failures surface instead of silently using rules.

## Review in the browser

```bash
python3 -m classifier.review data/sample_accountant_emails.json \
  --method rules \
  --labels data/eval_labels.json \
  --out data/review.html \
  --tag rules

python3 -m classifier.review data/sample_accountant_emails.json \
  --method llm \
  --labels data/eval_labels.json \
  --out data/review.html
```

Open `data/review.html` in a browser. The file is gitignored; regenerate anytime.

## Compare runs (check labels)

After several runs (rules, chat, tev, …), build a dashboard with overall scores and a per-email table. When ≥2 runs agree with each other but disagree with eval, the page shows a **label check?** badge so you can inspect the body and fix [`data/eval_labels.json`](data/eval_labels.json) if needed.

```bash
python3 -m classifier.compare \
  --emails data/sample_accountant_emails.json \
  --labels data/eval_labels.json \
  --outputs-dir runs/results \
  --out runs/compare.html
```

Open `runs/compare.html`. Pass `--inputs path1.json path2.json` to compare a subset of `runs/results/`.
