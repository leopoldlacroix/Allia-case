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
  methods/
    rules/                 # address + keyword rules
    llm/                   # OpenRouter + LangChain agent (optional deps)
data/
  sample_accountant_emails.json
  eval_labels.json
  review.html              # generated locally (gitignored)
tests/
docs/assignement.md
```

## Architecture

```text
emails JSON  →  executor  →  Classifier.classify(email)  →  Classification
                               ↑
                    registry: "rules" | "llm"
```

`--method llm` runs a LangChain agent on OpenRouter: helper tools (sender / bulk / topic scores) plus a final `ToolStrategy` schema (`source`, `topic`, `reason`). Native JSON `response_format` is not combined with tools (fragile on free models).

## Methods

### rules (default)

Deterministic, standard library only. See source/topic sections below.

### llm

Requires [`requirements.txt`](requirements.txt) and a `.env` (see [`.env.example`](.env.example)):

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U -r requirements.txt
# set OPENROUTER_API_KEY and OPENROUTER_MODEL in .env (free slug recommended)
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

## Review in the browser

```bash
python3 -m classifier.review data/sample_accountant_emails.json \
  --method rules \
  --labels data/eval_labels.json \
  --out data/review.html

python3 -m classifier.review data/sample_accountant_emails.json \
  --method llm \
  --labels data/eval_labels.json \
  --out data/review.html
```

Open `data/review.html` in a browser. The file is gitignored; regenerate anytime.
