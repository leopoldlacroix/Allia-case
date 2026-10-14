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
  methods/
    rules/                 # current method: address + keyword rules
      source.py
      topic.py
data/
  sample_accountant_emails.json   # input format (untouched)
  eval_labels.json                # provisional gold labels by id
tests/
docs/assignement.md
```

## Architecture

```text
emails JSON  →  executor  →  Classifier.classify(email)  →  Classification
                               ↑
                    registry: "rules" (later: "llm", "ml")
```

To add a method later: implement `Classifier` under `classifier/methods/`, register it in `registry.py`. The executor and CLI stay the same. Point the CLI at any dataset path (sample or synthetic) that uses `{"emails": [...]}`.

## Why rules first

Twenty unlabeled examples are enough to write the decision down, and too few to train or honestly score a model. Every result includes a `reason` that names the rule which fired. Standard library only.

## Source rules

The sender is the participant with `type` `from`. The firm domain is `accountingfirm.com`. First match wins.

1. Sender domain is the firm → `internal` (includes Sarah's replies, `billing@`, `support@`).
2. Bulk local-part (`noreply`, `statements`, `updates`, `events`, `winner`) or spam phrase (`lottery`, `you've won`, `claim your prize`) → `external`.
3. Anyone else → `client`.

## Topic rules

Topic follows content, separately from source. A tax newsletter is `external` and `taxes`.

Each keyword scores 2 in the subject and 1 in the body and attachment names (once per field). Score must reach 2; otherwise `other`. Tie-break: payroll, legal, taxes, accounting.

| Topic | Keywords |
| --- | --- |
| payroll | payroll, salary, w-2, withholding |
| legal | lawsuit, attorney, contract dispute |
| taxes | tax, deduction, filing, irs, r&d credit |
| accounting | financial statement, invoice, audit, reconciliation, bookkeeping, depreciation, p&l, bank statement, card statement |

## Judgment calls

- Source follows the sender of this message. Sarah's reply on a client thread is `internal`.
- A person outside the firm is a `client`, including first-contact prospects (FreshBites).
- FreshBites mixes tax and accounting asks below the threshold → topic `other`.
- Green Energy thank-you mentions audit only in the body → topic `other`.

## Run

```bash
python3 -m classifier data/sample_accountant_emails.json
python3 -m classifier --method rules data/sample_accountant_emails.json
python3 -m classifier path/to/synthetic_emails.json
python3 -m unittest discover -s tests
```

Output is a JSON list of `id`, `source`, `topic`, `reason`, and `method`.
