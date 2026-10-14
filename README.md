# Classifying an accountant's inbox

Accountants receive client mail, mail from colleagues, and everything else in the same inbox. This project reads one email in the shape of [`data/sample_accountant_emails.json`](data/sample_accountant_emails.json) and returns two labels:

- **source:** `client`, `internal`, or `external` (marketing, newsletters, spam, and other mail that is not from a person)
- **topic:** `accounting`, `taxes`, `payroll`, `legal`, or `other`

The sample has 20 emails and no answer key. The labels further down are the spec. [`tests/test_classify.py`](tests/test_classify.py) locks them, so a rule change that flips one email is visible.

## Why rules

Twenty unlabeled examples are enough to write the decision down, and too few to train or honestly score a model. Every result includes a `reason` that names the rule which fired. The classifier uses the Python standard library only.

## Source

The sender is the participant with `type` `from`. The firm domain is `accountingfirm.com`. The first matching rule wins.

1. The sender's domain is the firm, so the email is `internal`. That covers Sarah Johnson's own replies, and firm addresses such as `billing@` and `support@`.
2. The sender is bulk or the text is spam, so the email is `external`. Bulk local-parts are `noreply`, `statements`, `updates`, `events`, and `winner`. Spam phrases, matched as written, are `lottery`, `you've won`, and `claim your prize`.
3. Anyone else is a `client`.

## Topic

Topic follows the content, separately from the source. A newsletter about tax law is `external` and `taxes`.

Each keyword scores 2 points in the subject and 1 point in the body and attachment names. A keyword counts once in each place, so writing "audit" twice does not raise the score. The highest score wins when it reaches 2. A score of 0 or 1 stays `other`, which is what keeps a passing mention of an audit from becoming an accounting email. Equal scores break toward the more specific topic: payroll, then legal, then taxes, then accounting.

| Topic | Keywords |
| --- | --- |
| payroll | payroll, salary, w-2, withholding |
| legal | lawsuit, attorney, contract dispute |
| taxes | tax, deduction, filing, irs, r&d credit |
| accounting | financial statement, invoice, audit, reconciliation, bookkeeping, depreciation, p&l, bank statement, card statement |

`card statement` sits beside `bank statement` so a credit-card statement is accounting. The sample contains no payroll or legal mail. Those keywords are covered by unit tests.

## Judgment calls

- Source follows the sender of this message. Sarah's reply on the Smith Manufacturing thread is internal, because she sent it.
- A person writing from outside the firm is a client. Robert Martinez at FreshBites is a client on first contact.
- The FreshBites note asks for financial statements and tax preparation, one keyword each. Neither score reaches 2, so the topic is `other`.
- The Green Energy thank-you mentions an audit in the body only. The topic is `other`.
- Bank and card statements come from `statements@`, so they are external, and the topic is accounting.

## Labels on the sample

| Email | Source | Topic |
| --- | --- | --- |
| Q4 financial statements, from Smith Manufacturing | client | accounting |
| Team meeting on tax season, from David | internal | taxes |
| Missing audit documents, from Lisa | internal | accounting |
| QuickBooks renewal | external | other |
| Lottery prize | external | other |
| Invoice from billing@ | internal | accounting |
| Tax-law newsletter | external | taxes |
| Year-end tax planning, from TechStart | client | taxes |
| Bank statement | external | accounting |
| Client feedback on Green Energy, from David | internal | other |
| Reply on the Q4 statements, from Sarah | internal | accounting |
| Reply on the team meeting, from Sarah | internal | taxes |
| Reply on the audit documents, from Sarah | internal | accounting |
| New inquiry from FreshBites | client | other |
| Reply on the tax planning, from Sarah | internal | taxes |
| CPA webinar | external | taxes |
| Client portal update, from IT | internal | other |
| Credit card statement | external | accounting |
| Reply on the client feedback, from Sarah | internal | other |
| Client portal maintenance, from support@ | internal | other |

## Run

From the repository root:

```bash
python3 -m classifier data/sample_accountant_emails.json
python3 -m unittest discover -s tests
```

The classifier prints a JSON list of `id`, `source`, `topic`, and `reason`.
