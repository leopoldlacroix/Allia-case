"""Build a static HTML dashboard comparing saved runs under runs/results/.

Usage:
  python3 -m classifier.compare \\
    --emails data/sample_accountant_emails.json \\
    --labels data/eval_labels.json \\
    --outputs-dir runs/results \\
    --out runs/compare.html
"""

from __future__ import annotations

import argparse
import html
import sys
from collections import Counter
from pathlib import Path

from classifier.executor import load_emails, load_labels
from classifier.outputs import RunArtifact, load_runs
from classifier.types import Classification, Email, Source, Topic


def _escape(value: object) -> str:
    return html.escape(str(value), quote=True)


def _from_address(email: Email) -> str:
    for participant in email.get("participants", []):
        if participant.get("type", "").lower() == "from":
            return participant.get("email", "")
    return "(no sender)"


def _attachment_names(email: Email) -> str:
    names = [
        attachment.get("file_name", "")
        for attachment in email.get("attachments", [])
        if attachment.get("file_name")
    ]
    return ", ".join(names) if names else "—"


def _result_by_id(run: RunArtifact) -> dict[str, Classification]:
    return {result.id: result for result in run.results}


def _run_label(run: RunArtifact) -> str:
    settings = run.settings or {}
    bits = [run.method]
    if settings.get("api_mode"):
        bits.append(settings["api_mode"])
    if settings.get("model"):
        bits.append(settings["model"])
    if run.tag:
        bits.append(f"tag:{run.tag}")
    return " · ".join(bits)


def score_run(
    run: RunArtifact,
    labels: dict[str, tuple[Source, Topic]] | None,
) -> dict[str, float | int]:
    """Return match counts vs eval labels (0 compared → rates are 0)."""
    if not labels:
        return {
            "compared": 0,
            "both": 0,
            "source": 0,
            "topic": 0,
            "both_rate": 0.0,
            "source_rate": 0.0,
            "topic_rate": 0.0,
        }
    by_id = _result_by_id(run)
    compared = both = source_ok = topic_ok = 0
    for email_id, expected in labels.items():
        result = by_id.get(email_id)
        if result is None:
            continue
        compared += 1
        if result.source == expected[0]:
            source_ok += 1
        if result.topic == expected[1]:
            topic_ok += 1
        if (result.source, result.topic) == expected:
            both += 1
    return {
        "compared": compared,
        "both": both,
        "source": source_ok,
        "topic": topic_ok,
        "both_rate": (both / compared) if compared else 0.0,
        "source_rate": (source_ok / compared) if compared else 0.0,
        "topic_rate": (topic_ok / compared) if compared else 0.0,
    }


def consensus_vs_label(
    predictions: list[tuple[Source, Topic]],
    expected: tuple[Source, Topic] | None,
) -> tuple[Source, Topic] | None:
    """If ≥2 runs share a prediction that disagrees with eval, return that pair."""
    if expected is None or len(predictions) < 2:
        return None
    counts = Counter(predictions)
    if not counts:
        return None
    (top_pair, top_n) = counts.most_common(1)[0]
    if top_n >= 2 and top_pair != expected:
        return top_pair
    return None


def render_compare(
    emails: list[Email],
    runs: list[RunArtifact],
    labels: dict[str, tuple[Source, Topic]] | None,
    emails_path: str | Path,
) -> str:
    """Self-contained HTML: scores + per-email multi-run inspection."""
    scores = [score_run(run, labels) for run in runs]
    by_run = [_result_by_id(run) for run in runs]

    score_cards: list[str] = []
    for run, score in zip(runs, scores, strict=True):
        settings = run.settings or {}
        score_cards.append(
            f"""
<div class="score-card">
  <h2>{_escape(_run_label(run))}</h2>
  <p class="muted">{_escape(run.id)}</p>
  <p class="muted">{_escape(run.created_at)}
    · {_escape(settings.get("base_url", ""))}</p>
  <dl>
    <div><dt>Both</dt><dd><strong>{score["both"]}</strong>/{score["compared"]}
      ({score["both_rate"]:.0%})</dd></div>
    <div><dt>Source</dt><dd>{score["source"]}/{score["compared"]}
      ({score["source_rate"]:.0%})</dd></div>
    <div><dt>Topic</dt><dd>{score["topic"]}/{score["compared"]}
      ({score["topic_rate"]:.0%})</dd></div>
  </dl>
</div>
"""
        )

    email_cards: list[str] = []
    for index, email in enumerate(emails, start=1):
        email_id = email["id"]
        expected = labels.get(email_id) if labels else None
        preds: list[tuple[Source, Topic]] = []
        pred_rows: list[str] = []
        for run, mapping in zip(runs, by_run, strict=True):
            result = mapping.get(email_id)
            if result is None:
                pred_rows.append(
                    f"<tr><th>{_escape(_run_label(run))}</th>"
                    f"<td colspan='2'><em>missing</em></td></tr>"
                )
                continue
            preds.append((result.source, result.topic))
            agree = expected is not None and expected == (result.source, result.topic)
            cell_class = "ok" if agree else ("bad" if expected is not None else "")
            pred_rows.append(
                f"""
<tr class="{cell_class}">
  <th>{_escape(_run_label(run))}</th>
  <td><code>{_escape(result.source)}</code> /
      <code>{_escape(result.topic)}</code></td>
  <td class="reason">{_escape(result.reason)}</td>
</tr>
"""
            )

        suspect = consensus_vs_label(preds, expected)
        if expected is None:
            badge = '<span class="badge muted">no eval label</span>'
            expected_html = "<em>n/a</em>"
        elif suspect is not None:
            badge = '<span class="badge warn">label check?</span>'
            expected_html = f"{_escape(expected[0])} / {_escape(expected[1])}"
        elif all(
            mapping.get(email_id) is not None
            and (mapping[email_id].source, mapping[email_id].topic) == expected
            for mapping in by_run
        ):
            badge = '<span class="badge ok">all match</span>'
            expected_html = f"{_escape(expected[0])} / {_escape(expected[1])}"
        else:
            badge = '<span class="badge bad">disagreement</span>'
            expected_html = f"{_escape(expected[0])} / {_escape(expected[1])}"

        suspect_note = ""
        if suspect is not None:
            suspect_note = (
                f'<p class="suspect">Consensus <code>{_escape(suspect[0])}</code> / '
                f"<code>{_escape(suspect[1])}</code> disagrees with eval — "
                "inspect the email and consider fixing the label.</p>"
            )

        email_cards.append(
            f"""
<article class="card">
  <header>
    <span class="index">#{index}</span>
    {badge}
    <h2>{_escape(email.get("subject", "(no subject)"))}</h2>
  </header>
  <p class="meta"><strong>From:</strong> {_escape(_from_address(email))} ·
    <strong>Attachments:</strong> {_escape(_attachment_names(email))}</p>
  <p class="meta"><strong>Eval label:</strong> {expected_html}</p>
  {suspect_note}
  <table class="preds">
    <thead><tr><th>Run</th><th>Prediction</th><th>Reason</th></tr></thead>
    <tbody>
      {"".join(pred_rows)}
    </tbody>
  </table>
  <details>
    <summary>Body</summary>
    <pre>{_escape(email.get("body", ""))}</pre>
  </details>
</article>
"""
        )

    summary = (
        f"{len(emails)} emails · {len(runs)} runs · "
        f"source <code>{_escape(emails_path)}</code>"
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Classification compare</title>
  <style>
    :root {{
      --ink: #1a1a1a;
      --muted: #5c5c5c;
      --line: #e2e2e2;
      --bg: #f7f5f1;
      --card: #fff;
      --ok: #1f6b3a;
      --bad: #9b1c1c;
      --warn: #8a5a00;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font: 16px/1.45 Georgia, "Times New Roman", serif;
      color: var(--ink);
      background: var(--bg);
    }}
    main {{ max-width: 58rem; margin: 0 auto; padding: 2rem 1.25rem 4rem; }}
    h1 {{ font-size: 1.75rem; margin: 0 0 0.35rem; }}
    .summary {{ color: var(--muted); margin-bottom: 1.25rem; }}
    .scores {{
      display: grid;
      gap: 0.75rem;
      grid-template-columns: repeat(auto-fit, minmax(14rem, 1fr));
      margin-bottom: 1.75rem;
    }}
    .score-card {{
      background: var(--card);
      border: 1px solid var(--line);
      padding: 0.9rem 1rem;
    }}
    .score-card h2 {{ font-size: 1rem; margin: 0 0 0.35rem; }}
    .score-card dl {{
      display: grid;
      gap: 0.25rem 0.75rem;
      margin: 0.6rem 0 0;
      grid-template-columns: auto 1fr;
    }}
    .score-card dl > div {{ display: contents; }}
    .score-card dt {{ color: var(--muted); }}
    .score-card dd {{ margin: 0; }}
    .muted {{ color: var(--muted); font-size: 0.9rem; margin: 0.15rem 0; }}
    .card {{
      background: var(--card);
      border: 1px solid var(--line);
      padding: 1.1rem 1.25rem 1.25rem;
      margin-bottom: 1rem;
    }}
    .card header {{ display: flex; flex-wrap: wrap; gap: 0.5rem 0.75rem; align-items: baseline; }}
    .card h2 {{ flex: 1 1 100%; font-size: 1.15rem; margin: 0.15rem 0 0; }}
    .index {{ color: var(--muted); font-variant-numeric: tabular-nums; }}
    .meta {{ margin: 0.5rem 0; color: var(--muted); }}
    .suspect {{
      color: var(--warn);
      border-left: 3px solid var(--warn);
      padding-left: 0.65rem;
      margin: 0.5rem 0 0.75rem;
    }}
    table.preds {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.95rem;
      margin: 0.5rem 0 0.75rem;
    }}
    table.preds th, table.preds td {{
      border-top: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
      padding: 0.45rem 0.4rem;
    }}
    table.preds th {{ font-weight: 600; white-space: nowrap; }}
    table.preds tr.ok td {{ color: var(--ok); }}
    table.preds tr.bad td {{ color: var(--bad); }}
    table.preds .reason {{ color: var(--muted); font-size: 0.9rem; }}
    code {{ font-family: ui-monospace, Menlo, Consolas, monospace; font-size: 0.9em; }}
    .badge {{
      font-family: ui-sans-serif, system-ui, sans-serif;
      font-size: 0.75rem;
      letter-spacing: 0.02em;
      text-transform: uppercase;
      border: 1px solid var(--line);
      padding: 0.15rem 0.45rem;
    }}
    .badge.ok {{ color: var(--ok); border-color: var(--ok); }}
    .badge.bad {{ color: var(--bad); border-color: var(--bad); }}
    .badge.warn {{ color: var(--warn); border-color: var(--warn); }}
    .badge.muted {{ color: var(--muted); }}
    details summary {{ cursor: pointer; color: var(--muted); }}
    pre {{
      white-space: pre-wrap;
      font: 0.92rem/1.4 ui-monospace, Menlo, Consolas, monospace;
      background: #f0eee9;
      padding: 0.75rem;
      margin: 0.5rem 0 0;
      overflow-x: auto;
    }}
  </style>
</head>
<body>
  <main>
    <h1>Classification compare</h1>
    <p class="summary">{summary}</p>
    <section class="scores">{"".join(score_cards)}</section>
    {"".join(email_cards)}
  </main>
</body>
</html>
"""


def write_compare(
    emails_path: str | Path,
    out_path: str | Path,
    *,
    labels_path: str | Path | None = None,
    outputs_dir: str | Path = Path("runs/results"),
    inputs: list[str | Path] | None = None,
) -> Path:
    emails = load_emails(emails_path)
    runs = load_runs(inputs, outputs_dir=outputs_dir)
    if not runs:
        raise ValueError(
            f"no run artifacts found under {outputs_dir} "
            "(run classifier/review with saving enabled first)"
        )
    labels = load_labels(labels_path) if labels_path else None
    html_doc = render_compare(emails, runs, labels, emails_path)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html_doc, encoding="utf-8")
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare saved classification runs in a static HTML dashboard.",
    )
    parser.add_argument(
        "--emails",
        type=Path,
        default=Path("data/sample_accountant_emails.json"),
        help="Emails JSON used for body/from inspection.",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("data/eval_labels.json"),
        help="Eval labels JSON (default: data/eval_labels.json).",
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=Path("runs/results"),
        help="Directory of run JSON files (default: runs/results).",
    )
    parser.add_argument(
        "--inputs",
        type=Path,
        nargs="*",
        default=None,
        help="Specific run JSON files (default: all under --outputs-dir).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("runs/compare.html"),
        help="Output HTML path (default: runs/compare.html).",
    )
    parser.add_argument(
        "--no-labels",
        action="store_true",
        help="Skip eval labels / scores.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        out = write_compare(
            args.emails,
            args.out,
            labels_path=None if args.no_labels else args.labels,
            outputs_dir=args.outputs_dir,
            inputs=args.inputs,
        )
    except (OSError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
