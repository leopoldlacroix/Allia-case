"""Build a static HTML page to eyeball classifications against the emails.

Usage:
  python3 -m classifier.review data/sample_accountant_emails.json \\
    --method rules --labels data/eval_labels.json --out data/review.html
"""

from __future__ import annotations

import argparse
import html
import sys
from pathlib import Path

from classifier.cli_options import (
    add_llm_config_args,
    add_run_artifact_args,
    overrides_from_args,
)
from classifier.executor import load_emails, load_labels, run
from classifier.registry import available_methods
from classifier.types import Classification, Email, Source, Topic


def _from_address(email: Email) -> str:
    for participant in email.get("participants", []):
        if participant.get("type", "").lower() == "from":
            return participant.get("email", "")
    return "(no sender)"


def _escape(value: object) -> str:
    return html.escape(str(value), quote=True)


def render_review(
    emails: list[Email],
    results: list[Classification],
    labels: dict[str, tuple[Source, Topic]] | None,
    method: str,
    emails_path: str | Path,
) -> str:
    """Return a self-contained HTML document (no external assets)."""
    by_id = {result.id: result for result in results}
    rows: list[str] = []
    matches = 0
    compared = 0

    for index, email in enumerate(emails, start=1):
        result = by_id[email["id"]]
        expected = labels.get(email["id"]) if labels else None
        agree = expected is not None and expected == (result.source, result.topic)
        if expected is not None:
            compared += 1
            if agree:
                matches += 1

        if expected is None:
            badge = '<span class="badge muted">no eval label</span>'
            expected_html = "<em>n/a</em>"
        elif agree:
            badge = '<span class="badge ok">match</span>'
            expected_html = f"{_escape(expected[0])} / {_escape(expected[1])}"
        else:
            badge = '<span class="badge bad">mismatch</span>'
            expected_html = f"{_escape(expected[0])} / {_escape(expected[1])}"

        rows.append(
            f"""
<article class="card">
  <header>
    <span class="index">#{index}</span>
    {badge}
    <h2>{_escape(email.get("subject", "(no subject)"))}</h2>
  </header>
  <p class="meta"><strong>From:</strong> {_escape(_from_address(email))}</p>
  <dl class="labels">
    <div><dt>Predicted</dt><dd><code>{_escape(result.source)}</code> /
      <code>{_escape(result.topic)}</code></dd></div>
    <div><dt>Eval label</dt><dd>{expected_html}</dd></div>
    <div class="reason"><dt>Reason</dt><dd>{_escape(result.reason)}</dd></div>
  </dl>
  <details>
    <summary>Body</summary>
    <pre>{_escape(email.get("body", ""))}</pre>
  </details>
</article>
"""
        )

    summary = (
        f"{len(emails)} emails · method <code>{_escape(method)}</code> · "
        f"source <code>{_escape(emails_path)}</code>"
    )
    if labels is not None:
        summary += f" · eval match {matches}/{compared}"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Email classification review</title>
  <style>
    :root {{
      --ink: #1a1a1a;
      --muted: #5c5c5c;
      --line: #e2e2e2;
      --bg: #f7f5f1;
      --card: #fff;
      --ok: #1f6b3a;
      --bad: #9b1c1c;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font: 16px/1.45 Georgia, "Times New Roman", serif;
      color: var(--ink);
      background: var(--bg);
    }}
    main {{ max-width: 52rem; margin: 0 auto; padding: 2rem 1.25rem 4rem; }}
    h1 {{ font-size: 1.75rem; margin: 0 0 0.35rem; }}
    .summary {{ color: var(--muted); margin-bottom: 1.75rem; }}
    .card {{
      background: var(--card);
      border: 1px solid var(--line);
      padding: 1.1rem 1.25rem 1.25rem;
      margin-bottom: 1rem;
    }}
    .card header {{ display: flex; flex-wrap: wrap; gap: 0.5rem 0.75rem; align-items: baseline; }}
    .card h2 {{ flex: 1 1 100%; font-size: 1.15rem; margin: 0.15rem 0 0; }}
    .index {{ color: var(--muted); font-variant-numeric: tabular-nums; }}
    .meta {{ margin: 0.6rem 0 0.8rem; color: var(--muted); }}
    .labels {{
      display: grid;
      gap: 0.45rem 1rem;
      margin: 0 0 0.85rem;
      grid-template-columns: auto 1fr;
    }}
    .labels > div {{ display: contents; }}
    .labels dt {{ color: var(--muted); }}
    .labels dd {{ margin: 0; }}
    .labels .reason dd {{ color: var(--muted); font-size: 0.95rem; }}
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
    <h1>Email classification review</h1>
    <p class="summary">{summary}</p>
    {"".join(rows)}
  </main>
</body>
</html>
"""


def write_review(
    emails_path: str | Path,
    out_path: str | Path,
    method: str = "rules",
    labels_path: str | Path | None = None,
    *,
    env_file: str | Path | None = None,
    overrides=None,
    tag: str | None = None,
    save: bool = True,
    outputs_dir: str | Path = Path("runs/results"),
) -> Path:
    emails = load_emails(emails_path)
    results = run(
        emails_path,
        method=method,
        env_file=env_file,
        overrides=overrides,
        tag=tag,
        save=save,
        outputs_dir=outputs_dir,
    )
    labels = load_labels(labels_path) if labels_path else None
    html_doc = render_review(emails, results, labels, method, emails_path)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html_doc, encoding="utf-8")
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Write an HTML review of email classifications.",
    )
    parser.add_argument(
        "emails",
        type=Path,
        help="Path to a JSON file with an 'emails' list.",
    )
    parser.add_argument(
        "--method",
        default="rules",
        choices=available_methods(),
        help="Classification method (default: rules).",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=None,
        help="Optional eval labels JSON (e.g. data/eval_labels.json).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/review.html"),
        help="Output HTML path (default: data/review.html).",
    )
    add_llm_config_args(parser)
    add_run_artifact_args(parser)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        out = write_review(
            args.emails,
            args.out,
            method=args.method,
            labels_path=args.labels,
            env_file=args.env_file,
            overrides=overrides_from_args(args),
            tag=args.tag,
            save=not args.no_save,
            outputs_dir=args.outputs_dir,
        )
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
