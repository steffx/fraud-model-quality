"""Builds the static GitHub Pages report from the training artifacts and the pytest results.

    python -m fraudmodel.train
    pytest
    python report/build_site.py        # writes site/index.html
"""

from __future__ import annotations

import html
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from sklearn.metrics import precision_recall_curve

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"
SITE = ROOT / "site"
JUNIT = ROOT / "reports" / "junit.xml"

# Categorical slots 1-3 of the validated reference palette (light, dark).
SERIES = {
    "logistic_regression": ("--series-1", "Logistic regression"),
    "random_forest": ("--series-2", "Random forest"),
    "gradient_boosting": ("--series-3", "Gradient boosting"),
}

W, H = 640, 400
PAD = {"l": 56, "r": 150, "t": 16, "b": 48}


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def pr_curve_svg(y_true: np.ndarray, scores: dict[str, np.ndarray], champion: str) -> str:
    pw, ph = W - PAD["l"] - PAD["r"], H - PAD["t"] - PAD["b"]
    sx = lambda r: PAD["l"] + r * pw  # noqa: E731  recall on x
    sy = lambda p: PAD["t"] + (1 - p) * ph  # noqa: E731  precision on y

    parts = []
    for v in (0, 0.25, 0.5, 0.75, 1):
        parts.append(f'<line class="grid" x1="{sx(0)}" x2="{sx(1)}" y1="{sy(v):.1f}" y2="{sy(v):.1f}"/>')
        parts.append(f'<text class="tick" x="{sx(0) - 8}" y="{sy(v) + 4:.1f}" text-anchor="end">{int(v * 100)}%</text>')
        parts.append(f'<text class="tick" x="{sx(v):.1f}" y="{sy(0) + 20}" text-anchor="middle">{int(v * 100)}%</text>')
    parts.append(f'<text class="axis" x="{sx(0.5):.1f}" y="{H - 8}" text-anchor="middle">Recall (share of fraud caught)</text>')
    parts.append(
        f'<text class="axis" transform="translate(14 {sy(0.5):.1f}) rotate(-90)" text-anchor="middle">Precision (alerts that are real fraud)</text>'
    )

    label_ys = []
    for name, s in scores.items():
        var, label = SERIES[name]
        precision, recall, _ = precision_recall_curve(y_true, s)
        # Downsample to keep the SVG small; keep the order of the curve.
        idx = np.unique(np.linspace(0, len(recall) - 1, 160).astype(int))
        pts = " ".join(f"{sx(recall[i]):.1f},{sy(precision[i]):.1f}" for i in idx)
        width = 2.5 if name == champion else 2
        parts.append(f'<polyline fill="none" stroke="var({var})" stroke-width="{width}" stroke-linejoin="round" points="{pts}"/>')
        # Hover targets: bigger than the mark, native tooltip with exact values.
        for i in idx[::8]:
            parts.append(
                f'<circle class="hit" cx="{sx(recall[i]):.1f}" cy="{sy(precision[i]):.1f}" r="8">'
                f"<title>{label}: precision {pct(precision[i])}, recall {pct(recall[i])}</title></circle>"
            )
        # Direct label at the right end of the plot, nudged apart to avoid collisions.
        target = sy(precision[np.argmin(np.abs(recall - 0.5))])
        while any(abs(target - y) < 16 for y in label_ys):
            target += 16
        label_ys.append(target)
        parts.append(f'<circle cx="{sx(1) + 14}" cy="{target - 4:.1f}" r="5" fill="var({var})"/>')
        parts.append(f'<text class="label" x="{sx(1) + 24}" y="{target:.1f}">{label}</text>')

    return (
        f'<svg viewBox="0 0 {W} {H}" role="img" aria-labelledby="pr-title pr-desc" class="chart">'
        '<title id="pr-title">Precision-recall curves on the test set</title>'
        '<desc id="pr-desc">Higher and further right is better. Exact values are in the table below.</desc>'
        + "".join(parts)
        + "</svg>"
    )


def junit_summary() -> dict:
    if not JUNIT.exists():
        return {}
    root = ET.parse(JUNIT).getroot()
    suites = root.findall("testsuite") or [root]
    by_file: dict[str, dict] = {}
    for suite in suites:
        for case in suite.findall("testcase"):
            f = case.get("classname", "").split(".")[-1] or "other"
            entry = by_file.setdefault(f, {"passed": 0, "failed": 0, "skipped": 0})
            if case.find("failure") is not None or case.find("error") is not None:
                entry["failed"] += 1
            elif case.find("skipped") is not None:
                entry["skipped"] += 1
            else:
                entry["passed"] += 1
    return by_file


def build() -> Path:
    summary = json.loads((ARTIFACTS / "metrics.json").read_text())
    preds = np.load(ARTIFACTS / "test_predictions.npz")
    champion = summary["champion"]
    models = summary["models"]
    scores = {n: preds[f"scores_{n}"] for n in SERIES if f"scores_{n}" in preds}
    m = models[champion]["metrics"]
    cm = m["confusion_matrix"]
    bar = summary["quality_bar"]
    tests = junit_summary()
    total_passed = sum(t["passed"] for t in tests.values())
    total_failed = sum(t["failed"] for t in tests.values())

    model_rows = "".join(
        f"<tr><td><span class='swatch' style='background:var({SERIES[n][0]})'></span>{SERIES[n][1]}"
        f"{' <span class=pill>champion</span>' if n == champion else ''}</td>"
        f"<td>{x['metrics']['pr_auc']:.3f}</td><td>{pct(x['metrics']['precision'])}</td><td>{pct(x['metrics']['recall'])}</td>"
        f"<td>{pct(x['metrics']['false_positive_rate'])}</td><td>{x['metrics']['train_seconds']}s</td>"
        f"<td class='{'ok' if x['passes_quality_bar'] else 'bad'}'>{'✔ PASS' if x['passes_quality_bar'] else '✘ FAIL'}</td></tr>"
        for n, x in models.items()
    )
    check_rows = "".join(
        f"<tr><td>{html.escape(c['name'])}</td><td class='{'ok' if c['passed'] else 'bad'}'>{'✔ pass' if c['passed'] else '✘ fail'}</td>"
        f"<td>{html.escape(c['detail'])}</td></tr>"
        for c in summary["data"]["quality_checks"]
    )
    test_rows = "".join(
        f"<tr><td>{html.escape(f)}</td><td>{t['passed']}</td><td>{t['failed']}</td><td>{t['skipped']}</td></tr>"
        for f, t in sorted(tests.items())
    )

    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Fraud Model Quality Report</title>
<style>
:root {{
  color-scheme: light;
  --surface: #fcfcfb; --card: #ffffff; --border: #e3e2de;
  --text-primary: #0b0b0b; --text-secondary: #52514e; --grid: #ecebe7;
  --series-1: #2a78d6; --series-2: #eb6834; --series-3: #1baf7a;
  --good: #0b6b3a; --bad: #b42318;
  font-family: system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif;
}}
@media (prefers-color-scheme: dark) {{
  :root:where(:not([data-theme="light"])) {{
    color-scheme: dark;
    --surface: #1a1a19; --card: #232322; --border: #3a3936;
    --text-primary: #ffffff; --text-secondary: #c3c2b7; --grid: #34332f;
    --series-1: #3987e5; --series-2: #d95926; --series-3: #199e70;
    --good: #5cc98b; --bad: #f08a80;
  }}
}}
:root[data-theme="dark"] {{
  color-scheme: dark;
  --surface: #1a1a19; --card: #232322; --border: #3a3936;
  --text-primary: #ffffff; --text-secondary: #c3c2b7; --grid: #34332f;
  --series-1: #3987e5; --series-2: #d95926; --series-3: #199e70;
  --good: #5cc98b; --bad: #f08a80;
}}
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: var(--surface); color: var(--text-primary); }}
main {{ max-width: 980px; margin: 0 auto; padding: 24px 16px 48px; display: grid; gap: 16px; }}
h1 {{ margin: 0; font-size: 1.6rem; }} h2 {{ margin: 0 0 12px; font-size: 1.1rem; }}
p {{ color: var(--text-secondary); margin: 4px 0 0; }}
.card {{ background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 20px; min-width: 0; }}
.tiles {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; }}
.tile {{ border: 1px solid var(--border); border-radius: 10px; padding: 12px; }}
.tile span {{ color: var(--text-secondary); font-size: .85rem; }}
.tile b {{ display: block; font-size: 1.6rem; font-variant-numeric: tabular-nums; margin-top: 4px; }}
.table-wrap {{ overflow-x: auto; }}
table {{ width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; }}
th, td {{ text-align: left; padding: 8px; border-bottom: 1px solid var(--border); white-space: nowrap; }}
th {{ color: var(--text-secondary); font-size: .85rem; font-weight: 600; }}
td.ok {{ color: var(--good); font-weight: 600; }} td.bad {{ color: var(--bad); font-weight: 600; }}
.pill {{ font-size: .72rem; border: 1px solid var(--border); border-radius: 999px; padding: 1px 8px; margin-left: 6px; color: var(--text-secondary); }}
.swatch {{ display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 8px; }}
.chart {{ width: 100%; height: auto; display: block; }}
.chart .grid {{ stroke: var(--grid); stroke-width: 1; }}
.chart .tick {{ fill: var(--text-secondary); font-size: 12px; }}
.chart .axis {{ fill: var(--text-secondary); font-size: 13px; }}
.chart .label {{ fill: var(--text-primary); font-size: 13px; }}
.chart .hit {{ fill: transparent; cursor: crosshair; }}
.chart .hit:hover {{ fill: var(--text-secondary); fill-opacity: .25; }}
.cm {{ display: grid; grid-template-columns: auto 1fr 1fr; gap: 6px; max-width: 420px; }}
.cm div {{ border: 1px solid var(--border); border-radius: 8px; padding: 10px; font-variant-numeric: tabular-nums; }}
.cm .h {{ border: none; color: var(--text-secondary); font-size: .85rem; }}
</style>
</head>
<body>
<main>
  <header>
    <h1>Fraud model quality report</h1>
    <p>Champion <b>{SERIES[champion][1]}</b> · version {summary['model_version']} · trained {summary['trained_at']} ·
    {summary['data']['rows']['train']:,} training / {summary['data']['rows']['test']:,} test payments</p>
  </header>

  <section class="card" aria-label="Headline numbers">
    <div class="tiles">
      <div class="tile"><span>Tests</span><b>{total_passed} passed{f", {total_failed} failed" if total_failed else ""}</b></div>
      <div class="tile"><span>PR-AUC (bar ≥ {bar['min_pr_auc']})</span><b>{m['pr_auc']:.3f}</b></div>
      <div class="tile"><span>Precision (bar ≥ {pct(bar['min_precision'])})</span><b>{pct(m['precision'])}</b></div>
      <div class="tile"><span>Recall (bar ≥ {pct(bar['min_recall'])})</span><b>{pct(m['recall'])}</b></div>
      <div class="tile"><span>False positive rate (bar ≤ {pct(bar['max_fpr'])})</span><b>{pct(m['false_positive_rate'])}</b></div>
      <div class="tile"><span>Block precision</span><b>{pct(m['block_precision'])}</b></div>
    </div>
  </section>

  <section class="card">
    <h2>Precision vs. recall</h2>
    <p>Each line shows the trade-off for one model: moving the threshold catches more fraud (right) but raises more false alarms (down). Hover the lines for exact values.</p>
    {pr_curve_svg(preds['y_true'], scores, champion)}
  </section>

  <section class="card">
    <h2>Candidate models</h2>
    <div class="table-wrap"><table>
      <thead><tr><th>Model</th><th>PR-AUC</th><th>Precision</th><th>Recall</th><th>False positive rate</th><th>Training</th><th>Quality bar</th></tr></thead>
      <tbody>{model_rows}</tbody>
    </table></div>
  </section>

  <section class="card">
    <h2>Champion at the review threshold ({models[champion]['thresholds']['review']})</h2>
    <div class="cm" role="table" aria-label="Confusion matrix">
      <div class="h"></div><div class="h">Flagged</div><div class="h">Not flagged</div>
      <div class="h">Fraud</div><div>{cm['tp']} caught</div><div>{cm['fn']} missed</div>
      <div class="h">Genuine</div><div>{cm['fp']} false alarms</div><div>{cm['tn']:,} approved</div>
    </div>
  </section>

  <section class="card">
    <h2>Data quality checks</h2>
    <div class="table-wrap"><table><thead><tr><th>Check</th><th>Result</th><th>Detail</th></tr></thead><tbody>{check_rows}</tbody></table></div>
  </section>

  {"<section class='card'><h2>Test results</h2><div class='table-wrap'><table><thead><tr><th>Suite</th><th>Passed</th><th>Failed</th><th>Skipped</th></tr></thead><tbody>" + test_rows + "</tbody></table></div></section>" if tests else ""}
</main>
</body>
</html>"""
    SITE.mkdir(exist_ok=True)
    out = SITE / "index.html"
    out.write_text(page)
    return out


if __name__ == "__main__":
    path = build()
    print(f"Report written to {path.relative_to(ROOT)}")
    sys.exit(0)
