"""Render an AuditResult as a single self-contained HTML report (prints cleanly to PDF)."""
from __future__ import annotations

import math
from html import escape
from pathlib import Path

from .audit import AuditResult

STATUS_ICON = {"pass": "✓", "warn": "!", "fail": "✕"}
VERDICT_CLASS = {"Ready": "ok", "Almost ready": "warn", "Not ready": "bad", "Not assessed": "na"}

# which fixes unblock which AI use cases (shown in the action plan)
UNLOCKS = {
    "duplicate_contacts": ["Churn prediction", "AI sales assistant"],
    "overdue_open_deals": ["Sales forecasting"],
    "no_recent_activity": ["Churn prediction", "AI sales assistant"],
    "won_without_amount": ["Customer lifetime value"],
    "missing_amount_deals": ["Customer lifetime value", "Sales forecasting"],
    "inconsistent_stage_deals": ["Lead & deal scoring"],
    "missing_industry_contacts": ["Lead & deal scoring"],
    "missing_lead_source_contacts": ["Lead & deal scoring"],
    "invalid_email": ["AI sales assistant"],
    "orphan_deals": ["Customer lifetime value"],
    "close_before_created": ["Sales forecasting"],
}


def _tone(score: float) -> str:
    if score != score:  # NaN
        return "na"
    return "ok" if score >= 85 else "fair" if score >= 70 else "warn" if score >= 50 else "bad"


def _ring(score: float, size: int = 168) -> str:
    r = 70
    circ = 2 * math.pi * r
    frac = max(0.0, min(1.0, score / 100))
    return f"""
    <svg class="ring tone-{_tone(score)}" viewBox="0 0 168 168" width="{size}" height="{size}" role="img"
         aria-label="Health score {score:.0f} out of 100">
      <circle cx="84" cy="84" r="{r}" class="ring-track"/>
      <circle cx="84" cy="84" r="{r}" class="ring-value" stroke-dasharray="{circ * frac:.1f} {circ:.1f}"
              transform="rotate(-90 84 84)"/>
      <text x="84" y="86" class="ring-num">{score:.0f}</text>
      <text x="84" y="112" class="ring-sub">out of 100</text>
    </svg>"""


def render_html(result: AuditResult, client_name: str = "Sample CRM") -> str:
    h = result.health
    top = result.top_issues(5)
    flagged = result.flagged_records()
    n_flagged_records = flagged[["table", "record_id"]].drop_duplicates().shape[0] if len(flagged) else 0
    total_records = result.n_contacts + result.n_deals
    gain = top["points_recoverable"].sum()

    dims_html = "".join(
        f"""<div class="dim">
              <div class="dim-head"><span>{escape(row.dimension)}</span>
                <span class="dim-score tone-{_tone(row.score)}">{'—' if row.score != row.score else f'{row.score:.0f}'}</span></div>
              <div class="bar"><div class="fill tone-{_tone(row.score)}" style="width:{0 if row.score != row.score else row.score:.1f}%"></div></div>
            </div>"""
        for row in h.dimensions.itertuples()
    )

    issues_html = ""
    for i, row in enumerate(top.itertuples(), 1):
        unlocks = UNLOCKS.get(row.check_id, [])
        unlock_html = (f'<div class="unlocks">Unblocks: {", ".join(escape(u) for u in unlocks)}</div>' if unlocks else "")
        issues_html += f"""
        <div class="issue">
          <div class="issue-rank">{i}</div>
          <div class="issue-body">
            <div class="issue-top">
              <h3>{escape(row.title)}</h3>
              <span class="gain">+{row.points_recoverable:.1f} pts if fixed</span>
            </div>
            <div class="issue-meta">{row.affected:,} of {row.total:,} records · {row.rate:.1%} · {escape(row.dimension)}</div>
            <p class="impact">{escape(row.impact)}</p>
            <p class="fix"><b>Fix:</b> {escape(row.fix)}</p>
            {unlock_html}
          </div>
        </div>"""

    uc_html = ""
    for u in result.use_cases:
        reqs = "".join(
            f"""<li class="req st-{r.status}"><span class="ico">{STATUS_ICON[r.status]}</span>
                 <div><b>{escape(r.name)}</b><span>{escape(r.detail)}</span></div></li>"""
            for r in u.requirements
        )
        uc_html += f"""
        <div class="uc card">
          <div class="uc-head">
            <div><h3>{escape(u.name)}</h3><p class="q">{escape(u.question)}</p></div>
            <span class="pill v-{VERDICT_CLASS[u.verdict]}">{escape(u.verdict)}</span>
          </div>
          <ul class="reqs">{reqs}</ul>
        </div>"""

    rows = ""
    c = h.checks.sort_values(["dimension", "points_recoverable"], ascending=[True, False])
    for row in c.itertuples():
        if not row.assessed:
            rows += (f"<tr class='muted'><td>{escape(row.dimension)}</td><td>{escape(row.title)}</td>"
                     f"<td colspan='3'>{escape(row.note)}</td></tr>")
            continue
        rows += (f"<tr><td>{escape(row.dimension)}</td><td>{escape(row.title)}</td>"
                 f"<td class='num'>{row.affected:,} / {row.total:,}</td><td class='num'>{row.rate:.1%}</td>"
                 f"<td class='num'><span class='chip tone-{_tone(row.score)}'>{row.score:.0f}</span></td></tr>")

    n_ready = sum(u.verdict == "Ready" for u in result.use_cases)
    n_almost = sum(u.verdict == "Almost ready" for u in result.use_cases)
    ready_line = f"{n_ready} of {len(result.use_cases)}"
    ready_sub = f"AI use cases ready now ({n_almost} almost ready)"

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>CRM Data Health Audit — {escape(client_name)}</title>
<style>
  :root {{
    --bg:#f6f5f1; --paper:#ffffff; --ink:#1c1f24; --muted:#6b7079; --line:#e4e2dc;
    --ok:#1f8a5b; --fair:#6a9a2b; --warn:#c7801a; --bad:#c2412d; --na:#9aa0a6; --accent:#24486b;
  }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--ink);
         font:15px/1.55 "Inter", "Segoe UI", system-ui, -apple-system, sans-serif; }}
  .page {{ max-width:960px; margin:0 auto; padding:40px 16px 64px; }}
  header.top {{ display:flex; justify-content:space-between; align-items:flex-end; gap:16px; flex-wrap:wrap;
               border-bottom:2px solid var(--ink); padding-bottom:14px; margin-bottom:28px; }}
  .eyebrow {{ text-transform:uppercase; letter-spacing:.12em; font-size:12px; color:var(--muted); margin:0 0 4px; }}
  h1 {{ font-family:Georgia, "Times New Roman", serif; font-weight:600; font-size:34px; margin:0; letter-spacing:-.01em; }}
  .meta {{ color:var(--muted); font-size:13px; text-align:right; }}
  h2 {{ font-family:Georgia, serif; font-weight:600; font-size:22px; margin:44px 0 6px; }}
  .lede {{ color:var(--muted); margin:0 0 18px; max-width:68ch; }}
  .card {{ background:var(--paper); border:1px solid var(--line); border-radius:10px; padding:22px; }}
  .hero {{ display:grid; grid-template-columns:auto 1fr; gap:28px; align-items:center; }}
  .ring-track {{ fill:none; stroke:var(--line); stroke-width:14; }}
  .ring-value {{ fill:none; stroke-width:14; stroke-linecap:round; }}
  .ring.tone-ok .ring-value {{ stroke:var(--ok); }} .ring.tone-fair .ring-value {{ stroke:var(--fair); }}
  .ring.tone-warn .ring-value {{ stroke:var(--warn); }} .ring.tone-bad .ring-value {{ stroke:var(--bad); }}
  .ring-num {{ font:600 44px Georgia, serif; text-anchor:middle; fill:var(--ink); }}
  .ring-sub {{ font-size:12px; text-anchor:middle; fill:var(--muted); }}
  .verdict {{ font-family:Georgia, serif; font-size:26px; margin:0 0 4px; }}
  .verdict-sub {{ color:var(--muted); margin:0 0 16px; }}
  .kpis {{ display:grid; grid-template-columns:repeat(3, 1fr); gap:12px; }}
  .kpi {{ border-top:1px solid var(--line); padding-top:10px; }}
  .kpi b {{ display:block; font-size:22px; font-variant-numeric:tabular-nums; }}
  .kpi span {{ font-size:12.5px; color:var(--muted); }}
  .dims {{ display:grid; grid-template-columns:repeat(2, 1fr); gap:14px 28px; }}
  .dim-head {{ display:flex; justify-content:space-between; font-size:14px; margin-bottom:6px; }}
  .dim-score {{ font-weight:600; font-variant-numeric:tabular-nums; }}
  .bar {{ height:8px; background:var(--line); border-radius:99px; overflow:hidden; }}
  .fill {{ height:100%; border-radius:99px; }}
  .fill.tone-ok {{ background:var(--ok); }} .fill.tone-fair {{ background:var(--fair); }}
  .fill.tone-warn {{ background:var(--warn); }} .fill.tone-bad {{ background:var(--bad); }}
  .tone-ok {{ color:var(--ok); }} .tone-fair {{ color:var(--fair); }} .tone-warn {{ color:var(--warn); }}
  .tone-bad {{ color:var(--bad); }} .tone-na {{ color:var(--na); }}
  .issue {{ display:grid; grid-template-columns:36px 1fr; gap:14px; padding:18px 0; border-top:1px solid var(--line); }}
  .issue:first-child {{ border-top:0; padding-top:0; }}
  .issue-rank {{ width:32px; height:32px; border-radius:50%; background:var(--accent); color:#fff;
                display:grid; place-items:center; font-weight:600; }}
  .issue-top {{ display:flex; justify-content:space-between; gap:12px; align-items:baseline; flex-wrap:wrap; }}
  .issue h3, .uc h3 {{ margin:0; font-size:16.5px; }}
  .gain {{ font-size:13px; font-weight:600; color:var(--ok); background:#e8f4ee; padding:2px 10px; border-radius:99px; white-space:nowrap; }}
  .issue-meta {{ color:var(--muted); font-size:13px; margin:2px 0 8px; }}
  .impact {{ margin:0 0 6px; }} .fix {{ margin:0; color:#33373d; }}
  .unlocks {{ margin-top:8px; font-size:13px; color:var(--accent); }}
  .ucs {{ display:grid; gap:14px; }}
  .uc-head {{ display:flex; justify-content:space-between; gap:12px; align-items:flex-start; }}
  .q {{ margin:2px 0 0; color:var(--muted); font-size:13.5px; }}
  .pill {{ font-size:12.5px; font-weight:600; padding:4px 12px; border-radius:99px; white-space:nowrap; }}
  .v-ok {{ background:#e8f4ee; color:var(--ok); }} .v-warn {{ background:#fbf0de; color:#9a5f0c; }}
  .v-bad {{ background:#f9e5e1; color:var(--bad); }} .v-na {{ background:#eee; color:var(--na); }}
  .reqs {{ list-style:none; padding:0; margin:14px 0 0; display:grid; gap:8px; }}
  .req {{ display:grid; grid-template-columns:22px 1fr; gap:10px; font-size:13.5px; }}
  .req b {{ display:block; font-weight:600; }} .req span {{ color:#4a4f57; }}
  .ico {{ width:20px; height:20px; border-radius:50%; display:grid; place-items:center; font-size:11px; font-weight:700; color:#fff; margin-top:1px; }}
  .st-pass .ico {{ background:var(--ok); }} .st-warn .ico {{ background:var(--warn); }} .st-fail .ico {{ background:var(--bad); }}
  table {{ width:100%; border-collapse:collapse; font-size:13.5px; }}
  th, td {{ text-align:left; padding:8px 10px; border-bottom:1px solid var(--line); vertical-align:top; }}
  th {{ font-size:12px; text-transform:uppercase; letter-spacing:.06em; color:var(--muted); font-weight:600; }}
  td.num {{ text-align:right; font-variant-numeric:tabular-nums; white-space:nowrap; }}
  tr.muted td {{ color:var(--na); }}
  .chip {{ font-weight:600; }}
  .method {{ font-size:13.5px; color:#4a4f57; }}
  .method code {{ background:#f0eee8; padding:1px 5px; border-radius:4px; }}
  footer {{ margin-top:40px; font-size:12px; color:var(--muted); border-top:1px solid var(--line); padding-top:12px; }}
  @media (max-width:640px) {{
    .hero {{ grid-template-columns:1fr; justify-items:center; text-align:center; }}
    .kpis {{ grid-template-columns:1fr; }} .dims {{ grid-template-columns:1fr; }}
    .meta {{ text-align:left; }} h1 {{ font-size:28px; }}
    table {{ font-size:12.5px; }} th:nth-child(1), td:nth-child(1) {{ display:none; }}
  }}
  @media print {{ body {{ background:#fff; }} .card {{ break-inside:avoid; }} .page {{ padding-top:0; }} }}
</style></head>
<body><div class="page">

<header class="top">
  <div><p class="eyebrow">CRM Data Health &amp; AI-Readiness Audit</p><h1>{escape(client_name)}</h1></div>
  <div class="meta">Audit date {result.as_of:%d %b %Y}<br>{result.n_contacts:,} contacts · {result.n_deals:,} deals</div>
</header>

<section class="card hero">
  {_ring(h.overall)}
  <div>
    <p class="verdict tone-{_tone(h.overall)}">{escape(h.label)}</p>
    <p class="verdict-sub">{escape(h.meaning)}</p>
    <div class="kpis">
      <div class="kpi"><b>{n_flagged_records:,}</b><span>of {total_records:,} records have at least one issue</span></div>
      <div class="kpi"><b>+{gain:.0f} pts</b><span>available by fixing the top 5 issues</span></div>
      <div class="kpi"><b>{ready_line}</b><span>{ready_sub}</span></div>
    </div>
  </div>
</section>

<h2>Score by dimension</h2>
<p class="lede">Each dimension is the weighted average of its checks. A check scores 0 once its failure rate reaches the tolerance set for that problem.</p>
<section class="card dims">{dims_html}</section>

<h2>Top 5 problems, ranked by impact</h2>
<p class="lede">Ranked by how many points each problem costs the overall score — i.e. what fixing it first would recover.</p>
<section class="card">{issues_html or '<p>No issues found.</p>'}</section>

<h2>AI-readiness</h2>
<p class="lede">Clean data isn't automatically ML-ready data. Each AI use case has its own requirements for volume, label balance, history and leakage.</p>
<section class="ucs">{uc_html}</section>

<h2>All checks</h2>
<section class="card" style="padding:8px 12px; overflow-x:auto">
<table><thead><tr><th>Dimension</th><th>Check</th><th style="text-align:right">Affected</th><th style="text-align:right">Rate</th><th style="text-align:right">Score</th></tr></thead>
<tbody>{rows}</tbody></table>
</section>

<h2>Method</h2>
<section class="card method">
  <p><b>Check score</b> = 100 × (1 − failure rate ÷ tolerance), floored at 0. <b>Dimension score</b> = weighted average of its checks.
  <b>Overall score</b> = weighted average of dimensions (Completeness 25%, Uniqueness 20%, Validity, Consistency and Freshness 15% each, Integrity 10%).</p>
  <p><b>Duplicates</b> are found by exact email match, matching phone digits, and fuzzy name matching within the same company
  (company suffixes like “Pvt Ltd” ignored). <b>Inconsistent labels</b> are found by normalising case and punctuation, applying a
  dictionary of known synonyms (Bombay → Mumbai, BFSI → Financial Services), and fuzzy-merging misspellings.
  <b>Target leakage</b> is flagged when a single field separates won from lost deals ≥ 90% of the way from the majority-class baseline to perfect prediction.</p>
  <p>A record-level list of every flagged row is provided alongside this report so each issue can be fixed at source.</p>
</section>

<footer>Generated by CRM Health Audit · scores are relative to the tolerances documented above and should be re-run monthly to track progress.</footer>
</div></body></html>"""


def write_html(result: AuditResult, path: str | Path, client_name: str = "Sample CRM") -> Path:
    path = Path(path)
    path.write_text(render_html(result, client_name), encoding="utf-8")
    return path
