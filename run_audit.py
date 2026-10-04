"""Command-line audit.

    python run_audit.py --contacts sample_data/contacts.csv --deals sample_data/deals.csv
    python run_audit.py --contacts hubspot_contacts.csv --client "Acme Pvt Ltd" --out reports/acme

Writes <out>/report.html, <out>/audit.json and <out>/flagged_records.csv.
"""
import argparse
from pathlib import Path

import pandas as pd

from crm_audit.audit import run_audit
from crm_audit.report import write_html


def main() -> None:
    ap = argparse.ArgumentParser(description="CRM data health score + AI-readiness audit")
    ap.add_argument("--contacts", help="contacts export (.csv or .xlsx)")
    ap.add_argument("--deals", help="deals / opportunities export (.csv or .xlsx)")
    ap.add_argument("--client", default="Sample CRM", help="name shown on the report")
    ap.add_argument("--as-of", default=None, help="audit date, YYYY-MM-DD (default: today)")
    ap.add_argument("--out", default="reports/latest", help="output folder")
    ap.add_argument("--truth", help="optional injected_issues.csv to measure detection accuracy")
    args = ap.parse_args()

    result = run_audit(args.contacts, args.deals, as_of=args.as_of)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_html(result, out / "report.html", args.client)
    result.to_json(out / "audit.json")
    result.flagged_records().to_csv(out / "flagged_records.csv", index=False)

    h = result.health
    print(f"\nOverall health: {h.overall:.1f}/100 — {h.label}")
    for row in h.dimensions.itertuples():
        print(f"  {row.dimension:<13} {row.score:5.1f}")
    print("\nTop issues:")
    for row in result.top_issues().itertuples():
        print(f"  +{row.points_recoverable:4.1f} pts  {row.title} ({row.affected:,} records)")
    print("\nAI readiness:")
    for u in result.use_cases:
        print(f"  {u.verdict:<13} {u.name}")
    if args.truth:
        from crm_audit.evaluate import evaluate
        ev = evaluate(result, pd.read_csv(args.truth))
        print(f"\nDetection vs planted issues: recall {ev.caught.sum() / ev.planted.sum():.1%}, "
              f"precision {ev.caught.sum() / ev.flagged.sum():.1%}")
        ev.to_csv(out / "detection_accuracy.csv", index=False)
    print(f"\nReport written to {out.resolve() / 'report.html'}")


if __name__ == "__main__":
    main()
