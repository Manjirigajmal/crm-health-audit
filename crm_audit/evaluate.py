"""Score the audit against the issues deliberately planted in the synthetic data.

Precision = of the records a check flagged, how many really had that planted issue.
Recall    = of the planted issues, how many the check caught.
"""
from __future__ import annotations

import pandas as pd

from .audit import AuditResult

ISSUE_TO_CHECK = {
    "missing_email": "missing_email_contacts",
    "missing_phone": "missing_phone_contacts",
    "missing_industry": "missing_industry_contacts",
    "missing_lead_source": "missing_lead_source_contacts",
    "missing_owner": "missing_owner_contacts",
    "missing_job_title": "missing_job_title_contacts",
    "duplicate_contact": "duplicate_contacts",
    "invalid_email": "invalid_email",
    "invalid_phone": "invalid_phone",
    "inconsistent_city": "inconsistent_city_contacts",
    "inconsistent_industry": "inconsistent_industry_contacts",
    "inconsistent_country": "inconsistent_country_contacts",
    "inconsistent_lead_source": "inconsistent_lead_source_contacts",
    "inconsistent_stage": "inconsistent_stage_deals",
    "future_date": "future_created_contacts",
    "missing_amount": "missing_amount_deals",
    "negative_amount": "non_positive_amount",
    "close_before_created": "close_before_created",
    "orphan_deal": "orphan_deals",
    "stale_record": "stale_contacts",
    "overdue_open_deal": "overdue_open_deals",
}


def evaluate(result: AuditResult, issues: pd.DataFrame) -> pd.DataFrame:
    flagged = {r.check_id: set(r.flagged_ids) for r in result.results}
    rows = []
    for issue, check_id in ISSUE_TO_CHECK.items():
        truth = set(issues.loc[issues["issue"] == issue, "record_id"].astype(str))
        found = flagged.get(check_id, set())
        tp = len(truth & found)
        rows.append({
            "planted_issue": issue, "check": check_id, "planted": len(truth), "flagged": len(found),
            "caught": tp,
            "recall": tp / len(truth) if truth else float("nan"),
            "precision": tp / len(found) if found else float("nan"),
        })
    return pd.DataFrame(rows)
