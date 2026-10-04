"""One entry point: give it a contacts export and/or a deals export, get an audit back."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .checks import AuditContext, CheckResult, run_all_checks
from .readiness import UseCase, assess
from .schema import CONTACT_ALIASES, DEAL_ALIASES, standardize_columns
from .scoring import HealthScore, score


@dataclass
class AuditResult:
    as_of: pd.Timestamp
    n_contacts: int
    n_deals: int
    column_map: dict[str, dict[str, str]]
    results: list[CheckResult]
    health: HealthScore
    use_cases: list[UseCase]

    def top_issues(self, n: int = 5) -> pd.DataFrame:
        c = self.health.checks
        c = c[c["assessed"] & (c["affected"] > 0)]
        return c.sort_values("points_recoverable", ascending=False).head(n)

    def flagged_records(self) -> pd.DataFrame:
        """Long table of every flagged record: table, record_id, check, dimension."""
        rows = [{"table": r.table, "record_id": rid, "check_id": r.check_id, "check": r.title,
                 "dimension": r.dimension} for r in self.results for rid in r.flagged_ids]
        return pd.DataFrame(rows, columns=["table", "record_id", "check_id", "check", "dimension"])

    def to_dict(self) -> dict:
        return {
            "as_of": str(self.as_of.date()),
            "records": {"contacts": self.n_contacts, "deals": self.n_deals},
            "overall_score": self.health.overall,
            "label": self.health.label,
            "dimensions": self.health.dimensions.round(1).to_dict(orient="records"),
            "top_issues": self.top_issues()[["check_id", "title", "affected", "total", "rate",
                                              "points_recoverable", "impact", "fix"]].round(3).to_dict(orient="records"),
            "ai_readiness": [
                {"use_case": u.name, "verdict": u.verdict,
                 "requirements": [{"name": r.name, "status": r.status, "detail": r.detail} for r in u.requirements]}
                for u in self.use_cases
            ],
            "checks": self.health.checks.drop(columns=["impact", "fix"]).round(3).to_dict(orient="records"),
        }

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, default=str))


def _read(src) -> pd.DataFrame | None:
    if src is None:
        return None
    if isinstance(src, pd.DataFrame):
        return src.copy()
    name = getattr(src, "name", str(src)).lower()
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(src)
    return pd.read_csv(src, low_memory=False)


def run_audit(contacts=None, deals=None, as_of: str | pd.Timestamp | None = None) -> AuditResult:
    """Run the full audit.

    contacts / deals: a path, an uploaded file object, or a DataFrame. Either can be omitted.
    as_of: the date the audit is "run on" (defaults to today) — drives freshness checks.
    """
    c_raw, d_raw = _read(contacts), _read(deals)
    if c_raw is None and d_raw is None:
        raise ValueError("Provide at least a contacts or a deals export.")
    col_map: dict[str, dict[str, str]] = {}
    c = d = None
    if c_raw is not None:
        c, col_map["contacts"] = standardize_columns(c_raw, CONTACT_ALIASES)
    if d_raw is not None:
        d, col_map["deals"] = standardize_columns(d_raw, DEAL_ALIASES)
    ts = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp.today().normalize()

    ctx = AuditContext(c, d, ts)
    results = run_all_checks(ctx)
    return AuditResult(
        as_of=ts,
        n_contacts=0 if c is None else len(c),
        n_deals=0 if d is None else len(d),
        column_map=col_map,
        results=results,
        health=score(results),
        use_cases=assess(ctx),
    )
