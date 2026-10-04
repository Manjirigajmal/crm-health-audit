"""Turn individual check results into dimension scores and one overall score.

Scoring is deliberately transparent so a client can follow it:

* check score  = 100 × (1 − failure_rate / tolerance), floored at 0.
  "tolerance" is the failure rate at which a check is considered fully failed
  (e.g. 25% duplicates → 0/100).
* dimension score = weighted average of its check scores.
* overall score   = weighted average of dimension scores.
* "points recoverable" for a check = how much the overall score would rise if
  that one problem were fixed completely. This is what ranks the top issues.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .checks import DIMENSIONS, CheckResult

DIMENSION_WEIGHTS = {
    "Completeness": 0.25,
    "Uniqueness": 0.20,
    "Validity": 0.15,
    "Consistency": 0.15,
    "Freshness": 0.15,
    "Integrity": 0.10,
}

BANDS = [(85, "Healthy", "Data is in good shape; keep monitoring."),
         (70, "Fair", "Usable, but specific gaps will undermine reporting and AI."),
         (50, "Needs attention", "Problems are big enough to mislead reports and models."),
         (0, "Critical", "Data can't be trusted for decisions until it's repaired.")]


def band(score: float) -> tuple[str, str]:
    for cutoff, label, meaning in BANDS:
        if score >= cutoff:
            return label, meaning
    return BANDS[-1][1], BANDS[-1][2]


@dataclass
class HealthScore:
    overall: float
    label: str
    meaning: str
    dimensions: pd.DataFrame   # dimension, score, weight, checks
    checks: pd.DataFrame       # one row per check incl. points_recoverable


def score(results: list[CheckResult], weights: dict[str, float] | None = None) -> HealthScore:
    weights = weights or DIMENSION_WEIGHTS
    rows = []
    for r in results:
        rows.append({
            "check_id": r.check_id, "dimension": r.dimension, "table": r.table, "title": r.title,
            "affected": r.affected, "total": r.total, "rate": r.rate, "score": r.score,
            "weight": r.weight, "assessed": r.assessed, "impact": r.impact, "fix": r.fix, "note": r.note,
        })
    checks = pd.DataFrame(rows)
    assessed = checks[checks["assessed"]]

    dim_rows = []
    for dim in DIMENSIONS:
        sub = assessed[assessed["dimension"] == dim]
        if sub.empty:
            dim_rows.append({"dimension": dim, "score": np.nan, "weight": weights[dim], "checks": 0})
            continue
        s = float(np.average(sub["score"], weights=sub["weight"]))
        dim_rows.append({"dimension": dim, "score": s, "weight": weights[dim], "checks": len(sub)})
    dims = pd.DataFrame(dim_rows)
    live = dims.dropna(subset=["score"])
    total_w = live["weight"].sum()
    overall = float((live["score"] * live["weight"]).sum() / total_w) if total_w else float("nan")

    # how many overall points each check is costing
    dim_check_w = assessed.groupby("dimension")["weight"].sum()
    def recoverable(row) -> float:
        if not row["assessed"] or row["dimension"] not in live["dimension"].values:
            return 0.0
        dim_w = weights[row["dimension"]] / total_w
        return (100 - row["score"]) * row["weight"] / dim_check_w[row["dimension"]] * dim_w
    checks["points_recoverable"] = checks.apply(recoverable, axis=1)

    label, meaning = band(overall)
    return HealthScore(round(overall, 1), label, meaning, dims, checks)
