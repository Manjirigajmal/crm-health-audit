"""AI-readiness audit: can this CRM data support specific AI use cases?

A clean CRM is not automatically an ML-ready CRM. Each use case below has its
own requirements (volume, label balance, history length, feature coverage)
plus ML-specific risks a normal data check misses, such as target leakage.
Every requirement is graded pass / warn / fail and the use case gets a
traffic-light verdict.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .checks import AuditContext, find_duplicate_clusters, EMAIL_RE
from .schema import is_missing, outcome, to_datetime

STATUS_ORDER = {"pass": 0, "warn": 1, "fail": 2}
VERDICT = {0: "Ready", 1: "Almost ready", 2: "Not ready"}


@dataclass
class Requirement:
    name: str
    status: str            # pass | warn | fail
    detail: str


@dataclass
class UseCase:
    key: str
    name: str
    question: str
    requirements: list[Requirement] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        if not self.requirements:
            return "Not assessed"
        return VERDICT[max(STATUS_ORDER[r.status] for r in self.requirements)]

    @property
    def blockers(self) -> list[Requirement]:
        return [r for r in self.requirements if r.status != "pass"]


def _grade(value: float, good: float, ok: float, higher_is_better: bool = True) -> str:
    if higher_is_better:
        return "pass" if value >= good else "warn" if value >= ok else "fail"
    return "pass" if value <= good else "warn" if value <= ok else "fail"


# ---------------------------------------------------------------------------
# Target-leakage scan
# ---------------------------------------------------------------------------
EXCLUDE_FROM_LEAKAGE = {"deal_id", "deal_name", "stage", "contact_id", "company", "created_at", "close_date"}


def leakage_scan(deals: pd.DataFrame, threshold: float = 0.9) -> list[tuple[str, float, str]]:
    """Find columns that predict won/lost almost perfectly on their own.

    Such fields are usually filled in *because* the deal closed (an auto-set
    probability, a lost-reason only entered for losses). A model trained on
    them looks brilliant in testing and is useless on live, open deals.

    Returns [(column, separation 0–1, how)], where separation is how much of
    the gap between the majority-class baseline and perfect prediction the
    column closes by itself.
    """
    out = outcome(deals["stage"])
    closed = deals[out.isin(["won", "lost"])]
    y = (out[closed.index] == "won").astype(int)
    if len(closed) < 30 or y.nunique() < 2:
        return []
    baseline = max(y.mean(), 1 - y.mean())
    found = []
    for col in closed.columns:
        if col in EXCLUDE_FROM_LEAKAGE:
            continue
        s = closed[col]
        num = pd.to_numeric(s, errors="coerce")
        if num.notna().mean() > 0.8 and num.nunique() <= 20:
            how = "value"
            groups = num.astype(str).where(num.notna(), "__missing__")
        elif num.notna().mean() > 0.8:
            how = "value"
            try:
                bins = pd.qcut(num, q=min(10, num.nunique()), duplicates="drop").astype(str)
            except ValueError:
                bins = num.astype(str)
            groups = bins.where(num.notna(), "__missing__")
        else:
            missing = is_missing(s)
            if s.astype("string").nunique() > 0.5 * len(s) and missing.mean() < 0.05:
                continue  # free text / identifiers
            groups = s.astype("string").where(~missing, "__missing__").fillna("__missing__")
            how = "whether it is filled in" if missing.any() and (~missing).any() else "value"
        table = pd.crosstab(groups, y)
        purity = table.max(axis=1).sum() / len(y)
        separation = (purity - baseline) / (1 - baseline) if baseline < 1 else 0
        if separation >= threshold:
            found.append((col, round(float(separation), 3), how))
    return sorted(found, key=lambda t: -t[1])


# ---------------------------------------------------------------------------
# Shared facts
# ---------------------------------------------------------------------------
def _facts(ctx: AuditContext) -> dict:
    f: dict = {}
    c, d = ctx.contacts, ctx.deals
    if c is not None:
        clusters = find_duplicate_clusters(c)
        f["dup_rate"] = sum(len(x) - 1 for x in clusters) / max(len(c), 1)
        if "email" in c:
            em = c["email"].astype("string").str.strip()
            f["valid_email_share"] = float(em.str.match(EMAIL_RE.pattern).fillna(False).mean())
        if "last_activity_at" in c:
            ts = to_datetime(c["last_activity_at"])
            f["recent_activity_share"] = float((ts.notna() & ((ctx.as_of - ts).dt.days <= 180)).mean())
            f["any_activity_share"] = float(ts.notna().mean())
        if "owner" in c:
            f["owner_share"] = float((~is_missing(c["owner"])).mean())
    if d is not None and "stage" in d:
        out = outcome(d["stage"])
        f["outcome"] = out
        f["n_won"], f["n_lost"] = int((out == "won").sum()), int((out == "lost").sum())
        f["n_closed"] = f["n_won"] + f["n_lost"]
        f["minority_share"] = min(f["n_won"], f["n_lost"]) / f["n_closed"] if f["n_closed"] else 0.0
        amt = pd.to_numeric(d["amount"], errors="coerce") if "amount" in d else pd.Series(np.nan, index=d.index)
        cd = to_datetime(d["close_date"]) if "close_date" in d else pd.Series(pd.NaT, index=d.index)
        cr = to_datetime(d["created_at"]) if "created_at" in d else pd.Series(pd.NaT, index=d.index)
        won = d[out == "won"].assign(_amt=amt, _cd=cd)
        f["won_amount_valid"] = float((won["_amt"] > 0).mean()) if len(won) else 0.0
        won_dates = won["_cd"].dropna()
        f["won_span_months"] = ((won_dates.max() - won_dates.min()).days / 30.44) if len(won_dates) > 1 else 0.0
        closed_mask = out.isin(["won", "lost"])
        closed_dates = cd[closed_mask].dropna()
        f["closed_months_covered"] = int(closed_dates.dt.to_period("M").nunique()) if len(closed_dates) else 0
        f["closed_with_dates"] = int((closed_mask & cd.notna() & cr.notna() & (cd >= cr)).sum())
        is_open = out == "open"
        f["n_open"] = int(is_open.sum())
        f["overdue_share"] = float((is_open & cd.notna() & (cd < ctx.as_of)).sum() / max(is_open.sum(), 1))
        f["open_close_date_share"] = float(cd[is_open].notna().mean()) if is_open.any() else 1.0
        if c is not None and "contact_id" in d and "contact_id" in c:
            known = set(c["contact_id"].astype(str))
            f["won_orphan_share"] = float((~won["contact_id"].astype(str).isin(known)).mean()) if len(won) else 0.0
            buys = won[won["contact_id"].astype(str).isin(known)].groupby("contact_id").size()
            f["n_customers"] = int(len(buys))
            f["repeat_share"] = float((buys >= 2).mean()) if len(buys) else 0.0
            # feature coverage on closed deals, joining contact attributes
            closed = d[closed_mask].merge(c.drop_duplicates("contact_id"), on="contact_id", how="left",
                                          suffixes=("", "_contact"))
            feats = {}
            for col in ["lead_source", "industry", "job_title", "city"]:
                src = col if col in closed else f"{col}_contact"
                if src in closed:
                    feats[col] = float((~is_missing(closed[src])).mean())
            feats["amount"] = float((pd.to_numeric(closed["amount"], errors="coerce") > 0).mean()) if "amount" in closed else 0
            f["feature_coverage"] = feats
        stage_raw = d["stage"].astype("string").str.strip()
        canonical = {"Closed Won", "Closed Lost"}
        closed_raw = stage_raw[closed_mask]
        f["messy_outcome_labels"] = float((~closed_raw.isin(canonical)).mean()) if len(closed_raw) else 0.0
        f["leakage"] = leakage_scan(d)
    return f


# ---------------------------------------------------------------------------
# Use cases
# ---------------------------------------------------------------------------
def assess(ctx: AuditContext) -> list[UseCase]:
    f = _facts(ctx)
    cases: list[UseCase] = []

    # 1. Lead / deal scoring --------------------------------------------------
    uc = UseCase("lead_scoring", "Lead & deal scoring", "Which open deals are most likely to close?")
    if "n_closed" in f:
        n = f["n_closed"]
        uc.requirements.append(Requirement(
            "Enough closed deals to learn from", _grade(n, 300, 100),
            f"{n:,} closed deals ({f['n_won']:,} won / {f['n_lost']:,} lost). Aim for 300+."))
        ms = f["minority_share"]
        rare = "lost" if f["n_lost"] < f["n_won"] else "won"
        uc.requirements.append(Requirement(
            "Both outcomes well represented", _grade(ms, 0.15, 0.05),
            f"Only {ms:.0%} of closed deals are {rare}. Teams often stop logging {rare} deals, so the model "
            f"can't learn what failure looks like. Aim for 15%+, or back-fill old {rare} deals."))
        if f.get("feature_coverage"):
            cov = f["feature_coverage"]
            avg = float(np.mean(list(cov.values())))
            weakest = min(cov, key=cov.get)
            uc.requirements.append(Requirement(
                "Predictive fields filled in", _grade(avg, 0.85, 0.65),
                f"Key fields are {avg:.0%} complete on closed deals on average; weakest is '{weakest}' at {cov[weakest]:.0%}."))
        uc.requirements.append(Requirement(
            "Outcome labels standardised", _grade(f["messy_outcome_labels"], 0.02, 0.10, higher_is_better=False),
            f"{f['messy_outcome_labels']:.0%} of closed deals use a non-standard stage label (e.g. 'Won', 'closed won')."))
        leaks = f["leakage"]
        if leaks:
            names = ", ".join(f"'{c}' (by {how})" for c, _, how in leaks)
            verb, pron = ("predicts", "It") if len(leaks) == 1 else ("predict", "They")
            uc.requirements.append(Requirement(
                "No target leakage", "warn",
                f"{names} {verb} the outcome almost perfectly — a sign {pron.lower()} {'is' if len(leaks) == 1 else 'are'} only filled in after a deal closes. "
                f"{pron} must be excluded from training or the model will look great in testing and fail on live deals."))
        else:
            uc.requirements.append(Requirement("No target leakage", "pass", "No field separates won from lost suspiciously well."))
    cases.append(uc)

    # 2. Churn prediction ---------------------------------------------------------
    uc = UseCase("churn", "Churn prediction", "Which customers are about to stop buying?")
    if "n_customers" in f:
        uc.requirements.append(Requirement(
            "Enough customers", _grade(f["n_customers"], 300, 100),
            f"{f['n_customers']:,} customers with at least one won deal. Aim for 300+."))
        uc.requirements.append(Requirement(
            "Repeat-purchase history", _grade(f["repeat_share"], 0.20, 0.10),
            f"{f['repeat_share']:.0%} of customers have bought more than once. Churn is only definable for customers "
            f"expected to come back."))
        uc.requirements.append(Requirement(
            "Long enough history", _grade(f["won_span_months"], 24, 12),
            f"Won deals span {f['won_span_months']:.0f} months. You need at least one full renewal/repurchase cycle; aim for 24."))
    if "dup_rate" in f:
        uc.requirements.append(Requirement(
            "One record per customer", _grade(f["dup_rate"], 0.03, 0.10, higher_is_better=False),
            f"{f['dup_rate']:.0%} of contacts are duplicates. Duplicates split one customer's history across records, "
            f"which makes active customers look churned."))
    if "any_activity_share" in f:
        uc.requirements.append(Requirement(
            "Engagement signals logged", _grade(f["any_activity_share"], 0.70, 0.40),
            f"{f['any_activity_share']:.0%} of contacts have any logged activity. Engagement drop-off is usually the "
            f"strongest churn signal."))
    cases.append(uc)

    # 3. Customer lifetime value --------------------------------------------------
    uc = UseCase("clv", "Customer lifetime value", "How much will each customer be worth over the next 12 months?")
    if "won_amount_valid" in f:
        uc.requirements.append(Requirement(
            "Revenue recorded on won deals", _grade(f["won_amount_valid"], 0.95, 0.85),
            f"{f['won_amount_valid']:.0%} of won deals have a valid amount."))
        uc.requirements.append(Requirement(
            "Enough transaction history", _grade(f["won_span_months"], 12, 6),
            f"Won deals span {f['won_span_months']:.0f} months; BG/NBD-style models need 12+."))
    if "repeat_share" in f:
        uc.requirements.append(Requirement(
            "Repeat buyers present", _grade(f["repeat_share"], 0.20, 0.10),
            f"{f['repeat_share']:.0%} repeat buyers."))
        uc.requirements.append(Requirement(
            "Revenue linked to customers", _grade(f["won_orphan_share"], 0.02, 0.08, higher_is_better=False),
            f"{f['won_orphan_share']:.0%} of won deals point to a contact that doesn't exist, so that revenue can't be attributed."))
    cases.append(uc)

    # 4. Sales forecasting ---------------------------------------------------------
    uc = UseCase("forecasting", "Sales forecasting", "How much revenue will close next month/quarter?")
    if "closed_with_dates" in f:
        uc.requirements.append(Requirement(
            "Closed deals with clean timelines", _grade(f["closed_with_dates"], 200, 80),
            f"{f['closed_with_dates']:,} closed deals have a valid created → close timeline."))
        uc.requirements.append(Requirement(
            "Months of history", _grade(f["closed_months_covered"], 18, 12),
            f"Closed deals cover {f['closed_months_covered']} distinct months; seasonality needs 18+."))
        uc.requirements.append(Requirement(
            "Pipeline is current", _grade(f["overdue_share"], 0.15, 0.35, higher_is_better=False),
            f"{f['overdue_share']:.0%} of open deals are already past their close date. A forecast built on a stale "
            f"pipeline will be systematically too optimistic."))
        uc.requirements.append(Requirement(
            "Open deals have close dates", _grade(f["open_close_date_share"], 0.95, 0.80),
            f"{f['open_close_date_share']:.0%} of open deals have an expected close date."))
    cases.append(uc)

    # 5. AI sales assistant / agents ---------------------------------------------
    uc = UseCase("ai_agent", "AI sales assistant / agents",
                 "Can an AI agent draft follow-ups, prioritise reps' days, and update records?")
    if "valid_email_share" in f:
        uc.requirements.append(Requirement(
            "Reachable contacts", _grade(f["valid_email_share"], 0.90, 0.75),
            f"{f['valid_email_share']:.0%} of contacts have a valid email address."))
    if "recent_activity_share" in f:
        uc.requirements.append(Requirement(
            "Interaction history for context", _grade(f["recent_activity_share"], 0.60, 0.30),
            f"Only {f['recent_activity_share']:.0%} of contacts have activity logged in the last 6 months — an agent "
            f"drafting follow-ups would be working blind for the rest."))
    if "owner_share" in f:
        uc.requirements.append(Requirement(
            "Clear ownership", _grade(f["owner_share"], 0.95, 0.85),
            f"{f['owner_share']:.0%} of contacts have an owner the agent can route work to."))
    if "dup_rate" in f:
        uc.requirements.append(Requirement(
            "No duplicate people", _grade(f["dup_rate"], 0.03, 0.10, higher_is_better=False),
            f"{f['dup_rate']:.0%} duplicates — an agent would message the same person twice."))
    cases.append(uc)

    return cases
