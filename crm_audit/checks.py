"""The individual data-health checks, grouped into six dimensions.

Every check returns a CheckResult: how many records failed, out of how many,
which record IDs failed, and a plain-English impact + fix. Scoring is done in
scoring.py so the checks stay simple and testable.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from .schema import is_missing, outcome, to_datetime

DIMENSIONS = ["Completeness", "Uniqueness", "Validity", "Consistency", "Freshness", "Integrity"]

EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+'\-]+@[A-Za-z0-9\-]+(\.[A-Za-z0-9\-]+)*\.[A-Za-z]{2,}$")
COMPANY_SUFFIXES = re.compile(
    r"\b(pvt|private|ltd|limited|llp|llc|inc|corp|corporation|co|company|plc|gmbh|solutions|industries|group)\b\.?",
    re.I,
)

# General-purpose synonyms seen in real Indian/UK CRM exports. Fuzzy matching
# catches typos on top of this; anything else is left alone rather than guessed.
SYNONYMS: dict[str, dict[str, str]] = {
    "city": {"bombay": "mumbai", "bangalore": "bengaluru", "blr": "bengaluru", "new delhi": "delhi",
             "calcutta": "kolkata", "madras": "chennai", "poona": "pune", "gurgaon": "gurugram",
             "hyd": "hyderabad", "mum": "mumbai", "ahmadabad": "ahmedabad", "trivandrum": "thiruvananthapuram",
             "baroda": "vadodara", "mysore": "mysuru"},
    "country": {"in": "india", "ind": "india", "bharat": "india", "uk": "united kingdom", "gb": "united kingdom",
                "great britain": "united kingdom", "england": "united kingdom", "us": "united states",
                "usa": "united states", "u s a": "united states", "uae": "united arab emirates"},
    "industry": {"it": "information technology", "i t": "information technology", "info tech": "information technology",
                 "tech": "information technology", "finance": "financial services", "bfsi": "financial services",
                 "fin services": "financial services", "health care": "healthcare", "mfg": "manufacturing",
                 "realty": "real estate", "realestate": "real estate", "edtech": "education",
                 "retail e commerce": "retail", "ecommerce": "retail"},
    "lead_source": {"web": "website", "inbound website": "website", "linked in": "linkedin", "li": "linkedin",
                    "event": "events", "trade show": "events", "cold call": "cold outreach",
                    "google ads": "paid ads", "ppc": "paid ads", "adwords": "paid ads"},
    "stage": {"won": "closed won", "lost": "closed lost"},
}


@dataclass
class CheckResult:
    check_id: str
    dimension: str
    table: str
    title: str
    affected: int
    total: int
    tolerance: float          # failure rate at which this check scores 0
    weight: float = 1.0       # importance inside its dimension
    flagged_ids: list[str] = field(default_factory=list)
    impact: str = ""
    fix: str = ""
    assessed: bool = True
    note: str = ""

    @property
    def rate(self) -> float:
        return self.affected / self.total if self.total else 0.0

    @property
    def score(self) -> float:
        if not self.assessed:
            return np.nan
        return float(max(0.0, 1.0 - self.rate / self.tolerance) * 100)


@dataclass
class AuditContext:
    contacts: pd.DataFrame | None
    deals: pd.DataFrame | None
    as_of: pd.Timestamp

    def has(self, table: str, *cols: str) -> bool:
        df = getattr(self, table)
        return df is not None and all(c in df.columns for c in cols)


def _skip(check_id, dimension, table, title, missing_cols) -> CheckResult:
    return CheckResult(check_id, dimension, table, title, 0, 0, 1.0, assessed=False,
                       note=f"Not assessed — column(s) not found: {', '.join(missing_cols)}")


def _ids(df: pd.DataFrame, mask: pd.Series, id_col: str) -> list[str]:
    if id_col in df.columns:
        return df.loc[mask, id_col].astype(str).tolist()
    return [str(i) for i in df.index[mask]]


def _inr(x: float) -> str:
    if x >= 1e7:
        return f"₹{x / 1e7:,.2f} Cr"
    if x >= 1e5:
        return f"₹{x / 1e5:,.1f} L"
    return f"₹{x:,.0f}"


# ---------------------------------------------------------------------------
# Completeness
# ---------------------------------------------------------------------------
FIELD_RULES = {
    # table, column: (weight, tolerance, why it matters)
    ("contacts", "email"): (3, 0.30, "can't be emailed or matched to other systems"),
    ("contacts", "phone"): (2, 0.40, "can't be called or messaged"),
    ("contacts", "company"): (2, 0.30, "can't be grouped into accounts"),
    ("contacts", "industry"): (2, 0.50, "can't be segmented or used as a model feature"),
    ("contacts", "lead_source"): (2, 0.50, "hide which channels actually bring customers"),
    ("contacts", "owner"): (2, 0.25, "have nobody responsible for following up"),
    ("contacts", "job_title"): (1, 0.60, "can't be targeted by seniority or role"),
    ("deals", "amount"): (3, 0.25, "are missing a value, so pipeline and revenue reports are understated"),
    ("deals", "close_date"): (2, 0.30, "can't be included in forecasts"),
    ("deals", "owner"): (1, 0.25, "have no accountable rep"),
}


def completeness_checks(ctx: AuditContext) -> list[CheckResult]:
    out = []
    for (table, col), (weight, tol, why) in FIELD_RULES.items():
        title = f"Missing {col.replace('_', ' ')} ({table})"
        cid = f"missing_{col}_{table}"
        if not ctx.has(table, col):
            out.append(_skip(cid, "Completeness", table, title, [col]))
            continue
        df = getattr(ctx, table)
        mask = df[col].isna() if col == "amount" else is_missing(df[col])
        id_col = "contact_id" if table == "contacts" else "deal_id"
        n = int(mask.sum())
        out.append(CheckResult(
            cid, "Completeness", table, title, n, len(df), tol, weight, _ids(df, mask, id_col),
            impact=f"{n:,} {table} {why}.",
            fix=f"Make '{col}' a required field on forms/imports; backfill from emails, LinkedIn or enrichment.",
        ))
    return out


# ---------------------------------------------------------------------------
# Uniqueness
# ---------------------------------------------------------------------------
class _UnionFind:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, a: int) -> int:
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def _norm_text(s) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", str(s).lower()).strip() if pd.notna(s) else ""


def company_key(s) -> str:
    if pd.isna(s):
        return ""
    s = COMPANY_SUFFIXES.sub(" ", str(s).lower())
    return re.sub(r"[^a-z0-9]+", "", s)


def _names_match(f1: str, l1: str, f2: str, l2: str) -> bool:
    if not l1 or not l2:
        return False
    if fuzz.ratio(l1, l2) < 85:
        return False
    if not f1 or not f2:
        return False
    if len(f1) <= 2 or len(f2) <= 2:          # "A." vs "Aarav"
        return f1[0] == f2[0] and fuzz.ratio(l1, l2) >= 90
    return fuzz.ratio(f1, f2) >= 85


def find_duplicate_clusters(contacts: pd.DataFrame) -> list[list[int]]:
    """Cluster contact rows that are probably the same person.

    Signals: same normalised email, same last-10 phone digits, or a fuzzy
    name match inside the same company (company suffixes ignored).
    """
    df = contacts.reset_index(drop=True)
    uf = _UnionFind(len(df))

    def link_on(key: pd.Series) -> None:
        key = key[key.astype(bool)]
        for _, idx in key.groupby(key).groups.items():
            idx = list(idx)
            for other in idx[1:]:
                uf.union(idx[0], other)

    if "email" in df:
        em = df["email"].astype("string").str.strip().str.lower().fillna("")
        em = em.where(em.str.match(EMAIL_RE.pattern, case=False).fillna(False), "")
        link_on(em)
    if "phone" in df:
        digits = df["phone"].astype("string").fillna("").str.replace(r"\D", "", regex=True)
        has_letters = df["phone"].astype("string").fillna("").str.contains(r"[A-Za-z]")
        plausible = digits.str.len().between(10, 13) & ~has_letters
        key = digits.str[-10:].where(plausible, "")
        link_on(key)
    if {"first_name", "last_name", "company"} <= set(df.columns):
        f = df["first_name"].map(_norm_text)
        l = df["last_name"].map(_norm_text)
        ck = df["company"].map(company_key)
        for _, idx in ck[ck != ""].groupby(ck[ck != ""]).groups.items():
            idx = list(idx)
            for a_pos in range(len(idx)):
                for b_pos in range(a_pos + 1, len(idx)):
                    a, b = idx[a_pos], idx[b_pos]
                    if _names_match(f[a], l[a], f[b], l[b]):
                        uf.union(a, b)

    clusters: dict[int, list[int]] = {}
    for i in range(len(df)):
        clusters.setdefault(uf.find(i), []).append(i)
    return [c for c in clusters.values() if len(c) > 1]


def uniqueness_checks(ctx: AuditContext) -> list[CheckResult]:
    out = []
    if ctx.contacts is not None:
        df = ctx.contacts.reset_index(drop=True)
        clusters = find_duplicate_clusters(df)
        created = to_datetime(df["created_at"]) if "created_at" in df else pd.Series(pd.NaT, index=df.index)
        created = created.where(created <= ctx.as_of)  # a future date is a typo, not "newest"
        redundant: list[int] = []
        for c in clusters:
            keep = min(c, key=lambda i: (created[i] if pd.notna(created[i]) else pd.Timestamp.max, i))
            redundant += [i for i in c if i != keep]
        mask = pd.Series(False, index=df.index)
        mask[redundant] = True
        n = len(redundant)
        out.append(CheckResult(
            "duplicate_contacts", "Uniqueness", "contacts", "Duplicate contacts", n, len(df), 0.25, 3,
            _ids(df, mask, "contact_id"),
            impact=f"{n:,} redundant records across {len(clusters):,} people — reps may contact the same person twice "
                   f"and contact counts are inflated by {n / max(len(df) - n, 1):.0%}.",
            fix="Merge clusters, keeping the oldest record; add duplicate-blocking on email and phone at entry.",
        ))
        if "contact_id" in df:
            dmask = df["contact_id"].duplicated(keep="first") & df["contact_id"].notna()
            out.append(CheckResult(
                "duplicate_contact_ids", "Uniqueness", "contacts", "Repeated contact IDs", int(dmask.sum()),
                len(df), 0.02, 1, _ids(df, dmask, "contact_id"),
                impact=f"{int(dmask.sum()):,} rows reuse an ID that should be unique.",
                fix="Re-export with the system record ID; IDs must never be edited by hand.",
            ))
    if ctx.has("deals", "deal_id"):
        d = ctx.deals
        dmask = d["deal_id"].duplicated(keep="first") & d["deal_id"].notna()
        out.append(CheckResult(
            "duplicate_deal_ids", "Uniqueness", "deals", "Repeated deal IDs", int(dmask.sum()), len(d), 0.02, 1,
            _ids(d, dmask, "deal_id"),
            impact=f"{int(dmask.sum()):,} deal rows reuse an ID, double-counting pipeline.",
            fix="De-duplicate on the system deal ID before reporting.",
        ))
    return out


# ---------------------------------------------------------------------------
# Validity
# ---------------------------------------------------------------------------
def validity_checks(ctx: AuditContext) -> list[CheckResult]:
    out = []
    if ctx.has("contacts", "email"):
        df = ctx.contacts
        present = ~is_missing(df["email"])
        bad = present & ~df["email"].astype("string").str.strip().str.match(EMAIL_RE.pattern).fillna(False)
        out.append(CheckResult(
            "invalid_email", "Validity", "contacts", "Badly formatted emails", int(bad.sum()), int(present.sum()),
            0.15, 3, _ids(df, bad, "contact_id"),
            impact=f"{int(bad.sum()):,} emails will bounce and hurt sender reputation.",
            fix="Validate email format on entry; fix obvious typos (',in', '@@'); verify the rest with an email checker.",
        ))
    else:
        out.append(_skip("invalid_email", "Validity", "contacts", "Badly formatted emails", ["email"]))

    if ctx.has("contacts", "phone"):
        df = ctx.contacts
        raw = df["phone"].astype("string")
        present = ~is_missing(df["phone"])
        digits = raw.fillna("").str.replace(r"\D", "", regex=True)
        bad = present & (raw.fillna("").str.contains(r"[A-Za-z]") | (digits.str.len() < 10) | (digits.str.len() > 13))
        out.append(CheckResult(
            "invalid_phone", "Validity", "contacts", "Unusable phone numbers", int(bad.sum()), int(present.sum()),
            0.15, 2, _ids(df, bad, "contact_id"),
            impact=f"{int(bad.sum()):,} phone numbers can't be dialled (too short, too long, or contain text).",
            fix="Store phones in E.164 format (+91XXXXXXXXXX) and validate digit count on entry.",
        ))

    if ctx.has("deals", "amount"):
        d = ctx.deals
        amt = pd.to_numeric(d["amount"], errors="coerce")
        bad = amt.notna() & (amt <= 0)
        out.append(CheckResult(
            "non_positive_amount", "Validity", "deals", "Zero or negative deal amounts", int(bad.sum()),
            int(amt.notna().sum()), 0.05, 2, _ids(d, bad, "deal_id"),
            impact=f"{int(bad.sum()):,} deals have amounts ≤ 0, distorting revenue and average deal size.",
            fix="Block non-positive amounts; check whether negatives were meant as refunds/credit notes.",
        ))
    if ctx.has("deals", "probability"):
        d = ctx.deals
        p = pd.to_numeric(d["probability"], errors="coerce")
        bad = p.notna() & ((p < 0) | (p > 100))
        out.append(CheckResult(
            "probability_range", "Validity", "deals", "Deal probability outside 0–100%", int(bad.sum()),
            int(p.notna().sum()), 0.05, 1, _ids(d, bad, "deal_id"),
            impact=f"{int(bad.sum()):,} deals have impossible probabilities.", fix="Constrain to 0–100.",
        ))

    for table, id_col in [("contacts", "contact_id"), ("deals", "deal_id")]:
        if ctx.has(table, "created_at"):
            df = getattr(ctx, table)
            ts = to_datetime(df["created_at"])
            bad = ts > ctx.as_of + pd.Timedelta(days=1)
            out.append(CheckResult(
                f"future_created_{table}", "Validity", table, f"Created dates in the future ({table})",
                int(bad.sum()), int(ts.notna().sum()), 0.02, 1, _ids(df, bad, id_col),
                impact=f"{int(bad.sum()):,} {table} claim to be created after today — usually a date-format mix-up (DD/MM vs MM/DD).",
                fix="Standardise date formats on import (ISO YYYY-MM-DD).",
            ))
    return out


# ---------------------------------------------------------------------------
# Consistency
# ---------------------------------------------------------------------------
def inconsistent_labels(series: pd.Series, synonyms: dict[str, str] | None = None,
                        fuzzy_threshold: int = 88) -> tuple[pd.Series, dict[str, str]]:
    """Flag values written differently from the dominant spelling of the same thing.

    1. normalise case/punctuation, 2. apply known synonyms, 3. fuzzy-merge rare
    spellings into frequent ones, 4. the most common raw spelling in each group
    is treated as the standard; everything else in the group is flagged.
    Returns (mask, {raw_value: suggested_standard}).
    """
    synonyms = synonyms or {}
    raw = series.astype("string")
    present = ~is_missing(series)
    key = raw.fillna("").str.lower().str.replace(r"[^a-z0-9]+", " ", regex=True).str.strip()
    key = key.map(lambda k: synonyms.get(k, k))

    counts = key[present].value_counts()
    merge: dict[str, str] = {}
    keys_by_freq = list(counts.index)
    for i, k in enumerate(keys_by_freq):
        for bigger in keys_by_freq[:i]:
            target = merge.get(bigger, bigger)
            if len(k) >= 4 and fuzz.ratio(k, target) >= fuzzy_threshold:
                merge[k] = target
                break
    group = key.map(lambda k: merge.get(k, k))

    standard = raw[present].groupby(group[present]).agg(lambda s: s.value_counts().index[0])
    expected = group.map(standard)
    mask = present & (raw != expected)
    suggestions = {str(r): str(e) for r, e in zip(raw[mask], expected[mask])}
    return mask.fillna(False).astype(bool), suggestions


def consistency_checks(ctx: AuditContext) -> list[CheckResult]:
    out = []
    specs = [("contacts", "city", 2), ("contacts", "country", 1), ("contacts", "industry", 3),
             ("contacts", "lead_source", 2), ("deals", "stage", 3), ("deals", "lead_source", 1)]
    for table, col, weight in specs:
        title = f"Inconsistent {col.replace('_', ' ')} labels ({table})"
        cid = f"inconsistent_{col}_{table}"
        if not ctx.has(table, col):
            out.append(_skip(cid, "Consistency", table, title, [col]))
            continue
        df = getattr(ctx, table)
        mask, sugg = inconsistent_labels(df[col], SYNONYMS.get(col))
        examples = ", ".join(f"'{k}'→'{v}'" for k, v in list(dict.fromkeys(sugg.items()))[:4])
        n = int(mask.sum())
        out.append(CheckResult(
            cid, "Consistency", table, title, n, int((~is_missing(df[col])).sum()), 0.30, weight,
            _ids(df, mask, "contact_id" if table == "contacts" else "deal_id"),
            impact=f"{n:,} records use a non-standard spelling, splitting reports and segments"
                   + (f" (e.g. {examples})." if examples else "."),
            fix=f"Turn '{col}' into a picklist and remap existing variants to the standard value.",
        ))
    return out


# ---------------------------------------------------------------------------
# Freshness
# ---------------------------------------------------------------------------
def freshness_checks(ctx: AuditContext) -> list[CheckResult]:
    out = []
    if ctx.has("contacts", "last_modified_at"):
        df = ctx.contacts
        ts = to_datetime(df["last_modified_at"])
        stale = ts.notna() & ((ctx.as_of - ts).dt.days > 365)
        out.append(CheckResult(
            "stale_contacts", "Freshness", "contacts", "Contacts not updated in 12+ months", int(stale.sum()),
            int(ts.notna().sum()), 0.50, 2, _ids(df, stale, "contact_id"),
            impact=f"{int(stale.sum()):,} contacts haven't been touched in a year; B2B contact data decays roughly a "
                   f"third per year, so many of these people have likely changed roles.",
            fix="Re-verify or re-enrich stale contacts quarterly; archive ones that bounce.",
        ))
    else:
        out.append(_skip("stale_contacts", "Freshness", "contacts", "Contacts not updated in 12+ months", ["last_modified_at"]))

    if ctx.has("contacts", "last_activity_at"):
        df = ctx.contacts
        ts = to_datetime(df["last_activity_at"])
        cold = ts.isna() | ((ctx.as_of - ts).dt.days > 180)
        never = int(ts.isna().sum())
        out.append(CheckResult(
            "no_recent_activity", "Freshness", "contacts", "No logged activity in 6+ months", int(cold.sum()),
            len(df), 0.80, 2, _ids(df, cold, "contact_id"),
            impact=f"{int(cold.sum()):,} contacts show no logged call/email/meeting in 6 months ({never:,} have none at all) "
                   f"— either they're neglected, or activity isn't being logged.",
            fix="Auto-log email and calendar activity via the CRM's inbox integration instead of relying on manual entry.",
        ))

    if ctx.has("deals", "stage", "close_date"):
        d = ctx.deals
        cd = to_datetime(d["close_date"])
        is_open = outcome(d["stage"]) == "open"
        overdue = is_open & cd.notna() & (cd < ctx.as_of)
        value = pd.to_numeric(d.loc[overdue, "amount"], errors="coerce").clip(lower=0).sum() if "amount" in d else 0
        out.append(CheckResult(
            "overdue_open_deals", "Freshness", "deals", "Open deals past their close date", int(overdue.sum()),
            int(is_open.sum()), 0.60, 3, _ids(d, overdue, "deal_id"),
            impact=f"{int(overdue.sum()):,} open deals ({_inr(value)}) have already passed their close date — the "
                   f"pipeline and forecast are overstated until these are updated or closed out.",
            fix="Weekly pipeline review: every past-due deal gets a new date or is marked Closed Lost.",
        ))
    return out


# ---------------------------------------------------------------------------
# Integrity
# ---------------------------------------------------------------------------
def integrity_checks(ctx: AuditContext) -> list[CheckResult]:
    out = []
    if ctx.has("deals", "contact_id") and ctx.has("contacts", "contact_id"):
        d = ctx.deals
        known = set(ctx.contacts["contact_id"].astype(str))
        has_link = d["contact_id"].notna()
        orphan = has_link & ~d["contact_id"].astype(str).isin(known)
        out.append(CheckResult(
            "orphan_deals", "Integrity", "deals", "Deals linked to a contact that doesn't exist", int(orphan.sum()),
            int(has_link.sum()), 0.10, 3, _ids(d, orphan, "deal_id"),
            impact=f"{int(orphan.sum()):,} deals point to deleted or missing contacts, so they can't be attributed to a customer.",
            fix="Re-link to the correct contact (match on company/email) or restore the deleted contacts.",
        ))
    if ctx.has("deals", "created_at", "close_date"):
        d = ctx.deals
        cr, cl = to_datetime(d["created_at"]), to_datetime(d["close_date"])
        bad = cr.notna() & cl.notna() & (cl < cr)
        out.append(CheckResult(
            "close_before_created", "Integrity", "deals", "Deals that close before they were created",
            int(bad.sum()), int((cr.notna() & cl.notna()).sum()), 0.05, 2, _ids(d, bad, "deal_id"),
            impact=f"{int(bad.sum()):,} deals have impossible timelines, corrupting sales-cycle-length analysis.",
            fix="Add a validation rule: close date ≥ created date.",
        ))
    if ctx.has("deals", "stage", "amount"):
        d = ctx.deals
        amt = pd.to_numeric(d["amount"], errors="coerce")
        won = outcome(d["stage"]) == "won"
        bad = won & (amt.isna() | (amt <= 0))
        out.append(CheckResult(
            "won_without_amount", "Integrity", "deals", "Won deals with no valid amount", int(bad.sum()),
            int(won.sum()), 0.15, 3, _ids(d, bad, "deal_id"),
            impact=f"{int(bad.sum()):,} won deals carry no revenue, so booked revenue is under-reported.",
            fix="Require an amount before a deal can move to Closed Won.",
        ))
    if ctx.has("deals", "stage", "lost_reason"):
        d = ctx.deals
        lost = outcome(d["stage"]) == "lost"
        bad = lost & is_missing(d["lost_reason"])
        out.append(CheckResult(
            "lost_without_reason", "Integrity", "deals", "Lost deals with no lost reason", int(bad.sum()),
            int(lost.sum()), 0.30, 1, _ids(d, bad, "deal_id"),
            impact=f"{int(bad.sum()):,} lost deals don't say why — the most useful learning signal is thrown away.",
            fix="Make lost reason a required picklist when closing a deal as lost.",
        ))
    return out


ALL_CHECKS = [completeness_checks, uniqueness_checks, validity_checks,
              consistency_checks, freshness_checks, integrity_checks]


def run_all_checks(ctx: AuditContext) -> list[CheckResult]:
    results: list[CheckResult] = []
    for fn in ALL_CHECKS:
        results.extend(fn(ctx))
    return results
