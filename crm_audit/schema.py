"""Map whatever column names a CRM export uses onto one canonical schema.

HubSpot, Zoho, Salesforce and Pipedrive all name the same fields differently
("Email" / "Email Address" / "E-mail"). The audit only ever talks about the
canonical names below; anything it can't find is simply reported as
"not assessed" instead of crashing.
"""
from __future__ import annotations

import re

import pandas as pd

CONTACT_ALIASES: dict[str, list[str]] = {
    "contact_id": ["contact_id", "record_id", "id", "contact id", "lead id", "contactid"],
    "first_name": ["first_name", "first name", "firstname", "given name"],
    "last_name": ["last_name", "last name", "lastname", "surname", "family name"],
    "full_name": ["full_name", "name", "contact name", "full name"],
    "email": ["email", "email address", "e-mail", "email_address", "work email"],
    "phone": ["phone", "phone number", "mobile", "mobile phone", "phone_number", "contact number"],
    "company": ["company", "company name", "account", "account name", "organization", "organisation"],
    "job_title": ["job_title", "job title", "title", "designation", "role"],
    "industry": ["industry", "sector", "vertical"],
    "city": ["city", "town", "mailing city"],
    "country": ["country", "country/region", "mailing country"],
    "lead_source": ["lead_source", "lead source", "source", "original source", "lead_channel"],
    "owner": ["owner", "contact owner", "lead owner", "account owner", "assigned to", "sales rep"],
    "created_at": ["created_at", "create date", "created date", "created time", "date created", "createddate"],
    "last_modified_at": ["last_modified_at", "last modified date", "last modified", "modified time", "updated_at"],
    "last_activity_at": ["last_activity_at", "last activity date", "last activity", "last contacted", "last engagement date"],
}

DEAL_ALIASES: dict[str, list[str]] = {
    "deal_id": ["deal_id", "record_id", "id", "deal id", "opportunity id", "opportunityid"],
    "deal_name": ["deal_name", "deal name", "opportunity name", "name"],
    "contact_id": ["contact_id", "associated contact id", "contact id", "primary contact id", "contactid"],
    "company": ["company", "account name", "associated company", "company name"],
    "amount": ["amount", "deal value", "value", "deal amount", "opportunity amount", "revenue"],
    "stage": ["stage", "deal stage", "dealstage", "stagename", "status"],
    "probability": ["probability", "deal probability", "win probability"],
    "lead_source": ["lead_source", "lead source", "source", "original source"],
    "owner": ["owner", "deal owner", "opportunity owner", "assigned to"],
    "created_at": ["created_at", "create date", "created date", "created time", "createddate"],
    "close_date": ["close_date", "close date", "closing date", "expected close date", "closedate"],
    "lost_reason": ["lost_reason", "lost reason", "closed lost reason", "loss reason"],
}

WON_PATTERNS = re.compile(r"^\s*(closed[\s\-_]*)?won\s*$", re.I)
LOST_PATTERNS = re.compile(r"^\s*(closed[\s\-_]*)?lost\s*$", re.I)

MISSING_TOKENS = {"", "n/a", "na", "-", "--", "unknown", "null", "none", "tbd", "nil", "0", "test", "?"}


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(name).lower()).strip()


def standardize_columns(df: pd.DataFrame, aliases: dict[str, list[str]]) -> tuple[pd.DataFrame, dict[str, str]]:
    """Rename columns to canonical names. Returns the renamed frame and {canonical: original}."""
    lookup = {_norm(c): c for c in df.columns}
    mapping: dict[str, str] = {}
    used: set[str] = set()
    for canonical, options in aliases.items():
        for opt in options:
            original = lookup.get(_norm(opt))
            if original is not None and original not in used:
                mapping[canonical] = original
                used.add(original)
                break
    renamed = df.rename(columns={v: k for k, v in mapping.items()})
    if "full_name" in renamed and "first_name" not in renamed:
        parts = renamed["full_name"].astype("string").str.strip().str.split(r"\s+", n=1, expand=True)
        renamed["first_name"] = parts[0]
        renamed["last_name"] = parts[1] if parts.shape[1] > 1 else pd.NA
    return renamed, mapping


def is_missing(series: pd.Series) -> pd.Series:
    """True for NaN, empty strings and placeholder junk like 'N/A', '-', 'unknown'."""
    s = series.astype("string")
    return s.isna() | s.str.strip().str.lower().isin(MISSING_TOKENS)


def to_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", dayfirst=False)


def outcome(stage: pd.Series) -> pd.Series:
    """Map raw stage labels to 'won' / 'lost' / 'open' (tolerates messy spellings)."""
    s = stage.astype("string").fillna("")
    out = pd.Series("open", index=stage.index, dtype="object")
    out[s.str.match(WON_PATTERNS)] = "won"
    out[s.str.match(LOST_PATTERNS)] = "lost"
    return out
