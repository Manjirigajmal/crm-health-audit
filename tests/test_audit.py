import numpy as np
import pandas as pd
import pytest

from crm_audit.audit import run_audit
from crm_audit.checks import CheckResult, inconsistent_labels
from crm_audit.evaluate import evaluate
from crm_audit.readiness import leakage_scan
from crm_audit.scoring import score
from crm_audit.synthetic import generate

AS_OF = "2026-10-01"


@pytest.fixture(scope="module")
def crm():
    return generate(seed=7, as_of=AS_OF)


@pytest.fixture(scope="module")
def result(crm):
    return run_audit(crm.contacts, crm.deals, as_of=AS_OF)


def test_detects_planted_issues(result, crm):
    ev = evaluate(result, crm.issues)
    assert ev.caught.sum() / ev.planted.sum() > 0.93      # overall recall
    assert ev.caught.sum() / ev.flagged.sum() > 0.98      # overall precision
    dup = ev.set_index("planted_issue").loc["duplicate_contact"]
    assert dup.recall > 0.95 and dup.precision > 0.95


def test_score_is_bounded_and_labelled(result):
    assert 0 <= result.health.overall <= 100
    assert result.health.label in {"Healthy", "Fair", "Needs attention", "Critical"}
    assert result.health.dimensions["score"].between(0, 100).all()


def test_clean_data_scores_higher(crm):
    clean = generate(seed=7, as_of=AS_OF)
    # strip the planted duplicates: contacts never logged as duplicate
    dup_ids = set(clean.issues.loc[clean.issues.issue == "duplicate_contact", "record_id"])
    deduped = clean.contacts[~clean.contacts.contact_id.isin(dup_ids)]
    before = run_audit(clean.contacts, clean.deals, as_of=AS_OF).health.overall
    after = run_audit(deduped, clean.deals, as_of=AS_OF).health.overall
    assert after > before


def test_hubspot_style_headers_are_mapped():
    df = pd.DataFrame({
        "Record ID": ["1", "2"], "First Name": ["Asha", "Ravi"], "Last Name": ["Rao", "Shah"],
        "Email": ["asha@x.in", "bad-email"], "Phone Number": ["+91 9876543210", "123"],
        "Company Name": ["X Ltd", "Y Ltd"], "Create Date": ["2026-01-01", "2026-02-01"],
    })
    r = run_audit(contacts=df, as_of=AS_OF)
    assert r.column_map["contacts"]["email"] == "Email"
    invalid = next(c for c in r.results if c.check_id == "invalid_email")
    assert invalid.affected == 1


def test_missing_columns_are_skipped_not_crashing():
    df = pd.DataFrame({"contact_id": ["a", "b"], "email": ["a@b.com", None]})
    r = run_audit(contacts=df, as_of=AS_OF)
    skipped = [c for c in r.results if not c.assessed]
    assert skipped and all("Not assessed" in c.note for c in skipped)
    assert not np.isnan(r.health.overall)


def test_check_score_formula():
    c = CheckResult("x", "Completeness", "contacts", "x", affected=10, total=100, tolerance=0.2)
    assert c.score == pytest.approx(50.0)
    c0 = CheckResult("y", "Completeness", "contacts", "y", affected=50, total=100, tolerance=0.2)
    assert c0.score == 0.0
    h = score([c])
    assert h.overall == pytest.approx(50.0)


def test_inconsistent_labels_uses_synonyms_and_fuzzy():
    s = pd.Series(["Mumbai"] * 10 + ["Bombay", "mumbai", "Mumbaai", "Pune", "Pune"])
    mask, sugg = inconsistent_labels(s, {"bombay": "mumbai"})
    assert mask.sum() == 3 and sugg["Bombay"] == "Mumbai"


def test_leakage_scan_flags_post_outcome_fields():
    rng = np.random.default_rng(0)
    n = 400
    won = rng.random(n) < 0.7
    deals = pd.DataFrame({
        "deal_id": range(n),
        "stage": np.where(won, "Closed Won", "Closed Lost"),
        "amount": rng.lognormal(10, 1, n),                       # unrelated
        "lead_source": rng.choice(["Web", "Referral", "Ads"], n),  # unrelated
        "lost_reason": np.where(won, None, "Price"),             # leaks
    })
    leaks = [c for c, _, _ in leakage_scan(deals)]
    assert leaks == ["lost_reason"]
