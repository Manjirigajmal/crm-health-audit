"""Streamlit front end.

    streamlit run app.py                 # public demo: sample data only
    streamlit run app.py -- --private    # private mode on your own machine: upload client exports

The public demo never accepts uploads, so real customer data is only ever
processed locally in private mode.
"""
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

from crm_audit.audit import run_audit
from crm_audit.report import render_html

SAMPLE = Path(__file__).parent / "sample_data"
PRIVATE = "--private" in sys.argv[1:]
CONTACT_EMAIL = "careermanjiri@gmail.com"
LINKEDIN = "https://www.linkedin.com/in/manjirigajmal"
MAILTO = (f"mailto:{CONTACT_EMAIL}?subject=CRM%20audit%20request"
          "&body=Hi%20Manjiri%2C%20I%27d%20like%20an%20audit%20of%20our%20CRM%20data.")
VERDICT_ICON = {"Ready": "🟢", "Almost ready": "🟡", "Not ready": "🔴", "Not assessed": "⚪"}
STATUS_ICON = {"pass": "✅", "warn": "⚠️", "fail": "❌"}

st.set_page_config(page_title="CRM Health Audit", page_icon="🩺", layout="wide")
st.title("CRM Data Health & AI-Readiness Audit")
st.caption("A 0–100 health score for CRM data, the problems costing the most, "
           "and whether the data can support AI use cases.")


def contact_box():
    with st.container(border=True):
        st.markdown("**Want an audit of your own CRM?**")
        st.caption("Your data is audited privately on my machine, never on this public demo. "
                   "Free for the first 5 businesses.")
        st.link_button("✉️ Request an audit", MAILTO, width="stretch")
        st.link_button("💼 Connect on LinkedIn", LINKEDIN, width="stretch")


with st.sidebar:
    st.header("Data")
    contacts_file = deals_file = None
    if PRIVATE:
        use_sample = st.toggle("Use the messy sample CRM", value=True)
        if not use_sample:
            contacts_file = st.file_uploader("Contacts export (.csv / .xlsx)", type=["csv", "xlsx"])
            deals_file = st.file_uploader("Deals / opportunities export (.csv / .xlsx)", type=["csv", "xlsx"])
        client = st.text_input("Client name for the report", "Sample D2C Brand" if use_sample else "My Company")
        as_of = st.date_input("Audit date", pd.Timestamp("2026-10-01") if use_sample else pd.Timestamp.today())
    else:
        use_sample = True
        client = "Sample D2C Brand"
        as_of = pd.Timestamp("2026-10-01").date()
        st.info("This public demo runs on a **sample CRM** (1,320 contacts, 1,500 deals) "
                "with deliberately messy data.")
    st.divider()
    contact_box()

if not PRIVATE:
    a, b = st.columns([4, 1])
    a.info("👋 You're viewing a demo on sample data. To audit your own CRM, request a private audit. "
           "Your data never touches this public app.")
    b.link_button("Request an audit", MAILTO, type="primary", width="stretch")

if use_sample:
    contacts_file, deals_file = SAMPLE / "contacts.csv", SAMPLE / "deals.csv"

if contacts_file is None and deals_file is None:
    st.info("Upload at least one export in the sidebar, or switch on the sample data.")
    st.stop()


@st.cache_data(show_spinner="Auditing your CRM…")
def audit(c_bytes, c_name, d_bytes, d_name, as_of_str):
    import io

    def load(b, name):
        if b is None:
            return None
        return pd.read_excel(io.BytesIO(b)) if name.endswith((".xlsx", ".xls")) else pd.read_csv(io.BytesIO(b), low_memory=False)

    return run_audit(load(c_bytes, c_name), load(d_bytes, d_name), as_of=as_of_str)


def as_bytes(f):
    if f is None:
        return None, ""
    if isinstance(f, Path):
        return f.read_bytes(), f.name
    return f.getvalue(), f.name.lower()


cb, cn = as_bytes(contacts_file)
db, dn = as_bytes(deals_file)
result = audit(cb, cn, db, dn, str(as_of))
h = result.health

# ---- headline ------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Health score", f"{h.overall:.0f} / 100", h.label, delta_color="off")
c2.metric("Records audited", f"{result.n_contacts + result.n_deals:,}")
flag = result.flagged_records()
c3.metric("Records with an issue", f"{flag[['table', 'record_id']].drop_duplicates().shape[0]:,}")
c4.metric("Points from top-5 fixes", f"+{result.top_issues()['points_recoverable'].sum():.0f}")
st.write(h.meaning)

# ---- dimensions ------------------------------------------------------------
st.subheader("Score by dimension")
st.bar_chart(h.dimensions.set_index("dimension")["score"], horizontal=True, height=260)

# ---- top issues ---------------------------------------------------------------
st.subheader("Top 5 problems, ranked by impact")
for i, row in enumerate(result.top_issues().itertuples(), 1):
    with st.container(border=True):
        a, b = st.columns([5, 1])
        a.markdown(f"**{i}. {row.title}** — {row.affected:,} of {row.total:,} records ({row.rate:.1%})")
        b.markdown(f"**+{row.points_recoverable:.1f} pts**")
        st.write(row.impact)
        st.caption(f"Fix: {row.fix}")

# ---- AI readiness ---------------------------------------------------------------
st.subheader("AI-readiness")
cols = st.columns(len(result.use_cases))
for col, u in zip(cols, result.use_cases):
    col.markdown(f"{VERDICT_ICON[u.verdict]} **{u.name}**  \n{u.verdict}")
for u in result.use_cases:
    with st.expander(f"{VERDICT_ICON[u.verdict]} {u.name} — {u.question}"):
        for r in u.requirements:
            st.markdown(f"{STATUS_ICON[r.status]} **{r.name}** — {r.detail}")

# ---- details + downloads ------------------------------------------------------
with st.expander("All checks"):
    st.dataframe(h.checks[["dimension", "title", "affected", "total", "rate", "score", "points_recoverable"]]
                 .round(3), hide_index=True, width="stretch")
with st.expander("Column mapping used"):
    st.json(result.column_map)

st.subheader("Downloads")
d1, d2, d3 = st.columns(3)
d1.download_button("📄 Full HTML report", render_html(result, client), file_name="crm_health_report.html",
                   mime="text/html")
d2.download_button("🧾 Flagged records (CSV)", flag.to_csv(index=False), file_name="flagged_records.csv",
                   mime="text/csv")
d3.download_button("{ } Audit JSON", json.dumps(result.to_dict(), indent=2, default=str),
                   file_name="audit.json", mime="application/json")

if not PRIVATE:
    st.divider()
    left, mid, right = st.columns([1, 2, 1])
    with mid:
        st.markdown("### Want this for your own CRM?")
        st.write("Send me an export from HubSpot, Zoho, Salesforce, Pipedrive or Excel. I'll audit it privately "
                 "and walk you through the results and fixes. **Free for the first 5 businesses.**")
        c1, c2 = st.columns(2)
        c1.link_button("✉️ Request an audit", MAILTO, type="primary", width="stretch")
        c2.link_button("💼 Connect on LinkedIn", LINKEDIN, width="stretch")
