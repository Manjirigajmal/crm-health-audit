"""Generate a realistic, deliberately messy CRM export (contacts + deals).

Every problem injected into the data is logged in a ground-truth table, so the
audit can be scored on how many of the planted issues it actually catches.
"""
from __future__ import annotations

import random
import string
from dataclasses import dataclass

import numpy as np
import pandas as pd

FIRST_NAMES = [
    "Aarav", "Vivaan", "Aditya", "Arjun", "Rohan", "Karan", "Rahul", "Sahil", "Nikhil", "Varun",
    "Ananya", "Diya", "Isha", "Priya", "Sneha", "Kavya", "Neha", "Pooja", "Riya", "Meera",
    "Amit", "Vikram", "Sanjay", "Rajesh", "Suresh", "Deepak", "Manish", "Harsh", "Yash", "Kunal",
    "Anjali", "Shreya", "Tanvi", "Nisha", "Aditi", "Swati", "Divya", "Komal", "Payal", "Ritika",
    "James", "Oliver", "Emma", "Sophie", "Daniel", "Hannah", "Liam", "Grace", "Ethan", "Chloe",
]
LAST_NAMES = [
    "Sharma", "Verma", "Patel", "Shah", "Mehta", "Iyer", "Nair", "Reddy", "Rao", "Gupta",
    "Joshi", "Kulkarni", "Desai", "Kapoor", "Malhotra", "Bose", "Banerjee", "Chatterjee", "Das", "Singh",
    "Agarwal", "Bhatia", "Chopra", "Pillai", "Menon", "Thakur", "Pandey", "Mishra", "Saxena", "Khanna",
    "Smith", "Brown", "Taylor", "Wilson", "Clarke", "Hughes", "Walker", "Wright", "Hall", "Green",
]
COMPANY_A = ["Apex", "Nova", "Zenith", "Blue Lotus", "Sahyadri", "Vertex", "Orbit", "Pinnacle", "Kestrel",
             "Indus", "Saffron", "Monsoon", "Granite", "Lumen", "Harbor", "Cedar", "Quantum", "Everest",
             "Coral", "Silverline", "Banyan", "Peacock", "Trident", "Aurora", "Meridian"]
COMPANY_B = ["Logistics", "Retail", "Foods", "Health", "Tech", "Textiles", "Realty", "Learning",
             "Finserv", "Motors", "Labs", "Hospitality", "Pharma", "Analytics", "Ventures"]
COMPANY_C = ["Pvt Ltd", "Ltd", "LLP", "Solutions", "Industries", "Group"]

CITIES = {
    "Mumbai": ["Bombay", "mumbai", "MUMBAI", "Mumbai ", "Mum"],
    "Pune": ["pune", "Poona", "PUNE"],
    "Bengaluru": ["Bangalore", "bengaluru", "Banglore", "BLR"],
    "Delhi": ["New Delhi", "delhi", "Dilli", "NCR Delhi"],
    "Hyderabad": ["hyderabad", "Hyd", "Hyderbad"],
    "Chennai": ["Madras", "chennai"],
    "Ahmedabad": ["Ahmadabad", "ahmedabad", "Amdavad"],
    "Kolkata": ["Calcutta", "kolkata"],
}
INDUSTRIES = {
    "Information Technology": ["IT", "Info Tech", "information technology", "I.T."],
    "Retail": ["retail", "Retail & E-commerce", "RETAIL"],
    "Healthcare": ["Health Care", "healthcare", "Medical"],
    "Manufacturing": ["manufacturing", "Mfg", "Manufacture"],
    "Financial Services": ["Finance", "BFSI", "financial services", "Fin Services"],
    "Education": ["education", "EdTech", "Edu"],
    "Real Estate": ["real estate", "Realty", "RealEstate"],
    "Hospitality": ["hospitality", "Hotels", "F&B"],
}
COUNTRIES = {"India": ["IN", "IND", "india", "Bharat"]}
LEAD_SOURCES = {
    "Website": ["website", "Web", "Inbound - Website"],
    "LinkedIn": ["linkedin", "Linked In", "LI"],
    "Referral": ["referral", "Referal", "Word of mouth"],
    "Events": ["Event", "events", "Trade Show"],
    "Cold Outreach": ["Cold Call", "cold outreach", "Outbound"],
    "Paid Ads": ["Google Ads", "paid ads", "PPC"],
}
OWNERS = ["Priya Nair", "Rohit Kulkarni", "Ayesha Khan", "Vikram Rao",
          "Sneha Iyer", "Arjun Mehta", "Farhan Shaikh", "Neha Joshi"]
JOB_TITLES = ["Founder", "CEO", "Head of Sales", "Marketing Manager", "Operations Manager",
              "Procurement Lead", "CTO", "Finance Manager", "Business Analyst", "Director"]
OPEN_STAGES = ["Prospecting", "Qualification", "Proposal", "Negotiation"]
PLACEHOLDERS = ["N/A", "-", "unknown", "NA", "null", "TBD", " "]


@dataclass
class SyntheticCRM:
    contacts: pd.DataFrame
    deals: pd.DataFrame
    issues: pd.DataFrame  # ground truth: table, record_id, issue


def _slug(s: str) -> str:
    return "".join(ch for ch in s.lower() if ch.isalnum())


def _phone(rng: random.Random) -> str:
    return "+91 " + str(rng.randint(7, 9)) + "".join(str(rng.randint(0, 9)) for _ in range(9))


def generate(n_contacts: int = 1200, n_deals: int = 1500, seed: int = 42,
             as_of: str = "2026-10-01") -> SyntheticCRM:
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    as_of_ts = pd.Timestamp(as_of)
    start_ts = as_of_ts - pd.Timedelta(days=700)
    issues: list[dict] = []

    def log(table: str, rid: str, issue: str) -> None:
        issues.append({"table": table, "record_id": rid, "issue": issue})

    # ---------- companies ----------
    companies = []
    seen = set()
    while len(companies) < 320:
        name = f"{rng.choice(COMPANY_A)} {rng.choice(COMPANY_B)} {rng.choice(COMPANY_C)}"
        if name not in seen:
            seen.add(name)
            companies.append({
                "name": name,
                "domain": _slug(name.rsplit(" ", 1)[0] if name.endswith(("Pvt Ltd",)) else name)[:18] + ".in",
                "industry": rng.choice(list(INDUSTRIES)),
                "city": rng.choices(list(CITIES), weights=[30, 15, 18, 14, 8, 6, 5, 4])[0],
            })

    # ---------- clean contacts ----------
    rows = []
    used_people: set[tuple[str, str, str]] = set()
    for i in range(n_contacts):
        co = rng.choice(companies)
        fn, ln = rng.choice(FIRST_NAMES), rng.choice(LAST_NAMES)
        while (fn, ln, co["name"]) in used_people:   # distinct people get distinct identities
            fn, ln = rng.choice(FIRST_NAMES), rng.choice(LAST_NAMES)
        used_people.add((fn, ln, co["name"]))
        created = start_ts + pd.Timedelta(days=int(np_rng.integers(0, 690)))
        # modification recency: a long tail of records nobody has touched in ages
        days_since_mod = int(np_rng.exponential(200))
        modified = max(created, as_of_ts - pd.Timedelta(days=days_since_mod))
        has_activity = rng.random() > 0.35
        activity = (max(created, as_of_ts - pd.Timedelta(days=int(np_rng.exponential(120)))) if has_activity else pd.NaT)
        rows.append({
            "contact_id": f"C{10000 + i}",
            "first_name": fn,
            "last_name": ln,
            "email": f"{fn.lower()}.{ln.lower()}@{co['domain']}",
            "phone": _phone(rng),
            "company": co["name"],
            "job_title": rng.choice(JOB_TITLES),
            "industry": co["industry"],
            "city": co["city"],
            "country": "India",
            "lead_source": rng.choices(list(LEAD_SOURCES), weights=[25, 22, 18, 10, 15, 10])[0],
            "owner": rng.choice(OWNERS),
            "created_at": created,
            "last_modified_at": modified,
            "last_activity_at": activity,
        })
    contacts = pd.DataFrame(rows)

    # ---------- deals (generated from clean contacts, before duplicates) ----------
    # ~30% of contacts are repeat buyers -> gives churn/CLV something to learn from
    buyer_ids = contacts["contact_id"].tolist()
    weights = np.where(np_rng.random(len(buyer_ids)) < 0.3, 4.0, 1.0)
    weights = weights / weights.sum()
    deal_rows = []
    for j in range(n_deals):
        cid = buyer_ids[int(np_rng.choice(len(buyer_ids), p=weights))]
        c = contacts.loc[contacts["contact_id"] == cid].iloc[0]
        created = c["created_at"] + pd.Timedelta(days=int(np_rng.integers(0, 120)))
        if created > as_of_ts:
            created = as_of_ts - pd.Timedelta(days=int(np_rng.integers(1, 30)))
        age = (as_of_ts - created).days
        amount = float(np.round(np_rng.lognormal(mean=12.2, sigma=0.8), -2))  # ~ INR 2 lakh median
        # older deals are mostly closed; lost deals are under-logged (common in real CRMs)
        if age > 90 and rng.random() < 0.85:
            stage = "Closed Won" if rng.random() < 0.91 else "Closed Lost"
        else:
            stage = rng.choice(OPEN_STAGES)
        close_date = created + pd.Timedelta(days=int(np_rng.integers(20, 120)))
        closed = stage.startswith("Closed")
        if closed and close_date > as_of_ts:
            close_date = as_of_ts - pd.Timedelta(days=int(np_rng.integers(1, 20)))
        prob = {"Prospecting": 10, "Qualification": 25, "Proposal": 50, "Negotiation": 75,
                "Closed Won": 100, "Closed Lost": 0}[stage]
        deal_rows.append({
            "deal_id": f"D{50000 + j}",
            "deal_name": f"{c['company'].split()[0]} - {rng.choice(['Annual plan', 'Pilot', 'Expansion', 'Renewal', 'Analytics setup', 'Consulting'])}",
            "contact_id": cid,
            "company": c["company"],
            "amount": amount,
            "currency": "INR",
            "stage": stage,
            "probability": prob,                       # set automatically by stage -> target leakage
            "lead_source": c["lead_source"],
            "owner": c["owner"],
            "created_at": created,
            "close_date": close_date,
            "lost_reason": rng.choice(["Price", "Went with competitor", "No budget", "No response", "Timing"]) if stage == "Closed Lost" else np.nan,
        })
    deals = pd.DataFrame(deal_rows)

    # ---------- inject duplicate contacts ----------
    n_dups = int(n_contacts * 0.10)
    dup_sources = contacts.sample(n_dups, random_state=seed)
    dup_rows = []
    for k, (_, src) in enumerate(dup_sources.iterrows()):
        d = src.copy()
        d["contact_id"] = f"C{10000 + n_contacts + k}"
        variant = rng.choice(["case", "initial", "typo", "phone_fmt", "email_case", "company_suffix"])
        if variant == "case":
            d["first_name"], d["last_name"] = d["first_name"].upper(), d["last_name"].lower()
        elif variant == "initial":
            d["first_name"] = d["first_name"][0] + "."
        elif variant == "typo":
            name = list(d["last_name"])
            pos = rng.randrange(1, len(name))
            name[pos] = rng.choice(string.ascii_lowercase)
            d["last_name"] = "".join(name)
        elif variant == "phone_fmt":
            digits = d["phone"].replace("+91 ", "")
            d["phone"] = f"0{digits[:5]}-{digits[5:]}"
        elif variant == "email_case":
            d["email"] = d["email"].upper()
        elif variant == "company_suffix":
            d["company"] = d["company"].rsplit(" ", 1)[0] if " " in d["company"] else d["company"]
        d["created_at"] = d["created_at"] + pd.Timedelta(days=rng.randint(1, 200))
        d["created_at"] = min(d["created_at"], as_of_ts)
        d["last_modified_at"] = max(d["last_modified_at"], d["created_at"])
        dup_rows.append(d)
        log("contacts", d["contact_id"], "duplicate_contact")
    contacts = pd.concat([contacts, pd.DataFrame(dup_rows)], ignore_index=True)

    # ---------- inject contact-level issues ----------
    def pick(frac: float, df: pd.DataFrame) -> list[int]:
        return list(df.sample(frac=frac, random_state=rng.randint(0, 10**6)).index)

    contacts = contacts.astype({"email": object, "phone": object, "industry": object,
                                "lead_source": object, "owner": object, "city": object, "country": object})

    for col, frac, issue in [("email", 0.05, "missing_email"), ("phone", 0.09, "missing_phone"),
                             ("industry", 0.16, "missing_industry"), ("lead_source", 0.12, "missing_lead_source"),
                             ("owner", 0.04, "missing_owner"), ("job_title", 0.10, "missing_job_title")]:
        for idx in pick(frac, contacts):
            contacts.at[idx, col] = rng.choice([np.nan, np.nan, rng.choice(PLACEHOLDERS)])
            log("contacts", contacts.at[idx, "contact_id"], issue)

    def corrupt_email(e: str) -> str:
        return rng.choice([e.replace("@", ""), e.replace("@", "@@"), e.replace(".in", ",in"),
                           e.replace(".", " ", 1), e.split("@")[0] + "@"])

    has_email = contacts[contacts["email"].notna() & ~contacts["email"].isin(PLACEHOLDERS)]
    for idx in pick(0.04, has_email):
        contacts.at[idx, "email"] = corrupt_email(contacts.at[idx, "email"])
        log("contacts", contacts.at[idx, "contact_id"], "invalid_email")

    has_phone = contacts[contacts["phone"].notna() & ~contacts["phone"].isin(PLACEHOLDERS)]
    for idx in pick(0.03, has_phone):
        contacts.at[idx, "phone"] = rng.choice(["98765", "call reception", "+91 98XXXXXX12", "12345678901234567"])
        log("contacts", contacts.at[idx, "contact_id"], "invalid_phone")

    for col, mapping, frac, issue in [("city", CITIES, 0.12, "inconsistent_city"),
                                      ("industry", INDUSTRIES, 0.10, "inconsistent_industry"),
                                      ("country", COUNTRIES, 0.08, "inconsistent_country"),
                                      ("lead_source", LEAD_SOURCES, 0.08, "inconsistent_lead_source")]:
        valid = contacts[contacts[col].isin(list(mapping))]
        for idx in pick(frac, valid):
            contacts.at[idx, col] = rng.choice(mapping[contacts.at[idx, col]])
            log("contacts", contacts.at[idx, "contact_id"], issue)

    for idx in pick(0.005, contacts):
        contacts.at[idx, "created_at"] = as_of_ts + pd.Timedelta(days=rng.randint(30, 400))
        log("contacts", contacts.at[idx, "contact_id"], "future_date")

    # ---------- inject deal-level issues ----------
    deals = deals.astype({"amount": float, "contact_id": object})
    for idx in pick(0.05, deals):
        deals.at[idx, "amount"] = np.nan
        log("deals", deals.at[idx, "deal_id"], "missing_amount")
    has_amt = deals[deals["amount"].notna()]
    for idx in pick(0.01, has_amt):
        deals.at[idx, "amount"] = -abs(deals.at[idx, "amount"])
        log("deals", deals.at[idx, "deal_id"], "negative_amount")
    for idx in pick(0.02, deals):
        deals.at[idx, "close_date"] = deals.at[idx, "created_at"] - pd.Timedelta(days=rng.randint(5, 90))
        log("deals", deals.at[idx, "deal_id"], "close_before_created")
    for idx in pick(0.02, deals):
        deals.at[idx, "contact_id"] = f"C{90000 + rng.randint(0, 9999)}"
        log("deals", deals.at[idx, "deal_id"], "orphan_deal")
    for idx in pick(0.03, deals):
        deals.at[idx, "stage"] = rng.choice(["closed won", "Won", "Closed-Won", "negotiation", "Proposal Sent"])
        log("deals", deals.at[idx, "deal_id"], "inconsistent_stage")

    # natural (not injected) problems, logged after all edits so the truth matches the final data
    for _, r in contacts.iterrows():
        if (as_of_ts - r["last_modified_at"]).days > 365:
            log("contacts", r["contact_id"], "stale_record")
    still_open = ~deals["stage"].str.lower().str.replace("-", " ").isin(["closed won", "closed lost", "won", "lost"])
    for _, d in deals[still_open].iterrows():
        if d["close_date"] < as_of_ts:
            log("deals", d["deal_id"], "overdue_open_deal")

    # shuffle so duplicates aren't conveniently at the bottom
    contacts = contacts.sample(frac=1, random_state=seed).reset_index(drop=True)
    for col in ["created_at", "last_modified_at", "last_activity_at"]:
        contacts[col] = pd.to_datetime(contacts[col]).dt.strftime("%Y-%m-%d")
    for col in ["created_at", "close_date"]:
        deals[col] = pd.to_datetime(deals[col]).dt.strftime("%Y-%m-%d")

    return SyntheticCRM(contacts, deals, pd.DataFrame(issues))
