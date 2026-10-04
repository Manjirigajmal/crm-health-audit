"""Create the messy sample CRM export used for demos and tests.

    python generate_sample_data.py            # writes sample_data/*.csv
"""
from pathlib import Path

from crm_audit.synthetic import generate

OUT = Path(__file__).parent / "sample_data"

if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    crm = generate()
    crm.contacts.to_csv(OUT / "contacts.csv", index=False)
    crm.deals.to_csv(OUT / "deals.csv", index=False)
    crm.issues.to_csv(OUT / "injected_issues.csv", index=False)
    print(f"contacts: {len(crm.contacts):,} rows | deals: {len(crm.deals):,} rows")
    print("injected issues:")
    print(crm.issues["issue"].value_counts().to_string())
