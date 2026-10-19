"""Part A -- audit the raw exports, clean them with explicit rules, and store them as CSV + SQLite.
Run from the repository root:  python src/part_a_clean.py"""
import os
import sqlite3

import pandas as pd

RAW, CLEAN, DB = os.path.join("data", "raw"), os.path.join("data", "clean"), os.path.join("data", "instamart.db")


def audit_table(name, df):
    print(f"\n=== {name}: shape {df.shape} ===")
    rep = pd.DataFrame({"dtype": df.dtypes.astype(str), "missing": df.isna().sum(),
                        "missing_pct": (df.isna().mean() * 100).round(2)})
    print(rep.to_string())


def audit(customers, products, orders):
    for n, d in [("customers.csv", customers), ("products.csv", products), ("orders.csv", orders)]:
        audit_table(n, d)
    amt_num = pd.to_numeric(orders["order_amount"], errors="coerce")
    figures = {
        "raw orders rows": len(orders),
        "exact duplicate rows": int(orders.duplicated().sum()),
        "rows with status pending": int((orders["status"] == "pending").sum()),
        "distinct order_id": orders["order_id"].nunique(),
        "non-numeric order_amount": int((amt_num.isna() & orders["order_amount"].notna()).sum()),
        "negative delivery_minutes": int((orders["delivery_minutes"] < 0).sum()),
        "blank delivery_minutes": int(orders["delivery_minutes"].isna().sum()),
        "blank age": int(customers["age"].isna().sum()),
        "age outside 10-100": int((customers["age"].notna() & ~customers["age"].between(10, 100)).sum()),
        "distinct raw city spellings": customers["city"].nunique(),
        "distinct non-blank raw gender spellings": customers["gender"].dropna().nunique(),
        "blank gender": int(customers["gender"].isna().sum()),
    }
    print("\n=== Audit figures ===")
    for k, v in figures.items():
        print(f"{k:42s} {v:>8,}")
    return figures


def clean_customers(c):
    c = c.copy()
    c["city"] = c["city"].str.strip().str.title()
    gmap = {"male": "male", "m": "male", "female": "female", "f": "female"}
    c["gender"] = c["gender"].fillna("").str.strip().str.lower().map(gmap).fillna("unknown")
    impossible = c["age"].notna() & ~c["age"].between(10, 100)
    c.loc[impossible, "age"] = pd.NA
    n_missing = int(c["age"].isna().sum())
    c["age"] = c["age"].fillna(c.groupby("city")["age"].transform("median"))
    c["signup_date"] = pd.to_datetime(c["signup_date"])
    print(f"\n[customers] impossible ages set to missing: {int(impossible.sum())}; ages filled with city median: {n_missing}")
    print("[customers] cities:", c["city"].value_counts().to_dict())
    print("[customers] genders:", c["gender"].value_counts().to_dict())
    return c


def clean_orders(o, products):
    o = o.copy()
    log = {}
    o["order_ts"] = pd.to_datetime(o["order_ts"])
    n0 = len(o)
    o = o.drop_duplicates()
    log["exact duplicates removed"] = n0 - len(o)
    n1 = len(o)
    o = o.sort_values(["order_id", "order_ts"]).drop_duplicates("order_id", keep="last")
    log["superseded (earlier) records removed"] = n1 - len(o)

    o["order_amount"] = pd.to_numeric(o["order_amount"], errors="coerce")
    o = o.merge(products[["product_id", "category", "price_inr"]], on="product_id", how="left")
    miss_amt = o["order_amount"].isna()
    o.loc[miss_amt, "order_amount"] = (o.loc[miss_amt, "price_inr"] * o.loc[miss_amt, "quantity"]
                                       * (1 - o.loc[miss_amt, "discount_pct"] / 100)).round(2)
    log["order_amount filled from price"] = int(miss_amt.sum())

    neg = o["delivery_minutes"] < 0
    o.loc[neg, "delivery_minutes"] = pd.NA
    log["negative delivery_minutes set to missing"] = int(neg.sum())
    comp = o["status"] == "completed"
    cat_median = o[comp].groupby("category")["delivery_minutes"].median()
    print("[orders] median delivery minutes of completed orders by category:", cat_median.to_dict())
    fill_mask = comp & o["delivery_minutes"].isna()
    o.loc[comp, "delivery_minutes"] = o[comp].groupby("category")["delivery_minutes"].transform(
        lambda s: s.fillna(s.median()))
    log["completed delivery_minutes filled with category median"] = int(fill_mask.sum())

    o = o.drop(columns=["category", "price_inr"]).sort_values("order_id").reset_index(drop=True)
    print("\n=== Order cleaning log ===")
    for k, v in log.items():
        print(f"{k:55s} {v:>6,}")
    print("[orders] rows:", len(o), "| status:", o["status"].value_counts().to_dict())
    print("[orders] completed revenue INR %.2f" % o.loc[o["status"] == "completed", "order_amount"].sum())
    return o


def main():
    # only a truly blank cell counts as missing; text such as "N/A" stays text so the audit can count it
    raw = dict(keep_default_na=False, na_values=[""])
    customers = pd.read_csv(os.path.join(RAW, "customers.csv"), **raw)
    products = pd.read_csv(os.path.join(RAW, "products.csv"), **raw)
    orders = pd.read_csv(os.path.join(RAW, "orders.csv"), **raw)
    audit(customers, products, orders)
    tickets = pd.read_csv(os.path.join(RAW, "tickets.csv"), keep_default_na=False)
    print(f"\n[tickets] shape {tickets.shape}; empty texts: {(tickets['text'].str.strip() == '').sum()}")

    c = clean_customers(customers)
    o = clean_orders(orders, products)

    os.makedirs(CLEAN, exist_ok=True)
    c_out = c.assign(signup_date=c["signup_date"].dt.strftime("%Y-%m-%d"))
    o_out = o.assign(order_ts=o["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S"))
    c_out.to_csv(os.path.join(CLEAN, "customers.csv"), index=False)
    products.to_csv(os.path.join(CLEAN, "products.csv"), index=False)
    o_out.to_csv(os.path.join(CLEAN, "orders.csv"), index=False)
    if os.path.exists(DB):
        os.remove(DB)
    with sqlite3.connect(DB) as con:
        c_out.to_sql("customers", con, index=False)
        products.to_sql("products", con, index=False)
        o_out.to_sql("orders", con, index=False)
    print(f"\nwrote {CLEAN}/*.csv and {DB}")


if __name__ == "__main__":
    main()
