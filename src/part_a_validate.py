"""Part A -- assert the reference checkpoints on the clean tables and the database.
Run from the repository root:  python src/part_a_validate.py"""
import os
import sqlite3

import pandas as pd

CLEAN, DB = os.path.join("data", "clean"), os.path.join("data", "instamart.db")

c = pd.read_csv(os.path.join(CLEAN, "customers.csv"))
o = pd.read_csv(os.path.join(CLEAN, "orders.csv"))
p = pd.read_csv(os.path.join(CLEAN, "products.csv"))
comp, canc = o[o["status"] == "completed"], o[o["status"] == "cancelled"]

# customers
assert c["city"].value_counts().to_dict() == {"Pune": 227, "Delhi": 210, "Bengaluru": 197,
                                               "Chennai": 191, "Hyderabad": 191, "Mumbai": 184}
assert c["gender"].value_counts().to_dict() == {"female": 610, "male": 560, "unknown": 30}
assert c["age"].isna().sum() == 0
assert c["age"].between(18, 60).all()
# orders
assert len(o) == 12309 and o["order_id"].is_unique
assert len(comp) == 11110 and len(canc) == 1199
assert (o["status"] == "pending").sum() == 0
assert o["order_amount"].isna().sum() == 0
assert comp["delivery_minutes"].isna().sum() == 0
assert canc["delivery_minutes"].isna().sum() == 1199
assert (o["delivery_minutes"].dropna() >= 0).all()
assert round(comp["order_amount"].sum(), 2) == 2998668.80
assert len(p) == 40
# database
with sqlite3.connect(DB) as con:
    tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"customers", "products", "orders"} <= tables
    n, rev = con.execute("SELECT COUNT(*), ROUND(SUM(CASE WHEN status='completed' THEN order_amount END), 2) "
                         "FROM orders").fetchone()
    assert n == 12309 and rev == 2998668.80
    ts = con.execute("SELECT order_ts FROM orders LIMIT 1").fetchone()[0]
    assert isinstance(ts, str) and len(ts) == 19 and ts[10] == " "

print("ALL CHECKPOINTS PASSED")
