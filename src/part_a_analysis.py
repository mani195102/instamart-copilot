"""Part A -- run the seven SQL queries, cross-check Q2 in pandas, and run the exploratory statistics.
Run from the repository root:  python src/part_a_analysis.py"""
import os
import re
import sqlite3

import pandas as pd

DB, SQL = os.path.join("data", "instamart.db"), os.path.join("sql", "queries.sql")
pd.set_option("display.width", 140)


def load_queries(path):
    text = open(path, encoding="utf-8").read()
    parts = re.split(r"^-- name: (\w+)\s*$", text, flags=re.M)
    return {parts[i]: parts[i + 1].strip() for i in range(1, len(parts), 2)}


def run_sql():
    q = load_queries(SQL)
    with sqlite3.connect(DB) as con:
        r = {k: pd.read_sql_query(v, con) for k, v in q.items()}
    print("Q1 customers with > INR 8000 in completed orders above INR 500:", len(r["Q1"]))
    print(r["Q1"].head(5).to_string(index=False))
    q2 = r["Q2"]
    print(f"\nQ2 rows: {len(q2)}; total INR {q2['revenue_inr'].sum():,.2f}; largest cell:",
          q2.iloc[0].to_dict())
    print("\nQ3 products above average price:", len(r["Q3"]))
    print(r["Q3"].to_string(index=False))
    print("\nQ4 customers with no completed order:", len(r["Q4"]))
    q5 = r["Q5"]
    print(f"\nQ5 customers above average: {len(q5)}; average INR {q5['avg_spend_inr'].iloc[0]:,.2f}")
    q6 = r["Q6"]
    print("\nQ6 top customer per city:")
    print(q6[q6["row_num"] == 1][["city", "customer_id", "total_spend"]].to_string(index=False))
    diff = q6[q6["rnk"] != q6["row_num"]]
    print(f"Q6 rows where RANK() != ROW_NUMBER(): {len(diff)}")
    print(diff.to_string(index=False))
    q7 = r["Q7"]
    print(f"\nQ7 daily rows: {len(q7)}; last cumulative INR {q7['cumulative_revenue_inr'].iloc[-1]:,.2f}")
    return r


def pandas_checks(sql_total):
    o = pd.read_csv(os.path.join("data", "clean", "orders.csv"))
    c = pd.read_csv(os.path.join("data", "clean", "customers.csv"))
    p = pd.read_csv(os.path.join("data", "clean", "products.csv"))
    comp = o[o["status"] == "completed"].merge(c[["customer_id", "city"]], on="customer_id") \
                                        .merge(p[["product_id", "category"]], on="product_id")
    pv = pd.pivot_table(comp, values="order_amount", index="city", columns="category",
                        aggfunc="sum", margins=True, margins_name="Total")
    print("\n=== pandas pivot of Q2 (INR) ===")
    print(pv.round(2).to_string())
    grand = round(pv.loc["Total", "Total"], 2)
    print(f"pivot grand total INR {grand:,.2f} vs SQL total INR {sql_total:,.2f} -> equal: {abs(grand - sql_total) < 0.005}")

    by_city = comp.groupby("city").agg(total_inr=("order_amount", "sum"), mean_inr=("order_amount", "mean"),
                                       orders=("order_amount", "count")).round(2)
    print("\n=== completed order_amount by city ===")
    print(by_city.sort_values("total_inr", ascending=False).to_string())

    a = comp["order_amount"]
    q1, q3 = a.quantile(.25), a.quantile(.75)
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    print("\n=== completed order_amount distribution (INR) ===")
    print(f"mean {a.mean():.2f}  median {a.median():.2f}  skew {a.skew():.3f}  std {a.std():.2f}")
    print(f"min {a.min():.2f}  Q1 {q1:.2f}  Q2 {a.median():.2f}  Q3 {q3:.2f}  max {a.max():.2f}  IQR {iqr:.2f}")
    print(f"lower fence {lo:.2f} -> {(a < lo).sum()} outliers below; upper fence {hi:.2f} -> {(a > hi).sum()} outliers above")
    corr = comp[["order_amount", "discount_pct", "quantity", "delivery_minutes"]].corr()
    print("\n=== correlation matrix (completed orders) ===")
    print(corr.round(3).to_string())
    pairs = corr.where(~pd.DataFrame([[i >= j for j in range(4)] for i in range(4)],
                                     index=corr.index, columns=corr.columns)).stack()
    top = pairs.abs().idxmax()
    print(f"largest |correlation|: {top} = {pairs[top]:.3f}")


if __name__ == "__main__":
    res = run_sql()
    pandas_checks(round(res["Q2"]["revenue_inr"].sum(), 2))
