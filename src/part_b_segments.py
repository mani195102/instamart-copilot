"""Part B (tasks 1-3) -- RFM table, K-means model selection, segment profiles, PCA, customer_segments.csv.
Run from the repository root:  python src/part_b_segments.py"""
import os

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

SNAPSHOT = pd.Timestamp("2025-10-01 00:00:00")
FINAL_K = 4
OUT = "outputs"


def build_rfm(orders):
    comp = orders[orders["status"] == "completed"]
    rfm = comp.groupby("customer_id").agg(last_ts=("order_ts", "max"), frequency=("order_id", "nunique"),
                                          monetary=("order_amount", "sum"))
    rfm["recency"] = (SNAPSHOT - rfm["last_ts"]).dt.days
    rfm["monetary"] = rfm["monetary"].round(2)
    return rfm[["recency", "frequency", "monetary"]].reset_index()


def name_segments(profile):
    """Name clusters from their profile, never from the arbitrary cluster number."""
    names, actions = {}, {}
    premium = profile["monetary"].idxmax()
    names[premium] = "Premium Big Spenders"
    actions[premium] = "Priority 1-hour support, early access to electronics launches, no-cost EMI offers."
    rest = profile.drop(index=premium)
    lapsing = rest["recency"].idxmax()
    names[lapsing] = "Dormant / At-Risk"
    actions[lapsing] = "Win-back push with a time-limited coupon on staple groceries; stop after two ignored offers."
    rest = rest.drop(index=lapsing)
    loyal = rest["frequency"].idxmax()
    names[loyal] = "Loyal Regulars"
    actions[loyal] = "Subscription / membership offer for free delivery; reward streaks instead of discounting."
    for k in rest.drop(index=loyal).index:
        names[k] = "Occasional Shoppers"
        actions[k] = "Cross-sell dairy and snacks bundles to lift order frequency; nudge with basket-size thresholds."
    return names, actions, premium


def main():
    orders = pd.read_csv(os.path.join("data", "clean", "orders.csv"), parse_dates=["order_ts"])
    rfm = build_rfm(orders)
    print(f"RFM customers: {len(rfm)}")
    print(rfm[["recency", "frequency", "monetary"]].describe().round(2).to_string())
    print(f"recency median {rfm['recency'].median():.0f}")

    X = StandardScaler().fit_transform(rfm[["recency", "frequency", "monetary"]])
    rows = []
    for k in range(2, 9):
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
        rows.append({"K": k, "inertia": round(km.inertia_, 1),
                     "silhouette": round(silhouette_score(X, km.labels_), 3)})
    scan = pd.DataFrame(rows)
    scan["inertia_drop_pct"] = (-scan["inertia"].pct_change() * 100).round(1)
    print("\n=== K-means scan ===")
    print(scan.to_string(index=False))

    km = KMeans(n_clusters=FINAL_K, n_init=10, random_state=42).fit(X)
    rfm["segment_id"] = km.labels_
    profile = rfm.groupby("segment_id").agg(size=("customer_id", "count"), recency=("recency", "mean"),
                                            frequency=("frequency", "mean"), monetary=("monetary", "mean"))
    names, actions, premium = name_segments(profile)
    profile["segment_name"] = profile.index.map(names)
    profile["action"] = profile.index.map(actions)
    print(f"\n=== Segment profile, K={FINAL_K} (cluster ids are labels only, no order) ===")
    print(profile.round(2).to_string())

    pca = PCA(n_components=2).fit(X)
    ev = pca.explained_variance_ratio_
    print(f"\nPCA explained_variance_ratio_: {ev[0]:.3f}, {ev[1]:.3f} (total {ev.sum():.3f})")
    print("PCA loadings (rows=component, cols=recency, frequency, monetary):")
    print(pd.DataFrame(pca.components_, columns=["recency", "frequency", "monetary"]).round(3).to_string())

    rfm["segment_name"] = rfm["segment_id"].map(names)
    rfm["is_premium"] = rfm["segment_id"] == premium
    os.makedirs(OUT, exist_ok=True)
    rfm[["customer_id", "recency", "frequency", "monetary", "segment_id", "segment_name", "is_premium"]] \
        .to_csv(os.path.join(OUT, "customer_segments.csv"), index=False)
    print(f"\nwrote {OUT}/customer_segments.csv ({int(rfm['is_premium'].sum())} premium customers)")


if __name__ == "__main__":
    main()
