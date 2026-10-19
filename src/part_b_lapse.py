"""Part B (tasks 4-8) -- lapse label, leakage-free features, tuned models, class balance, persistence, retention list.
Run from the repository root (after part_b_segments.py):  python src/part_b_lapse.py"""
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils import resample

CUTOFF, LABEL_END = pd.Timestamp("2025-09-01"), pd.Timestamp("2025-10-01")
OUT = "outputs"
NUM = ["recency", "frequency", "monetary", "avg_discount_pct", "cancel_rate", "avg_delivery_minutes",
       "age", "tenure_days", "avg_order_value", "distinct_categories"]
CAT = ["city", "gender", "fav_category"]


def build_dataset():
    o = pd.read_csv(os.path.join("data", "clean", "orders.csv"), parse_dates=["order_ts"])
    c = pd.read_csv(os.path.join("data", "clean", "customers.csv"), parse_dates=["signup_date"])
    p = pd.read_csv(os.path.join("data", "clean", "products.csv"))
    o = o.merge(p[["product_id", "category"]], on="product_id")

    before = o[o["order_ts"] < CUTOFF]                       # the ONLY rows features may use
    comp = before[before["status"] == "completed"]
    window = o[(o["order_ts"] >= CUTOFF) & (o["order_ts"] < LABEL_END) & (o["status"] == "completed")]

    g = comp.groupby("customer_id")
    f = pd.DataFrame({
        "recency": (CUTOFF - g["order_ts"].max()).dt.days,
        "frequency": g["order_id"].nunique(),
        "monetary": g["order_amount"].sum(),
        "avg_discount_pct": g["discount_pct"].mean(),
        "avg_delivery_minutes": g["delivery_minutes"].mean(),
        "distinct_categories": g["category"].nunique(),
        # most frequent category; ties broken alphabetically so the result is deterministic
        "fav_category": g["category"].agg(lambda s: s.value_counts().sort_index().idxmax()),
    })
    f["avg_order_value"] = f["monetary"] / f["frequency"]
    final = before[before["status"].isin(["completed", "cancelled"])]
    f["cancel_rate"] = (final["status"] == "cancelled").groupby(final["customer_id"]).mean().reindex(f.index)
    f = f.join(c.set_index("customer_id")[["city", "age", "gender", "signup_date"]])
    f["tenure_days"] = (CUTOFF - f["signup_date"]).dt.days
    f["lapsed"] = (~f.index.isin(window["customer_id"])).astype(int)
    return f.drop(columns="signup_date")


def lapsed_metrics(y_true, y_pred):
    return {"precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0)}


def make_prep():
    return ColumnTransformer([("num", StandardScaler(), NUM),
                              ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CAT)])


def show(name, y_true, y_pred):
    m = lapsed_metrics(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    print(f"{name}: confusion [[TN FP] [FN TP]] = {cm.tolist()}  precision {m['precision']:.3f}  "
          f"recall {m['recall']:.3f}  F1 {m['f1']:.3f}")
    return m


def main():
    df = build_dataset()
    print(f"cohort {len(df)}; lapsed {df['lapsed'].sum()} ({df['lapsed'].mean():.2%})")
    X, y = df[NUM + CAT], df["lapsed"]
    assert X.isna().sum().sum() == 0, "a feature column contains missing values"
    print("missing values in feature columns:", int(X.isna().sum().sum()))

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    print(f"train {len(X_tr)} rows / test {len(X_te)} rows; train lapsed {y_tr.sum()}, not lapsed {(y_tr == 0).sum()}")

    base = np.zeros(len(y_te), dtype=int)
    print(f"\nbaseline 'never lapsed': accuracy {accuracy_score(y_te, base):.3f}, lapsed F1 "
          f"{f1_score(y_te, base, zero_division=0):.3f}")

    lr_gs = GridSearchCV(Pipeline([("prep", make_prep()), ("clf", LogisticRegression(max_iter=1000))]),
                         {"clf__C": [0.01, 0.1, 1, 10, 100]}, cv=5, scoring="f1").fit(X_tr, y_tr)
    rf_gs = GridSearchCV(Pipeline([("prep", make_prep()),
                                   ("clf", RandomForestClassifier(n_estimators=100, random_state=42))]),
                         {"clf__max_depth": [3, 5, 8], "clf__min_samples_leaf": [5, 10, 20]},
                         cv=5, scoring="f1").fit(X_tr, y_tr)
    print(f"\nLogReg best {lr_gs.best_params_}  CV F1 {lr_gs.best_score_:.3f}")
    print(f"RandomForest best {rf_gs.best_params_}  CV F1 {rf_gs.best_score_:.3f}  "
          f"({len(rf_gs.cv_results_['params']) * 5} fits)")

    lr, rf = lr_gs.best_estimator_, rf_gs.best_estimator_
    scaler = lr.named_steps["prep"].named_transformers_["num"]
    print(f"\nscaler mean of frequency (train rows only) {scaler.mean_[NUM.index('frequency')]:.4f} vs "
          f"full-cohort mean {X['frequency'].mean():.4f}")

    print("\n=== test set, lapsed class ===")
    m_lr = show("LogReg      ", y_te, lr.predict(X_te))
    m_rf = show("RandomForest", y_te, rf.predict(X_te))
    best, best_name = (rf, "RandomForest") if m_rf["f1"] >= m_lr["f1"] else (lr, "LogReg")
    print(f"selected model: {best_name}")

    # class balance: over-sample lapsed TRAIN rows only, test set untouched
    tr = X_tr.assign(lapsed=y_tr)
    maj, mino = tr[tr["lapsed"] == 0], tr[tr["lapsed"] == 1]
    extra = resample(mino, replace=True, n_samples=len(maj) - len(mino), random_state=42)
    bal = pd.concat([tr, extra])
    print(f"\nbalanced training set: {bal['lapsed'].value_counts().to_dict()}")
    lr_bal = Pipeline([("prep", make_prep()),
                       ("clf", LogisticRegression(max_iter=1000, C=lr_gs.best_params_["clf__C"]))]) \
        .fit(bal[NUM + CAT], bal["lapsed"])
    show("LogReg unbalanced", y_te, lr.predict(X_te))
    show("LogReg balanced  ", y_te, lr_bal.predict(X_te))

    names = rf.named_steps["prep"].get_feature_names_out()
    imp = pd.Series(rf.named_steps["clf"].feature_importances_, index=names).sort_values(ascending=False)
    print("\nRandomForest top-5 importances:")
    print(imp.head(5).round(4).to_string())
    coef = pd.Series(lr.named_steps["clf"].coef_[0], index=lr.named_steps["prep"].get_feature_names_out())
    print("\nLogReg coefficients (standardised features):")
    print(coef.sort_values().round(3).to_string())

    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "lapse_model.joblib")
    joblib.dump(best, path)
    reloaded = joblib.load(path)
    same = np.array_equal(reloaded.predict(X_te), best.predict(X_te))
    print(f"\nreloaded {path}: identical test predictions = {same}")
    assert same

    seg = pd.read_csv(os.path.join(OUT, "customer_segments.csv")).set_index("customer_id")
    te = X_te.assign(lapse_probability=reloaded.predict_proba(X_te)[:, 1], actually_lapsed=y_te)
    te = te[te.index.isin(seg.index[seg["is_premium"]])]
    ret = te.sort_values("lapse_probability", ascending=False).head(10)
    ret = ret.reset_index()[["customer_id", "lapse_probability", "actually_lapsed"]].round({"lapse_probability": 4})
    ret.to_csv(os.path.join(OUT, "retention_list.csv"), index=False)
    print(f"\npremium customers in test set: {len(te)}; retention list (top {len(ret)}):")
    print(ret.to_string(index=False))
    print(f"truly lapsed in retention list: {int(ret['actually_lapsed'].sum())} of {len(ret)}")


if __name__ == "__main__":
    main()
