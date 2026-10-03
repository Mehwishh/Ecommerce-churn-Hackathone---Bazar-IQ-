"""Retrain the churn model with the INSTALLED scikit-learn.

Exact copy of notebook Task D (D.1 SQL, D.3 split, D.4 pipeline, D.9 save).
Nothing is changed: same SQL, cutoff, target window, MODEL_FEATURES, preprocessing,
classifier and random seed.

Run from the project folder (where ecommerce_clean.db and app.py are):
    python retrain_churn_model.py
"""
import json
import sqlite3
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

print("scikit-learn version used for training:", sklearn.__version__)

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "ecommerce_clean.db"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

# ---------------- settings (identical to the notebook and app.py) ----------------
RANDOM_STATE = 42
FEATURE_CUTOFF = "2026-05-31"
TARGET_START, TARGET_END = "2026-06-01", "2026-08-31"

NUMERIC_FEATURES = ["total_orders", "total_spending", "avg_order_value", "days_since_last_order",
                    "return_rate", "avg_delivery_days", "age"]
CATEGORICAL_FEATURES = ["membership_type"]
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES    # same order as MODEL_FEATURES in app.py
TARGET = "churn"

# ---------------- D.1 churn dataset (same SQL as the notebook) ----------------
CHURN_DATASET_SQL = """
WITH history AS (
    SELECT * FROM orders WHERE order_date <= :cutoff
),
features AS (
    SELECT
        customer_id,
        COUNT(*)                                                             AS total_orders,
        SUM(CASE WHEN returned = 0
                 THEN quantity * unit_price * (1 - discount) ELSE 0 END)     AS total_spending,
        AVG(quantity * unit_price * (1 - discount))                          AS avg_order_value,
        CAST(julianday(:cutoff) - julianday(MAX(order_date)) AS INTEGER)     AS days_since_last_order,
        AVG(returned)                                                        AS return_rate,
        AVG(delivery_days)                                                   AS avg_delivery_days
    FROM history
    GROUP BY customer_id
),
active_in_target AS (
    SELECT DISTINCT customer_id
    FROM orders
    WHERE order_date BETWEEN :target_start AND :target_end
)
SELECT
    f.*,
    c.age,
    c.membership_type,
    CASE WHEN a.customer_id IS NULL THEN 1 ELSE 0 END AS churn
FROM features f
JOIN customers c        ON f.customer_id = c.customer_id
LEFT JOIN active_in_target a ON f.customer_id = a.customer_id;
"""

with sqlite3.connect(DB_PATH) as conn:
    churn_df = pd.read_sql(CHURN_DATASET_SQL, conn, params={
        "cutoff": FEATURE_CUTOFF, "target_start": TARGET_START, "target_end": TARGET_END})
print(f"Churn dataset: {len(churn_df):,} customers | churn rate {churn_df[TARGET].mean():.2%}")

# ---------------- D.3 split ----------------
X = churn_df[MODEL_FEATURES]
y = churn_df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

# ---------------- D.4 pipeline: impute + scale + one-hot + Logistic Regression ----------------
preprocess = ColumnTransformer([
    ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                      ("scale", StandardScaler())]), NUMERIC_FEATURES),
    ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
])
model = Pipeline([
    ("preprocess", preprocess),
    ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
])
model.fit(X_train, y_train)

# ---------------- evaluate (test set) ----------------
proba = model.predict_proba(X_test)[:, 1]
pred = (proba >= 0.5).astype(int)
test_metrics = {"accuracy": accuracy_score(y_test, pred), "precision": precision_score(y_test, pred),
                "recall": recall_score(y_test, pred), "f1": f1_score(y_test, pred),
                "roc_auc": roc_auc_score(y_test, proba)}
test_metrics = {k: round(float(v), 3) for k, v in test_metrics.items()}
print("Test metrics:", test_metrics)

# ---------------- save + reload test ----------------
pkl_path = MODELS_DIR / "churn_model.pkl"
joblib.dump(model, pkl_path)

loaded_model = joblib.load(pkl_path)
predictions = loaded_model.predict_proba(X_test)[:, 1]
print("First 10 reloaded predictions:", np.round(predictions[:10], 4))
assert np.allclose(predictions, proba), "Reloaded model gives different probabilities"
assert list(loaded_model.feature_names_in_) == MODEL_FEATURES, "Feature names/order differ from MODEL_FEATURES"

# Same call the Streamlit app makes on the At-risk page
scored = X_test.copy()
scored["churn_probability"] = loaded_model.predict_proba(scored[MODEL_FEATURES])[:, 1]
print("App-style scoring works on", len(scored), "customers")

# app.py loads models/churn_model.joblib, so save the same pipeline there too (no app change needed)
joblib.dump(loaded_model, MODELS_DIR / "churn_model.joblib")

# Keep every key app.py reads from the metadata; add the version used
metadata = {
    "model": "Logistic Regression",
    "numeric_features": NUMERIC_FEATURES,
    "categorical_features": CATEGORICAL_FEATURES,
    "membership_types": sorted(churn_df["membership_type"].unique().tolist()),
    "feature_cutoff": FEATURE_CUTOFF,
    "target_window": [TARGET_START, TARGET_END],
    "test_metrics": test_metrics,
    "sklearn_version": sklearn.__version__,
}
(MODELS_DIR / "churn_metadata.json").write_text(json.dumps(metadata, indent=2))

print(f"Saved models/churn_model.pkl and models/churn_model.joblib with scikit-learn {sklearn.__version__}")
print(f"Put scikit-learn=={sklearn.__version__} in requirements.txt")
