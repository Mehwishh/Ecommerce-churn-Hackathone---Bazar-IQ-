"""Retrain the sentiment model with the INSTALLED scikit-learn (same as notebook Task F).

Run from the project folder:  python retrain_sentiment_model.py
"""
import json
import sqlite3
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline

print("scikit-learn version used for training:", sklearn.__version__)
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)
RANDOM_STATE = 42

with sqlite3.connect(BASE_DIR / "ecommerce_clean.db") as conn:
    reviews = pd.read_sql("""
        SELECT review_id, rating, review_text,
               CASE WHEN rating <= 2 THEN 'Negative'
                    WHEN rating = 3  THEN 'Neutral'
                    ELSE 'Positive' END AS sentiment
        FROM reviews;""", conn)

splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
train_idx, test_idx = next(splitter.split(reviews, groups=reviews["review_text"]))
train, test = reviews.iloc[train_idx], reviews.iloc[test_idx]

model = Pipeline([
    ("tfidf", TfidfVectorizer(lowercase=True, token_pattern=r"\b[a-z]{2,}\b",
                              ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
    ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)),
]).fit(train["review_text"], train["sentiment"])

pred = model.predict(test["review_text"])
scores = {"accuracy": accuracy_score(test["sentiment"], pred),
          "precision_macro": precision_score(test["sentiment"], pred, average="macro"),
          "recall_macro": recall_score(test["sentiment"], pred, average="macro"),
          "f1_macro": f1_score(test["sentiment"], pred, average="macro")}
scores = {k: round(float(v), 3) for k, v in scores.items()}
print("Test metrics:", scores)

path = MODELS_DIR / "sentiment_model.joblib"
joblib.dump(model, path)
loaded = joblib.load(path)
print("Reloaded test:", loaded.predict(["Very satisfied, good value for money", "quality bohat kharab thi"]))
(MODELS_DIR / "sentiment_metadata.json").write_text(json.dumps({
    "model": "TF-IDF + Logistic Regression",
    "labels": {"Negative": "rating 1-2", "Neutral": "rating 3", "Positive": "rating 4-5"},
    "test_metrics": scores, "sklearn_version": sklearn.__version__}, indent=2))
print(f"Saved models/sentiment_model.joblib with scikit-learn {sklearn.__version__}")
