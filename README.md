# 🛍️ BazaarIQ: E-Commerce Customer Intelligence System

**Live app:** (https://bazariq.streamlit.app/)
**Hackathon:** Data Science Final Hackathon (6 hours)

BazaarIQ starts from a raw SQLite database and ends with a deployed Streamlit app that tells the business **how it is doing, which customers are about to leave, why, and what customers feel about their orders**.

---

## What the project does

| Step | What we did | Result |
|---|---|---|
| **A. Cleaning** | Inspected 4 tables (100,000 rows). Fixed messy text, negative prices and missing labels; removed only rows that could not be trusted | Removed only **0.12% of orders** and **0.44% of reviews**; all fixed issues re-checked to 0 |
| **B. SQL analysis** | 5 business queries (revenue, top customers, categories, monthly trend, top products) | [`sql/business_queries.sql`](sql/business_queries.sql) |
| **C. EDA** | 4 charts + business insights | See insights below |
| **D. Churn model** | Logistic Regression vs Random Forest, features from orders up to 31 May 2026, target = no purchase Jun–Aug 2026 | **Logistic Regression chosen**: ROC-AUC 0.785, recall 0.84, no overfitting |
| **E. Deep learning** | Feed-forward network 32 → 16 → 1 on the same test set | ROC-AUC 0.787 with 77× more parameters, so **not worth the complexity** |
| **F. Sentiment NLP** | TF-IDF + Logistic Regression, split by unique text to avoid leakage | 100% on test, but limited to the dataset's template phrases (see limitation) |
| **G. Streamlit app** | Dashboard, churn prediction, at-risk customer list, sentiment analysis | Live link above |

**Net revenue** = `quantity × unit_price × (1 − discount)` for non-returned orders.

---

## Three business insights

**1. The business depends heavily on Electronics.**
Electronics brings **69.9% of net revenue**, and its returns cost **PKR 51.8M**, which is **73% of all returned value**. A drop in Electronics would hit the whole company. Reducing Electronics returns is the biggest money lever, and growing other categories would lower the risk.

**2. Fashion has the highest return rate.**
Fashion returns are **11.25%**, which is **1.7× the store average (6.69%)**. This usually points to size or fit problems. Size guides, fit notes and clearer photos could reduce returns without reducing sales.

**3. Revenue is concentrated in two cities, but smaller cities have more valuable customers.**
Karachi and Lahore bring **44.5% of revenue**, but a **Sargodha** customer spends **PKR 171,842** on average, which is **41% above the average customer**. Marketing to win new customers in high-value smaller cities may give a better return.

*Bonus:* revenue grew **+172% year over year** (Aug 2026 vs Aug 2025), but fell **11.6%** from July to August 2026. This is worth watching, and it is why churn prediction matters.

---

## The app

| Page | What it does |
|---|---|
| **Dashboard** | Net revenue, orders, customers, return rate, monthly / category / city / return charts. **Every number comes from SQL** on the SQLite database |
| **Churn prediction** | Enter a customer's history, or **look up a real customer**. Shows churn probability, risk level, suggested action, **why** the model decided, and what really happened |
| **At-risk customers** | Scores every customer, ranks them by **value at risk** (probability × spending), with filters and a CSV download |
| **Sentiment analysis** | Paste a review (English or Roman Urdu) and get Negative / Neutral / Positive with confidence |

---

## Repository structure

```
├── app.py                       # Streamlit app
├── retrain_churn_model.py       # retrains the churn model with the installed scikit-learn
├── retrain_sentiment_model.py   # retrains the sentiment model with the installed scikit-learn
├── requirements.txt
├── ecommerce_hackathon.db       # original raw database (never modified)
├── ecommerce_clean.db           # cleaned database used by the SQL and the app
├── .streamlit/config.toml       # app theme
├── models/
│   ├── churn_model.joblib       # Logistic Regression pipeline (also saved as churn_model.pkl)
│   ├── churn_metadata.json
│   ├── sentiment_model.joblib   # TF-IDF + Logistic Regression pipeline
│   └── sentiment_metadata.json
├── sql/
│   └── business_queries.sql     # the 5 required business queries
├── E-Commerce_Customer_Intelligence_System.ipynb     # cleaning, SQL, EDA, ML, DL, NLP
└── er_diagram.png               # database schema
```

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

**Retrain the models** (needed if your scikit-learn version differs from the one in `models/*_metadata.json`):

```bash
python retrain_churn_model.py
python retrain_sentiment_model.py
```

Then put the printed scikit-learn version in `requirements.txt`. The full analysis (cleaning, EDA, deep learning) is in `E-Commerce_Customer_Intelligence_System.ipynb` (it needs `tensorflow` for Task E only).

---

## Key decisions

- **No data leakage in churn:** features only use orders up to 31 May 2026; imputation and scaling are inside a scikit-learn `Pipeline` fitted on training data only; the test set is used once.
- **Balanced target (49% churn):** no SMOTE or class weights needed for churn.
- **Model choice:** Logistic Regression matched Random Forest on test ROC-AUC, caught more churners, did not overfit (Random Forest: train 0.96 vs test 0.78) and is easy to explain.
- **Sentiment leakage check:** 83% of reviews are exact repeats, so we split by unique text to make sure no test review was seen during training.

## Limitations

- The review dataset is template-generated, so the sentiment model does not generalize well to real customer wording (e.g. *"I love it"* is labelled Neutral).
- The churn model uses 8 simple features; adding product categories, discounts and review sentiment per customer could improve it.
