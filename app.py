"""BazaarIQ: customer intelligence app for the e-commerce hackathon.

Pages
- Dashboard: KPIs and charts, all computed with SQL on the SQLite database
- Churn Prediction: saved Logistic Regression pipeline, with customer lookup and "why" explanation
- At-Risk Customers: scores every customer and ranks them by value at risk
- Sentiment Analysis: saved TF-IDF + Logistic Regression pipeline
"""
import json
import sqlite3
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------- settings
APP_NAME = "BazaarIQ"
APP_TAGLINE = "Customer intelligence for a Pakistani online store"

BASE_DIR = Path(__file__).resolve().parent          # relative paths work locally and on Streamlit Cloud
DB_PATH = BASE_DIR / "ecommerce_clean.db"
MODELS_DIR = BASE_DIR / "models"

INDIGO, EMERALD, MADDER, SAFFRON, SLATE = "#1F2A5A", "#0E7C5A", "#A3242B", "#E0A526", "#6B7393"
SENTIMENT_COLORS = {"Negative": MADDER, "Neutral": SAFFRON, "Positive": EMERALD}

NET_REVENUE = "SUM(CASE WHEN o.returned = 0 THEN o.quantity * o.unit_price * (1 - o.discount) ELSE 0 END)"

# ---------------------------------------------------------------- SQL (every dashboard number comes from here)
SQL = {
    "kpis": f"""
        SELECT ROUND({NET_REVENUE}, 2)                         AS net_revenue,
               COUNT(o.order_id)                               AS total_orders,
               (SELECT COUNT(*) FROM customers)                AS total_customers,
               COUNT(DISTINCT o.customer_id)                   AS buying_customers,
               ROUND(100.0 * AVG(o.returned), 2)               AS return_rate_pct
        FROM orders o;""",
    "monthly": f"""
        SELECT strftime('%Y-%m', o.order_date) AS month,
               ROUND({NET_REVENUE}, 2)        AS net_revenue,
               COUNT(o.order_id)              AS orders
        FROM orders o
        GROUP BY month
        ORDER BY month;""",
    "category": f"""
        SELECT p.category,
               ROUND({NET_REVENUE}, 2)                                   AS net_revenue,
               COUNT(o.order_id)                                         AS orders,
               ROUND(100.0 * SUM(o.returned) / COUNT(o.order_id), 2)     AS return_rate_pct
        FROM orders o
        JOIN products p ON o.product_id = p.product_id
        GROUP BY p.category
        ORDER BY net_revenue DESC;""",
    "city": f"""
        SELECT c.city,
               ROUND({NET_REVENUE}, 2)       AS net_revenue,
               COUNT(DISTINCT o.customer_id) AS customers
        FROM orders o
        JOIN customers c ON o.customer_id = c.customer_id
        GROUP BY c.city
        ORDER BY net_revenue DESC;""",
}

# Same feature definitions as the notebook (Task D), for every customer at a chosen cutoff date
CUSTOMER_FEATURES_SQL = """
    WITH history AS (
        SELECT * FROM orders WHERE order_date <= :cutoff
    )
    SELECT c.customer_id,
           c.customer_name,
           c.city,
           COUNT(*)                                                               AS total_orders,
           SUM(CASE WHEN h.returned = 0
                    THEN h.quantity * h.unit_price * (1 - h.discount) ELSE 0 END) AS total_spending,
           AVG(h.quantity * h.unit_price * (1 - h.discount))                      AS avg_order_value,
           CAST(julianday(:cutoff) - julianday(MAX(h.order_date)) AS INTEGER)     AS days_since_last_order,
           AVG(h.returned)                                                        AS return_rate,
           AVG(h.delivery_days)                                                   AS avg_delivery_days,
           c.age,
           c.membership_type
    FROM history h
    JOIN customers c ON h.customer_id = c.customer_id
    GROUP BY c.customer_id;"""

ORDERS_IN_WINDOW_SQL = """
    SELECT COUNT(*) AS orders
    FROM orders
    WHERE customer_id = :customer_id AND order_date BETWEEN :start AND :end;"""

MODEL_FEATURES = ["total_orders", "total_spending", "avg_order_value", "days_since_last_order",
                  "return_rate", "avg_delivery_days", "age", "membership_type"]

FEATURE_LABELS = {
    "total_orders": "Number of orders",
    "total_spending": "Total spending",
    "avg_order_value": "Average order value",
    "days_since_last_order": "Days since last order",
    "return_rate": "Return rate",
    "avg_delivery_days": "Delivery time",
    "age": "Age",
}

DEFAULT_CUSTOMER = {"total_orders": 3, "total_spending": 22_654.0, "avg_order_value": 7_179.0,
                    "days_since_last_order": 64, "return_rate_pct": 0, "avg_delivery_days": 3.5,
                    "age": 31, "membership_type": "Standard"}

# ---------------------------------------------------------------- page setup and style
st.set_page_config(page_title=APP_NAME, page_icon="🛍️", layout="wide")

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"], .stMarkdown, .stButton button, input, textarea {{
    font-family: 'Plus Jakarta Sans', 'Segoe UI', sans-serif;
}}
.block-container {{ padding-top: 1.6rem; max-width: 1250px; }}

/* Header band with an ajrak-inspired stripe: the one bold element */
.biq-header {{
    background: {INDIGO};
    color: #FFFFFF;
    border-radius: 14px;
    padding: 1.4rem 1.8rem 1.2rem;
    margin-bottom: 1.4rem;
    position: relative;
    overflow: hidden;
}}
.biq-header::after {{
    content: "";
    position: absolute; left: 0; right: 0; bottom: 0; height: 8px;
    background: repeating-linear-gradient(90deg,
        {MADDER} 0 18px, {INDIGO} 18px 22px, {SAFFRON} 22px 30px, {INDIGO} 30px 34px, {EMERALD} 34px 52px, {INDIGO} 52px 56px);
}}
.biq-header h1 {{ font-size: 1.9rem; font-weight: 800; margin: 0; color: #FFFFFF; letter-spacing: -0.02em; }}
.biq-header p {{ margin: 0.25rem 0 0; color: #C9CEE6; font-size: 0.98rem; }}

/* KPI cards: quiet, a coloured left edge carries the meaning */
.kpi {{
    background: #FFFFFF;
    border: 1px solid #E3E6F0;
    border-left: 5px solid var(--edge);
    border-radius: 10px;
    padding: 0.9rem 1.1rem;
}}
.kpi .label {{ color: {SLATE}; font-size: 0.85rem; font-weight: 600; }}
.kpi .value {{ color: {INDIGO}; font-size: 1.75rem; font-weight: 800; line-height: 1.25; }}
.kpi .note {{ color: {SLATE}; font-size: 0.78rem; }}

.section-title {{ color: {INDIGO}; font-weight: 700; font-size: 1.12rem; margin: 0.8rem 0 0.2rem; }}

.result {{
    border-radius: 12px; padding: 1.1rem 1.3rem;
    background: var(--bg); color: var(--fg, #FFFFFF);
}}
.result .big {{ font-size: 1.6rem; font-weight: 800; }}
.result .small {{ font-size: 0.92rem; opacity: 0.92; }}

[data-testid="stSidebar"] {{ background: #FFFFFF; border-right: 1px solid #E3E6F0; }}
.stButton button, .stFormSubmitButton button {{ border-radius: 8px; font-weight: 700; }}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------- helpers
@st.cache_data(show_spinner=False)
def run_query(query, params=None):
    """Run a SQL query on the SQLite database and return a DataFrame."""
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql(query, conn, params=params)


def customer_features(cutoff):
    """One row of model features per customer, using only orders up to the cutoff date."""
    return run_query(CUSTOMER_FEATURES_SQL, {"cutoff": cutoff})


def risk_level(probability):
    if probability >= 0.7:
        return "High risk"
    if probability >= 0.5:
        return "Medium risk"
    return "Low risk"


def explain_prediction(model, customer):
    """Each feature's push on the churn score, compared with an average customer.

    Features are standardised inside the pipeline, so (scaled value x coefficient)
    shows how much that feature moves this customer's score away from the average.
    """
    scaled = model.named_steps["preprocess"].transform(customer[MODEL_FEATURES])[0]
    names = [n.split("__")[1] for n in model.named_steps["preprocess"].get_feature_names_out()]
    pushes = pd.Series(scaled * model.named_steps["model"].coef_[0], index=names)
    pushes = pushes[[n for n in names if n in FEATURE_LABELS]]      # numeric features only
    top = pushes.reindex(pushes.abs().sort_values(ascending=False).index).head(4)
    return top.rename(FEATURE_LABELS)


@st.cache_resource(show_spinner=False)
def load_model(filename):
    return joblib.load(MODELS_DIR / filename)


@st.cache_data(show_spinner=False)
def load_metadata(filename):
    with open(MODELS_DIR / filename) as f:
        return json.load(f)


def header(title, subtitle):
    st.markdown(f'<div class="biq-header"><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)


def kpi(column, label, value, note, edge=INDIGO):
    column.markdown(f'<div class="kpi" style="--edge:{edge}"><div class="label">{label}</div>'
                    f'<div class="value">{value}</div><div class="note">{note}</div></div>',
                    unsafe_allow_html=True)


def pkr(value):
    """Short PKR format: 865.6M, 1.2B."""
    if value >= 1e9:
        return f"PKR {value / 1e9:,.2f}B"
    if value >= 1e6:
        return f"PKR {value / 1e6:,.1f}M"
    return f"PKR {value:,.0f}"


def style_chart(fig, height=360):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10),
                      plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
                      font=dict(family="Plus Jakarta Sans, sans-serif", color=INDIGO),
                      showlegend=False)
    fig.update_xaxes(gridcolor="#EEF0F6")
    fig.update_yaxes(gridcolor="#EEF0F6")
    return fig


def show_sql(*names):
    with st.expander("See the SQL behind this"):
        for name in names:
            st.code(SQL[name].strip(), language="sql")


# ---------------------------------------------------------------- page 1: dashboard
def dashboard_page():
    header(APP_NAME, f"{APP_TAGLINE}. Every number on this page is calculated live with SQL on the SQLite database.")

    kpis = run_query(SQL["kpis"]).iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    kpi(c1, "Net revenue", pkr(kpis["net_revenue"]), "after discounts and returns", EMERALD)
    kpi(c2, "Total orders", f"{kpis['total_orders']:,.0f}", "all valid orders", INDIGO)
    kpi(c3, "Total customers", f"{kpis['total_customers']:,.0f}",
        f"{kpis['buying_customers']:,.0f} have placed an order", SAFFRON)
    kpi(c4, "Return rate", f"{kpis['return_rate_pct']:.1f}%", "orders sent back", MADDER)

    st.markdown('<div class="section-title">Monthly net revenue</div>', unsafe_allow_html=True)
    monthly = run_query(SQL["monthly"])
    monthly["month"] = pd.to_datetime(monthly["month"])
    fig = px.area(monthly, x="month", y="net_revenue",
                  labels={"month": "", "net_revenue": "Net revenue (PKR)"})
    fig.update_traces(line_color=EMERALD, fillcolor="rgba(14,124,90,0.12)",
                      hovertemplate="%{x|%b %Y}<br>PKR %{y:,.0f}<extra></extra>")
    st.plotly_chart(style_chart(fig, 330), width="stretch")

    left, right = st.columns(2)
    with left:
        st.markdown('<div class="section-title">Revenue by category</div>', unsafe_allow_html=True)
        category = run_query(SQL["category"]).sort_values("net_revenue")
        fig = px.bar(category, x="net_revenue", y="category", orientation="h",
                     labels={"net_revenue": "Net revenue (PKR)", "category": ""})
        fig.update_traces(marker_color=[MADDER if c == category["category"].iloc[-1] else INDIGO
                                        for c in category["category"]],
                          hovertemplate="%{y}<br>PKR %{x:,.0f}<extra></extra>")
        st.plotly_chart(style_chart(fig), width="stretch")
    with right:
        st.markdown('<div class="section-title">Revenue by city</div>', unsafe_allow_html=True)
        city = run_query(SQL["city"]).sort_values("net_revenue")
        fig = px.bar(city, x="net_revenue", y="city", orientation="h",
                     labels={"net_revenue": "Net revenue (PKR)", "city": ""})
        fig.update_traces(marker_color=INDIGO, hovertemplate="%{y}<br>PKR %{x:,.0f}<extra></extra>")
        st.plotly_chart(style_chart(fig), width="stretch")

    st.markdown('<div class="section-title">Return rate by category</div>', unsafe_allow_html=True)
    returns = run_query(SQL["category"]).sort_values("return_rate_pct", ascending=False)
    fig = px.bar(returns, x="category", y="return_rate_pct",
                 labels={"category": "", "return_rate_pct": "Return rate (%)"})
    fig.update_traces(marker_color=[MADDER if r > kpis["return_rate_pct"] else SLATE
                                    for r in returns["return_rate_pct"]],
                      hovertemplate="%{x}<br>%{y:.2f}% returned<extra></extra>")
    fig.add_hline(y=kpis["return_rate_pct"], line_dash="dash", line_color=INDIGO,
                  annotation_text=f"Store average {kpis['return_rate_pct']:.1f}%")
    st.plotly_chart(style_chart(fig, 300), width="stretch")
    st.caption("Red bars return more than the store average.")

    show_sql("kpis", "monthly", "category", "city")


# ---------------------------------------------------------------- page 2: churn prediction
def fill_inputs_from_customer(row):
    """Copy a real customer's features into the form fields."""
    st.session_state["in_total_orders"] = int(row["total_orders"])
    st.session_state["in_total_spending"] = round(float(row["total_spending"]), 2)
    st.session_state["in_avg_order_value"] = round(float(row["avg_order_value"]), 2)
    st.session_state["in_days_since_last_order"] = int(row["days_since_last_order"])
    st.session_state["in_return_rate_pct"] = int(round(row["return_rate"] * 100))
    st.session_state["in_avg_delivery_days"] = round(float(row["avg_delivery_days"]), 1) \
        if pd.notna(row["avg_delivery_days"]) else DEFAULT_CUSTOMER["avg_delivery_days"]
    st.session_state["in_age"] = int(row["age"]) if pd.notna(row["age"]) else DEFAULT_CUSTOMER["age"]
    st.session_state["in_membership_type"] = row["membership_type"]


def churn_page():
    header("Churn prediction", "Check how likely a customer is to stop buying, and why.")
    model = load_model("churn_model.joblib")
    meta = load_metadata("churn_metadata.json")
    cutoff = meta["feature_cutoff"]
    window_start, window_end = meta["target_window"]

    for key, value in DEFAULT_CUSTOMER.items():
        st.session_state.setdefault(f"in_{key}", value)

    mode = st.radio("How do you want to enter the customer?",
                    ["Look up a real customer", "Enter details manually"], horizontal=True)

    looked_up = None
    if mode == "Look up a real customer":
        customers = customer_features(cutoff)
        names = customers.set_index("customer_id")[["customer_name", "city"]]
        selected_id = st.selectbox(
            "Customer", customers["customer_id"].tolist(),
            format_func=lambda cid: f"#{cid}  {names.loc[cid, 'customer_name']} ({names.loc[cid, 'city']})",
            help="Type a name or ID to search.")
        looked_up = customers.loc[customers["customer_id"] == selected_id].iloc[0]
        if st.session_state.get("loaded_customer") != selected_id:
            fill_inputs_from_customer(looked_up)
            st.session_state["loaded_customer"] = selected_id
        st.caption(f"Details filled from the database using orders up to {cutoff}. You can still edit them below.")

    with st.form("churn_form"):
        st.markdown('<div class="section-title">Customer history</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.number_input("Total orders", min_value=1, max_value=500, step=1, key="in_total_orders")
        c2.number_input("Total spending (PKR)", min_value=0.0, step=1_000.0, key="in_total_spending")
        c3.number_input("Average order value (PKR)", min_value=0.0, step=500.0, key="in_avg_order_value")
        c4.number_input("Days since last order", min_value=0, max_value=2_000, step=1, key="in_days_since_last_order")

        c5, c6, c7, c8 = st.columns(4)
        c5.slider("Return rate (%)", 0, 100, key="in_return_rate_pct")
        c6.number_input("Average delivery days", min_value=1.0, max_value=10.0, step=0.5, key="in_avg_delivery_days")
        c7.number_input("Age", min_value=18, max_value=80, step=1, key="in_age")
        c8.selectbox("Membership", meta["membership_types"], key="in_membership_type")
        submitted = st.form_submit_button("Predict churn", type="primary", width="stretch")

    if not submitted and looked_up is None:
        st.info("Fill in the customer's history and select **Predict churn**, or look up a real customer.")
        return

    ss = st.session_state
    customer = pd.DataFrame([{
        "total_orders": ss.in_total_orders, "total_spending": ss.in_total_spending,
        "avg_order_value": ss.in_avg_order_value, "days_since_last_order": ss.in_days_since_last_order,
        "return_rate": ss.in_return_rate_pct / 100, "avg_delivery_days": ss.in_avg_delivery_days,
        "age": ss.in_age, "membership_type": ss.in_membership_type,
    }])
    probability = float(model.predict_proba(customer[MODEL_FEATURES])[0, 1])
    risk = risk_level(probability)
    color = {"High risk": MADDER, "Medium risk": SAFFRON, "Low risk": EMERALD}[risk]
    action = {"High risk": "Reach out now with a personal win-back offer or discount.",
              "Medium risk": "Send a reminder with recommended products.",
              "Low risk": "Keep them engaged with regular updates. No discount needed."}[risk]

    left, right = st.columns([1, 1.3])
    with left:
        gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=probability * 100, number={"suffix": "%", "font": {"color": INDIGO}},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": color},
                   "steps": [{"range": [0, 50], "color": "#E6F2EE"}, {"range": [50, 70], "color": "#FBF1DC"},
                             {"range": [70, 100], "color": "#F6E1E2"}],
                   "threshold": {"line": {"color": INDIGO, "width": 3}, "value": 50}}))
        gauge.update_layout(height=260, margin=dict(l=20, r=20, t=30, b=0),
                            font=dict(family="Plus Jakarta Sans, sans-serif"))
        st.plotly_chart(gauge, width="stretch")
    with right:
        verdict = "Likely to churn" if probability >= 0.5 else "Likely to stay"
        text_color = INDIGO if color == SAFFRON else "#FFFFFF"   # dark text on yellow for readability
        st.markdown(f'<div class="result" style="--bg:{color};--fg:{text_color}"><div class="big">{verdict}</div>'
                    f'<div class="small">{risk}. Churn probability: {probability:.1%}</div></div>',
                    unsafe_allow_html=True)
        st.markdown(f"**Suggested action:** {action}")

        if looked_up is not None:
            actual_orders = int(run_query(ORDERS_IN_WINDOW_SQL, {
                "customer_id": int(looked_up["customer_id"]), "start": window_start, "end": window_end
            })["orders"].iloc[0])
            actually_churned = actual_orders == 0
            correct = actually_churned == (probability >= 0.5)
            outcome = "did not buy again (churned)" if actually_churned else f"placed {actual_orders} order(s) (stayed)"
            st.markdown(f"**What really happened ({window_start} to {window_end}):** this customer {outcome}. "
                        f"{'✅ The model was right.' if correct else '❌ The model was wrong this time.'}")

    st.markdown('<div class="section-title">Why this prediction?</div>', unsafe_allow_html=True)
    reasons = explain_prediction(model, customer)
    fig = go.Figure(go.Bar(
        x=reasons.values, y=reasons.index, orientation="h",
        marker_color=[MADDER if v > 0 else EMERALD for v in reasons.values],
        hovertemplate="%{y}<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title_text="← lowers churn risk          raises churn risk →", zeroline=True,
                     zerolinecolor=INDIGO, showticklabels=False)
    st.plotly_chart(style_chart(fig, 230), width="stretch")
    st.caption("Each bar shows how much this customer's value pushes the score compared with an average customer. "
               "Red raises the risk, green lowers it.")
    m = meta["test_metrics"]
    st.caption(f"Model: {meta['model']}, trained on orders up to {cutoff}. "
               f"Test ROC-AUC {m['roc_auc']:.2f}, recall {m['recall']:.2f}, precision {m['precision']:.2f}."
               + (" This customer may have been part of the training data." if looked_up is not None else ""))


# ---------------------------------------------------------------- page 3: at-risk customers
@st.cache_data(show_spinner="Scoring every customer...")
def score_all_customers(cutoff):
    model = load_model("churn_model.joblib")
    scored = customer_features(cutoff).copy()
    scored["churn_probability"] = model.predict_proba(scored[MODEL_FEATURES])[:, 1]
    scored["risk"] = scored["churn_probability"].apply(risk_level)
    scored["value_at_risk"] = scored["churn_probability"] * scored["total_spending"]
    return scored.sort_values("value_at_risk", ascending=False)


def at_risk_page():
    latest = run_query("SELECT MAX(order_date) AS latest FROM orders;")["latest"].iloc[0]
    header("At-risk customers",
           f"Every customer scored with the churn model, using all orders up to {latest}. "
           "Who should the team call first?")
    scored = score_all_customers(latest)

    high = scored[scored["risk"] == "High risk"]
    c1, c2, c3, c4 = st.columns(4)
    kpi(c1, "Customers scored", f"{len(scored):,}", "everyone who has ordered", INDIGO)
    kpi(c2, "High risk", f"{len(high):,}", f"{len(high) / len(scored):.0%} of customers", MADDER)
    kpi(c3, "Value at risk", pkr(high["value_at_risk"].sum()), "from high-risk customers", SAFFRON)
    kpi(c4, "Worth saving first", f"{(high['total_spending'] >= scored['total_spending'].quantile(0.75)).sum():,}",
        "high-risk and in the top 25% by spending", EMERALD)

    st.markdown('<div class="section-title">Filter the list</div>', unsafe_allow_html=True)
    f1, f2, f3, f4 = st.columns([1.2, 1.2, 1.2, 1])
    risks = f1.multiselect("Risk level", ["High risk", "Medium risk", "Low risk"], default=["High risk"])
    cities = f2.multiselect("City", sorted(scored["city"].unique()), placeholder="All cities")
    tiers = f3.multiselect("Membership", sorted(scored["membership_type"].unique()), placeholder="All memberships")
    min_spend = f4.number_input("Minimum spending (PKR)", min_value=0, value=0, step=10_000)

    view = scored[scored["risk"].isin(risks or scored["risk"].unique())
                  & scored["city"].isin(cities or scored["city"].unique())
                  & scored["membership_type"].isin(tiers or scored["membership_type"].unique())
                  & (scored["total_spending"] >= min_spend)]

    st.markdown(f'<div class="section-title">{len(view):,} customers, sorted by value at risk</div>',
                unsafe_allow_html=True)
    columns = ["customer_id", "customer_name", "city", "membership_type", "total_orders", "total_spending",
               "days_since_last_order", "churn_probability", "risk", "value_at_risk"]
    st.dataframe(
        view[columns], hide_index=True, height=420, width="stretch",
        column_config={
            "customer_id": st.column_config.NumberColumn("ID", format="%d"),
            "customer_name": "Name", "city": "City", "membership_type": "Membership",
            "total_orders": st.column_config.NumberColumn("Orders", format="%d"),
            "total_spending": st.column_config.NumberColumn("Spent (PKR)", format="%,.0f"),
            "days_since_last_order": st.column_config.NumberColumn("Days since last order", format="%d"),
            "churn_probability": st.column_config.ProgressColumn("Churn probability", format="percent",
                                                                 min_value=0.0, max_value=1.0),
            "risk": "Risk",
            "value_at_risk": st.column_config.NumberColumn("Value at risk (PKR)", format="%,.0f"),
        })
    st.caption("Value at risk = churn probability × what the customer has spent so far. "
               "It puts big spenders who are about to leave at the top of the list.")
    st.download_button("Download this list (CSV)", view[columns].to_csv(index=False).encode("utf-8"),
                       file_name="at_risk_customers.csv", mime="text/csv", type="primary")

    if len(view):
        st.markdown('<div class="section-title">Where the value at risk is</div>', unsafe_allow_html=True)
        by_city = view.groupby("city", as_index=False)["value_at_risk"].sum().sort_values("value_at_risk")
        fig = px.bar(by_city, x="value_at_risk", y="city", orientation="h",
                     labels={"value_at_risk": "Value at risk (PKR)", "city": ""})
        fig.update_traces(marker_color=MADDER, hovertemplate="%{y}<br>PKR %{x:,.0f}<extra></extra>")
        st.plotly_chart(style_chart(fig, 330), width="stretch")


# ---------------------------------------------------------------- page 3: sentiment analysis
def sentiment_page():
    header("Review sentiment", "Paste a customer review to see if it is negative, neutral or positive.")
    model = load_model("sentiment_model.joblib")

    review = st.text_area("Customer review", height=130,
                          placeholder="e.g. Very satisfied, quality feels premium. product bilkul theek nikla")
    if st.button("Analyze sentiment", type="primary"):
        if not review.strip():
            st.warning("Type or paste a review first.")
            return
        label = model.predict([review])[0]
        probabilities = pd.Series(model.predict_proba([review])[0], index=model.classes_)

        left, right = st.columns([1, 1.3])
        with left:
            text_color = INDIGO if label == "Neutral" else "#FFFFFF"
            st.markdown(f'<div class="result" style="--bg:{SENTIMENT_COLORS[label]};--fg:{text_color}"><div class="big">{label}</div>'
                        f'<div class="small">Confidence: {probabilities[label]:.0%}</div></div>',
                        unsafe_allow_html=True)
        with right:
            chart = probabilities.reindex(["Negative", "Neutral", "Positive"]).reset_index()
            chart.columns = ["sentiment", "probability"]
            fig = px.bar(chart, x="probability", y="sentiment", orientation="h",
                         labels={"probability": "Probability", "sentiment": ""}, range_x=[0, 1])
            fig.update_traces(marker_color=[SENTIMENT_COLORS[s] for s in chart["sentiment"]],
                              hovertemplate="%{y}: %{x:.0%}<extra></extra>")
            st.plotly_chart(style_chart(fig, 200), width="stretch")

    st.caption("Works best on review-style English and Roman Urdu. Very short or unusual phrases "
               "(e.g. \"I love it\") may be labelled Neutral because the training reviews use a small, fixed vocabulary.")


# ---------------------------------------------------------------- navigation
def main():
    pages = [
        st.Page(dashboard_page, title="Dashboard", icon=":material/monitoring:", default=True),
        st.Page(churn_page, title="Churn prediction", icon=":material/person_remove:", url_path="churn"),
        st.Page(at_risk_page, title="At-risk customers", icon=":material/crisis_alert:", url_path="at-risk"),
        st.Page(sentiment_page, title="Sentiment analysis", icon=":material/reviews:", url_path="sentiment"),
    ]
    with st.sidebar:
        st.markdown(f"### 🛍️ {APP_NAME}")
        st.caption(APP_TAGLINE)
    st.navigation(pages).run()
    with st.sidebar:
        st.divider()
        st.caption("Data: ecommerce_clean.db (SQLite). Models: scikit-learn pipelines saved with joblib.")


if __name__ == "__main__":
    main()
