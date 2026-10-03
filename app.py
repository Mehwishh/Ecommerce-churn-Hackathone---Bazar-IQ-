"""BazaarIQ: seller analytics and customer retention for an online store.

Tabs
- Overview          : store KPIs and charts, every number computed with SQL on SQLite
- Churn Predictor   : saved Logistic Regression pipeline, manual customer input
- Review Sentiment  : saved TF-IDF + Logistic Regression pipeline
- Retention Center  : customer 360, ranked at-risk list, win-back planner, win-back email/SMS campaign
"""
import json
import smtplib
import sqlite3
import ssl
from datetime import datetime
from email.message import EmailMessage
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_option_menu import option_menu

# ------------------------------------------------------------------ settings
APP_NAME = "BazaarIQ"
BASE_DIR = Path(__file__).resolve().parent          # relative paths: work locally and on Streamlit Cloud
DB_PATH = BASE_DIR / "ecommerce_clean.db"
MODELS_DIR = BASE_DIR / "models"

INK, INK_SOFT = "#14213D", "#5A6478"
EMERALD, MADDER, SAFFRON, SKY = "#0E7C5A", "#B42318", "#DC8A00", "#2F6FED"
CANVAS, LINE = "#F5F6F8", "#E4E7EE"
RISK_COLORS = {"High risk": MADDER, "Medium risk": SAFFRON, "Low risk": EMERALD}
SENTIMENT_COLORS = {"Negative": MADDER, "Neutral": SAFFRON, "Positive": EMERALD}

NET = "CASE WHEN o.returned = 0 THEN o.quantity * o.unit_price * (1 - o.discount) ELSE 0 END"

# ------------------------------------------------------------------ SQL (all dashboard numbers come from here)
SQL = {
    "kpis": f"""
        SELECT ROUND(SUM({NET}), 2)                    AS net_revenue,
               COUNT(o.order_id)                       AS total_orders,
               SUM(1 - o.returned)                     AS completed_orders,
               (SELECT COUNT(*) FROM customers)        AS total_customers,
               COUNT(DISTINCT o.customer_id)           AS buying_customers,
               ROUND(100.0 * AVG(o.returned), 2)       AS return_rate_pct,
               MAX(o.order_date)                       AS data_until
        FROM orders o;""",
    "monthly": f"""
        SELECT strftime('%Y-%m', o.order_date) AS month,
               ROUND(SUM({NET}), 2)           AS net_revenue,
               COUNT(o.order_id)              AS orders
        FROM orders o
        GROUP BY month
        ORDER BY month;""",
    "category": f"""
        SELECT p.category,
               ROUND(SUM({NET}), 2)                                    AS net_revenue,
               COUNT(o.order_id)                                       AS orders,
               SUM(o.quantity)                                         AS units,
               ROUND(100.0 * SUM(o.returned) / COUNT(o.order_id), 2)   AS return_rate_pct
        FROM orders o
        JOIN products p ON o.product_id = p.product_id
        GROUP BY p.category
        ORDER BY net_revenue DESC;""",
    "city": f"""
        SELECT c.city,
               ROUND(SUM({NET}), 2)          AS net_revenue,
               COUNT(DISTINCT o.customer_id) AS customers
        FROM orders o
        JOIN customers c ON o.customer_id = c.customer_id
        GROUP BY c.city
        ORDER BY net_revenue DESC;""",
    "top_products": f"""
        SELECT p.product_name, p.category, SUM(o.quantity * (1 - o.returned)) AS units_sold,
               ROUND(SUM({NET}), 2) AS net_revenue
        FROM orders o
        JOIN products p ON o.product_id = p.product_id
        GROUP BY p.product_id
        ORDER BY net_revenue DESC
        LIMIT 5;""",
    "top_customers": f"""
        SELECT c.customer_name, c.city, SUM(1 - o.returned) AS orders,
               ROUND(SUM({NET}), 2) AS total_spending
        FROM orders o
        JOIN customers c ON o.customer_id = c.customer_id
        GROUP BY c.customer_id
        ORDER BY total_spending DESC
        LIMIT 5;""",
    "review_mood": """
        SELECT p.category,
               ROUND(100.0 * AVG(CASE WHEN r.rating <= 2 THEN 1 ELSE 0 END), 2) AS negative_pct,
               ROUND(100.0 * AVG(CASE WHEN r.rating = 3  THEN 1 ELSE 0 END), 2) AS neutral_pct,
               ROUND(100.0 * AVG(CASE WHEN r.rating >= 4 THEN 1 ELSE 0 END), 2) AS positive_pct,
               COUNT(*) AS reviews
        FROM reviews r
        JOIN products p ON r.product_id = p.product_id
        GROUP BY p.category
        ORDER BY negative_pct DESC;""",
}

# Same churn features as the notebook (Task D), for every customer at a chosen cutoff date
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
    SELECT COUNT(*) AS orders FROM orders
    WHERE customer_id = :customer_id AND order_date BETWEEN :start AND :end;"""

CUSTOMER_PROFILE_SQL = """
    SELECT customer_name, gender, age, city, membership_type, signup_date
    FROM customers WHERE customer_id = :customer_id;"""

RECENT_ORDERS_SQL = """
    SELECT o.order_date, p.product_name, p.category, o.quantity,
           ROUND(o.quantity * o.unit_price * (1 - o.discount), 0) AS amount,
           CASE WHEN o.returned = 1 THEN 'Returned' ELSE 'Kept' END AS status
    FROM orders o JOIN products p ON o.product_id = p.product_id
    WHERE o.customer_id = :customer_id AND o.order_date <= :cutoff
    ORDER BY o.order_date DESC
    LIMIT 5;"""

CONTACTS_SQL = "SELECT customer_id, email, phone FROM customers;"

MODEL_FEATURES = ["total_orders", "total_spending", "avg_order_value", "days_since_last_order",
                  "return_rate", "avg_delivery_days", "age", "membership_type"]
FEATURE_LABELS = {"total_orders": "Number of orders", "total_spending": "Total spending",
                  "avg_order_value": "Average order value", "days_since_last_order": "Days since last order",
                  "return_rate": "Return rate", "avg_delivery_days": "Delivery time", "age": "Age"}
DEMO_CUSTOMER_ID = 4755

# ------------------------------------------------------------------ page setup and style
st.set_page_config(page_title=f"{APP_NAME} | Seller analytics", page_icon="🛍️", layout="wide",
                   initial_sidebar_state="collapsed")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stMarkdown, input, textarea, button, select {
    font-family: 'Plus Jakarta Sans', 'Segoe UI', sans-serif !important;
}
#MainMenu, footer, header[data-testid="stHeader"], [data-testid="stToolbar"] { visibility: hidden; height: 0; }
[data-testid="stSidebar"], [data-testid="collapsedControl"] { display: none; }
.stApp { background: __CANVAS__; }
.block-container { padding-top: 1rem; padding-bottom: 3rem; max-width: 1320px; }

/* top bar */
.topbar { display: flex; align-items: center; justify-content: space-between;
          background: #FFFFFF; border: 1px solid __LINE__; border-radius: 14px;
          padding: 0.85rem 1.3rem; position: relative; overflow: hidden; margin-bottom: 0.8rem; }
.topbar::after { content: ""; position: absolute; left: 0; right: 0; bottom: 0; height: 4px;
    background: repeating-linear-gradient(90deg, __MADDER__ 0 16px, __INK__ 16px 19px, __SAFFRON__ 19px 26px,
                                          __INK__ 26px 29px, __EMERALD__ 29px 45px, __INK__ 45px 48px); }
.brand { display: flex; align-items: center; gap: 0.75rem; }
.logo { width: 40px; height: 40px; border-radius: 10px; background: __INK__; color: #FFFFFF;
        display: grid; place-items: center; font-weight: 800; font-size: 1.05rem; }
.brand-name { font-size: 1.3rem; font-weight: 800; color: __INK__; letter-spacing: -0.02em; line-height: 1.1; }
.brand-sub { font-size: 0.82rem; color: __INK_SOFT__; }
.status { font-size: 0.82rem; color: __INK_SOFT__; text-align: right; }
.status b { color: __EMERALD__; }

/* page intro */
.page-title { font-size: 1.55rem; font-weight: 800; color: __INK__; margin: 0.9rem 0 0.1rem; letter-spacing: -0.02em; }
.page-sub { color: __INK_SOFT__; font-size: 0.95rem; margin-bottom: 0.9rem; }
.section { font-size: 1.02rem; font-weight: 700; color: __INK__; margin: 0.4rem 0 0.35rem; }

/* KPI cards */
.kpi { background: #FFFFFF; border: 1px solid __LINE__; border-radius: 12px; padding: 0.95rem 1.1rem; height: 100%; margin-bottom: 0.6rem; }
.kpi .label { color: __INK_SOFT__; font-size: 0.82rem; font-weight: 600; }
.kpi .value { color: __INK__; font-size: 1.6rem; font-weight: 800; letter-spacing: -0.02em; margin: 0.15rem 0; }
.kpi .note { color: __INK_SOFT__; font-size: 0.78rem; }
.pill { display: inline-block; padding: 0.08rem 0.5rem; border-radius: 999px; font-size: 0.75rem; font-weight: 700; margin-right: 0.3rem; }
.pill.up { background: #E7F4EE; color: __EMERALD__; }
.pill.down { background: #FBEAE8; color: __MADDER__; }
.pill.neutral { background: #EEF1F6; color: __INK_SOFT__; }

/* panels: Streamlit bordered containers */
[data-testid="stVerticalBlockBorderWrapper"] { background: #FFFFFF; border-color: __LINE__ !important; border-radius: 14px !important; }

/* result banner */
.result { border-radius: 12px; padding: 1rem 1.2rem; background: var(--bg); color: var(--fg, #FFFFFF); }
.result .big { font-size: 1.45rem; font-weight: 800; }
.result .small { font-size: 0.9rem; opacity: 0.95; }

/* profile card */
.profile { display: flex; gap: 0.9rem; align-items: center; }
.avatar { width: 52px; height: 52px; border-radius: 50%; background: #E9EDF5; color: __INK__;
          display: grid; place-items: center; font-weight: 800; font-size: 1.1rem; }
.profile .name { font-weight: 800; color: __INK__; font-size: 1.15rem; }
.profile .meta { color: __INK_SOFT__; font-size: 0.85rem; }
.tag { display: inline-block; padding: 0.12rem 0.55rem; border-radius: 6px; font-size: 0.75rem; font-weight: 700;
       background: #EEF1F6; color: __INK__; margin-right: 0.3rem; }

/* tabs and buttons */
.stTabs [data-baseweb="tab-list"] { gap: 0.4rem; border-bottom: 1px solid __LINE__; }
.stTabs [data-baseweb="tab"] { font-weight: 700; padding: 0.5rem 0.9rem; }
.stButton button, .stFormSubmitButton button, .stDownloadButton button { border-radius: 9px; font-weight: 700; }
.stButton button[kind="primary"], .stFormSubmitButton button[kind="primary"], .stDownloadButton button[kind="primary"] {
    background: __INK__; border-color: __INK__; }
:focus-visible { outline: 2px solid __SKY__ !important; outline-offset: 2px; }
</style>
"""
for token, value in {"__INK_SOFT__": INK_SOFT, "__INK__": INK, "__EMERALD__": EMERALD, "__MADDER__": MADDER,
                     "__SAFFRON__": SAFFRON, "__SKY__": SKY, "__CANVAS__": CANVAS, "__LINE__": LINE}.items():
    CSS = CSS.replace(token, value)
st.markdown(CSS, unsafe_allow_html=True)


# ------------------------------------------------------------------ data and model helpers
@st.cache_data(show_spinner=False)
def run_query(query, params=None):
    """Run a SQL query on the SQLite database and return a DataFrame."""
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql(query, conn, params=params)


@st.cache_resource(show_spinner=False)
def load_model(filename):
    return joblib.load(MODELS_DIR / filename)


@st.cache_data(show_spinner=False)
def load_metadata(filename):
    with open(MODELS_DIR / filename) as f:
        return json.load(f)


def customer_features(cutoff):
    return run_query(CUSTOMER_FEATURES_SQL, {"cutoff": cutoff})


def risk_level(probability):
    if probability >= 0.7:
        return "High risk"
    if probability >= 0.5:
        return "Medium risk"
    return "Low risk"


@st.cache_data(show_spinner="Scoring every customer...")
def score_all_customers(cutoff):
    model = load_model("churn_model.joblib")
    scored = customer_features(cutoff).copy()
    scored["churn_probability"] = model.predict_proba(scored[MODEL_FEATURES])[:, 1]
    scored["risk"] = scored["churn_probability"].apply(risk_level)
    scored["value_at_risk"] = scored["churn_probability"] * scored["total_spending"]
    return scored.sort_values("value_at_risk", ascending=False)


def explain_prediction(model, customer):
    """How much each feature pushes this customer's score, compared with an average customer.

    Features are standardised inside the pipeline, so scaled value x coefficient is that push.
    """
    scaled = model.named_steps["preprocess"].transform(customer[MODEL_FEATURES])[0]
    names = [n.split("__")[1] for n in model.named_steps["preprocess"].get_feature_names_out()]
    pushes = pd.Series(scaled * model.named_steps["model"].coef_[0], index=names)
    pushes = pushes[[n for n in names if n in FEATURE_LABELS]]
    return pushes.reindex(pushes.abs().sort_values(ascending=False).index).head(4).rename(FEATURE_LABELS)


# ------------------------------------------------------------------ UI helpers
def pkr(value):
    if value >= 1e9:
        return f"PKR {value / 1e9:,.2f}B"
    if value >= 1e6:
        return f"PKR {value / 1e6:,.1f}M"
    if value >= 1e3:
        return f"PKR {value / 1e3:,.0f}K"
    return f"PKR {value:,.0f}"


def page_intro(title, subtitle):
    st.markdown(f'<div class="page-title">{title}</div><div class="page-sub">{subtitle}</div>',
                unsafe_allow_html=True)


def section(title):
    st.markdown(f'<div class="section">{title}</div>', unsafe_allow_html=True)


def kpi(column, label, value, note="", change=None, good_when_up=True):
    pill = ""
    if change is not None:
        up = change >= 0
        kind = "up" if up == good_when_up else "down"
        pill = f'<span class="pill {kind}">{"▲" if up else "▼"} {abs(change):.1f}%</span>'
    column.markdown(f'<div class="kpi"><div class="label">{label}</div><div class="value">{value}</div>'
                    f'<div class="note">{pill}{note}</div></div>', unsafe_allow_html=True)


def result_banner(title, detail, color):
    text = INK if color == SAFFRON else "#FFFFFF"
    st.markdown(f'<div class="result" style="--bg:{color};--fg:{text}"><div class="big">{title}</div>'
                f'<div class="small">{detail}</div></div>', unsafe_allow_html=True)


def style_chart(fig, height=340, legend=False):
    fig.update_layout(height=height, margin=dict(l=8, r=8, t=8, b=8), plot_bgcolor="#FFFFFF",
                      paper_bgcolor="#FFFFFF", showlegend=legend,
                      font=dict(family="Plus Jakarta Sans, Segoe UI, sans-serif", color=INK, size=12),
                      hoverlabel=dict(bgcolor="#FFFFFF", font_size=12))
    fig.update_xaxes(gridcolor="#EEF0F5", zeroline=False)
    fig.update_yaxes(gridcolor="#EEF0F5", zeroline=False)
    return fig


def churn_gauge(probability, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=probability * 100, number={"suffix": "%", "font": {"color": INK, "size": 40}},
        gauge={"axis": {"range": [0, 100], "tickcolor": INK_SOFT}, "bar": {"color": color, "thickness": 0.3},
               "steps": [{"range": [0, 50], "color": "#E7F4EE"}, {"range": [50, 70], "color": "#FDF1DC"},
                         {"range": [70, 100], "color": "#FBEAE8"}],
               "threshold": {"line": {"color": INK, "width": 3}, "value": 50}}))
    fig.update_layout(height=230, margin=dict(l=25, r=25, t=25, b=0), paper_bgcolor="#FFFFFF",
                      font=dict(family="Plus Jakarta Sans, Segoe UI, sans-serif"))
    return fig


def reasons_chart(reasons):
    fig = go.Figure(go.Bar(x=reasons.values, y=reasons.index, orientation="h",
                           marker_color=[MADDER if v > 0 else EMERALD for v in reasons.values],
                           hovertemplate="%{y}<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title_text="← lowers churn risk        raises churn risk →", showticklabels=False,
                     zeroline=True, zerolinecolor=INK)
    return style_chart(fig, 220)


RISK_ACTIONS = {"High risk": "Call or message now with a personal win-back offer.",
                "Medium risk": "Send a reminder with products they are likely to buy.",
                "Low risk": "Keep them engaged with normal updates. No discount needed."}


# ------------------------------------------------------------------ tab 1: overview
def overview_page():
    kpis = run_query(SQL["kpis"]).iloc[0]
    monthly = run_query(SQL["monthly"])
    last, prev = monthly.iloc[-1], monthly.iloc[-2]
    month_name = pd.to_datetime(last["month"]).strftime("%B %Y")
    page_intro("Store overview", f"How the store is doing. {month_name} is compared with the month before.")

    aov = kpis["net_revenue"] / kpis["completed_orders"]
    c1, c2, c3, c4, c5 = st.columns(5)
    kpi(c1, "Net revenue", pkr(kpis["net_revenue"]), "last month",
        change=(last["net_revenue"] / prev["net_revenue"] - 1) * 100)
    kpi(c2, "Total orders", f"{kpis['total_orders']:,.0f}", "last month",
        change=(last["orders"] / prev["orders"] - 1) * 100)
    kpi(c3, "Total customers", f"{kpis['total_customers']:,.0f}", f"{kpis['buying_customers']:,.0f} have ordered")
    kpi(c4, "Average order value", pkr(aov), "per completed order")
    kpi(c5, "Return rate", f"{kpis['return_rate_pct']:.1f}%", "of all orders")
    st.write("")

    left, right = st.columns([2, 1])
    with left, st.container(border=True):
        section("Monthly net revenue")
        chart = monthly.assign(month=pd.to_datetime(monthly["month"]))
        fig = px.area(chart, x="month", y="net_revenue", labels={"month": "", "net_revenue": "PKR"})
        fig.update_traces(line_color=EMERALD, line_width=2.5, fillcolor="rgba(14,124,90,0.10)",
                          hovertemplate="%{x|%b %Y}<br>PKR %{y:,.0f}<extra></extra>")
        st.plotly_chart(style_chart(fig, 320), width="stretch")
    with right, st.container(border=True):
        section("Revenue share by category")
        category = run_query(SQL["category"])
        palette = [INK, EMERALD, SKY, SAFFRON, MADDER, "#7A5AF8", "#8C95A6", "#C9CFDA"]
        fig = go.Figure(go.Pie(labels=category["category"], values=category["net_revenue"], hole=0.62,
                               marker=dict(colors=palette), sort=False, textinfo="none",
                               hovertemplate="%{label}<br>PKR %{value:,.0f} (%{percent})<extra></extra>"))
        top = category.iloc[0]
        fig.add_annotation(text=f"<b>{top['net_revenue'] / category['net_revenue'].sum():.0%}</b><br>{top['category']}",
                           showarrow=False, font=dict(size=15, color=INK))
        fig.update_layout(legend=dict(orientation="h", y=-0.05, font=dict(size=11)))
        st.plotly_chart(style_chart(fig, 320, legend=True), width="stretch")

    left, right = st.columns(2)
    with left, st.container(border=True):
        section("Revenue by category")
        data = category.sort_values("net_revenue")
        fig = px.bar(data, x="net_revenue", y="category", orientation="h", labels={"net_revenue": "PKR", "category": ""})
        fig.update_traces(marker_color=[EMERALD if c == top["category"] else INK for c in data["category"]],
                          hovertemplate="%{y}<br>PKR %{x:,.0f}<extra></extra>")
        st.plotly_chart(style_chart(fig, 320), width="stretch")
    with right, st.container(border=True):
        section("Revenue by city")
        city = run_query(SQL["city"]).sort_values("net_revenue")
        fig = px.bar(city, x="net_revenue", y="city", orientation="h", labels={"net_revenue": "PKR", "city": ""})
        fig.update_traces(marker_color=INK, hovertemplate="%{y}<br>PKR %{x:,.0f}<extra></extra>")
        st.plotly_chart(style_chart(fig, 320), width="stretch")

    with st.container(border=True):
        section("Return rate by category")
        data = category.sort_values("return_rate_pct", ascending=False)
        fig = px.bar(data, x="category", y="return_rate_pct", labels={"category": "", "return_rate_pct": "Returned (%)"})
        fig.update_traces(marker_color=[MADDER if r > kpis["return_rate_pct"] else "#C9CFDA" for r in data["return_rate_pct"]],
                          hovertemplate="%{x}<br>%{y:.2f}% returned<extra></extra>")
        fig.add_hline(y=kpis["return_rate_pct"], line_dash="dot", line_color=INK,
                      annotation_text=f"Store average {kpis['return_rate_pct']:.1f}%", annotation_position="top right")
        st.plotly_chart(style_chart(fig, 280), width="stretch")
        st.caption("Red: categories that return more than the store average.")

    left, right = st.columns(2)
    with left, st.container(border=True):
        section("Best-selling products")
        st.dataframe(run_query(SQL["top_products"]), hide_index=True, width="stretch", column_config={
            "product_name": "Product", "category": "Category", "units_sold": "Units",
            "net_revenue": st.column_config.NumberColumn("Net revenue (PKR)", format="%,.0f")})
    with right, st.container(border=True):
        section("Top customers")
        st.dataframe(run_query(SQL["top_customers"]), hide_index=True, width="stretch", column_config={
            "customer_name": "Customer", "city": "City", "orders": "Orders",
            "total_spending": st.column_config.NumberColumn("Spent (PKR)", format="%,.0f")})

    with st.expander("See the SQL behind this page"):
        for name in ["kpis", "monthly", "category", "city", "top_products", "top_customers"]:
            st.code(SQL[name].strip(), language="sql")


# ------------------------------------------------------------------ tab 2: churn predictor
def churn_page():
    page_intro("Churn predictor", "Enter a customer's purchase history to see how likely they are to stop buying.")
    model = load_model("churn_model.joblib")
    meta = load_metadata("churn_metadata.json")

    with st.form("churn_form"), st.container():
        section("Customer history")
        c1, c2, c3, c4 = st.columns(4)
        total_orders = c1.number_input("Total orders", min_value=1, max_value=500, value=3, step=1)
        total_spending = c2.number_input("Total spending (PKR)", min_value=0.0, value=22_654.0, step=1_000.0)
        avg_order_value = c3.number_input("Average order value (PKR)", min_value=0.0, value=7_179.0, step=500.0)
        days_since = c4.number_input("Days since last order", min_value=0, max_value=2_000, value=64, step=1)
        c5, c6, c7, c8 = st.columns(4)
        return_rate_pct = c5.slider("Return rate (%)", 0, 100, 0)
        delivery = c6.number_input("Average delivery days", min_value=1.0, max_value=10.0, value=3.5, step=0.5)
        age = c7.number_input("Age", min_value=18, max_value=80, value=31, step=1)
        membership = c8.selectbox("Membership", meta["membership_types"],
                                  index=meta["membership_types"].index("Standard"))
        submitted = st.form_submit_button("Predict churn", type="primary", width="stretch")

    if not submitted:
        st.info("Fill in the customer's history and select **Predict churn**. "
                "To check a real customer, open the **Retention Center** tab.")
        return

    customer = pd.DataFrame([{"total_orders": total_orders, "total_spending": total_spending,
                              "avg_order_value": avg_order_value, "days_since_last_order": days_since,
                              "return_rate": return_rate_pct / 100, "avg_delivery_days": delivery,
                              "age": age, "membership_type": membership}])
    probability = float(model.predict_proba(customer[MODEL_FEATURES])[0, 1])
    risk = risk_level(probability)
    prediction = "Likely to churn" if probability >= 0.5 else "Likely to stay"

    left, right = st.columns([1, 1.25])
    with left, st.container(border=True):
        st.plotly_chart(churn_gauge(probability, RISK_COLORS[risk]), width="stretch")
    with right, st.container(border=True):
        result_banner(prediction, f"{risk}. Churn probability: {probability:.1%}", RISK_COLORS[risk])
        st.markdown(f"**Suggested action:** {RISK_ACTIONS[risk]}")
        section("Why this prediction?")
        st.plotly_chart(reasons_chart(explain_prediction(model, customer)), width="stretch")
    m = meta["test_metrics"]
    st.caption(f"Model: {meta['model']} trained on orders up to {meta['feature_cutoff']}. "
               f"Test ROC-AUC {m['roc_auc']:.2f}, recall {m['recall']:.2f}, precision {m['precision']:.2f}.")


# ------------------------------------------------------------------ tab 3: review sentiment
EXAMPLE_REVIEWS = {
    "Happy customer": "Very satisfied, quality feels premium. Good value for money.",
    "Roman Urdu": "quality bohat kharab thi, packaging was damaged",
    "Mixed": "It does the job, packaging was average. Three stars overall.",
}


def sentiment_page():
    page_intro("Review sentiment", "Paste a customer review to see whether it is negative, neutral or positive.")
    model = load_model("sentiment_model.joblib")

    left, right = st.columns([1.3, 1])
    with left, st.container(border=True):
        section("Analyze a review")
        st.caption("Try an example:")
        cols = st.columns(len(EXAMPLE_REVIEWS))
        for col, (label, text) in zip(cols, EXAMPLE_REVIEWS.items()):
            if col.button(label, width="stretch"):
                st.session_state["review_text"] = text
        review = st.text_area("Customer review", key="review_text", height=120,
                              placeholder="e.g. Very satisfied, product bilkul theek nikla")
        analyze = st.button("Analyze sentiment", type="primary")

        if analyze and not review.strip():
            st.warning("Type or paste a review first.")
        elif analyze:
            label = model.predict([review])[0]
            probabilities = pd.Series(model.predict_proba([review])[0], index=model.classes_)
            result_banner(label, f"Confidence: {probabilities[label]:.0%}", SENTIMENT_COLORS[label])
            chart = probabilities.reindex(["Negative", "Neutral", "Positive"]).rename_axis("sentiment").reset_index(name="p")
            fig = px.bar(chart, x="p", y="sentiment", orientation="h", range_x=[0, 1],
                         labels={"p": "Probability", "sentiment": ""})
            fig.update_traces(marker_color=[SENTIMENT_COLORS[s] for s in chart["sentiment"]],
                              hovertemplate="%{y}: %{x:.0%}<extra></extra>")
            st.plotly_chart(style_chart(fig, 170), width="stretch")
        st.caption("Works best on review-style English and Roman Urdu. Very short phrases like \"I love it\" "
                   "may show as Neutral because the training reviews use a small, fixed vocabulary.")

    with right, st.container(border=True):
        section("What customers say, by category")
        mood = run_query(SQL["review_mood"]).sort_values("negative_pct")
        fig = go.Figure()
        for column, name, color in [("negative_pct", "Negative", MADDER), ("neutral_pct", "Neutral", SAFFRON),
                                    ("positive_pct", "Positive", EMERALD)]:
            fig.add_bar(y=mood["category"], x=mood[column], name=name, orientation="h", marker_color=color,
                        hovertemplate="%{y}<br>" + name + ": %{x:.1f}%<extra></extra>")
        fig.update_layout(barmode="stack", legend=dict(orientation="h", y=1.08, x=0))
        fig.update_xaxes(title_text="% of reviews", range=[0, 100])
        st.plotly_chart(style_chart(fig, 360, legend=True), width="stretch")
        st.caption("Based on star ratings in the reviews table (SQL).")


# ------------------------------------------------------------------ tab 4: retention center (enhanced feature)
def customer_360(meta, model):
    cutoff = meta["feature_cutoff"]
    start, end = meta["target_window"]
    customers = customer_features(cutoff)
    ids = customers["customer_id"].tolist()
    names = customers.set_index("customer_id")[["customer_name", "city"]]
    default = ids.index(DEMO_CUSTOMER_ID) if DEMO_CUSTOMER_ID in ids else 0

    selected = st.selectbox("Find a customer", ids, index=default,
                            format_func=lambda cid: f"#{cid}  {names.loc[cid, 'customer_name']} ({names.loc[cid, 'city']})",
                            help="Type a name or ID to search.")
    row = customers.loc[customers["customer_id"] == selected].iloc[[0]]
    profile = run_query(CUSTOMER_PROFILE_SQL, {"customer_id": int(selected)}).iloc[0]
    probability = float(model.predict_proba(row[MODEL_FEATURES])[0, 1])
    risk = risk_level(probability)
    actual_orders = int(run_query(ORDERS_IN_WINDOW_SQL, {"customer_id": int(selected), "start": start,
                                                         "end": end})["orders"].iloc[0])

    left, right = st.columns([1.15, 1])
    with left, st.container(border=True):
        initials = "".join(part[0] for part in profile["customer_name"].split()[:2]).upper()
        age = f"{int(profile['age'])} years" if pd.notna(profile["age"]) else "age unknown"
        st.markdown(f'<div class="profile"><div class="avatar">{initials}</div><div>'
                    f'<div class="name">{profile["customer_name"]}</div>'
                    f'<div class="meta">{profile["gender"]}, {age}, {profile["city"]}. Customer since {profile["signup_date"]}</div>'
                    f'<div style="margin-top:0.35rem"><span class="tag">{profile["membership_type"]}</span>'
                    f'<span class="tag">#{selected}</span></div></div></div>', unsafe_allow_html=True)
        st.write("")
        c1, c2, c3 = st.columns(3)
        kpi(c1, "Orders", f"{int(row['total_orders'].iloc[0])}", f"up to {cutoff}")
        kpi(c2, "Spent", pkr(row["total_spending"].iloc[0]), "net of returns")
        kpi(c3, "Last order", f"{int(row['days_since_last_order'].iloc[0])} days", "before the cutoff")
        section("Recent orders")
        st.dataframe(run_query(RECENT_ORDERS_SQL, {"customer_id": int(selected), "cutoff": cutoff}),
                     hide_index=True, width="stretch", column_config={
                         "order_date": "Date", "product_name": "Product", "category": "Category",
                         "quantity": "Qty", "amount": st.column_config.NumberColumn("Amount (PKR)", format="%,.0f"),
                         "status": "Status"})
    with right, st.container(border=True):
        st.plotly_chart(churn_gauge(probability, RISK_COLORS[risk]), width="stretch")
        result_banner("Likely to churn" if probability >= 0.5 else "Likely to stay",
                      f"{risk}. {RISK_ACTIONS[risk]}", RISK_COLORS[risk])
        churned = actual_orders == 0
        correct = churned == (probability >= 0.5)
        outcome = "did not buy again" if churned else f"placed {actual_orders} order(s)"
        st.markdown(f"**What really happened ({start} to {end}):** {outcome}. "
                    f"{'✅ The model was right.' if correct else '❌ The model was wrong this time.'}")
        section("Why this score?")
        st.plotly_chart(reasons_chart(explain_prediction(model, row)), width="stretch")
    st.caption(f"Prediction uses only orders up to {cutoff}, then we check the real outcome. "
               "This customer may have been part of the training data.")


def at_risk_list(scored, latest):
    high = scored[scored["risk"] == "High risk"]
    c1, c2, c3, c4 = st.columns(4)
    kpi(c1, "Customers scored", f"{len(scored):,}", f"orders up to {latest}")
    kpi(c2, "High risk", f"{len(high):,}", f"{len(high) / len(scored):.0%} of customers")
    kpi(c3, "Value at risk", pkr(high["value_at_risk"].sum()), "from high-risk customers")
    kpi(c4, "Call first", f"{(high['total_spending'] >= scored['total_spending'].quantile(0.75)).sum():,}",
        "high risk and top 25% spenders")
    st.write("")

    with st.container(border=True):
        f1, f2, f3, f4 = st.columns([1.2, 1.2, 1.2, 1])
        risks = f1.multiselect("Risk level", list(RISK_COLORS), default=["High risk"])
        cities = f2.multiselect("City", sorted(scored["city"].unique()), placeholder="All cities")
        tiers = f3.multiselect("Membership", sorted(scored["membership_type"].unique()), placeholder="All memberships")
        min_spend = f4.number_input("Minimum spending (PKR)", min_value=0, value=0, step=10_000)
        view = scored[scored["risk"].isin(risks or list(RISK_COLORS))
                      & scored["city"].isin(cities or scored["city"].unique())
                      & scored["membership_type"].isin(tiers or scored["membership_type"].unique())
                      & (scored["total_spending"] >= min_spend)]

        section(f"{len(view):,} customers, most valuable first")
        columns = ["customer_id", "customer_name", "city", "membership_type", "total_orders", "total_spending",
                   "days_since_last_order", "churn_probability", "value_at_risk"]
        st.dataframe(view[columns], hide_index=True, height=400, width="stretch", column_config={
            "customer_id": st.column_config.NumberColumn("ID", format="%d"),
            "customer_name": "Name", "city": "City", "membership_type": "Membership",
            "total_orders": st.column_config.NumberColumn("Orders", format="%d"),
            "total_spending": st.column_config.NumberColumn("Spent (PKR)", format="%,.0f"),
            "days_since_last_order": st.column_config.NumberColumn("Days since last order", format="%d"),
            "churn_probability": st.column_config.ProgressColumn("Churn probability", format="percent",
                                                                 min_value=0.0, max_value=1.0),
            "value_at_risk": st.column_config.NumberColumn("Value at risk (PKR)", format="%,.0f")})
        a, b = st.columns([3, 1])
        a.caption("Value at risk = churn probability × what the customer has spent. "
                  "Big spenders about to leave come first.")
        b.download_button("Download call list (CSV)", view[columns].to_csv(index=False).encode("utf-8"),
                          file_name="bazaariq_call_list.csv", mime="text/csv", type="primary", width="stretch")


def winback_planner(scored):
    st.markdown("Plan a win-back campaign: choose who to target and your offer, and see if it pays off.")
    left, right = st.columns([1, 1.4])
    with left, st.container(border=True):
        section("Campaign settings")
        risks = st.multiselect("Target customers", list(RISK_COLORS), default=["High risk"], key="plan_risk")
        min_spend = st.number_input("Only customers who spent at least (PKR)", min_value=0, value=50_000,
                                    step=10_000, key="plan_spend")
        discount = st.slider("Discount on their next order (%)", 0, 50, 15, key="plan_discount")
        win_rate = st.slider("Customers expected to come back (%)", 1, 60, 20, key="plan_win",
                             help="Your assumption. 10-30% is common for win-back offers.")
    target = scored[scored["risk"].isin(risks or list(RISK_COLORS)) & (scored["total_spending"] >= min_spend)]
    returning = len(target) * win_rate / 100
    protected = target["value_at_risk"].sum() * win_rate / 100
    cost = (target["avg_order_value"] * discount / 100).sum() * win_rate / 100
    net = protected - cost

    with right, st.container(border=True):
        section("Expected result")
        c1, c2 = st.columns(2)
        kpi(c1, "Customers targeted", f"{len(target):,}", f"about {returning:,.0f} expected back")
        kpi(c2, "Revenue protected", pkr(protected), "value at risk × come-back rate")
        c3, c4 = st.columns(2)
        kpi(c3, "Discount cost", pkr(cost), "discount on one order each")
        kpi(c4, "Net benefit", pkr(net), f"about {protected / cost:,.0f}× the cost" if cost else "no cost")
        if len(target):
            by_city = (target.groupby("city", as_index=False)["value_at_risk"].sum()
                       .sort_values("value_at_risk", ascending=False).head(6))
            fig = px.bar(by_city, x="city", y="value_at_risk", labels={"city": "", "value_at_risk": "Value at risk (PKR)"})
            fig.update_traces(marker_color=INK, hovertemplate="%{x}<br>PKR %{y:,.0f}<extra></extra>")
            st.plotly_chart(style_chart(fig, 220), width="stretch")
            st.caption("Where the targeted value is. Start the campaign in these cities.")
    st.caption("These are estimates based on your assumptions, not guarantees.")


# ------------------------------------------------------------------ win-back campaign (messages)
DEFAULT_SUBJECT = "We miss you, {first_name}! {discount}% off your next order"
DEFAULT_EMAIL_BODY = """Assalam o Alaikum {first_name},

It has been a while since your last order, and we miss you.
Here is {discount}% off your next order, just for you.

Your code: {code} (valid for 14 days)

Thank you for shopping with us.
Team BazaarIQ

You are receiving this because you are our customer. Reply STOP to unsubscribe."""
DEFAULT_SMS = "Hi {first_name}, we miss you! Get {discount}% off your next order with code {code}. Valid 14 days. Reply STOP to opt out."


class _KeepUnknown(dict):
    """Leave unknown {placeholders} as they are instead of crashing."""
    def __missing__(self, key):
        return "{" + key + "}"


def personalise(template, row, discount):
    values = _KeepUnknown(first_name=str(row["customer_name"]).split()[0], name=row["customer_name"],
                          city=row["city"], discount=discount, code=f"COMEBACK{discount}-{int(row['customer_id'])}")
    return template.format_map(values)


def smtp_settings():
    """SMTP details from Streamlit secrets, or None if they are not set up."""
    try:
        return dict(st.secrets["smtp"])
    except Exception:
        return None


def send_email(settings, to_address, subject, body):
    message = EmailMessage()
    message["From"] = settings.get("sender", settings["user"])
    message["To"] = to_address
    message["Subject"] = subject
    message.set_content(body)
    port = int(settings.get("port", 465))
    if port == 465:
        with smtplib.SMTP_SSL(settings["host"], port, context=ssl.create_default_context(), timeout=20) as server:
            server.login(settings["user"], settings["password"])
            server.send_message(message)
    else:
        with smtplib.SMTP(settings["host"], port, timeout=20) as server:
            server.starttls(context=ssl.create_default_context())
            server.login(settings["user"], settings["password"])
            server.send_message(message)


def winback_campaign(scored):
    st.markdown("Write one message, and every at-risk customer gets a personal copy with their name and discount code.")
    contacts = run_query(CONTACTS_SQL)

    left, right = st.columns([1, 1.25])
    with left, st.container(border=True):
        section("Who gets it")
        risks = st.multiselect("Customers", list(RISK_COLORS), default=["High risk"], key="camp_risk")
        min_spend = st.number_input("Only customers who spent at least (PKR)", min_value=0, value=50_000,
                                    step=10_000, key="camp_spend")
        max_people = st.number_input("Maximum people (most valuable first)", min_value=1, max_value=5_000,
                                     value=50, step=10, key="camp_max")
        channel = st.radio("Channel", ["Email", "SMS"], horizontal=True, key="camp_channel")
        discount = st.slider("Discount (%)", 5, 50, 15, step=5, key="camp_discount")

        section("Message")
        st.caption("You can use {first_name}, {name}, {city}, {discount} and {code}.")
        if channel == "Email":
            subject = st.text_input("Subject", DEFAULT_SUBJECT, key="camp_subject")
            body = st.text_area("Email text", DEFAULT_EMAIL_BODY, height=230, key="camp_body")
        else:
            subject = ""
            body = st.text_area("SMS text", DEFAULT_SMS, height=110, key="camp_sms")

    targets = (scored[scored["risk"].isin(risks or list(RISK_COLORS)) & (scored["total_spending"] >= min_spend)]
               .head(int(max_people)).merge(contacts, on="customer_id", how="left"))
    if not targets.empty:
        targets["code"] = [f"COMEBACK{discount}-{cid}" for cid in targets["customer_id"]]
        targets["subject"] = [personalise(subject, r, discount) for _, r in targets.iterrows()]
        targets["message"] = [personalise(body, r, discount) for _, r in targets.iterrows()]
        targets["send_to"] = targets["email"] if channel == "Email" else targets["phone"]

    with right, st.container(border=True):
        section("Preview")
        if targets.empty:
            st.info("No customers match these settings. Try a lower minimum spending or another risk level.")
            return
        first = targets.iloc[0]
        to_line = f"<b>To:</b> {first['send_to']}" + (f"<br><b>Subject:</b> {first['subject']}" if channel == "Email" else "")
        st.markdown(f'<div class="kpi" style="font-size:0.88rem;color:{INK_SOFT}">{to_line}</div>', unsafe_allow_html=True)
        st.code(first["message"], language=None, wrap_lines=True)
        if channel == "SMS":
            st.caption(f"{len(first['message'])} characters" + (" (more than one SMS)" if len(first["message"]) > 160 else ""))

        c1, c2 = st.columns(2)
        kpi(c1, "Recipients", f"{len(targets):,}", f"{channel.lower()} messages")
        kpi(c2, "Value at risk", pkr(targets["value_at_risk"].sum()), "in this group")

        export_cols = ["customer_id", "customer_name", "city", "send_to", "code", "subject", "message"]
        if channel == "SMS":
            export_cols.remove("subject")
        b1, b2 = st.columns(2)
        if b1.button(f"Send {channel.lower()} campaign", type="primary", width="stretch"):
            st.session_state.setdefault("campaign_log", []).insert(0, {
                "time": datetime.now().strftime("%Y-%m-%d %H:%M"), "channel": channel,
                "recipients": len(targets), "discount": f"{discount}%",
                "value_at_risk": round(float(targets["value_at_risk"].sum())), "status": "Queued (demo)"})
            st.success(f"{len(targets):,} personalised {channel.lower()} messages prepared and logged.")
        b2.download_button("Download messages (CSV)", targets[export_cols].to_csv(index=False).encode("utf-8"),
                           file_name=f"bazaariq_{channel.lower()}_campaign.csv", mime="text/csv", width="stretch")
        st.caption("Demo mode: the contacts in this dataset are sample data, so messages are prepared and logged, "
                   "not delivered. Upload the CSV to an email or SMS service to send them for real.")

    section("Recipients")
    st.dataframe(targets[["customer_id", "customer_name", "city", "send_to", "code", "churn_probability", "value_at_risk"]],
                 hide_index=True, height=260, width="stretch", column_config={
                     "customer_id": st.column_config.NumberColumn("ID", format="%d"),
                     "customer_name": "Name", "city": "City", "send_to": "Send to", "code": "Discount code",
                     "churn_probability": st.column_config.ProgressColumn("Churn probability", format="percent",
                                                                          min_value=0.0, max_value=1.0),
                     "value_at_risk": st.column_config.NumberColumn("Value at risk (PKR)", format="%,.0f")})

    if st.session_state.get("campaign_log"):
        section("Campaign history (this session)")
        st.dataframe(pd.DataFrame(st.session_state["campaign_log"]), hide_index=True, width="stretch")

    if channel == "Email":
        with st.expander("Send a real test email to yourself"):
            settings = smtp_settings()
            test_to = st.text_input("Your email address", key="camp_test_to")
            if settings is None:
                st.info("To send real emails, add your SMTP details in Streamlit Cloud under "
                        "**App settings → Secrets** (see the format below), then reload the app.")
                st.code('[smtp]\nhost = "smtp.gmail.com"\nport = 465\nuser = "you@gmail.com"\n'
                        'password = "your-16-letter-app-password"\nsender = "BazaarIQ <you@gmail.com>"', language="toml")
            elif st.button("Send test email", disabled=not test_to.strip()):
                try:
                    send_email(settings, test_to.strip(), first["subject"], first["message"])
                    st.success(f"Test email sent to {test_to.strip()}. Check your inbox (and spam folder).")
                except Exception as error:
                    st.error(f"Could not send the email: {error}. Check the SMTP details in Secrets.")


def retention_page():
    latest = run_query("SELECT MAX(order_date) AS latest FROM orders;")["latest"].iloc[0]
    page_intro("Retention Center", "Find who is about to leave, understand why, and plan how to win them back.")
    model = load_model("churn_model.joblib")
    meta = load_metadata("churn_metadata.json")
    scored = score_all_customers(latest)

    tab1, tab2, tab3, tab4 = st.tabs(["Customer 360", "At-risk customers", "Win-back planner", "Win-back campaign"])
    with tab1:
        customer_360(meta, model)
    with tab2:
        at_risk_list(scored, latest)
    with tab3:
        winback_planner(scored)
    with tab4:
        winback_campaign(scored)


# ------------------------------------------------------------------ layout
PAGES = {
    "Overview": ("speedometer2", overview_page),
    "Churn Predictor": ("person-dash", churn_page),
    "Review Sentiment": ("chat-square-heart", sentiment_page),
    "Retention Center": ("stars", retention_page),
}


def top_bar():
    data_until = run_query(SQL["kpis"])["data_until"].iloc[0]
    st.markdown(f'<div class="topbar"><div class="brand"><div class="logo">B</div><div>'
                f'<div class="brand-name">{APP_NAME}</div>'
                f'<div class="brand-sub">Seller analytics and customer retention</div></div></div>'
                f'<div class="status">Live from <b>ecommerce_clean.db</b><br>Orders up to {data_until}</div></div>',
                unsafe_allow_html=True)


def main():
    top_bar()
    choice = option_menu(
        None, list(PAGES), icons=[icon for icon, _ in PAGES.values()], orientation="horizontal", default_index=0,
        styles={
            "container": {"padding": "5px", "background-color": "#FFFFFF", "border": f"1px solid {LINE}",
                          "border-radius": "12px", "max-width": "100%"},
            "icon": {"font-size": "15px"},
            "nav-link": {"font-size": "14px", "font-weight": "600", "color": INK, "margin": "0 3px",
                         "border-radius": "9px", "--hover-color": "#EEF1F6",
                         "font-family": "Plus Jakarta Sans, Segoe UI, sans-serif"},
            "nav-link-selected": {"background-color": INK, "color": "#FFFFFF", "font-weight": "700"},
        })
    PAGES[choice or "Overview"][1]()


if __name__ == "__main__":
    main()
