import pandas as pd
from dash import Dash, dcc, html, dash_table, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import os
import logging
from huggingface_hub import InferenceClient

# =========================
# LOGGING
# =========================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("sales_db.csv")
df["Date"] = pd.to_datetime(df["Date"])
df["Month"] = df["Date"].dt.strftime("%m")
df["Year"] = df["Date"].dt.year

# =========================
# HF CLIENT
# =========================
HF_TOKEN = os.getenv("HF_TOKEN")

client = None
if HF_TOKEN:
    try:
        client = InferenceClient(
            model="HuggingFaceH4/zephyr-7b-beta",
            token=HF_TOKEN
        )
    except Exception as e:
        logger.error(f"HF init error: {e}")

# =========================
# AI INSIGHTS
# =========================
def generate_insights(dataframe):
    if dataframe.empty or client is None:
        return "AI insights unavailable."

    try:
        summary_text = f"""
Revenue: {dataframe['Total (EGP)'].sum():,.0f}
Orders: {dataframe['Sale ID'].nunique()}
Customers: {dataframe['Customer'].nunique()}
Top City: {dataframe.groupby('City')['Total (EGP)'].sum().idxmax()}
Top Category: {dataframe.groupby('Category')['Total (EGP)'].sum().idxmax()}
Top Rep: {dataframe.groupby('Rep')['Total (EGP)'].sum().idxmax()}
"""

        prompt = f"""
You are a senior business analyst.
Write 5 short insights based on:

{summary_text}
"""

        response = client.text_generation(
            prompt,
            max_new_tokens=200,
            temperature=0.7,
            top_p=0.9
        )

        return response

    except Exception as e:
        return f"AI error: {str(e)[:200]}"
# =========================
# KPI
# =========================
def calculate_kpis(df_):
    if df_.empty:
        return 0, 0, 0, 0

    return (
        df_["Total (EGP)"].sum(),
        df_["Sale ID"].nunique(),
        df_["Customer"].nunique(),
        df_["Total (EGP)"].mean()
    )

# =========================
# APP
# =========================
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server
app.title = "Sales Dashboard"

# =========================
# LAYOUT (clean - no logos)
# =========================
app.layout = dbc.Container([

    html.H2("Sales Dashboard", className="text-center my-3"),

    # FILTERS
    dbc.Row([

        dbc.Col(dcc.Dropdown(
            id="year-filter",
            options=[{"label": str(y), "value": y} for y in sorted(df["Year"].unique())],
            value=list(df["Year"].unique()),
            multi=True
        ), md=4),

        dbc.Col(dcc.Dropdown(
            id="category-filter",
            options=[{"label": c, "value": c} for c in sorted(df["Category"].unique())],
            value=list(df["Category"].unique()),
            multi=True
        ), md=4),

        dbc.Col(dcc.Dropdown(
            id="city-filter",
            options=[{"label": c, "value": c} for c in sorted(df["City"].unique())],
            value=list(df["City"].unique()),
            multi=True
        ), md=4),

    ], className="mb-3"),

    # KPI
    dbc.Row([

        dbc.Col(dbc.Card(dbc.CardBody([html.H5("Revenue"), html.H3(id="total-sales")])), md=3),
        dbc.Col(dbc.Card(dbc.CardBody([html.H5("Orders"), html.H3(id="total-orders")])), md=3),
        dbc.Col(dbc.Card(dbc.CardBody([html.H5("Customers"), html.H3(id="total-customers")])), md=3),
        dbc.Col(dbc.Card(dbc.CardBody([html.H5("Avg Order"), html.H3(id="avg-order")])), md=3),

    ], className="mb-3"),

    # AI
    html.Div([
        html.H4("AI Insights"),
        html.Pre(id="ai-insights")
    ], className="mb-3"),

    # CHARTS
    dbc.Row([
        dbc.Col(dcc.Graph(id="sales-trend"), md=6),
        dbc.Col(dcc.Graph(id="category-chart"), md=6),
    ]),

    dbc.Row([
        dbc.Col(dcc.Graph(id="city-chart"), md=6),
        dbc.Col(dcc.Graph(id="rep-chart"), md=6),
    ]),

    # TABLE
    html.H4("Transactions", className="mt-4"),
    dash_table.DataTable(
        id="sales-table",
        page_size=10,
        style_table={"overflowX": "auto"},
        style_cell={"textAlign": "left"}
    )

], fluid=True)

# =========================
# CALLBACK
# =========================
@app.callback(
    [
        Output("total-sales", "children"),
        Output("total-orders", "children"),
        Output("total-customers", "children"),
        Output("avg-order", "children"),
        Output("sales-trend", "figure"),
        Output("category-chart", "figure"),
        Output("city-chart", "figure"),
        Output("rep-chart", "figure"),
        Output("sales-table", "data"),
        Output("sales-table", "columns"),
        Output("ai-insights", "children"),
    ],
    [
        Input("year-filter", "value"),
        Input("category-filter", "value"),
        Input("city-filter", "value"),
    ]
)
def update_dashboard(years, categories, cities):

    # SAFE DEFAULTS (prevents crash)
    years = years or list(df["Year"].unique())
    categories = categories or list(df["Category"].unique())
    cities = cities or list(df["City"].unique())

    filtered = df[
        df["Year"].isin(years) &
        df["Category"].isin(categories) &
        df["City"].isin(cities)
    ]

    total, orders, customers, avg = calculate_kpis(filtered)

    # charts
    trend = px.line(filtered.groupby("Month")["Total (EGP)"].sum().reset_index(),
                    x="Month", y="Total (EGP)") if not filtered.empty else px.line()

    category = px.bar(filtered.groupby("Category")["Total (EGP)"].sum().reset_index(),
                      x="Category", y="Total (EGP)") if not filtered.empty else px.bar()

    city = px.pie(filtered.groupby("City")["Total (EGP)"].sum().reset_index(),
                  names="City", values="Total (EGP)") if not filtered.empty else px.pie()

    rep = px.bar(filtered.groupby("Rep")["Total (EGP)"].sum().reset_index(),
                 x="Rep", y="Total (EGP)") if not filtered.empty else px.bar()

    return (
        f"{total:,.0f}",
        f"{orders}",
        f"{customers}",
        f"{avg:,.0f}",
        trend,
        category,
        city,
        rep,
        filtered.to_dict("records"),
        [{"name": i, "id": i} for i in filtered.columns],
        generate_insights(filtered)
    )

# =========================
# RUN
# =========================
if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=int(os.getenv("PORT", 8050)))