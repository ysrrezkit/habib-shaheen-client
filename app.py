
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
MODEL = None

if HF_TOKEN:
    try:
        client = InferenceClient(token=HF_TOKEN)
        logger.info("Hugging Face client initialized successfully.")
        MODEL = "meta-llama/Meta-Llama-3-8B-Instruct" 
    except Exception as e:
        logger.error(f"HF init error: {e}")

# =========================
# AI INSIGHTS
# =========================
def generate_insights(dataframe):
    if dataframe.empty or client is None or MODEL is None:
        return "⚡ AI Insights are currently warming up. Select filters to update analysis."

    try:
        dataframe = dataframe.fillna(0)
        summary_text = f"""
Revenue: {dataframe['Total (EGP)'].sum():,.0f}
Orders: {dataframe['Sale ID'].nunique()}
Customers: {dataframe['Customer'].nunique()}
Top City: {dataframe.groupby('City')['Total (EGP)'].sum().idxmax()}
Top Category: {dataframe.groupby('Category')['Total (EGP)'].sum().idxmax()}
Top Rep: {dataframe.groupby('Rep')['Total (EGP)'].sum().idxmax()}
"""

        messages = [
            {"role": "system", "content": "You are a senior business analyst. Provide 5 short, actionable, and professional insights based on the data summary provided. your answers is breif short no more than 3 lines and no bolding only the insights."},
            {"role": "user", "content": summary_text}
        ]

        response = client.chat_completion(
            model=MODEL,
            messages=messages,
            max_tokens=300,
            temperature=0.5,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"AI system response: {str(e)[:200]}"

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
# THEME STYLING CONFIG (Cyber Dark)
# =========================
DARK_STYLE = {
    "background-color": "#0b0f19",
    "color": "#f1f5f9",
    "font-family": "'Inter', 'Segoe UI', sans-serif",
    "min-height": "100vh",
    "padding": "24px"
}

CARD_STYLE = {
    "background": "linear-gradient(145deg, #111827, #1f2937)",
    "border": "1px solid #2d3748",
    "border-radius": "12px",
    "box-shadow": "0 4px 20px 0 rgba(0, 0, 0, 0.3)",
    "padding": "15px",
    "transition": "transform 0.2s"
}

PLOTLY_DARK_LAYOUT = {
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "font": {"color": "#94a3b8", "family": "'Inter', sans-serif"},
    "xaxis": {"gridcolor": "#1e293b", "zerolinecolor": "#1e293b"},
    "yaxis": {"gridcolor": "#1e293b", "zerolinecolor": "#1e293b"},
    "margin": {"t": 40, "b": 40, "l": 40, "r": 40}
}

# =========================
# APP INITIALIZATION
# =========================
app = Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
server = app.server
app.title = "Executive Sales Command"

# =========================
# LAYOUT
# =========================
app.layout = html.Div(style=DARK_STYLE, children=[
    dbc.Container([
        
        # HEADER
        html.Div([
            html.H1("EXECUTIVE SALES COMMAND", 
                    style={"letter-spacing": "2px", "font-weight": "800", "background": "linear-gradient(to right, #38bdf8, #818cf8)", "-webkit-background-clip": "text", "-webkit-text-fill-color": "transparent"}),
            html.P("Real-time Business Intelligence Engine", style={"color": "#64748b", "font-size": "14px", "margin-top": "-5px"})
        ], className="text-center my-4"),

        # FILTERS
        dbc.Row([
            dbc.Col([
                html.Label("📊 Year Window", style={"color": "#38bdf8", "font-weight": "600", "margin-bottom": "6px"}),
                dcc.Dropdown(
                    id="year-filter",
                    options=[{"label": str(y), "value": y} for y in sorted(df["Year"].unique())],
                    value=list(df["Year"].unique()),
                    multi=True,
                    className="dash-bootstrap"
                )
            ], md=4, className="mb-3"),

            dbc.Col([
                html.Label("🛍️ Product Category", style={"color": "#38bdf8", "font-weight": "600", "margin-bottom": "6px"}),
                dcc.Dropdown(
                    id="category-filter",
                    options=[{"label": c, "value": c} for c in sorted(df["Category"].unique())],
                    value=list(df["Category"].unique()),
                    multi=True,
                    className="dash-bootstrap"
                )
            ], md=4, className="mb-3"),

            dbc.Col([
                html.Label("📍 Regional City", style={"color": "#38bdf8", "font-weight": "600", "margin-bottom": "6px"}),
                dcc.Dropdown(
                    id="city-filter",
                    options=[{"label": c, "value": c} for c in sorted(df["City"].unique())],
                    value=list(df["City"].unique()),
                    multi=True,
                    className="dash-bootstrap"
                )
            ], md=4, className="mb-3"),
        ], className="mb-4"),

        # KPI CARDS
        dbc.Row([
            dbc.Col(html.Div(style=CARD_STYLE, children=[
                html.H6("TOTAL REVENUE", style={"color": "#64748b", "font-weight": "700", "letter-spacing": "1px"}),
                html.H2(id="total-sales", style={"color": "#34d399", "font-weight": "700"})
            ]), md=3, className="mb-3"),
            
            dbc.Col(html.Div(style=CARD_STYLE, children=[
                html.H6("VOLUME ORDERS", style={"color": "#64748b", "font-weight": "700", "letter-spacing": "1px"}),
                html.H2(id="total-orders", style={"color": "#38bdf8", "font-weight": "700"})
            ]), md=3, className="mb-3"),
            
            dbc.Col(html.Div(style=CARD_STYLE, children=[
                html.H6("ACTIVE CUSTOMERS", style={"color": "#64748b", "font-weight": "700", "letter-spacing": "1px"}),
                html.H2(id="total-customers", style={"color": "#a78bfa", "font-weight": "700"})
            ]), md=3, className="mb-3"),
            
            dbc.Col(html.Div(style=CARD_STYLE, children=[
                html.H6("AVG BASKET VALUE", style={"color": "#64748b", "font-weight": "700", "letter-spacing": "1px"}),
                html.H2(id="avg-order", style={"color": "#fb923c", "font-weight": "700"})
            ]), md=3, className="mb-3"),
        ], className="mb-4"),

        # AI INSIGHTS BLOCK
        html.Div(style={**CARD_STYLE, "background": "linear-gradient(145deg, #1e1b4b, #111827)", "border-color": "#4338ca"}, children=[
            html.H5("✨ Neural AI Business Insights", style={"color": "#818cf8", "font-weight": "700", "margin-bottom": "12px"}),
            html.Div(id="ai-insights", style={
                "color": "#cbd5e1", 
                "whiteSpace": "pre-line", 
                "font-size": "14px", 
                "line-height": "1.7",
                "font-family": "inherit"
            })
        ], className="mb-4"),

        # CHARTS GRID
        dbc.Row([
            dbc.Col(html.Div(style=CARD_STYLE, children=[dcc.Graph(id="sales-trend", config={"displayModeBar": False})]), md=6, className="mb-4"),
            dbc.Col(html.Div(style=CARD_STYLE, children=[dcc.Graph(id="category-chart", config={"displayModeBar": False})]), md=6, className="mb-4"),
        ]),

        dbc.Row([
            dbc.Col(html.Div(style=CARD_STYLE, children=[dcc.Graph(id="city-chart", config={"displayModeBar": False})]), md=6, className="mb-4"),
            dbc.Col(html.Div(style=CARD_STYLE, children=[dcc.Graph(id="rep-chart", config={"displayModeBar": False})]), md=6, className="mb-4"),
        ]),

        # DATATABLE
        html.H4("Detailed Transaction Ledger", className="mt-2 mb-3", style={"color": "#94a3b8", "font-weight": "600"}),
        html.Div(style={"border-radius": "12px", "overflow": "hidden", "border": "1px solid #2d3748"}, children=[
            dash_table.DataTable(
                id="sales-table",
                page_size=10,
                style_table={"overflowX": "auto"},
                style_cell={
                    "textAlign": "left", 
                    "backgroundColor": "#111827", 
                    "color": "#cbd5e1",
                    "border": "1px solid #1f2937",
                    "padding": "12px 15px",
                    "font-family": "'Inter', sans-serif"
                },
                style_header={
                    "backgroundColor": "#1f2937",
                    "color": "#38bdf8",
                    "fontWeight": "bold",
                    "border": "1px solid #2d3748"
                },
                style_data_conditional=[{
                    'if': {'row_index': 'odd'},
                    'backgroundColor': '#1f2937',
                }]
            )
        ], className="mb-5")

    ], fluid=True)
])

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
    years = years or list(df["Year"].unique())
    categories = categories or list(df["Category"].unique())
    cities = cities or list(df["City"].unique())

    filtered = df[
        df["Year"].isin(years) &
        df["Category"].isin(categories) &
        df["City"].isin(cities)
    ]

    total, orders, customers, avg = calculate_kpis(filtered)

    if not filtered.empty:
        # Trend
        trend_df = filtered.groupby("Month")["Total (EGP)"].sum().reset_index()
        trend = px.line(trend_df, x="Month", y="Total (EGP)", title="Monthly Performance Scale Grid")
        trend.update_traces(line=dict(color="#38bdf8", width=3), mode='lines+markers')
        
        # Category
        cat_df = filtered.groupby("Category")["Total (EGP)"].sum().reset_index()
        category = px.bar(cat_df, x="Category", y="Total (EGP)", title="Revenue Distribution by Verticals")
        category.update_traces(marker_color="#818cf8", marker_line_color="#4338ca", marker_line_width=1.5)
        
        # City
        city_df = filtered.groupby("City")["Total (EGP)"].sum().reset_index()
        city = px.pie(city_df, names="City", values="Total (EGP)", title="Market Contribution Share", hole=0.4)
        city.update_traces(textinfo='percent+label', marker=dict(colors=["#38bdf8", "#818cf8", "#a78bfa", "#f43f5e"]))
        
        # Reps
        rep_df = filtered.groupby("Rep")["Total (EGP)"].sum().reset_index().sort_values(by="Total (EGP)", ascending=True)
        rep = px.bar(rep_df, x="Total (EGP)", y="Rep", orientation='h', title="Top Representative Pipelines")
        rep.update_traces(marker_color="#34d399", marker_line_color="#059669", marker_line_width=1.5)
    else:
        trend, category, city, rep = px.line(), px.bar(), px.pie(), px.bar()

    for fig in [trend, category, city, rep]:
        fig.update_layout(**PLOTLY_DARK_LAYOUT)
        fig.update_layout(title={"font": {"size": 15, "color": "#f1f5f9"}})

    city.update_layout(showlegend=False)

    return (
        f"{total:,.0f} EGP",
        f"{orders:,}",
        f"{customers:,}",
        f"{avg:,.0f} EGP",
        trend,
        category,
        city,
        rep,
        filtered.to_dict("records"),
        [{"name": i, "id": i} for i in filtered.columns if i not in ["Month", "Year"]],
        generate_insights(filtered)
    )

# =========================
# RUN
# =========================
if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=int(os.getenv("PORT", 8050)))
