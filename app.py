import pandas as pd
from dash import Dash, dcc, html, dash_table, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import os
import logging
from huggingface_hub import InferenceClient

# =========================
# LOGGING SETUP
# =========================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =========================
# Load Data
# =========================
try:
    df = pd.read_csv("sales_db.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df["Month"] = df["Date"].dt.strftime("%m")
    df["Year"] = df["Date"].dt.year
    logger.info("✅ Data loaded successfully")
except Exception as e:
    logger.error(f"❌ Failed to load data: {e}")
    raise

# =========================
# LLM CONFIG (Hugging Face)
# =========================
HF_TOKEN = os.getenv("HF_TOKEN")

# Initialize client safely
client = None
if HF_TOKEN:
    try:
        client = InferenceClient(
            model="mistralai/Mistral-7B-Instruct-v0.2",
            token=HF_TOKEN
        )
        logger.info("✅ HuggingFace client initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize HF client: {e}")
        client = None
else:
    logger.warning("⚠️ HF_TOKEN not set - AI insights will be unavailable")

def generate_insights(dataframe):
    """Generate AI insights from sales data"""
    
    # Handle empty dataframe
    if dataframe.empty:
        return "No data available for insights."
    
    # Handle missing client
    if not client:
        return "AI insights unavailable - HF_TOKEN not configured."
    
    try:
        summary_text = f"""
Sales Summary:
- Total Revenue: {dataframe['Total (EGP)'].sum():,.0f}
- Orders: {dataframe['Sale ID'].nunique()}
- Customers: {dataframe['Customer'].nunique()}
- Top City: {dataframe.groupby('City')['Total (EGP)'].sum().idxmax()}
- Top Category: {dataframe.groupby('Category')['Total (EGP)'].sum().idxmax()}
- Top Rep: {dataframe.groupby('Rep')['Total (EGP)'].sum().idxmax()}
"""

        prompt = f"""You are a senior business analyst. Based on the data below, write 5 short business insights:

{summary_text}"""

        logger.info("🔄 Requesting AI insights...")
        
        message = client.text_generation(prompt,max_new_tokens=200,temperature=0.7,top_p=0.9,timeout=30) #type: ignore
        logger.info("✅ AI insights generated")
        return message
        
    except TimeoutError:
        logger.warning("⏱️ HF API request timed out")
        return "AI insights generation timed out. Please try again."
    except Exception as e:
        logger.error(f"❌ Error generating insights: {type(e).__name__}: {e}")
        return f"AI insights error: {str(e)[:100]}"

# =========================
# KPI FUNCTION
# =========================
def calculate_kpis(dataframe):
    """Calculate key performance indicators"""
    if dataframe.empty:
        return 0, 0, 0, 0
    
    total_sales = dataframe["Total (EGP)"].sum()
    total_orders = dataframe["Sale ID"].nunique()
    total_customers = dataframe["Customer"].nunique()
    avg_order = dataframe["Total (EGP)"].mean()

    return total_sales, total_orders, total_customers, avg_order

# =========================
# APP SETUP
# =========================
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server
app.title = "IT Sales Dashboard"

# =========================
# LAYOUT
# =========================
app.layout = dbc.Container([

    # ================= HEADER =================
    dbc.Row([
        dbc.Col([
            html.H1("IT Sales Dashboard", className="text-center mt-4 mb-4")
        ])
    ]),

    # ================= FILTERS =================
    dbc.Row([

        dbc.Col([
            html.Label("Select Year"),
            dcc.Dropdown(
                id="year-filter",
                options=[{"label": str(y), "value": y} for y in sorted(df["Year"].unique())],
                multi=True,
                value=sorted(df["Year"].unique())
            )
        ], md=4),

        dbc.Col([
            html.Label("Select Category"),
            dcc.Dropdown(
                id="category-filter",
                options=[{"label": c, "value": c} for c in sorted(df["Category"].unique())],
                multi=True,
                value=sorted(df["Category"].unique())
            )
        ], md=4),

        dbc.Col([
            html.Label("Select City"),
            dcc.Dropdown(
                id="city-filter",
                options=[{"label": c, "value": c} for c in sorted(df["City"].unique())],
                multi=True,
                value=sorted(df["City"].unique())
            )
        ], md=4),

    ], className="mb-4"),

    # ================= KPI CARDS =================
    dbc.Row([

        dbc.Col(dbc.Card(dbc.CardBody([
            html.H4("Total Revenue"),
            html.H2(id="total-sales")
        ]), color="primary", inverse=True), md=3),

        dbc.Col(dbc.Card(dbc.CardBody([
            html.H4("Orders"),
            html.H2(id="total-orders")
        ]), color="success", inverse=True), md=3),

        dbc.Col(dbc.Card(dbc.CardBody([
            html.H4("Customers"),
            html.H2(id="total-customers")
        ]), color="warning", inverse=True), md=3),

        dbc.Col(dbc.Card(dbc.CardBody([
            html.H4("Average Order"),
            html.H2(id="avg-order")
        ]), color="danger", inverse=True), md=3),

    ], className="mb-4"),

    # ================= AI INSIGHTS =================
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H3("AI Business Insights"),
                    html.Div(id="ai-insights", style={"whiteSpace": "pre-line"})
                ])
            ])
        ])
    ], className="mb-4"),

    # ================= CHARTS =================
    dbc.Row([
        dbc.Col(dcc.Graph(id="sales-trend"), md=6),
        dbc.Col(dcc.Graph(id="category-chart"), md=6),
    ]),

    dbc.Row([
        dbc.Col(dcc.Graph(id="city-chart"), md=6),
        dbc.Col(dcc.Graph(id="rep-chart"), md=6),
    ]),

    # ================= TABLE =================
    dbc.Row([
        dbc.Col([
            html.H3("Sales Transactions", className="mt-4"),
            dash_table.DataTable(
                id="sales-table",
                page_size=10,
                style_table={"overflowX": "auto"},
                style_cell={"textAlign": "left", "padding": "10px"},
                style_header={"fontWeight": "bold"}
            )
        ])
    ])

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
def update_dashboard(selected_years, selected_categories, selected_cities):
    """Update all dashboard components"""
    
    try:
        # ================= FILTER DATA =================
        filtered_df = df[
            (df["Year"].isin(selected_years)) &
            (df["Category"].isin(selected_categories)) &
            (df["City"].isin(selected_cities))
        ]

        # ================= KPIs =================
        total_sales, total_orders, total_customers, avg_order = calculate_kpis(filtered_df)

        # ================= INSIGHTS =================
        insights = generate_insights(filtered_df)

        # ================= TREND =================
        if not filtered_df.empty:
            monthly_sales = filtered_df.groupby("Month")["Total (EGP)"].sum().reset_index()
            sales_fig = px.line(monthly_sales, x="Month", y="Total (EGP)", title="Monthly Revenue Trend", markers=True)
        else:
            sales_fig = px.line(title="Monthly Revenue Trend")

        # ================= CATEGORY =================
        if not filtered_df.empty:
            category_sales = filtered_df.groupby("Category")["Total (EGP)"].sum().reset_index()
            category_fig = px.bar(category_sales, x="Category", y="Total (EGP)", title="Revenue by Category")
        else:
            category_fig = px.bar(title="Revenue by Category")

        # ================= CITY =================
        if not filtered_df.empty:
            city_sales = filtered_df.groupby("City")["Total (EGP)"].sum().reset_index()
            city_fig = px.pie(city_sales, names="City", values="Total (EGP)", title="Revenue by City")
        else:
            city_fig = px.pie(title="Revenue by City")

        # ================= REP =================
        if not filtered_df.empty:
            rep_sales = filtered_df.groupby("Rep")["Total (EGP)"].sum().reset_index()
            rep_fig = px.bar(rep_sales, x="Rep", y="Total (EGP)", title="Sales by Representative", color="Rep")
        else:
            rep_fig = px.bar(title="Sales by Representative")

        # ================= TABLE =================
        table_data = filtered_df.to_dict("records")
        table_columns = [{"name": col, "id": col} for col in filtered_df.columns]

        return (
            f"EGP {total_sales:,.0f}",
            f"{total_orders:,}",
            f"{total_customers:,}",
            f"EGP {avg_order:,.0f}",

            sales_fig,
            category_fig,
            city_fig,
            rep_fig,

            table_data,
            table_columns,

            insights
        )
        
    except Exception as e:
        logger.error(f"❌ Callback error: {e}")
        return ("Error", "Error", "Error", "Error", {}, {}, {}, {}, [], [], f"Error: {str(e)[:100]}")

# =========================
# RUN SERVER
# =========================
if __name__ == "__main__":
    PORT = int(os.environ.get("PORT", 8050))
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    
    logger.info(f"🚀 Starting dashboard on port {PORT}")
    app.run(host="0.0.0.0", port=PORT, debug=DEBUG)