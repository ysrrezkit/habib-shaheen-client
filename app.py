import pandas as pd
from dash import Dash, dcc, html, dash_table, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import os 

def calculate_kpis(dataframe):
    total_sales = dataframe["Total (EGP)"].sum()
    total_orders = dataframe["Sale ID"].nunique()
    total_customers = dataframe["Customer"].nunique()
    avg_order = dataframe["Total (EGP)"].mean()

    return total_sales, total_orders, total_customers, avg_order

df = pd.read_excel('sales_db.xlsx')

df["Date"] = pd.to_datetime(df["Date"])
df["Month"] = df["Date"].dt.strftime("%m")
df["Year"] = df["Date"].dt.year

app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP]
)

server = app.server
app.title = "IT Sales Dashboard"



import os


app.layout = dbc.Container([

    dbc.Row([
        dbc.Col([
            html.H1(
                "IT Sales Dashboard",
                className="text-center mt-4 mb-4"
            )
        ])
    ]),

    # =========================
    # Filters
    # =========================

    dbc.Row([

        dbc.Col([
            html.Label("Select Year"),
            dcc.Dropdown(
                id="year-filter",
                options=[
                    {"label": str(year), "value": year}
                    for year in sorted(df["Year"].unique())
                ],
                multi=True,
                value=sorted(df["Year"].unique())
            )
        ], md=4),

        dbc.Col([
            html.Label("Select Category"),
            dcc.Dropdown(
                id="category-filter",
                options=[
                    {"label": cat, "value": cat}
                    for cat in sorted(df["Category"].unique())
                ],
                multi=True,
                value=sorted(df["Category"].unique())
            )
        ], md=4),

        dbc.Col([
            html.Label("Select City"),
            dcc.Dropdown(
                id="city-filter",
                options=[
                    {"label": city, "value": city}
                    for city in sorted(df["City"].unique())
                ],
                multi=True,
                value=sorted(df["City"].unique())
            )
        ], md=4),

    ], className="mb-4"),

    # =========================
    # KPI Cards
    # =========================

    dbc.Row([

        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Total Revenue"),
                    html.H2(id="total-sales")
                ])
            ], color="primary", inverse=True)
        ], md=3),

        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Orders"),
                    html.H2(id="total-orders")
                ])
            ], color="success", inverse=True)
        ], md=3),

        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Customers"),
                    html.H2(id="total-customers")
                ])
            ], color="warning", inverse=True)
        ], md=3),

        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Average Order"),
                    html.H2(id="avg-order")
                ])
            ], color="danger", inverse=True)
        ], md=3),

    ], className="mb-4"),

    # =========================
    # Charts Row 1
    # =========================

    dbc.Row([

        dbc.Col([
            dcc.Graph(id="sales-trend")
        ], md=6),

        dbc.Col([
            dcc.Graph(id="category-chart")
        ], md=6),

    ]),

    # =========================
    # Charts Row 2
    # =========================

    dbc.Row([

        dbc.Col([
            dcc.Graph(id="city-chart")
        ], md=6),

        dbc.Col([
            dcc.Graph(id="rep-chart")
        ], md=6),

    ]),

    # =========================
    # Data Table
    # =========================

    dbc.Row([
        dbc.Col([
            html.H3("Sales Transactions", className="mt-4"),
            dash_table.DataTable(
                id="sales-table",
                page_size=10,
                style_table={"overflowX": "auto"},
                style_cell={
                    "textAlign": "left",
                    "padding": "10px"
                },
                style_header={
                    "fontWeight": "bold"
                }
            )
        ])
    ])

], fluid=True)

# =========================
# Callback
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
    ],
    [
        Input("year-filter", "value"),
        Input("category-filter", "value"),
        Input("city-filter", "value"),
    ]
)

def update_dashboard(selected_years, selected_categories, selected_cities):

    filtered_df = df[
        (df["Year"].isin(selected_years)) &
        (df["Category"].isin(selected_categories)) &
        (df["City"].isin(selected_cities))
    ]

    # =========================
    # KPIs
    # =========================

    total_sales, total_orders, total_customers, avg_order = calculate_kpis(filtered_df)

    # =========================
    # Sales Trend
    # =========================

    monthly_sales = (
        filtered_df
        .groupby("Month")["Total (EGP)"]
        .sum()
        .reset_index()
    )

    sales_fig = px.line(
        monthly_sales,
        x="Month",
        y="Total (EGP)",
        title="Monthly Revenue Trend",
        markers=True
    )

    # =========================
    # Category Chart
    # =========================

    category_sales = (
        filtered_df
        .groupby("Category")["Total (EGP)"]
        .sum()
        .reset_index()
    )

    category_fig = px.bar(
        category_sales,
        x="Category",
        y="Total (EGP)",
        title="Revenue by Category"
    )

    # =========================
    # City Chart
    # =========================

    city_sales = (
        filtered_df
        .groupby("City")["Total (EGP)"]
        .sum()
        .reset_index()
    )

    city_fig = px.pie(
        city_sales,
        names="City",
        values="Total (EGP)",
        title="Revenue by City"
    )

    # =========================
    # Sales Rep Chart
    # =========================

    rep_sales = (
        filtered_df
        .groupby("Rep")["Total (EGP)"]
        .sum()
        .reset_index()
    )

    rep_fig = px.bar(
        rep_sales,
        x="Rep",
        y="Total (EGP)",
        title="Sales by Representative",
        color="Rep"
    )

    # =========================
    # Table
    # =========================

    table_data = filtered_df.to_dict("records")
    table_columns = [
        {"name": col, "id": col}
        for col in filtered_df.columns
    ]

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
    )

# =========================
# Run App
# =========================

if __name__ == "__main__":

	PORT = int(os.environ.get("PORT", 8050))

	app.run(
   		host="0.0.0.0",
    	port=PORT
	)

