# sesi6/app_dash_plotly.py — menampilkan chart Plotly di aplikasi Dash
from pathlib import Path
import pandas as pd
import plotly.express as px
from dash import Dash, html, dcc

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data" / "customer_churn.csv", parse_dates=["signup_date"])
df["status"] = df["churn"].map({0: "Bertahan", 1: "Churn"})
WARNA = {"Bertahan": "#4C72B0", "Churn": "#C44E52"}

hist = px.histogram(df, x="tenure_months", color="status", nbins=36, barmode="overlay",
                    opacity=0.7, color_discrete_map=WARNA, template="plotly_white",
                    labels={"tenure_months": "Tenure (bulan)", "status": "Status"},
                    title="Churn terkonsentrasi pada tenure pendek")

scatter = px.scatter(df, x="tenure_months", y="monthly_charges", color="status",
                     color_discrete_map=WARNA, opacity=0.55, template="plotly_white",
                     hover_name="customer_id", hover_data=["region", "contract_type", "num_complaints"],
                     labels={"tenure_months": "Tenure (bulan)", "monthly_charges": "Tagihan (Rp)"},
                     title="Klik legenda untuk menyembunyikan/menampilkan kelompok")

# Filter visual tanpa callback: tombol dropdown bawaan Plotly (updatemenus)
bar_df = df.groupby(["region", "contract_type"], as_index=False)["churn"].mean()
bar = px.bar(bar_df, x="region", y="churn", color="contract_type", barmode="group",
             template="plotly_white", labels={"churn": "Churn rate", "region": "Region",
                                              "contract_type": "Kontrak"},
             title="Churn rate per region dan kontrak")
bar.update_yaxes(tickformat=".0%")
bar.update_layout(updatemenus=[dict(type="buttons", direction="right", x=1, y=1.15, buttons=[
    dict(label="Grouped", method="relayout", args=[{"barmode": "group"}]),
    dict(label="Stacked", method="relayout", args=[{"barmode": "stack"}]),
])])

app = Dash(__name__, title="Plotly di Dash")
app.layout = html.Div(style={"maxWidth": "1150px", "margin": "auto", "fontFamily": "Arial"}, children=[
    html.H1("Interactive Visualization with Plotly"),
    html.P("Arahkan kursor (hover), zoom dengan drag, double-click untuk reset, klik legenda untuk filter."),
    html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "12px"}, children=[
        dcc.Graph(figure=hist), dcc.Graph(figure=scatter)]),
    dcc.Graph(figure=bar, config={"displaylogo": False,
                                  "toImageButtonOptions": {"format": "png", "scale": 2}}),
])

if __name__ == "__main__":
    app.run(debug=True, port=8051)
