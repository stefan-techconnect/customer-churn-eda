# sesi6/app_dash_plotly.py — menampilkan chart Plotly di aplikasi Dash
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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
                     size="support_calls", size_max=28,
                     color_discrete_map=WARNA, opacity=0.55, template="plotly_white",
                     hover_name="customer_id",
                     hover_data=["region", "contract_type", "num_complaints", "support_calls"],
                     labels={"tenure_months": "Tenure (bulan)", "monthly_charges": "Tagihan (Rp)",
                             "support_calls": "Panggilan CS"},
                     title="Bubble chart: ukuran titik = jumlah panggilan ke CS")

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

pm = (df.groupby("payment_method", as_index=False)["churn"].mean()
        .sort_values("churn", ascending=True))
pay = px.bar(pm, x="churn", y="payment_method", orientation="h", template="plotly_white",
             color_discrete_sequence=["#C44E52"],
             labels={"churn": "Churn rate", "payment_method": "Metode pembayaran"},
             title="Churn rate per metode pembayaran")
pay.add_vline(x=df.churn.mean(), line_dash="dot", annotation_text="rata-rata")
pay.update_xaxes(tickformat=".0%")

cnt = (df.groupby(["region", "segment", "contract_type"]).size()
         .reset_index(name="pelanggan"))
sun = px.sunburst(cnt, path=["region", "segment", "contract_type"], values="pelanggan",
                  color_discrete_sequence=["#4C72B0", "#55A868", "#C44E52", "#8172B3", "#CCB974"],
                  title="Pelanggan: region → segmen → kontrak")

gauge = go.Figure(go.Indicator(
    mode="gauge+number",
    value=df.churn.mean() * 100,
    number={"suffix": "%"},
    title={"text": "Churn rate keseluruhan"},
    gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#C44E52"}},
))
gauge.update_layout(template="plotly_white", height=360)

app = Dash(__name__, title="Plotly di Dash")
app.layout = html.Div(style={"maxWidth": "1150px", "margin": "auto", "fontFamily": "Arial"}, children=[
    html.H1("Interactive Visualization with Plotly"),
    html.P("Arahkan kursor (hover), zoom dengan drag, double-click untuk reset, klik legenda untuk filter."),
    html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "12px"}, children=[
        dcc.Graph(figure=hist), dcc.Graph(figure=scatter)]),
    dcc.Graph(figure=bar, config={"displaylogo": False,
                                  "toImageButtonOptions": {"format": "png", "scale": 2}}),
    dcc.Graph(figure=pay),
    html.Div(style={"display": "grid", "gridTemplateColumns": "1.4fr 0.8fr", "gap": "12px"}, children=[
        html.Div([
            dcc.Graph(figure=sun),
            html.P("Cara membaca sunburst: luas setiap irisan sebanding dengan jumlah pelanggan. "
                   "Lingkar dalam adalah region, tengah segmen, dan luar jenis kontrak. "
                   "Klik irisan untuk memperbesar cabang tersebut.",
                   style={"color": "#555", "marginTop": 0}),
        ]),
        dcc.Graph(figure=gauge),
    ]),
])

if __name__ == "__main__":
    app.run(debug=True, port=8051)
