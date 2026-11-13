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

def gelap(fig):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0",
        title_font_color="#f8fafc",
        legend_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.08)", zerolinecolor="rgba(255,255,255,0.12)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.08)", zerolinecolor="rgba(255,255,255,0.12)")
    return fig

for figur in (hist, scatter, bar, pay, sun, gauge):
    gelap(figur)

def panel(figure, config=None):
    return html.Div(
        className="spotlight overflow-hidden rounded-3xl border border-white/10 bg-white/5 shadow-2xl backdrop-blur-xl",
        children=dcc.Graph(figure=figure, config=config or {"displayModeBar": False}),
    )

app = Dash(
    __name__,
    title="Plotly di Dash",
    external_scripts=["https://cdn.tailwindcss.com"],
    assets_folder=str(Path(__file__).resolve().parent / "assets"),
)
app.layout = html.Div(className="relative min-h-screen overflow-hidden bg-slate-950 text-slate-100", children=[
    html.Div(className="aurora pointer-events-none absolute inset-0"),
    html.Div(className="relative mx-auto max-w-6xl px-6 py-10", children=[
        html.P("Sesi 6", className="mb-3 text-xs font-semibold uppercase tracking-[0.28em] text-cyan-300/80"),
        html.H1("Interactive Visualization", className="gradient-text text-4xl font-semibold tracking-tight sm:text-5xl"),
        html.P("Hover untuk spotlight, zoom dengan drag, double-click untuk reset, dan klik legenda untuk menyaring.",
               className="mt-3 max-w-2xl text-sm text-slate-300"),
        html.Div(className="mt-8 grid gap-4 md:grid-cols-2", children=[
            panel(hist),
            panel(scatter),
        ]),
        html.Div(className="mt-4", children=panel(
            bar,
            {"displaylogo": False, "toImageButtonOptions": {"format": "png", "scale": 2}},
        )),
        html.Div(className="mt-4", children=panel(pay)),
        html.Div(className="mt-4 grid items-start gap-4 lg:grid-cols-[1.4fr_0.8fr]", children=[
            html.Div(className="spotlight overflow-hidden rounded-3xl border border-white/10 bg-white/5 p-4 shadow-2xl backdrop-blur-xl", children=[
                dcc.Graph(figure=sun, config={"displayModeBar": False}),
                html.P("Cara membaca sunburst: luas setiap irisan sebanding dengan jumlah pelanggan. "
                       "Lingkar dalam adalah region, tengah segmen, dan luar jenis kontrak. "
                       "Klik irisan untuk memperbesar cabang tersebut.",
                       className="px-2 pb-2 text-sm leading-6 text-slate-300"),
            ]),
            panel(gauge),
        ]),
    ]),
])

if __name__ == "__main__":
    app.run(debug=True, port=8051)
