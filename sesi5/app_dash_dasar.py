# sesi5/app_dash_dasar.py — struktur dasar aplikasi Dash (tanpa callback)
from pathlib import Path
import pandas as pd
from dash import Dash, html, dcc, dash_table

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data" / "customer_churn.csv", parse_dates=["signup_date"])

ringkas = (df.groupby("contract_type")
             .agg(pelanggan=("customer_id", "count"), churn_rate=("churn", "mean"))
             .reset_index())
ringkas["churn_rate"] = (ringkas["churn_rate"] * 100).round(1)
region = (df.groupby("region")
            .agg(churn_rate=("churn", "mean"))
            .reset_index()
            .sort_values("churn_rate", ascending=False))
region["churn_rate"] = (region["churn_rate"] * 100).round(1)

# Bootstrap 5 mengatur grid, kartu, dan tipografi. sesi5/assets/style.css menyesuaikan warna dashboard.
BOOTSTRAP = "https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css"
app = Dash(
    __name__,
    title="Dash Dasar",
    external_stylesheets=[BOOTSTRAP],
    assets_folder=str(Path(__file__).resolve().parent / "assets"),
)

def gaya_grafik(judul):
    return {
        "title": {"text": judul, "font": {"size": 16, "color": "#1f2a44"}},
        "yaxis": {"title": "Churn rate (%)", "gridcolor": "#eef2f7"},
        "xaxis": {"title": ""},
        "height": 340,
        "margin": {"l": 48, "r": 16, "t": 48, "b": 40},
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"family": "Segoe UI, sans-serif", "color": "#526077"},
    }

def kartu(judul, nilai, aksen):
    return html.Div(className="col-12 col-sm-6 col-xl", children=[
        html.Div(className="card kartu h-100 border-0 shadow-sm", children=[
            html.Div(className="card-body", children=[
                html.Div(className="kartu-aksen", style={"background": aksen}),
                html.P(judul, className="text-secondary small mb-1"),
                html.H2(nilai, className="mb-0 fw-bold"),
            ]),
        ]),
    ])

fig_kontrak = {
    "data": [{"type": "bar", "x": ringkas["contract_type"], "y": ringkas["churn_rate"],
              "marker": {"color": "#e34948", "cornerradius": 6}}],
    "layout": gaya_grafik("Churn rate per jenis kontrak"),
}
fig_region = {
    "data": [{"type": "bar", "x": region["region"], "y": region["churn_rate"],
              "marker": {"color": "#2f5da8", "cornerradius": 6}}],
    "layout": gaya_grafik("Churn rate per region"),
}

# 3. Layout = pohon komponen
app.layout = html.Div(className="halaman", children=[
    html.Div(className="container py-4 py-lg-5", children=[
        html.Div(className="mb-4", children=[
            html.P("Customer churn", className="text-uppercase small text-secondary fw-semibold mb-1"),
            html.H1("Dashboard Churn", className="judul-halaman fw-bold mb-2"),
            html.P("Header, KPI, dua grafik, dan tabel ringkasan dari data latihan.",
                   className="text-secondary mb-0"),
        ]),

        html.Div(className="row g-3 mb-4", children=[
            kartu("Total pelanggan", f"{len(df):,}", "#2f5da8"),
            kartu("Churn rate", f"{df.churn.mean():.1%}", "#e34948"),
            kartu("Rata-rata tenure", f"{df.tenure_months.mean():.1f} bln", "#1baf7a"),
            kartu("Region", df.region.nunique(), "#eda100"),
            kartu("Rata-rata keluhan", f"{df.num_complaints.mean():.2f}", "#4a3aa7"),
        ]),

        html.Div(className="row g-3 mb-4", children=[
            html.Div(className="col-lg-6", children=[
                html.Div(className="card panel border-0 shadow-sm h-100", children=[
                    dcc.Graph(figure=fig_kontrak, config={"displayModeBar": False}),
                ]),
            ]),
            html.Div(className="col-lg-6", children=[
                html.Div(className="card panel border-0 shadow-sm h-100", children=[
                    dcc.Graph(figure=fig_region, config={"displayModeBar": False}),
                ]),
            ]),
        ]),

        html.Div(className="card panel border-0 shadow-sm", children=[
            html.Div(className="card-body", children=[
                html.H2("Ringkasan per kontrak", className="h5 fw-bold mb-3"),
                dash_table.DataTable(
                    data=ringkas.to_dict("records"),
                    columns=[{"name": c, "id": c} for c in ringkas.columns],
                    sort_action="native", page_size=5,
                    style_table={"overflowX": "auto"},
                    style_cell={"padding": "12px 14px", "textAlign": "left",
                                "fontFamily": "Segoe UI, sans-serif", "fontSize": "14px",
                                "border": "none", "borderBottom": "1px solid #eef2f7"},
                    style_header={"fontWeight": "600", "background": "#f8fafc",
                                  "color": "#526077", "border": "none"},
                    style_data={"color": "#1f2a44"},
                ),
            ]),
        ]),

        html.P("Sumber: data latihan sintetis customer_churn.csv",
               className="text-secondary small mt-3 mb-0"),
    ]),
])

# 4. Menjalankan server
if __name__ == "__main__":
    app.run(debug=True, port=8050)
