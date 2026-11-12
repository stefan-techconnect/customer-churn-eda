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

# 1. Membuat objek aplikasi
app = Dash(__name__, title="Dash Dasar")

# 2. Gaya sederhana (dict Python = CSS inline)
KARTU = {"flex": 1, "padding": "16px", "borderRadius": "10px",
         "background": "#F4F6FA", "textAlign": "center"}

def kartu(judul, nilai):
    return html.Div([html.P(judul, style={"margin": 0, "color": "#555"}),
                     html.H2(nilai, style={"margin": "4px 0 0 0"})], style=KARTU)

# 3. Layout = pohon komponen
app.layout = html.Div(style={"fontFamily": "Arial, sans-serif", "maxWidth": "1100px",
                             "margin": "auto", "padding": "20px"}, children=[
    html.H1("Dashboard Churn — Versi Dash"),
    html.P("Contoh layout statis: header, KPI cards, grafik, dan tabel."),

    html.Div(style={"display": "flex", "gap": "12px"}, children=[
        kartu("Total pelanggan", f"{len(df):,}"),
        kartu("Churn rate", f"{df.churn.mean():.1%}"),
        kartu("Rata-rata tenure", f"{df.tenure_months.mean():.1f} bln"),
        kartu("Region", df.region.nunique()),
    ]),

    html.H3("Churn rate per jenis kontrak"),
    dcc.Graph(figure={
        "data": [{"type": "bar", "x": ringkas["contract_type"], "y": ringkas["churn_rate"],
                  "marker": {"color": "#C44E52"}}],

        "layout": {"yaxis": {"title": "Churn rate (%)"}, "height": 350},
    }),

    html.H3("Ringkasan per kontrak"),
    dash_table.DataTable(data=ringkas.to_dict("records"),
                         columns=[{"name": c, "id": c} for c in ringkas.columns],
                         style_cell={"padding": "6px", "textAlign": "left"},
                         style_header={"fontWeight": "bold", "background": "#E9EDF5"}),

    html.Hr(),
    html.Small("Sumber: data latihan sintetis customer_churn.csv"),
])

# 4. Menjalankan server
if __name__ == "__main__":
    app.run(debug=True, port=8050)
