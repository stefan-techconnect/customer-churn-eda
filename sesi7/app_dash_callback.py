# sesi7/app_dash_callback.py — sistem callback Dash: filter -> KPI, chart, tabel
import os
from pathlib import Path
import pandas as pd
import plotly.express as px
from dash import Dash, html, dcc, dash_table, Input, Output, State, callback, ctx, no_update
from dash.exceptions import PreventUpdate

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data" / "customer_churn.csv", parse_dates=["signup_date"])
df["status"] = df["churn"].map({0: "Bertahan", 1: "Churn"})
WARNA = {"Bertahan": "#60a5fa", "Churn": "#fb7185"}
TABEL_COLS = ["customer_id", "region", "segment", "contract_type", "tenure_months",
              "monthly_charges", "num_complaints", "status"]
TABEL_LABEL = {
    "customer_id": "ID", "region": "Region", "segment": "Segmen",
    "contract_type": "Kontrak", "tenure_months": "Tenure",
    "monthly_charges": "Tagihan", "num_complaints": "Keluhan", "status": "Status",
}

app = Dash(
    __name__,
    title="Dash Callback",
    external_scripts=["https://cdn.tailwindcss.com"],
    assets_folder=str(Path(__file__).resolve().parent / "assets"),
)
app.layout = html.Div(className="relative min-h-screen overflow-hidden bg-slate-950 text-slate-100", children=[
    html.Div(className="aurora pointer-events-none absolute inset-0"),
    html.Div(className="relative mx-auto max-w-6xl px-4 py-8 sm:px-6", children=[
        html.P("Sesi 7", className="mb-2 text-xs font-semibold uppercase tracking-[0.28em] text-cyan-300/80"),
        html.H1("Dashboard Churn", className="gradient-text text-4xl font-semibold tracking-tight"),
        html.P("Filter di kartu atas mengubah KPI, grafik, dan tabel sekaligus.",
               className="mt-2 text-sm text-slate-300"),

        html.Div(className="filter-card spotlight mt-6 rounded-3xl border border-white/10 bg-white/5 p-4 shadow-2xl backdrop-blur-xl", children=[
            html.Div(className="mb-4", children=[
                html.Label("Jenis kontrak", className="mb-2 block text-xs uppercase tracking-wide text-slate-400"),
                dcc.RadioItems(
                    id="rd-kontrak", inline=True, value="Semua",
                    options=["Semua", "Bulanan", "Tahunan", "Dua Tahun"],
                    className="flex flex-wrap gap-x-5 gap-y-2 text-sm text-slate-100",
                    inputStyle={"marginRight": "8px", "accentColor": "#22d3ee"},
                    labelStyle={"display": "inline-flex", "alignItems": "center", "marginRight": "4px"},
                ),
            ]),
            html.Div(className="mb-4", children=[
                html.Label("Layanan internet", className="mb-2 block text-xs uppercase tracking-wide text-slate-400"),
                dcc.Checklist(
                    id="cl-service", options=sorted(df.internet_service.unique()),
                    value=[], inline=True,
                    className="flex flex-wrap gap-x-5 gap-y-2 text-sm text-slate-100",
                    inputStyle={"marginRight": "8px", "accentColor": "#22d3ee"},
                    labelStyle={"display": "inline-flex", "alignItems": "center", "marginRight": "4px"},
                ),
            ]),
            html.Div(className="mb-4", children=[
                html.Label("Tenure (bulan)", className="mb-1 block text-xs uppercase tracking-wide text-slate-400"),
                dcc.RangeSlider(id="rs-tenure", min=1, max=72, step=1, value=[1, 72],
                                marks={i: str(i) for i in (1, 12, 24, 48, 72)},
                                tooltip={"placement": "bottom", "always_visible": False}),
            ]),
            html.Div(className="mb-5 flex flex-wrap items-center gap-3", children=[
                html.Label("Minimal keluhan", className="text-sm text-slate-300"),
                dcc.Input(id="in-keluhan", type="number", min=0, max=10, value=0,
                          className="w-20 rounded-lg border border-white/10 bg-slate-900 px-2 py-1 text-white"),
                html.Button("Terapkan", id="btn-terapkan", n_clicks=0,
                            className="rounded-lg bg-cyan-400 px-3 py-1.5 text-sm font-semibold text-slate-950"),
                html.Button("Reset filter", id="btn-reset", n_clicks=0,
                            className="rounded-lg border border-white/15 px-3 py-1.5 text-sm text-slate-200"),
                html.Span(id="info-keluhan", className="text-sm text-slate-400"),
            ]),
            html.Div(className="dropdown-row grid gap-4 md:grid-cols-2", children=[
                html.Div([
                    html.Label("Region", className="mb-1 block text-xs uppercase tracking-wide text-slate-400"),
                    dcc.Dropdown(id="dd-region", options=sorted(df.region.unique()), multi=True,
                                 placeholder="Semua region", className="text-slate-900"),
                ]),
                html.Div([
                    html.Label("Segmen", className="mb-1 block text-xs uppercase tracking-wide text-slate-400"),
                    dcc.Dropdown(id="dd-segment", multi=True, placeholder="Semua segmen",
                                 className="text-slate-900"),
                ]),
            ]),
        ]),
        dcc.Store(id="store-keluhan", data=0),
        dcc.Store(id="store-filter"),

        html.Div(id="kpi", className="mt-4 grid gap-3 sm:grid-cols-2 xl:grid-cols-4"),

        html.Div(className="mt-4 grid gap-4 lg:grid-cols-2", children=[
            html.Div(className="spotlight min-h-[440px] overflow-hidden rounded-3xl border border-white/10 bg-white/5",
                     children=dcc.Graph(id="g-bar", style={"height": "440px"}, config={"displayModeBar": False})),
            html.Div(className="spotlight min-h-[440px] overflow-hidden rounded-3xl border border-white/10 bg-white/5",
                     children=dcc.Graph(id="g-scatter", style={"height": "440px"}, config={"displayModeBar": False})),
            html.Div(className="spotlight min-h-[440px] overflow-hidden rounded-3xl border border-white/10 bg-white/5",
                     children=dcc.Graph(id="g-hist", style={"height": "440px"}, config={"displayModeBar": False})),
            html.Div(id="detail", className="spotlight flex min-h-[440px] items-center rounded-3xl border border-white/10 bg-white/5 p-6 text-sm text-slate-300"),
        ]),
        html.P("Klik batang region, klik satu titik scatter, atau seleksi beberapa titik untuk memfilter tabel.",
               className="mt-3 text-sm text-slate-400"),
        html.H2(id="judul-tabel", className="mt-6 text-lg font-semibold"),
        html.Div(className="spotlight mt-3 overflow-x-auto rounded-3xl border border-white/10 bg-white/5 p-3", children=[
            dash_table.DataTable(
                id="tbl", page_size=10, sort_action="native",
                columns=[{"name": TABEL_LABEL[c], "id": c} for c in TABEL_COLS],
                style_table={"overflowX": "auto", "minWidth": "100%"},
                style_cell={"padding": "10px 12px", "textAlign": "left", "minWidth": "110px",
                            "fontFamily": "Segoe UI, sans-serif", "fontSize": "13px",
                            "backgroundColor": "transparent", "color": "#e2e8f0",
                            "border": "none", "borderBottom": "1px solid rgba(255,255,255,0.08)"},
                style_header={"fontWeight": "600", "backgroundColor": "rgba(15,23,42,0.9)",
                              "color": "#94a3b8", "border": "none"},
                style_data_conditional=[{"if": {"filter_query": "{status} = 'Churn'"},
                                         "backgroundColor": "rgba(251,113,133,0.16)"}],
            ),
        ]),
    ]),
])


def filter_df(region, segment, kontrak, tenure, min_keluhan, layanan=None):
    d = df
    if region:
        d = d[d.region.isin(region)]
    if segment:
        d = d[d.segment.isin(segment)]
    if layanan:
        d = d[d.internet_service.isin(layanan)]
    if kontrak and kontrak != "Semua":
        d = d[d.contract_type == kontrak]
    d = d[d.tenure_months.between(tenure[0], tenure[1])]
    return d[d.num_complaints >= (min_keluhan or 0)]


# 1) Chained callback: opsi segmen bergantung pada region terpilih
@callback(Output("dd-segment", "options"), Input("dd-region", "value"))
def update_segment_options(region):
    d = df[df.region.isin(region)] if region else df
    return sorted(d.segment.unique())


# 2) State + tombol: nilai keluhan baru dipakai setelah tombol diklik; ctx untuk tombol mana
@callback(Output("store-keluhan", "data"), Output("in-keluhan", "value"),
          Output("dd-region", "value"), Output("dd-segment", "value"),
          Output("rd-kontrak", "value"), Output("rs-tenure", "value"),
          Output("cl-service", "value"),
          Input("btn-terapkan", "n_clicks"), Input("btn-reset", "n_clicks"),
          State("in-keluhan", "value"), prevent_initial_call=True)
def terapkan_atau_reset(n_apply, n_reset, nilai):
    if ctx.triggered_id == "btn-reset":
        return 0, 0, None, None, "Semua", [1, 72], []
    if nilai is None:
        raise PreventUpdate                       # input kosong: jangan ubah apa pun
    return nilai, no_update, no_update, no_update, no_update, no_update, no_update


# 3) Callback utama: banyak input -> banyak output
@callback(Output("kpi", "children"), Output("g-bar", "figure"), Output("g-scatter", "figure"),
          Output("g-hist", "figure"), Output("info-keluhan", "children"),
          Output("store-filter", "data"),
          Input("dd-region", "value"), Input("dd-segment", "value"), Input("rd-kontrak", "value"),
          Input("rs-tenure", "value"), Input("store-keluhan", "data"), Input("cl-service", "value"))
def update_dashboard(region, segment, kontrak, tenure, min_keluhan, layanan):
    d = filter_df(region, segment, kontrak, tenure, min_keluhan, layanan)
    info = f"Filter keluhan aktif: ≥ {min_keluhan}"
    if d.empty:
        kosong = px.scatter(title="Tidak ada data untuk filter ini")
        return [html.Div("Data kosong", className="text-slate-300")], kosong, kosong, kosong, info, []

    def kartu(j, v):
        return html.Div(className="spotlight rounded-2xl border border-white/10 bg-white/5 px-4 py-3", children=[
            html.P(j, className="text-xs uppercase tracking-wide text-slate-400"),
            html.P(v, className="mt-1 text-2xl font-semibold text-white"),
        ])
    kpi = [kartu("Pelanggan", f"{len(d):,}"), kartu("Churn rate", f"{d.churn.mean():.1%}"),
           kartu("Rata-rata tagihan", f"Rp{d.monthly_charges.mean():,.0f}".replace(",", ".")),

           kartu("Rata-rata keluhan", f"{d.num_complaints.mean():.2f}")]

    reg = d.groupby("region", as_index=False)["churn"].mean().sort_values("churn", ascending=False)
    bar = px.bar(reg, x="region", y="churn", template="plotly_white", title="Churn rate per region",
                 labels={"churn": "Churn rate", "region": ""}, color_discrete_sequence=["#C44E52"])
    bar.update_yaxes(tickformat=".0%")
    bar.update_layout(height=400, margin=dict(l=48, r=16, t=56, b=72))
    sc = px.scatter(d, x="tenure_months", y="monthly_charges", color="status", opacity=0.6,
                    color_discrete_map=WARNA, template="plotly_white", hover_name="customer_id",
                    custom_data=["customer_id"],
                    title="Tenure vs tagihan", labels={"tenure_months": "Tenure (bulan)",
                                                       "monthly_charges": "Tagihan (Rp)"})
    sc.update_layout(height=400, margin=dict(l=56, r=16, t=56, b=56))
    hist = px.histogram(d, x="tenure_months", color="status", nbins=30, barmode="overlay",
                        opacity=0.7, color_discrete_map=WARNA, template="plotly_white",
                        title="Distribusi tenure (ikut filter)",
                        labels={"tenure_months": "Tenure (bulan)", "status": "Status"})
    hist.update_layout(height=400, margin=dict(l=48, r=16, t=56, b=64), bargap=0.05)
    for figur in (bar, sc, hist):
        figur.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                            font_color="#e2e8f0", title_font_color="#f8fafc",
                            legend_bgcolor="rgba(0,0,0,0)")
        figur.update_xaxes(gridcolor="rgba(255,255,255,0.08)")
        figur.update_yaxes(gridcolor="rgba(255,255,255,0.08)")
    return kpi, bar, sc, hist, info, d[TABEL_COLS].to_dict("records")


# 4) Interaksi antarkomponen: klik batang chart -> filter tabel
@callback(Output("tbl", "data"), Output("judul-tabel", "children"),
          Input("g-bar", "clickData"), Input("g-scatter", "selectedData"),
          Input("store-filter", "data"))
def update_tabel(click, selected, records):
    if not records:
        return [], "Tidak ada data"
    d = pd.DataFrame(records)
    judul = "Detail pelanggan (hasil filter)"
    if ctx.triggered_id == "g-bar" and click:
        reg = click["points"][0]["x"]
        d = d[d.region == reg]
        judul = f"Detail pelanggan — region {reg}"
    elif ctx.triggered_id == "g-scatter" and selected and selected.get("points"):
        ids = [p["customdata"][0] for p in selected["points"]]
        d = d[d.customer_id.isin(ids)]
        judul = f"Detail pelanggan — {len(ids)} titik terpilih"
    return d[TABEL_COLS].to_dict("records"), judul


@callback(Output("detail", "children"), Input("g-scatter", "clickData"))
def tampil_detail(click):
    if not click:
        raise PreventUpdate
    cid = click["points"][0]["customdata"][0]
    r = df.loc[df.customer_id == cid].iloc[0]
    return html.Ul(className="space-y-2 text-base text-slate-100", children=[
        html.Li(f"ID: {r.customer_id}"),
        html.Li(f"Region: {r.region}"),
        html.Li(f"Kontrak: {r.contract_type}"),
        html.Li(f"Keluhan: {r.num_complaints}"),
    ])


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8052"))
    app.run(host="0.0.0.0", port=port, debug=port == 8052)
