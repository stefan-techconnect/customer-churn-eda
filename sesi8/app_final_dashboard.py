# sesi8/app_final_dashboard.py — FINAL DASHBOARD PROJECT (Dash + Plotly + callback + ML inference)
import sys
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, html, dcc, dash_table, Input, Output, State, ALL, ctx, callback
from dash.exceptions import PreventUpdate

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from utils.model_utils import load_model, load_metadata, load_registry, model_path, validate_input, predict

# ============ 1. DATA + TIGA MODEL TERBAIK (ROC-AUC holdout) ============
BASE_META = dict(load_metadata())
REGISTRY = load_registry()
CHOICES = sorted(REGISTRY["models"], key=lambda m: m["holdout_metrics"]["roc_auc"], reverse=True)[:3]
BY_KEY = {m["key"]: m for m in CHOICES}
BEST = CHOICES[0]

def meta_for(entry):
    meta = dict(BASE_META)
    meta.update(model_name=entry["name"], algorithm=entry["algorithm"],
                holdout_metrics=entry["holdout_metrics"])
    return meta

raw = pd.read_csv(ROOT / "data" / "customer_churn.csv", parse_dates=["signup_date"])
raw["total_charges"] = raw["total_charges"].fillna(raw["monthly_charges"] * raw["tenure_months"])
FULL_CHURN = float(raw["churn"].mean())
MODELS, SCORED = {}, {}
for entry in CHOICES:
    MODELS[entry["key"]] = load_model(model_path(entry))
    scored = predict(MODELS[entry["key"]], raw, meta_for(entry))
    scored["status"] = scored["churn"].map({0: "Bertahan", 1: "Churn"})
    scored["periode"] = scored["signup_date"].dt.to_period("Q").astype(str)
    SCORED[entry["key"]] = scored
RISK_ORDER = ["Rendah", "Sedang", "Tinggi"]
RISK_COLOR = {"Rendah": "#55A868", "Sedang": "#DD8452", "Tinggi": "#C44E52"}
CATS = BASE_META["categories"]

# ============ 2. STYLE ============
C = {"muted": "#64748b", "accent": "#1d4ed8"}
PANEL = "spotlight overflow-hidden rounded-3xl border border-slate-200 bg-white shadow-sm"
def label(t):
    return html.Label(t, className="mb-1 block text-xs font-semibold uppercase tracking-wide text-slate-500")
def kpi(judul, nilai, sub=""):
    return html.Div(className="spotlight rounded-2xl border border-slate-200 bg-white px-4 py-3 shadow-sm", children=[
        html.Div(judul, className="text-xs uppercase tracking-wide text-slate-500"),
        html.Div(nilai, className="mt-1 text-2xl font-semibold text-slate-900"),
        html.Div(sub, className="mt-1 text-xs text-slate-500")])
def rupiah(x): return f"Rp{x:,.0f}".replace(",", ".")
def chart_card(graph_id):
    return html.Div(className=PANEL, children=dcc.Graph(
        id=graph_id, style={"height": "380px"}, config={"displayModeBar": False}))
def card_class(active):
    return "model-card spotlight" + (" is-active" if active else "")

# ============ 3. LAYOUT ============
sidebar = html.Aside(className="side-card spotlight w-full shrink-0 rounded-3xl border border-slate-200 bg-white p-4 shadow-sm lg:w-72", children=[
    html.H2("Filter", className="text-base font-semibold text-slate-900"),
    html.Div(className="mt-4", children=[
        label("Jenis kontrak"),
        dcc.Checklist(id="f-kontrak", options=CATS["contract_type"], value=CATS["contract_type"],
                      labelStyle={"display": "flex", "gap": "8px", "marginBottom": "6px"}),
    ]),
    html.Div(className="mt-4", children=[
        label("Kategori risiko"),
        dcc.Checklist(id="f-risk", options=RISK_ORDER, value=RISK_ORDER,
                      labelStyle={"display": "flex", "gap": "8px", "marginBottom": "6px"}),
    ]),
    html.Div(className="mt-4", children=[
        label("Tenure (bulan)"),
        dcc.RangeSlider(id="f-tenure", min=1, max=72, value=[1, 72], step=1,
                        marks={1: "1", 24: "24", 48: "48", 72: "72"}),
    ]),
    html.Div(className="dropdown-row mt-4 grid gap-3", children=[
        html.Div([label("Region"), dcc.Dropdown(id="f-region", options=CATS["region"], multi=True,
                                                placeholder="Semua")]),
        html.Div([label("Segmen"), dcc.Dropdown(id="f-segment", options=CATS["segment"], multi=True,
                                                placeholder="Semua")]),
    ]),
])

tab_overview = html.Div([
    html.Div(id="kpi-row", className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4"),
    html.Div(className="mt-4 grid gap-4 lg:grid-cols-2", children=[
        chart_card("g-donut"), chart_card("g-region-risk"),
        chart_card("g-trend"), chart_card("g-contract"),
        chart_card("g-risk"), chart_card("g-actual-pred")]),
    html.Div(className=f"{PANEL} mt-4 p-4", children=[
        html.H3(id="tbl-title", className="mb-3 text-base font-semibold text-slate-900"),
        dash_table.DataTable(
            id="tbl-risk", page_size=10, sort_action="native", filter_action="native",

            columns=[{"name": n, "id": i, **({"type": "numeric", "format": {"specifier": ".1%"}}
                                             if i == "churn_probability" else {})}
                     for n, i in [("ID", "customer_id"), ("Region", "region"), ("Segmen", "segment"),
                                  ("Kontrak", "contract_type"), ("Tenure", "tenure_months"),
                                  ("Keluhan", "num_complaints"), ("Prob. churn", "churn_probability"),
                                  ("Risiko", "risk_category")]],
            style_cell={"padding": "6px", "fontSize": "13px", "textAlign": "left"},
            style_header={"fontWeight": "bold", "background": "#E9EDF5"},
            style_data_conditional=[
                {"if": {"filter_query": "{risk_category} = 'Tinggi'", "column_id": "risk_category"},
                 "color": "white", "backgroundColor": RISK_COLOR["Tinggi"]},
                {"if": {"filter_query": "{risk_category} = 'Sedang'", "column_id": "risk_category"},
                 "backgroundColor": "#FBE3D3"}])]),
])

def num_input(id_, lab, val, mn, mx, step=1):
    return html.Div([label(lab), dcc.Input(id=id_, type="number", value=val, min=mn, max=mx,
                                           step=step, style={"width": "100%"})])
def dd_input(id_, lab, opts):
    return html.Div([label(lab), dcc.Dropdown(id=id_, options=opts, value=opts[0], clearable=False)])

tab_predict = html.Div(className="grid gap-4 lg:grid-cols-2", children=[
    html.Div(className=f"{PANEL} p-4", children=[
        html.H3("Simulasi prediksi pelanggan", className="mb-3 text-base font-semibold"),
        html.Div(className="grid gap-3 sm:grid-cols-2", children=[
            dd_input("p-region", "Region", CATS["region"]),
            dd_input("p-segment", "Segmen", CATS["segment"]),
            dd_input("p-contract", "Kontrak", CATS["contract_type"]),
            dd_input("p-payment", "Pembayaran", CATS["payment_method"]),
            dd_input("p-service", "Layanan", CATS["internet_service"]),
            num_input("p-tenure", "Tenure (bulan)", 6, 1, 72),
            num_input("p-monthly", "Tagihan bulanan (Rp)", 450000, 100000, 2000000, 10000),
            num_input("p-complaints", "Jumlah keluhan", 2, 0, 10),
            num_input("p-calls", "Panggilan CS", 3, 0, 15)]),
        html.Button("Jalankan prediksi", id="btn-predict", n_clicks=0,
                    className="mt-4 rounded-xl bg-blue-700 px-4 py-2 text-sm font-semibold text-white")]),
    html.Div(className=f"{PANEL} p-4", children=[html.H3("Hasil prediksi", className="mb-3 text-base font-semibold"),
                                   html.Div(id="pred-output",
                                            children=html.P("Isi form lalu klik tombol.")),
                                   dcc.Graph(id="g-gauge", style={"height": "260px"})]),
])

app = Dash(
    __name__,
    title="Final Dashboard — Customer Churn",
    external_scripts=["https://cdn.tailwindcss.com"],
    assets_folder=str(ROOT / "sesi8" / "assets"),
)
server = app.server                                  # untuk deployment (gunicorn)
app.layout = html.Div(className="relative min-h-screen overflow-hidden bg-white text-slate-900", children=[
    html.Div(className="aurora pointer-events-none absolute inset-0"),
    html.Div(className="relative mx-auto max-w-7xl px-4 py-8", children=[
    html.Div(className="mb-6", children=[
        html.P("Sesi 8", className="mb-2 text-xs font-semibold uppercase tracking-[0.28em] text-blue-700"),
        html.H1("Customer Churn Intelligence", className="gradient-text text-4xl font-semibold tracking-tight"),
        html.P(id="model-blurb",
               children=(f"Siapa yang berisiko berhenti, dan apa yang harus dilakukan? "
                         f"Prediksi {BEST['name']} · ROC-AUC {BEST['holdout_metrics']['roc_auc']:.3f}"),
               className="mt-2 max-w-3xl text-sm text-slate-600")]),
    html.Div(className="mb-6 grid gap-3 md:grid-cols-3", children=[
        html.Button(
            id={"type": "pick-model", "key": m["key"]},
            n_clicks=0,
            className=card_class(m["key"] == BEST["key"]),
            children=[
                html.Div(f"Peringkat {i}", className="text-[11px] font-semibold uppercase tracking-[0.16em] text-blue-700"),
                html.Div(m["name"], className="mt-1 text-base font-semibold text-slate-900"),
                html.Div(f"Holdout ROC-AUC {m['holdout_metrics']['roc_auc']:.3f}",
                         className="mt-1 text-sm text-slate-500"),
            ],
        ) for i, m in enumerate(CHOICES, start=1)
    ]),
    html.Div(id="model-note", className="mb-6 text-sm text-slate-500"),
    dcc.Store(id="f-model", data=BEST["key"]),
    html.Div(className="flex flex-col items-start gap-4 lg:flex-row", children=[
        sidebar,
        html.Div(className="min-w-0 flex-1", children=[
            dcc.Tabs(id="tabs", value="overview", className="churn-tabs", children=[
                dcc.Tab(label="Overview & Risiko", value="overview", children=[tab_overview]),
                dcc.Tab(label="Simulasi Prediksi", value="predict", children=[tab_predict])])])]),
    ]),
])

# ============ 4. CALLBACKS ============
def active_entry(model_key):
    return BY_KEY.get(model_key, BEST)

def model_copy(model_key):
    entry = active_entry(model_key)
    blurb = (f"Siapa yang berisiko berhenti, dan apa yang harus dilakukan? "
             f"Prediksi {entry['name']} · ROC-AUC {entry['holdout_metrics']['roc_auc']:.3f}")
    note = entry["description"]
    if entry["oversampling"]:
        note += " Dilatih dengan oversampling, jadi probabilitas cenderung lebih tinggi."
    return blurb, note

def apply_filters(model_key, region, segment, kontrak, risk, tenure):
    d = SCORED[active_entry(model_key)["key"]]
    if region: d = d[d.region.isin(region)]
    if segment: d = d[d.segment.isin(segment)]
    d = d[d.contract_type.isin(kontrak or []) & d.risk_category.isin(risk or [])]
    return d[d.tenure_months.between(*tenure)]

@callback(Output("f-model", "data"), Output({"type": "pick-model", "key": ALL}, "className"),
          Input({"type": "pick-model", "key": ALL}, "n_clicks"), State("f-model", "data"))
def choose_model(clicks, current):
    key = current or BEST["key"]
    if isinstance(ctx.triggered_id, dict) and any(clicks or []):
        key = ctx.triggered_id["key"]
    return key, [card_class(m["key"] == key) for m in CHOICES]

FILTERS = [Input("f-model", "data"), Input("f-region", "value"), Input("f-segment", "value"),
           Input("f-kontrak", "value"), Input("f-risk", "value"), Input("f-tenure", "value")]

@callback(Output("kpi-row", "children"), Output("g-donut", "figure"), Output("g-region-risk", "figure"),
          Output("g-trend", "figure"), Output("g-contract", "figure"),
          Output("g-risk", "figure"), Output("g-actual-pred", "figure"),
          Output("tbl-risk", "data"), Output("tbl-title", "children"),
          Output("model-blurb", "children"), Output("model-note", "children"), *FILTERS)
def update_overview(model_key, region, segment, kontrak, risk, tenure):
    blurb, note = model_copy(model_key)
    d = apply_filters(model_key, region, segment, kontrak, risk, tenure)
    if d.empty:
        f = px.scatter(title="Tidak ada data — longgarkan filter", template="plotly_white")
        return [kpi("Pelanggan", "0")], f, f, f, f, f, f, [], "Tidak ada data", blurb, note
    tinggi = (d.risk_category == "Tinggi").sum()
    kpis = [kpi("Pelanggan", f"{len(d):,}", "sesuai filter"),
            kpi("Churn rate aktual", f"{d.churn.mean():.1%}", f"total: {FULL_CHURN:.1%}"),
            kpi("Rata-rata prob. churn", f"{d.churn_probability.mean():.1%}", "prediksi model"),
            kpi("Risiko tinggi", f"{tinggi:,}", rupiah(d.loc[d.risk_category == 'Tinggi',
                                                             'monthly_charges'].sum()) + "/bln berisiko")]
    g0 = px.pie(d, names="risk_category", hole=0.55, color="risk_category",
                color_discrete_map=RISK_COLOR, category_orders={"risk_category": RISK_ORDER},
                title="Komposisi kategori risiko")
    g0.update_traces(textinfo="percent+label", sort=False)
    g0.add_annotation(text=f"<b>{len(d):,}</b><br>pelanggan", showarrow=False, font_size=14)
    g0.update_layout(showlegend=False)
    rr = (d.assign(tinggi=d.risk_category.eq("Tinggi")).groupby("region", as_index=False)["tinggi"]
            .mean().sort_values("tinggi"))
    g5 = px.bar(rr, x="tinggi", y="region", orientation="h", text_auto=".1%",
                template="plotly_white", color_discrete_sequence=[RISK_COLOR["Tinggi"]],
                title="Porsi pelanggan risiko tinggi per region",
                labels={"tinggi": "Porsi risiko tinggi", "region": ""})
    g5.update_xaxes(tickformat=".0%")
    tr = d.groupby("periode").agg(aktual=("churn", "mean"), prediksi=("churn_probability",
"mean")).reset_index()
    g1 = px.line(tr.melt("periode", var_name="seri", value_name="rate"), x="periode", y="rate",
                 color="seri", markers=True, template="plotly_white",
                 title="Churn aktual vs rata-rata prediksi per kuartal pendaftaran",
                 labels={"periode": "Kuartal pendaftaran", "rate": "Rate", "seri": ""})
    g1.update_yaxes(tickformat=".0%")
    ck = d.groupby(["contract_type", "risk_category"]).size().reset_index(name="n")
    g2 = px.bar(ck, x="contract_type", y="n", color="risk_category", color_discrete_map=RISK_COLOR,
                category_orders={"risk_category": RISK_ORDER,
                                 "contract_type": ["Bulanan", "Tahunan", "Dua Tahun"]},
                template="plotly_white", title="Komposisi risiko per jenis kontrak",
                labels={"contract_type": "", "n": "Pelanggan", "risk_category": "Risiko"})
    g3 = px.histogram(d, x="churn_probability", nbins=30, color="risk_category",
                      color_discrete_map=RISK_COLOR, category_orders={"risk_category": RISK_ORDER},
                      template="plotly_white", title="Distribusi probabilitas churn (prediksi)",
                      labels={"churn_probability": "Probabilitas churn", "risk_category": "Risiko"})
    g3.add_vline(x=BASE_META["decision_threshold"], line_dash="dash", annotation_text="threshold")
    rg = d.groupby("region").agg(aktual=("churn", "mean"), prediksi=("churn_probability", "mean"),
                                 n=("customer_id", "count")).reset_index()
    g4 = px.scatter(rg, x="aktual", y="prediksi", size="n", text="region", template="plotly_white",
                    title="Region: churn aktual vs prediksi (ukuran = jumlah pelanggan)",
                    labels={"aktual": "Churn rate aktual", "prediksi": "Rata-rata prediksi"})
    g4.update_traces(textposition="top center", cliponaxis=False)
    pad = 0.03
    g4.update_xaxes(tickformat=".0%", range=[rg.aktual.min() - pad, rg.aktual.max() + pad])
    g4.update_yaxes(tickformat=".0%", range=[rg.prediksi.min() - pad, rg.prediksi.max() + pad])
    top = d.sort_values("churn_probability", ascending=False).head(200)
    entry = active_entry(model_key)
    return (kpis, g0, g5, g1, g2, g3, g4, top.to_dict("records"),
            f"Prioritas retensi — {min(len(d), 200)} pelanggan · {entry['name']}",
            blurb, note)

@callback(Output("pred-output", "children"), Output("g-gauge", "figure"),
          Input("btn-predict", "n_clicks"), Input("f-model", "data"),
          State("p-region", "value"), State("p-segment", "value"), State("p-contract", "value"),
          State("p-payment", "value"), State("p-service", "value"), State("p-tenure", "value"),
          State("p-monthly", "value"), State("p-complaints", "value"), State("p-calls", "value"),
          prevent_initial_call=True)
def run_prediction(n, model_key, region, segment, contract, payment, service, tenure, monthly,
                   complaints, calls):
    if not n or None in (tenure, monthly, complaints, calls):
        raise PreventUpdate
    entry = active_entry(model_key)
    meta = meta_for(entry)
    row = pd.DataFrame([{"tenure_months": tenure, "monthly_charges": monthly,
                         "total_charges": monthly * tenure, "num_complaints": complaints,
                         "support_calls": calls, "region": region, "segment": segment,
                         "contract_type": contract, "payment_method": payment,
                         "internet_service": service}])
    errs = validate_input(row, meta)
    if errs:
        return html.Div([html.P(e, style={"color": "crimson"}) for e in errs]), px.scatter()
    scored = {m["key"]: predict(MODELS[m["key"]], row, meta_for(m)).iloc[0] for m in CHOICES}
    r = scored[entry["key"]]
    saran = {"Tinggi": "Prioritaskan retensi: hubungi dalam 48 jam, tawarkan kontrak tahunan.",
             "Sedang": "Pantau: kirim survei kepuasan dan penawaran loyalitas.",
             "Rendah": "Pertahankan layanan standar."}[r.risk_category]
    banding = html.Div(className="mt-4 grid grid-cols-3 gap-2", children=[
        html.Div(className="model-card" + (" is-active" if m["key"] == entry["key"] else ""), children=[
            html.Div(m["name"], className="text-xs text-slate-500"),
            html.Div(f"{scored[m['key']].churn_probability:.1%}", className="text-lg font-semibold",
                     style={"color": RISK_COLOR[scored[m["key"]].risk_category]}),
            html.Div(scored[m["key"]].risk_category, className="text-xs text-slate-500"),
        ]) for m in CHOICES])
    out = html.Div([
        html.Div(f"{r.churn_probability:.1%}", style={"fontSize": "40px", "fontWeight": 700,
                                                     "color": RISK_COLOR[r.risk_category]}),
        html.Div(f"{entry['name']} · Prediksi: {'Churn' if r.predicted_churn else 'Bertahan'} "
                 f"· Risiko {r.risk_category}"),
        html.P(saran, style={"color": C["muted"]}),
        banding])
    gauge = go.Figure(go.Indicator(mode="gauge+number", value=r.churn_probability * 100,
                                   number={"suffix": "%"}, gauge={
                                       "axis": {"range": [0, 100]},
                                       "bar": {"color": RISK_COLOR[r.risk_category]},
                                       "steps": [{"range": [0, 30], "color": "#E6F4EA"},
                                                 {"range": [30, 60], "color": "#FDF0E6"},
                                                 {"range": [60, 100], "color": "#FBE3E3"}]}))
    gauge.update_layout(margin=dict(t=20, b=10, l=20, r=20), height=240)
    return out, gauge

if __name__ == "__main__":
    app.run(debug=True, port=8053)
