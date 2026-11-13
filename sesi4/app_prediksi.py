# sesi4/app_prediksi.py — Integrasi final trained model ke dashboard Streamlit
import sys
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))                      # agar folder utils bisa di-import
from utils.model_utils import (load_model, load_metadata, load_registry, load_holdout,
                               model_path, validate_input, predict)

st.set_page_config(page_title="EDA & Prediksi Churn", page_icon="📊", layout="wide")

st.markdown(
    """
    <style>
    [class*="st-key-nav-"] button {
        width: 100%;
        justify-content: flex-start !important;
        text-align: left !important;
        background: transparent !important;
        color: inherit !important;
        border: 1px solid transparent !important;
        border-left: 3px solid transparent !important;
        border-radius: 10px !important;
        box-shadow: none !important;
        font-weight: 400 !important;
        padding: 0.4rem 0.65rem 0.4rem 0.9rem !important;
    }
    [class*="st-key-nav-"]:hover button {
        background: rgba(42, 120, 214, 0.12) !important;
    }
    [class*="st-key-nav-on-"] button {
        background: rgba(42, 120, 214, 0.22) !important;
        border-left-color: #2a78d6 !important;
        font-weight: 600 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

STATUS = {"Bertahan": "#2a78d6", "Churn": "#eb6834"}
KATEGORI = ["#2a78d6", "#1baf7a", "#eda100", "#4a3aa7", "#e87ba4"]
KONTRAK = ["Bulanan", "Tahunan", "Dua Tahun"]
EDA_BAGIAN = ["🥧 Komposisi", "📊 Perbandingan", "📈 Distribusi", "⏱️ Tren", "🔗 Hubungan"]
PRED_BANDING = "🏆 Perbandingan model"
PRED_SATU = "👤 Prediksi satu pelanggan"
PRED_BATCH = "📂 Prediksi batch (CSV)"
DATA = "📋 Data"
PRED_BAGIAN = [PRED_BANDING, PRED_SATU, PRED_BATCH, DATA]
METRIK = {"accuracy": "Accuracy", "precision": "Precision", "recall": "Recall",
          "f1": "F1", "roc_auc": "ROC-AUC"}

# 1) LOAD MODEL — tiap file model dimuat sekali saja, disimpan di cache (kunci = path)
@st.cache_resource
def get_model(path):
    return load_model(path)

@st.cache_data
def get_meta():
    return load_metadata()

@st.cache_data
def get_registry():
    return load_registry()

@st.cache_data
def get_holdout():
    return load_holdout()

def metrik_holdout(y_true, proba, threshold):
    """Metrik pada data uji untuk threshold tertentu (ROC-AUC tidak bergantung threshold)."""
    pred = (proba >= threshold).astype(int)
    return {"accuracy": accuracy_score(y_true, pred),
            "precision": precision_score(y_true, pred, zero_division=0),
            "recall": recall_score(y_true, pred), "f1": f1_score(y_true, pred),
            "roc_auc": roc_auc_score(y_true, proba)}

@st.cache_data
def load_eda():
    df = pd.read_csv(ROOT / "data" / "customer_churn.csv", parse_dates=["signup_date"])
    df["total_charges_missing"] = df["total_charges"].isna()
    df["total_charges"] = df["total_charges"].fillna(df["total_charges"].median())
    df["status"] = df["churn"].map({0: "Bertahan", 1: "Churn"})
    df["kelompok_tenure"] = pd.cut(
        df["tenure_months"], [0, 12, 24, 48, 72],
        labels=["0-12", "13-24", "25-48", "49-72"],
    ).astype(str)
    return df

def pilih_navigasi(label):
    aktif = st.session_state.bagian == label
    slug = "".join(ch.lower() if ch.isalnum() else "-" for ch in label)
    slug = "-".join(part for part in slug.split("-") if part)
    if st.sidebar.button(label, key=f"nav-{'on' if aktif else 'off'}-{slug}", width="stretch"):
        st.session_state.bagian = label
        st.rerun()

def gaya(fig, tinggi=360):
    fig.update_layout(
        template="plotly_white", height=tinggi, margin=dict(l=10, r=10, t=50, b=10),
        title_font_size=15, legend_title_text="",
    )
    return fig

registry = get_registry()
MODELS = {m["key"]: m for m in registry["models"]}
model_key = st.sidebar.selectbox(
    "Model prediksi", list(MODELS),
    format_func=lambda k: f"{MODELS[k]['name']} · AUC {MODELS[k]['holdout_metrics']['roc_auc']:.3f}",
    help="Semua model memakai fitur & preprocessing yang sama; pilih untuk membandingkan hasil.",
)
aktif = MODELS[model_key]
model = get_model(model_path(aktif))
base_meta = get_meta()
threshold = st.sidebar.slider(
    "Threshold keputusan", 0.3, 0.7, 0.5, 0.05,
    help="Pelanggan diprediksi churn jika probabilitas ≥ threshold.",
)
meta = dict(base_meta)
meta["decision_threshold"] = float(threshold)
meta.update(model_name=f"Customer Churn Classifier — {aktif['name']}",
            algorithm=aktif["algorithm"], holdout_metrics=aktif["holdout_metrics"])
st.sidebar.caption("Mengubah threshold mengubah jumlah Prediksi churn.")
st.sidebar.divider()
eda = load_eda()
if "bagian" not in st.session_state:
    st.session_state.bagian = EDA_BAGIAN[0]
st.sidebar.markdown("**EDA**")
for item in EDA_BAGIAN:
    pilih_navigasi(item)
st.sidebar.markdown("**Prediction**")
for item in PRED_BAGIAN:
    pilih_navigasi(item)
bagian = st.session_state.bagian
st.sidebar.caption(f"{len(eda):,} pelanggan · churn {eda.churn.mean():.1%}")

st.title("📊 EDA & Prediksi Risiko Churn")
st.caption("Eksplorasi customer_churn.csv, lalu inference model final pada pelanggan baru.")
with st.expander(f"ℹ️ Model aktif: {aktif['name']}", expanded=False):
    st.write(f"**{meta['model_name']}** · versi {meta['model_version']} · {meta['algorithm']}")
    st.caption(aktif["description"])
    if aktif["oversampling"]:
        st.info("Model ini dilatih dengan **oversampling** (RandomOverSampler): kelas churn diseimbangkan "
                "pada data latih. Recall lebih tinggi, tetapi probabilitas cenderung lebih tinggi "
                "dibanding model tanpa oversampling.")
    kolom_metrik = st.columns(len(METRIK) + 1)
    for kol, (k, label) in zip(kolom_metrik, METRIK.items()):
        kol.metric(label, f"{aktif['holdout_metrics'][k]:.3f}")
    kolom_metrik[-1].metric("CV ROC-AUC", f"{aktif['cv_roc_auc']['mean']:.3f}",
                            help=f"5-fold, ± {aktif['cv_roc_auc']['std']:.3f}")
    st.caption(f"Metrik hasil training pada data uji {registry['split']['n_test']} pelanggan "
               f"(threshold {registry['metrics_threshold']}).")
    st.write("Fitur input:", ", ".join(meta["feature_order"]))
    st.write(f"Threshold keputusan: {meta['decision_threshold']} · "
             f"Risk bins: {meta['risk_bins']}")
    st.caption("Model tidak dilatih ulang di aplikasi ini; aplikasi hanya melakukan inference. "
               "Untuk melatih ulang: `uv run python models/train_models.py`.")

COLOR = {"Rendah": "🟢", "Sedang": "🟡", "Tinggi": "🔴"}

if bagian in EDA_BAGIAN or bagian == DATA:
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total pelanggan", f"{len(eda):,}")
    k2.metric("Churn rate", f"{eda.churn.mean():.1%}")
    k3.metric("Rata-rata tagihan", f"Rp{eda.monthly_charges.mean():,.0f}".replace(",", "."))
    k4.metric("Rata-rata tenure", f"{eda.tenure_months.mean():.1f} bln")
    k5.metric("Missing total_charges", f"{int(eda.total_charges_missing.sum())} baris", help="Diisi median")

    if bagian == "🥧 Komposisi":
        c1, c2 = st.columns(2)
        pie = px.pie(eda, names="status", color="status", color_discrete_map=STATUS,
                     title="Satu dari tiga pelanggan churn")
        pie.update_traces(textinfo="percent+label", textfont_size=14)
        c1.plotly_chart(gaya(pie), width="stretch")
        donut = px.pie(eda, names="contract_type", hole=0.55, color_discrete_sequence=KATEGORI,
                       category_orders={"contract_type": KONTRAK},
                       title="Separuh pelanggan berkontrak bulanan")
        donut.update_traces(textinfo="percent+label")
        donut.add_annotation(text=f"<b>{len(eda):,}</b><br>pelanggan", showarrow=False, font_size=16)
        c2.plotly_chart(gaya(donut), width="stretch")
        tm = eda.groupby(["region", "segment"], as_index=False).agg(
            pelanggan=("customer_id", "count"), churn_rate=("churn", "mean"))
        tree = px.treemap(tm, path=["region", "segment"], values="pelanggan", color="churn_rate",
                          color_continuous_scale="Oranges",
                          title="Treemap: luas = jumlah pelanggan, warna = churn rate")
        st.plotly_chart(gaya(tree, 420), width="stretch")
    elif bagian == "📊 Perbandingan":
        kolom = st.selectbox("Bandingkan churn rate berdasarkan",
                             ["contract_type", "region", "segment", "payment_method", "internet_service"])
        rk = (eda.groupby(kolom, as_index=False)
              .agg(pelanggan=("customer_id", "count"), churn_rate=("churn", "mean"))
              .sort_values("churn_rate"))
        bar = px.bar(rk, x="churn_rate", y=kolom, orientation="h", text_auto=".1%",
                     color_discrete_sequence=["#eb6834"], title=f"Churn rate per {kolom}")
        bar.add_vline(x=eda.churn.mean(), line_dash="dot", annotation_text="rata-rata")
        bar.update_xaxes(tickformat=".0%")
        c1, c2 = st.columns([3, 2])
        c1.plotly_chart(gaya(bar), width="stretch")
        c2.dataframe(rk.sort_values("churn_rate", ascending=False)
                     .style.format({"churn_rate": "{:.1%}"})
                     .background_gradient(subset=["churn_rate"], cmap="Oranges"),
                     width="stretch", hide_index=True)
        c3, c4 = st.columns(2)
        gb = eda.groupby(["contract_type", "segment"], as_index=False)["churn"].mean()
        grouped = px.bar(gb, x="contract_type", y="churn", color="segment", barmode="group",
                         color_discrete_sequence=KATEGORI, category_orders={"contract_type": KONTRAK},
                         title="Grouped bar: kontrak × segmen")
        grouped.update_yaxes(tickformat=".0%")
        c3.plotly_chart(gaya(grouped), width="stretch")
        stacked = px.histogram(eda, y="payment_method", color="status", barnorm="percent",
                               color_discrete_map=STATUS, title="Stacked bar 100%: status per pembayaran")
        stacked.update_layout(xaxis_title="Proporsi (%)", yaxis_title="")
        c4.plotly_chart(gaya(stacked), width="stretch")
    elif bagian == "📈 Distribusi":
        num = st.selectbox("Variabel numerik",
                           ["tenure_months", "monthly_charges", "num_complaints", "support_calls"])
        c1, c2 = st.columns(2)
        hist = px.histogram(eda, x=num, color="status", barmode="overlay", opacity=0.7, nbins=30,
                            color_discrete_map=STATUS, marginal="box", title=f"Histogram {num}")
        c1.plotly_chart(gaya(hist, 400), width="stretch")
        vio = px.violin(eda, x="contract_type", y=num, color="contract_type", box=True,
                        color_discrete_sequence=KATEGORI, category_orders={"contract_type": KONTRAK},
                        title=f"Violin {num} per kontrak")
        c2.plotly_chart(gaya(vio, 400), width="stretch")
    elif bagian == "⏱️ Tren":
        bulanan = (eda.set_index("signup_date").resample("ME")["customer_id"].count()
                   .rename("pelanggan").reset_index())
        bulanan["rata2_3bln"] = bulanan["pelanggan"].rolling(3).mean()
        line = px.line(bulanan, x="signup_date", y=["pelanggan", "rata2_3bln"], markers=True,
                       color_discrete_sequence=["#9dbbe6", "#2a78d6"],
                       title="Pelanggan baru per bulan dan rata-rata bergerak 3 bulan")
        st.plotly_chart(gaya(line), width="stretch")
        c1, c2 = st.columns(2)
        kt = eda.groupby("kelompok_tenure", as_index=False)["churn"].mean()
        lt = px.line(kt, x="kelompok_tenure", y="churn", markers=True,
                     text=kt["churn"].map("{:.1%}".format), color_discrete_sequence=["#eb6834"],
                     title="Churn rate per kelompok tenure")
        lt.update_traces(textposition="top center", marker_size=10)
        lt.update_yaxes(tickformat=".0%", range=[0, 0.65])
        c1.plotly_chart(gaya(lt), width="stretch")
        q = (eda.assign(kuartal=eda.signup_date.dt.to_period("Q").astype(str))
             .groupby(["kuartal", "segment"], as_index=False)["customer_id"].count())
        area = px.area(q, x="kuartal", y="customer_id", color="segment",
                       color_discrete_sequence=KATEGORI, title="Area: pelanggan baru per kuartal")
        c2.plotly_chart(gaya(area), width="stretch")
    elif bagian == "🔗 Hubungan":
        c1, c2 = st.columns(2)
        sc = px.scatter(eda, x="tenure_months", y="monthly_charges", color="status", opacity=0.55,
                        color_discrete_map=STATUS, hover_name="customer_id",
                        hover_data=["contract_type", "num_complaints"], title="Tenure vs tagihan")
        c1.plotly_chart(gaya(sc, 420), width="stretch")
        corr = eda[["tenure_months", "monthly_charges", "num_complaints", "support_calls", "churn"]].corr()
        hm = px.imshow(corr.round(2), text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                       title="Heatmap korelasi")
        c2.plotly_chart(gaya(hm, 420), width="stretch")
        pv = pd.pivot_table(eda, values="churn", index="region", columns="segment", aggfunc="mean")
        hm2 = px.imshow(pv * 100, text_auto=".1f", color_continuous_scale="Oranges",
                        labels={"color": "Churn (%)"}, title="Heatmap churn rate region × segmen (%)")
        st.plotly_chart(gaya(hm2, 360), width="stretch")
    elif bagian == "📋 Data":
        profil = pd.DataFrame({
            "tipe": eda.dtypes.astype(str),
            "missing": eda.isna().sum().where(eda.columns != "total_charges", eda.total_charges_missing.sum()),
            "unik": eda.nunique(),
        }).reset_index(names="kolom")
        c1, c2 = st.columns([2, 3])
        c1.markdown("**Profil kolom**")
        c1.dataframe(profil, width="stretch", hide_index=True)
        c2.markdown("**Statistik deskriptif**")
        c2.dataframe(eda.describe(include="number").T.round(1), width="stretch")
        st.markdown("**10 baris pertama**")
        st.dataframe(eda.head(10), width="stretch", hide_index=True)

# PERBANDINGAN MODEL — metrik hasil training pada data uji (holdout) yang sama
elif bagian == PRED_BANDING:
    holdout = get_holdout()
    y_uji = holdout["churn"]
    st.caption(f"Semua model dinilai pada data uji yang sama: {registry['split']['n_test']} pelanggan "
               f"(20% data, stratified, random_state={registry['split']['random_state']}). "
               f"Metrik di bawah dihitung ulang pada threshold sidebar = **{threshold:.2f}**.")

    # Ketimpangan kelas: data latih sebelum vs sesudah oversampling
    jumlah = registry["train_class_counts"]
    kelas = pd.DataFrame([{"data latih": tahap.capitalize(), "status": s, "pelanggan": n}
                          for tahap, isi in jumlah.items() for s, n in isi.items()])
    c1, c2 = st.columns([2, 3])
    dist = px.bar(kelas, x="data latih", y="pelanggan", color="status", barmode="group",
                  text_auto=True, color_discrete_map=STATUS,
                  title="Data latih sebelum & sesudah oversampling")
    dist.update_layout(xaxis_title="")
    c1.plotly_chart(gaya(dist, 330), width="stretch")
    with c2:
        st.markdown("**Stratified sampling vs oversampling**")
        st.markdown(
            f"- **Stratified sampling** — split train/test dan 5 fold CV menjaga rasio churn "
            f"~{y_uji.mean():.0%} di setiap bagian, sehingga evaluasi adil. Rasio kelas tidak berubah.\n"
            f"- **Oversampling (RandomOverSampler)** — baris churn di data latih diduplikasi acak "
            f"({jumlah['sebelum']['Churn']:,} → {jumlah['sesudah']['Churn']:,}) agar model belajar "
            f"seimbang dari kedua kelas. Hanya data latih; data uji tetap data asli.\n"
            f"- Efeknya: **recall & F1 naik**, precision turun, ROC-AUC hampir sama. Probabilitas "
            f"model oversampling cenderung lebih tinggi, jadi lebih banyak pelanggan masuk risiko Tinggi.")

    rows = []
    for k, m in MODELS.items():
        r = metrik_holdout(y_uji, holdout[k], threshold)
        r.update(model=m["name"], algoritma=m["family"],
                 sampling="Oversampling" if m["oversampling"] else "Tanpa oversampling",
                 cv_recall=m["cv_metrics"]["recall"], cv_f1=m["cv_metrics"]["f1"],
                 cv_roc_auc=m["cv_metrics"]["roc_auc"])
        rows.append(r)
    tabel = pd.DataFrame(rows).set_index("model")
    terbaik = tabel["roc_auc"].idxmax()
    st.success(f"ROC-AUC tertinggi: **{terbaik}** ({tabel.loc[terbaik, 'roc_auc']:.3f}). "
               f"Recall tertinggi pada threshold ini: **{tabel['recall'].idxmax()}** "
               f"({tabel['recall'].max():.1%} pelanggan churn tertangkap). "
               f"F1 tertinggi: **{tabel['f1'].idxmax()}** ({tabel['f1'].max():.3f}).")

    angka = [*METRIK, "cv_recall", "cv_f1", "cv_roc_auc"]
    st.dataframe(
        tabel[angka].rename(columns={**METRIK, "cv_recall": "CV Recall", "cv_f1": "CV F1",
                                     "cv_roc_auc": "CV ROC-AUC"})
        .style.format("{:.3f}").highlight_max(axis=0, color="rgba(27, 175, 122, 0.30)"),
        width="stretch",
    )
    st.caption("Kolom CV = rata-rata 5-fold stratified cross-validation (threshold 0,5); "
               "oversampling diterapkan di dalam setiap fold latih saja.")

    panjang = (tabel[["algoritma", "sampling", *METRIK]].rename(columns=METRIK)
               .melt(id_vars=["algoritma", "sampling"], var_name="metrik", value_name="nilai"))
    gb = px.bar(panjang, x="algoritma", y="nilai", color="sampling", barmode="group",
                facet_col="metrik", text_auto=".2f",
                color_discrete_map={"Tanpa oversampling": "#9dbbe6", "Oversampling": "#2a78d6"},
                title=f"Efek oversampling per algoritma (holdout, threshold {threshold:.2f})")
    gb.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    gb.update_yaxes(range=[0, 1])
    gb.update_xaxes(title="", tickangle=-35)
    st.plotly_chart(gaya(gb, 430), width="stretch")

    WARNA_ALGO = dict(zip(dict.fromkeys(m["family"] for m in MODELS.values()), KATEGORI))
    c3, c4 = st.columns(2)
    roc = go.Figure()
    for k, m in MODELS.items():
        fpr, tpr, _ = roc_curve(y_uji, holdout[k])
        roc.add_scatter(x=fpr, y=tpr, mode="lines",
                        line=dict(color=WARNA_ALGO[m["family"]], width=3.5 if k == model_key else 1.5,
                                  dash="dash" if m["oversampling"] else "solid"),
                        name=f"{m['name']} ({tabel.loc[m['name'], 'roc_auc']:.3f})")
    roc.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dot", color="gray"),
                    name="Tebakan acak", showlegend=False)
    roc.update_layout(title="Kurva ROC (putus-putus = oversampling, tebal = model aktif)",
                      xaxis_title="False positive rate", yaxis_title="True positive rate",
                      legend=dict(font_size=10))
    c3.plotly_chart(gaya(roc, 440), width="stretch")

    cm = confusion_matrix(y_uji, (holdout[model_key] >= threshold).astype(int))
    cmf = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                    x=["Prediksi Bertahan", "Prediksi Churn"], y=["Aktual Bertahan", "Aktual Churn"],
                    title=f"Confusion matrix — {aktif['name']}")
    cmf.update_coloraxes(showscale=False)
    c4.plotly_chart(gaya(cmf, 330), width="stretch")
    tn, fp, fn, tp = cm.ravel()
    c4.markdown(
        f"- **{tp}** pelanggan churn berhasil terdeteksi, **{fn}** lolos tidak terdeteksi.\n"
        f"- **{fp}** pelanggan setia ikut ditandai churn (biaya retensi terbuang).\n"
        f"- Oversampling atau threshold lebih rendah menangkap lebih banyak churn (recall naik), "
        f"dengan konsekuensi lebih banyak alarm palsu (precision turun).")

# 2) PREDIKSI SATU PELANGGAN — input lewat form
elif bagian == PRED_SATU:
    cats, rng = meta["categories"], meta["numeric_ranges"]
    with st.form("form_prediksi"):
        c1, c2, c3 = st.columns(3)
        region = c1.selectbox("Region", cats["region"])
        segment = c1.selectbox("Segmen", cats["segment"])
        contract = c2.selectbox("Jenis kontrak", cats["contract_type"])
        payment = c2.selectbox("Metode pembayaran", cats["payment_method"])
        service = c3.selectbox("Layanan internet", cats["internet_service"])
        tenure = c3.number_input("Tenure (bulan)", 1, 72, 6)

        d1, d2, d3 = st.columns(3)
        monthly = d1.number_input("Tagihan bulanan (Rp)", 100_000, 2_000_000, 450_000, step=10_000)
        complaints = d2.number_input("Jumlah keluhan", 0, 10, 2)
        calls = d3.number_input("Panggilan ke CS", 0, 15, 3)
        submitted = st.form_submit_button("🔮 Prediksi", type="primary")

    if submitted:
        row = pd.DataFrame([{
            "tenure_months": tenure, "monthly_charges": monthly,
            "total_charges": monthly * tenure, "num_complaints": complaints,
            "support_calls": calls, "region": region, "segment": segment,
            "contract_type": contract, "payment_method": payment, "internet_service": service,
        }])
        errs = validate_input(row, meta)
        if errs:
            st.error("\n".join(errs))
        else:
            res = predict(model, row, meta).iloc[0]
            m1, m2, m3 = st.columns(3)
            m1.metric("Probabilitas churn", f"{res.churn_probability:.1%}")
            m2.metric("Prediksi", "Churn" if res.predicted_churn else "Bertahan")
            m3.metric("Kategori risiko", f"{COLOR[res.risk_category]} {res.risk_category}")
            st.progress(float(res.churn_probability))
            if res.risk_category == "Tinggi":
                st.error("Rekomendasi: prioritaskan retensi — tawarkan migrasi ke kontrak tahunan "
                         "dan tindak lanjuti keluhan dalam 48 jam.")
            elif res.risk_category == "Sedang":
                st.warning("Rekomendasi: pantau, kirim survei kepuasan dan penawaran loyalitas.")
            else:
                st.success("Rekomendasi: pertahankan layanan standar.")

            with st.expander("Bandingkan dengan semua model"):
                semua = pd.DataFrame([
                    {"model": m["name"],
                     **predict(get_model(model_path(m)), row, meta).iloc[0][
                         ["churn_probability", "predicted_churn", "risk_category"]].to_dict()}
                    for m in MODELS.values()
                ])
                semua["predicted_churn"] = semua["predicted_churn"].map({1: "Churn", 0: "Bertahan"})
                st.dataframe(semua, width="stretch", hide_index=True, column_config={
                    "churn_probability": st.column_config.ProgressColumn(
                        "Probabilitas churn", min_value=0.0, max_value=1.0, format="%.2f")})

# 3) PREDIKSI BATCH — upload CSV pelanggan baru
elif bagian == PRED_BATCH:
    st.caption("Untuk latihan validasi, unggah sesi4/contoh_kontrak_salah.csv (satu kontrak Mingguan).")
    up = st.file_uploader("Upload CSV pelanggan baru (format sama dengan data/new_customers.csv)",
                          type="csv")
    pakai_contoh = st.checkbox("Gunakan contoh data/new_customers.csv", value=up is None)
    if up is not None:
        data = pd.read_csv(up)
    elif pakai_contoh:
        data = pd.read_csv(ROOT / "data" / "new_customers.csv")
    else:
        st.info("Upload file atau centang contoh data.")
        st.stop()

    errs = validate_input(data, meta)
    if errs:
        for e in errs:
            st.error(e)
        st.stop()

    hasil = predict(model, data, meta)
    berisiko = hasil.loc[hasil.risk_category == "Tinggi", "monthly_charges"].sum()
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Pelanggan diproses", f"{len(hasil):,}")
    k2.metric("Prediksi churn", f"{int(hasil.predicted_churn.sum()):,}")
    k3.metric("Rata-rata probabilitas", f"{hasil.churn_probability.mean():.1%}")
    k4.metric("Risiko tinggi", f"{(hasil.risk_category == 'Tinggi').sum():,}")
    k5.metric("Pendapatan berisiko", f"Rp{berisiko:,.0f}".replace(",", "."))

    RISK_COLOR = {"Rendah": "#1baf7a", "Sedang": "#eda100", "Tinggi": "#e34948"}
    a, b = st.columns(2)
    donut = px.pie(hasil, names="risk_category", hole=0.55, color="risk_category",
                   color_discrete_map=RISK_COLOR, title="Komposisi kategori risiko",
                   category_orders={"risk_category": ["Rendah", "Sedang", "Tinggi"]})
    donut.update_traces(textinfo="percent+label")
    a.plotly_chart(donut.update_layout(height=330, showlegend=False), width="stretch")
    pk = hasil.groupby("contract_type", as_index=False)["churn_probability"].mean()
    bar = px.bar(pk, x="contract_type", y="churn_probability", text_auto=".1%",
                 color_discrete_sequence=["#eb6834"], title="Rata-rata probabilitas churn per kontrak")
    bar.update_yaxes(tickformat=".0%")
    b.plotly_chart(bar.update_layout(height=330, template="plotly_white", xaxis_title=""),
                   width="stretch")

    st.markdown("**Jumlah pelanggan per region × kategori risiko**")
    ringkas = pd.crosstab(hasil["region"], hasil["risk_category"])
    for kolom in ("Rendah", "Sedang", "Tinggi"):
        if kolom not in ringkas.columns:
            ringkas[kolom] = 0
    st.dataframe(ringkas[["Rendah", "Sedang", "Tinggi"]], width="stretch")

    risiko = st.multiselect("Tampilkan kategori risiko", ["Tinggi", "Sedang", "Rendah"],
                            default=["Tinggi"])
    tampil = hasil[hasil.risk_category.isin(risiko)].sort_values("churn_probability",
                                                                 ascending=False)

    st.dataframe(tampil, width="stretch", height=320,
                 column_config={"churn_probability": st.column_config.ProgressColumn(
                     "Probabilitas churn", min_value=0.0, max_value=1.0, format="%.2f")})
    st.download_button("⬇️ Unduh hasil prediksi", hasil.to_csv(index=False).encode("utf-8"),
                       "hasil_prediksi_churn.csv", "text/csv")
