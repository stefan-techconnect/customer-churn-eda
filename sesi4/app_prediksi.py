# sesi4/app_prediksi.py — Integrasi final trained model ke dashboard Streamlit
import sys
from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))                      # agar folder utils bisa di-import
from utils.model_utils import load_model, load_metadata, validate_input, predict

st.set_page_config(page_title="EDA & Prediksi Churn", page_icon="📊", layout="wide")

st.markdown(
    """
    <style>
    [data-testid="stSidebar"] [data-testid="stRadioGroup"] > div { margin-bottom: 0.2rem; }
    [data-testid="stSidebar"] [data-testid="stRadioOption"] {
        display: flex; align-items: center; width: 100%;
        padding: 0.55rem 0.8rem; border-radius: 10px;
        border-left: 3px solid transparent; cursor: pointer; background: transparent;
    }
    [data-testid="stSidebar"] [data-testid="stRadioOption"]:hover {
        background: rgba(42, 120, 214, 0.12);
    }
    [data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected="true"] {
        background: rgba(42, 120, 214, 0.22); border-left-color: #2a78d6; font-weight: 600;
    }
    [data-testid="stSidebar"] [data-testid="stRadioOption"] > div > div:first-child { display: none; }
    [data-testid="stSidebar"] [data-testid="stRadioOption"] p { margin: 0; line-height: 1.3; }
    </style>
    """,
    unsafe_allow_html=True,
)

STATUS = {"Bertahan": "#2a78d6", "Churn": "#eb6834"}
KATEGORI = ["#2a78d6", "#1baf7a", "#eda100", "#4a3aa7", "#e87ba4"]
KONTRAK = ["Bulanan", "Tahunan", "Dua Tahun"]
EDA_BAGIAN = ["🥧 Komposisi", "📊 Perbandingan", "📈 Distribusi", "⏱️ Tren", "🔗 Hubungan", "📋 Data"]
PRED_SATU = "👤 Prediksi satu pelanggan"
PRED_BATCH = "📂 Prediksi batch (CSV)"
BAGIAN = EDA_BAGIAN + [PRED_SATU, PRED_BATCH]

# 1) LOAD MODEL — sekali saja, disimpan di cache
@st.cache_resource
def get_model():
    return load_model()

@st.cache_data
def get_meta():
    return load_metadata()

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

def gaya(fig, tinggi=360):
    fig.update_layout(
        template="plotly_white", height=tinggi, margin=dict(l=10, r=10, t=50, b=10),
        title_font_size=15, legend_title_text="",
    )
    return fig

model = get_model()
base_meta = get_meta()
threshold = st.sidebar.slider(
    "Threshold keputusan", 0.3, 0.7, 0.5, 0.05,
    help="Pelanggan diprediksi churn jika probabilitas ≥ threshold.",
)
meta = dict(base_meta)
meta["decision_threshold"] = float(threshold)
st.sidebar.caption("Mengubah threshold mengubah jumlah Prediksi churn.")
st.sidebar.divider()
eda = load_eda()
st.sidebar.markdown("### Navigasi")
st.sidebar.caption("Pilih satu tampilan")
bagian = st.sidebar.radio("Bagian", BAGIAN, label_visibility="collapsed")
st.sidebar.caption(f"{len(eda):,} pelanggan · churn {eda.churn.mean():.1%}")

st.title("📊 EDA & Prediksi Risiko Churn")
st.caption("Eksplorasi customer_churn.csv, lalu inference model final pada pelanggan baru.")
with st.expander("ℹ️ Informasi model final", expanded=False):
    st.write(f"**{meta['model_name']}** · versi {meta['model_version']} · {meta['algorithm']}")
    st.write("Fitur input:", ", ".join(meta["feature_order"]))
    st.write(f"Threshold keputusan: {meta['decision_threshold']} · "
             f"Risk bins: {meta['risk_bins']}")
    st.caption("Model tidak dilatih ulang di aplikasi ini; aplikasi hanya melakukan inference.")

COLOR = {"Rendah": "🟢", "Sedang": "🟡", "Tinggi": "🔴"}

if bagian in EDA_BAGIAN:
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
