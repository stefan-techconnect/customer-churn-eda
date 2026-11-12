# sesi2/app_eda.py — EDA interaktif di Streamlit dengan beragam visualisasi Plotly
from pathlib import Path
import pandas as pd
import plotly.express as px

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
st.set_page_config(page_title="EDA Churn", page_icon="🔎", layout="wide")

st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        border-right: 1px solid rgba(128, 128, 128, 0.25);
    }
    [data-testid="stSidebar"] [data-testid="stRadioGroup"] > div {
        margin-bottom: 0.2rem;
    }
    [data-testid="stSidebar"] [data-testid="stRadioOption"] {
        display: flex;
        align-items: center;
        width: 100%;
        padding: 0.55rem 0.8rem;
        border-radius: 10px;
        border-left: 3px solid transparent;
        cursor: pointer;
        background: transparent;
    }
    [data-testid="stSidebar"] [data-testid="stRadioOption"]:hover {
        background: rgba(42, 120, 214, 0.12);
    }
    [data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected="true"] {
        background: rgba(42, 120, 214, 0.22);
        border-left-color: #2a78d6;
        font-weight: 600;
    }
    [data-testid="stSidebar"] [data-testid="stRadioOption"] > div > div:first-child {
        display: none;
    }
    [data-testid="stSidebar"] [data-testid="stRadioOption"] p {
        margin: 0;
        line-height: 1.3;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

STATUS = {"Bertahan": "#2a78d6", "Churn": "#eb6834"}
KATEGORI = ["#2a78d6", "#1baf7a", "#eda100", "#4a3aa7", "#e87ba4"]
KONTRAK = ["Bulanan", "Tahunan", "Dua Tahun"]


@st.cache_data
def load_data():
    df = pd.read_csv(ROOT / "data" / "customer_churn.csv", parse_dates=["signup_date"])
    df["total_charges_missing"] = df["total_charges"].isna()  # catat sebelum imputasi
    df["total_charges"] = df["total_charges"].fillna(df["total_charges"].median())
    df["status"] = df["churn"].map({0: "Bertahan", 1: "Churn"})
    df["kelompok_tenure"] = pd.cut(
        df["tenure_months"],
        [0, 12, 24, 48, 72],
        labels=["0-12", "13-24", "25-48", "49-72"],
    ).astype(str)
    return df


def gaya(fig, tinggi=360):
    fig.update_layout(
        template="plotly_white",
        height=tinggi,
        margin=dict(l=10, r=10, t=50, b=10),
        title_font_size=15,
        legend_title_text="",
    )
    return fig


df = load_data()
st.title("🔎 Exploratory Data Analysis — Customer Churn")
st.caption(
    "Data latihan sintetis · 3.000 pelanggan · pendaftaran Januari 2023 – Desember 2025"
)

# ---------- KPI CARDS ----------
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total pelanggan", f"{len(df):,}")
k2.metric("Churn rate", f"{df.churn.mean():.1%}")
k3.metric("Rata-rata tagihan", f"Rp{df.monthly_charges.mean():,.0f}".replace(",", "."))
k4.metric("Rata-rata tenure", f"{df.tenure_months.mean():.1f} bln")
k5.metric(
    "Missing total_charges",
    f"{df.total_charges_missing.sum()} baris",
    help="Diisi median",
)

BAGIAN = [
    "🥧 Komposisi",
    "📊 Perbandingan",
    "📈 Distribusi",
    "⏱️ Tren",
    "🔗 Hubungan",
    "📋 Data",
]
with st.sidebar:
    st.markdown("### 🔎 EDA Churn")
    st.caption("Pilih bagian analisis")
    bagian = st.radio("Bagian", BAGIAN, label_visibility="collapsed")
    st.divider()
    st.caption("Dataset latihan")
    st.markdown(f"**{len(df):,}** pelanggan  \nChurn **{df.churn.mean():.1%}**")

# ---------- 1. KOMPOSISI: pie, donut, treemap ----------
if bagian == "🥧 Komposisi":
    c1, c2 = st.columns(2)
    pie = px.pie(
        df,
        names="status",
        color="status",
        color_discrete_map=STATUS,
        title="Satu dari tiga pelanggan churn",
    )
    pie.update_traces(textinfo="percent+label", textfont_size=14)
    c1.plotly_chart(gaya(pie), width="stretch")
    donut = px.pie(
        df,
        names="contract_type",
        hole=0.55,
        color_discrete_sequence=KATEGORI,
        category_orders={"contract_type": KONTRAK},
        title="Separuh pelanggan berkontrak bulanan",
    )
    donut.update_traces(textinfo="percent+label")
    donut.add_annotation(
        text=f"<b>{len(df):,}</b><br>pelanggan", showarrow=False, font_size=16
    )
    c2.plotly_chart(gaya(donut), width="stretch")
    tm = df.groupby(["region", "segment"], as_index=False).agg(
        pelanggan=("customer_id", "count"), churn_rate=("churn", "mean")
    )
    tree = px.treemap(
        tm,
        path=["region", "segment"],
        values="pelanggan",
        color="churn_rate",
        color_continuous_scale="Oranges",
        title="Treemap: luas = jumlah pelanggan, warna = churn rate",
    )
    st.plotly_chart(gaya(tree, 420), width="stretch")

# ---------- 2. PERBANDINGAN: bar, grouped bar, stacked 100% ----------
elif bagian == "📊 Perbandingan":
    kolom = st.selectbox(
        "Bandingkan churn rate berdasarkan",
        ["contract_type", "region", "segment", "payment_method", "internet_service"],
    )
    rk = (
        df.groupby(kolom, as_index=False)
        .agg(pelanggan=("customer_id", "count"), churn_rate=("churn", "mean"))
        .sort_values("churn_rate")
    )
    bar = px.bar(
        rk,
        x="churn_rate",
        y=kolom,
        orientation="h",
        text_auto=".1%",
        color_discrete_sequence=["#eb6834"],
        title=f"Churn rate per {kolom}",
    )
    bar.add_vline(x=df.churn.mean(), line_dash="dot", annotation_text="rata-rata")
    bar.update_xaxes(tickformat=".0%")
    c1, c2 = st.columns([3, 2])
    c1.plotly_chart(gaya(bar), width="stretch")
    c2.dataframe(
        rk.sort_values("churn_rate", ascending=False)
        .style.format({"churn_rate": "{:.1%}"})
        .background_gradient(subset=["churn_rate"], cmap="Oranges"),
        width="stretch",
        hide_index=True,
    )
    c3, c4 = st.columns(2)
    gb = df.groupby(["contract_type", "segment"], as_index=False)["churn"].mean()
    grouped = px.bar(
        gb,
        x="contract_type",
        y="churn",
        color="segment",
        barmode="group",
        color_discrete_sequence=KATEGORI,
        category_orders={"contract_type": KONTRAK},
        title="Grouped bar: kontrak × segmen",
    )
    grouped.update_yaxes(tickformat=".0%")
    c3.plotly_chart(gaya(grouped), width="stretch")
    stacked = px.histogram(
        df,
        y="payment_method",
        color="status",
        barnorm="percent",
        color_discrete_map=STATUS,
        title="Stacked bar 100%: status per pembayaran",
    )
    stacked.update_layout(xaxis_title="Proporsi (%)", yaxis_title="")
    c4.plotly_chart(gaya(stacked), width="stretch")

# ---------- 3. DISTRIBUSI: histogram, box, violin ----------
elif bagian == "📈 Distribusi":
    num = st.selectbox(
        "Variabel numerik",
        ["tenure_months", "monthly_charges", "num_complaints", "support_calls"],
    )
    c1, c2 = st.columns(2)
    hist = px.histogram(
        df,
        x=num,
        color="status",
        barmode="overlay",
        opacity=0.7,
        nbins=30,
        color_discrete_map=STATUS,
        marginal="box",
        title=f"Histogram {num}",
    )
    c1.plotly_chart(gaya(hist, 400), width="stretch")
    vio = px.violin(
        df,
        x="contract_type",
        y=num,
        color="contract_type",
        box=True,
        color_discrete_sequence=KATEGORI,
        category_orders={"contract_type": KONTRAK},
        title=f"Violin {num} per kontrak",
    )
    c2.plotly_chart(gaya(vio, 400), width="stretch")

# ---------- 4. TREN: line, area ----------
elif bagian == "⏱️ Tren":
    bulanan = (
        df.set_index("signup_date")
        .resample("ME")["customer_id"]
        .count()
        .rename("pelanggan")
        .reset_index()
    )
    bulanan["rata2_3bln"] = bulanan["pelanggan"].rolling(3).mean()
    line = px.line(
        bulanan,
        x="signup_date",
        y=["pelanggan", "rata2_3bln"],
        markers=True,
        color_discrete_sequence=["#9dbbe6", "#2a78d6"],
        title="Pelanggan baru per bulan dan rata-rata bergerak 3 bulan",
    )
    st.plotly_chart(gaya(line), width="stretch")
    c1, c2 = st.columns(2)
    kt = df.groupby("kelompok_tenure", as_index=False)["churn"].mean()
    lt = px.line(
        kt,
        x="kelompok_tenure",
        y="churn",
        markers=True,
        text=kt["churn"].map("{:.1%}".format),
        color_discrete_sequence=["#eb6834"],
        title="Churn rate per kelompok tenure",
    )
    lt.update_traces(textposition="top center", marker_size=10)
    lt.update_yaxes(tickformat=".0%", range=[0, 0.65])
    c1.plotly_chart(gaya(lt), width="stretch")
    q = (
        df.assign(kuartal=df.signup_date.dt.to_period("Q").astype(str))
        .groupby(["kuartal", "segment"], as_index=False)["customer_id"]
        .count()
    )
    area = px.area(
        q,
        x="kuartal",
        y="customer_id",
        color="segment",
        color_discrete_sequence=KATEGORI,
        title="Area: pelanggan baru per kuartal",
    )
    c2.plotly_chart(gaya(area), width="stretch")

# ---------- 5. HUBUNGAN: scatter, heatmap ----------
elif bagian == "🔗 Hubungan":
    c1, c2 = st.columns(2)
    sc = px.scatter(
        df,
        x="tenure_months",
        y="monthly_charges",
        color="status",
        opacity=0.55,
        color_discrete_map=STATUS,
        hover_name="customer_id",
        hover_data=["contract_type", "num_complaints"],
        title="Tenure vs tagihan",
    )
    c1.plotly_chart(gaya(sc, 420), width="stretch")
    corr = df[
        ["tenure_months", "monthly_charges", "num_complaints", "support_calls", "churn"]
    ].corr()
    hm = px.imshow(
        corr.round(2),
        text_auto=True,
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Heatmap korelasi",
    )
    c2.plotly_chart(gaya(hm, 420), width="stretch")
    pv = pd.pivot_table(
        df, values="churn", index="region", columns="segment", aggfunc="mean"
    )
    hm2 = px.imshow(
        pv * 100,
        text_auto=".1f",
        color_continuous_scale="Oranges",
        labels={"color": "Churn (%)"},
        title="Heatmap churn rate region × segmen (%)",
    )
    st.plotly_chart(gaya(hm2, 360), width="stretch")

# ---------- 6. DATA: profil + tabel ----------
elif bagian == "📋 Data":
    profil = pd.DataFrame(
        {
            "tipe": df.dtypes.astype(str),
            "missing": df.isna()
            .sum()
            .where(df.columns != "total_charges", df.total_charges_missing.sum()),
            "unik": df.nunique(),
        }
    ).reset_index(names="kolom")
    c1, c2 = st.columns([2, 3])
    c1.markdown("**Profil kolom**")
    c1.dataframe(profil, width="stretch", hide_index=True)
    c2.markdown("**Statistik deskriptif**")
    c2.dataframe(df.describe(include="number").T.round(1), width="stretch")
    st.markdown("**10 baris pertama**")

    st.dataframe(df.head(10), width="stretch", hide_index=True)
