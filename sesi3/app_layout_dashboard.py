# sesi3/app_layout_dashboard.py — Dashboard Streamlit dengan sidebar, columns, tabs
from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
st.set_page_config(page_title="Dashboard Churn", page_icon="📊", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv(ROOT / "data" / "customer_churn.csv", parse_dates=["signup_date"])
    df["total_charges"] = df["total_charges"].fillna(df["total_charges"].median())
    return df

def rupiah(x):
    return f"Rp{x:,.0f}".replace(",", ".")

df = load_data()

# ---------- SIDEBAR: filter ----------
st.sidebar.header("🎛️ Filter Data")
region = st.sidebar.multiselect("Region", sorted(df["region"].unique()),
                                default=sorted(df["region"].unique()))
segment = st.sidebar.multiselect("Segmen", sorted(df["segment"].unique()),
                                 default=sorted(df["segment"].unique()))
kontrak = st.sidebar.radio("Jenis kontrak", ["Semua"] + sorted(df["contract_type"].unique()))
tmin, tmax = int(df.tenure_months.min()), int(df.tenure_months.max())
tenure = st.sidebar.slider("Tenure (bulan)", tmin, tmax, (tmin, tmax))
tgl = st.sidebar.date_input("Periode pendaftaran",
                            (df.signup_date.min().date(), df.signup_date.max().date()))

mask = (df["region"].isin(region) & df["segment"].isin(segment)
        & df["tenure_months"].between(*tenure))
if kontrak != "Semua":
    mask &= df["contract_type"] == kontrak
if isinstance(tgl, tuple) and len(tgl) == 2:
    mask &= df["signup_date"].between(pd.Timestamp(tgl[0]), pd.Timestamp(tgl[1]))
dff = df[mask]

st.sidebar.caption(f"{len(dff):,} dari {len(df):,} pelanggan terpilih")

# ---------- HEADER ----------
st.title("📊 Dashboard Customer Churn")
st.caption("Data latihan sintetis · periode pendaftaran 2023–2025")

if dff.empty:
    st.warning("Tidak ada data untuk kombinasi filter ini. Longgarkan filter di sidebar.")
    st.stop()

# ---------- KPI CARDS (4 kolom) ----------
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total pelanggan", f"{len(dff):,}")
k2.metric("Churn rate", f"{dff.churn.mean():.1%}",
          delta=f"{(dff.churn.mean() - df.churn.mean()) * 100:+.1f} pp vs total",
          delta_color="inverse")
k3.metric("Rata-rata tagihan", rupiah(dff.monthly_charges.mean()))
k4.metric("Rata-rata tenure", f"{dff.tenure_months.mean():.1f} bln")

# ---------- TABS ----------
tab1, tab2, tab3 = st.tabs(["📈 Ringkasan", "🔍 Segmentasi", "📋 Data Detail"])


with tab1:
    c1, c2 = st.columns([3, 2])
    with c1:
        tren = (dff.set_index("signup_date").resample("ME")["customer_id"].count()
                  .rename("pelanggan").reset_index())
        fig_tren = px.line(tren, x="signup_date", y="pelanggan", markers=True,
                           title="Pelanggan baru per bulan", color_discrete_sequence=["#2a78d6"])
        st.plotly_chart(fig_tren.update_layout(height=320, template="plotly_white"), width="stretch")
    with c2:
        fig_donut = px.pie(dff, names="segment", hole=0.55, title="Komposisi segmen",
                           color_discrete_sequence=["#2a78d6", "#1baf7a", "#eda100"])
        fig_donut.update_traces(textinfo="percent+label")
        st.plotly_chart(fig_donut.update_layout(height=320, showlegend=False), width="stretch")
    ck = (dff.groupby("contract_type", as_index=False)["churn"].mean()
             .sort_values("churn", ascending=False))
    fig_bar = px.bar(ck, x="contract_type", y="churn", text_auto=".1%",
                     title="Churn rate per jenis kontrak", color_discrete_sequence=["#eb6834"])
    fig_bar.update_yaxes(tickformat=".0%", title="Churn rate")
    st.plotly_chart(fig_bar.update_layout(height=300, template="plotly_white", xaxis_title=""),
                    width="stretch")

with tab2:
    pilih = st.selectbox("Segmentasi berdasarkan", ["region", "segment", "payment_method",
                                                    "internet_service"])
    seg = (dff.groupby(pilih).agg(pelanggan=("customer_id", "count"),
                                  churn_rate=("churn", "mean"),
                                  tagihan=("monthly_charges", "mean"))
              .sort_values("churn_rate", ascending=False))
    a, b = st.columns(2)
    fig_seg = px.bar(seg.reset_index(), x=pilih, y="churn_rate", text_auto=".1%",
                     color_discrete_sequence=["#2a78d6"], title=f"Churn rate per {pilih}")
    fig_seg.update_yaxes(tickformat=".0%")
    a.plotly_chart(fig_seg.update_layout(height=320, template="plotly_white"), width="stretch")
    b.dataframe(seg.style.format({"churn_rate": "{:.1%}", "tagihan": rupiah})
                   .background_gradient(subset=["churn_rate"], cmap="Oranges"), width="stretch")

with tab3:
    with st.expander("Pilih kolom yang ditampilkan", expanded=False):
        cols = st.multiselect("Kolom", list(dff.columns), default=list(dff.columns)[:8])
    st.dataframe(dff[cols], width="stretch", height=360)
    st.download_button("⬇️ Unduh CSV", dff.to_csv(index=False).encode("utf-8"),
                       file_name="churn_filtered.csv", mime="text/csv")
