# sesi1/app_komponen.py — tur komponen dasar Streamlit
import datetime as dt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Komponen Dasar", page_icon="🧩", layout="centered")

# 1. Teks
st.title("Komponen Dasar Streamlit")
st.header("1. Menampilkan teks")
st.subheader("Subjudul")
st.markdown("Teks **tebal**, *miring*, dan `kode` dengan Markdown.")
st.caption("Caption untuk keterangan kecil.")
st.code("print('Halo')", language="python")

# 2. Data
st.header("2. Menampilkan data")
df = pd.DataFrame({"produk": ["A", "B", "C"], "penjualan": [120, 95, 143]})
st.dataframe(df, width="stretch")
st.metric(label="Total penjualan", value=f"{df['penjualan'].sum()} unit", delta="+12 vs bulan lalu")

# 3. Input widget
st.header("3. Input pengguna")
umur = st.slider("Umur", min_value=17, max_value=70, value=30)
kota = st.selectbox("Kota", ["Jakarta", "Makassar", "Medan", "Surabaya"])
minat = st.multiselect("Minat", ["Data", "Desain", "Bisnis"], default=["Data"])
tanggal = st.date_input("Tanggal mulai", dt.date(2026, 1, 1))
setuju = st.checkbox("Saya setuju dengan syarat & ketentuan")
st.write(f"Umur: {umur} | Kota: {kota} | Minat: {', '.join(minat)} | Mulai: {tanggal}")

# 4. Tombol dan status
st.header("4. Tombol & pesan status")
if st.button("Kirim"):
    if setuju:
        st.success("Data terkirim.")
    else:
        st.warning("Centang persetujuan terlebih dahulu.")
st.info("st.info untuk informasi netral.")

# 5. Grafik bawaan
st.header("5. Grafik bawaan")
rng = np.random.default_rng(0)
tren = pd.DataFrame({"hari": range(1, 31), "pengunjung": rng.integers(80, 160, 30)}).set_index("hari")
st.line_chart(tren)
