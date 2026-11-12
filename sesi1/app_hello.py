# sesi1/app_hello.py — aplikasi Streamlit pertama

import streamlit as st

st.title("Halo, Streamlit! 👋")
st.write("Ini aplikasi data interaktif pertama saya.")

nama = st.text_input("Siapa nama Anda?")
if nama:
    st.success(f"Selamat datang di pelatihan dashboard, {nama}!")
