# sesi6/plotly_galeri.py — galeri 12 chart Plotly (jalankan: python sesi6/plotly_galeri.py)

from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parents[1]
STATUS = {"Bertahan": "#2a78d6", "Churn": "#eb6834"}
KATEGORI = ["#2a78d6", "#1baf7a", "#eda100", "#4a3aa7", "#e87ba4"]
KONTRAK = ["Bulanan", "Tahunan", "Dua Tahun"]


def load():
    df = pd.read_csv(ROOT / "data" / "customer_churn.csv", parse_dates=["signup_date"])
    df["status"] = df["churn"].map({0: "Bertahan", 1: "Churn"})
    df["tagihan_ribu"] = df["monthly_charges"] / 1000
    return df


def buat_figur(df):
    fig = {}
    # 1. Bar chart (Plotly Express) + hover kustom
    bar = df.groupby(["contract_type", "segment"], as_index=False)["churn"].mean()
    bar["churn_pct"] = bar["churn"] * 100
    f = px.bar(bar, x="contract_type", y="churn_pct", color="segment", barmode="group",
               color_discrete_sequence=KATEGORI, category_orders={"contract_type": KONTRAK},
               labels={"contract_type": "Jenis kontrak", "churn_pct": "Churn rate (%)", "segment": "Segmen"},
               title="Kontrak bulanan memiliki churn rate tertinggi di semua segmen")
    f.update_traces(hovertemplate="Kontrak: %{x}<br>Churn rate: %{y:.1f}%<extra></extra>")
    fig["bar"] = f

    # 2. Pie chart
    f = px.pie(df, names="status", color="status", color_discrete_map=STATUS,
               title="Proporsi status pelanggan")
    f.update_traces(textinfo="percent+label", pull=[0, 0.06])
    fig["pie"] = f

    # 3. Donut chart dengan angka di tengah
    f = px.pie(df, names="internet_service", hole=0.55, color_discrete_sequence=KATEGORI,
               title="Komposisi layanan internet")
    f.add_annotation(text=f"<b>{len(df):,}</b><br>pelanggan", showarrow=False, font_size=15)
    fig["donut"] = f

    # 4. Histogram dengan warna kategori
    fig["histogram"] = px.histogram(
        df, x="tenure_months", color="status", nbins=36, barmode="overlay", opacity=0.7,
        color_discrete_map=STATUS, labels={"tenure_months": "Tenure (bulan)", "status": "Status"},
        title="Churn terkonsentrasi pada pelanggan dengan tenure pendek")

    # 5. Scatter interaktif: hover_data, facet
    fig["scatter"] = px.scatter(
        df, x="tenure_months", y="tagihan_ribu", color="status", color_discrete_map=STATUS,
        hover_name="customer_id", hover_data={"region": True, "num_complaints": True, "tagihan_ribu": ":.0f"},
        facet_col="contract_type", category_orders={"contract_type": KONTRAK}, opacity=0.6,
        labels={"tenure_months": "Tenure (bulan)", "tagihan_ribu": "Tagihan (ribu Rp)"},
        title="Tenure vs tagihan per jenis kontrak")

    # 6. Line chart tren bulanan + range slider
    tren = (df.set_index("signup_date").groupby([pd.Grouper(freq="ME"), "status"])
              .size().reset_index(name="jumlah"))
    f = px.line(tren, x="signup_date", y="jumlah", color="status", markers=True,
                color_discrete_map=STATUS, labels={"signup_date": "Bulan pendaftaran", "jumlah": "Pelanggan"},
                title="Pelanggan baru per bulan menurut status akhir")
    f.update_xaxes(rangeslider_visible=True)
    fig["line"] = f

    # 7. Box plot
    fig["box"] = px.box(
        df, x="num_complaints", y="tagihan_ribu", color="status", color_discrete_map=STATUS,
        labels={"num_complaints": "Jumlah keluhan", "tagihan_ribu": "Tagihan (ribu Rp)"},
        title="Distribusi tagihan menurut jumlah keluhan")


    # 8. Graph Objects: bar dengan anotasi teks + garis referensi rata-rata
    reg = df.groupby("region", as_index=False)["churn"].mean().sort_values("churn", ascending=False)
    f = go.Figure(go.Bar(x=reg["region"], y=reg["churn"] * 100, marker_color="#eb6834",
                         text=(reg["churn"] * 100).round(1).astype(str) + "%", textposition="outside"))
    f.add_hline(y=df.churn.mean() * 100, line_dash="dot",
                annotation_text="rata-rata nasional", annotation_position="top right")
    f.update_layout(title="Churn rate per region (Graph Objects)", yaxis_title="Churn rate (%)",
                    yaxis_range=[0, 45])
    fig["go_bar"] = f

    # 9. Heatmap churn rate region x segmen
    pv = pd.pivot_table(df, values="churn", index="region", columns="segment", aggfunc="mean") * 100
    fig["heatmap"] = px.imshow(pv.round(1), text_auto=True, color_continuous_scale="Oranges",
                               aspect="auto", labels={"color": "Churn (%)"},
                               title="Heatmap churn rate region × segmen")

    # 10. Treemap & 11. Sunburst — data hierarkis
    cnt = df.groupby(["region", "segment", "contract_type"], as_index=False).agg(
        pelanggan=("customer_id", "count"), churn_rate=("churn", "mean"))
    fig["treemap"] = px.treemap(cnt, path=[px.Constant("Semua"), "region", "segment"],
                                values="pelanggan", color="churn_rate",
                                color_continuous_scale="Oranges",
                                title="Treemap: luas = pelanggan, warna = churn rate")
    fig["sunburst"] = px.sunburst(cnt, path=["contract_type", "segment"], values="pelanggan",
                                  color="contract_type", color_discrete_sequence=KATEGORI,
                                  title="Sunburst: kontrak → segmen")

    # 12. Funnel — dari seluruh pelanggan ke kelompok paling berisiko
    m1 = df.contract_type == "Bulanan"
    m2 = m1 & (df.tenure_months <= 12)
    m3 = m2 & (df.num_complaints >= 2)
    tahap = pd.DataFrame({
        "tahap": ["Semua pelanggan", "Kontrak bulanan", "+ tenure ≤ 12 bulan",
                  "+ keluhan ≥ 2", "Akhirnya churn"],
        "jumlah": [len(df), m1.sum(), m2.sum(), m3.sum(), (m3 & (df.churn == 1)).sum()]})
    f = px.funnel(tahap, x="jumlah", y="tahap", title="Funnel: penyempitan kelompok berisiko")
    f.update_traces(marker_color="#2a78d6", textinfo="value+percent initial")
    fig["funnel"] = f

    # 13. Indicator / gauge — KPI tunggal
    fig["gauge"] = go.Figure(go.Indicator(
        mode="gauge+number+delta", value=df.churn.mean() * 100, number={"suffix": "%"},
        delta={"reference": 30, "increasing": {"color": "#e34948"}, "suffix": " pp"},
        title={"text": "Churn rate vs target 30%"},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#eb6834"},
               "threshold": {"line": {"color": "black", "width": 3}, "value": 30}}))

    for f in fig.values():
        f.update_layout(template="plotly_white")
    return fig


if __name__ == "__main__":
    OUT = ROOT / "sesi6" / "output"; OUT.mkdir(exist_ok=True)
    figs = buat_figur(load())
    for nama, f in figs.items():
        f.write_html(OUT / f"{nama}.html", include_plotlyjs="cdn")
    print(f"{len(figs)} chart HTML tersimpan di", OUT)
