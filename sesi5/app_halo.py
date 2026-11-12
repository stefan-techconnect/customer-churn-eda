from dash import Dash, html

app = Dash(__name__)                     # 1. buat aplikasi
app.layout = html.Div([                  # 2. definisikan layout
    html.H1("Halo, Dash!"),
    html.P("Aplikasi Dash pertama saya."),
])

if __name__ == "__main__":               # 3. jalankan server
    app.run(debug=True)
