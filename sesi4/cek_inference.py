# sesi4/cek_inference.py — uji inference di terminal sebelum masuk dashboard
import sys
from pathlib import Path
import pandas as pd
pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 20)

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from utils.model_utils import load_model, load_metadata, validate_input, predict

meta = load_metadata()
model = load_model()
print("Model :", type(model).__name__, "->", [name for name, _ in model.steps])
print("Fitur :", meta["feature_order"])

baru = pd.read_csv(ROOT / "data" / "new_customers.csv")
print("Error validasi:", validate_input(baru, meta) or "tidak ada")

hasil = predict(model, baru, meta)
print(hasil[["customer_id", "contract_type", "num_complaints",
             "churn_probability", "predicted_churn", "risk_category"]].head())
print("\nJumlah per kategori risiko:\n", hasil["risk_category"].value_counts())
