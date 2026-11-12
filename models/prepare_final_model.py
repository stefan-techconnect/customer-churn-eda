"""models/prepare_final_model.py — dokumentasi asal-usul model final.

Bukan bagian sesi. Peserta tidak melatih ulang model saat praktikum.
File churn_model.joblib dari paket fasilitator tidak ada di folder ini,
jadi skrip ini membangun Pipeline dengan struktur yang sama:
imputasi median, standarisasi, one-hot encoding, lalu GradientBoostingClassifier.
Split 80/20 terstratifikasi (random_state=42) menghasilkan angka holdout
yang sama dengan buku: ROC-AUC 0,7887 · akurasi 0,7183 · F1 0,5786.
"""
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "customer_churn.csv"
MODEL_PATH = ROOT / "models" / "churn_model.joblib"
META_PATH = ROOT / "models" / "model_metadata.json"

NUMERIC = ["tenure_months", "monthly_charges", "total_charges", "num_complaints", "support_calls"]
CATEGORICAL = ["region", "segment", "contract_type", "payment_method", "internet_service"]
FEATURES = NUMERIC + CATEGORICAL


def build_pipeline() -> Pipeline:
    preprocess = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                NUMERIC,
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL,
            ),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocess),
            (
                "clf",
                GradientBoostingClassifier(
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=3,
                    random_state=42,
                ),
            ),
        ]
    )


def main() -> None:
    df = pd.read_csv(DATA)
    X = df[FEATURES]
    y = df["churn"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = build_pipeline()
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    encoder = model.named_steps["preprocess"].named_transformers_["cat"]
    categories = {
        col: [str(v) for v in values]
        for col, values in zip(CATEGORICAL, encoder.categories_)
    }
    ranges = {
        col: [float(X[col].min(skipna=True)), float(X[col].max(skipna=True))]
        for col in NUMERIC
    }
    meta = {
        "model_name": "Customer Churn Classifier",
        "model_version": "1.0.0",
        "algorithm": "GradientBoostingClassifier (scikit-learn Pipeline)",
        "sklearn_version": "1.9.1",
        "numeric_features": NUMERIC,
        "categorical_features": CATEGORICAL,
        "feature_order": FEATURES,
        "categories": categories,
        "numeric_ranges": ranges,
        "decision_threshold": 0.5,
        "risk_bins": {"Rendah": [0.0, 0.3], "Sedang": [0.3, 0.6], "Tinggi": [0.6, 1.0]},
        "holdout_metrics": {
            "roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
            "accuracy": round(float(accuracy_score(y_test, pred)), 4),
            "f1": round(float(f1_score(y_test, pred)), 4),
        },
    }

    joblib.dump(model, MODEL_PATH)
    META_PATH.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Model tersimpan di", MODEL_PATH)
    print("Metadata:", json.dumps(meta["holdout_metrics"]))
    print("Langkah pipeline:", [name for name, _ in model.steps])


if __name__ == "__main__":
    main()
