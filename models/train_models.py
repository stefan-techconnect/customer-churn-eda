"""models/train_models.py — melatih model pembanding untuk dashboard Sesi 4.

Jalankan dari root proyek:  uv run python models/train_models.py

Algoritma (preprocessing sama persis dengan churn_model.joblib):
  - Gradient Boosting    : model final asli (churn_model.joblib, TIDAK dilatih ulang)
  - Logistic Regression  : model linear, baseline kuat & mudah dijelaskan
  - Random Forest        : ensemble bagging dari banyak decision tree
  - SVM (RBF kernel)     : model berbasis kernel/margin

Setiap algoritma juga dilatih dalam varian "+ Oversampling" (RandomOverSampler).
Pelanggan churn hanya ~35% data, sehingga model cenderung belajar lebih banyak dari
pelanggan bertahan. RandomOverSampler menduplikasi baris churn secara acak sampai
jumlahnya sama dengan pelanggan bertahan.

Dua teknik berbeda, jangan tertukar:
  - Stratified sampling : split train/test & fold CV menjaga rasio churn ~35% di setiap
                          bagian, agar evaluasi adil. Tidak mengubah ketimpangan kelas.
  - Oversampling        : menyeimbangkan kelas HANYA pada data latih. Sampler berada di
                          dalam imblearn Pipeline, sehingga hanya aktif saat fit — data uji
                          dan fold validasi CV tetap berisi data asli (tidak bocor).

Catatan: model oversampling "melihat" churn 50%, sehingga probabilitasnya cenderung
lebih tinggi daripada model biasa (recall naik, precision turun, ROC-AUC hampir sama).

Output:
  models/churn_model_<key>.joblib  — Pipeline siap inference per model
  models/model_registry.json       — daftar model + metrik holdout & cross-validation
  models/holdout_predictions.csv   — label asli & probabilitas tiap model pada data uji
                                     (dipakai dashboard untuk menghitung metrik per threshold)
"""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(Path(__file__).resolve().parent))
from prepare_final_model import DATA, FEATURES, MODEL_PATH, build_pipeline  # noqa: E402

MODELS_DIR = ROOT / "models"
REGISTRY_PATH = MODELS_DIR / "model_registry.json"
HOLDOUT_PATH = MODELS_DIR / "holdout_predictions.csv"
SCORING = ["accuracy", "precision", "recall", "f1", "roc_auc"]

# key -> (nama tampilan, deskripsi, classifier). Preprocessing diambil dari build_pipeline().
ALGORITHMS = {
    "gradient_boosting": (
        "Gradient Boosting",
        "Ensemble boosting 200 pohon dangkal; struktur sama dengan model final asli.",
        build_pipeline().named_steps["clf"],
    ),
    "logreg": (
        "Logistic Regression",
        "Model linear; koefisiennya mudah dijelaskan ke tim bisnis.",
        LogisticRegression(max_iter=2000),
    ),
    "random_forest": (
        "Random Forest",
        "Ensemble bagging 400 decision tree; menangkap interaksi non-linear.",
        RandomForestClassifier(n_estimators=400, min_samples_leaf=5, random_state=42, n_jobs=-1),
    ),
    "svm": (
        "SVM (RBF kernel)",
        "Support Vector Machine kernel RBF; probabilitas dari kalibrasi sigmoid (Platt).",
        CalibratedClassifierCV(SVC(kernel="rbf"), method="sigmoid", ensemble=False),
    ),
}


def make_pipeline(clf, oversample: bool):
    """Preprocessing + classifier; dengan oversample=True sampler ditaruh paling depan."""
    base = clone(build_pipeline()).set_params(clf=clone(clf))
    if not oversample:
        return base
    return ImbPipeline([("sampler", RandomOverSampler(random_state=42)), *base.steps])


def evaluate(y_true, proba, threshold=0.5) -> dict:
    pred = (proba >= threshold).astype(int)
    return {
        "accuracy": round(float(accuracy_score(y_true, pred)), 4),
        "precision": round(float(precision_score(y_true, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, pred)), 4),
        "f1": round(float(f1_score(y_true, pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, proba)), 4),
    }


def main() -> None:
    df = pd.read_csv(DATA)
    X, y = df[FEATURES], df["churn"]
    # Stratified split identik dengan prepare_final_model.py: rasio churn sama di train & test.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    holdout = pd.DataFrame({"churn": y_test.to_numpy()})
    registry = []

    _, y_resampled = RandomOverSampler(random_state=42).fit_resample(X_train, y_train)
    class_counts = {
        "sebelum": {"Bertahan": int((y_train == 0).sum()), "Churn": int((y_train == 1).sum())},
        "sesudah": {"Bertahan": int((y_resampled == 0).sum()), "Churn": int((y_resampled == 1).sum())},
    }

    for algo_key, (algo_name, description, clf) in ALGORITHMS.items():
        for oversample in (False, True):
            key = f"{algo_key}_ros" if oversample else algo_key
            name = f"{algo_name} + Oversampling" if oversample else algo_name
            template = make_pipeline(clf, oversample)

            if key == "gradient_boosting":
                # Model final asli: hanya dievaluasi, tidak ditimpa.
                path, model = MODEL_PATH, joblib.load(MODEL_PATH)
                desc = "Model final asli dari fasilitator (churn_model.joblib)."
            else:
                path = MODELS_DIR / f"churn_model_{key}.joblib"
                model = clone(template).fit(X_train, y_train)
                joblib.dump(model, path, compress=3)  # kompres agar Random Forest tidak membengkak
                desc = description

            proba = model.predict_proba(X_test)[:, 1]
            holdout[key] = proba.round(4)
            cv_res = cross_validate(template, X, y, cv=cv, scoring=SCORING, n_jobs=-1)
            algorithm = ("SVC RBF + CalibratedClassifierCV" if algo_key == "svm"
                         else type(clf).__name__)
            entry = {
                "key": key,
                "name": name,
                "family": algo_name,
                "oversampling": "RandomOverSampler" if oversample else None,
                "description": desc,
                "algorithm": f"{algorithm} (scikit-learn Pipeline)"
                             + (" + RandomOverSampler (imblearn)" if oversample else ""),
                "file": path.name,
                "holdout_metrics": evaluate(y_test, proba),
                "cv_metrics": {m: round(float(cv_res[f"test_{m}"].mean()), 4) for m in SCORING},
                "cv_roc_auc": {"mean": round(float(cv_res["test_roc_auc"].mean()), 4),
                               "std": round(float(cv_res["test_roc_auc"].std()), 4)},
            }
            registry.append(entry)
            print(f"{name:<36} {entry['holdout_metrics']}  CV AUC {entry['cv_roc_auc']['mean']}")

    meta = {
        "split": {"test_size": 0.2, "stratify": "churn", "random_state": 42,
                  "n_train": int(len(X_train)), "n_test": int(len(X_test))},
        "cv": "StratifiedKFold(5, shuffle=True, random_state=42) pada seluruh data",
        "metrics_threshold": 0.5,
        "train_class_counts": class_counts,
        "models": registry,
    }
    REGISTRY_PATH.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    holdout.to_csv(HOLDOUT_PATH, index=False)
    print("Distribusi kelas data latih:", class_counts)
    print("Registry:", REGISTRY_PATH)
    print("Holdout :", HOLDOUT_PATH)


if __name__ == "__main__":
    main()
