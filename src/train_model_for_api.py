from pathlib import Path
import json
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
APP_DIR = ROOT / "app"
TARGET = "churn_next_60d"
SPLIT = "split"
EXCLUDE = ["customer_id", "snapshot_date", TARGET, SPLIT]
SNAPSHOT_DATE = "2025-09-30"


def one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def choose_threshold(y_true, prob):
    best = {"threshold": 0.5, "f1": -1, "precision": 0, "recall": 0}
    for t in np.round(np.arange(0.10, 0.91, 0.01), 2):
        pred = (prob >= t).astype(int)
        p = precision_score(y_true, pred, zero_division=0)
        r = recall_score(y_true, pred, zero_division=0)
        f = f1_score(y_true, pred, zero_division=0)
        # Prefer high recall but avoid nearly-zero precision.
        if p >= 0.55 and r >= 0.75 and f > best["f1"]:
            best = {"threshold": float(t), "f1": float(f), "precision": float(p), "recall": float(r)}
    if best["f1"] < 0:
        for t in np.round(np.arange(0.10, 0.91, 0.01), 2):
            pred = (prob >= t).astype(int)
            f = f1_score(y_true, pred, zero_division=0)
            if f > best["f1"]:
                best = {"threshold": float(t), "f1": float(f), "precision": float(precision_score(y_true, pred, zero_division=0)), "recall": float(recall_score(y_true, pred, zero_division=0))}
    return best


def main() -> None:
    APP_DIR.mkdir(exist_ok=True)
    df = pd.read_csv(DATA_DIR / "rfm_modeling_snapshot.csv")
    X = df.drop(columns=EXCLUDE)
    y = df[TARGET].astype(int)
    cat = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num = [c for c in X.columns if c not in cat]

    preprocessor = ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num),
        ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", one_hot_encoder())]), cat),
    ], sparse_threshold=0)

    model = Pipeline([
        ("preprocess", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=300,
            max_depth=7,
            min_samples_leaf=8,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )),
    ])

    train = df[SPLIT].eq("train")
    validation = df[SPLIT].eq("validation")
    model.fit(X[train], y[train])
    val_prob = model.predict_proba(X[validation])[:, 1]
    threshold_info = choose_threshold(y[validation], val_prob)

    artifact = {
        "model": model,
        "threshold": threshold_info["threshold"],
        "threshold_info": threshold_info,
        "feature_columns": X.columns.tolist(),
        "numeric_columns": num,
        "categorical_columns": cat,
        "target": TARGET,
        "snapshot_date": SNAPSHOT_DATE,
    }
    joblib.dump(artifact, APP_DIR / "model.pkl")

    # Create a sample request from a real row, but do not include target/split/snapshot_date.
    sample = X.iloc[0].replace({np.nan: None}).to_dict()
    sample["customer_id"] = df.loc[0, "customer_id"]
    (ROOT / "sample_payload.json").write_text(json.dumps(sample, indent=2), encoding="utf-8")
    print(f"Saved {APP_DIR / 'model.pkl'}")
    print(f"Threshold info: {threshold_info}")
    print("Saved sample_payload.json")


if __name__ == "__main__":
    main()
