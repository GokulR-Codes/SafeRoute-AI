"""
Namma Safe BLR — ML Risk Prediction Model
Trains multiple classifiers to predict crime risk level.
Model is persisted to models/risk_model.pkl
"""

import os
import pandas as pd
import numpy as np
import pickle

from sklearn.ensemble        import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model    import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing   import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.metrics         import classification_report
from sklearn.pipeline        import Pipeline
from xgboost                 import XGBClassifier
from lightgbm                import LGBMClassifier
from catboost                import CatBoostClassifier

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "../data/bangalore_crime_dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "../models/risk_model.pkl")


def build_features(df: pd.DataFrame):
    df = df.copy()

    if "severity_norm" not in df.columns:
        df["severity_norm"] = MinMaxScaler().fit_transform(df[["crime_severity"]])

    RISK_MAP = {
        0:0.95, 1:0.98, 2:0.99, 3:1.0,  4:0.90, 5:0.75,
        6:0.55, 7:0.40, 8:0.35, 9:0.30, 10:0.28, 11:0.25,
        12:0.30, 13:0.28, 14:0.25, 15:0.28, 16:0.35, 17:0.45,
        18:0.60, 19:0.72, 20:0.82, 21:0.88, 22:0.92, 23:0.94,
    }
    df["time_risk"] = df["hour"].map(RISK_MAP).fillna(0.5)

    df["composite"] = (
        0.40 * df["severity_norm"] +
        0.20 * (1 - df["lighting_score"]) +
        0.20 * (1 - df["cctv_score"]) +
        0.10 * (1 - df["crowd_density"]) +
        0.10 * df["time_risk"]
    )

    df["risk_level"] = pd.cut(
        df["composite"],
        bins   = [0, 0.30, 0.55, 0.75, 1.01],
        labels = ["LOW", "MEDIUM", "HIGH", "CRITICAL"],
        right  = False,
    ).astype(str)

    FEATURE_COLS = [
        "severity_norm", "time_risk", "lighting_score",
        "cctv_score", "crowd_density", "police_proximity",
        "is_night", "hour",
    ]
    return df[FEATURE_COLS], df["risk_level"]


def train(data_path: str = DATA_PATH, model_path: str = MODEL_PATH):
    print("─" * 60)
    print("Namma Safe BLR — ML Model Training")
    print("─" * 60)

    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} records")

    X, y_str = build_features(df)
    print(f"Class distribution:\n{y_str.value_counts()}\n")

    # ── Encode string labels → integers (fixes XGBoost error) ────────────────
    le = LabelEncoder()
    y  = le.fit_transform(y_str)
    # le.classes_ will be ['CRITICAL','HIGH','LOW','MEDIUM'] (alphabetical)
    print(f"Label encoding: {dict(enumerate(le.classes_))}\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    candidates = {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(max_iter=500, C=1.0, random_state=42)),
        ]),
        "RandomForest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)),
        ]),
        "ExtraTrees": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    ExtraTreesClassifier(n_estimators=200, random_state=42, n_jobs=-1)),
        ]),
        "XGBoost": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    XGBClassifier(
                            n_estimators=300, max_depth=6, learning_rate=0.1,
                            random_state=42, eval_metric="mlogloss", verbosity=0,
                            # num_class handled automatically
                       )),
        ]),
        "LightGBM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LGBMClassifier(
                            n_estimators=500, learning_rate=0.05,
                            num_leaves=63, random_state=42, verbose=-1,
                       )),
        ]),
        "CatBoost": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    CatBoostClassifier(
                            iterations=500, learning_rate=0.05,
                            depth=6, random_seed=42, verbose=0,
                       )),
        ]),
    }

    results = {}
    for name, pipe in candidates.items():
        try:
            cv_scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring="accuracy")
            results[name] = cv_scores.mean()
            print(f"{name:25s}  CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        except Exception as e:
            print(f"{name:25s}  FAILED: {e}")
            results[name] = 0.0

    best_name = max(results, key=results.get)
    best_pipe = candidates[best_name]
    best_pipe.fit(X_train, y_train)

    y_pred   = best_pipe.predict(X_test)
    test_acc = (y_pred == y_test).mean()

    # Decode back to string labels for the report
    print(f"\n✅ Best model  : {best_name}")
    print(f"✅ Test accuracy: {test_acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(le.inverse_transform(y_test),
                                 le.inverse_transform(y_pred)))

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    bundle = {
        "model":         best_pipe,
        "model_name":    best_name,
        "feature_cols":  list(X.columns),
        "test_accuracy": test_acc,
        "label_encoder": le,           # saved so inference can decode predictions
        "all_results":   results,
    }
    with open(model_path, "wb") as f:
        pickle.dump(bundle, f)
    print(f"\n✅ Model saved → {model_path}")
    return bundle


def load_model(model_path: str = MODEL_PATH) -> dict:
    with open(model_path, "rb") as f:
        return pickle.load(f)


def predict_risk(bundle: dict, features: dict) -> dict:
    X          = pd.DataFrame([features])[bundle["feature_cols"]]
    model      = bundle["model"]
    le         = bundle["label_encoder"]
    pred_int   = model.predict(X)[0]
    proba      = model.predict_proba(X)[0]
    risk_label = le.inverse_transform([pred_int])[0]
    classes    = le.inverse_transform(model.classes_)

    return {
        "risk_level":    risk_label,
        "probabilities": dict(zip(classes.tolist(), proba.round(3).tolist())),
    }


if __name__ == "__main__":
    train()
