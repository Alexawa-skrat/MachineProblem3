"""Train the MP2 ensemble and save artifacts used by app.py.

Usage:  python train_model.py [--data data/augmented_balanced_dataset_m.csv] [--out artifacts]
"""
import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (ExtraTreesClassifier, HistGradientBoostingClassifier,
                              VotingClassifier)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from features import FEATURES, TARGET, engineer_features


def train(data_path: str, out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(data_path)
    print("Dataset Shape:", df.shape)

    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")

    X = engineer_features(df)
    le = LabelEncoder()
    y = le.fit_transform(df[TARGET])  # High -> 0, Low -> 1
    print("Class mapping:", dict(zip(le.classes_, le.transform(le.classes_))))
    print("Engineered feature count:", X.shape[1])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)   # fit on training data only
    X_test_s = scaler.transform(X_test)

    hgb = HistGradientBoostingClassifier(max_iter=350, max_depth=14, learning_rate=0.07,
                                         l2_regularization=2.0, random_state=42)
    et = ExtraTreesClassifier(n_estimators=400, max_depth=20, min_samples_split=2,
                              random_state=42)
    model = VotingClassifier(estimators=[("hgb", hgb), ("et", et)], voting="soft")
    model.fit(X_train_s, y_train)

    y_pred = model.predict(X_test_s)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nTest Accuracy: {acc * 100:.2f}%")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    print("Confusion Matrix:\n", cm)

    joblib.dump({"model": model, "scaler": scaler, "classes": list(le.classes_),
                 "feature_columns": list(X.columns)},
                os.path.join(out_dir, "model.joblib"), compress=3)

    metrics = {"accuracy": acc, "report": report, "confusion_matrix": cm.tolist(),
               "classes": list(le.classes_), "n_total": int(len(df)),
               "n_train": int(len(X_train)), "n_test": int(len(X_test))}
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Dataset statistics for slider ranges, presets and "high/low" feature insights
    grid = list(range(0, 101, 5))
    stats = {"percentile_grid": grid, "features": {}, "class_medians": {}}
    for c in FEATURES:
        stats["features"][c] = {
            "min": float(df[c].min()), "max": float(df[c].max()),
            "mean": float(df[c].mean()), "median": float(df[c].median()),
            "quantiles": [float(v) for v in np.percentile(df[c], grid)],
        }
    for cls in le.classes_:
        stats["class_medians"][cls] = {c: float(df.loc[df[TARGET] == cls, c].median())
                                       for c in FEATURES}
    with open(os.path.join(out_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nArtifacts saved to '{out_dir}/'")
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/augmented_balanced_dataset_m.csv")
    ap.add_argument("--out", default="artifacts")
    args = ap.parse_args()
    train(args.data, args.out)
