from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from project_utils import ConfigError, ensure_parent, get_required, load_config


DEFAULT_CONFIG = ROOT / "configs" / "config.yaml"


def create_model(algo: str):
    if algo == "rf":
        return RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
    if algo == "mlp":
        return MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
    raise ValueError(f"Unsupported algorithm: {algo}. Choose rf or mlp.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a cup-clearing high-level policy model.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to config.yaml")
    parser.add_argument("--data", required=True, help="Input CSV path")
    parser.add_argument("--model", required=True, help="Output joblib path")
    parser.add_argument("--algo", default="rf", choices=["rf", "mlp"], help="Training algorithm")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = load_config(args.config)
        feature_names = get_required(config, ["policy", "model_input_features"])
    except ConfigError as exc:
        print(f"[ERROR] {exc}")
        return 1

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"[ERROR] Dataset not found: {data_path}")
        return 1

    df = pd.read_csv(data_path)
    missing_columns = [column for column in feature_names + ["label"] if column not in df.columns]
    if missing_columns:
        print(f"[ERROR] Dataset is missing required columns: {missing_columns}")
        return 1

    x_data = df[feature_names]
    y_data = df["label"]
    x_train, x_val, y_train, y_val = train_test_split(
        x_data,
        y_data,
        test_size=0.2,
        random_state=42,
        stratify=y_data,
    )

    model = create_model(args.algo)
    model.fit(x_train, y_train)
    predictions = model.predict(x_val)
    labels = sorted(y_data.unique())
    accuracy = accuracy_score(y_val, predictions)
    report = classification_report(y_val, predictions, labels=labels)
    matrix = confusion_matrix(y_val, predictions, labels=labels)

    model_path = ensure_parent(args.model)
    bundle = {
        "model": model,
        "feature_names": list(feature_names),
        "labels": labels,
        "algo": args.algo,
    }
    joblib.dump(bundle, model_path)

    results_dir = model_path.parent
    report_path = ensure_parent(results_dir / "classification_report.txt")
    report_path.write_text(f"accuracy: {accuracy:.4f}\n\n{report}", encoding="utf-8")

    figure, axis = plt.subplots(figsize=(6, 6))
    ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=labels).plot(ax=axis, cmap="Blues", colorbar=False)
    axis.set_title("Decision Policy Confusion Matrix")
    figure.tight_layout()
    confusion_path = ensure_parent(results_dir / "confusion_matrix.png")
    figure.savefig(confusion_path, dpi=150)
    plt.close(figure)

    print(f"Validation accuracy: {accuracy:.4f}")
    print(report)
    print(f"Saved model bundle to {model_path}")
    print(f"Saved classification report to {report_path}")
    print(f"Saved confusion matrix to {confusion_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

