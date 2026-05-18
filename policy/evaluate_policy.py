from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from policy.infer_policy import load_model_bundle
from project_utils import ConfigError, get_required, load_config


DEFAULT_CONFIG = ROOT / "configs" / "config.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained decision policy on a CSV dataset.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to config.yaml")
    parser.add_argument("--data", required=True, help="Evaluation CSV path")
    parser.add_argument("--model", required=True, help="Path to model bundle")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = load_config(args.config)
        feature_names = get_required(config, ["policy", "model_input_features"])
        model_bundle = load_model_bundle(args.model)
    except (ConfigError, FileNotFoundError) as exc:
        print(f"[ERROR] {exc}")
        return 1

    df = pd.read_csv(args.data)
    x_data = df[feature_names]
    y_true = df["label"]
    y_pred = model_bundle["model"].predict(x_data)
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(classification_report(y_true, y_pred))
    return 0


if __name__ == "__main__":
    sys.exit(main())

