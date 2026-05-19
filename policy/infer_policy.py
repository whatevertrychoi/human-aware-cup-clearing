from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from project_utils import ConfigError, get_required, load_config


DEFAULT_CONFIG = ROOT / "configs" / "config.yaml"


def load_model_bundle(model_path: str | Path) -> dict:
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    return joblib.load(path)


def preprocess_inference_frame(frame: pd.DataFrame, model_bundle: dict) -> pd.DataFrame:
    processed = frame.copy()
    preprocessing = model_bundle.get("preprocessing", {})
    hand_distance_clip_upper = float(preprocessing.get("hand_distance_clip_upper", 2.0))
    last_touched_time_clip_upper = float(preprocessing.get("last_touched_time_clip_upper", 60.0))
    user_absent_time_clip_upper = float(preprocessing.get("user_absent_time_clip_upper", 60.0))

    if "hand_distance" in processed.columns:
        processed["hand_distance"] = processed["hand_distance"].clip(upper=hand_distance_clip_upper)
    if "last_touched_time" in processed.columns:
        processed["last_touched_time"] = processed["last_touched_time"].clip(upper=last_touched_time_clip_upper)
    if "user_absent_time" in processed.columns:
        processed["user_absent_time"] = processed["user_absent_time"].clip(upper=user_absent_time_clip_upper)
    return processed


def predict_actions(cup_features: dict | list[dict], model_bundle: dict, config: dict) -> list[dict]:
    items = [cup_features] if isinstance(cup_features, dict) else list(cup_features)
    feature_names = model_bundle["feature_names"]
    threshold = float(get_required(config, ["policy", "confidence_threshold"]))
    frame = pd.DataFrame(items)
    missing = [feature for feature in feature_names if feature not in frame.columns]
    if missing:
        raise ValueError(f"Input cup features are missing required fields: {missing}")

    frame = preprocess_inference_frame(frame, model_bundle)
    x_data = frame[feature_names]
    model = model_bundle["model"]
    raw_actions = model.predict(x_data)
    probabilities = model.predict_proba(x_data)
    class_names = list(model.classes_)

    predictions: list[dict] = []
    for item, raw_action, probability_vector in zip(items, raw_actions, probabilities):
        confidence = float(max(probability_vector))
        action = str(raw_action)
        uncertainty_override = False
        if raw_action != "WAIT" and confidence < threshold:
            action = "ASK"
            uncertainty_override = True

        output = {
            "cup_id": int(item.get("cup_id", -1)),
            "action": action,
            "confidence": confidence,
            "raw_action": str(raw_action),
            "probabilities": {
                label: float(probability_vector[index]) for index, label in enumerate(class_names)
            },
        }
        if uncertainty_override:
            output["uncertainty_override"] = True
        predictions.append(output)
    return predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Infer cup-clearing high-level actions from a trained model.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to config.yaml")
    parser.add_argument("--model", required=True, help="Path to model bundle")
    parser.add_argument(
        "--sample",
        default='[{"cup_id":1,"x":0.41,"y":0.08,"hand_distance":0.34,"last_touched_time":4.2,"touch_count":1,"moved_recently":1,"distance_to_tray":0.31,"user_present":1,"user_absent_time":0.0}]',
        help="JSON object or list of cup feature dicts",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = load_config(args.config)
        sample = json.loads(args.sample)
        model_bundle = load_model_bundle(args.model)
        predictions = predict_actions(sample, model_bundle, config)
    except (ConfigError, json.JSONDecodeError, FileNotFoundError, ValueError) as exc:
        print(f"[ERROR] {exc}")
        return 1

    print(json.dumps(predictions, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
