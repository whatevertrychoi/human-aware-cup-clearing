from __future__ import annotations

import argparse
import csv
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
VALID_LABELS = ["WAIT", "ASK", "IDLE", "CLEANUP_CANDIDATE"]


def create_model(algo: str):
    if algo == "rf":
        return RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
    if algo == "mlp":
        return MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
    raise ValueError(f"Unsupported algorithm: {algo}. Choose rf or mlp.")


def get_preprocessing_config(config: dict) -> dict:
    defaults = {
        "hand_distance_clip_upper": 2.0,
        "last_touched_time_clip_upper": 60.0,
        "user_absent_time_clip_upper": 60.0,
        "time_near_cup_clip_upper": 30.0,
        "time_since_release_clip_upper": 60.0,
        "cup_motion_distance_clip_upper": 2.0,
        "stationary_time_clip_upper": 60.0,
    }
    policy_cfg = config.get("policy", {}) if isinstance(config, dict) else {}
    preprocess_cfg = policy_cfg.get("preprocessing", {}) if isinstance(policy_cfg, dict) else {}
    for key in defaults:
        if key in preprocess_cfg:
            defaults[key] = float(preprocess_cfg[key])
    return defaults


def preprocess_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    processed = df.copy()
    clip_cfg = get_preprocessing_config(config)
    if "hand_distance" in processed.columns:
        processed["hand_distance"] = processed["hand_distance"].clip(upper=clip_cfg["hand_distance_clip_upper"])
    if "last_touched_time" in processed.columns:
        processed["last_touched_time"] = processed["last_touched_time"].clip(upper=clip_cfg["last_touched_time_clip_upper"])
    if "user_absent_time" in processed.columns:
        processed["user_absent_time"] = processed["user_absent_time"].clip(upper=clip_cfg["user_absent_time_clip_upper"])
    if "time_near_cup" in processed.columns:
        processed["time_near_cup"] = processed["time_near_cup"].clip(upper=clip_cfg["time_near_cup_clip_upper"])
    if "time_since_release" in processed.columns:
        processed["time_since_release"] = processed["time_since_release"].clip(upper=clip_cfg["time_since_release_clip_upper"])
    if "cup_motion_distance" in processed.columns:
        processed["cup_motion_distance"] = processed["cup_motion_distance"].clip(upper=clip_cfg["cup_motion_distance_clip_upper"])
    if "stationary_time" in processed.columns:
        processed["stationary_time"] = processed["stationary_time"].clip(upper=clip_cfg["stationary_time_clip_upper"])
    return processed


def apply_conservative_override(
    raw_predictions, probabilities, threshold: float, features: pd.DataFrame
) -> tuple[list[str], int]:
    adjusted: list[str] = []
    override_count = 0
    for row_index, (raw_action, probability_vector) in enumerate(zip(raw_predictions, probabilities)):
        confidence = float(max(probability_vector))
        if raw_action == "WAIT":
            adjusted.append("WAIT")
        elif raw_action == "CLEANUP_CANDIDATE" and confidence < threshold:
            used_cup_candidate = int(features.iloc[row_index].get("used_cup_candidate", 1))
            adjusted.append("ASK" if used_cup_candidate else "IDLE")
            override_count += 1
        elif raw_action == "ASK" and int(features.iloc[row_index].get("user_present", 0)) == 1 and int(features.iloc[row_index].get("used_cup_candidate", 1)) == 0:
            adjusted.append("IDLE")
        else:
            adjusted.append(str(raw_action))
    return adjusted, override_count


def compute_safety_metrics(y_true: pd.Series, y_pred: list[str], ask_override_count: int) -> dict[str, float]:
    y_true_series = pd.Series(y_true).reset_index(drop=True)
    y_pred_series = pd.Series(y_pred)
    risky_mask = y_true_series.isin(["WAIT", "ASK", "IDLE"])
    predicted_cleanup_mask = y_pred_series == "CLEANUP_CANDIDATE"
    true_cleanup_mask = y_true_series == "CLEANUP_CANDIDATE"
    true_wait_mask = y_true_series == "WAIT"
    predicted_ask_mask = y_pred_series == "ASK"
    true_ask_mask = y_true_series == "ASK"
    true_idle_mask = y_true_series == "IDLE"
    predicted_idle_mask = y_pred_series == "IDLE"

    wrong_cleanup_rate = float(((risky_mask) & predicted_cleanup_mask).sum()) / float(max(len(y_true_series), 1))
    cleanup_precision_den = int(predicted_cleanup_mask.sum())
    cleanup_candidate_precision = (
        float((predicted_cleanup_mask & true_cleanup_mask).sum()) / float(cleanup_precision_den)
        if cleanup_precision_den > 0
        else 0.0
    )
    unnecessary_ask_rate = float(((~true_ask_mask) & predicted_ask_mask).sum()) / float(max(len(y_true_series), 1))
    idle_precision_den = int(predicted_idle_mask.sum())
    idle_precision = (
        float((predicted_idle_mask & true_idle_mask).sum()) / float(idle_precision_den)
        if idle_precision_den > 0
        else 0.0
    )
    wait_recall_den = int(true_wait_mask.sum())
    wait_recall = (
        float(((y_pred_series == "WAIT") & true_wait_mask).sum()) / float(wait_recall_den)
        if wait_recall_den > 0
        else 0.0
    )
    return {
        "wrong_cleanup_rate": wrong_cleanup_rate,
        "ask_override_count": float(ask_override_count),
        "unnecessary_ask_rate": unnecessary_ask_rate,
        "idle_precision": idle_precision,
        "cleanup_candidate_precision": cleanup_candidate_precision,
        "wait_recall": wait_recall,
    }


def validate_feature_names(feature_names: list[str], df: pd.DataFrame) -> list[str]:
    missing_columns = [column for column in feature_names + ["label"] if column not in df.columns]
    if missing_columns:
        return missing_columns

    forbidden = {"cup_id", "color", "active_cup_id"}
    present_forbidden = [feature for feature in feature_names if feature in forbidden]
    if present_forbidden:
        raise ValueError(
            f"Model input features must not include identity helpers such as cup_id or color: {present_forbidden}"
        )
    return []


def build_output_paths(model_path: Path) -> dict[str, Path]:
    results_dir = model_path.parent
    suffix = ""
    if model_path.stem.startswith("decision_model"):
        suffix = model_path.stem[len("decision_model"):]
    else:
        suffix = f"_{model_path.stem}"
    return {
        "report": results_dir / f"classification_report{suffix}.txt",
        "confusion": results_dir / f"confusion_matrix{suffix}.png",
        "evaluation": results_dir / f"evaluation_summary{suffix}.csv",
    }


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
    df = preprocess_features(df, config)
    try:
        missing_columns = validate_feature_names(feature_names, df)
    except ValueError as exc:
        print(f"[ERROR] {exc}")
        return 1
    if missing_columns:
        print(f"[ERROR] Dataset is missing required columns: {missing_columns}")
        return 1
    invalid_labels = sorted(set(df["label"].dropna().unique()) - set(VALID_LABELS))
    if invalid_labels:
        print(f"[ERROR] Dataset contains invalid labels: {invalid_labels}")
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
    raw_predictions = model.predict(x_val)
    probabilities = model.predict_proba(x_val)
    labels = sorted(y_data.unique())
    confidence_threshold = float(get_required(config, ["policy", "confidence_threshold"]))
    predictions, ask_override_count = apply_conservative_override(raw_predictions, probabilities, confidence_threshold, x_val)
    accuracy = accuracy_score(y_val, predictions)
    report = classification_report(y_val, predictions, labels=labels)
    matrix = confusion_matrix(y_val, predictions, labels=labels)
    safety_metrics = compute_safety_metrics(y_val, predictions, ask_override_count)

    model_path = ensure_parent(args.model)
    bundle = {
        "model": model,
        "feature_names": list(feature_names),
        "labels": labels,
        "algo": args.algo,
        "preprocessing": get_preprocessing_config(config),
        "confidence_threshold": confidence_threshold,
    }
    joblib.dump(bundle, model_path)

    output_paths = build_output_paths(model_path)
    report_path = ensure_parent(output_paths["report"])
    metrics_block = "\n".join(f"{key}: {value:.4f}" for key, value in safety_metrics.items() if key != "ask_override_count")
    report_path.write_text(
        f"accuracy: {accuracy:.4f}\n"
        f"ask_override_count: {int(ask_override_count)}\n"
        f"{metrics_block}\n\n{report}",
        encoding="utf-8",
    )

    figure, axis = plt.subplots(figsize=(6, 6))
    ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=labels).plot(ax=axis, cmap="Blues", colorbar=False)
    axis.set_title("Decision Policy Confusion Matrix")
    figure.tight_layout()
    confusion_path = ensure_parent(output_paths["confusion"])
    figure.savefig(confusion_path, dpi=150)
    plt.close(figure)

    evaluation_path = ensure_parent(output_paths["evaluation"])
    with evaluation_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset", "algo", "metric", "value"])
        writer.writeheader()
        writer.writerow({"dataset": str(data_path), "algo": args.algo, "metric": "accuracy", "value": f"{accuracy:.4f}"})
        for key, value in safety_metrics.items():
            metric_value = int(value) if key == "ask_override_count" else f"{value:.4f}"
            writer.writerow({"dataset": str(data_path), "algo": args.algo, "metric": key, "value": metric_value})

    print(f"Validation accuracy: {accuracy:.4f}")
    print(report)
    print("Safety metrics:")
    for key, value in safety_metrics.items():
        if key == "ask_override_count":
            print(f"- {key}: {int(value)}")
        else:
            print(f"- {key}: {value:.4f}")
    print(f"Saved model bundle to {model_path}")
    print(f"Saved classification report to {report_path}")
    print(f"Saved confusion matrix to {confusion_path}")
    print(f"Saved evaluation summary to {evaluation_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
