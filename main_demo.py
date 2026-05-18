from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from perception.detect_liquid_local import detect_liquid_local, make_mock_liquid_frame
from policy.infer_policy import load_model_bundle, predict_actions
from project_utils import ConfigError, load_config
from robot import mock_robot


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs" / "config.yaml"


def build_mock_cup_features() -> list[dict]:
    return [
        {
            "cup_id": 0,
            "x": 0.32,
            "y": -0.10,
            "hand_distance": 0.05,
            "last_touched_time": 0.0,
            "touch_count": 2,
            "moved_recently": 1,
            "distance_to_tray": 0.40,
            "user_present": 1,
            "user_absent_time": 0.0,
        },
        {
            "cup_id": 1,
            "x": 0.41,
            "y": 0.08,
            "hand_distance": 0.34,
            "last_touched_time": 4.2,
            "touch_count": 1,
            "moved_recently": 1,
            "distance_to_tray": 0.31,
            "user_present": 1,
            "user_absent_time": 0.0,
        },
        {
            "cup_id": 2,
            "x": 0.28,
            "y": 0.15,
            "hand_distance": 0.60,
            "last_touched_time": 55.0,
            "touch_count": 0,
            "moved_recently": 0,
            "distance_to_tray": 0.25,
            "user_present": 0,
            "user_absent_time": 12.0,
        },
    ]


def choose_liquid_frame(cup_id: int) -> np.ndarray:
    return make_mock_liquid_frame(fill=(cup_id % 2 == 1))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the cup clearing mock demo.")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to trained decision model bundle.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use built-in mock features and mock liquid verification frames.",
    )
    parser.add_argument(
        "--mock-responses",
        default="",
        help="Comma-separated mock user responses such as y,n for non-interactive ASK flow.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"[ERROR] {exc}")
        return 1

    model_bundle = load_model_bundle(args.model)
    cups = build_mock_cup_features() if args.mock else build_mock_cup_features()
    predictions = predict_actions(cups, model_bundle, config)
    mock_responses = [item.strip().lower() for item in args.mock_responses.split(",") if item.strip()]

    for prediction in predictions:
        cup_id = prediction["cup_id"]
        action = prediction["action"]
        confidence = prediction["confidence"]
        raw_action = prediction["raw_action"]
        print(f"Cup {cup_id} -> {action} (raw={raw_action}, confidence={confidence:.2f})")

        if action == "WAIT":
            mock_robot.wait()
            print()
            continue

        if action == "ASK":
            user_accepted = mock_robot.ask_user(cup_id, mock_responses)
            if not user_accepted:
                mock_robot.skip_cup(cup_id)
                print()
                continue

        mock_robot.approach_for_liquid_check(cup_id)
        frame = choose_liquid_frame(cup_id) if args.mock else choose_liquid_frame(cup_id)
        liquid_result = detect_liquid_local(frame, config)
        liquid_state = liquid_result["liquid_state"]
        print(
            f"[LOCAL VISION] cup {cup_id} -> {liquid_state} "
            f"(confidence={liquid_result['confidence']:.2f})"
        )

        if liquid_state == "EMPTY":
            mock_robot.clear_cup(cup_id)
        else:
            mock_robot.spill_safe_clear_cup(cup_id)
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())

