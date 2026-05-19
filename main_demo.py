from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from perception.detect_cups import detect_cups
from perception.detect_hand import detect_hand, has_mediapipe_solutions
from perception.detect_liquid_local import detect_liquid_local, make_mock_liquid_frame
from perception.detect_user_presence import detect_user_presence
from policy.infer_policy import load_model_bundle, predict_actions
from project_utils import ConfigError, ensure_parent, get_required, load_config
from robot import mock_robot
from tracking.interaction_tracker import InteractionTracker
from tracking.user_presence_tracker import UserPresenceTracker


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs" / "config.yaml"
BACKEND_MAP = {
    "auto": None,
    "dshow": getattr(cv2, "CAP_DSHOW", None),
    "msmf": getattr(cv2, "CAP_MSMF", None),
}


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
    parser = argparse.ArgumentParser(description="Run the cup clearing demo.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to config.yaml")
    parser.add_argument("--model", default=None, help="Path to trained decision model bundle")
    parser.add_argument("--mock", action="store_true", help="Use built-in mock features and mock liquid verification frames")
    parser.add_argument("--mock-responses", default="", help="Comma-separated mock user responses such as y,n")
    parser.add_argument("--camera-index", type=int, default=None, help="Camera index for live perception")
    parser.add_argument("--backend", default="auto", choices=["auto", "dshow", "msmf"], help="OpenCV backend")
    parser.add_argument("--width", type=int, default=None, help="Override capture width")
    parser.add_argument("--height", type=int, default=None, help="Override capture height")
    parser.add_argument("--debug-perception", action="store_true", help="Run live cup or hand or user perception debug view")
    parser.add_argument("--live-policy", action="store_true", help="Run live policy inference on webcam perception")
    parser.add_argument(
        "--policy-mode",
        default="safety_guard",
        choices=["model_only", "safety_guard", "arbitration"],
        help="Choose model-only, lightweight safety-guard, or full arbitration live policy mode",
    )
    parser.add_argument("--log-live-eval", default=None, help="Optional CSV path for live policy evaluation logging")
    return parser.parse_args()


def open_camera(camera_index: int, backend: str, width: int, height: int):
    backend_id = BACKEND_MAP[backend]
    cap = cv2.VideoCapture(camera_index) if backend_id is None else cv2.VideoCapture(camera_index, backend_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def create_mediapipe_contexts():
    if not has_mediapipe_solutions():
        return None, None

    import mediapipe as mp

    hands_ctx = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.35,
        min_tracking_confidence=0.35,
    )
    face_ctx = [
        mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.35),
        mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.35),
    ]
    return hands_ctx, face_ctx


def get_live_policy_thresholds(config: dict) -> dict:
    tracking_cfg = config.get("tracking", {}) if isinstance(config, dict) else {}
    policy_cfg = config.get("policy", {}) if isinstance(config, dict) else {}
    return {
        "touch_threshold": float(tracking_cfg.get("touch_threshold", 0.12)),
        "user_absence_threshold": float(tracking_cfg.get("user_absence_threshold", 10.0)),
        "recent_touch_threshold": float(policy_cfg.get("recent_touch_threshold", 10.0)),
        "post_active_idle_ask_threshold": float(policy_cfg.get("post_active_idle_ask_threshold", 20.0)),
        "untouched_idle_ask_threshold": float(policy_cfg.get("untouched_idle_ask_threshold", 60.0)),
        "cleanup_time_threshold": float(policy_cfg.get("cleanup_time_threshold", 30.0)),
        "stationary_threshold": float(policy_cfg.get("stationary_threshold", 3.0)),
        "confidence_threshold": float(policy_cfg.get("confidence_threshold", 0.65)),
    }


def collect_live_perception(frame, config: dict, hands_ctx, face_ctx, interaction_tracker, user_presence_tracker, dt: float):
    tray_position = get_required(config, ["robot", "tray_position"])
    cups = detect_cups(frame, config)
    for cup in cups:
        cx, cy = cup["center_pixel"]
        cup["x"] = float(cx)
        cup["y"] = float(cy)
        cup["frame_width"] = frame.shape[1]
        cup["frame_height"] = frame.shape[0]
        cup["x_norm"] = float(cx) / float(max(frame.shape[1], 1))
        cup["y_norm"] = float(cy) / float(max(frame.shape[0], 1))
        cup["distance_to_tray"] = float(
            ((cup["x_norm"] - float(tray_position["x"])) ** 2 + (cup["y_norm"] - float(tray_position["y"])) ** 2) ** 0.5
        )

    hand = detect_hand(frame, hands=hands_ctx)
    user_presence = detect_user_presence(frame, face_detector=face_ctx, hand_detection=hand)
    user_state = user_presence_tracker.update(bool(user_presence["user_present"]), dt)
    user_state["face_center"] = user_presence.get("face_center")
    user_state["confidence"] = user_presence.get("confidence", 0.0)
    user_state["source"] = user_presence.get("source", "none")
    tracked_cups = interaction_tracker.update(cups, hand, dt)
    return tracked_cups, hand, user_state


def compute_idle_duration(cup: dict) -> float:
    if int(cup.get("used_cup_candidate", 0)) == 1:
        return float(cup.get("last_touched_time", 0.0))
    return float(cup.get("stationary_time", 0.0))


def draw_perception_debug(frame, cups: list[dict], hand: dict, user_state: dict) -> np.ndarray:
    output = frame.copy()
    if hand.get("hand_visible") and hand.get("hand_center") is not None:
        hx, hy = hand["hand_center"]
        cv2.circle(output, (hx, hy), 10, (0, 255, 255), -1)
        cv2.putText(output, "hand", (hx + 10, hy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        for cup in cups:
            cx, cy = cup["center_pixel"]
            cv2.line(output, (hx, hy), (cx, cy), (0, 255, 255), 1)

    if user_state.get("face_center") is not None:
        fx, fy = user_state["face_center"]
        cv2.circle(output, (fx, fy), 10, (255, 0, 255), -1)
        cv2.putText(output, "user", (fx + 10, fy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    info_lines = [
        f"user_present={int(user_state.get('user_present', 0))}",
        f"user_absent_time={user_state.get('user_absent_time', 0.0):.1f}s",
        f"hand_visible={int(hand.get('hand_visible', False))}",
        "q: quit",
    ]
    for index, line in enumerate(info_lines):
        cv2.putText(output, line, (10, 30 + (index * 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    color_map = {"green": (0, 220, 0), "red": (0, 0, 255), "blue": (255, 120, 0)}
    for cup in cups:
        x1, y1, x2, y2 = cup["bbox"]
        cx, cy = cup["center_pixel"]
        draw_color = color_map.get(cup["color"], (0, 255, 0))
        cv2.rectangle(output, (x1, y1), (x2, y2), draw_color, 2)
        cv2.circle(output, (cx, cy), 5, draw_color, -1)
        cv2.putText(
            output,
            f"Cup {cup['cup_id']} {cup['color']} touches={cup.get('touch_count', 0)} last={cup.get('last_touched_time', 0.0):.1f}s",
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            draw_color,
            2,
        )
        cv2.putText(
            output,
            (
                f"near={cup.get('time_near_cup', 0.0):.1f}s "
                f"rel={cup.get('release_count', 0)} "
                f"stat={cup.get('stationary_time', 0.0):.1f}s "
                f"used={cup.get('used_cup_candidate', 0)}"
            ),
            (x1, min(output.shape[0] - 10, y2 + 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            draw_color,
            1,
        )
        if cup.get("is_active_cup", 0):
            cv2.putText(output, "ACTIVE", (x1, min(output.shape[0] - 28, y2 + 36)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    return output


def apply_live_safety_guard(cups: list[dict], predictions: list[dict], user_state: dict, hand: dict, config: dict) -> list[dict]:
    thresholds = get_live_policy_thresholds(config)
    prediction_map = {int(item["cup_id"]): item for item in predictions}
    results: list[dict] = []

    for cup in cups:
        cup_id = int(cup["cup_id"])
        prediction = prediction_map.get(
            cup_id,
            {"cup_id": cup_id, "action": "IDLE", "raw_action": "IDLE", "confidence": 0.0, "probabilities": {}},
        )
        raw_action = prediction.get("raw_action", prediction.get("action", "IDLE"))
        confidence = float(prediction.get("confidence", 0.0))
        hand_distance = float(cup.get("hand_distance", 999.0))
        used_cup_candidate = bool(cup.get("used_cup_candidate", 0))
        user_present = int(user_state.get("user_present", 0))
        active_and_near = bool(hand.get("hand_visible")) and bool(cup.get("is_active_cup", 0)) and hand_distance < thresholds["touch_threshold"]
        cleanup_ready = (
            user_present == 0
            and float(user_state.get("user_absent_time", 0.0)) > thresholds["user_absence_threshold"]
            and float(cup.get("stationary_time", 0.0)) > thresholds["stationary_threshold"]
        ) or (
            user_present == 0
            and float(cup.get("last_touched_time", 999.0)) > thresholds["cleanup_time_threshold"]
            and float(cup.get("stationary_time", 0.0)) > thresholds["stationary_threshold"]
        )

        action = str(prediction.get("action", raw_action))
        reason = "model_first"
        uncertainty_override = False

        if active_and_near:
            action = "WAIT"
            reason = "safety_wait_override"
        elif action == "CLEANUP_CANDIDATE" and confidence < thresholds["confidence_threshold"]:
            action = "ASK" if used_cup_candidate else "IDLE"
            reason = "low_confidence_cleanup_guard"
            uncertainty_override = True
        elif action == "ASK" and user_present == 1 and not used_cup_candidate:
            action = "IDLE"
            reason = "unused_cup_ask_suppressed"
        elif action == "CLEANUP_CANDIDATE" and not cleanup_ready:
            action = "ASK" if used_cup_candidate else "IDLE"
            reason = "cleanup_requires_abandonment"

        merged = dict(prediction)
        merged["action"] = action
        merged["raw_action"] = raw_action
        merged["confidence"] = confidence
        merged["reason"] = reason
        merged["used_cup_candidate"] = used_cup_candidate
        merged["is_active_cup"] = bool(cup.get("is_active_cup", 0))
        merged["uncertainty_override"] = uncertainty_override or bool(prediction.get("uncertainty_override", False))
        results.append(merged)
    return results


def apply_live_arbitration(cups: list[dict], predictions: list[dict], user_state: dict, hand: dict, config: dict) -> list[dict]:
    thresholds = get_live_policy_thresholds(config)
    prediction_map = {int(item["cup_id"]): item for item in predictions}
    results: list[dict] = []

    for cup in cups:
        cup_id = int(cup["cup_id"])
        prediction = prediction_map.get(
            cup_id,
            {"cup_id": cup_id, "action": "IDLE", "raw_action": "IDLE", "confidence": 0.0, "probabilities": {}},
        )
        raw_action = prediction.get("raw_action", prediction.get("action", "IDLE"))
        confidence = float(prediction.get("confidence", 0.0))
        hand_distance = float(cup.get("hand_distance", 999.0))
        used_cup_candidate = bool(cup.get("used_cup_candidate", 0))
        user_present = int(user_state.get("user_present", 0))
        active_and_near = bool(hand.get("hand_visible")) and bool(cup.get("is_active_cup", 0)) and hand_distance < thresholds["touch_threshold"]
        post_active_idle_ready = (
            user_present == 1
            and used_cup_candidate
            and not active_and_near
            and hand_distance >= thresholds["touch_threshold"]
            and float(cup.get("last_touched_time", 999.0)) >= thresholds["post_active_idle_ask_threshold"]
        )
        untouched_idle_ready = (
            user_present == 1
            and not used_cup_candidate
            and not active_and_near
            and float(cup.get("stationary_time", 0.0)) >= thresholds["untouched_idle_ask_threshold"]
        )
        cleanup_ready = (
            user_present == 0
            and float(user_state.get("user_absent_time", 0.0)) > thresholds["user_absence_threshold"]
            and float(cup.get("stationary_time", 0.0)) > thresholds["stationary_threshold"]
        ) or (
            user_present == 0
            and float(cup.get("last_touched_time", 999.0)) > thresholds["cleanup_time_threshold"]
            and float(cup.get("stationary_time", 0.0)) > thresholds["stationary_threshold"]
        )

        action = str(prediction.get("action", raw_action))
        reason = "model_first"
        uncertainty_override = False

        if active_and_near:
            action = "WAIT"
            reason = "safety_wait_override"
        elif post_active_idle_ready:
            action = "ASK"
            reason = "post_active_idle_timeout"
        elif untouched_idle_ready:
            action = "ASK"
            reason = "untouched_idle_timeout"
        elif user_present == 1 and not used_cup_candidate:
            action = "IDLE"
            reason = "present_unused_idle_suppression"
        elif user_present == 1 and used_cup_candidate and hand_distance >= thresholds["touch_threshold"] and float(cup.get("last_touched_time", 999.0)) < thresholds["post_active_idle_ask_threshold"]:
            action = "IDLE"
            reason = "used_cup_idle_grace_period"
        elif action == "CLEANUP_CANDIDATE" and confidence < thresholds["confidence_threshold"]:
            action = "ASK" if used_cup_candidate else "IDLE"
            reason = "low_confidence_cleanup_guard"
            uncertainty_override = True
        elif action == "ASK" and user_present == 1 and not used_cup_candidate:
            action = "IDLE"
            reason = "unused_cup_ask_suppressed"
        elif action == "CLEANUP_CANDIDATE" and not cleanup_ready:
            action = "ASK" if used_cup_candidate else "IDLE"
            reason = "cleanup_requires_abandonment"

        merged = dict(prediction)
        merged["action"] = action
        merged["raw_action"] = raw_action
        merged["confidence"] = confidence
        merged["reason"] = reason
        merged["used_cup_candidate"] = used_cup_candidate
        merged["is_active_cup"] = bool(cup.get("is_active_cup", 0))
        merged["uncertainty_override"] = uncertainty_override or bool(prediction.get("uncertainty_override", False))
        results.append(merged)
    return results


def apply_model_only(cups: list[dict], predictions: list[dict]) -> list[dict]:
    prediction_map = {int(item["cup_id"]): item for item in predictions}
    results: list[dict] = []
    for cup in cups:
        cup_id = int(cup["cup_id"])
        prediction = prediction_map.get(
            cup_id,
            {"cup_id": cup_id, "action": "IDLE", "raw_action": "IDLE", "confidence": 0.0, "probabilities": {}},
        )
        merged = dict(prediction)
        merged["action"] = str(prediction.get("action", "IDLE"))
        merged["raw_action"] = str(prediction.get("raw_action", merged["action"]))
        merged["reason"] = "model_only"
        merged["used_cup_candidate"] = bool(cup.get("used_cup_candidate", 0))
        merged["is_active_cup"] = bool(cup.get("is_active_cup", 0))
        merged["uncertainty_override"] = bool(prediction.get("uncertainty_override", False))
        results.append(merged)
    return results


def draw_live_policy_overlay(frame, cups: list[dict], hand: dict, user_state: dict, predictions: list[dict], policy_mode: str) -> np.ndarray:
    output = frame.copy()
    prediction_map = {item["cup_id"]: item for item in predictions}
    action_colors = {
        "WAIT": (0, 215, 255),
        "ASK": (0, 165, 255),
        "CLEANUP_CANDIDATE": (0, 0, 255),
        "IDLE": (180, 180, 180),
    }

    if hand.get("hand_visible") and hand.get("hand_center") is not None:
        hx, hy = hand["hand_center"]
        cv2.circle(output, (hx, hy), 10, (0, 255, 255), -1)
        cv2.putText(output, "hand", (hx + 10, hy), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

    if user_state.get("face_center") is not None:
        fx, fy = user_state["face_center"]
        cv2.circle(output, (fx, fy), 10, (255, 0, 255), -1)
        cv2.putText(output, "user", (fx + 10, fy), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 2)

    info_lines = [
        f"policy_mode={policy_mode}",
        f"user_present={int(user_state.get('user_present', 0))}",
        f"user_absent_time={user_state.get('user_absent_time', 0.0):.1f}s",
        f"hand_visible={int(hand.get('hand_visible', False))}",
        "live policy | q: quit",
    ]
    for index, line in enumerate(info_lines):
        cv2.putText(output, line, (10, 30 + (index * 24)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    ask_message_drawn = False
    for cup in cups:
        prediction = prediction_map.get(int(cup["cup_id"]), {})
        action = prediction.get("action", "UNKNOWN")
        confidence = prediction.get("confidence", 0.0)
        raw_action = prediction.get("raw_action", action)
        override_reason = prediction.get("reason", "none")
        x1, y1, x2, y2 = cup["bbox"]
        cx, cy = cup["center_pixel"]
        draw_color = action_colors.get(action, (255, 255, 255))

        cv2.rectangle(output, (x1, y1), (x2, y2), draw_color, 3)
        cv2.circle(output, (cx, cy), 5, draw_color, -1)
        cv2.putText(output, f"Cup {cup['cup_id']} raw={raw_action} final={action}", (x1, max(22, y1 - 44)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, draw_color, 2)
        cv2.putText(output, f"conf={confidence:.2f} reason={override_reason}", (x1, max(42, y1 - 22)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, draw_color, 2)
        cv2.putText(
            output,
            f"d={cup.get('hand_distance', 999.0):.2f} touch={cup.get('touch_count', 0)} last={cup.get('last_touched_time', 0.0):.1f}s",
            (x1, max(62, y1 - 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            draw_color,
            2,
        )
        cv2.putText(
            output,
            (
                f"near={cup.get('time_near_cup', 0.0):.1f}s "
                f"release={cup.get('release_count', 0)} "
                f"stat={cup.get('stationary_time', 0.0):.1f}s "
                f"used={cup.get('used_cup_candidate', 0)}"
            ),
            (x1, min(output.shape[0] - 42, y2 + 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            draw_color,
            1,
        )
        if cup.get("is_active_cup", 0):
            cv2.putText(output, "ACTIVE", (x1, min(output.shape[0] - 20, y2 + 38)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 2)
        if cup.get("used_cup_candidate", 0):
            cv2.putText(output, "USED", (x2 - 58, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, draw_color, 2)
        if action != raw_action:
            cv2.putText(output, f"override={override_reason}", (x1, min(output.shape[0] - 8, y2 + 56)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, draw_color, 1)
        elif action == "CLEANUP_CANDIDATE":
            cv2.putText(output, "cleanup candidate", (x1, min(output.shape[0] - 8, y2 + 56)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, draw_color, 2)
        elif action == "IDLE":
            cv2.putText(output, "IDLE", (x1, min(output.shape[0] - 8, y2 + 56)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, draw_color, 2)

        if action == "ASK" and not ask_message_drawn:
            cv2.putText(output, "Will you clear this cup?", (10, output.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, draw_color, 3)
            ask_message_drawn = True
    return output


def write_live_eval_rows(log_path: Path, tracked_cups: list[dict], predictions: list[dict], user_state: dict, policy_mode: str, timestamp_now: float) -> None:
    fieldnames = [
        "timestamp",
        "cup_id",
        "raw_action",
        "final_action",
        "confidence",
        "override_reason",
        "policy_mode",
        "hand_distance",
        "last_touched_time",
        "touch_count",
        "user_present",
        "user_absent_time",
        "is_active_cup",
        "time_near_cup",
        "time_since_release",
        "release_count",
        "stationary_time",
        "used_cup_candidate",
        "idle_duration",
    ]
    prediction_map = {int(item["cup_id"]): item for item in predictions}
    file_exists = log_path.exists()
    with log_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for cup in tracked_cups:
            prediction = prediction_map.get(int(cup["cup_id"]), {})
            writer.writerow(
                {
                    "timestamp": round(timestamp_now, 3),
                    "cup_id": int(cup["cup_id"]),
                    "raw_action": prediction.get("raw_action", prediction.get("action", "IDLE")),
                    "final_action": prediction.get("action", "IDLE"),
                    "confidence": round(float(prediction.get("confidence", 0.0)), 4),
                    "override_reason": prediction.get("reason", "none"),
                    "policy_mode": policy_mode,
                    "hand_distance": round(float(cup.get("hand_distance", 999.0)), 4),
                    "last_touched_time": round(float(cup.get("last_touched_time", 999.0)), 4),
                    "touch_count": int(cup.get("touch_count", 0)),
                    "user_present": int(user_state.get("user_present", 0)),
                    "user_absent_time": round(float(user_state.get("user_absent_time", 0.0)), 4),
                    "is_active_cup": int(cup.get("is_active_cup", 0)),
                    "time_near_cup": round(float(cup.get("time_near_cup", 0.0)), 4),
                    "time_since_release": round(float(cup.get("time_since_release", 999.0)), 4),
                    "release_count": int(cup.get("release_count", 0)),
                    "stationary_time": round(float(cup.get("stationary_time", 0.0)), 4),
                    "used_cup_candidate": int(cup.get("used_cup_candidate", 0)),
                    "idle_duration": round(compute_idle_duration(cup), 4),
                }
            )


def build_interaction_tracker(config: dict) -> InteractionTracker:
    tracking_cfg = get_required(config, ["tracking"])
    return InteractionTracker(
        touch_threshold=float(tracking_cfg.get("touch_threshold", 0.12)),
        default_last_touched_time=float(tracking_cfg.get("default_last_touched_time", 999.0)),
        time_near_threshold=float(tracking_cfg.get("time_near_threshold", 0.8)),
        motion_distance_threshold=float(tracking_cfg.get("motion_distance_threshold", 0.03)),
        stationary_motion_threshold=float(tracking_cfg.get("stationary_motion_threshold", 0.01)),
        touch_count_used_threshold=int(tracking_cfg.get("touch_count_used_threshold", 1)),
    )


def run_perception_debug(args: argparse.Namespace, config: dict) -> int:
    camera_cfg = get_required(config, ["camera"])
    tracking_cfg = get_required(config, ["tracking"])
    camera_index = int(args.camera_index if args.camera_index is not None else camera_cfg.get("global_index", 0))
    width = int(args.width if args.width is not None else camera_cfg.get("width", 1280))
    height = int(args.height if args.height is not None else camera_cfg.get("height", 720))

    cap = open_camera(camera_index, args.backend, width, height)
    if not cap.isOpened():
        print(f"[ERROR] Could not open camera index {camera_index} with backend {args.backend}.")
        return 1

    hands_ctx, face_ctx = create_mediapipe_contexts()
    interaction_tracker = build_interaction_tracker(config)
    user_presence_tracker = UserPresenceTracker(absence_threshold=float(tracking_cfg.get("user_absence_threshold", 10.0)))
    previous_time = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[ERROR] Failed to read frame from camera.")
                return 1

            current_time = time.time()
            dt = max(0.0, current_time - previous_time)
            previous_time = current_time

            tracked_cups, hand, user_state = collect_live_perception(
                frame,
                config,
                hands_ctx,
                face_ctx,
                interaction_tracker,
                user_presence_tracker,
                dt,
            )

            debug_frame = draw_perception_debug(frame, tracked_cups, hand, user_state)
            cv2.imshow("Perception Debug", debug_frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        if hands_ctx is not None:
            hands_ctx.close()
        if face_ctx is not None:
            for detector in face_ctx:
                detector.close()
        cap.release()
        cv2.destroyAllWindows()
    return 0


def run_live_policy(args: argparse.Namespace, config: dict) -> int:
    if not args.model:
        print("[ERROR] --model is required when using --live-policy.")
        return 1

    camera_cfg = get_required(config, ["camera"])
    tracking_cfg = get_required(config, ["tracking"])
    camera_index = int(args.camera_index if args.camera_index is not None else camera_cfg.get("global_index", 0))
    width = int(args.width if args.width is not None else camera_cfg.get("width", 1280))
    height = int(args.height if args.height is not None else camera_cfg.get("height", 720))
    model_bundle = load_model_bundle(args.model)
    log_path = ensure_parent(args.log_live_eval) if args.log_live_eval else None

    cap = open_camera(camera_index, args.backend, width, height)
    if not cap.isOpened():
        print(f"[ERROR] Could not open camera index {camera_index} with backend {args.backend}.")
        return 1

    hands_ctx, face_ctx = create_mediapipe_contexts()
    interaction_tracker = build_interaction_tracker(config)
    user_presence_tracker = UserPresenceTracker(absence_threshold=float(tracking_cfg.get("user_absence_threshold", 10.0)))
    previous_time = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[ERROR] Failed to read frame from camera.")
                return 1

            current_time = time.time()
            dt = max(0.0, current_time - previous_time)
            previous_time = current_time

            tracked_cups, hand, user_state = collect_live_perception(
                frame,
                config,
                hands_ctx,
                face_ctx,
                interaction_tracker,
                user_presence_tracker,
                dt,
            )

            feature_rows = [
                {
                    "cup_id": int(cup["cup_id"]),
                    "x": float(cup.get("x_norm", 0.0)),
                    "y": float(cup.get("y_norm", 0.0)),
                    "hand_distance": float(cup.get("hand_distance", 999.0)),
                    "last_touched_time": float(cup.get("last_touched_time", 999.0)),
                    "touch_count": int(cup.get("touch_count", 0)),
                    "moved_recently": int(cup.get("moved_recently", 0)),
                    "distance_to_tray": float(cup.get("distance_to_tray", 0.0)),
                    "user_present": int(user_state.get("user_present", 0)),
                    "user_absent_time": float(user_state.get("user_absent_time", 0.0)),
                    "is_active_cup": int(cup.get("is_active_cup", 0)),
                    "time_near_cup": float(cup.get("time_near_cup", 0.0)),
                    "time_since_release": float(cup.get("time_since_release", 999.0)),
                    "release_count": int(cup.get("release_count", 0)),
                    "cup_motion_distance": float(cup.get("cup_motion_distance", 0.0)),
                    "stationary_time": float(cup.get("stationary_time", 0.0)),
                    "was_moved": int(cup.get("was_moved", 0)),
                    "used_cup_candidate": int(cup.get("used_cup_candidate", 0)),
                    "idle_candidate": int(cup.get("idle_candidate", 0)),
                }
                for cup in tracked_cups
            ]

            raw_predictions = predict_actions(feature_rows, model_bundle, config) if feature_rows else []
            if args.policy_mode == "model_only":
                predictions = apply_model_only(tracked_cups, raw_predictions) if raw_predictions else []
            elif args.policy_mode == "safety_guard":
                predictions = apply_live_safety_guard(tracked_cups, raw_predictions, user_state, hand, config) if raw_predictions else []
            else:
                predictions = apply_live_arbitration(tracked_cups, raw_predictions, user_state, hand, config) if raw_predictions else []

            if log_path is not None and predictions:
                write_live_eval_rows(log_path, tracked_cups, predictions, user_state, args.policy_mode, current_time)

            overlay = draw_live_policy_overlay(frame, tracked_cups, hand, user_state, predictions, args.policy_mode)
            cv2.imshow("Live Policy Inference", overlay)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        if hands_ctx is not None:
            hands_ctx.close()
        if face_ctx is not None:
            for detector in face_ctx:
                detector.close()
        cap.release()
        cv2.destroyAllWindows()
    return 0


def run_mock_policy(args: argparse.Namespace, config: dict) -> int:
    if not args.model:
        print("[ERROR] --model is required unless --debug-perception is used.")
        return 1

    model_bundle = load_model_bundle(args.model)
    cups = build_mock_cup_features()
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
        frame = choose_liquid_frame(cup_id)
        liquid_result = detect_liquid_local(frame, config)
        liquid_state = liquid_result["liquid_state"]
        print(f"[LOCAL VISION] cup {cup_id} -> {liquid_state} (confidence={liquid_result['confidence']:.2f})")

        if liquid_state == "EMPTY":
            mock_robot.clear_cup(cup_id)
        else:
            mock_robot.spill_safe_clear_cup(cup_id)
        print()
    return 0


def main() -> int:
    args = parse_args()
    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"[ERROR] {exc}")
        return 1

    if args.debug_perception:
        return run_perception_debug(args, config)
    if args.live_policy:
        return run_live_policy(args, config)
    return run_mock_policy(args, config)


if __name__ == "__main__":
    sys.exit(main())
