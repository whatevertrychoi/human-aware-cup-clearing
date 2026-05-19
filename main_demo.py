from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from perception.detect_cups import detect_cups
from perception.detect_hand import detect_hand
from perception.detect_liquid_local import detect_liquid_local, make_mock_liquid_frame
from perception.detect_user_presence import detect_user_presence
from policy.infer_policy import load_model_bundle, predict_actions
from project_utils import ConfigError, get_required, load_config
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
    parser = argparse.ArgumentParser(description="Run the cup clearing mock demo.")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--model",
        default=None,
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
    parser.add_argument("--camera-index", type=int, default=None, help="Camera index for live perception debug")
    parser.add_argument("--backend", default="auto", choices=["auto", "dshow", "msmf"], help="OpenCV backend")
    parser.add_argument("--width", type=int, default=None, help="Override capture width")
    parser.add_argument("--height", type=int, default=None, help="Override capture height")
    parser.add_argument("--debug-perception", action="store_true", help="Run live cup/hand/user perception debug view")
    return parser.parse_args()


def open_camera(camera_index: int, backend: str, width: int, height: int):
    backend_id = BACKEND_MAP[backend]
    cap = cv2.VideoCapture(camera_index) if backend_id is None else cv2.VideoCapture(camera_index, backend_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def draw_perception_debug(frame, cups: list[dict], hand: dict, user_state: dict) -> np.ndarray:
    output = frame.copy()
    if hand.get("hand_visible") and hand.get("hand_center") is not None:
        hx, hy = hand["hand_center"]
        cv2.circle(output, (hx, hy), 10, (0, 255, 255), -1)
        cv2.putText(output, "hand", (hx + 10, hy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        for cup in cups:
            cx, cy = cup["center_pixel"]
            cv2.line(output, (hx, hy), (cx, cy), (0, 255, 255), 1)
            cv2.putText(
                output,
                f"d={cup.get('hand_distance', 999.0):.2f}",
                (cx + 10, cy - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 255),
                1,
            )

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
        cv2.putText(
            output,
            line,
            (10, 30 + (index * 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )

    for cup in cups:
        x1, y1, x2, y2 = cup["bbox"]
        cx, cy = cup["center_pixel"]
        label = (
            f"Cup {cup['cup_id']} {cup['color']} "
            f"touches={cup.get('touch_count', 0)} "
            f"last={cup.get('last_touched_time', 0.0):.1f}s"
        )
        color_map = {
            "green": (0, 220, 0),
            "red": (0, 0, 255),
            "blue": (255, 120, 0),
        }
        draw_color = color_map.get(cup["color"], (0, 255, 0))
        cv2.rectangle(output, (x1, y1), (x2, y2), draw_color, 2)
        cv2.circle(output, (cx, cy), 5, draw_color, -1)
        cv2.putText(output, label, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, draw_color, 2)
    return output


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

    try:
        import mediapipe as mp
    except ImportError:
        mp = None

    hands_ctx = None
    face_ctx = None
    if mp is not None:
        hands_ctx = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.35,
            min_tracking_confidence=0.35,
        )
        face_ctx = [
            mp.solutions.face_detection.FaceDetection(
                model_selection=0,
                min_detection_confidence=0.35,
            ),
            mp.solutions.face_detection.FaceDetection(
                model_selection=1,
                min_detection_confidence=0.35,
            ),
        ]

    interaction_tracker = InteractionTracker(
        touch_threshold=float(tracking_cfg.get("touch_threshold", 0.12)),
        default_last_touched_time=float(tracking_cfg.get("default_last_touched_time", 999.0)),
    )
    user_presence_tracker = UserPresenceTracker(
        absence_threshold=float(tracking_cfg.get("user_absence_threshold", 10.0))
    )
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

            cups = detect_cups(frame, config)
            for cup in cups:
                cx, cy = cup["center_pixel"]
                cup["x"] = float(cx)
                cup["y"] = float(cy)
                cup["frame_width"] = frame.shape[1]
                cup["frame_height"] = frame.shape[0]

            hand = detect_hand(frame, hands=hands_ctx)
            user_presence = detect_user_presence(frame, face_detector=face_ctx, hand_detection=hand)
            user_state = user_presence_tracker.update(bool(user_presence["user_present"]), dt)
            user_state["face_center"] = user_presence.get("face_center")
            user_state["confidence"] = user_presence.get("confidence", 0.0)
            user_state["source"] = user_presence.get("source", "none")
            tracked_cups = interaction_tracker.update(cups, hand, dt)

            debug_frame = draw_perception_debug(frame, tracked_cups, hand, user_state)
            cv2.imshow("Perception Debug", debug_frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
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


def main() -> int:
    args = parse_args()
    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"[ERROR] {exc}")
        return 1

    if args.debug_perception:
        return run_perception_debug(args, config)

    if not args.model:
        print("[ERROR] --model is required unless --debug-perception is used.")
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
