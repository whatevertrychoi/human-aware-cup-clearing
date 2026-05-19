from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from perception.detect_cups import detect_cups
from perception.detect_hand import detect_hand
from perception.detect_user_presence import detect_user_presence
from policy.expert_policy import expert_high_level_policy
from project_utils import ConfigError, ensure_parent, get_required, load_config
from tracking.interaction_tracker import InteractionTracker
from tracking.user_presence_tracker import UserPresenceTracker


BACKEND_MAP = {
    "auto": None,
    "dshow": getattr(cv2, "CAP_DSHOW", None),
    "msmf": getattr(cv2, "CAP_MSMF", None),
}

FIELDNAMES = [
    "scene_id",
    "timestamp",
    "cup_id",
    "x",
    "y",
    "hand_distance",
    "last_touched_time",
    "touch_count",
    "moved_recently",
    "distance_to_tray",
    "user_present",
    "user_absent_time",
    "active_cup_id",
    "is_active_cup",
    "time_near_cup",
    "time_since_release",
    "release_count",
    "cup_motion_distance",
    "stationary_time",
    "was_moved",
    "used_cup_candidate",
    "label",
]
MOCK_DATASET_NAME = "dataset_decision.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record real interaction dataset from USB webcam perception.")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config.yaml")
    parser.add_argument("--camera-index", type=int, default=None, help="Override camera index from config")
    parser.add_argument("--backend", default="auto", choices=["auto", "dshow", "msmf"], help="OpenCV backend")
    parser.add_argument("--width", type=int, default=None, help="Override capture width")
    parser.add_argument("--height", type=int, default=None, help="Override capture height")
    parser.add_argument("--out", required=True, help="Output CSV path")
    parser.add_argument("--interval", type=float, default=0.5, help="Minimum seconds between saved scenes")
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to an existing output CSV instead of refusing to run",
    )
    return parser.parse_args()


def open_camera(camera_index: int, backend: str, width: int, height: int) -> cv2.VideoCapture:
    backend_id = BACKEND_MAP[backend]
    capture = cv2.VideoCapture(camera_index) if backend_id is None else cv2.VideoCapture(camera_index, backend_id)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return capture


def get_next_scene_id(csv_path: Path) -> int:
    if not csv_path.exists():
        return 0

    max_scene_id = -1
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                max_scene_id = max(max_scene_id, int(row["scene_id"]))
            except (KeyError, TypeError, ValueError):
                continue
    return max_scene_id + 1


def validate_output_path(output_path: Path, append: bool) -> None:
    if output_path.name == MOCK_DATASET_NAME:
        raise ValueError(
            "Refusing to write real interaction data into data/processed/dataset_decision.csv. "
            "That file is reserved for the v0.1 mock dataset. "
            "Use a scene-specific file such as interaction_green.csv or interaction_clutter.csv."
        )

    if output_path.exists() and not append:
        raise ValueError(
            f"Output file already exists: {output_path}\n"
            "Use a different scene-specific CSV path or re-run with --append if you intentionally want to keep adding rows."
        )


def normalize_center(center_pixel: list[int], frame_width: int, frame_height: int) -> tuple[float, float]:
    cx, cy = center_pixel
    return float(cx) / float(max(frame_width, 1)), float(cy) / float(max(frame_height, 1))


def compute_distance_to_tray(x_norm: float, y_norm: float, config: dict) -> float:
    tray_position = get_required(config, ["robot", "tray_position"])
    tray_x = float(tray_position["x"])
    tray_y = float(tray_position["y"])
    return float(((x_norm - tray_x) ** 2 + (y_norm - tray_y) ** 2) ** 0.5)


def make_dataset_rows(
    cups: list[dict],
    user_state: dict,
    scene_id: int,
    timestamp: float,
    frame_width: int,
    frame_height: int,
    config: dict,
) -> list[dict]:
    rows: list[dict] = []
    for cup in cups:
        x_norm, y_norm = normalize_center(cup["center_pixel"], frame_width, frame_height)
        feature_row = {
            "scene_id": scene_id,
            "timestamp": round(timestamp, 3),
            "cup_id": int(cup["cup_id"]),
            "x": round(x_norm, 4),
            "y": round(y_norm, 4),
            "hand_distance": round(float(cup.get("hand_distance", 999.0)), 4),
            "last_touched_time": round(float(cup.get("last_touched_time", 999.0)), 3),
            "touch_count": int(cup.get("touch_count", 0)),
            "moved_recently": int(cup.get("moved_recently", 0)),
            "distance_to_tray": round(compute_distance_to_tray(x_norm, y_norm, config), 4),
            "user_present": int(user_state.get("user_present", 0)),
            "user_absent_time": round(float(user_state.get("user_absent_time", 0.0)), 3),
            "active_cup_id": int(cup["active_cup_id"]) if cup.get("active_cup_id") is not None else -1,
            "is_active_cup": int(cup.get("is_active_cup", 0)),
            "time_near_cup": round(float(cup.get("time_near_cup", 0.0)), 3),
            "time_since_release": round(float(cup.get("time_since_release", 999.0)), 3),
            "release_count": int(cup.get("release_count", 0)),
            "cup_motion_distance": round(float(cup.get("cup_motion_distance", 0.0)), 4),
            "stationary_time": round(float(cup.get("stationary_time", 0.0)), 3),
            "was_moved": int(cup.get("was_moved", 0)),
            "used_cup_candidate": int(cup.get("used_cup_candidate", 0)),
        }
        feature_row["label"] = expert_high_level_policy(feature_row, config)
        rows.append(feature_row)
    return rows


def append_rows(csv_path: Path, rows: list[dict]) -> None:
    file_exists = csv_path.exists()
    with csv_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)


def draw_debug_overlay(
    frame,
    cups: list[dict],
    hand: dict,
    user_state: dict,
    sample_count: int,
    scene_id: int,
    interval: float,
) -> None:
    lines = [
        f"samples={sample_count}",
        f"next_scene_id={scene_id}",
        f"user_present={int(user_state.get('user_present', 0))}",
        f"user_absent_time={user_state.get('user_absent_time', 0.0):.1f}s",
        f"hand_visible={int(hand.get('hand_visible', False))}",
        f"save_interval={interval:.2f}s",
        "q: quit",
    ]
    for index, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (10, 30 + (index * 24)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )

    if hand.get("hand_visible") and hand.get("hand_center") is not None:
        hx, hy = hand["hand_center"]
        cv2.circle(frame, (hx, hy), 10, (0, 255, 255), -1)
        cv2.putText(frame, "hand", (hx + 10, hy), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

    if user_state.get("face_center") is not None:
        fx, fy = user_state["face_center"]
        cv2.circle(frame, (fx, fy), 10, (255, 0, 255), -1)
        cv2.putText(frame, "user", (fx + 10, fy), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 2)

    color_map = {
        "green": (0, 220, 0),
        "red": (0, 0, 255),
        "blue": (255, 120, 0),
    }
    for cup in cups:
        x1, y1, x2, y2 = cup["bbox"]
        cx, cy = cup["center_pixel"]
        draw_color = color_map.get(cup["color"], (0, 255, 0))
        label = (
            f"Cup {cup['cup_id']} {cup['color']} "
            f"d={cup.get('hand_distance', 999.0):.2f} "
            f"touch={cup.get('touch_count', 0)} "
            f"last={cup.get('last_touched_time', 0.0):.1f}s"
        )
        cv2.rectangle(frame, (x1, y1), (x2, y2), draw_color, 2)
        cv2.circle(frame, (cx, cy), 5, draw_color, -1)
        cv2.putText(frame, label, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, draw_color, 2)
        cv2.putText(
            frame,
            (
                f"near={cup.get('time_near_cup', 0.0):.1f}s "
                f"rel={cup.get('release_count', 0)} "
                f"stat={cup.get('stationary_time', 0.0):.1f}s "
                f"used={cup.get('used_cup_candidate', 0)}"
            ),
            (x1, min(frame.shape[0] - 10, y2 + 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            draw_color,
            1,
        )
        if cup.get("is_active_cup", 0):
            cv2.putText(frame, "ACTIVE", (x1, min(frame.shape[0] - 28, y2 + 36)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 2)


def main() -> int:
    args = parse_args()
    try:
        config = load_config(args.config)
        camera_cfg = get_required(config, ["camera"])
        tracking_cfg = get_required(config, ["tracking"])
    except ConfigError as exc:
        print(f"[ERROR] {exc}")
        return 1

    camera_index = int(args.camera_index if args.camera_index is not None else camera_cfg.get("global_index", 0))
    width = int(args.width if args.width is not None else camera_cfg.get("width", 1280))
    height = int(args.height if args.height is not None else camera_cfg.get("height", 720))

    output_path = ensure_parent(args.out)
    try:
        validate_output_path(output_path, args.append)
    except ValueError as exc:
        print(f"[ERROR] {exc}")
        return 1

    scene_id = get_next_scene_id(output_path)
    sample_count = 0

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
    if mp is not None and hasattr(mp, "solutions"):
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
        time_near_threshold=float(tracking_cfg.get("time_near_threshold", 0.8)),
        motion_distance_threshold=float(tracking_cfg.get("motion_distance_threshold", 0.03)),
        stationary_motion_threshold=float(tracking_cfg.get("stationary_motion_threshold", 0.01)),
        touch_count_used_threshold=int(tracking_cfg.get("touch_count_used_threshold", 1)),
    )
    user_presence_tracker = UserPresenceTracker(
        absence_threshold=float(tracking_cfg.get("user_absence_threshold", 10.0))
    )

    previous_time = time.time()
    last_saved_time = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[ERROR] Failed to read frame from camera.")
                return 1

            now = time.time()
            dt = max(0.0, now - previous_time)
            previous_time = now

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

            tracked_cups = interaction_tracker.update(cups, hand, dt)

            if tracked_cups and (now - last_saved_time) >= args.interval:
                rows = make_dataset_rows(
                    tracked_cups,
                    user_state,
                    scene_id=scene_id,
                    timestamp=now,
                    frame_width=frame.shape[1],
                    frame_height=frame.shape[0],
                    config=config,
                )
                append_rows(output_path, rows)
                sample_count += len(rows)
                print(f"[SAVE] scene_id={scene_id} rows={len(rows)} total_samples={sample_count}")
                scene_id += 1
                last_saved_time = now

            debug = frame.copy()
            draw_debug_overlay(debug, tracked_cups, hand, user_state, sample_count, scene_id, args.interval)
            cv2.imshow("Interaction Dataset Recorder", debug)
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

    print(f"Saved dataset to {output_path}")
    print(f"Final sample count: {sample_count}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
