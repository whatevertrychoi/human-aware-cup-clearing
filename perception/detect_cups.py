from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from project_utils import ConfigError, get_required, load_config


def _mask_from_color(hsv_frame: np.ndarray, color_cfg: dict) -> np.ndarray:
    if "lower1" in color_cfg and "upper1" in color_cfg:
        mask1 = cv2.inRange(hsv_frame, np.array(color_cfg["lower1"]), np.array(color_cfg["upper1"]))
        mask2 = cv2.inRange(hsv_frame, np.array(color_cfg["lower2"]), np.array(color_cfg["upper2"]))
        return cv2.bitwise_or(mask1, mask2)
    return cv2.inRange(hsv_frame, np.array(color_cfg["lower"]), np.array(color_cfg["upper"]))


def detect_cups(frame: np.ndarray, config: dict) -> list[dict]:
    cup_cfg = get_required(config, ["cup_detection"])
    colors = get_required(cup_cfg, ["colors"])
    min_area = int(get_required(cup_cfg, ["min_area"]))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    detections: list[dict] = []

    for color_name, color_cfg in colors.items():
        mask = _mask_from_color(hsv, color_cfg)
        mask = cv2.medianBlur(mask, 5)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        largest = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(largest))
        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(largest)
        detections.append(
            {
                "cup_id": int(color_cfg["cup_id"]),
                "color": color_name,
                "bbox": [int(x), int(y), int(x + w), int(y + h)],
                "center_pixel": [int(x + w / 2), int(y + h / 2)],
                "area": area,
            }
        )

    return sorted(detections, key=lambda item: item["cup_id"])


def draw_detections(frame: np.ndarray, detections: list[dict]) -> np.ndarray:
    output = frame.copy()
    for item in detections:
        x1, y1, x2, y2 = item["bbox"]
        label = f"Cup {item['cup_id']} ({item['color']})"
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(output, label, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect green, red, and blue cups from a webcam.")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = load_config(args.config)
        camera_cfg = get_required(config, ["camera"])
    except ConfigError as exc:
        print(f"[ERROR] {exc}")
        return 1

    cap = cv2.VideoCapture(int(camera_cfg.get("global_index", 0)))
    if not cap.isOpened():
        print("[ERROR] Could not open global camera.")
        return 1

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[ERROR] Failed to read frame from camera.")
                return 1

            detections = detect_cups(frame, config)
            debug = draw_detections(frame, detections)
            cv2.imshow("Cup Detection", debug)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())

