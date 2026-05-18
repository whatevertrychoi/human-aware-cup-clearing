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


def _extract_roi(frame: np.ndarray, config: dict) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    roi_cfg = get_required(config, ["local_liquid_detection", "roi"])
    h, w = frame.shape[:2]
    x = int(float(roi_cfg["x"]) * w)
    y = int(float(roi_cfg["y"]) * h)
    roi_w = int(float(roi_cfg["w"]) * w)
    roi_h = int(float(roi_cfg["h"]) * h)
    x2 = min(w, x + roi_w)
    y2 = min(h, y + roi_h)
    return frame[y:y2, x:x2], (x, y, x2, y2)


def detect_liquid_local(frame: np.ndarray, config: dict) -> dict:
    local_cfg = get_required(config, ["local_liquid_detection"])
    roi_frame, _ = _extract_roi(frame, config)
    hsv = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)

    total_pixels = max(1, roi_frame.shape[0] * roi_frame.shape[1])
    combined_mask = np.zeros((roi_frame.shape[0], roi_frame.shape[1]), dtype=np.uint8)
    for color_cfg in get_required(local_cfg, ["liquid_colors"]).values():
        combined_mask = cv2.bitwise_or(combined_mask, _mask_from_color(hsv, color_cfg))

    gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
    color_ratio = float(np.count_nonzero(combined_mask)) / float(total_pixels)
    dark_ratio = float(np.count_nonzero(gray < 70)) / float(total_pixels)

    min_color_ratio = float(get_required(local_cfg, ["min_color_ratio"]))
    min_dark_ratio = float(get_required(local_cfg, ["min_dark_ratio"]))
    non_empty = color_ratio >= min_color_ratio or dark_ratio >= min_dark_ratio
    confidence = min(1.0, max(color_ratio / max(min_color_ratio, 1e-6), dark_ratio / max(min_dark_ratio, 1e-6)))

    return {
        "liquid_state": "NON_EMPTY" if non_empty else "EMPTY",
        "confidence": float(confidence),
        "color_ratio": color_ratio,
        "dark_ratio": dark_ratio,
    }


def make_mock_liquid_frame(fill: bool, width: int = 640, height: int = 480) -> np.ndarray:
    frame = np.full((height, width, 3), 220, dtype=np.uint8)
    center = (width // 2, height // 2)
    cv2.circle(frame, center, 110, (245, 245, 245), -1)
    cv2.circle(frame, center, 100, (255, 255, 255), -1)
    if fill:
        cv2.ellipse(frame, center, (80, 55), 0, 0, 360, (0, 80, 220), -1)
    return frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect cup interior liquid state from a local camera.")
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

    cap = cv2.VideoCapture(int(camera_cfg.get("local_index", 1)))
    if not cap.isOpened():
        print("[ERROR] Could not open local camera.")
        return 1

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[ERROR] Failed to read frame from camera.")
                return 1

            result = detect_liquid_local(frame, config)
            _, (x1, y1, x2, y2) = _extract_roi(frame, config)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            label = (
                f"{result['liquid_state']} "
                f"c={result['color_ratio']:.2f} d={result['dark_ratio']:.2f}"
            )
            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow("Local Liquid Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())

