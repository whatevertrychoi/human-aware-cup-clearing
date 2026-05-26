from __future__ import annotations

"""HSV-based global cup detector used by the top-down policy pipeline.

This detector is intentionally lightweight and explainable. It finds one best
candidate per configured color and returns stable metadata such as:
- cup ID
- bbox
- center pixel
- contour area

Those outputs become the shared perception backbone for tracking and policy.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from project_utils import ConfigError, get_required, load_config


BACKEND_MAP = {
    "auto": None,
    "dshow": getattr(cv2, "CAP_DSHOW", None),
    "msmf": getattr(cv2, "CAP_MSMF", None),
}


def _mask_from_color(hsv_frame: np.ndarray, color_cfg: dict) -> np.ndarray:
    """Build a binary mask for one color range, including split red ranges."""
    if "lower1" in color_cfg and "upper1" in color_cfg:
        mask1 = cv2.inRange(hsv_frame, np.array(color_cfg["lower1"]), np.array(color_cfg["upper1"]))
        mask2 = cv2.inRange(hsv_frame, np.array(color_cfg["lower2"]), np.array(color_cfg["upper2"]))
        return cv2.bitwise_or(mask1, mask2)
    return cv2.inRange(hsv_frame, np.array(color_cfg["lower"]), np.array(color_cfg["upper"]))


def _preprocess_mask(mask: np.ndarray) -> np.ndarray:
    """Remove small speckles and fill small holes before contour search."""
    kernel_open = np.ones((5, 5), dtype=np.uint8)
    kernel_close = np.ones((7, 7), dtype=np.uint8)
    cleaned = cv2.medianBlur(mask, 5)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel_open)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_close)
    return cleaned


def _score_contour(contour, cup_cfg: dict, frame_area: float) -> tuple[float, dict] | None:
    """Score a contour and return metadata if it still looks cup-like."""
    min_area = float(cup_cfg.get("min_area", 500))
    max_area = float(cup_cfg.get("max_area", frame_area))
    min_box_size = int(cup_cfg.get("min_box_size", 18))
    min_fill_ratio = float(cup_cfg.get("min_fill_ratio", 0.22))
    min_aspect_ratio = float(cup_cfg.get("min_aspect_ratio", 0.22))
    max_aspect_ratio = float(cup_cfg.get("max_aspect_ratio", 4.2))
    min_solidity = float(cup_cfg.get("min_solidity", 0.0))
    min_circularity = float(cup_cfg.get("min_circularity", 0.0))
    max_bbox_area_ratio = float(cup_cfg.get("max_bbox_area_ratio", 1.0))
    min_score = float(cup_cfg.get("min_score", float("-inf")))

    area = float(cv2.contourArea(contour))
    if area < min_area:
        return None
    if area > max_area:
        return None

    x, y, w, h = cv2.boundingRect(contour)
    bbox_area = float(max(w * h, 1))
    if bbox_area / float(max(frame_area, 1.0)) > max_bbox_area_ratio:
        return None

    fill_ratio = area / bbox_area
    aspect_ratio = float(w) / float(max(h, 1))
    extent_penalty = abs(np.log(max(aspect_ratio, 1e-6)))
    perimeter = float(cv2.arcLength(contour, True))
    circularity = (
        float((4.0 * np.pi * area) / max(perimeter * perimeter, 1e-6))
        if perimeter > 0.0
        else 0.0
    )
    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))
    solidity = area / hull_area if hull_area > 0.0 else 0.0

    # Cups can appear upright, lying down, or upside down, so the filter should
    # be permissive while still rejecting thin tape-like regions.
    if w < min_box_size or h < min_box_size:
        return None
    if fill_ratio < min_fill_ratio:
        return None
    if aspect_ratio > max_aspect_ratio or aspect_ratio < min_aspect_ratio:
        return None
    if solidity < min_solidity:
        return None
    if circularity < min_circularity:
        return None

    score = (
        area
        + (fill_ratio * 500.0)
        + (solidity * 300.0)
        + (circularity * 250.0)
        - (extent_penalty * 120.0)
    )
    if score < min_score:
        return None

    metadata = {
        "area": area,
        "bbox": [int(x), int(y), int(x + w), int(y + h)],
        "center_pixel": [int(x + w / 2), int(y + h / 2)],
        "fill_ratio": fill_ratio,
        "aspect_ratio": aspect_ratio,
        "solidity": solidity,
        "circularity": circularity,
    }
    return score, metadata


def detect_cups(frame: np.ndarray, config: dict) -> list[dict]:
    """Return one best detection per configured cup color."""
    cup_cfg = get_required(config, ["cup_detection"])
    colors = get_required(cup_cfg, ["colors"])
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    frame_area = float(frame.shape[0] * frame.shape[1])
    detections: list[dict] = []

    for color_name, color_cfg in colors.items():
        mask = _preprocess_mask(_mask_from_color(hsv, color_cfg))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        contour_cfg = dict(cup_cfg)
        contour_cfg.update(color_cfg.get("detector_overrides", {}))

        best_candidate = None
        for contour in contours:
            candidate = _score_contour(contour, contour_cfg, frame_area)
            if candidate is None:
                continue
            if best_candidate is None or candidate[0] > best_candidate[0]:
                best_candidate = candidate

        if best_candidate is None:
            continue

        _, metadata = best_candidate
        detections.append(
            {
                "cup_id": int(color_cfg["cup_id"]),
                "color": color_name,
                "bbox": metadata["bbox"],
                "center_pixel": metadata["center_pixel"],
                "area": metadata["area"],
            }
        )

    return sorted(detections, key=lambda item: item["cup_id"])


def build_mask_debug_view(frame: np.ndarray, config: dict) -> np.ndarray:
    cup_cfg = get_required(config, ["cup_detection"])
    colors = get_required(cup_cfg, ["colors"])
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    debug_panels = []
    for color_name, color_cfg in colors.items():
        mask = _preprocess_mask(_mask_from_color(hsv, color_cfg))
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        cv2.putText(mask_bgr, color_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        debug_panels.append(mask_bgr)

    if not debug_panels:
        return np.zeros_like(frame)

    return cv2.hconcat(debug_panels)


def draw_detections(frame: np.ndarray, detections: list[dict]) -> np.ndarray:
    output = frame.copy()
    for item in detections:
        x1, y1, x2, y2 = item["bbox"]
        cx, cy = item["center_pixel"]
        label = f"Cup {item['cup_id']} | {item['color']} | area={int(item['area'])}"
        color_map = {
            "green": (0, 220, 0),
            "red": (0, 0, 255),
            "blue": (255, 120, 0),
        }
        draw_color = color_map.get(item["color"], (0, 255, 0))
        cv2.rectangle(output, (x1, y1), (x2, y2), draw_color, 2)
        cv2.circle(output, (cx, cy), 5, draw_color, -1)
        cv2.putText(output, label, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(output, f"({cx}, {cy})", (x1, min(output.shape[0] - 10, y2 + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, draw_color, 2)
    cv2.putText(
        output,
        f"detections={len(detections)} | q: quit",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect green, red, and blue cups from a webcam.")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--camera-index", type=int, default=None, help="Override camera index from config")
    parser.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "dshow", "msmf"],
        help="Preferred OpenCV backend on Windows",
    )
    parser.add_argument("--width", type=int, default=None, help="Override capture width")
    parser.add_argument("--height", type=int, default=None, help="Override capture height")
    parser.add_argument("--show-mask-debug", action="store_true", help="Show HSV mask debug window")
    return parser.parse_args()


def open_camera(camera_index: int, backend: str, width: int, height: int):
    backend_id = BACKEND_MAP[backend]
    capture = cv2.VideoCapture(camera_index) if backend_id is None else cv2.VideoCapture(camera_index, backend_id)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return capture


def main() -> int:
    args = parse_args()
    try:
        config = load_config(args.config)
        camera_cfg = get_required(config, ["camera"])
    except ConfigError as exc:
        print(f"[ERROR] {exc}")
        return 1

    camera_index = int(args.camera_index if args.camera_index is not None else camera_cfg.get("global_index", 0))
    width = int(args.width if args.width is not None else camera_cfg.get("width", 1280))
    height = int(args.height if args.height is not None else camera_cfg.get("height", 720))

    cap = open_camera(camera_index, args.backend, width, height)
    if not cap.isOpened():
        print(f"[ERROR] Could not open global camera index {camera_index} with backend {args.backend}.")
        print("Try:")
        print(f"python perception/detect_cups.py --config {args.config} --camera-index 1 --backend dshow")
        print(f"python perception/detect_cups.py --config {args.config} --camera-index 1 --backend msmf")
        return 1

    print(f"[INFO] Opened global camera index {camera_index} with backend {args.backend}.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[ERROR] Failed to read frame from camera.")
                return 1

            detections = detect_cups(frame, config)
            debug = draw_detections(frame, detections)
            cv2.imshow("Cup Detection", debug)
            if args.show_mask_debug:
                mask_view = build_mask_debug_view(frame, config)
                cv2.imshow("Cup Mask Debug", mask_view)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
