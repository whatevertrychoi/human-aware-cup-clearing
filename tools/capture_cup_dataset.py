from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from project_utils import ConfigError, ensure_directory, load_config


CUP_KEYS = {
    "g": ("green", "green"),
    "r": ("red", "red"),
    "b": ("blue", "blue"),
}

BACKEND_CANDIDATES = [
    ("AUTO", None),
    ("DSHOW", getattr(cv2, "CAP_DSHOW", None)),
    ("MSMF", getattr(cv2, "CAP_MSMF", None)),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture USB webcam images for cup detection dataset collection.")
    parser.add_argument("--camera-index", type=int, default=0, help="OpenCV camera index for the USB webcam")
    parser.add_argument("--out-dir", type=str, default="data/raw", help="Base output directory for captured images")
    parser.add_argument("--width", type=int, default=1280, help="Camera capture width")
    parser.add_argument("--height", type=int, default=720, help="Camera capture height")
    parser.add_argument("--fps", type=int, default=30, help="Requested camera FPS")
    parser.add_argument(
        "--backend",
        type=str,
        default="auto",
        choices=["auto", "dshow", "msmf"],
        help="Preferred OpenCV backend on Windows",
    )
    parser.add_argument(
        "--scan-max-index",
        type=int,
        default=5,
        help="Maximum index to suggest when scanning for available cameras",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Optional config path used as a fallback for camera resolution values",
    )
    return parser.parse_args()


def load_camera_defaults(config_path: str) -> dict:
    try:
        config = load_config(config_path)
    except ConfigError:
        return {}
    return config.get("camera", {}) if isinstance(config, dict) else {}


def get_existing_count(directory: Path, prefix: str) -> int:
    return len(list(directory.glob(f"{prefix}_*.jpg")))


def get_next_index(directory: Path, prefix: str) -> int:
    max_index = 0
    for image_path in directory.glob(f"{prefix}_*.jpg"):
        stem = image_path.stem
        try:
            max_index = max(max_index, int(stem.split("_")[-1]))
        except ValueError:
            continue
    return max_index + 1


def build_output_dirs(base_dir: str) -> dict[str, Path]:
    root = Path(base_dir)
    return {
        color: ensure_directory(root / color)
        for color in ["green", "red", "blue"]
    }


def open_camera_with_backend(
    camera_index: int,
    width: int,
    height: int,
    fps: int,
    backend_id: int | None,
) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(camera_index) if backend_id is None else cv2.VideoCapture(camera_index, backend_id)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    capture.set(cv2.CAP_PROP_FPS, fps)
    return capture


def choose_backend_order(preferred_backend: str) -> list[tuple[str, int | None]]:
    if preferred_backend == "dshow":
        return [
            ("DSHOW", getattr(cv2, "CAP_DSHOW", None)),
            ("AUTO", None),
            ("MSMF", getattr(cv2, "CAP_MSMF", None)),
        ]
    if preferred_backend == "msmf":
        return [
            ("MSMF", getattr(cv2, "CAP_MSMF", None)),
            ("AUTO", None),
            ("DSHOW", getattr(cv2, "CAP_DSHOW", None)),
        ]
    return BACKEND_CANDIDATES


def try_open_camera(
    camera_index: int,
    width: int,
    height: int,
    fps: int,
    preferred_backend: str,
):
    attempts: list[tuple[str, str]] = []
    for backend_name, backend_id in choose_backend_order(preferred_backend):
        if backend_name != "AUTO" and backend_id is None:
            attempts.append((backend_name, "backend not available in this OpenCV build"))
            continue

        capture = open_camera_with_backend(
            camera_index,
            width,
            height,
            fps,
            backend_id,
        )
        if not capture.isOpened():
            capture.release()
            attempts.append((backend_name, "open failed"))
            continue

        ok, _ = capture.read()
        if not ok:
            capture.release()
            attempts.append((backend_name, "opened but frame read failed"))
            continue

        return capture, backend_name, attempts

    return None, None, attempts


def scan_camera_indices(
    width: int,
    height: int,
    fps: int,
    preferred_backend: str,
    scan_max_index: int,
) -> list[str]:
    found: list[str] = []
    for index in range(scan_max_index + 1):
        capture, backend_name, _ = try_open_camera(
            index,
            width,
            height,
            fps,
            preferred_backend,
        )
        if capture is not None:
            found.append(f"index {index} via {backend_name}")
            capture.release()
    return found


def draw_overlay(frame, counts: dict[str, int]) -> None:
    count_text = f"Green: {counts['green']} / Red: {counts['red']} / Blue: {counts['blue']}"
    help_text = "Press g: save green | r: save red | b: save blue | q or ESC: quit"
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 70), (0, 0, 0), -1)
    cv2.putText(frame, count_text, (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, help_text, (20, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def save_frame(frame, directory: Path, prefix: str) -> Path:
    image_index = get_next_index(directory, prefix)
    output_path = directory / f"{prefix}_{image_index:04d}.jpg"
    cv2.imwrite(str(output_path), frame)
    return output_path


def main() -> int:
    args = parse_args()
    camera_defaults = load_camera_defaults(args.config)
    width = args.width or int(camera_defaults.get("width", 1280))
    height = args.height or int(camera_defaults.get("height", 720))
    fps = args.fps or 30
    output_dirs = build_output_dirs(args.out_dir)
    counts = {
        color: get_existing_count(directory, color)
        for color, directory in output_dirs.items()
    }

    capture, backend_name, attempts = try_open_camera(
        args.camera_index,
        width,
        height,
        fps,
        args.backend,
    )
    if capture is None:
        print(f"[ERROR] Could not open camera index {args.camera_index}.")
        print("Tried backends:")
        for attempt_backend, reason in attempts:
            print(f"  - {attempt_backend}: {reason}")
        print("Try another index:")
        print(f"python tools/capture_cup_dataset.py --camera-index {args.camera_index + 1}")
        print(f"python tools/capture_cup_dataset.py --camera-index {args.camera_index + 2}")
        print("Try an explicit backend:")
        print(f"python tools/capture_cup_dataset.py --camera-index {args.camera_index} --backend dshow")
        print(f"python tools/capture_cup_dataset.py --camera-index {args.camera_index} --backend msmf")
        available_cameras = scan_camera_indices(
            width,
            height,
            fps,
            args.backend,
            args.scan_max_index,
        )
        if available_cameras:
            print("Detected working camera candidates:")
            for item in available_cameras:
                print(f"  - {item}")
        else:
            print(
                "No working cameras were detected in the quick scan. "
                "Check whether another app is already using the webcam, whether the USB cable is connected, "
                "and whether Windows Camera privacy permissions are enabled."
            )
        return 1

    print(f"[INFO] Opened camera index {args.camera_index} with backend {backend_name}.")

    window_name = "Cup Dataset Capture"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, width, height)

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                print("[ERROR] Failed to read frame from the USB webcam.")
                return 1

            display = frame.copy()
            draw_overlay(display, counts)
            cv2.imshow(window_name, display)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

            key_char = chr(key).lower() if key != 255 else ""
            if key_char in CUP_KEYS:
                color_name, prefix = CUP_KEYS[key_char]
                saved_path = save_frame(frame, output_dirs[color_name], prefix)
                counts[color_name] += 1
                print(f"[SAVE] {saved_path}")
    finally:
        capture.release()
        cv2.destroyAllWindows()

    print("Final counts:")
    print(f"green: {counts['green']}")
    print(f"red: {counts['red']}")
    print(f"blue: {counts['blue']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
