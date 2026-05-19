from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

try:
    import mediapipe as mp
except ImportError:  # pragma: no cover - runtime fallback
    mp = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BACKEND_MAP = {
    "auto": None,
    "dshow": getattr(cv2, "CAP_DSHOW", None),
    "msmf": getattr(cv2, "CAP_MSMF", None),
}

MEDIAPIPE_SOLUTIONS_ERROR = """[ERROR] This script requires MediaPipe Solutions API.
Current mediapipe package does not provide mp.solutions.
Use Python 3.12 environment with mediapipe==0.10.13:
conda create -n cup_cleanup_mp312 python=3.12 -y
conda activate cup_cleanup_mp312
pip install mediapipe==0.10.13"""


def has_mediapipe_solutions() -> bool:
    return mp is not None and hasattr(mp, "solutions")


def detect_hand(frame, hands=None) -> dict:
    result = {"hand_visible": False, "hand_center": None, "landmarks": []}
    if not has_mediapipe_solutions():
        return result

    owns_context = hands is None
    if owns_context:
        hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.35,
            min_tracking_confidence=0.35,
        )

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hand_result = hands.process(rgb)
    if hand_result.multi_hand_landmarks:
        landmarks = hand_result.multi_hand_landmarks[0].landmark
        h, w = frame.shape[:2]
        pixels = [(int(point.x * w), int(point.y * h)) for point in landmarks]
        center_x = int(sum(x for x, _ in pixels) / len(pixels))
        center_y = int(sum(y for _, y in pixels) / len(pixels))
        result = {
            "hand_visible": True,
            "hand_center": [center_x, center_y],
            "landmarks": pixels,
        }

    if owns_context and hands is not None:
        hands.close()
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect a hand center using MediaPipe Hands.")
    parser.add_argument("--camera-index", type=int, default=0, help="Camera index")
    parser.add_argument("--backend", default="auto", choices=["auto", "dshow", "msmf"], help="OpenCV backend")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if mp is None:
        print("[ERROR] mediapipe is not installed. Run `pip install -r requirements.txt` first.")
        return 1
    if not has_mediapipe_solutions():
        print(MEDIAPIPE_SOLUTIONS_ERROR)
        return 1

    backend_id = BACKEND_MAP[args.backend]
    cap = cv2.VideoCapture(args.camera_index) if backend_id is None else cv2.VideoCapture(args.camera_index, backend_id)
    if not cap.isOpened():
        print("[ERROR] Could not open camera.")
        return 1

    hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.35,
        min_tracking_confidence=0.35,
    )
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[ERROR] Failed to read frame from camera.")
                return 1

            detection = detect_hand(frame, hands=hands)
            if detection["hand_visible"]:
                cx, cy = detection["hand_center"]
                cv2.circle(frame, (cx, cy), 10, (0, 255, 0), -1)
                cv2.putText(frame, "Hand", (cx + 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow("Hand Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        hands.close()
        cap.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
