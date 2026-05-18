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


def detect_hand(frame, hands=None) -> dict:
    result = {"hand_visible": False, "hand_center": None, "landmarks": []}
    if mp is None:
        return result

    owns_context = hands is None
    if owns_context:
        hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if mp is None:
        print("[ERROR] mediapipe is not installed. Run `pip install -r requirements.txt` first.")
        return 1

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        print("[ERROR] Could not open camera.")
        return 1

    hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
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

