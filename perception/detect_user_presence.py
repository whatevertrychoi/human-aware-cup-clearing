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

from perception.detect_hand import detect_hand


def detect_user_presence(frame, face_detector=None) -> dict:
    if mp is None:
        hand_detection = detect_hand(frame)
        return {
            "user_present": hand_detection["hand_visible"],
            "face_center": None,
            "confidence": 0.3 if hand_detection["hand_visible"] else 0.0,
        }

    owns_context = face_detector is None
    if owns_context:
        face_detector = mp.solutions.face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.5,
        )

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_detector.process(rgb)
    response = {"user_present": False, "face_center": None, "confidence": 0.0}
    if result.detections:
        bbox = result.detections[0].location_data.relative_bounding_box
        h, w = frame.shape[:2]
        center_x = int((bbox.xmin + bbox.width / 2.0) * w)
        center_y = int((bbox.ymin + bbox.height / 2.0) * h)
        response = {
            "user_present": True,
            "face_center": [center_x, center_y],
            "confidence": float(result.detections[0].score[0]),
        }
    else:
        hand_detection = detect_hand(frame)
        if hand_detection["hand_visible"]:
            response = {"user_present": True, "face_center": None, "confidence": 0.3}

    if owns_context and face_detector is not None:
        face_detector.close()
    return response


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect user presence using MediaPipe Face Detection with fallback.")
    parser.add_argument("--camera-index", type=int, default=0, help="Camera index")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        print("[ERROR] Could not open camera.")
        return 1

    face_detector = None
    if mp is not None:
        face_detector = mp.solutions.face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.5,
        )

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[ERROR] Failed to read frame from camera.")
                return 1

            detection = detect_user_presence(frame, face_detector=face_detector)
            label = f"user_present={detection['user_present']} conf={detection['confidence']:.2f}"
            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if detection["face_center"]:
                cv2.circle(frame, tuple(detection["face_center"]), 10, (255, 0, 0), -1)
            cv2.imshow("User Presence Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        if face_detector is not None:
            face_detector.close()
        cap.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())

