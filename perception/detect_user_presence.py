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
from perception.detect_hand import has_mediapipe_solutions
from perception.detect_hand import MEDIAPIPE_SOLUTIONS_ERROR

BACKEND_MAP = {
    "auto": None,
    "dshow": getattr(cv2, "CAP_DSHOW", None),
    "msmf": getattr(cv2, "CAP_MSMF", None),
}


def detect_user_presence(frame, face_detector=None, hand_detection=None) -> dict:
    if not has_mediapipe_solutions():
        hand_detection = hand_detection or detect_hand(frame)
        return {
            "user_present": hand_detection["hand_visible"],
            "face_center": None,
            "confidence": 0.3 if hand_detection["hand_visible"] else 0.0,
            "source": "hand_fallback",
        }

    owns_context = face_detector is None
    if owns_context:
        face_detector = [
            mp.solutions.face_detection.FaceDetection(
                model_selection=0,
                min_detection_confidence=0.35,
            ),
            mp.solutions.face_detection.FaceDetection(
                model_selection=1,
                min_detection_confidence=0.35,
            ),
        ]

    detectors = face_detector if isinstance(face_detector, list) else [face_detector]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    response = {"user_present": False, "face_center": None, "confidence": 0.0, "source": "none"}
    best_detection = None
    for detector in detectors:
        result = detector.process(rgb)
        if not result.detections:
            continue

        detection = max(result.detections, key=lambda item: float(item.score[0]))
        score = float(detection.score[0])
        if best_detection is None or score > best_detection[0]:
            best_detection = (score, detection)

    if best_detection is not None:
        _, detection = best_detection
        bbox = detection.location_data.relative_bounding_box
        h, w = frame.shape[:2]
        center_x = int((bbox.xmin + bbox.width / 2.0) * w)
        center_y = int((bbox.ymin + bbox.height / 2.0) * h)
        response = {
            "user_present": True,
            "face_center": [center_x, center_y],
            "confidence": float(detection.score[0]),
            "source": "face",
        }
    else:
        hand_detection = hand_detection or detect_hand(frame)
        if hand_detection["hand_visible"]:
            response = {"user_present": True, "face_center": None, "confidence": 0.3, "source": "hand_fallback"}

    if owns_context:
        for detector in detectors:
            detector.close()
    return response


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect user presence using MediaPipe Face Detection with fallback.")
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

    face_detector = None
    if mp is not None:
        face_detector = [
            mp.solutions.face_detection.FaceDetection(
                model_selection=0,
                min_detection_confidence=0.35,
            ),
            mp.solutions.face_detection.FaceDetection(
                model_selection=1,
                min_detection_confidence=0.35,
            ),
        ]

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[ERROR] Failed to read frame from camera.")
                return 1

            hand_detection = detect_hand(frame)
            detection = detect_user_presence(frame, face_detector=face_detector, hand_detection=hand_detection)
            label = (
                f"user_present={detection['user_present']} "
                f"conf={detection['confidence']:.2f} src={detection.get('source', 'n/a')}"
            )
            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if detection["face_center"]:
                cv2.circle(frame, tuple(detection["face_center"]), 10, (255, 0, 0), -1)
            cv2.imshow("User Presence Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        if face_detector is not None:
            for detector in face_detector:
                detector.close()
        cap.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
