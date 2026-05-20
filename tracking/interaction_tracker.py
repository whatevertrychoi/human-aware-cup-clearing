from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class InteractionTracker:
    touch_threshold: float = 0.12
    touch_release_threshold: float = 0.16
    default_last_touched_time: float = 999.0
    time_near_threshold: float = 0.8
    motion_distance_threshold: float = 0.03
    stationary_motion_threshold: float = 0.01
    touch_count_used_threshold: int = 1
    release_debounce_seconds: float = 0.35
    release_cooldown_seconds: float = 1.0
    motion_window_seconds: float = 2.0
    hand_near_recent_window: float = 2.0
    cup_capacity_ml: float = 390.0
    estimated_sip_ml: float = 22.0
    drink_time_threshold_strict: float = 2.0
    drink_motion_threshold_strict: float = 0.08
    drink_event_cooldown: float = 3.0
    face_drink_distance_threshold: float = 0.22
    states: dict[int, dict] = field(default_factory=dict)

    @staticmethod
    def _get_cup_position(cup: dict) -> np.ndarray:
        if "center_pixel" in cup and cup["center_pixel"] is not None:
            return np.array(cup["center_pixel"], dtype=float)
        return np.array([cup["x"], cup["y"]], dtype=float)

    @staticmethod
    def _distance_scale(cup: dict) -> float:
        width = float(cup.get("frame_width", 1.0))
        height = float(cup.get("frame_height", 1.0))
        diagonal = float(np.hypot(width, height))
        return max(diagonal, 1.0)

    def update(self, cups: list[dict], hand: dict, dt: float, face_center: tuple[int, int] | None = None) -> list[dict]:
        for state in self.states.values():
            state["last_touched_time"] += max(0.0, dt)
            state["time_since_release"] += max(0.0, dt)
            state["stationary_time"] += max(0.0, dt)
            state["moved_recently"] = 0

        hand_center = hand.get("hand_center")
        nearest_cup_id = None
        nearest_distance = float("inf")

        for cup in cups:
            cup_id = int(cup["cup_id"])
            current_pos = self._get_cup_position(cup)
            state = self.states.setdefault(
                cup_id,
                {
                    "last_touched_time": self.default_last_touched_time,
                    "touch_count": 0,
                    "moved_recently": 0,
                    "time_near_cup": 0.0,
                    "time_since_release": self.default_last_touched_time,
                    "release_count": 0,
                    "cup_motion_distance": 0.0,
                    "stationary_time": 0.0,
                    "was_moved": 0,
                    "prev_near": False,
                    "near_state": False,
                    "far_duration": 0.0,
                    "last_release_event_time": self.default_last_touched_time,
                    "time_since_hand_near": self.default_last_touched_time,
                    "recent_motion_samples": [],
                    "prev_position": current_pos.copy(),
                    "drink_count": 0,
                    "estimated_consumed_ml": 0.0,
                    "estimated_drink_progress": 0.0,
                    "last_drink_event_time": self.default_last_touched_time,
                    "recent_drink_like_hold": 0.0,
                    "min_face_distance_during_hold": self.default_last_touched_time,
                    "face_approach_detected": 0,
                },
            )

            scale = self._distance_scale(cup)
            moved_distance_pixels = float(np.linalg.norm(current_pos - state["prev_position"]))
            moved_distance = moved_distance_pixels / scale
            state["prev_position"] = current_pos.copy()
            if moved_distance > self.motion_distance_threshold:
                state["recent_motion_samples"].append({"age": 0.0, "distance": moved_distance})
            for sample in state["recent_motion_samples"]:
                sample["age"] += max(0.0, dt)
            state["recent_motion_samples"] = [
                sample for sample in state["recent_motion_samples"] if sample["age"] <= self.motion_window_seconds
            ]
            recent_motion_distance = float(sum(sample["distance"] for sample in state["recent_motion_samples"]))

            state["moved_recently"] = int(moved_distance > self.stationary_motion_threshold)
            if moved_distance > self.stationary_motion_threshold:
                state["stationary_time"] = 0.0
            if moved_distance > self.motion_distance_threshold:
                state["was_moved"] = 1

            if hand.get("hand_visible") and hand_center is not None:
                hand_vec = np.array(hand_center, dtype=float)
                hand_distance_pixels = float(np.linalg.norm(current_pos - hand_vec))
                hand_distance = hand_distance_pixels / scale
                if hand_distance < nearest_distance:
                    nearest_distance = hand_distance
                    nearest_cup_id = cup_id
            else:
                hand_distance_pixels = 9999.0
                hand_distance = 999.0

            is_near_raw = hand_distance < self.touch_threshold
            if state["near_state"]:
                if hand_distance >= self.touch_release_threshold:
                    state["far_duration"] += max(0.0, dt)
                    if state["far_duration"] >= self.release_debounce_seconds:
                        state["near_state"] = False
                        state["far_duration"] = 0.0
                else:
                    state["far_duration"] = 0.0
            else:
                if hand_distance < self.touch_threshold:
                    state["near_state"] = True
                    state["far_duration"] = 0.0

            is_near = bool(state["near_state"])
            if is_near:
                state["time_near_cup"] += max(0.0, dt)
                state["time_since_hand_near"] = 0.0
                state["stationary_time"] = 0.0
                if face_center is not None:
                    face_vec = np.array(face_center, dtype=float)
                    face_distance_pixels = float(np.linalg.norm(current_pos - face_vec))
                    face_distance = face_distance_pixels / scale
                    state["min_face_distance_during_hold"] = min(
                        float(state.get("min_face_distance_during_hold", self.default_last_touched_time)),
                        face_distance,
                    )
            else:
                state["time_since_hand_near"] += max(0.0, dt)
            if state["prev_near"] and not is_near and state["last_release_event_time"] >= self.release_cooldown_seconds:
                state["release_count"] += 1
                state["time_since_release"] = 0.0
                state["stationary_time"] = 0.0
                state["last_release_event_time"] = 0.0
                face_approach_detected = (
                    float(state.get("min_face_distance_during_hold", self.default_last_touched_time))
                    <= self.face_drink_distance_threshold
                )
                state["face_approach_detected"] = int(face_approach_detected)
                if (
                    state["recent_drink_like_hold"] >= self.drink_time_threshold_strict
                    and (recent_motion_distance >= self.drink_motion_threshold_strict or state["was_moved"])
                    and state["last_drink_event_time"] >= self.drink_event_cooldown
                    and face_approach_detected
                ):
                    state["drink_count"] += 1
                    state["last_drink_event_time"] = 0.0
                state["recent_drink_like_hold"] = 0.0
                state["min_face_distance_during_hold"] = self.default_last_touched_time
            state["prev_near"] = is_near

            if is_near:
                if state["last_touched_time"] >= self.touch_threshold:
                    state["touch_count"] += 1
                state["last_touched_time"] = 0.0
                if recent_motion_distance >= (self.drink_motion_threshold_strict * 0.5) or is_near_raw:
                    state["recent_drink_like_hold"] += max(0.0, dt)
            else:
                state["last_drink_event_time"] += max(0.0, dt)
                state["last_release_event_time"] += max(0.0, dt)
                if state["recent_drink_like_hold"] > 0.0 and recent_motion_distance < self.stationary_motion_threshold:
                    state["recent_drink_like_hold"] = max(0.0, state["recent_drink_like_hold"] - max(0.0, dt))

            state["estimated_consumed_ml"] = float(state["drink_count"]) * float(self.estimated_sip_ml)
            state["estimated_drink_progress"] = min(
                state["estimated_consumed_ml"] / max(float(self.cup_capacity_ml), 1.0),
                1.0,
            )

            cup["hand_distance"] = hand_distance
            cup["hand_distance_pixels"] = hand_distance_pixels
            cup["last_touched_time"] = float(state["last_touched_time"])
            cup["touch_count"] = int(state["touch_count"])
            cup["moved_recently"] = int(state["moved_recently"])
            cup["is_touching"] = bool(is_near)
            cup["time_near_cup"] = float(state["time_near_cup"])
            cup["time_since_release"] = float(state["time_since_release"])
            cup["release_count"] = int(state["release_count"])
            cup["cup_motion_distance"] = recent_motion_distance
            cup["stationary_time"] = float(state["stationary_time"])
            cup["was_moved"] = int(state["was_moved"])
            cup["drink_count"] = int(state["drink_count"])
            cup["estimated_consumed_ml"] = float(state["estimated_consumed_ml"])
            cup["estimated_drink_progress"] = float(state["estimated_drink_progress"])
            cup["last_drink_event_time"] = float(state["last_drink_event_time"])
            cup["min_face_distance"] = float(state.get("min_face_distance_during_hold", self.default_last_touched_time))
            cup["face_approach_detected"] = int(state.get("face_approach_detected", 0))

        active_cup_id = nearest_cup_id if hand.get("hand_visible") and nearest_cup_id is not None else None
        for cup in cups:
            cup_id = int(cup["cup_id"])
            state = self.states[cup_id]
            cup["active_cup_id"] = active_cup_id
            cup["is_active_cup"] = int(active_cup_id == cup_id) if active_cup_id is not None else 0
            motion_with_recent_hand = (
                cup.get("cup_motion_distance", 0.0) > self.motion_distance_threshold
                and state.get("time_since_hand_near", self.default_last_touched_time) <= self.hand_near_recent_window
            )
            used_cup_candidate = (
                cup.get("time_near_cup", 0.0) > self.time_near_threshold
                or cup.get("release_count", 0) > 0
                or cup.get("touch_count", 0) >= self.touch_count_used_threshold
                or motion_with_recent_hand
            )
            cup["used_cup_candidate"] = int(used_cup_candidate)
            cup["idle_candidate"] = int(not used_cup_candidate)

        return cups
