from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class InteractionTracker:
    touch_threshold: float = 0.12
    default_last_touched_time: float = 999.0
    states: dict[int, dict] = field(default_factory=dict)

    @staticmethod
    def _get_cup_position(cup: dict) -> np.ndarray:
        if "center_pixel" in cup and cup["center_pixel"] is not None:
            return np.array(cup["center_pixel"], dtype=float)
        return np.array([cup["x"], cup["y"]], dtype=float)

    @staticmethod
    def _distance_scale(cup: dict, hand_center) -> float:
        width = float(cup.get("frame_width", 1.0))
        height = float(cup.get("frame_height", 1.0))
        diagonal = float(np.hypot(width, height))
        return max(diagonal, 1.0)

    def update(self, cups: list[dict], hand: dict, dt: float) -> list[dict]:
        for state in self.states.values():
            state["last_touched_time"] += max(0.0, dt)
            state["moved_recently"] = 0

        hand_center = hand.get("hand_center")
        for cup in cups:
            cup_id = int(cup["cup_id"])
            current_pos = self._get_cup_position(cup)
            state = self.states.setdefault(
                cup_id,
                {
                    "last_touched_time": self.default_last_touched_time,
                    "touch_count": 0,
                    "moved_recently": 0,
                    "prev_position": current_pos.copy(),
                },
            )

            moved_distance = float(np.linalg.norm(current_pos - state["prev_position"]))
            scale = self._distance_scale(cup, hand_center)
            state["moved_recently"] = int((moved_distance / scale) > 0.02)
            state["prev_position"] = current_pos.copy()

            if hand.get("hand_visible") and hand_center is not None:
                hand_vec = np.array(hand_center, dtype=float)
                hand_distance_pixels = float(np.linalg.norm(current_pos - hand_vec))
                hand_distance = hand_distance_pixels / scale
            else:
                hand_distance_pixels = 9999.0
                hand_distance = 999.0

            is_touching = hand_distance < self.touch_threshold
            if is_touching:
                if state["last_touched_time"] >= self.touch_threshold:
                    state["touch_count"] += 1
                state["last_touched_time"] = 0.0

            cup["hand_distance"] = hand_distance
            cup["hand_distance_pixels"] = hand_distance_pixels
            cup["last_touched_time"] = float(state["last_touched_time"])
            cup["touch_count"] = int(state["touch_count"])
            cup["moved_recently"] = int(state["moved_recently"])
            cup["is_touching"] = bool(is_touching)

        return cups
