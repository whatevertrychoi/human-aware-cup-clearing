from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class InteractionTracker:
    touch_threshold: float = 0.12
    default_last_touched_time: float = 999.0
    states: dict[int, dict] = field(default_factory=dict)

    def update(self, cups: list[dict], hand: dict, dt: float) -> list[dict]:
        for state in self.states.values():
            state["last_touched_time"] += max(0.0, dt)
            state["moved_recently"] = 0

        hand_center = hand.get("hand_center")
        for cup in cups:
            cup_id = int(cup["cup_id"])
            state = self.states.setdefault(
                cup_id,
                {
                    "last_touched_time": self.default_last_touched_time,
                    "touch_count": 0,
                    "moved_recently": 0,
                    "prev_position": np.array([cup["x"], cup["y"]], dtype=float),
                },
            )

            current_pos = np.array([cup["x"], cup["y"]], dtype=float)
            moved_distance = float(np.linalg.norm(current_pos - state["prev_position"]))
            state["moved_recently"] = int(moved_distance > 0.02)
            state["prev_position"] = current_pos

            if hand.get("hand_visible") and hand_center is not None:
                hand_vec = np.array(hand_center, dtype=float)
                cup_vec = np.array([cup["x"], cup["y"]], dtype=float)
                hand_distance = float(np.linalg.norm(cup_vec - hand_vec))
            else:
                hand_distance = 999.0

            is_touching = hand_distance < self.touch_threshold
            if is_touching:
                if state["last_touched_time"] >= self.touch_threshold:
                    state["touch_count"] += 1
                state["last_touched_time"] = 0.0

            cup["hand_distance"] = hand_distance
            cup["last_touched_time"] = float(state["last_touched_time"])
            cup["touch_count"] = int(state["touch_count"])
            cup["moved_recently"] = int(state["moved_recently"])
            cup["is_touching"] = bool(is_touching)

        return cups

