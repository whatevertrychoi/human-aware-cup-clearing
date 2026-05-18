from __future__ import annotations

from dataclasses import dataclass


@dataclass
class UserPresenceTracker:
    absence_threshold: float = 10.0
    user_present: bool = False
    user_absent_time: float = 0.0

    def update(self, observed_user_present: bool, dt: float) -> dict:
        if observed_user_present:
            self.user_present = True
            self.user_absent_time = 0.0
        else:
            self.user_absent_time += max(0.0, dt)
            self.user_present = self.user_absent_time <= self.absence_threshold

        return {
            "user_present": int(self.user_present),
            "user_absent_time": float(self.user_absent_time),
        }

