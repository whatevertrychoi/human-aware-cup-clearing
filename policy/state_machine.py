from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CupRuntimeState:
    state: str = "IDLE"
    previous_state: str = "IDLE"
    reuse_count: int = 0
    last_reuse_at: float | None = None
    ask_cancelled_by_reuse: bool = False
    last_asked_at: float | None = None
    ask_count: int = 0
    ask_pending: bool = False
    cooldown_until: float | None = None
    user_response: str | None = None
    ready_to_clear: bool = False


@dataclass
class SoftTransitionStateMachine:
    thresholds: dict
    states: dict[int, CupRuntimeState] = field(default_factory=dict)

    def reset(self) -> None:
        self.states.clear()

    def _get_runtime_state(self, cup_id: int) -> CupRuntimeState:
        return self.states.setdefault(cup_id, CupRuntimeState())

    def apply_user_response(self, cup_id: int, response: str, timestamp_now: float) -> None:
        runtime_state = self._get_runtime_state(cup_id)
        if not runtime_state.ask_pending:
            return

        if response == "yes":
            runtime_state.ask_pending = False
            runtime_state.ready_to_clear = True
            runtime_state.user_response = "yes"
            runtime_state.cooldown_until = None
            runtime_state.previous_state = runtime_state.state
            runtime_state.state = "READY_TO_CLEAR"
        elif response == "no":
            runtime_state.ask_pending = False
            runtime_state.ready_to_clear = False
            runtime_state.user_response = "no"
            runtime_state.cooldown_until = timestamp_now + float(self.thresholds.get("ask_cooldown_seconds", 30.0))
            runtime_state.previous_state = runtime_state.state
            runtime_state.state = "ASK_COOLDOWN"

    def get_pending_cup_id(self) -> int | None:
        for cup_id, runtime_state in self.states.items():
            if runtime_state.ask_pending:
                return cup_id
        return None

    def update_cup_state(
        self,
        cup: dict,
        prediction: dict,
        user_state: dict,
        hand: dict,
        timestamp_now: float,
    ) -> dict:
        cup_id = int(cup["cup_id"])
        runtime_state = self._get_runtime_state(cup_id)

        raw_action = str(prediction.get("raw_action", prediction.get("action", "IDLE")))
        predicted_action = str(prediction.get("action", raw_action))
        confidence = float(prediction.get("confidence", 0.0))

        hand_distance = float(cup.get("hand_distance", 999.0))
        used_cup_candidate = bool(cup.get("used_cup_candidate", 0))
        user_present = int(user_state.get("user_present", 0))
        user_absent_time = float(user_state.get("user_absent_time", 0.0))
        is_active_cup = bool(cup.get("is_active_cup", 0))
        time_since_release = float(cup.get("time_since_release", 999.0))
        release_count = int(cup.get("release_count", 0))
        stationary_time = float(cup.get("stationary_time", 0.0))

        active_and_near = bool(hand.get("hand_visible")) and is_active_cup and hand_distance < self.thresholds["touch_threshold"]
        cleanup_ready = (
            user_present == 0
            and user_absent_time > self.thresholds["user_absence_threshold"]
            and stationary_time > self.thresholds["stationary_threshold"]
        ) or (
            user_present == 0
            and float(cup.get("last_touched_time", 999.0)) > self.thresholds["cleanup_time_threshold"]
            and stationary_time > self.thresholds["stationary_threshold"]
        )

        observe_window = max(
            float(self.thresholds.get("observe_min_duration", 3.0)),
            float(self.thresholds.get("ask_delay_after_release", 20.0)),
        )
        cooldown_seconds = float(self.thresholds.get("ask_cooldown_seconds", 30.0))
        ask_repeat_limit = int(self.thresholds.get("ask_repeat_limit", 1))
        ask_pending_timeout = float(self.thresholds.get("ask_pending_timeout", 20.0))

        previous_state = runtime_state.state
        state = predicted_action
        final_action = predicted_action
        override_reason = "model_first"
        reuse_event = False

        runtime_state.ask_cancelled_by_reuse = False
        if (
            active_and_near
            and previous_state in {"OBSERVE", "ASK", "ASK_PENDING", "ASK_COOLDOWN", "READY_TO_CLEAR"}
        ):
            reuse_event = True
            runtime_state.reuse_count += 1
            runtime_state.last_reuse_at = timestamp_now
            runtime_state.ask_cancelled_by_reuse = True
            runtime_state.ask_pending = False
            runtime_state.cooldown_until = None
            runtime_state.user_response = None
            runtime_state.ready_to_clear = False

        cooldown_active = runtime_state.cooldown_until is not None and timestamp_now < runtime_state.cooldown_until
        if runtime_state.cooldown_until is not None and not cooldown_active:
            runtime_state.cooldown_until = None
            runtime_state.ask_count = 0
            if runtime_state.state == "ASK_COOLDOWN":
                runtime_state.state = "IDLE"

        if runtime_state.ask_pending and runtime_state.last_asked_at is not None:
            if (timestamp_now - runtime_state.last_asked_at) >= ask_pending_timeout:
                runtime_state.ask_pending = False
                runtime_state.cooldown_until = timestamp_now + cooldown_seconds
                runtime_state.user_response = "timeout"
                runtime_state.ready_to_clear = False
                cooldown_active = True

        if active_and_near:
            state = "WAIT"
            final_action = "WAIT"
            override_reason = "reuse_detected" if reuse_event else "safety_wait_override"
        elif runtime_state.ready_to_clear:
            state = "READY_TO_CLEAR"
            final_action = "READY_TO_CLEAR"
            override_reason = "user_accepted_cleanup"
        elif runtime_state.ask_pending:
            state = "ASK_PENDING"
            final_action = "ASK_PENDING"
            override_reason = "waiting_for_user_response"
        elif cooldown_active:
            state = "ASK_COOLDOWN"
            final_action = "IDLE"
            override_reason = "ask_cooldown_active"
        elif user_present == 1 and not used_cup_candidate:
            if stationary_time >= self.thresholds["never_active_ask_delay"]:
                if runtime_state.ask_count < ask_repeat_limit:
                    runtime_state.last_asked_at = timestamp_now
                    runtime_state.ask_count += 1
                    runtime_state.ask_pending = True
                    runtime_state.user_response = None
                    state = "ASK_PENDING"
                    final_action = "ASK"
                    override_reason = "ask_once_triggered"
                else:
                    state = "ASK_COOLDOWN"
                    final_action = "IDLE"
                    override_reason = "ask_repeat_limit_reached"
            else:
                state = "IDLE"
                final_action = "IDLE"
                override_reason = "present_unused_idle_suppression"
        elif user_present == 1 and used_cup_candidate:
            if release_count <= 0 or hand_distance < self.thresholds["touch_threshold"]:
                state = "IDLE"
                final_action = "IDLE"
                override_reason = "awaiting_release"
            elif time_since_release < observe_window or stationary_time < self.thresholds["observe_min_duration"]:
                state = "OBSERVE"
                final_action = "OBSERVE"
                override_reason = "post_release_observe"
            else:
                if runtime_state.ask_count < ask_repeat_limit:
                    runtime_state.last_asked_at = timestamp_now
                    runtime_state.ask_count += 1
                    runtime_state.ask_pending = True
                    runtime_state.user_response = None
                    state = "ASK_PENDING"
                    final_action = "ASK"
                    override_reason = "ask_once_triggered"
                else:
                    runtime_state.cooldown_until = timestamp_now + cooldown_seconds
                    state = "ASK_COOLDOWN"
                    final_action = "IDLE"
                    override_reason = "ask_repeat_limit_reached"
        elif predicted_action == "CLEANUP_CANDIDATE" and cleanup_ready:
            state = "CLEANUP_CANDIDATE"
            final_action = "CLEANUP_CANDIDATE"
            override_reason = "cleanup_ready"
        elif predicted_action == "CLEANUP_CANDIDATE" and not cleanup_ready:
            state = "IDLE"
            final_action = "IDLE"
            override_reason = "cleanup_requires_abandonment"
        elif predicted_action == "ASK" and confidence < self.thresholds["confidence_threshold"]:
            state = "IDLE"
            final_action = "IDLE"
            override_reason = "low_confidence_ask_guard"
        else:
            state = predicted_action
            final_action = predicted_action

        runtime_state.previous_state = previous_state
        runtime_state.state = state

        last_reuse_time = (
            999.0
            if runtime_state.last_reuse_at is None
            else max(0.0, float(timestamp_now) - float(runtime_state.last_reuse_at))
        )
        last_asked_time = (
            999.0
            if runtime_state.last_asked_at is None
            else max(0.0, float(timestamp_now) - float(runtime_state.last_asked_at))
        )
        cooldown_remaining = (
            0.0
            if runtime_state.cooldown_until is None
            else max(0.0, float(runtime_state.cooldown_until) - float(timestamp_now))
        )

        merged = dict(prediction)
        merged["state"] = state
        merged["previous_state"] = previous_state
        merged["action"] = final_action
        merged["raw_action"] = raw_action
        merged["confidence"] = confidence
        merged["reason"] = override_reason
        merged["used_cup_candidate"] = used_cup_candidate
        merged["is_active_cup"] = is_active_cup
        merged["reuse_event"] = reuse_event
        merged["reuse_count"] = int(runtime_state.reuse_count)
        merged["last_reuse_time"] = float(last_reuse_time)
        merged["ask_cancelled_by_reuse"] = bool(runtime_state.ask_cancelled_by_reuse)
        merged["ask_count"] = int(runtime_state.ask_count)
        merged["ask_pending"] = bool(runtime_state.ask_pending)
        merged["last_asked_time"] = float(last_asked_time)
        merged["cooldown_remaining"] = float(cooldown_remaining)
        merged["user_response"] = runtime_state.user_response or "none"
        merged["ready_to_clear"] = bool(runtime_state.ready_to_clear)
        return merged
