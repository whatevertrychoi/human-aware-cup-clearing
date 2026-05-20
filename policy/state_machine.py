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
    asked_drink_milestones: set[int] = field(default_factory=set)
    exclude_from_policy: bool = False
    handled_reason: str | None = None
    liquid_check_result: str | None = None


@dataclass
class SoftTransitionStateMachine:
    thresholds: dict
    states: dict[int, CupRuntimeState] = field(default_factory=dict)

    def reset(self) -> None:
        self.states.clear()

    def _get_runtime_state(self, cup_id: int) -> CupRuntimeState:
        return self.states.setdefault(cup_id, CupRuntimeState())

    def _normalize(self, value: float, upper: float) -> float:
        if upper <= 0.0:
            return 0.0
        return max(0.0, min(float(value) / float(upper), 1.0))

    def _used_strength(self, cup: dict) -> float:
        near_threshold = max(float(self.thresholds.get("time_near_threshold", 0.8)), 0.1)
        touch_threshold = max(float(self.thresholds.get("touch_count_used_threshold", 1)), 1.0)
        near_score = self._normalize(float(cup.get("time_near_cup", 0.0)), near_threshold * 3.0)
        release_score = self._normalize(float(cup.get("release_count", 0.0)), 3.0)
        touch_score = self._normalize(float(cup.get("touch_count", 0.0)), touch_threshold * 3.0)
        motion_score = self._normalize(float(cup.get("cup_motion_distance", 0.0)), 0.25)
        used_flag = 1.0 if int(cup.get("used_cup_candidate", 0)) == 1 else 0.0
        return min(
            1.0,
            0.25 * near_score + 0.20 * release_score + 0.20 * touch_score + 0.20 * motion_score + 0.15 * used_flag,
        )

    def _compute_ask_priority(self, cup: dict, prediction: dict) -> tuple[tuple[float, ...], float]:
        confidence = float(prediction.get("confidence", 0.0))
        hand_distance = float(cup.get("hand_distance", 999.0))
        far_score = self._normalize(min(hand_distance, 2.0), 2.0)
        drink_progress = float(cup.get("estimated_drink_progress", 0.0))
        used_strength = self._used_strength(cup)
        release_score = self._normalize(
            max(0.0, float(cup.get("time_since_release", 0.0)) - float(self.thresholds.get("ask_delay_after_release", 20.0))),
            40.0,
        )
        stationary_score = self._normalize(float(cup.get("stationary_time", 0.0)), 60.0)
        score = (
            0.30 * far_score
            + 0.20 * drink_progress
            + 0.18 * used_strength
            + 0.14 * release_score
            + 0.10 * stationary_score
            + 0.08 * confidence
        )
        priority_key = (
            far_score,
            drink_progress,
            used_strength,
            release_score,
            stationary_score,
            confidence,
        )
        return priority_key, score

    def _infer_ask_reason(self, cup: dict, prediction: dict) -> str:
        far_score = self._normalize(min(float(cup.get("hand_distance", 999.0)), 2.0), 2.0)
        drink_progress = float(cup.get("estimated_drink_progress", 0.0))
        used_strength = self._used_strength(cup)
        release_score = self._normalize(
            max(0.0, float(cup.get("time_since_release", 0.0)) - float(self.thresholds.get("ask_delay_after_release", 20.0))),
            40.0,
        )
        if far_score >= 0.75:
            return "far_from_user"
        if drink_progress >= float(self.thresholds.get("drink_progress_ask_threshold", 0.65)):
            return "high_drink_progress"
        if used_strength >= 0.55:
            return "used_cup_candidate"
        if release_score >= 0.30:
            return "post_release_timeout"
        if int(cup.get("used_cup_candidate", 0)) == 0 and float(cup.get("stationary_time", 0.0)) >= float(
            self.thresholds.get("never_active_ask_delay", 60.0)
        ):
            return "untouched_idle_timeout"
        return "highest_priority_candidate"

    def _activate_ask(self, cup_id: int, timestamp_now: float) -> CupRuntimeState:
        runtime_state = self._get_runtime_state(cup_id)
        runtime_state.previous_state = runtime_state.state
        runtime_state.state = "ASK_PENDING"
        runtime_state.last_asked_at = timestamp_now
        runtime_state.ask_count += 1
        runtime_state.ask_pending = True
        runtime_state.user_response = None
        runtime_state.ready_to_clear = False
        runtime_state.liquid_check_result = None
        return runtime_state

    def _next_available_drink_milestone(self, drink_count: int, runtime_state: CupRuntimeState) -> int | None:
        milestones = [int(value) for value in self.thresholds.get("ask_drink_count_milestones", [5, 8, 10])]
        for milestone in sorted(milestones):
            if drink_count >= milestone and milestone not in runtime_state.asked_drink_milestones:
                return milestone
        return None

    def apply_user_response(self, cup_id: int, response: str, timestamp_now: float) -> None:
        runtime_state = self._get_runtime_state(cup_id)
        if not runtime_state.ask_pending:
            return

        if response == "yes":
            runtime_state.ask_pending = False
            runtime_state.ready_to_clear = True
            runtime_state.user_response = "yes"
            runtime_state.cooldown_until = None
            runtime_state.exclude_from_policy = True
            runtime_state.handled_reason = "accepted_for_cleanup"
            runtime_state.liquid_check_result = None
            runtime_state.previous_state = runtime_state.state
            runtime_state.state = "READY_TO_CLEAR"
        elif response == "no":
            runtime_state.ask_pending = False
            runtime_state.ready_to_clear = False
            runtime_state.user_response = "no"
            runtime_state.cooldown_until = timestamp_now + float(self.thresholds.get("ask_cooldown_seconds", 30.0))
            runtime_state.exclude_from_policy = False
            runtime_state.handled_reason = "user_rejected_cleanup"
            runtime_state.liquid_check_result = None
            runtime_state.previous_state = runtime_state.state
            runtime_state.state = "ASK_COOLDOWN"

    def get_pending_cup_id(self) -> int | None:
        for cup_id, runtime_state in self.states.items():
            if runtime_state.ask_pending:
                return cup_id
        return None

    def get_selected_liquid_check_cup_id(self) -> int | None:
        for cup_id, runtime_state in self.states.items():
            if runtime_state.state == "NEEDS_LIQUID_CHECK":
                return cup_id
        return None

    def apply_liquid_check_response(self, cup_id: int, response: str, timestamp_now: float) -> None:
        runtime_state = self._get_runtime_state(cup_id)
        if runtime_state.state != "NEEDS_LIQUID_CHECK":
            return

        runtime_state.previous_state = runtime_state.state
        runtime_state.ask_pending = False
        runtime_state.cooldown_until = None
        runtime_state.user_response = None
        runtime_state.exclude_from_policy = True

        if response == "yes":
            runtime_state.ready_to_clear = True
            runtime_state.handled_reason = "liquid_check_empty_clear"
            runtime_state.liquid_check_result = "EMPTY"
            runtime_state.state = "READY_TO_CLEAR"
        elif response == "no":
            runtime_state.ready_to_clear = False
            runtime_state.handled_reason = "liquid_check_non_empty_restore"
            runtime_state.liquid_check_result = "NON_EMPTY"
            runtime_state.state = "HANDLED"

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
        drink_count = int(cup.get("drink_count", 0))
        drink_progress = float(cup.get("estimated_drink_progress", 0.0))

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
        drink_progress_ask_threshold = float(self.thresholds.get("drink_progress_ask_threshold", 0.65))

        previous_state = runtime_state.state
        state = predicted_action
        final_action = predicted_action
        override_reason = "model_first"
        reuse_event = False
        ask_candidate = False
        ask_reason = "none"
        ask_priority = 0.0
        selected_for_ask = False
        ask_candidate_rank = 0
        verification_required = False
        selected_for_liquid_check = False
        liquid_check_status = "none"

        if runtime_state.exclude_from_policy:
            if runtime_state.ready_to_clear:
                state = "READY_TO_CLEAR"
                final_action = "READY_TO_CLEAR"
                override_reason = "liquid_check_empty_clear" if runtime_state.liquid_check_result == "EMPTY" else "excluded_after_accept"
            else:
                state = "HANDLED"
                final_action = "IDLE"
                override_reason = "liquid_check_non_empty_restore" if runtime_state.liquid_check_result == "NON_EMPTY" else "excluded_from_policy"
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
            merged["reuse_event"] = False
            merged["reuse_count"] = int(runtime_state.reuse_count)
            merged["last_reuse_time"] = float(last_reuse_time)
            merged["ask_cancelled_by_reuse"] = bool(runtime_state.ask_cancelled_by_reuse)
            merged["ask_count"] = int(runtime_state.ask_count)
            merged["ask_pending"] = bool(runtime_state.ask_pending)
            merged["last_asked_time"] = float(last_asked_time)
            merged["cooldown_remaining"] = float(cooldown_remaining)
            merged["user_response"] = runtime_state.user_response or "none"
            merged["ready_to_clear"] = bool(runtime_state.ready_to_clear)
            merged["ask_candidate"] = False
            merged["ask_priority"] = 0.0
            merged["ask_reason"] = "none"
            merged["selected_for_ask"] = False
            merged["ask_candidate_rank"] = 0
            merged["verification_required"] = bool(runtime_state.ready_to_clear)
            merged["selected_for_liquid_check"] = bool(runtime_state.ready_to_clear)
            if runtime_state.liquid_check_result == "EMPTY":
                merged["liquid_check_status"] = "empty"
            elif runtime_state.liquid_check_result == "NON_EMPTY":
                merged["liquid_check_status"] = "non_empty"
            else:
                merged["liquid_check_status"] = "pending" if runtime_state.ready_to_clear else "none"
            merged["exclude_from_policy"] = True
            merged["handled_reason"] = runtime_state.handled_reason or "accepted_for_cleanup"
            merged["liquid_check_result"] = runtime_state.liquid_check_result or "none"
            return merged

        runtime_state.ask_cancelled_by_reuse = False
        if active_and_near and previous_state in {
            "OBSERVE",
            "ASK",
            "ASK_PENDING",
            "ASK_COOLDOWN",
            "READY_TO_CLEAR",
            "NEEDS_LIQUID_CHECK",
            "SPILL_SAFE_CLEAR",
        }:
            reuse_event = True
            runtime_state.reuse_count += 1
            runtime_state.last_reuse_at = timestamp_now
            runtime_state.ask_cancelled_by_reuse = True
            runtime_state.ask_pending = False
            runtime_state.cooldown_until = None
            runtime_state.user_response = None
            runtime_state.ready_to_clear = False
            runtime_state.exclude_from_policy = False
            runtime_state.handled_reason = None
            runtime_state.liquid_check_result = None

        cooldown_active = runtime_state.cooldown_until is not None and timestamp_now < runtime_state.cooldown_until
        if runtime_state.cooldown_until is not None and not cooldown_active:
            runtime_state.cooldown_until = None
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
            if stationary_time >= self.thresholds["never_active_ask_delay"] and runtime_state.ask_count < ask_repeat_limit:
                ask_candidate = True
                state = "ASK"
                final_action = "ASK"
                override_reason = "ask_candidate"
                ask_reason = "untouched_idle_timeout"
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
                next_drink_milestone = self._next_available_drink_milestone(drink_count, runtime_state)
                milestone_ready = next_drink_milestone is not None
                progress_ready = drink_progress >= drink_progress_ask_threshold
                if not milestone_ready and not progress_ready:
                    state = "IDLE"
                    final_action = "IDLE"
                    override_reason = "insufficient_drink_progress"
                else:
                    ask_candidate = True
                    state = "ASK"
                    final_action = "ASK"
                    override_reason = "ask_candidate"
                    if milestone_ready:
                        ask_reason = f"drink_count_milestone_{next_drink_milestone}"
                    elif progress_ready:
                        ask_reason = "high_drink_progress"
        elif cleanup_ready:
            state = "NEEDS_LIQUID_CHECK"
            final_action = "NEEDS_LIQUID_CHECK"
            override_reason = "cleanup_ready"
            verification_required = True
            liquid_check_status = "pending"
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

        if ask_candidate:
            _, ask_priority = self._compute_ask_priority(cup, prediction)
            ask_reason = self._infer_ask_reason(cup, prediction)

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
        merged["ask_candidate"] = bool(ask_candidate)
        merged["ask_priority"] = float(ask_priority)
        merged["ask_reason"] = ask_reason
        merged["selected_for_ask"] = bool(selected_for_ask)
        merged["ask_candidate_rank"] = int(ask_candidate_rank)
        merged["verification_required"] = bool(verification_required)
        merged["selected_for_liquid_check"] = bool(selected_for_liquid_check)
        merged["liquid_check_status"] = liquid_check_status
        merged["exclude_from_policy"] = bool(runtime_state.exclude_from_policy)
        merged["handled_reason"] = runtime_state.handled_reason or "none"
        merged["liquid_check_result"] = runtime_state.liquid_check_result or "none"
        return merged

    def finalize_frame(
        self,
        cups: list[dict],
        frame_predictions: list[dict],
        timestamp_now: float,
    ) -> list[dict]:
        cup_map = {int(cup["cup_id"]): cup for cup in cups}
        results = [dict(item) for item in frame_predictions]

        pending_exists = any(item.get("state") == "ASK_PENDING" for item in results)
        ask_candidates = []
        for item in results:
            cup_id = int(item["cup_id"])
            cup = cup_map[cup_id]
            if item.get("exclude_from_policy", False):
                continue
            if not item.get("ask_candidate", False):
                continue
            runtime_state = self._get_runtime_state(cup_id)
            if runtime_state.ask_pending:
                continue
            if runtime_state.cooldown_until is not None and timestamp_now < runtime_state.cooldown_until:
                continue
            if int(cup.get("used_cup_candidate", 0)) == 0 and float(cup.get("stationary_time", 0.0)) < float(
                self.thresholds.get("never_active_ask_delay", 60.0)
            ):
                continue
            priority_key, priority_score = self._compute_ask_priority(cup, item)
            ask_candidates.append(
                {
                    "cup_id": cup_id,
                    "priority_key": priority_key,
                    "priority_score": priority_score,
                    "ask_reason": item.get("ask_reason", "none") if item.get("ask_reason", "none") != "none" else self._infer_ask_reason(cup, item),
                }
            )

        ask_candidates.sort(key=lambda item: (item["priority_key"], item["priority_score"]), reverse=True)
        selected_ask_cup_id = ask_candidates[0]["cup_id"] if ask_candidates and not pending_exists else None
        ask_candidate_rank_map = {item["cup_id"]: rank + 1 for rank, item in enumerate(ask_candidates)}
        ask_priority_map = {item["cup_id"]: item["priority_score"] for item in ask_candidates}
        ask_reason_map = {item["cup_id"]: item["ask_reason"] for item in ask_candidates}

        for item in results:
            cup_id = int(item["cup_id"])
            if cup_id in ask_candidate_rank_map:
                item["ask_priority"] = float(ask_priority_map[cup_id])
                item["ask_reason"] = ask_reason_map[cup_id]
                item["ask_candidate_rank"] = int(ask_candidate_rank_map[cup_id])
                item["selected_for_ask"] = bool(cup_id == selected_ask_cup_id)
            else:
                item["ask_priority"] = float(item.get("ask_priority", 0.0))
                item["ask_reason"] = item.get("ask_reason", "none")
                item["ask_candidate_rank"] = int(item.get("ask_candidate_rank", 0))
                item["selected_for_ask"] = False

            if item.get("ask_candidate", False):
                if pending_exists and item.get("state") != "ASK_PENDING":
                    item["state"] = "WAITING_QUEUE"
                    item["action"] = "IDLE"
                    item["reason"] = "ask_pending_exists"
                elif cup_id == selected_ask_cup_id:
                    runtime_state = self._activate_ask(cup_id, timestamp_now)
                    drink_count = int(cup.get("drink_count", 0))
                    triggered_milestone = self._next_available_drink_milestone(drink_count, runtime_state)
                    if triggered_milestone is not None:
                        runtime_state.asked_drink_milestones.add(triggered_milestone)
                    item["state"] = "ASK_PENDING"
                    item["action"] = "ASK"
                    item["reason"] = "ask_once_triggered"
                    item["ask_pending"] = True
                    item["ask_count"] = int(runtime_state.ask_count)
                    item["last_asked_time"] = 0.0
                    item["cooldown_remaining"] = 0.0
                    item["ask_reason"] = ask_reason_map[cup_id]
                else:
                    item["state"] = "WAITING_QUEUE"
                    item["action"] = "IDLE"
                    item["reason"] = "ask_waiting_queue"

        liquid_candidates = [
            item for item in results if item.get("state") == "NEEDS_LIQUID_CHECK" and item.get("verification_required", False)
        ]
        liquid_candidates.sort(
            key=lambda item: (
                float(cup_map[int(item["cup_id"])].get("stationary_time", 0.0)),
                float(cup_map[int(item["cup_id"])].get("last_touched_time", 0.0)),
                float(item.get("confidence", 0.0)),
            ),
            reverse=True,
        )
        selected_liquid_cup_id = liquid_candidates[0]["cup_id"] if liquid_candidates else None
        for item in results:
            if item.get("state") != "NEEDS_LIQUID_CHECK":
                continue
            if int(item["cup_id"]) == int(selected_liquid_cup_id):
                item["selected_for_liquid_check"] = True
                item["verification_required"] = True
                item["liquid_check_status"] = "pending"
            else:
                item["selected_for_liquid_check"] = False
                item["verification_required"] = True
                item["liquid_check_status"] = "queued"
                item["reason"] = "liquid_check_waiting_queue"

        return results
