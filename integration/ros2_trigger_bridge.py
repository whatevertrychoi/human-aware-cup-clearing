from __future__ import annotations

"""ROS2 trigger bridge for exporting high-level policy decisions.

This module is the boundary between the local cup-cleanup policy runtime and
the external ROS2 robot stack. The policy still reasons in terms of internal
cup IDs and abstract states such as ASK or NEEDS_LIQUID_CHECK. The bridge turns
those into transport-friendly one-shot JSON trigger messages.

Important design points:
- internal policy IDs are remapped to robot-side class labels here
- repeated per-frame policy outputs are collapsed into one-shot trigger events
- cleanup cancellation is handled as a session-level concern, not just a single
  frame-level state drop
- ASK events may be routed to a different topic than robot motion events
"""

import json
from dataclasses import dataclass, field


ASK_SESSION_STATES = {"ASK_PENDING", "READY_TO_CLEAR"}
ASK_REARM_DELAY_SEC = 1.5
# Keep the ASK session latched to robot execution. This timeout is only a
# fallback watchdog in case robot feedback is never received.
ASK_STATE_CLEAR_GRACE_SEC = 600.0
ROBOT_FEEDBACK_TOPIC = "/cup_cleanup/robot_feedback"


@dataclass
class ROS2TriggerBridge:
    """Publish policy-side social or cleanup events onto a ROS2 topic.

    The bridge has two jobs:
    1. convert per-frame policy outputs into one-shot robot trigger messages
    2. keep enough session state so the robot is not confused by frame flicker

    In practice, this means ASK and cleanup events are not emitted on every
    frame. Instead, the bridge remembers which cup currently owns the active
    social or cleanup session and only publishes the next trigger when the
    previous session is truly finished.
    """

    enabled: bool
    ask_topic_name: str
    robot_topic_name: str
    node_name: str
    cup_id_to_robot_label: dict[int, int] = field(default_factory=dict)
    ask_publisher: object | None = None
    robot_publisher: object | None = None
    feedback_subscription: object | None = None
    node: object | None = None
    _rclpy: object | None = None
    _std_msgs: object | None = None
    _last_ask_signature: tuple[int, int] | None = None
    _last_cancel_signature: tuple[int, int] | None = None
    _active_liquid_check_cups: set[int] = field(default_factory=set)
    _active_ask_cups: set[int] = field(default_factory=set)
    _active_ask_session_cup: int | None = None
    _pending_ask_events: dict[int, dict] = field(default_factory=dict)
    _pending_robot_feedback: list[dict] = field(default_factory=list)
    _ask_rearm_at: float = 0.0
    _ask_state_cleared_since: float | None = None
    _liquid_cleanup_session_active: bool = False
    _liquid_cleanup_empty_since: float | None = None

    @classmethod
    def from_config(cls, config: dict, active: bool) -> "ROS2TriggerBridge":
        """Build and optionally initialize the bridge from config.yaml."""
        cfg = config.get("ros2_trigger", {}) if isinstance(config, dict) else {}
        enabled = bool(cfg.get("enabled", False)) and active
        legacy_topic_name = str(cfg.get("topic", "/cup_cleanup/trigger"))
        ask_topic_name = str(cfg.get("ask_topic", legacy_topic_name))
        robot_topic_name = str(cfg.get("robot_topic", legacy_topic_name))
        node_name = str(cfg.get("node_name", "cup_cleanup_trigger_bridge"))
        cup_id_map_raw = cfg.get("cup_id_to_robot_label", {})
        cup_id_to_robot_label: dict[int, int] = {}
        if isinstance(cup_id_map_raw, dict):
            for key, value in cup_id_map_raw.items():
                try:
                    cup_id_to_robot_label[int(key)] = int(value)
                except (TypeError, ValueError):
                    continue
        bridge = cls(
            enabled=enabled,
            ask_topic_name=ask_topic_name,
            robot_topic_name=robot_topic_name,
            node_name=node_name,
            cup_id_to_robot_label=cup_id_to_robot_label,
        )
        bridge._initialize_ros()
        return bridge

    def _initialize_ros(self) -> None:
        """Create the ROS2 node, publishers, and feedback subscription."""
        if not self.enabled:
            return
        try:
            import rclpy
            from std_msgs.msg import String
        except ImportError:
            print("[WARN] ROS2 trigger bridge enabled but rclpy/std_msgs are unavailable. Trigger publishing disabled.")
            self.enabled = False
            return

        self._rclpy = rclpy
        self._std_msgs = String
        if not rclpy.ok():
            rclpy.init(args=None)
        self.node = rclpy.create_node(self.node_name)
        self.ask_publisher = self.node.create_publisher(String, self.ask_topic_name, 10)
        self.robot_publisher = self.node.create_publisher(String, self.robot_topic_name, 10)
        self.feedback_subscription = self.node.create_subscription(
            String,
            ROBOT_FEEDBACK_TOPIC,
            self._handle_robot_feedback,
            10,
        )
        print(
            f"[INFO] ROS2 trigger bridge enabled: "
            f"ask_topic={self.ask_topic_name}, robot_topic={self.robot_topic_name}"
        )

    def close(self) -> None:
        if not self.enabled or self.node is None or self._rclpy is None:
            return
        try:
            self.node.destroy_node()
        finally:
            self.node = None
            self.ask_publisher = None
            self.robot_publisher = None

    def _map_cup_id(self, source_cup_id: int) -> int:
        return int(self.cup_id_to_robot_label.get(int(source_cup_id), int(source_cup_id)))

    def _publish_to(self, publisher, payload: dict) -> None:
        if not self.enabled or publisher is None or self._std_msgs is None:
            return
        message = self._std_msgs()
        message.data = json.dumps(payload, ensure_ascii=True)
        publisher.publish(message)

    def _publish_ask(self, payload: dict) -> None:
        self._publish_to(self.ask_publisher, payload)

    def _publish_robot(self, payload: dict) -> None:
        self._publish_to(self.robot_publisher, payload)

    def _clear_active_ask_session(self, timestamp_now: float) -> None:
        """Release ASK ownership and arm a short delay before the next ASK.

        The re-arm delay prevents an old ASK session from clearing and a new ASK
        from being published in the exact same frame, which previously caused
        duplicate or confusing back-to-back social prompts.
        """
        if self._active_ask_session_cup is not None:
            self._active_ask_cups.discard(self._active_ask_session_cup)
        self._active_ask_session_cup = None
        self._ask_rearm_at = float(timestamp_now) + ASK_REARM_DELAY_SEC
        self._ask_state_cleared_since = None

    def _pump_robot_feedback(self) -> None:
        if self._rclpy is None or self.node is None:
            return
        try:
            self._rclpy.spin_once(self.node, timeout_sec=0.0)
        except Exception:
            pass

    def drain_robot_feedback(self) -> list[dict]:
        self._pump_robot_feedback()
        feedback_events = list(self._pending_robot_feedback)
        self._pending_robot_feedback.clear()
        return feedback_events

    def _handle_robot_feedback(self, msg) -> None:
        """End the robot-owned ASK session only when the robot reports done.

        Policy state can flicker for a frame while the robot is still moving.
        Robot feedback is therefore treated as the authoritative signal that the
        ask-related execution really completed, aborted, or was cancelled.
        """
        try:
            payload = json.loads(msg.data)
        except Exception:
            return
        if payload.get("event_type") != "ASK_ACTION_FINISHED":
            return
        source_cup_id = int(payload.get("source_cup_id", -1))
        status = str(payload.get("status", "unknown"))
        if status not in {"completed", "aborted", "cancelled"}:
            return
        self._pending_robot_feedback.append(
            {
                "event_type": "ASK_ACTION_FINISHED",
                "source_cup_id": source_cup_id,
                "status": status,
                "timestamp": float(payload.get("timestamp", 0.0) or 0.0),
                "reason": str(payload.get("reason", "none")),
            }
        )
        if self._active_ask_session_cup != source_cup_id:
            return
        print(
            f"[INFO] ASK session released from robot feedback: "
            f"source_cup_id={source_cup_id}, status={status}"
        )
        self._clear_active_ask_session(float(payload.get("timestamp", 0.0) or 0.0))

    def _queue_or_publish_ask(self, payload: dict, ask_signature: tuple[int, int]) -> None:
        """Publish immediately when idle, otherwise queue behind the active ASK."""
        source_cup_id = int(payload["source_cup_id"])
        payload_copy = dict(payload)
        payload_copy["ask_signature"] = ask_signature
        if self._active_ask_session_cup is None or self._active_ask_session_cup == source_cup_id:
            self._last_ask_signature = ask_signature
            self._active_ask_cups.add(source_cup_id)
            self._active_ask_session_cup = source_cup_id
            self._ask_state_cleared_since = None
            self._publish_ask(payload)
            return
        self._pending_ask_events[source_cup_id] = payload_copy

    def _drain_pending_ask(self, timestamp_now: float) -> None:
        """Promote the oldest queued ASK once the active session is gone."""
        if self._active_ask_session_cup is not None or not self._pending_ask_events:
            return
        if float(timestamp_now) < float(self._ask_rearm_at):
            return
        source_cup_id = next(iter(self._pending_ask_events))
        payload = self._pending_ask_events.pop(source_cup_id)
        ask_signature = tuple(payload.pop("ask_signature"))
        payload["timestamp"] = float(timestamp_now)
        self._last_ask_signature = ask_signature
        self._active_ask_cups.add(source_cup_id)
        self._active_ask_session_cup = source_cup_id
        self._ask_state_cleared_since = None
        self._publish_ask(payload)

    def apply_action_latches(self, predictions: list[dict]) -> list[dict]:
        """Keep the active ASK cup visually latched to ASK in the overlay.

        The policy runtime may already be transitioning through intermediate
        states while the robot is still executing the ASK flow. For operator
        clarity, the UI keeps showing ASK until the robot feedback releases the
        session.
        """
        if self._active_ask_session_cup is None or not predictions:
            return predictions

        latched_predictions: list[dict] = []
        for item in predictions:
            merged = dict(item)
            if int(merged.get("cup_id", -1)) == int(self._active_ask_session_cup):
                merged["robot_ask_latched"] = True
                merged["latched_previous_state"] = merged.get("state", merged.get("action", "IDLE"))
                merged["state"] = "ASK"
                merged["action"] = "ASK"
                merged["reason"] = "robot_ask_latched"
            latched_predictions.append(merged)
        return latched_predictions

    def process_predictions(self, predictions: list[dict], timestamp_now: float) -> None:
        """Convert one frame of policy predictions into ROS2 trigger events.

        This method is called once per live-policy frame. It:
        - receives any pending robot feedback first
        - scans the current policy outputs
        - emits one-shot ASK / cancel / cleanup triggers when conditions match
        - maintains session state for ASK ownership and cleanup lifecycle
        """
        if not self.enabled:
            return
        self._pump_robot_feedback()

        # `current_liquid_cups` tracks which cups still belong to the current
        # policy-side liquid-check candidate set in this frame.
        current_liquid_cups: set[int] = set()
        current_ask_cups: set[int] = set()
        prediction_map = {int(item.get("cup_id", -1)): item for item in predictions}
        selected_liquid_item: dict | None = None
        for item in predictions:
            source_cup_id = int(item.get("cup_id", -1))
            robot_cup_id = self._map_cup_id(source_cup_id)

            if (
                item.get("action") == "ASK"
                or item.get("state") in ASK_SESSION_STATES
                or bool(item.get("ask_pending", False))
            ):
                current_ask_cups.add(source_cup_id)

            # ASK is exported only on the one-shot frame where the state machine
            # explicitly selected this cup for a real ask event.
            if (
                item.get("action") == "ASK"
                and item.get("reason") == "ask_once_triggered"
                and bool(item.get("selected_for_ask", False))
            ):
                current_ask_cups.add(source_cup_id)
                ask_signature = (source_cup_id, int(item.get("ask_count", 0)))
                if ask_signature != self._last_ask_signature:
                    self._queue_or_publish_ask(
                        {
                            "event_type": "ASK_TRIGGER",
                            "cup_id": robot_cup_id,
                            "source_cup_id": source_cup_id,
                            "ask_reason": item.get("ask_reason", "none"),
                            "timestamp": float(timestamp_now),
                        },
                        ask_signature,
                    )
            # If a previously active ASK session becomes invalid due to reuse,
            # publish a matching cancel event so the robot or downstream client
            # can abandon the pending ask flow.
            if bool(item.get("ask_cancelled_by_reuse", False)):
                cancel_signature = (source_cup_id, int(item.get("reuse_count", 0)))
                if source_cup_id in self._active_ask_cups and cancel_signature != self._last_cancel_signature:
                    self._last_cancel_signature = cancel_signature
                    self._active_ask_cups.discard(source_cup_id)
                    if self._active_ask_session_cup == source_cup_id:
                        self._clear_active_ask_session(float(timestamp_now))
                    self._publish_ask(
                        {
                            "event_type": "CANCEL_ASK_TRIGGER",
                            "cup_id": robot_cup_id,
                            "source_cup_id": source_cup_id,
                            "reason": "reuse_detected",
                            "timestamp": float(timestamp_now),
                        }
                    )
                    self._pending_ask_events.pop(source_cup_id, None)

            # The bridge now interprets policy-side liquid-check membership as a
            # cleanup-session signal. We still remember which specific cup was
            # selected, but the robot can treat this as "start cleanup cycle".
            if (
                item.get("state") == "NEEDS_LIQUID_CHECK"
                and bool(item.get("selected_for_liquid_check", False))
                and bool(item.get("verification_required", False))
            ):
                selected_liquid_item = item
                current_liquid_cups.add(source_cup_id)
            elif (
                item.get("state") == "NEEDS_LIQUID_CHECK"
                and bool(item.get("verification_required", False))
            ):
                current_liquid_cups.add(source_cup_id)

        # Start exactly one cleanup session when the policy-side liquid-check
        # set becomes non-empty.
        if current_liquid_cups and not self._liquid_cleanup_session_active:
            trigger_item = selected_liquid_item
            if trigger_item is None:
                trigger_item = next(
                    (
                        item for item in predictions
                        if item.get("state") == "NEEDS_LIQUID_CHECK"
                        and bool(item.get("verification_required", False))
                    ),
                    None,
                )
            if trigger_item is not None:
                source_cup_id = int(trigger_item.get("cup_id", -1))
                robot_cup_id = self._map_cup_id(source_cup_id)
                cleanup_source_cup_ids = sorted(current_liquid_cups)
                cleanup_robot_cup_ids: list[int] = []
                for cleanup_source_cup_id in cleanup_source_cup_ids:
                    mapped_robot_cup_id = self._map_cup_id(cleanup_source_cup_id)
                    if mapped_robot_cup_id not in cleanup_robot_cup_ids:
                        cleanup_robot_cup_ids.append(mapped_robot_cup_id)
                self._publish_robot(
                    {
                        "event_type": "ROBOT_LIQUID_CHECK_TRIGGER",
                        "cup_id": robot_cup_id,
                        "source_cup_id": source_cup_id,
                        "cleanup_source_cup_ids": cleanup_source_cup_ids,
                        "cleanup_robot_cup_ids": cleanup_robot_cup_ids,
                        "reason": "cleanup_session_start",
                        "timestamp": float(timestamp_now),
                    }
                )
                self._liquid_cleanup_session_active = True
                self._liquid_cleanup_empty_since = None

        if current_liquid_cups:
            self._liquid_cleanup_empty_since = None

        if self._active_ask_session_cup is not None:
            if self._active_ask_session_cup in current_ask_cups:
                self._ask_state_cleared_since = None
            else:
                # If the policy no longer shows ASK for the active cup, start a
                # watchdog timer. We still prefer robot feedback for release, but
                # this avoids a permanently stuck session when feedback is lost.
                if self._ask_state_cleared_since is None:
                    self._ask_state_cleared_since = float(timestamp_now)
                cleared_duration = float(timestamp_now) - float(self._ask_state_cleared_since)
                if cleared_duration >= ASK_STATE_CLEAR_GRACE_SEC:
                    print(
                        f"[WARN] ASK session watchdog released without robot feedback: "
                        f"source_cup_id={self._active_ask_session_cup}, "
                        f"cleared_duration={cleared_duration:.1f}s"
                    )
                    self._clear_active_ask_session(float(timestamp_now))

        # Pending ASK events are activated only after the current ASK fully ends.
        self._drain_pending_ask(timestamp_now)

        completed_liquid_results = {"EMPTY", "NON_EMPTY"}
        if self._liquid_cleanup_session_active and not current_liquid_cups:
            # A brief empty set can happen from transient perception or
            # arbitration flicker. Wait a short grace period before cancelling
            # the entire cleanup session.
            if self._liquid_cleanup_empty_since is None:
                self._liquid_cleanup_empty_since = float(timestamp_now)
            empty_duration = float(timestamp_now) - self._liquid_cleanup_empty_since
            if empty_duration < 2.0:
                self._active_liquid_check_cups = current_liquid_cups
                return

            cancel_source_cup_id = -1
            cancel_reason = "liquid_check_cancelled"
            should_publish_cancel = True
            if self._active_liquid_check_cups:
                cancel_source_cup_id = sorted(self._active_liquid_check_cups)[0]
                item = prediction_map.get(cancel_source_cup_id, {})
                liquid_check_result = str(item.get("liquid_check_result", "none"))
                if liquid_check_result in completed_liquid_results:
                    should_publish_cancel = False
                else:
                    cancel_reason = str(item.get("reason", "liquid_check_cancelled"))
            if should_publish_cancel:
                self._publish_robot(
                    {
                        "event_type": "CANCEL_ROBOT_LIQUID_CHECK_TRIGGER",
                        "cup_id": self._map_cup_id(cancel_source_cup_id),
                        "source_cup_id": cancel_source_cup_id,
                        "reason": cancel_reason,
                        "timestamp": float(timestamp_now),
                    }
                )
            self._liquid_cleanup_session_active = False
            self._liquid_cleanup_empty_since = None

        self._active_liquid_check_cups = current_liquid_cups
