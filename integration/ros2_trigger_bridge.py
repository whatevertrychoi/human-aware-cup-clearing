from __future__ import annotations

import json
from dataclasses import dataclass, field


@dataclass
class ROS2TriggerBridge:
    enabled: bool
    topic_name: str
    node_name: str
    cup_id_to_robot_label: dict[int, int] = field(default_factory=dict)
    publisher: object | None = None
    node: object | None = None
    _rclpy: object | None = None
    _std_msgs: object | None = None
    _last_ask_signature: tuple[int, int] | None = None
    _last_cancel_signature: tuple[int, int] | None = None
    _active_liquid_check_cups: set[int] = field(default_factory=set)
    _active_ask_cups: set[int] = field(default_factory=set)

    @classmethod
    def from_config(cls, config: dict, active: bool) -> "ROS2TriggerBridge":
        cfg = config.get("ros2_trigger", {}) if isinstance(config, dict) else {}
        enabled = bool(cfg.get("enabled", False)) and active
        topic_name = str(cfg.get("topic", "/cup_cleanup/trigger"))
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
            topic_name=topic_name,
            node_name=node_name,
            cup_id_to_robot_label=cup_id_to_robot_label,
        )
        bridge._initialize_ros()
        return bridge

    def _initialize_ros(self) -> None:
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
        self.publisher = self.node.create_publisher(String, self.topic_name, 10)
        print(f"[INFO] ROS2 trigger bridge enabled on topic {self.topic_name}")

    def close(self) -> None:
        if not self.enabled or self.node is None or self._rclpy is None:
            return
        try:
            self.node.destroy_node()
        finally:
            self.node = None
            self.publisher = None

    def _map_cup_id(self, source_cup_id: int) -> int:
        return int(self.cup_id_to_robot_label.get(int(source_cup_id), int(source_cup_id)))

    def _publish(self, payload: dict) -> None:
        if not self.enabled or self.publisher is None or self._std_msgs is None:
            return
        message = self._std_msgs()
        message.data = json.dumps(payload, ensure_ascii=True)
        self.publisher.publish(message)

    def process_predictions(self, predictions: list[dict], timestamp_now: float) -> None:
        if not self.enabled:
            return

        current_liquid_cups: set[int] = set()
        prediction_map = {int(item.get("cup_id", -1)): item for item in predictions}
        for item in predictions:
            source_cup_id = int(item.get("cup_id", -1))
            robot_cup_id = self._map_cup_id(source_cup_id)

            if (
                item.get("action") == "ASK"
                and item.get("reason") == "ask_once_triggered"
                and bool(item.get("selected_for_ask", False))
            ):
                ask_signature = (source_cup_id, int(item.get("ask_count", 0)))
                if ask_signature != self._last_ask_signature:
                    self._last_ask_signature = ask_signature
                    self._active_ask_cups.add(source_cup_id)
                    self._publish(
                        {
                            "event_type": "ASK_TRIGGER",
                            "cup_id": robot_cup_id,
                            "source_cup_id": source_cup_id,
                            "ask_reason": item.get("ask_reason", "none"),
                            "timestamp": float(timestamp_now),
                        }
                    )

            if bool(item.get("ask_cancelled_by_reuse", False)):
                cancel_signature = (source_cup_id, int(item.get("reuse_count", 0)))
                if source_cup_id in self._active_ask_cups and cancel_signature != self._last_cancel_signature:
                    self._last_cancel_signature = cancel_signature
                    self._active_ask_cups.discard(source_cup_id)
                    self._publish(
                        {
                            "event_type": "CANCEL_ASK_TRIGGER",
                            "cup_id": robot_cup_id,
                            "source_cup_id": source_cup_id,
                            "reason": "reuse_detected",
                            "timestamp": float(timestamp_now),
                        }
                    )

            if (
                item.get("state") == "NEEDS_LIQUID_CHECK"
                and bool(item.get("selected_for_liquid_check", False))
                and bool(item.get("verification_required", False))
            ):
                current_liquid_cups.add(source_cup_id)
                if source_cup_id not in self._active_liquid_check_cups:
                    self._publish(
                        {
                            "event_type": "ROBOT_LIQUID_CHECK_TRIGGER",
                            "cup_id": robot_cup_id,
                            "source_cup_id": source_cup_id,
                            "timestamp": float(timestamp_now),
                        }
                    )

        completed_liquid_results = {"EMPTY", "NON_EMPTY"}
        for source_cup_id in self._active_liquid_check_cups - current_liquid_cups:
            item = prediction_map.get(source_cup_id, {})
            liquid_check_result = str(item.get("liquid_check_result", "none"))
            if liquid_check_result in completed_liquid_results:
                continue
            robot_cup_id = self._map_cup_id(source_cup_id)
            self._publish(
                {
                    "event_type": "CANCEL_ROBOT_LIQUID_CHECK_TRIGGER",
                    "cup_id": robot_cup_id,
                    "source_cup_id": source_cup_id,
                    "reason": str(item.get("reason", "liquid_check_cancelled")),
                    "timestamp": float(timestamp_now),
                }
            )

        self._active_liquid_check_cups = current_liquid_cups
