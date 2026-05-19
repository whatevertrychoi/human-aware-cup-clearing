from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from project_utils import ConfigError, get_required, load_config


def expert_high_level_policy(cup_feature: dict, config: dict) -> str:
    touch_threshold = float(get_required(config, ["tracking", "touch_threshold"]))
    tracking_cfg = config.get("tracking", {}) if isinstance(config, dict) else {}
    policy_cfg = config.get("policy", {}) if isinstance(config, dict) else {}
    recent_touch_threshold = float(policy_cfg.get("recent_touch_threshold", 10.0))
    cleanup_time_threshold = float(policy_cfg.get("cleanup_time_threshold", 30.0))
    user_absence_threshold = float(tracking_cfg.get("user_absence_threshold", 10.0))
    stationary_threshold = float(policy_cfg.get("stationary_threshold", 3.0))

    is_active_cup = bool(cup_feature.get("is_active_cup", 0))
    used_cup_candidate = bool(cup_feature.get("used_cup_candidate", 0))
    user_present = int(cup_feature.get("user_present", 0))
    hand_distance = float(cup_feature.get("hand_distance", 999.0))
    last_touched_time = float(cup_feature.get("last_touched_time", 999.0))
    user_absent_time = float(cup_feature.get("user_absent_time", 0.0))
    time_since_release = float(cup_feature.get("time_since_release", 999.0))
    stationary_time = float(cup_feature.get("stationary_time", 0.0))

    if is_active_cup and hand_distance < touch_threshold:
        return "WAIT"

    if used_cup_candidate and user_present == 1 and time_since_release < recent_touch_threshold:
        return "ASK"

    if user_present == 1 and not used_cup_candidate:
        return "IDLE"

    if user_present == 0 and user_absent_time > user_absence_threshold and stationary_time > stationary_threshold:
        return "CLEANUP_CANDIDATE"

    if user_present == 0 and last_touched_time > cleanup_time_threshold and stationary_time > stationary_threshold:
        return "CLEANUP_CANDIDATE"

    return "IDLE"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run expert high-level cup clearing policy on a JSON sample.")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument(
        "--sample",
        default='{"hand_distance":0.2,"user_present":1,"last_touched_time":5,"user_absent_time":0,"is_active_cup":0,"used_cup_candidate":1,"time_since_release":2.0,"stationary_time":4.0}',
        help="JSON dict representing a single cup feature",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = load_config(args.config)
        sample = json.loads(args.sample)
    except (ConfigError, json.JSONDecodeError) as exc:
        print(f"[ERROR] {exc}")
        return 1

    print(expert_high_level_policy(sample, config))
    return 0


if __name__ == "__main__":
    sys.exit(main())
