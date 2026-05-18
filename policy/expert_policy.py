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
    if float(cup_feature["hand_distance"]) < touch_threshold:
        return "WAIT"

    if int(cup_feature["user_present"]) == 1 and float(cup_feature["last_touched_time"]) < 10:
        return "ASK"

    if int(cup_feature["user_present"]) == 0 and float(cup_feature["user_absent_time"]) > 10:
        return "CLEANUP_CANDIDATE"

    if float(cup_feature["last_touched_time"]) > 30:
        return "CLEANUP_CANDIDATE"

    return "ASK"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run expert high-level cup clearing policy on a JSON sample.")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument(
        "--sample",
        default='{"hand_distance":0.2,"user_present":1,"last_touched_time":5,"user_absent_time":0}',
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

