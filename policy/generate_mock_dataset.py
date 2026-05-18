from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from policy.expert_policy import expert_high_level_policy
from project_utils import ConfigError, ensure_parent, load_config


DEFAULT_CONFIG = ROOT / "configs" / "config.yaml"
LABELS = ["WAIT", "ASK", "CLEANUP_CANDIDATE"]


def generate_sample_for_label(label: str) -> dict:
    cup_id = random.randint(0, 2)
    base = {
        "scene_id": 0,
        "cup_id": cup_id,
        "x": round(random.uniform(0.2, 0.6), 3),
        "y": round(random.uniform(-0.2, 0.2), 3),
        "touch_count": random.randint(0, 4),
        "moved_recently": random.randint(0, 1),
        "distance_to_tray": round(random.uniform(0.1, 0.6), 3),
    }

    if label == "WAIT":
        base.update(
            {
                "hand_distance": round(random.uniform(0.01, 0.11), 3),
                "last_touched_time": round(random.uniform(0.0, 6.0), 2),
                "user_present": 1,
                "user_absent_time": 0.0,
            }
        )
    elif label == "ASK":
        if random.random() < 0.6:
            base.update(
                {
                    "hand_distance": round(random.uniform(0.13, 0.45), 3),
                    "last_touched_time": round(random.uniform(0.1, 9.5), 2),
                    "user_present": 1,
                    "user_absent_time": 0.0,
                }
            )
        else:
            base.update(
                {
                    "hand_distance": round(random.uniform(0.13, 0.55), 3),
                    "last_touched_time": round(random.uniform(10.0, 29.5), 2),
                    "user_present": 1,
                    "user_absent_time": 0.0,
                }
            )
    else:
        if random.random() < 0.5:
            base.update(
                {
                    "hand_distance": round(random.uniform(0.15, 0.7), 3),
                    "last_touched_time": round(random.uniform(0.0, 25.0), 2),
                    "user_present": 0,
                    "user_absent_time": round(random.uniform(10.5, 25.0), 2),
                }
            )
        else:
            base.update(
                {
                    "hand_distance": round(random.uniform(0.15, 0.7), 3),
                    "last_touched_time": round(random.uniform(30.5, 90.0), 2),
                    "user_present": random.randint(0, 1),
                    "user_absent_time": 0.0 if random.random() < 0.5 else round(random.uniform(0.0, 25.0), 2),
                }
            )
    return base


def build_dataset(config: dict, n_samples: int) -> pd.DataFrame:
    rows: list[dict] = []
    per_label = n_samples // len(LABELS)

    for label in LABELS:
        for _ in range(per_label):
            sample = generate_sample_for_label(label)
            sample["label"] = expert_high_level_policy(sample, config)
            rows.append(sample)

    while len(rows) < n_samples:
        label = LABELS[len(rows) % len(LABELS)]
        sample = generate_sample_for_label(label)
        sample["label"] = expert_high_level_policy(sample, config)
        rows.append(sample)

    for scene_id, row in enumerate(rows):
        row["scene_id"] = scene_id

    df = pd.DataFrame(rows)
    return df.sample(frac=1.0, random_state=42).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a mock dataset for cup-clearing high-level decisions.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to config.yaml")
    parser.add_argument("--out", required=True, help="Output CSV path")
    parser.add_argument("--n", type=int, default=1000, help="Number of samples to generate")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"[ERROR] {exc}")
        return 1

    dataset = build_dataset(config, args.n)
    output_path = ensure_parent(args.out)
    dataset.to_csv(output_path, index=False)
    print(f"Saved mock dataset to {output_path} ({len(dataset)} rows)")
    print(dataset["label"].value_counts().to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())

