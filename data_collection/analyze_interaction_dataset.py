from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze a recorded interaction dataset CSV.")
    parser.add_argument("--data", required=True, help="Path to interaction dataset CSV")
    return parser.parse_args()


def print_describe(df: pd.DataFrame, column: str) -> None:
    if column not in df.columns:
        print(f"\n{column}: missing")
        return
    print(f"\n{column}:")
    print(df[column].describe().to_string())


FEATURE_COLUMNS = [
    "hand_distance",
    "last_touched_time",
    "user_absent_time",
    "time_near_cup",
    "time_since_release",
    "release_count",
    "cup_motion_distance",
    "stationary_time",
]


def main() -> int:
    args = parse_args()
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"[ERROR] Dataset not found: {data_path}")
        return 1

    df = pd.read_csv(data_path)
    print(f"Dataset: {data_path}")
    print(f"Row count: {len(df)}")
    print("\nLabel distribution:")
    if "label" in df.columns:
        print(df["label"].value_counts().to_string())
    else:
        print("label column missing")

    print("\nCup ID sample count:")
    if "cup_id" in df.columns:
        print(df.groupby("cup_id").size().to_string())
    else:
        print("cup_id column missing")

    if "source_file" in df.columns:
        print("\nSource file row count:")
        print(df["source_file"].value_counts().to_string())

    if "label" in df.columns:
        available = [feature for feature in FEATURE_COLUMNS if feature in df.columns]
        if available:
            print("\nLabel-wise feature mean/std/min/max:")
            stats = df.groupby("label")[available].agg(["mean", "std", "min", "max"])
            print(stats.to_string())

    for feature in FEATURE_COLUMNS:
        print_describe(df, feature)
    return 0


if __name__ == "__main__":
    sys.exit(main())
