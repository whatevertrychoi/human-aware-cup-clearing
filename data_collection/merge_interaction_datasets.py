from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from project_utils import ensure_parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge scene-specific interaction datasets into one CSV.")
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=[
            "data/processed/interaction_green.csv",
            "data/processed/interaction_red.csv",
            "data/processed/interaction_blue.csv",
            "data/processed/interaction_clutter.csv",
        ],
        help="Input CSV paths to merge",
    )
    parser.add_argument(
        "--out",
        default="data/processed/interaction_dataset_all.csv",
        help="Merged output CSV path",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    frames = []
    file_counts: list[tuple[str, int]] = []
    for input_path_str in args.inputs:
        input_path = Path(input_path_str)
        if not input_path.exists():
            print(f"[WARN] Missing input file, skipping: {input_path}")
            continue
        frame = pd.read_csv(input_path)
        if frame.empty:
            print(f"[WARN] Empty input file, skipping: {input_path}")
            continue
        frame["source_file"] = input_path.name
        file_counts.append((input_path.name, len(frame)))
        frames.append(frame)

    if not frames:
        print("[ERROR] No valid input CSV files were found.")
        return 1

    merged = pd.concat(frames, ignore_index=True)
    out_path = ensure_parent(args.out)
    merged.to_csv(out_path, index=False)

    print(f"Saved merged dataset to {out_path}")
    print(f"Total rows: {len(merged)}")
    print("\nInput file row counts:")
    for name, count in file_counts:
        print(f"{name}: {count}")
    print("\nLabel distribution:")
    print(merged["label"].value_counts().to_string())
    print("\nCup ID sample counts:")
    print(merged.groupby("cup_id").size().to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
