#!/usr/bin/env python3
"""
Remove the 'position' column from master_dataset.csv.

This script reads master_dataset.csv, removes the 'position' column,
and saves the result back to the same file (or optionally to a new file).
"""

from pathlib import Path
import pandas as pd
import sys

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
INPUT_FILE = DATA_DIR / "master_dataset.csv"
OUTPUT_FILE = DATA_DIR / "master_dataset_no_position.csv"  # Create a copy


def main():
    """Remove position column from master_dataset.csv."""
    if not INPUT_FILE.exists():
        print(f"Error: Input file not found: {INPUT_FILE}", file=sys.stderr)
        sys.exit(1)

    print(f"Reading {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)

    print(f"Original shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    if "position" not in df.columns:
        print("Warning: 'position' column not found in the dataset.", file=sys.stderr)
        print("No changes made.")
        return

    # Remove the position column
    df = df.drop(columns=["position"])

    print(f"New shape: {df.shape}")
    print(f"Removed 'position' column.")

    # Save the result
    print(f"Saving to {OUTPUT_FILE}...")
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"✓ Successfully created copy without 'position' column: {OUTPUT_FILE}")
    print(f"  Original file unchanged: {INPUT_FILE}")


if __name__ == "__main__":
    main()

