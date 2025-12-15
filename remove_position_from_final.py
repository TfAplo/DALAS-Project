#!/usr/bin/env python3
"""
Create a copy of final_dataset.csv without the 'position' column.

This script reads final_dataset.csv, removes the 'position' column,
and saves the result to final_dataset_no_position.csv.
"""

from pathlib import Path
import pandas as pd
import sys

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
INPUT_FILE = DATA_DIR / "final_dataset.csv"
OUTPUT_FILE = DATA_DIR / "final_dataset_no_position.csv"


def main():
    """Remove position column from final_dataset.csv and save as a copy."""
    if not INPUT_FILE.exists():
        print(f"Error: Input file not found: {INPUT_FILE}", file=sys.stderr)
        sys.exit(1)

    print(f"Reading {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)

    print(f"Original shape: {df.shape}")
    print(f"Columns: {len(df.columns)} columns")

    if "position" not in df.columns:
        print("Warning: 'position' column not found in the dataset.", file=sys.stderr)
        print("No changes made, but creating copy anyway.")
    else:
        # Remove the position column
        df = df.drop(columns=["position"])
        print(f"Removed 'position' column.")
        print(f"New shape: {df.shape}")

    # Save the result as a copy
    print(f"Saving copy to {OUTPUT_FILE}...")
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"[OK] Successfully created copy without 'position' column: {OUTPUT_FILE}")
    print(f"     Original file unchanged: {INPUT_FILE}")


if __name__ == "__main__":
    main()

