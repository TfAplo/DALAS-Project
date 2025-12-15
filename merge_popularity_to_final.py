#!/usr/bin/env python3
"""
Merge popularity scores from master_dataset.csv into final_dataset.csv.

This script:
  1. Loads final_dataset.csv
  2. Loads master_dataset.csv
  3. Merges popularity scores based on title + artist matching
  4. Updates final_dataset.csv with popularity scores
"""

from pathlib import Path
import pandas as pd
import sys
import re

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
FINAL_DATASET_PATH = DATA_DIR / "final_dataset.csv"
MASTER_DATASET_PATH = DATA_DIR / "master_dataset.csv"


def normalize_text(s):
    """Normalize text for matching."""
    if pd.isna(s):
        return ""
    t = str(s).lower().strip()
    # Drop wrapping quotes and normalize whitespace
    t = re.sub(r'^[\'"]+|[\'"]+$', "", t).strip()
    t = re.sub(r"\s+", " ", t)
    return t


def main():
    """Merge popularity from master_dataset into final_dataset."""
    print("Loading datasets...")
    
    if not FINAL_DATASET_PATH.exists():
        print(f"Error: {FINAL_DATASET_PATH} not found", file=sys.stderr)
        sys.exit(1)
    
    if not MASTER_DATASET_PATH.exists():
        print(f"Error: {MASTER_DATASET_PATH} not found", file=sys.stderr)
        sys.exit(1)
    
    # Load datasets
    print(f"1. Loading {FINAL_DATASET_PATH}...")
    final_df = pd.read_csv(FINAL_DATASET_PATH)
    print(f"   Loaded {len(final_df)} rows")
    
    print(f"2. Loading {MASTER_DATASET_PATH}...")
    master_df = pd.read_csv(MASTER_DATASET_PATH)
    print(f"   Loaded {len(master_df)} rows")
    
    # Check if popularity column exists in master
    if 'popularity' not in master_df.columns:
        print("Error: 'popularity' column not found in master_dataset.csv", file=sys.stderr)
        sys.exit(1)
    
    # Create normalized keys for matching
    print("3. Preparing merge keys...")
    final_df['_norm_title'] = final_df['title'].apply(normalize_text)
    final_df['_norm_artist'] = final_df['artist'].apply(normalize_text)
    
    master_df['_norm_title'] = master_df['title'].apply(normalize_text)
    master_df['_norm_artist'] = master_df['artist'].apply(normalize_text)
    
    # Get popularity from master (keep only title, artist, popularity)
    master_pop = master_df[['_norm_title', '_norm_artist', 'popularity']].copy()
    master_pop = master_pop.dropna(subset=['_norm_title', '_norm_artist', 'popularity'])
    
    # Merge popularity
    print("4. Merging popularity scores...")
    before_count = final_df['popularity'].notna().sum()
    
    # Merge on normalized keys
    final_df = final_df.merge(
        master_pop,
        on=['_norm_title', '_norm_artist'],
        how='left',
        suffixes=('', '_from_master')
    )
    
    # Update popularity: use master's Spotify-derived value wherever it exists.
    # We always trust master (which we refresh directly from Spotify) over any
    # popularity carried in from earlier Kaggle merges.
    if 'popularity_from_master' in final_df.columns:
        # Coerce to numeric for robust comparisons/merging
        final_df['popularity'] = pd.to_numeric(final_df['popularity'], errors='coerce')
        final_df['popularity_from_master'] = pd.to_numeric(final_df['popularity_from_master'], errors='coerce')

        mask_has_master = final_df['popularity_from_master'].notna()
        final_df.loc[mask_has_master, 'popularity'] = final_df.loc[mask_has_master, 'popularity_from_master']
        final_df = final_df.drop(columns=['popularity_from_master'])
    
    after_count = final_df['popularity'].notna().sum()
    added = after_count - before_count
    
    print(f"   Before: {before_count} rows with popularity")
    print(f"   After:  {after_count} rows with popularity")
    print(f"   Added:  {added} new popularity scores")
    
    # Clean up temporary columns
    final_df = final_df.drop(columns=['_norm_title', '_norm_artist'])
    
    # Save updated dataset
    print("5. Saving updated final_dataset.csv...")
    final_df.to_csv(FINAL_DATASET_PATH, index=False)
    print(f"   [OK] Saved to {FINAL_DATASET_PATH}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()

