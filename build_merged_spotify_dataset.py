#!/usr/bin/env python3
"""
Download and merge multiple Kaggle Spotify datasets into a unified lookup table.

This script:
1. Downloads multiple Kaggle datasets containing Spotify audio features
2. Merges them into a single unified dataset
3. Deduplicates by track_id (keeping highest popularity when duplicates exist)
4. Saves to a merged CSV file for use in enrichment scripts
"""

import sys
from pathlib import Path
import pandas as pd
from typing import List, Optional
import warnings

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_CSV = PROJECT_ROOT / "spotify_tracks_merged.csv"

# List of Kaggle datasets to download and merge
# Format: (dataset_name, expected_csv_filename_in_dataset)
# Note: You can add more datasets here. Common ones include:
# - "yamaerenay/spotify-dataset-19212020-600k-tracks" (600k+ tracks)
# - "vatsavm/spotify-tracks-db" (large collection)
# - "nadintamer/top-tracks-of-2017" (2017 tracks)
# - "tomigelo/spotify-audio-features" (various years)
KAGGLE_DATASETS = [
    # Primary dataset (already have this locally as spotify_tracks.csv)
    ("maharshipandya/-spotify-tracks-dataset", "spotify_tracks.csv"),
    
    # Additional large datasets with audio features
    # Uncomment to download more datasets (requires Kaggle API):
    # ("yamaerenay/spotify-dataset-19212020-600k-tracks", "tracks.csv"),
    # ("vatsavm/spotify-tracks-db", "tracks.csv"),
    # ("nadintamer/top-tracks-of-2017", "featuresdf.csv"),
]

# Alternative: Use local CSV files if Kaggle download fails
LOCAL_CSV_PATHS = [
    PROJECT_ROOT / "spotify_tracks.csv",  # Existing local file
    # Add more local CSV paths here if you have them
]


def normalize_track_id(track_id: str) -> Optional[str]:
    """Normalize track_id to standard format (22-char Spotify ID)."""
    if pd.isna(track_id):
        return None
    track_id = str(track_id).strip()
    # Extract 22-char ID if embedded in longer string
    if len(track_id) >= 22:
        # Try to find a 22-char substring that looks like a Spotify ID
        for i in range(len(track_id) - 21):
            candidate = track_id[i : i + 22]
            if candidate.isalnum():
                return candidate
    return track_id if len(track_id) == 22 else None


def load_kaggle_dataset(dataset_name: str, csv_filename: str) -> Optional[pd.DataFrame]:
    """Download and load a Kaggle dataset."""
    try:
        import kagglehub
        print(f"  Downloading {dataset_name}...")
        path = kagglehub.dataset_download(dataset_name)
        csv_path = Path(path) / csv_filename
        if not csv_path.exists():
            # Try to find any CSV file in the directory
            csv_files = list(Path(path).glob("*.csv"))
            if csv_files:
                csv_path = csv_files[0]
                print(f"    Using {csv_path.name} instead of {csv_filename}")
            else:
                print(f"    [WARN] {csv_filename} not found in dataset")
                return None
        
        print(f"    Loading {csv_path}...")
        df = pd.read_csv(csv_path, on_bad_lines="skip", engine="python")
        print(f"    Loaded {len(df)} rows")
        return df
    except Exception as e:
        print(f"    [ERROR] Failed to download {dataset_name}: {e}")
        return None


def load_local_csv(csv_path: Path) -> Optional[pd.DataFrame]:
    """Load a local CSV file."""
    if not csv_path.exists():
        return None
    try:
        print(f"  Loading local file: {csv_path.name}...")
        df = pd.read_csv(csv_path, on_bad_lines="skip", engine="python")
        print(f"    Loaded {len(df)} rows")
        return df
    except Exception as e:
        print(f"    [ERROR] Failed to load {csv_path}: {e}")
        return None


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names across different datasets."""
    # Common column name variations
    column_mapping = {
        "id": "track_id",
        "track_id": "track_id",
        "spotify_id": "track_id",
        "track_uri": "track_id",  # Extract ID from URI
        "uri": "track_id",
        
        "name": "track_name",
        "track_name": "track_name",
        "title": "track_name",
        "song": "track_name",
        
        "artist": "artists",
        "artists": "artists",
        "artist_name": "artists",
        "artist_names": "artists",
        
        "album": "album_name",
        "album_name": "album_name",
    }
    
    # Rename columns
    df = df.rename(columns=column_mapping)
    
    # Extract track_id from URI if needed
    if "track_id" in df.columns:
        df["track_id"] = df["track_id"].astype(str)
        # If track_id looks like a URI, extract the ID
        mask = df["track_id"].str.startswith("spotify:track:")
        df.loc[mask, "track_id"] = df.loc[mask, "track_id"].str.replace("spotify:track:", "", regex=False)
        mask = df["track_id"].str.startswith("https://open.spotify.com/track/")
        df.loc[mask, "track_id"] = df.loc[mask, "track_id"].str.replace("https://open.spotify.com/track/", "", regex=False).str.split("?").str[0]
    
    return df


def extract_audio_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract and standardize audio feature columns."""
    # Required columns for audio features
    audio_cols = [
        "track_id",
        "artists",
        "album_name",
        "track_name",
        "popularity",
        "duration_ms",
        "explicit",
        "danceability",
        "energy",
        "key",
        "loudness",
        "mode",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence",
        "tempo",
        "time_signature",
        "track_genre",
    ]
    
    # Keep only columns that exist
    available_cols = [c for c in audio_cols if c in df.columns]
    if "track_id" not in available_cols:
        print(f"    [WARN] No track_id column found, skipping this dataset")
        return pd.DataFrame()
    
    df_subset = df[available_cols].copy()
    
    # Normalize track_id
    df_subset["track_id"] = df_subset["track_id"].apply(normalize_track_id)
    df_subset = df_subset[df_subset["track_id"].notna()]
    df_subset = df_subset[df_subset["track_id"].str.len() == 22]
    
    # Convert popularity to numeric
    if "popularity" in df_subset.columns:
        df_subset["popularity"] = pd.to_numeric(df_subset["popularity"], errors="coerce")
    
    return df_subset


def merge_datasets(datasets: List[pd.DataFrame]) -> pd.DataFrame:
    """Merge multiple datasets, deduplicating by track_id (keep highest popularity)."""
    if not datasets:
        return pd.DataFrame()
    
    print(f"\nMerging {len(datasets)} datasets...")
    
    # Combine all datasets
    combined = pd.concat(datasets, ignore_index=True)
    print(f"  Combined: {len(combined)} total rows")
    
    # Deduplicate by track_id, keeping row with highest popularity
    if "popularity" in combined.columns:
        combined = combined.sort_values("popularity", ascending=False, na_position="last")
        combined = combined.drop_duplicates(subset=["track_id"], keep="first")
    else:
        combined = combined.drop_duplicates(subset=["track_id"], keep="first")
    
    print(f"  After deduplication: {len(combined)} unique tracks")
    
    return combined


def main():
    """Main function to download, merge, and save datasets."""
    print("=" * 60)
    print("Building Merged Spotify Dataset")
    print("=" * 60)
    
    datasets = []
    
    # 1. Try loading from Kaggle
    print("\n1. Downloading Kaggle datasets...")
    for dataset_name, csv_filename in KAGGLE_DATASETS:
        df = load_kaggle_dataset(dataset_name, csv_filename)
        if df is not None and not df.empty:
            df = standardize_columns(df)
            df = extract_audio_features(df)
            if not df.empty:
                datasets.append(df)
                print(f"    [OK] Added {len(df)} tracks from {dataset_name}")
    
    # 2. Load local CSV files
    print("\n2. Loading local CSV files...")
    for csv_path in LOCAL_CSV_PATHS:
        df = load_local_csv(csv_path)
        if df is not None and not df.empty:
            df = standardize_columns(df)
            df = extract_audio_features(df)
            if not df.empty:
                datasets.append(df)
                print(f"    [OK] Added {len(df)} tracks from {csv_path.name}")
    
    if not datasets:
        print("\n[ERROR] No datasets loaded! Please check:")
        print("  - Kaggle API authentication (kagglehub)")
        print("  - Local CSV file paths")
        return 1
    
    # 3. Merge datasets
    merged = merge_datasets(datasets)
    
    if merged.empty:
        print("\n[ERROR] Merged dataset is empty!")
        return 1
    
    # 4. Save merged dataset
    print(f"\n3. Saving merged dataset to {OUTPUT_CSV}...")
    merged.to_csv(OUTPUT_CSV, index=False)
    
    # 5. Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Total unique tracks: {len(merged)}")
    
    if "popularity" in merged.columns:
        pop_count = merged["popularity"].notna().sum()
        print(f"Tracks with popularity: {pop_count} ({pop_count/len(merged)*100:.1f}%)")
    
    audio_cols = ["danceability", "energy", "valence", "tempo", "loudness", 
                  "acousticness", "instrumentalness", "speechiness", "liveness"]
    audio_complete = merged[audio_cols].apply(lambda x: pd.to_numeric(x, errors="coerce").notna().all(), axis=1)
    complete_count = audio_complete.sum()
    print(f"Tracks with all 9 audio features: {complete_count} ({complete_count/len(merged)*100:.1f}%)")
    
    print(f"\n[OK] Merged dataset saved to: {OUTPUT_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

