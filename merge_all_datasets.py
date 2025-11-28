"""
Merge all available chart/song datasets into a single unified CSV,
and join with Spotify track/playlist data (from Kaggle).

Inputs:
  - data/songs_latest.csv
  - data/playlists_latest.csv
  - data/official_charts/*.csv
  - spotify_tracks.csv (or downloaded via kagglehub)
  - spotify_dataset.csv (downloaded via kagglehub)

Outputs:
  - data/final_dataset.csv
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Dict, Any
import sys
import pandas as pd
import kagglehub
import os
import re

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"

TARGET_COLUMNS = [
    "source", "domain", "chart", "region", "date",
    "position", "title", "artist", "url", "scraped_at",
]


def coerce_int(val) -> Optional[int]:
    try:
        if pd.isna(val):
            return None
        s = str(val)
        digits = "".join(ch for ch in s if ch.isdigit())
        return int(digits) if digits else None
    except Exception:
        return None


def normalize_columns(df: pd.DataFrame, assumed_source: str = "") -> pd.DataFrame:
    """
    Normalize a DataFrame to TARGET_COLUMNS. Missing columns are added.
    """
    rename_map = {}
    if "source_domain" in df.columns and "domain" not in df.columns:
        rename_map["source_domain"] = "domain"
    df = df.rename(columns=rename_map)

    for col in TARGET_COLUMNS:
        if col not in df.columns:
            if col == "source" and assumed_source:
                df[col] = assumed_source
            else:
                df[col] = pd.NA

    # Keep only target columns and order them
    df = df[TARGET_COLUMNS]

    # Coerce position to int where possible
    df["position"] = df["position"].map(coerce_int)
    return df


def read_with_header_or_infer(path: Path, expected_cols: int = 10) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        # Check if columns match known schema (at least a few)
        intersection = set(df.columns) & set(TARGET_COLUMNS)
        if len(intersection) < 3:
            raise ValueError("Likely headerless CSV")
        return df
    except Exception:
        try:
            df = pd.read_csv(path, header=None)
            if df.shape[1] == expected_cols:
                df.columns = [
                    "source", "domain", "chart", "region", "date",
                    "position", "title", "artist", "url", "scraped_at",
                ]
            return df
        except Exception:
            return pd.DataFrame()


def load_scraped_data() -> pd.DataFrame:
    """
    Loads and merges all scraped chart data from data/ directory.
    """
    parts: List[pd.DataFrame] = []

    # songs_latest.csv
    songs_path = DATA_DIR / "songs_latest.csv"
    if songs_path.exists():
        df = read_with_header_or_infer(songs_path)
        if not df.empty:
            parts.append(normalize_columns(df))

    # additional_charts.csv (historical Billboard data)
    additional_path = DATA_DIR / "additional_charts.csv"
    if additional_path.exists():
        df = read_with_header_or_infer(additional_path)
        if not df.empty:
            parts.append(normalize_columns(df))

    # playlists_latest.csv
    playlists_path = DATA_DIR / "playlists_latest.csv"
    if playlists_path.exists():
        df = read_with_header_or_infer(playlists_path)
        if not df.empty:
            parts.append(normalize_columns(df))

    # official_charts/*.csv
    official_dir = DATA_DIR / "official_charts"
    if official_dir.exists():
        for csv_file in sorted(official_dir.glob("*.csv")):
            try:
                df = pd.read_csv(csv_file)
            except Exception:
                continue
            if df.empty:
                continue
            parts.append(normalize_columns(df, assumed_source="official_site"))

    if not parts:
        print("No scraped datasets found.", file=sys.stderr)
        return pd.DataFrame(columns=TARGET_COLUMNS)

    merged = pd.concat(parts, ignore_index=True)

    # Cleanup
    dedup_keys = ["domain", "chart", "date", "position", "title", "artist"]
    merged = merged.drop_duplicates(subset=dedup_keys, keep="first")

    return merged


def load_spotify_data() -> pd.DataFrame:
    """
    Loads Spotify tracks and playlists, merges them into track_playlists.
    Replicates logic from Data_collection.ipynb.
    """
    print("Loading Spotify data...")
    
    # 1. Load Tracks
    tracks_path = PROJECT_ROOT / "spotify_tracks.csv"
    if not tracks_path.exists():
        print("spotify_tracks.csv not found locally. Downloading...")
        path = kagglehub.dataset_download("maharshipandya/-spotify-tracks-dataset")
        # Find the csv in the downloaded path
        found = list(Path(path).glob("*.csv"))
        if found:
            tracks_path = found[0]
        else:
            raise FileNotFoundError("Could not find CSV in downloaded spotify tracks dataset")
            
    tracks = pd.read_csv(tracks_path, on_bad_lines="skip", engine="python")
    # Drop first 2 columns (index and track_id) as done in notebook
    tracks = tracks.drop(tracks.columns[:2], axis=1)
    
    # 2. Load Playlists
    # Try to find spotify_dataset.csv in likely locations
    playlists_path = PROJECT_ROOT / "spotify_dataset.csv"
    if not playlists_path.exists():
        # Try to download
        print("spotify_dataset.csv not found locally. Downloading...")
        path = kagglehub.dataset_download("andrewmvd/spotify-playlists")
        found = list(Path(path).glob("*.csv"))
        if found:
            playlists_path = found[0]
        else:
            # Fallback if file is named differently in the cache
            playlists_path = Path(path) / "spotify_dataset.csv"
    
    if not playlists_path.exists():
         raise FileNotFoundError("Could not find spotify_dataset.csv (playlists)")

    playlists = pd.read_csv(playlists_path, on_bad_lines="skip", engine="python")
    playlists.columns = playlists.columns.str.strip()
    
    # 3. Clean and Merge
    # Create DF from playlists
    df = pd.DataFrame()
    
    # Map standard names
    col_map = {c: c.replace('"', '') for c in playlists.columns}
    playlists = playlists.rename(columns=col_map)
    
    df["artistname"] = playlists["artistname"]
    df["trackname"] = playlists["trackname"]
    df["playlistname"] = playlists["playlistname"]
    
    df = df.dropna()
    
    # Filter to valid tracks (intersection)
    valid_tracks = set(zip(tracks['track_name'], tracks['artists']))
    
    # Check validity
    df_pairs = list(zip(df['trackname'], df['artistname']))
    df['is_valid'] = [pair in valid_tracks for pair in df_pairs]
    
    # Filter playlists where ALL tracks are valid (as per notebook logic)
    valid_playlists = df.groupby('playlistname')['is_valid'].all()
    valid_playlists_names = valid_playlists[valid_playlists].index
    df_filtered = df[df['playlistname'].isin(valid_playlists_names)].drop(columns='is_valid')
    
    # Merge both
    merged = df_filtered.merge(
        tracks,
        left_on=['trackname', 'artistname'],
        right_on=['track_name', 'artists'],
        how='outer'
    )
    
    # Group by track/artist to get list of playlists
    agg_dict = {col: 'first' for col in tracks.columns if col not in ['track_name', 'artists']}
    
    def collapse_playlists(s: pd.Series) -> str:
        # Drop NaNs, turn into strings, get uniques, sort for determinism
        values = sorted({str(v) for v in s.dropna()})
        return ";".join(values) if values else ""
    
    agg_dict['playlistname'] = collapse_playlists
    
    track_playlists = merged.groupby(['track_name', 'artists'], as_index=False).agg(agg_dict)
    
    return track_playlists


def clean_text(s: Any) -> str:
    """
    Aggressive text cleaning for normalization.
    Removes parentheses content, quotes, punctuation, 'feat.', extra whitespace, and lowercases.
    """
    if not isinstance(s, str):
        return ""
    # Remove content in brackets/parentheses (often feat. info or remastered info)
    s = re.sub(r"[\(\[].*?[\)\]]", "", s)
    # Remove quotes
    s = re.sub(r'["\']', "", s)
    # Remove common feature markers if not in brackets
    s = re.sub(r"\b(feat\.|ft\.|featuring|with)\b.*", "", s, flags=re.IGNORECASE)
    # Remove punctuation and non-alphanumeric
    s = re.sub(r"[^\w\s]", "", s)
    return s.lower().strip()


def primary_artist(s: Any) -> str:
    """
    Take the first artist from a ';'-separated list and clean it.
    If there is no ';', we just clean the whole string.
    """
    if not isinstance(s, str):
        return ""
    first = s.split(";", 1)[0]
    return clean_text(first)


def has_features(s: Any) -> bool:
    """
    Check if a string contains feature/remix indicators.
    Returns True if it contains 'feat', 'featuring', 'with', 'remix', etc.
    """
    if not isinstance(s, str):
        return False
    s_lower = s.lower()
    indicators = ['feat', 'featuring', 'ft.', 'ft ', 'with ', 'remix', 'mix', 'edit']
    return any(ind in s_lower for ind in indicators)


def build_track_master(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a track-level master table from appearance-level data.
    Aggregates all chart appearances for each unique track into a single row.
    """
    if df.empty:
        return pd.DataFrame()
    
    # Decide grouping key
    if 'track_id' in df.columns:
        group_key = ['track_id']
    else:
        group_key = ['track_name', 'artists']
    
    # Ensure required columns exist
    required_cols = ['track_name', 'artists', 'position', 'date', 'chart']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Warning: Missing columns for track master: {missing_cols}")
        return pd.DataFrame()
    
    # Convert date to datetime if it's not already
    df = df.copy()
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Basic scalar aggregations
    grouped = df.groupby(group_key, as_index=False)
    
    # Build aggregation dictionary
    agg_dict = {
        # If using track_id as key, keep name/artists for readability
        'track_name': ('track_name', 'first'),
        'artists': ('artists', 'first'),
        'peak_position': ('position', 'min'),          # best chart rank
        'first_chart_date': ('date', 'min'),
        'last_chart_date': ('date', 'max'),
        'n_chart_appearances': ('position', 'size'),   # total rows / appearances
        'n_distinct_charts': ('chart', 'nunique')      # number of distinct charts/sources
    }
    
    # Add any Spotify features from the first appearance (they should be the same per track)
    spotify_feature_cols = [
        'album_name', 'popularity', 'duration_ms', 'explicit',
        'danceability', 'energy', 'key', 'loudness', 'mode',
        'speechiness', 'acousticness', 'instrumentalness', 'liveness',
        'valence', 'tempo', 'time_signature', 'track_genre', 'playlistname'
    ]
    
    for col in spotify_feature_cols:
        if col in df.columns:
            agg_dict[col] = (col, 'first')
    
    # Use **agg_dict to unpack the dictionary for named aggregation
    track_master = grouped.agg(**agg_dict)
    
    # Add "peak chart(s)" (where the track achieved its best rank)
    # Compute peak position per track key
    peak_pos = df.groupby(group_key)['position'].min().reset_index(name='peak_position')
    
    # Merge back
    df_with_peak = df.merge(peak_pos, on=group_key, how='left')
    
    # Filter to only rows at the peak position
    at_peak = df_with_peak[df_with_peak['position'] == df_with_peak['peak_position']]
    
    # Collect the charts where the peak happened (as semicolon-separated string)
    def collapse_peak_charts(s: pd.Series) -> str:
        values = sorted({str(v) for v in s.dropna()})
        return ";".join(values) if values else ""
    
    peak_charts = (
        at_peak
        .groupby(group_key)['chart']
        .apply(collapse_peak_charts)
        .reset_index(name='peak_charts')
    )
    
    # Merge into track_master
    track_master = track_master.merge(peak_charts, on=group_key, how='left')
    
    return track_master


def deduplicate_chart_entries(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate chart entries (same source, chart, region, date, position, title, artist).
    When multiple Spotify matches exist for the same chart slot, keep the simpler/canonical version:
    - Prefer tracks without 'feat', 'featuring', 'with', 'remix' indicators
    - Prefer higher popularity if both have/not have features
    - Prefer simpler artist names (fewer ';' separators)
    """
    if df.empty:
        return df
    
    # Chart entry key: same chart slot
    chart_key_cols = ['source', 'domain', 'chart', 'region', 'date', 'position', 'title', 'artist']
    
    # Check which columns exist
    available_key_cols = [col for col in chart_key_cols if col in df.columns]
    
    if not available_key_cols:
        # No chart metadata, can't dedupe by chart entry
        return df
    
    # Add helper columns for ranking
    df = df.copy()
    df['_has_features_title'] = df.get('track_name', df.get('title', '')).apply(has_features)
    df['_has_features_artist'] = df.get('artists', df.get('artist', '')).apply(has_features)
    df['_has_features'] = df['_has_features_title'] | df['_has_features_artist']
    df['_artist_complexity'] = df.get('artists', df.get('artist', '')).apply(
        lambda x: str(x).count(';') if isinstance(x, str) else 0
    )
    df['_popularity'] = df.get('popularity', 0).fillna(0)
    
    # Sort to prefer: no features > has features, then higher popularity, then simpler artist
    df = df.sort_values(
        by=['_has_features', '_popularity', '_artist_complexity'],
        ascending=[True, False, True]  # False popularity = higher first
    )
    
    # Keep first (best) match per chart entry
    df_deduped = df.drop_duplicates(subset=available_key_cols, keep='first')
    
    # Remove helper columns
    df_deduped = df_deduped.drop(columns=['_has_features_title', '_has_features_artist', '_has_features', '_artist_complexity', '_popularity'])
    
    return df_deduped


def relaxed_merge(spotify_df: pd.DataFrame, scraped_df: pd.DataFrame) -> pd.DataFrame:
    """
    STRICT merge (no fuzzy logic):
    - Normalize track_name/title (remove parentheses, punctuation, lowercase, etc.).
    - Normalize artists/artist by taking only the first ';'-separated artist, then cleaning.
    - Inner-join on (norm_title, norm_artist).
    """
    print("Preparing strict merge on normalized title + primary artist...")
    
    spotify_df = spotify_df.copy()
    scraped_df = scraped_df.copy()
    
    # Normalize Spotify side
    spotify_df["norm_title"] = spotify_df["track_name"].apply(clean_text)
    spotify_df["norm_artist"] = spotify_df["artists"].apply(primary_artist)
    
    # Normalize scraped chart side
    scraped_df["norm_title"] = scraped_df["title"].apply(clean_text)
    scraped_df["norm_artist"] = scraped_df["artist"].apply(primary_artist)
    
    # Drop rows without a usable normalized title or artist
    spotify_df = spotify_df[(spotify_df["norm_title"] != "") & (spotify_df["norm_artist"] != "")]
    scraped_df = scraped_df[(scraped_df["norm_title"] != "") & (scraped_df["norm_artist"] != "")]
    
    print(f"Spotify rows after normalization: {len(spotify_df)}")
    print(f"Scraped rows after normalization: {len(scraped_df)}")
    
    # Inner join on normalized title + normalized primary artist
    merged = scraped_df.merge(
        spotify_df,
        on=["norm_title", "norm_artist"],
        how="inner",
        suffixes=("", "_spotify"),
    )
    
    print(f"Found {len(merged)} strict matches.")
    
    # We don't need the helper columns in the final dataset
    merged = merged.drop(columns=["norm_title", "norm_artist"])
    
    return merged


def main():
    # 1. Load Scraped Data
    print("Loading scraped charts data...")
    scraped_df = load_scraped_data()
    print(f"Loaded {len(scraped_df)} rows of scraped data.")
    
    if scraped_df.empty:
        print("Warning: Scraped data is empty.")

    # 2. Load Spotify Data (Track Playlists)
    try:
        spotify_df = load_spotify_data()
        print(f"Loaded {len(spotify_df)} rows of Spotify track/playlist data.")
    except Exception as e:
        print(f"Error loading Spotify data: {e}")
        return

    # 3. Strict Merge (normalized title + primary artist)
    final_df = relaxed_merge(spotify_df, scraped_df)

    # 4. Remove duplicate chart entries (keep canonical version without features)
    if not final_df.empty:
        before = len(final_df)
        final_df = deduplicate_chart_entries(final_df)
        after = len(final_df)
        dropped = before - after
        if dropped > 0:
            print(f"Removed {dropped} duplicate chart entries. Final dataset has {after} unique chart entries.")
        else:
            print(f"Merged dataset has {after} rows (no duplicate chart entries removed).")
        
        # Also drop any exact duplicate rows (bit-for-bit identical)
        before_exact = len(final_df)
        final_df = final_df.drop_duplicates()
        after_exact = len(final_df)
        if before_exact != after_exact:
            print(f"Removed {before_exact - after_exact} exact duplicate rows.")
    else:
        print("Merged dataset is empty.")
    
    # 5. Build track-level master table
    if not final_df.empty:
        print("\nBuilding track-level master table...")
        track_master = build_track_master(final_df)
        if not track_master.empty:
            print(f"Track master table has {len(track_master)} unique tracks.")
            track_master_path = DATA_DIR / "track_master.csv"
            track_master.to_csv(track_master_path, index=False)
            print(f"Saved track master table to {track_master_path}")
    
    # 6. Save appearance-level dataset
    output_path = DATA_DIR / "final_dataset.csv"
    final_df.to_csv(output_path, index=False)
    print(f"Saved appearance-level dataset to {output_path}")

if __name__ == "__main__":
    main()
