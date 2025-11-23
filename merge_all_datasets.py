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
from difflib import SequenceMatcher

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
    agg_dict['playlistname'] = list
    
    track_playlists = merged.groupby(['track_name', 'artists'], as_index=False).agg(agg_dict)
    
    return track_playlists


def clean_text(s: Any) -> str:
    """
    Aggressive text cleaning for fuzzy matching.
    Removes punctuation, 'feat.', extra whitespace, and lowercases.
    """
    if not isinstance(s, str):
        return ""
    # Remove content in brackets/parentheses (often feat. info or remastered info)
    s = re.sub(r"[\(\[].*?[\)\]]", "", s)
    # Remove common feature markers if not in brackets
    s = re.sub(r"\b(feat\.|ft\.|featuring|with)\b.*", "", s, flags=re.IGNORECASE)
    # Remove punctuation and non-alphanumeric
    s = re.sub(r"[^\w\s]", "", s)
    return s.lower().strip()


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def relaxed_merge(spotify_df: pd.DataFrame, scraped_df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs a relaxed merge between Spotify and Scraped data using:
    1. Exact match on aggressively cleaned Title.
    2. Fuzzy match on Artist (threshold > 0.6).
    """
    print("Preparing relaxed merge...")
    
    # Prepare keys
    spotify_df = spotify_df.copy()
    scraped_df = scraped_df.copy()
    
    spotify_df["clean_title"] = spotify_df["track_name"].apply(clean_text)
    spotify_df["clean_artist"] = spotify_df["artists"].apply(clean_text)
    
    scraped_df["clean_title"] = scraped_df["title"].apply(clean_text)
    scraped_df["clean_artist"] = scraped_df["artist"].apply(clean_text)
    
    # Build lookup from clean_title to list of (index, clean_artist) in scraped_df
    # We iterate Spotify rows and look up in this map (Scraped is smaller, but Spotify is what we want to enrich?)
    # Actually, we want to keep Scraped rows that have a match in Spotify.
    # Or Spotify rows that have a match in Scraped?
    # The goal is "final_dataset" -> likely Spotify tracks enriched with Chart data, or Chart data enriched with Spotify features?
    # The schema of final_dataset suggests: Spotify features + Chart info.
    # So we want Scraped Data rows that we can find audio features for.
    
    # Optimizing:
    # Group Spotify by clean_title
    spotify_map = {}
    for idx, row in spotify_df.iterrows():
        t = row["clean_title"]
        if not t: continue
        if t not in spotify_map:
            spotify_map[t] = []
        spotify_map[t].append((idx, row["clean_artist"]))
        
    matches = []
    
    print(f"Scanning {len(scraped_df)} scraped rows against {len(spotify_map)} unique Spotify titles...")
    
    # Iterate scraped data
    for s_idx, s_row in scraped_df.iterrows():
        s_title = s_row["clean_title"]
        s_artist = s_row["clean_artist"]
        
        if not s_title:
            continue
            
        # Look for title match (exact)
        candidates = spotify_map.get(s_title)
        
        # If no exact title match, try fuzzy title matching
        if not candidates:
            # Find similar titles (fuzzy match)
            for sp_title, sp_candidates in spotify_map.items():
                title_sim = similarity(s_title, sp_title)
                if title_sim > 0.85:  # Very similar titles
                    candidates = sp_candidates
                    break
        
        if candidates:
            # Check artists
            best_match_idx = -1
            best_score = -1.0
            
            for sp_idx, sp_artist in candidates:
                # If artist is empty, maybe just match title? (Dangerous)
                # Let's require some artist similarity
                if not s_artist or not sp_artist:
                    score = 0.5 # Ambiguous
                else:
                    score = similarity(s_artist, sp_artist)
                
                # Threshold - lowered to 0.4 for more matches
                if score > 0.4: # Very loose
                    if score > best_score:
                        best_score = score
                        best_match_idx = sp_idx
            
            if best_match_idx != -1:
                # Found a match!
                # Combine data.
                # We take the Spotify row and add Scraped columns
                sp_row = spotify_df.loc[best_match_idx].to_dict()
                sc_row = scraped_df.loc[s_idx].to_dict()
                
                combined = {**sp_row, **sc_row} # Overwrite Spotify cols with Scraped if collision?
                # Actually, we want Spotify features (danceability, etc) AND Chart info (position, date, etc)
                # Collisions: 'title', 'artist' vs 'track_name', 'artists'.
                # We keep both or rename.
                matches.append(combined)
    
    if not matches:
        print("No matches found with relaxed logic.")
        return pd.DataFrame()
        
    result = pd.DataFrame(matches)
    # Clean up temp cols
    result = result.drop(columns=["clean_title", "clean_artist"])
    return result


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

    # 3. Relaxed Merge
    final_df = relaxed_merge(spotify_df, scraped_df)

    # 4. Drop exact duplicate rows to keep final_dataset unique
    if not final_df.empty:
        before = len(final_df)
        final_df = final_df.drop_duplicates()
        after = len(final_df)
        dropped = before - after
        if dropped > 0:
            print(f"Removed {dropped} duplicate rows. Final dataset has {after} unique rows.")
        else:
            print(f"Merged dataset has {after} rows (no duplicates removed).")
    else:
        print("Merged dataset is empty.")
    
    # 5. Save
    output_path = DATA_DIR / "final_dataset.csv"
    final_df.to_csv(output_path, index=False)
    print(f"Saved final dataset to {output_path}")

if __name__ == "__main__":
    main()
