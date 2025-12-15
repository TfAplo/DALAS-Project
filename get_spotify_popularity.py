"""
Fetch popularity scores for all songs in master_dataset.csv from Spotify API.

This script:
  1. Loads data/master_dataset.csv
  2. For each song, fetches popularity from Spotify API
  3. Uses track_id if available, otherwise searches by title + artist
  4. Handles rate limiting and retries
  5. Updates the dataset with popularity scores
  6. Saves back to master_dataset.csv (with backup)

Requirements:
  - spotipy library: pip install spotipy
  - Spotify API credentials (Client ID and Client Secret)
    Set as environment variables:
      SPOTIPY_CLIENT_ID
      SPOTIPY_CLIENT_SECRET
    Or create a .env file with these values
"""

from __future__ import annotations

import os
import sys
import time
import re
from pathlib import Path
from typing import Optional

import pandas as pd

try:
    import spotipy
    from spotipy.oauth2 import SpotifyClientCredentials
except ImportError:
    print(
        "Error: spotipy library not found. Install it with: pip install spotipy",
        file=sys.stderr,
    )
    sys.exit(1)

PROJECT_ROOT = Path(__file__).parent

# Try to load .env file
def load_env_file():
    """Load environment variables from .env file if it exists."""
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        try:
            loaded_count = 0
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, value = line.split("=", 1)
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")
                        if key and value:
                            os.environ[key] = value
                            loaded_count += 1
            if loaded_count > 0:
                print(f"Loaded {loaded_count} environment variable(s) from .env file")
        except Exception as e:
            print(f"Warning: Could not read .env file: {e}", file=sys.stderr)
    else:
        print(f"Note: .env file not found at {env_path}")

# Try python-dotenv first, fallback to manual loading
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    load_env_file()  # Manual .env loading

DATA_DIR = PROJECT_ROOT / "data"
MASTER_PATH = DATA_DIR / "master_dataset.csv"

# Rate limiting: Spotify allows 300 requests per minute.
# We'll be conservative and use ~50 requests per minute (1.2 seconds between requests).
REQUEST_DELAY = 1.2
BATCH_SIZE = 50  # Process in batches to allow for rate limit reset


def get_spotify_client() -> spotipy.Spotify:
    """Initialize and return a Spotify API client."""
    client_id = os.getenv("SPOTIPY_CLIENT_ID")
    client_secret = os.getenv("SPOTIPY_CLIENT_SECRET")

    if not client_id or not client_secret:
        env_path = PROJECT_ROOT / ".env"
        print(
            "Error: Spotify API credentials not found.\n"
            f"Please set SPOTIPY_CLIENT_ID and SPOTIPY_CLIENT_SECRET as environment variables,\n"
            f"or create a .env file at {env_path} with these values:\n"
            f"  SPOTIPY_CLIENT_ID=your_client_id\n"
            f"  SPOTIPY_CLIENT_SECRET=your_client_secret\n\n"
            "You can get credentials from: https://developer.spotify.com/dashboard",
            file=sys.stderr,
        )
        if env_path.exists():
            print(f"\nNote: .env file exists at {env_path} but credentials were not loaded.", file=sys.stderr)
            print("Please check that the file contains SPOTIPY_CLIENT_ID and SPOTIPY_CLIENT_SECRET.", file=sys.stderr)
        sys.exit(1)

    client_credentials_manager = SpotifyClientCredentials(
        client_id=client_id, client_secret=client_secret
    )
    return spotipy.Spotify(client_credentials_manager=client_credentials_manager)


def get_popularity_by_track_id(sp: spotipy.Spotify, track_id: str) -> Optional[int]:
    """Get popularity for a track using its Spotify track ID."""
    try:
        track = sp.track(track_id)
        return track.get("popularity")
    except Exception as e:
        print(f"  Error fetching track {track_id}: {e}", file=sys.stderr)
        return None


def search_track(
    sp: spotipy.Spotify, title: str, artist: str
) -> Optional[dict]:
    """
    Search for a track by title and artist.
    Returns the first matching track's data, or None if not found.
    """
    if not title or not artist:
        return None

    def clean_title_for_search(s: str) -> str:
        s = str(s).strip()
        # Remove wrapping quotes and normalize whitespace
        s = re.sub(r'^[\'"]+|[\'"]+$', "", s).strip()
        s = re.sub(r"\s+", " ", s)
        # Drop trailing metadata after " - " (e.g., remastered/edit versions)
        if " - " in s:
            s = s.split(" - ", 1)[0].strip()
        return s

    def clean_artist_for_search(s: str) -> str:
        s = str(s).strip()
        s = re.sub(r'^[\'"]+|[\'"]+$', "", s).strip()
        # Remove featuring/with segments
        s = re.split(r"\b(feat\.|ft\.|featuring|with)\b", s, flags=re.IGNORECASE)[0]
        # Prefer the primary artist when multiple are present
        for sep in [";", ",", "&", " x ", " X ", " and "]:
            if sep in s:
                s = s.split(sep, 1)[0]
        s = re.sub(r"\s+", " ", s).strip()
        return s

    # Clean up the search query
    title_q = clean_title_for_search(title)
    artist_q = clean_artist_for_search(artist)
    if not title_q or not artist_q:
        return None
    query = f"track:{title_q} artist:{artist_q}"
    try:
        results = sp.search(q=query, type="track", limit=1)
        tracks = results.get("tracks", {}).get("items", [])
        if tracks:
            return tracks[0]
    except Exception as e:
        print(f"  Error searching for '{title}' by {artist}: {e}", file=sys.stderr)
    return None


def get_popularity_for_row(
    sp: spotipy.Spotify, row: pd.Series, use_search: bool = True
) -> Optional[int]:
    """
    Get popularity for a single row.
    
    Args:
        sp: Spotify client
        row: DataFrame row with title, artist, and optionally track_id
        use_search: If True, search by title+artist when track_id is missing
    
    Returns:
        Popularity score (0-100) or None if not found
    """
    # First, try using track_id if available
    track_id = row.get("track_id")
    if pd.notna(track_id) and track_id:
        track_id_str = str(track_id).strip()
        # Check if it looks like a valid Spotify track ID (22 alphanumeric chars)
        if len(track_id_str) == 22 and track_id_str.isalnum():
            popularity = get_popularity_by_track_id(sp, track_id_str)
            if popularity is not None:
                return popularity

    # If track_id didn't work or wasn't available, try searching
    if use_search:
        title = str(row.get("title", "")).strip()
        artist = str(row.get("artist", "")).strip()
        if title and artist:
            track = search_track(sp, title, artist)
            if track:
                return track.get("popularity")

    return None


def main() -> int:
    if not MASTER_PATH.exists():
        print(f"Error: {MASTER_PATH} not found.", file=sys.stderr)
        return 1

    print("Loading master_dataset.csv...")
    df = pd.read_csv(MASTER_PATH)
    print(f"Loaded {len(df)} rows.")

    # For full accuracy, ignore any existing popularity values and refetch everything from Spotify.
    # This guarantees that master_dataset.csv reflects current Spotify popularity for every track we can match.
    df["popularity"] = pd.NA
    rows_needing_popularity = pd.Series(True, index=df.index)
    print("Will fetch popularity for ALL rows from Spotify (ignoring any existing values).")

    # Initialize Spotify client
    print("\nInitializing Spotify API client...")
    try:
        sp = get_spotify_client()
        print("Spotify client initialized successfully.")
    except Exception as e:
        print(f"Error initializing Spotify client: {e}", file=sys.stderr)
        return 1

    # Process rows that need popularity (here: all rows)
    rows_to_process = df[rows_needing_popularity].copy()
    print(f"\nFetching popularity for {len(rows_to_process)} rows...")
    print(f"Using delay of {REQUEST_DELAY} seconds between requests to respect rate limits.")

    fetched_count = 0
    failed_count = 0

    def safe_console(s: object) -> str:
        """Avoid Windows console UnicodeEncodeError by forcing ASCII-safe output."""
        try:
            return str(s).encode("ascii", "replace").decode("ascii")
        except Exception:
            return "<unprintable>"

    for idx, (row_idx, row) in enumerate(rows_to_process.iterrows(), 1):
        title = str(row.get("title", "")).strip()
        artist = str(row.get("artist", "")).strip()
        
        print(
            f"[{idx}/{len(rows_to_process)}] Fetching: '{safe_console(title)}' by {safe_console(artist)}...",
            end=" ",
        )

        popularity = get_popularity_for_row(sp, row, use_search=True)
        
        if popularity is not None:
            df.at[row_idx, "popularity"] = popularity
            fetched_count += 1
            print(f"OK - Popularity: {popularity}")
        else:
            failed_count += 1
            print("FAILED - Not found")

        # Rate limiting: delay between requests
        if idx < len(rows_to_process):
            time.sleep(REQUEST_DELAY)

        # Progress update every 50 rows
        if idx % 50 == 0:
            print(f"\nProgress: {idx}/{len(rows_to_process)} processed. "
                  f"Fetched: {fetched_count}, Failed: {failed_count}")

    print(f"\n\nCompleted!")
    print(f"  Successfully fetched: {fetched_count}")
    print(f"  Failed/not found: {failed_count}")
    print(f"  Total rows with popularity: {df['popularity'].notna().sum()}/{len(df)}")

    # Backup original file
    backup_path = MASTER_PATH.with_suffix(".backup_before_popularity.csv")
    if not backup_path.exists():
        df_before = pd.read_csv(MASTER_PATH)
        df_before.to_csv(backup_path, index=False)
        print(f"\nBacked up original to: {backup_path}")

    # Save updated dataset
    df.to_csv(MASTER_PATH, index=False)
    print(f"Saved updated dataset to: {MASTER_PATH}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

