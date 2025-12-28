"""
Verify and fix Spotify search cache issues.

The cache keys are generated using aggressive normalization which can cause:
1. Key collisions (different songs get same key)
2. Wrong matches (cache key doesn't match the actual song)

This script:
1. Loads the cache and source data
2. Verifies cache entries match their keys
3. Identifies problematic entries
4. Optionally fixes the cache
"""

import json
import pandas as pd
import re
import unicodedata
from pathlib import Path
from typing import Dict, Any, Tuple, List

DATA_DIR = Path("data")
CACHE_PATH = DATA_DIR / "cache" / "spotify_search_cache_final_dataset_additional.json"
SOURCE_DATA = DATA_DIR / "final_dataset_additional.csv"


def normalize_string(value: Any) -> str:
    """Normalize to a join key: ascii, lowercase, alnum words only."""
    if value is None or pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKD", str(value))
    text = text.encode("ascii", "ignore").decode("ascii")
    # Drop trailing metadata after " - " (common in Spotify track names)
    if " - " in text:
        text = text.split(" - ", 1)[0]
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def primary_artist_raw(value: Any) -> str:
    """Extract primary artist portion before features / joiners."""
    if value is None or pd.isna(value):
        return ""
    s = str(value).strip()
    # Remove featuring/with segments
    s = re.split(r"\b(feat\.|ft\.|featuring|with)\b", s, flags=re.IGNORECASE)[0]
    # Prefer primary when multiple are present
    for sep in [";", ",", "&", " x ", " X ", " and "]:
        if sep in s:
            s = s.split(sep, 1)[0]
    # Remove parentheses qualifiers like (FIN)
    s = re.sub(r"\s*[\(\[].*?[\)\]]\s*", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def make_track_key(title: Any, artist: Any) -> str:
    return f"{normalize_string(title)}||{normalize_string(primary_artist_raw(artist))}"


def verify_cache_entry(cache_key: str, cache_value: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Verify if a cache entry's key matches the stored song data.
    Returns (is_valid, error_message)
    """
    cached_name = cache_value.get("name", "")
    cached_artists = cache_value.get("artists", "")
    
    # Generate expected key from cached song data
    expected_key = make_track_key(cached_name, cached_artists)
    
    if cache_key != expected_key:
        return False, f"Key mismatch: expected '{expected_key}' but got '{cache_key}'"
    
    return True, ""


def find_problematic_entries(cache: Dict[str, Dict[str, Any]]) -> List[Tuple[str, Dict, str]]:
    """Find cache entries where the key doesn't match the stored song."""
    problems = []
    
    for key, value in cache.items():
        is_valid, error = verify_cache_entry(key, value)
        if not is_valid:
            problems.append((key, value, error))
    
    return problems


def check_source_data_matches(cache: Dict[str, Dict[str, Any]], source_df: pd.DataFrame) -> pd.DataFrame:
    """Check if source data matches correctly with cache."""
    results = []
    
    for idx, row in source_df.iterrows():
        title = row.get("title", "")
        artist = row.get("artist", "")
        
        if pd.isna(title) or pd.isna(artist) or not title or not artist:
            continue
        
        key = make_track_key(title, artist)
        cached = cache.get(key)
        
        if cached:
            cached_name = cached.get("name", "")
            cached_artists = cached.get("artists", "")
            
            # Check if the match seems correct
            title_match = normalize_string(title) == normalize_string(cached_name)
            artist_match = normalize_string(primary_artist_raw(artist)) == normalize_string(primary_artist_raw(cached_artists))
            
            results.append({
                "source_title": title,
                "source_artist": artist,
                "cache_key": key,
                "cached_name": cached_name,
                "cached_artists": cached_artists,
                "title_matches": title_match,
                "artist_matches": artist_match,
                "seems_correct": title_match and artist_match
            })
    
    return pd.DataFrame(results)


def main():
    print("=" * 80)
    print("Spotify Cache Verification")
    print("=" * 80)
    
    # 1. Load cache
    print("\n1. Loading cache...")
    if not CACHE_PATH.exists():
        print(f"   ERROR: Cache file not found: {CACHE_PATH}")
        return 1
    
    with open(CACHE_PATH, 'r', encoding='utf-8') as f:
        cache = json.load(f)
    
    print(f"   Loaded {len(cache)} cache entries")
    
    # 2. Verify cache entries
    print("\n2. Verifying cache entries...")
    problems = find_problematic_entries(cache)
    print(f"   Found {len(problems)} problematic entries")
    
    if problems:
        print("\n   Sample problematic entries:")
        for key, value, error in problems[:10]:
            print(f"   - Key: {key}")
            print(f"     Song: {value.get('name', 'N/A')} by {value.get('artists', 'N/A')}")
            print(f"     Error: {error}")
            print()
    
    # 3. Check source data matches
    print("\n3. Checking source data matches...")
    if SOURCE_DATA.exists():
        source_df = pd.read_csv(SOURCE_DATA)
        print(f"   Loaded {len(source_df)} source rows")
        
        matches_df = check_source_data_matches(cache, source_df)
        print(f"   Found {len(matches_df)} cache matches")
        
        if len(matches_df) > 0:
            incorrect = matches_df[~matches_df['seems_correct']]
            print(f"   Incorrect matches: {len(incorrect)}")
            
            if len(incorrect) > 0:
                print("\n   Sample incorrect matches:")
                for idx, row in incorrect.head(10).iterrows():
                    print(f"   - Source: '{row['source_title']}' by '{row['source_artist']}'")
                    print(f"     Cached: '{row['cached_name']}' by '{row['cached_artists']}'")
                    print(f"     Key: {row['cache_key']}")
                    print()
    else:
        print(f"   WARNING: Source data not found: {SOURCE_DATA}")
    
    # 4. Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Total cache entries: {len(cache)}")
    print(f"Problematic entries: {len(problems)}")
    if SOURCE_DATA.exists() and len(matches_df) > 0:
        print(f"Incorrect source matches: {len(incorrect) if 'incorrect' in locals() else 0}")
    
    print("\n" + "=" * 80)
    print("Recommendation:")
    print("=" * 80)
    if len(problems) > 0 or (SOURCE_DATA.exists() and len(incorrect) > 0 if 'incorrect' in locals() else False):
        print("The cache has issues. Consider:")
        print("1. Regenerating the cache with better matching logic")
        print("2. Using track_id for matching instead of normalized keys")
        print("3. Adding validation before saving cache entries")
    else:
        print("Cache appears to be valid!")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

