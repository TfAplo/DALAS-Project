"""
Fix Spotify cache by regenerating keys from actual song data.

The cache has keys generated from source data, but the stored songs don't match.
This script regenerates the cache with correct keys.
"""

import json
import re
import unicodedata
from pathlib import Path
from typing import Dict, Any

DATA_DIR = Path("data")
CACHE_PATH = DATA_DIR / "cache" / "spotify_search_cache_final_dataset_additional.json"
BACKUP_PATH = DATA_DIR / "cache" / "spotify_search_cache_final_dataset_additional.json.backup"


def normalize_string(value: Any) -> str:
    """Normalize to a join key: ascii, lowercase, alnum words only."""
    if value is None:
        return ""
    text = unicodedata.normalize("NFKD", str(value))
    text = text.encode("ascii", "ignore").decode("ascii")
    if " - " in text:
        text = text.split(" - ", 1)[0]
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def primary_artist_raw(value: Any) -> str:
    """Extract primary artist portion before features / joiners."""
    if value is None:
        return ""
    s = str(value).strip()
    s = re.split(r"\b(feat\.|ft\.|featuring|with)\b", s, flags=re.IGNORECASE)[0]
    for sep in [";", ",", "&", " x ", " X ", " and "]:
        if sep in s:
            s = s.split(sep, 1)[0]
    s = re.sub(r"\s*[\(\[].*?[\)\]]\s*", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def make_track_key(title: Any, artist: Any) -> str:
    return f"{normalize_string(title)}||{normalize_string(primary_artist_raw(artist))}"


def fix_cache(cache: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Regenerate cache with correct keys based on stored song data."""
    fixed_cache = {}
    duplicates = {}
    fixed_count = 0
    duplicate_count = 0
    
    for old_key, value in cache.items():
        # Generate new key from actual song data
        song_name = value.get("name", "")
        song_artists = value.get("artists", "")
        
        if not song_name or not song_artists:
            # Skip entries without valid data
            continue
        
        new_key = make_track_key(song_name, song_artists)
        
        if new_key in fixed_cache:
            # Handle duplicates - keep the one with higher popularity
            duplicate_count += 1
            existing_pop = fixed_cache[new_key].get("popularity", 0) or 0
            new_pop = value.get("popularity", 0) or 0
            
            if new_pop > existing_pop:
                duplicates[new_key] = duplicates.get(new_key, []) + [fixed_cache[new_key]]
                fixed_cache[new_key] = value
            else:
                duplicates[new_key] = duplicates.get(new_key, []) + [value]
        else:
            fixed_cache[new_key] = value
            if old_key != new_key:
                fixed_count += 1
    
    print(f"Fixed {fixed_count} keys")
    print(f"Found {duplicate_count} duplicate keys (kept highest popularity)")
    
    if duplicates:
        print(f"\nDuplicate keys (kept best, others discarded):")
        for key, dup_list in list(duplicates.items())[:10]:
            print(f"  {key}: {len(dup_list) + 1} entries")
    
    return fixed_cache


def main():
    print("=" * 80)
    print("Fixing Spotify Cache")
    print("=" * 80)
    
    # 1. Load cache
    print("\n1. Loading cache...")
    if not CACHE_PATH.exists():
        print(f"   ERROR: Cache file not found: {CACHE_PATH}")
        return 1
    
    with open(CACHE_PATH, 'r', encoding='utf-8') as f:
        cache = json.load(f)
    
    print(f"   Loaded {len(cache)} cache entries")
    
    # 2. Backup original
    print("\n2. Creating backup...")
    import shutil
    shutil.copy(CACHE_PATH, BACKUP_PATH)
    print(f"   Backup created: {BACKUP_PATH}")
    
    # 3. Fix cache
    print("\n3. Fixing cache keys...")
    fixed_cache = fix_cache(cache)
    
    # 4. Save fixed cache
    print("\n4. Saving fixed cache...")
    with open(CACHE_PATH, 'w', encoding='utf-8') as f:
        json.dump(fixed_cache, f, ensure_ascii=True, indent=2)
    
    print(f"   Saved {len(fixed_cache)} entries to {CACHE_PATH}")
    
    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)
    print("\nNote: The cache keys are now based on the actual song data.")
    print("You may want to re-run the enrichment script to fix any incorrect")
    print("matches in final.csv that were caused by the old cache keys.")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

