"""
Verify if final.csv has incorrect song matches due to bad cache keys.

This script checks if songs in final.csv have correct Spotify data by:
1. Comparing source title/artist with track_name/artists
2. Checking if the normalization would match correctly
3. Flagging potential mismatches
"""

import pandas as pd
import re
import unicodedata
from pathlib import Path
from typing import Any

DATA_DIR = Path("data")
FINAL_CSV = DATA_DIR / "final.csv"


def normalize_string(value: Any) -> str:
    """Normalize to a join key: ascii, lowercase, alnum words only."""
    if value is None or pd.isna(value):
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
    if value is None or pd.isna(value):
        return ""
    s = str(value).strip()
    s = re.split(r"\b(feat\.|ft\.|featuring|with)\b", s, flags=re.IGNORECASE)[0]
    for sep in [";", ",", "&", " x ", " X ", " and "]:
        if sep in s:
            s = s.split(sep, 1)[0]
    s = re.sub(r"\s*[\(\[].*?[\)\]]\s*", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def check_match(source_title: str, source_artist: str, 
                spotify_name: str, spotify_artists: str) -> tuple[bool, str]:
    """
    Check if source and Spotify data match.
    Returns (matches, reason)
    """
    if pd.isna(source_title) or pd.isna(source_artist):
        return False, "Missing source data"
    
    if pd.isna(spotify_name) or pd.isna(spotify_artists):
        return False, "Missing Spotify data"
    
    # Normalize both
    source_title_norm = normalize_string(source_title)
    source_artist_norm = normalize_string(primary_artist_raw(source_artist))
    spotify_name_norm = normalize_string(spotify_name)
    spotify_artists_norm = normalize_string(primary_artist_raw(spotify_artists))
    
    title_match = source_title_norm == spotify_name_norm
    artist_match = source_artist_norm == spotify_artists_norm
    
    if title_match and artist_match:
        return True, "Perfect match"
    elif title_match:
        return False, f"Title matches but artist differs: '{source_artist}' vs '{spotify_artists}'"
    elif artist_match:
        return False, f"Artist matches but title differs: '{source_title}' vs '{spotify_name}'"
    else:
        return False, f"Both differ: '{source_title}'/'{source_artist}' vs '{spotify_name}'/'{spotify_artists}'"


def main():
    print("=" * 80)
    print("Verifying final.csv Song Matches")
    print("=" * 80)
    
    # Load data
    print("\n1. Loading final.csv...")
    df = pd.read_csv(FINAL_CSV)
    print(f"   Loaded {len(df)} rows")
    
    # Check matches
    print("\n2. Checking song matches...")
    results = []
    
    for idx, row in df.iterrows():
        source_title = row.get("title", "")
        source_artist = row.get("artist", "")
        spotify_name = row.get("track_name", "")
        spotify_artists = row.get("artists", "")
        
        # Only check rows with both source and Spotify data
        if (pd.notna(source_title) and pd.notna(source_artist) and 
            pd.notna(spotify_name) and pd.notna(spotify_artists)):
            
            matches, reason = check_match(source_title, source_artist, 
                                         spotify_name, spotify_artists)
            
            results.append({
                "index": idx,
                "source_title": source_title,
                "source_artist": source_artist,
                "spotify_name": spotify_name,
                "spotify_artists": spotify_artists,
                "track_id": row.get("track_id", ""),
                "matches": matches,
                "reason": reason
            })
    
    results_df = pd.DataFrame(results)
    
    if len(results_df) > 0:
        matches = results_df[results_df['matches']]
        mismatches = results_df[~results_df['matches']]
        
        print(f"   Total checked: {len(results_df)}")
        print(f"   Matches: {len(matches)} ({100*len(matches)/len(results_df):.1f}%)")
        print(f"   Mismatches: {len(mismatches)} ({100*len(mismatches)/len(results_df):.1f}%)")
        
        if len(mismatches) > 0:
            print("\n3. Sample mismatches:")
            for idx, row in mismatches.head(20).iterrows():
                print(f"\n   Row {row['index']}:")
                print(f"     Source: '{row['source_title']}' by '{row['source_artist']}'")
                print(f"     Spotify: '{row['spotify_name']}' by '{row['spotify_artists']}'")
                print(f"     Track ID: {row['track_id']}")
                print(f"     Reason: {row['reason']}")
    else:
        print("   No rows with both source and Spotify data found")
    
    # Save report
    if len(results_df) > 0:
        report_path = DATA_DIR / "song_match_verification_report.csv"
        results_df.to_csv(report_path, index=False)
        print(f"\n4. Saved detailed report to: {report_path}")
    
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    if len(results_df) > 0:
        print(f"Total songs checked: {len(results_df)}")
        print(f"Correct matches: {len(matches)}")
        print(f"Potential mismatches: {len(mismatches)}")
        
        if len(mismatches) > 0:
            print("\nRecommendation:")
            print("If there are many mismatches, consider re-running the enrichment")
            print("script with the fixed cache to correct the data.")
    else:
        print("No data to verify")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

