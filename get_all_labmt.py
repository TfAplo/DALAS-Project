#!/usr/bin/env python3
"""
Download all LabMT language dictionaries from Hedonometer API.
"""

import requests
import pandas as pd
from pathlib import Path
import time

LABMT_URL = "http://hedonometer.org/api/v1/words/"
DATA_DIR = Path("data")

# Language dictionaries to download
LANGUAGES = {
    "en": "labMT-en-v2",
    "de": "labMT-de-v2",
    "ko": "labMT-ko-v2",
    "es": "labMT-es-v2",
    "ru": "labMT-ru-v2",
    "zh": "labMT-zh-v2",
    "ar": "labMT-ar-v2",
    "pt": "labMT-pt-v2",
    "fr": "labMT-fr-v2",
    "uk": "labMT-uk-ru",  # Note: different naming for Ukrainian
}

def download_labmt(lang_code: str, wordlist_title: str):
    """Download a LabMT dictionary for a specific language."""
    output_file = DATA_DIR / f"labMT-{lang_code}-v2.csv"
    
    # Ukrainian has different naming
    if lang_code == "uk":
        output_file = DATA_DIR / "labMT-uk-ru.csv"
    
    print(f"Downloading {wordlist_title} ({lang_code})...")
    
    params = {
        "format": "json",
        "wordlist__title": wordlist_title,
        "limit": 15000,  # Should cover all words
        "offset": 0
    }
    
    try:
        resp = requests.get(LABMT_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        
        objects = data.get("objects", [])
        if not objects:
            print(f"  Warning: No objects found for {wordlist_title}")
            return False
            
        # Convert to DataFrame
        df = pd.DataFrame(objects)
        
        # Keep relevant columns
        if 'word' in df.columns and 'happs' in df.columns:
            # Keep word, happs, and stdDev if available
            cols_to_keep = ['word', 'happs']
            if 'stdDev' in df.columns:
                cols_to_keep.append('stdDev')
            df = df[cols_to_keep]
            
            df.to_csv(output_file, index=False)
            print(f"  ✓ Saved {len(df)} words to {output_file}")
            return True
        else:
            print(f"  ✗ Unexpected JSON structure. Columns: {df.columns}")
            return False
            
    except Exception as e:
        print(f"  ✗ Error downloading {wordlist_title}: {e}")
        return False

def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    print("Downloading all LabMT language dictionaries...\n")
    
    success_count = 0
    for lang_code, wordlist_title in LANGUAGES.items():
        if download_labmt(lang_code, wordlist_title):
            success_count += 1
        time.sleep(1)  # Be respectful to the API
    
    print(f"\n✓ Successfully downloaded {success_count}/{len(LANGUAGES)} dictionaries")

if __name__ == "__main__":
    main()

