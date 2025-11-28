import requests
import json
import pandas as pd
from pathlib import Path
import time

LABMT_URL = "http://hedonometer.org/api/v1/words/"
OUTPUT_FILE = Path("data/labMT-en-v2.csv")

def download_labmt():
    print("Downloading LabMT-en-v2 dictionary...")
    
    # Try fetching with a large limit
    params = {
        "format": "json",
        "wordlist__title": "labMT-en-v2",
        "limit": 11000, # Should cover all ~10k words
        "offset": 0
    }
    
    try:
        resp = requests.get(LABMT_URL, params=params)
        resp.raise_for_status()
        data = resp.json()
        
        objects = data.get("objects", [])
        if not objects:
            print("No objects found in response.")
            return
            
        # Convert to DataFrame
        # We need 'word' and 'happs' (happiness score)
        df = pd.DataFrame(objects)
        
        # Keep relevant columns
        if 'word' in df.columns and 'happs' in df.columns:
            df = df[['word', 'happs', 'stdDev']]
            df.to_csv(OUTPUT_FILE, index=False)
            print(f"Saved {len(df)} words to {OUTPUT_FILE}")
        else:
            print("Unexpected JSON structure. Columns:", df.columns)
            
    except Exception as e:
        print(f"Error downloading LabMT: {e}")

if __name__ == "__main__":
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    download_labmt()


