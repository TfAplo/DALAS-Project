import pandas as pd
from pathlib import Path
import glob
from difflib import SequenceMatcher
import re

# Constants
TOP25_DIR = Path("data/research_paper/Top25 Tracks/20220119")
SPOTIFY_TRACKS_PATH = Path("spotify_tracks.csv")
FINAL_DATASET_PATH = Path("data/final_dataset.csv")

def clean_text(s):
    if not isinstance(s, str):
        return ""
    # Remove content in brackets/parentheses
    s = re.sub(r"[\(\[].*?[\)\]]", "", s)
    # Remove common feature markers
    s = re.sub(r"\b(feat\.|ft\.|featuring|with)\b.*", "", s, flags=re.IGNORECASE)
    # Remove punctuation and non-alphanumeric
    s = re.sub(r"[^\w\s]", "", s)
    return s.lower().strip()

def load_top25_unique():
    print("Loading Top 25 Tracks...")
    all_files = glob.glob(str(TOP25_DIR / "*.xlsx"))
    all_data = []
    
    for f in all_files:
        try:
            df = pd.read_excel(f)
            df.columns = [str(c).lower().strip() for c in df.columns]
            if 'song' in df.columns and 'artist' in df.columns:
                 # Keep only necessary cols to save memory
                 subset = df[['song', 'artist']].copy()
                 # Add city/date context if needed later, but for now we just want unique songs
                 all_data.append(subset)
        except Exception as e:
            pass # Skip bad files
            
    if not all_data:
        return pd.DataFrame()
        
    combined = pd.concat(all_data, ignore_index=True)
    combined.dropna(subset=['song', 'artist'], inplace=True)
    
    # Deduplicate
    unique_songs = combined.drop_duplicates(subset=['song', 'artist']).copy()
    print(f"Found {len(unique_songs)} unique songs in Top 25 files.")
    
    unique_songs['clean_title'] = unique_songs['song'].apply(clean_text)
    unique_songs['clean_artist'] = unique_songs['artist'].apply(clean_text)
    
    return unique_songs

def load_spotify_data():
    print("Loading Spotify tracks...")
    if not SPOTIFY_TRACKS_PATH.exists():
        print("spotify_tracks.csv not found.")
        return pd.DataFrame()
        
    # Load specific columns to save memory if file is huge, but we need features
    # Columns in spotify_tracks.csv usually: 
    # artists, album_name, track_name, popularity, duration_ms, explicit, danceability, energy, key, loudness, mode, speechiness, acousticness, instrumentalness, liveness, valence, tempo, time_signature, track_genre
    df = pd.read_csv(SPOTIFY_TRACKS_PATH, on_bad_lines='skip')
    df['clean_title'] = df['track_name'].apply(clean_text)
    df['clean_artist'] = df['artists'].apply(clean_text)
    return df

def main():
    # 1. Load Top 25
    top25_df = load_top25_unique()
    if top25_df.empty:
        print("No Top 25 data found.")
        return

    # 2. Load Spotify Data
    spotify_df = load_spotify_data()
    if spotify_df.empty:
        print("No Spotify data found.")
        return

    # 3. Match
    print("Matching Top 25 songs against Spotify database...")
    
    # Exact match on cleaned title + cleaned artist
    # Create a lookup key
    spotify_df['match_key'] = spotify_df['clean_title'] + "|" + spotify_df['clean_artist']
    top25_df['match_key'] = top25_df['clean_title'] + "|" + top25_df['clean_artist']
    
    # Drop duplicates in spotify lookup to avoid explosion
    spotify_lookup = spotify_df.drop_duplicates(subset=['match_key'])
    
    matched = pd.merge(top25_df, spotify_lookup, on='match_key', how='inner', suffixes=('_scraped', ''))
    
    print(f"Matched {len(matched)} songs from Top 25 to Spotify data.")
    
    # 4. Load existing final_dataset
    if FINAL_DATASET_PATH.exists():
        final_df = pd.read_csv(FINAL_DATASET_PATH)
    else:
        final_df = pd.DataFrame()
        
    # 5. Append new matches
    # Map matched columns to final_dataset columns
    # final_dataset cols: track_name, artists, album_name, popularity, ... source, domain, ...
    
    # matched has columns from spotify_tracks.csv + song, artist from top25
    # We need to add 'source', 'domain', 'chart', etc. for consistency
    
    new_rows = matched.copy()
    new_rows['source'] = 'top25_research_paper'
    new_rows['domain'] = 'research_paper'
    new_rows['chart'] = 'top25_cities'
    new_rows['region'] = 'global' # or various
    new_rows['date'] = '2022-01-19' # approximate from folder name
    new_rows['position'] = -1 # Unknown/mixed
    new_rows['title'] = new_rows['song'] # original title
    new_rows['artist'] = new_rows['artist'] # original artist
    new_rows['scraped_at'] = pd.Timestamp.now().isoformat()
    new_rows['url'] = ''
    
    # Drop helper columns
    cols_to_drop = ['match_key', 'clean_title_scraped', 'clean_artist_scraped', 'clean_title', 'clean_artist', 'song', 'artist']
    # Note: 'song' and 'artist' (from top25) might be needed as 'title' and 'artist' cols in final dataset
    # but we already assigned them.
    
    # Filter to match final_df columns if it exists
    if not final_df.empty:
        common_cols = [c for c in final_df.columns if c in new_rows.columns]
        # Also keep columns that are in new_rows but maybe not in final_df if we want to expand?
        # For now, align to final_df schema + new ones
        
        # Ensure we have all columns from spotify_tracks that are useful
        useful_spotify_cols = ['track_name', 'artists', 'album_name', 'popularity', 'duration_ms', 'explicit', 
                               'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 
                               'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature', 'track_genre']
        
        for c in useful_spotify_cols:
            if c not in final_df.columns and c in new_rows.columns:
                 final_df[c] = pd.NA # Add col to final_df if missing
                 
        # Re-align
        new_rows_aligned = new_rows[final_df.columns.intersection(new_rows.columns)]
        final_df = pd.concat([final_df, new_rows_aligned], ignore_index=True)
    else:
        final_df = new_rows
        
    # Deduplicate final result
    print("Deduplicating merged dataset...")
    final_df['clean_t'] = final_df['track_name'].fillna(final_df['title']).apply(clean_text)
    final_df['clean_a'] = final_df['artists'].fillna(final_df['artist']).apply(clean_text)
    
    final_df = final_df.drop_duplicates(subset=['clean_t', 'clean_a'])
    final_df.drop(columns=['clean_t', 'clean_a'], inplace=True)
    
    print(f"Final dataset size: {len(final_df)}")
    final_df.to_csv(FINAL_DATASET_PATH, index=False)
    print(f"Saved to {FINAL_DATASET_PATH}")

if __name__ == "__main__":
    main()


