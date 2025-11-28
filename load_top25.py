import pandas as pd
from pathlib import Path
import glob

TOP25_DIR = Path("data/research_paper/Top25 Tracks/20220119")

def load_top25_tracks():
    all_files = glob.glob(str(TOP25_DIR / "*.xlsx"))
    all_data = []
    
    print(f"Found {len(all_files)} files in {TOP25_DIR}")
    
    for f in all_files:
        city_name = Path(f).stem
        try:
            # Read excel file, assuming headers are on the first row
            # Adjust header row if necessary after inspecting one file
            df = pd.read_excel(f)
            
            # Standardize column names (convert to lower case)
            df.columns = [str(c).lower().strip() for c in df.columns]
            
            # We expect columns like 'artist', 'track name' or similar.
            # Let's verify content by looking for common columns.
            # Usually these files have 'artist_name', 'track_name', etc.
            
            # Add source info
            df['city'] = city_name
            
            all_data.append(df)
        except Exception as e:
            print(f"Error reading {f}: {e}")
            
    if not all_data:
        return pd.DataFrame()
        
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"Combined Top 25 data has {len(combined_df)} rows.")
    return combined_df

if __name__ == "__main__":
    df = load_top25_tracks()
    print(df.head())
    print(df.columns)


