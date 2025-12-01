"""
Analyze correlations between genre-level happiness profiles and MXMH survey data.

This script:
1. Loads genre happiness profiles from our dataset
2. Loads MXMH survey data
3. Calculates average mental health scores by favorite genre
4. Explores correlations between genre happiness/arousal and mental health outcomes
"""

import pandas as pd
import numpy as np
from pathlib import Path
import kagglehub

DATA_DIR = Path("data")
FINAL_DATASET_PATH = DATA_DIR / "final_dataset.csv"
MXMH_PATH = DATA_DIR / "mxmh_survey_results.csv"
GENRE_PROFILES_PATH = DATA_DIR / "genre_happiness_profiles.csv"


def load_mxmh_survey() -> pd.DataFrame:
    """Load MXMH survey data."""
    if MXMH_PATH.exists():
        return pd.read_csv(MXMH_PATH)
    
    # Try downloading from Kaggle
    try:
        path = kagglehub.dataset_download("catherinerasgaitis/mxmh-survey-results")
        found = list(Path(path).glob("*.csv"))
        if found:
            df = pd.read_csv(found[0])
            df.to_csv(MXMH_PATH, index=False)
            return df
    except Exception as e:
        print(f"Error downloading MXMH: {e}")
        return pd.DataFrame()
    
    return pd.DataFrame()


def normalize_genre_names(genre: str) -> str:
    """Normalize genre names for matching."""
    import re
    if pd.isna(genre):
        return ""
    genre = str(genre).lower().strip()
    # Remove special characters and normalize spaces to hyphens
    genre = re.sub(r'[^\w\s]', '', genre)
    genre = re.sub(r'\s+', '-', genre)
    # Common mappings
    mappings = {
        'hip-hop': 'hip-hop', 'hip hop': 'hip-hop',
        'r&b': 'r-n-b', 'r-and-b': 'r-n-b', 'r-n-b': 'r-n-b', 'rnb': 'r-n-b',
        'rock': 'rock',
        'pop': 'pop',
        'country': 'country',
        'jazz': 'jazz',
        'classical': 'classical',
        'electronic': 'electro', 'edm': 'edm',
        'metal': 'metal',
        'punk': 'punk-rock', 'punk-rock': 'punk-rock',
        'folk': 'folk',
        'indie': 'indie-pop', 'indie-pop': 'indie-pop',
        'alternative': 'alternative', 'alt-rock': 'alt-rock',
        'latin': 'latin',
        'k-pop': 'k-pop', 'kpop': 'k-pop',
        'j-pop': 'j-pop', 'jpop': 'j-pop',
        'video-game-music': 'video-game-music', 'video game music': 'video-game-music',
        'lofi': 'lofi',
        'gospel': 'gospel',
    }
    return mappings.get(genre, genre)


def main():
    print("=" * 80)
    print("MXMH Survey - Genre Happiness Correlation Analysis")
    print("=" * 80)
    
    # 1. Load genre profiles
    print("\n1. Loading genre happiness profiles...")
    if GENRE_PROFILES_PATH.exists():
        genre_profiles = pd.read_csv(GENRE_PROFILES_PATH)
        print(f"   ✓ Loaded {len(genre_profiles)} genres")
    else:
        print("   ⚠ Genre profiles not found. Run calculate_comprehensive_happiness.py first.")
        # Calculate from final_dataset
        if FINAL_DATASET_PATH.exists():
            df = pd.read_csv(FINAL_DATASET_PATH)
            if 'track_genre' in df.columns and 'h_track' in df.columns:
                genre_profiles = df.groupby('track_genre').agg({
                    'h_track': ['mean', 'std', 'count'],
                    'a_audio_norm': ['mean', 'std'],
                    'catharsis_score': 'mean',
                }).round(4)
                genre_profiles.columns = ['_'.join(col).strip() for col in genre_profiles.columns]
                genre_profiles = genre_profiles.reset_index()
                genre_profiles.columns = [col.replace('track_genre', 'genre') if 'track_genre' in col else col for col in genre_profiles.columns]
                print(f"   ✓ Calculated profiles for {len(genre_profiles)} genres")
            else:
                print("   ✗ Missing required columns in final_dataset.csv")
                return
        else:
            print("   ✗ final_dataset.csv not found")
            return
    
    # 2. Load MXMH survey
    print("\n2. Loading MXMH survey data...")
    mxmh_df = load_mxmh_survey()
    if mxmh_df.empty:
        print("   ✗ Could not load MXMH survey data")
        return
    
    print(f"   ✓ Loaded {len(mxmh_df)} survey responses")
    print(f"   Columns: {list(mxmh_df.columns)}")
    
    # 3. Extract mental health scores by genre
    print("\n3. Analyzing mental health by favorite genre...")
    
    # Common MXMH columns (adjust based on actual dataset)
    genre_col = None
    for col in mxmh_df.columns:
        if 'genre' in col.lower() or 'fav' in col.lower():
            genre_col = col
            break
    
    if genre_col is None:
        print("   ⚠ Could not find genre column in MXMH data")
        print(f"   Available columns: {list(mxmh_df.columns)}")
        return
    
    # Mental health columns (common names)
    mh_columns = []
    for col in mxmh_df.columns:
        col_lower = col.lower()
        if any(term in col_lower for term in ['depression', 'anxiety', 'ocd', 'insomnia', 'mental']):
            mh_columns.append(col)
    
    if not mh_columns:
        print("   ⚠ Could not find mental health columns")
        print(f"   Available columns: {list(mxmh_df.columns)}")
        return
    
    # Calculate average mental health scores by genre
    mxmh_df['genre_normalized'] = mxmh_df[genre_col].apply(normalize_genre_names)
    
    # Check what column name genre_profiles uses
    genre_col_name = None
    for col in genre_profiles.columns:
        if 'genre' in col.lower():
            genre_col_name = col
            break
    
    if genre_col_name is None:
        print("   ✗ Could not find genre column in genre profiles")
        print(f"   Available columns: {list(genre_profiles.columns)}")
        return
    
    genre_profiles['genre_normalized'] = genre_profiles[genre_col_name].apply(normalize_genre_names)
    
    # Convert mental health columns to numeric before calculating mean
    mxmh_df_numeric = mxmh_df.copy()
    for mh_col in mh_columns:
        if mh_col in mxmh_df_numeric.columns:
            mxmh_df_numeric[mh_col] = pd.to_numeric(mxmh_df_numeric[mh_col], errors='coerce')
    
    mh_by_genre = mxmh_df_numeric.groupby('genre_normalized')[mh_columns].mean()
    mh_by_genre = mh_by_genre.reset_index()
    
    print(f"   ✓ Calculated mental health scores for {len(mh_by_genre)} genres")
    
    # 4. Merge with happiness profiles
    print("\n4. Merging happiness profiles with mental health data...")
    merged = genre_profiles.merge(
        mh_by_genre,
        on='genre_normalized',
        how='inner',
        suffixes=('', '_mxmh')
    )
    
    if merged.empty:
        print("   ⚠ No matching genres found between datasets")
        our_genres = sorted(genre_profiles['genre_normalized'].dropna().unique()[:10])
        mxmh_genres = sorted(mxmh_df['genre_normalized'].dropna().unique()[:10])
        print(f"   Our genres (sample): {our_genres}")
        print(f"   MXMH genres (sample): {mxmh_genres}")
        return
    
    print(f"   ✓ Found {len(merged)} matching genres")
    
    # 5. Calculate correlations
    print("\n5. Calculating correlations...")
    
    correlations = {}
    for mh_col in mh_columns:
        if mh_col in merged.columns:
            corr_h = merged['h_track_mean'].corr(merged[mh_col])
            corr_a = merged['a_audio_norm_mean'].corr(merged[mh_col])
            corr_c = merged['catharsis_score_mean'].corr(merged[mh_col])
            
            correlations[mh_col] = {
                'happiness': corr_h,
                'arousal': corr_a,
                'catharsis': corr_c
            }
    
    # 6. Print results
    print("\n" + "=" * 80)
    print("Correlation Results")
    print("=" * 80)
    
    print("\nCorrelations between genre characteristics and mental health scores:")
    print(f"{'Mental Health Metric':<30} {'Happiness':<12} {'Arousal':<12} {'Catharsis':<12}")
    print("-" * 80)
    for mh_col, corrs in correlations.items():
        print(f"{mh_col:<30} {corrs['happiness']:>10.3f}  {corrs['arousal']:>10.3f}  {corrs['catharsis']:>10.3f}")
    
    # 7. Save merged results
    output_path = DATA_DIR / "genre_mxmh_correlations.csv"
    merged.to_csv(output_path, index=False)
    print(f"\n✓ Saved merged data to {output_path}")
    
    # 8. Summary statistics
    print("\n" + "=" * 80)
    print("Summary Statistics")
    print("=" * 80)
    print(f"\nGenres analyzed: {len(merged)}")
    # Get the genre column name for display
    display_genre_col = genre_col_name if genre_col_name else 'genre'
    
    print(f"\nTop 5 genres by happiness:")
    top_happy = merged.nlargest(5, 'h_track_mean')[[display_genre_col, 'h_track_mean', 'a_audio_norm_mean', 'catharsis_score_mean']]
    for idx, row in top_happy.iterrows():
        genre_name = str(row[display_genre_col]).encode('ascii', 'replace').decode('ascii')
        print(f"  {genre_name}: H={row['h_track_mean']:.3f}, A={row['a_audio_norm_mean']:.3f}, C={row['catharsis_score_mean']:.3f}")
    
    print(f"\nTop 5 genres by catharsis score:")
    top_cathartic = merged.nlargest(5, 'catharsis_score_mean')[[display_genre_col, 'h_track_mean', 'a_audio_norm_mean', 'catharsis_score_mean']]
    for idx, row in top_cathartic.iterrows():
        genre_name = str(row[display_genre_col]).encode('ascii', 'replace').decode('ascii')
        print(f"  {genre_name}: H={row['h_track_mean']:.3f}, A={row['a_audio_norm_mean']:.3f}, C={row['catharsis_score_mean']:.3f}")
    
    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)


if __name__ == "__main__":
    main()

