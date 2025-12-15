"""
Comprehensive Happiness Scoring System
Based on research from:
- Liew et al. (2023): Danceability/energy as cultural affordances for high-arousal negative emotions
- Anglada-Tort et al.: Music features reflect mood regulation mechanisms
- labMT/Hedonometer: Word-level happiness ratings for lyric sentiment
- MXMH Survey: Mental health and music preference data

This script calculates:
1. Audio Valence (V_audio): Spotify valence + mode (major/minor)
2. Audio Arousal (A_audio): Energy + danceability + tempo
3. Lyric Happiness (H_lyrics): labMT-based word happiness scores
4. Track Happiness (H_track): Weighted combination of lyric and audio valence
5. Catharsis Score: High arousal × low happiness (cathartic listening)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re
from typing import Dict, Optional, Any
import kagglehub

# Try to import scipy for sigmoid, fallback to manual implementation
try:
    from scipy.special import expit
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    def expit(x):
        """Sigmoid function: 1 / (1 + exp(-x))"""
        return 1 / (1 + np.exp(-np.clip(x, -250, 250)))  # Clip to avoid overflow

# Try to import langdetect
try:
    from langdetect import detect, LangDetectException
    HAS_LANGDETECT = True
except ImportError:
    HAS_LANGDETECT = False
    print("Warning: langdetect not installed. Install with: pip install langdetect")

# Paths
FINAL_DATASET_PATH = Path("data/final_dataset.csv")
DATA_DIR = Path("data")
MXMH_PATH = Path("data/mxmh_survey_results.csv")

# Language dictionaries mapping
LABMT_LANGUAGES = {
    "en": "labMT-en-v2.csv",
    "de": "labMT-de-v2.csv",
    "ko": "labMT-ko-v2.csv",
    "es": "labMT-es-v2.csv",
    "ru": "labMT-ru-v2.csv",
    "zh": "labMT-zh-v2.csv",
    "ar": "labMT-ar-v2.csv",
    "pt": "labMT-pt-v2.csv",
    "fr": "labMT-fr-v2.csv",
    "uk": "labMT-uk-ru.csv",
}

# Parameters (can be tuned)
ALPHA_V = 0.8  # Weight for valence in audio valence
ALPHA_M = 0.2  # Weight for mode in audio valence
BETA_E = 1/3   # Weight for energy in arousal
BETA_D = 1/3   # Weight for danceability in arousal
BETA_T = 1/3   # Weight for tempo in arousal
LAMBDA = 0.6   # Weight for lyrics in track happiness (0.6 = lyrics dominate)

# Neutral word range for labMT (words with scores 4-6 are often filtered)
NEUTRAL_MIN = 4.0
NEUTRAL_MAX = 6.0
FILTER_NEUTRAL = True  # Whether to filter neutral words


def load_all_labmt_dicts() -> Dict[str, Dict[str, float]]:
    """Load all language dictionaries into a dict of {lang_code: {word: happs}}"""
    labmt_dicts = {}
    
    for lang_code, filename in LABMT_LANGUAGES.items():
        filepath = DATA_DIR / "hedonometer" / filename
        if filepath.exists():
            try:
                df = pd.read_csv(filepath)
                if 'word' in df.columns and 'happs' in df.columns:
                    labmt_dicts[lang_code] = dict(zip(df['word'], df['happs']))
                    print(f"  [OK] Loaded {lang_code}: {len(labmt_dicts[lang_code])} words")
                else:
                    print(f"  [ERROR] {filename}: missing 'word' or 'happs' columns")
            except Exception as e:
                print(f"  [ERROR] Error loading {filename}: {e}")
        else:
            print(f"  [ERROR] {filename} not found")
    
    return labmt_dicts


def detect_language(lyrics: str) -> str:
    """Detect the language of lyrics text."""
    if not lyrics or not isinstance(lyrics, str) or len(lyrics.strip()) < 10:
        return "en"
    
    if HAS_LANGDETECT:
        try:
            detected = detect(lyrics)
            if detected in LABMT_LANGUAGES:
                return detected
            lang_map = {"zh-cn": "zh", "zh-tw": "zh"}
            return lang_map.get(detected, "en")
        except LangDetectException:
            return "en"
    else:
        return "en"


def calculate_lyric_happiness(lyrics: str, labmt_dict: Dict[str, float], 
                             filter_neutral: bool = FILTER_NEUTRAL) -> float:
    """
    Calculate lyric happiness using labMT dictionary.
    
    Formula: H_lyrics = sum(f(w) * h(w)) / sum(f(w))
    where f(w) is word frequency and h(w) is happiness score.
    
    Optionally filters neutral words (4 <= h(w) <= 6).
    """
    if not lyrics or not isinstance(lyrics, str):
        return np.nan
    
    # Tokenize: lowercase, extract words
    words = re.findall(r"\b\w+\b", lyrics.lower())
    
    # For CJK languages, also try character-based matching
    if not words:
        chars = list(lyrics)
        words = [c for c in chars if c.strip()]
    
    # Count word frequencies and collect happiness scores
    word_freqs = {}
    word_scores = {}
    
    for w in words:
        if w in labmt_dict:
            h_score = labmt_dict[w]
            
            # Filter neutral words if requested
            if filter_neutral and (NEUTRAL_MIN <= h_score <= NEUTRAL_MAX):
                continue
            
            word_freqs[w] = word_freqs.get(w, 0) + 1
            word_scores[w] = h_score
    
    if not word_freqs:
        return np.nan
    
    # Calculate frequency-weighted average
    numerator = sum(word_freqs[w] * word_scores[w] for w in word_freqs)
    denominator = sum(word_freqs.values())
    
    if denominator == 0:
        return np.nan
    
    return numerator / denominator


def calculate_audio_valence(df: pd.DataFrame) -> pd.Series:
    """
    Calculate Audio Valence: V_audio = α_v * z_valence + α_m * m
    
    where:
    - z_valence: standardized valence (z-score)
    - m: mode encoding (+1 for major, -1 for minor)
    """
    # Standardize valence (handle NaN)
    valid_valence = df['valence'].dropna()
    if len(valid_valence) == 0:
        return pd.Series(np.nan, index=df.index)
    
    valence_mean = valid_valence.mean()
    valence_std = valid_valence.std()
    if valence_std == 0:
        z_valence = pd.Series(0.0, index=df.index)
    else:
        z_valence = (df['valence'] - valence_mean) / valence_std
    
    # Encode mode: +1 for major (mode=1), -1 for minor (mode=0)
    # Handle NaN: default to 0 (neutral)
    m = df['mode'].apply(lambda x: 1.0 if pd.notna(x) and x == 1 else (-1.0 if pd.notna(x) and x == 0 else 0.0))
    
    # Calculate raw audio valence
    v_audio_raw = ALPHA_V * z_valence + ALPHA_M * m
    
    return v_audio_raw


def calculate_audio_arousal(df: pd.DataFrame) -> pd.Series:
    """
    Calculate Audio Arousal: A_audio = β_e * z_energy + β_d * z_danceability + β_t * z_tempo
    """
    # Standardize each component (handle NaN)
    def standardize_col(col_name):
        valid_col = df[col_name].dropna()
        if len(valid_col) == 0:
            return pd.Series(np.nan, index=df.index)
        col_mean = valid_col.mean()
        col_std = valid_col.std()
        if col_std == 0:
            return pd.Series(0.0, index=df.index)
        return (df[col_name] - col_mean) / col_std
    
    z_energy = standardize_col('energy')
    z_danceability = standardize_col('danceability')
    z_tempo = standardize_col('tempo')
    
    # Calculate raw arousal
    a_audio_raw = BETA_E * z_energy + BETA_D * z_danceability + BETA_T * z_tempo
    
    return a_audio_raw


def normalize_to_0_1(series: pd.Series, use_sigmoid: bool = True) -> pd.Series:
    """
    Normalize a series to [0, 1] range.
    If use_sigmoid=True, uses sigmoid function (expit).
    Otherwise, uses min-max normalization.
    """
    if use_sigmoid:
        # Use sigmoid: 1 / (1 + exp(-x))
        # Handle NaN values and ensure numeric type
        valid_mask = series.notna()
        result = pd.Series(np.nan, index=series.index, dtype=float)
        if valid_mask.any():
            valid_values = series.loc[valid_mask]
            # Ensure numeric and convert to numpy array of float64
            valid_array = pd.to_numeric(valid_values, errors='coerce').astype(np.float64).values
            # Remove any remaining NaN that might have been introduced
            valid_mask_clean = ~np.isnan(valid_array)
            if valid_mask_clean.any():
                # Convert to float64 explicitly for expit
                clean_array = valid_array[valid_mask_clean].astype(np.float64)
                sigmoid_values = expit(clean_array)
                # Map back to original indices
                valid_indices = valid_values.index[valid_mask_clean]
                result.loc[valid_indices] = sigmoid_values.astype(float)
        return result
    else:
        # Min-max normalization
        valid_series = series.dropna()
        if len(valid_series) == 0:
            return pd.Series(np.nan, index=series.index, dtype=float)
        min_val = valid_series.min()
        max_val = valid_series.max()
        if max_val == min_val:
            result = pd.Series(0.5, index=series.index, dtype=float)
            result[series.isna()] = np.nan
            return result
        result = (series - min_val) / (max_val - min_val)
        return result


def load_mxmh_survey() -> Optional[pd.DataFrame]:
    """Load MXMH survey data from Kaggle or local file."""
    # Try local file first
    if MXMH_PATH.exists():
        try:
            df = pd.read_csv(MXMH_PATH)
            print(f"  [OK] Loaded MXMH survey from {MXMH_PATH}")
            return df
        except Exception as e:
                print(f"  [ERROR] Error loading local MXMH: {e}")
    
    # Try downloading from Kaggle
    try:
        print("  Attempting to download MXMH survey from Kaggle...")
        path = kagglehub.dataset_download("catherinerasgaitis/mxmh-survey-results")
        found = list(Path(path).glob("*.csv"))
        if found:
            df = pd.read_csv(found[0])
            # Save locally for future use
            df.to_csv(MXMH_PATH, index=False)
            print(f"  [OK] Downloaded and saved MXMH survey to {MXMH_PATH}")
            return df
    except Exception as e:
        print(f"  [ERROR] Could not download MXMH survey: {e}")
        print("  Continuing without MXMH data...")
    
    return None


def calculate_genre_happiness_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate mean happiness and arousal by genre."""
    if 'track_genre' not in df.columns:
        return pd.DataFrame()
    
    genre_profiles = df.groupby('track_genre').agg({
        'h_track': ['mean', 'std', 'count'],
        'a_audio_norm': ['mean', 'std'],
        'catharsis_score': 'mean',
    }).round(4)
    
    genre_profiles.columns = ['_'.join(col).strip() for col in genre_profiles.columns]
    genre_profiles = genre_profiles.reset_index()
    
    return genre_profiles


def main():
    print("=" * 80)
    print("Comprehensive Happiness Scoring System")
    print("=" * 80)
    
    # 1. Load dataset
    if not FINAL_DATASET_PATH.exists():
        print(f"Error: {FINAL_DATASET_PATH} not found.")
        return
    
    print("\n1. Loading final dataset...")
    df = pd.read_csv(FINAL_DATASET_PATH)
    print(f"   Loaded {len(df)} tracks")
    
    # 2. Load LabMT dictionaries
    print("\n2. Loading LabMT dictionaries...")
    labmt_dicts = load_all_labmt_dicts()
    if not labmt_dicts:
        print("   Error: No LabMT dictionaries found!")
        return
    default_dict = labmt_dicts.get("en", {})
    
    # 3. Calculate Audio Valence and Arousal
    print("\n3. Calculating audio features...")
    
    # Check required columns
    required_audio_cols = ['valence', 'mode', 'energy', 'danceability', 'tempo']
    missing_cols = [col for col in required_audio_cols if col not in df.columns]
    if missing_cols:
        print(f"   Error: Missing required columns: {missing_cols}")
        return
    
    # Calculate raw scores
    df['v_audio_raw'] = calculate_audio_valence(df)
    df['a_audio_raw'] = calculate_audio_arousal(df)
    
    # Normalize to [0, 1]
    df['v_audio_norm'] = normalize_to_0_1(df['v_audio_raw'], use_sigmoid=True)
    df['a_audio_norm'] = normalize_to_0_1(df['a_audio_raw'], use_sigmoid=True)
    
    print(f"   [OK] Audio Valence: mean={df['v_audio_norm'].mean():.3f}, std={df['v_audio_norm'].std():.3f}")
    print(f"   [OK] Audio Arousal: mean={df['a_audio_norm'].mean():.3f}, std={df['a_audio_norm'].std():.3f}")
    
    # 4. Load lyrics from dataset_lyrics.csv if available
    lyrics_df_path = DATA_DIR / "dataset_lyrics.csv"
    if lyrics_df_path.exists():
        print("\n4. Loading lyrics from dataset_lyrics.csv...")
        try:
            lyrics_df = pd.read_csv(lyrics_df_path)
            # Merge lyrics into main dataset
            if 'lyrics' in lyrics_df.columns:
                # Match on track_name and artists
                base_cols = ['track_name', 'artists', 'lyrics']
                merge_cols = base_cols.copy()
                if 'happiness_from_lyrics' in lyrics_df.columns:
                    merge_cols.append('happiness_from_lyrics')
                lyrics_df_clean = lyrics_df[merge_cols].dropna(subset=['track_name', 'artists'])
                
                # Merge lyrics, and also preserve happiness_from_lyrics if it exists
                df = df.merge(
                    lyrics_df_clean,
                    on=['track_name', 'artists'],
                    how='left',
                    suffixes=('', '_from_file')
                )
                # Use lyrics from lyrics file if main dataset doesn't have them
                if 'lyrics_from_file' in df.columns:
                    df['lyrics'] = df['lyrics'].fillna(df['lyrics_from_file'])
                    df = df.drop(columns=['lyrics_from_file'])
                # Also fill happiness_from_lyrics if needed
                if 'happiness_from_lyrics_from_file' in df.columns:
                    if 'happiness_from_lyrics' not in df.columns:
                        df['happiness_from_lyrics'] = df['happiness_from_lyrics_from_file']
                    else:
                        # Only fill NaN values from file
                        df['happiness_from_lyrics'] = df['happiness_from_lyrics'].fillna(
                            df['happiness_from_lyrics_from_file']
                        )
                    df = df.drop(columns=['happiness_from_lyrics_from_file'])
                    print(f"   [OK] Preserved BERT scores: {df['happiness_from_lyrics'].notna().sum()} tracks")
                print(f"   [OK] Merged lyrics for {df['lyrics'].notna().sum()} tracks")
        except Exception as e:
            print(f"   [WARN] Error loading lyrics file: {e}")
    
    # 5. Calculate Lyric Happiness
    print("\n4. Calculating lyric happiness...")
    
    # Initialize columns
    if 'lyrics' not in df.columns:
        df['lyrics'] = pd.NA
    if 'lyrics_language' not in df.columns:
        df['lyrics_language'] = pd.NA
    if 'h_lyrics_raw' not in df.columns:
        df['h_lyrics_raw'] = pd.NA
    
    # Process lyrics
    lyrics_processed = 0
    for idx, row in df.iterrows():
        lyrics = row.get('lyrics')
        
        if pd.isna(lyrics) or lyrics == "":
            continue
        
        # Detect language if not already set
        detected_lang = row.get('lyrics_language')
        if pd.isna(detected_lang):
            detected_lang = detect_language(str(lyrics))
            df.at[idx, 'lyrics_language'] = detected_lang
        
        # Get appropriate dictionary
        labmt_dict = labmt_dicts.get(detected_lang, default_dict)
        
        # Calculate lyric happiness
        h_lyrics = calculate_lyric_happiness(str(lyrics), labmt_dict, filter_neutral=FILTER_NEUTRAL)
        if pd.notna(h_lyrics):
            df.at[idx, 'h_lyrics_raw'] = h_lyrics
            lyrics_processed += 1
    
    print(f"   [OK] Processed lyrics for {lyrics_processed} tracks")
    
    # Normalize lyric happiness to [0, 1]
    # First convert from [1, 9] scale to [0, 1], then standardize
    df['h_lyrics_01'] = (df['h_lyrics_raw'] - 1) / 8  # Map [1,9] -> [0,1]
    
    # Standardize and normalize
    h_lyrics_mean = df['h_lyrics_01'].mean()
    h_lyrics_std = df['h_lyrics_01'].std()
    if h_lyrics_std > 0:
        z_lyrics = (df['h_lyrics_01'] - h_lyrics_mean) / h_lyrics_std
        df['h_lyrics_norm'] = normalize_to_0_1(z_lyrics, use_sigmoid=True)
    else:
        df['h_lyrics_norm'] = df['h_lyrics_01']
    
    print(f"   [OK] Lyric Happiness: mean={df['h_lyrics_norm'].mean():.3f} (from {df['h_lyrics_raw'].notna().sum()} tracks with lyrics)")
    
    # 6. Calculate Track Happiness
    print("\n6. Calculating track-level happiness...")
    
    # H_track = λ * H_lyrics + (1-λ) * V_audio
    # If lyrics missing, use only audio valence
    df['h_track'] = (
        LAMBDA * df['h_lyrics_norm'].fillna(0) + 
        (1 - LAMBDA) * df['v_audio_norm']
    )
    
    # If we have lyrics, use the weighted combination
    has_lyrics = df['h_lyrics_norm'].notna()
    df.loc[has_lyrics, 'h_track'] = (
        LAMBDA * df.loc[has_lyrics, 'h_lyrics_norm'] + 
        (1 - LAMBDA) * df.loc[has_lyrics, 'v_audio_norm']
    )
    
    print(f"   [OK] Track Happiness: mean={df['h_track'].mean():.3f}, std={df['h_track'].std():.3f}")
    
    # 7. Calculate Catharsis Score
    print("\n7. Calculating catharsis score...")
    
    # C_catharsis = A_audio * (1 - H_track)
    # High when energetic but lyrically/valence-wise negative
    df['catharsis_score'] = df['a_audio_norm'] * (1 - df['h_track'])
    
    print(f"   [OK] Catharsis Score: mean={df['catharsis_score'].mean():.3f}, std={df['catharsis_score'].std():.3f}")
    
    # 8. Load and integrate MXMH survey data
    print("\n8. Loading MXMH survey data...")
    mxmh_df = load_mxmh_survey()
    
    if mxmh_df is not None:
        print(f"   [OK] MXMH survey loaded: {len(mxmh_df)} respondents")
        
        # Calculate genre-level profiles from our dataset
        genre_profiles = calculate_genre_happiness_profiles(df)
        if not genre_profiles.empty:
            genre_profiles_path = DATA_DIR / "genre_happiness_profiles.csv"
            genre_profiles.to_csv(genre_profiles_path, index=False)
            print(f"   [OK] Saved genre profiles to {genre_profiles_path}")
            
            # Try to match genres from MXMH to our track genres
            if 'Fav genre' in mxmh_df.columns:
                mxmh_genres = mxmh_df['Fav genre'].dropna().unique()
                our_genres = df['track_genre'].dropna().unique()
                common_genres = set(mxmh_genres) & set(our_genres)
                print(f"   [OK] Found {len(common_genres)} common genres between MXMH and our dataset")
    else:
        print("   [WARN] Continuing without MXMH data")
    
    # 9. Save updated dataset
    print("\n9. Saving updated dataset...")
    df.to_csv(FINAL_DATASET_PATH, index=False)
    print(f"   [OK] Saved to {FINAL_DATASET_PATH}")
    
    # 10. Print summary statistics
    print("\n" + "=" * 80)
    print("Summary Statistics")
    print("=" * 80)
    print(f"\nTotal tracks: {len(df)}")
    print(f"Tracks with lyrics: {df['h_lyrics_raw'].notna().sum()}")
    print(f"\nHappiness Scores (0-1 scale):")
    print(f"  Audio Valence:     mean={df['v_audio_norm'].mean():.3f}, std={df['v_audio_norm'].std():.3f}")
    print(f"  Audio Arousal:    mean={df['a_audio_norm'].mean():.3f}, std={df['a_audio_norm'].std():.3f}")
    print(f"  Lyric Happiness:  mean={df['h_lyrics_norm'].mean():.3f}, std={df['h_lyrics_norm'].std():.3f} (n={df['h_lyrics_norm'].notna().sum()})")
    print(f"  Track Happiness:  mean={df['h_track'].mean():.3f}, std={df['h_track'].std():.3f}")
    print(f"  Catharsis Score:  mean={df['catharsis_score'].mean():.3f}, std={df['catharsis_score'].std():.3f}")
    
    # Top and bottom tracks by happiness
    print(f"\nTop 5 happiest tracks:")
    top_happy = df.nlargest(5, 'h_track')[['track_name', 'artists', 'h_track', 'a_audio_norm', 'catharsis_score']]
    for idx, row in top_happy.iterrows():
        track_name = str(row['track_name']).encode('ascii', 'replace').decode('ascii')
        artists = str(row['artists']).encode('ascii', 'replace').decode('ascii')
        print(f"  {track_name} - {artists}: H={row['h_track']:.3f}, A={row['a_audio_norm']:.3f}, C={row['catharsis_score']:.3f}")
    
    print(f"\nTop 5 most cathartic tracks (high arousal, low happiness):")
    top_cathartic = df.nlargest(5, 'catharsis_score')[['track_name', 'artists', 'h_track', 'a_audio_norm', 'catharsis_score']]
    for idx, row in top_cathartic.iterrows():
        track_name = str(row['track_name']).encode('ascii', 'replace').decode('ascii')
        artists = str(row['artists']).encode('ascii', 'replace').decode('ascii')
        print(f"  {track_name} - {artists}: H={row['h_track']:.3f}, A={row['a_audio_norm']:.3f}, C={row['catharsis_score']:.3f}")
    
    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)


if __name__ == "__main__":
    main()

