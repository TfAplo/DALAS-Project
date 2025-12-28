"""
Compute h_track, catharsis_score, lyrics_language, and embeddings for new songs in final.csv.

This script:
1. Loads final.csv
2. Computes h_track, catharsis_score, and lyrics_language for songs missing these values
3. Computes embeddings for all songs
4. Runs PCA on embeddings
5. Adds PCA components to the dataset
6. Saves the updated dataset
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path
from typing import Dict, Optional
import sys

# Import functions from existing modules
try:
    from calculate_comprehensive_happiness import (
        load_all_labmt_dicts,
        detect_language,
        calculate_lyric_happiness,
        calculate_audio_valence,
        calculate_audio_arousal,
        normalize_to_0_1,
    )
    LAMBDA = 0.6  # Weight for lyrics in track happiness (from calculate_comprehensive_happiness.py)
    FILTER_NEUTRAL = True
    NEUTRAL_MIN = 4.0
    NEUTRAL_MAX = 6.0
except ImportError:
    print("Warning: Could not import from calculate_comprehensive_happiness.py")
    print("Will use inline implementations")
    LAMBDA = 0.6
    FILTER_NEUTRAL = True
    NEUTRAL_MIN = 4.0
    NEUTRAL_MAX = 6.0

try:
    from langdetect import detect, LangDetectException
    HAS_LANGDETECT = True
except ImportError:
    HAS_LANGDETECT = False
    print("Warning: langdetect not installed. Install with: pip install langdetect")

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    print("Warning: sentence-transformers not installed. Install with: pip install sentence-transformers")

try:
    from sklearn.decomposition import PCA
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("Warning: scikit-learn not installed. Install with: pip install scikit-learn")

try:
    from scipy.special import expit
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    def expit(x):
        """Sigmoid function: 1 / (1 + exp(-x))"""
        return 1 / (1 + np.exp(-np.clip(x, -250, 250)))

# Paths
DATA_DIR = Path("data")
FINAL_CSV = DATA_DIR / "final.csv"
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

# Audio feature weights (from calculate_comprehensive_happiness.py)
ALPHA_V = 0.8  # Weight for valence in audio valence
ALPHA_M = 0.2  # Weight for mode in audio valence
BETA_E = 1/3   # Weight for energy in arousal
BETA_D = 1/3   # Weight for danceability in arousal
BETA_T = 1/3   # Weight for tempo in arousal


def load_all_labmt_dicts() -> Dict[str, Dict[str, float]]:
    """Load all language dictionaries into a dict of {lang_code: {word: happs}}"""
    labmt_dicts = {}
    hed = DATA_DIR / "hedonometer"
    
    for lang_code, filename in LABMT_LANGUAGES.items():
        filepath = hed / filename
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
            print(f"  [WARN] {filename} not found")
    
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


def compute_happiness_features(df: pd.DataFrame, labmt_dicts: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """Compute h_track, catharsis_score, and lyrics_language for songs missing these values."""
    df = df.copy()
    
    default_dict = labmt_dicts.get("en", {})
    
    # Initialize columns if they don't exist
    if 'lyrics_language' not in df.columns:
        df['lyrics_language'] = pd.NA
    if 'h_lyrics_raw' not in df.columns:
        df['h_lyrics_raw'] = pd.NA
    if 'h_track' not in df.columns:
        df['h_track'] = pd.NA
    if 'catharsis_score' not in df.columns:
        df['catharsis_score'] = pd.NA
    
    # Identify songs that need processing (missing lyrics_language or h_track)
    needs_language = df['lyrics_language'].isna() & df['lyrics'].notna()
    needs_happiness = df['h_track'].isna()
    
    print(f"\nSongs needing language detection: {needs_language.sum()}")
    print(f"Songs needing happiness computation: {needs_happiness.sum()}")
    
    # Process lyrics for language detection and happiness
    processed = 0
    for idx, row in df.iterrows():
        lyrics = row.get('lyrics')
        
        if pd.isna(lyrics) or lyrics == "":
            continue
        
        # Detect language if not already set
        if pd.isna(row.get('lyrics_language')):
            detected_lang = detect_language(str(lyrics))
            df.at[idx, 'lyrics_language'] = detected_lang
        else:
            detected_lang = row.get('lyrics_language')
        
        # Get appropriate dictionary
        labmt_dict = labmt_dicts.get(str(detected_lang), default_dict)
        
        # Calculate lyric happiness if not already computed
        if pd.isna(row.get('h_lyrics_raw')):
            h_lyrics = calculate_lyric_happiness(str(lyrics), labmt_dict, filter_neutral=FILTER_NEUTRAL)
            if pd.notna(h_lyrics):
                df.at[idx, 'h_lyrics_raw'] = h_lyrics
                processed += 1
    
    print(f"Processed lyrics for {processed} tracks")
    
    # Normalize lyric happiness to [0, 1]
    df['h_lyrics_01'] = (df['h_lyrics_raw'] - 1) / 8  # Map [1,9] -> [0,1]
    
    # Standardize and normalize
    h_lyrics_mean = df['h_lyrics_01'].mean()
    h_lyrics_std = df['h_lyrics_01'].std()
    if h_lyrics_std > 0:
        z_lyrics = (df['h_lyrics_01'] - h_lyrics_mean) / h_lyrics_std
        df['h_lyrics_norm'] = normalize_to_0_1(z_lyrics, use_sigmoid=True)
    else:
        df['h_lyrics_norm'] = df['h_lyrics_01']
    
    # Calculate Audio Valence and Arousal (for all songs, needed for h_track and catharsis)
    print("\nCalculating audio features...")
    df['v_audio_raw'] = calculate_audio_valence(df)
    df['a_audio_raw'] = calculate_audio_arousal(df)
    df['v_audio_norm'] = normalize_to_0_1(df['v_audio_raw'], use_sigmoid=True)
    df['a_audio_norm'] = normalize_to_0_1(df['a_audio_raw'], use_sigmoid=True)
    
    # Calculate Track Happiness
    print("\nCalculating track-level happiness...")
    # H_track = λ * H_lyrics + (1-λ) * V_audio
    # If lyrics missing, use only audio valence
    has_lyrics = df['h_lyrics_norm'].notna()
    has_audio = df['v_audio_norm'].notna()
    
    # Initialize h_track
    df['h_track'] = np.nan
    
    # If both lyrics and audio exist: weighted combination
    df.loc[has_lyrics & has_audio, 'h_track'] = (
        LAMBDA * df.loc[has_lyrics & has_audio, 'h_lyrics_norm'] + 
        (1 - LAMBDA) * df.loc[has_lyrics & has_audio, 'v_audio_norm']
    )
    
    # If only lyrics exist: use lyrics
    df.loc[has_lyrics & ~has_audio, 'h_track'] = df.loc[has_lyrics & ~has_audio, 'h_lyrics_norm']
    
    # If only audio exists: use audio valence
    df.loc[~has_lyrics & has_audio, 'h_track'] = df.loc[~has_lyrics & has_audio, 'v_audio_norm']
    
    print(f"Track Happiness: mean={df['h_track'].mean():.3f}, std={df['h_track'].std():.3f}")
    
    # Calculate Catharsis Score
    print("\nCalculating catharsis score...")
    # C_catharsis = A_audio * (1 - H_track)
    df['catharsis_score'] = df['a_audio_norm'] * (1 - df['h_track'])
    
    print(f"Catharsis Score: mean={df['catharsis_score'].mean():.3f}, std={df['catharsis_score'].std():.3f}")
    
    return df


def compute_embeddings(df: pd.DataFrame) -> pd.DataFrame:
    """Compute embeddings for all songs using SentenceTransformers."""
    if not HAS_SENTENCE_TRANSFORMERS:
        print("ERROR: sentence-transformers not available. Cannot compute embeddings.")
        return df
    
    print("\nComputing embeddings...")
    model_name = 'all-MiniLM-L6-v2'
    print(f"Loading model: {model_name}...")
    model = SentenceTransformer(model_name)
    
    # Clean lyrics for embedding
    def clean_lyrics_text(text):
        if not isinstance(text, str):
            return ""
        t = text.replace("\r", " ").replace("\n", " ")
        t = t.replace("\\r", " ").replace("\\n", " ")
        t = re.sub(r"\[[^\]]+\]", " ", t)
        t = re.sub(r"\s+", " ", t).strip()
        return t
    
    # Prepare lyrics
    df['lyrics_clean'] = df['lyrics'].fillna("").apply(clean_lyrics_text)
    
    # Compute embeddings
    print(f"Embedding {len(df)} songs...")
    lyrics_list = df['lyrics_clean'].tolist()
    
    embeddings = model.encode(
        lyrics_list,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    
    # Store embeddings
    df['embedding'] = list(embeddings)
    df['embedding_model'] = model_name
    print(f"Embeddings computed: shape {embeddings.shape}")
    
    return df


def run_pca_on_embeddings(df: pd.DataFrame, n_components: Optional[int] = None) -> pd.DataFrame:
    """Run PCA on embeddings and add components to dataframe."""
    if not HAS_SKLEARN:
        print("ERROR: scikit-learn not available. Cannot run PCA.")
        return df
    
    if 'embedding' not in df.columns:
        print("ERROR: No embeddings found. Run compute_embeddings first.")
        return df
    
    print("\nRunning PCA on embeddings...")
    
    # Convert embeddings to numpy array
    embeddings_list = df['embedding'].tolist()
    embeddings_matrix = np.vstack(embeddings_list)
    
    print(f"Embeddings matrix shape: {embeddings_matrix.shape}")
    
    # Determine number of components
    if n_components is None:
        # Use enough components to explain 95% of variance
        pca_temp = PCA()
        pca_temp.fit(embeddings_matrix)
        cumsum_variance = np.cumsum(pca_temp.explained_variance_ratio_)
        n_components = np.argmax(cumsum_variance >= 0.95) + 1
        print(f"Selected {n_components} components to explain 95% of variance")
    
    # Run PCA
    pca = PCA(n_components=n_components)
    pca_results = pca.fit_transform(embeddings_matrix)
    
    # Calculate variance explained
    variance_ratio = pca.explained_variance_ratio_
    total_variance = sum(variance_ratio) * 100
    print(f"Explained variance ratio: {variance_ratio}")
    print(f"Total variance explained: {total_variance:.2f}%")
    print(f"Number of components kept: {n_components} (out of {embeddings_matrix.shape[1]} original dimensions)")
    
    # Add PCA components to dataframe (use concat to avoid fragmentation)
    pca_df = pd.DataFrame(
        pca_results,
        columns=[f'embedding_pc_{i+1}' for i in range(n_components)],
        index=df.index
    )
    df = pd.concat([df, pca_df], axis=1)
    
    return df, n_components, total_variance


def main():
    print("=" * 80)
    print("Computing Features for New Songs in final.csv")
    print("=" * 80)
    
    # 1. Load dataset
    if not FINAL_CSV.exists():
        print(f"Error: {FINAL_CSV} not found.")
        return 1
    
    print(f"\n1. Loading {FINAL_CSV}...")
    df = pd.read_csv(FINAL_CSV)
    print(f"   Loaded {len(df)} tracks")
    
    # 2. Load LabMT dictionaries
    print("\n2. Loading LabMT dictionaries...")
    labmt_dicts = load_all_labmt_dicts()
    if not labmt_dicts:
        print("   Error: No LabMT dictionaries found!")
        return 1
    
    # 3. Compute happiness features (h_track, catharsis_score, lyrics_language)
    print("\n3. Computing happiness features...")
    df = compute_happiness_features(df, labmt_dicts)
    
    # 4. Compute embeddings
    print("\n4. Computing embeddings...")
    if HAS_SENTENCE_TRANSFORMERS:
        df = compute_embeddings(df)
    else:
        print("   Skipping embeddings (sentence-transformers not available)")
    
    # 5. Run PCA on embeddings
    print("\n5. Running PCA on embeddings...")
    if 'embedding' in df.columns and HAS_SKLEARN:
        df, n_components, variance_explained = run_pca_on_embeddings(df)
        print(f"\n   PCA Summary:")
        print(f"   - Original embedding dimensions: {df['embedding'].iloc[0].shape[0]}")
        print(f"   - Components kept: {n_components}")
        print(f"   - Variance explained: {variance_explained:.2f}%")
        print(f"   - Reason: Selected {n_components} components to explain 95% of variance")
    else:
        print("   Skipping PCA (embeddings or sklearn not available)")
    
    # 6. Remove intermediate normalized columns (as requested - pipeline will normalize)
    print("\n6. Cleaning up intermediate columns...")
    intermediate_cols = ['v_audio_raw', 'a_audio_raw', 'v_audio_norm', 'a_audio_norm', 
                        'h_lyrics_raw', 'h_lyrics_01', 'h_lyrics_norm', 'lyrics_clean']
    cols_to_remove = [c for c in intermediate_cols if c in df.columns]
    if cols_to_remove:
        df = df.drop(columns=cols_to_remove)
        print(f"   Removed intermediate columns: {cols_to_remove}")
    
    # 7. Save updated dataset
    print("\n7. Saving updated dataset...")
    df.to_csv(FINAL_CSV, index=False)
    print(f"   [OK] Saved to {FINAL_CSV}")
    
    # 8. Print summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"\nTotal tracks: {len(df)}")
    print(f"Tracks with lyrics_language: {df['lyrics_language'].notna().sum()}")
    print(f"Tracks with h_track: {df['h_track'].notna().sum()}")
    print(f"Tracks with catharsis_score: {df['catharsis_score'].notna().sum()}")
    if 'embedding' in df.columns:
        print(f"Tracks with embeddings: {df['embedding'].notna().sum()}")
    if 'embedding_pc_1' in df.columns:
        pca_cols = [c for c in df.columns if c.startswith('embedding_pc_')]
        print(f"PCA components added: {len(pca_cols)}")
    
    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

