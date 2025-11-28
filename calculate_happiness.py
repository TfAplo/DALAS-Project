import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
import time
from pathlib import Path
import numpy as np

# Try to import langdetect, fallback to simple heuristic if not available
try:
    from langdetect import detect, LangDetectException
    HAS_LANGDETECT = True
except ImportError:
    HAS_LANGDETECT = False
    print("Warning: langdetect not installed. Install with: pip install langdetect")
    print("Will use dictionary matching heuristic instead.")

# Paths
FINAL_DATASET_PATH = Path("data/final_dataset.csv")
DATA_DIR = Path("data")

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
    "uk": "labMT-uk-ru.csv",  # Ukrainian has different filename
}

# Headers for scraping
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

def clean_text_for_url(s):
    """Clean string for AZLyrics URL format (lowercase, no punctuation/spaces)"""
    if not isinstance(s, str):
        return ""
    s = s.lower()
    # Remove "the" from start of artist if present? AZLyrics usually does.
    if s.startswith("the "):
        s = s[4:]
    s = re.sub(r"[^a-z0-9]", "", s)
    return s

def fetch_lyrics_az(artist, title):
    """Try to fetch lyrics from AZLyrics"""
    a_clean = clean_text_for_url(artist)
    t_clean = clean_text_for_url(title)
    
    if not a_clean or not t_clean:
        return None
        
    url = f"https://www.azlyrics.com/lyrics/{a_clean}/{t_clean}.html"
    
    try:
        resp = requests.get(url, headers=HEADERS, timeout=5)
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, "html.parser")
            # AZLyrics: lyrics are in a div with no class, but there is a comment <!-- Usage of azlyrics.com content by any third-party prohibited ... -->
            # Or finding the div after the ringtone div?
            
            # Div structure:
            # <div class="ringtone">...</div>
            # <b>"Title"</b><br>
            # <br>
            # <div>
            # <!-- Usage of azlyrics.com content by any third-party prohibited ... -->
            # [Lyrics]
            # </div>
            
            # Find div with the comment
            for div in soup.find_all("div"):
                if not div.attrs and "Usage of azlyrics.com content" in str(div):
                    return div.get_text(strip=True)
                    
            # Fallback: look for div with significant text content near the center
            # This is risky.
            
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        
    return None

def detect_language(lyrics: str) -> str:
    """
    Detect the language of lyrics text.
    Returns language code (e.g., 'en', 'es', 'fr') or 'en' as default.
    """
    if not lyrics or not isinstance(lyrics, str) or len(lyrics.strip()) < 10:
        return "en"  # Default to English
    
    if HAS_LANGDETECT:
        try:
            # Use langdetect
            detected = detect(lyrics)
            # Map to our supported languages
            if detected in LABMT_LANGUAGES:
                return detected
            # Some mappings for common variations
            lang_map = {
                "zh-cn": "zh",
                "zh-tw": "zh",
            }
            return lang_map.get(detected, "en")
        except LangDetectException:
            return "en"
    else:
        # Simple heuristic: try each dictionary and see which matches most words
        # This is a fallback if langdetect is not available
        return "en"  # Default to English for now


def calculate_lyric_happiness(lyrics, labmt_dict):
    """
    Calculate average happiness score of lyrics using LabMT dictionary.
    labmt_dict: {word: happs}
    """
    if not lyrics or not isinstance(lyrics, str):
        return np.nan
    
    # For non-Latin scripts, we need different word tokenization
    # For now, use simple regex for Latin scripts
    words = re.findall(r"\b\w+\b", lyrics.lower())
    
    # For Chinese/Japanese/Korean, we might need character-based tokenization
    # For Arabic, we might need different handling
    # For now, try word-based matching first
    
    scores = []
    
    for w in words:
        if w in labmt_dict:
            scores.append(labmt_dict[w])
    
    # If no word matches found, try character-based for CJK languages
    if not scores and len(lyrics) > 0:
        # Try matching individual characters for Chinese/Japanese
        chars = list(lyrics)
        for char in chars:
            if char in labmt_dict:
                scores.append(labmt_dict[char])
    
    if not scores:
        return np.nan
        
    return np.mean(scores)

def load_all_labmt_dicts():
    """Load all language dictionaries into a dict of {lang_code: {word: happs}}"""
    labmt_dicts = {}
    
    for lang_code, filename in LABMT_LANGUAGES.items():
        filepath = DATA_DIR / filename
        if filepath.exists():
            try:
                df = pd.read_csv(filepath)
                if 'word' in df.columns and 'happs' in df.columns:
                    labmt_dicts[lang_code] = dict(zip(df['word'], df['happs']))
                    print(f"  ✓ Loaded {lang_code}: {len(labmt_dicts[lang_code])} words")
                else:
                    print(f"  ✗ {filename}: missing 'word' or 'happs' columns")
            except Exception as e:
                print(f"  ✗ Error loading {filename}: {e}")
        else:
            print(f"  ✗ {filename} not found")
    
    return labmt_dicts


def main():
    if not FINAL_DATASET_PATH.exists():
        print("final_dataset.csv not found")
        return
        
    print("Loading datasets...")
    df = pd.read_csv(FINAL_DATASET_PATH)
    
    print("Loading LabMT dictionaries...")
    labmt_dicts = load_all_labmt_dicts()
    
    if not labmt_dicts:
        print("No LabMT dictionaries found! Please run get_all_labmt.py first.")
        return
    
    # Default to English if available
    default_dict = labmt_dicts.get("en", {})

    # Add columns if missing
    if 'lyrics' not in df.columns:
        df['lyrics'] = pd.NA
    if 'happiness_score' not in df.columns:
        df['happiness_score'] = pd.NA
    if 'lyric_happiness' not in df.columns:
        df['lyric_happiness'] = pd.NA
    if 'lyrics_language' not in df.columns:
        df['lyrics_language'] = pd.NA
        
    print(f"Processing {len(df)} songs...")
    
    processed_count = 0
    
    for idx, row in df.iterrows():
        # Skip if we already have a score (unless force update?)
        # For now, try to fill missing
        
        title = row.get('track_name', row.get('title', ''))
        artist = row.get('artists', row.get('artist', ''))
        
        # Normalize artist (take first if multiple)
        if isinstance(artist, str) and ';' in artist:
            artist = artist.split(';')[0]
        if isinstance(artist, str) and ',' in artist:
            artist = artist.split(',')[0]
            
        lyrics = row.get('lyrics')
        
        # 1. Fetch lyrics if missing
        if pd.isna(lyrics) or lyrics == "":
            print(f"Fetching lyrics for: {title} - {artist}")
            lyrics = fetch_lyrics_az(artist, title)
            if lyrics:
                df.at[idx, 'lyrics'] = lyrics
                time.sleep(2) # Respect rate limits
            else:
                time.sleep(0.5) # Short delay even on miss
        
        # 2. Calculate Lyric Happiness (with language detection)
        lyric_score = df.at[idx, 'lyric_happiness']
        detected_lang = df.at[idx, 'lyrics_language']
        
        if pd.isna(lyric_score) and lyrics:
            # Detect language and use appropriate dictionary
            if pd.isna(detected_lang):
                detected_lang = detect_language(lyrics)
                df.at[idx, 'lyrics_language'] = detected_lang
            
            labmt_dict = labmt_dicts.get(detected_lang, default_dict)
            
            if detected_lang != "en" and detected_lang in labmt_dicts:
                print(f"  Using {detected_lang} dictionary for: {title} - {artist}")
            
            lyric_score = calculate_lyric_happiness(lyrics, labmt_dict)
            df.at[idx, 'lyric_happiness'] = lyric_score
            
        # 3. Calculate Composite Happiness Score
        # Components:
        # - Lyric Happiness (1-9, center ~5) -> Normalize to 0-1? (x-1)/8
        # - Valence (0-1)
        # - Popularity (0-100) -> Normalize 0-1
        # - Chart Position (if available) -> lower is better. 
        
        # Using only what we have:
        valence = row.get('valence', np.nan)
        popularity = row.get('popularity', 0)
        
        # Normalize
        # LabMT: 1 to 9. (val - 1) / 8 -> 0 to 1.
        norm_lyric = (lyric_score - 1) / 8 if pd.notna(lyric_score) else 0.5 # Default to neutral if no lyrics
        
        # Valence: 0 to 1.
        norm_valence = valence if pd.notna(valence) else 0.5
        
        # Popularity: 0 to 100 -> 0 to 1.
        norm_pop = popularity / 100 if pd.notna(popularity) else 0.0
        
        # Weights (Arbitrary but reasonable)
        # Lyrics: 40%, Valence: 40%, Popularity: 20%
        # Or user said "account for position on chart... and how it charted around the world".
        # If we have 'position', we can use it.
        # 'position' in dataset is mostly for specific charts.
        # 'chart' column tells us which chart.
        
        # Let's keep it simple: Acoustic + Lyrics + Pop
        
        composite = (0.4 * norm_lyric) + (0.4 * norm_valence) + (0.2 * norm_pop)
        
        # Scale to 0-100 for final score? Or 0-10? Let's do 0-100.
        final_score = composite * 100
        
        df.at[idx, 'happiness_score'] = round(final_score, 2)
        processed_count += 1
        
        if processed_count % 10 == 0:
            print(f"Processed {processed_count}/{len(df)}")
            # Save intermediate?
            # df.to_csv(FINAL_DATASET_PATH, index=False)

    df.to_csv(FINAL_DATASET_PATH, index=False)
    print(f"Updated {FINAL_DATASET_PATH} with happiness scores.")

if __name__ == "__main__":
    main()


