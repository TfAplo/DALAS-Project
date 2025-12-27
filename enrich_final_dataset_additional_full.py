"""
End-to-end enrichment for `data/final_dataset_additional.csv`.

This script produces a NEW dataset (does not overwrite the input) by:
  1) Filling missing Spotify popularity + track metadata via Spotify Search API
     (client-credentials), with on-disk caching + 429 backoff to avoid rate-limits.
  2) Filling Spotify audio features ONLY from local `spotify_tracks.csv` (Kaggle-derived)
     because Spotify's Web API audio-features endpoints return 403 for many apps
     (deprecated/restricted).
  3) Computing happiness scores using the same approach as `calculate_comprehensive_happiness.py`:
     - Audio valence/arousal (z-score then sigmoid normalization)
     - labMT lyric happiness (language-aware where possible)
     - Track happiness (lyrics+audio when both exist; otherwise falls back)
     - Catharsis score
  4) Attaching ONE chart appearance (if present) from the scraped chart catalog
     using normalized (title, artist) matching, and filling region/country/continent.

Then run:
  - `python embed/embed_lyrics.py <output_csv> lyrics --output <out>.pkl --keep_all`
  - `python pca/run_pca.py <embeddings.pkl> embedding --output_csv <pca>.pkl --output_plot <plot>.png`
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import spotipy  # type: ignore
    from spotipy.oauth2 import SpotifyClientCredentials  # type: ignore
    from spotipy.exceptions import SpotifyException  # type: ignore
except Exception:  # pragma: no cover
    spotipy = None  # type: ignore
    SpotifyClientCredentials = None  # type: ignore
    SpotifyException = Exception  # type: ignore

try:
    from langdetect import detect  # type: ignore
    from langdetect.lang_detect_exception import LangDetectException  # type: ignore

    HAS_LANGDETECT = True
except Exception:  # pragma: no cover
    detect = None  # type: ignore
    LangDetectException = Exception  # type: ignore
    HAS_LANGDETECT = False


PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"

DEFAULT_INPUT = DATA_DIR / "final_dataset_additional.csv"
DEFAULT_OUTPUT = DATA_DIR / "final_dataset_additional_enriched.csv"

SPOTIFY_TRACKS_CSV = PROJECT_ROOT / "spotify_tracks.csv"
SPOTIFY_TRACKS_MERGED_CSV = PROJECT_ROOT / "spotify_tracks_merged.csv"  # Merged from multiple Kaggle datasets
RECORD_CHARTS = DATA_DIR / "record_charts.csv"

CACHE_DIR = DATA_DIR / "cache"
CACHE_PATH = CACHE_DIR / "spotify_search_cache_final_dataset_additional.json"


# -----------------------------
# Happiness configuration
# -----------------------------

LAMBDA = 0.7  # lyrics weight in track happiness

FILTER_NEUTRAL = True
NEUTRAL_MIN = 4.0
NEUTRAL_MAX = 6.0

# Audio valence (z(valence) + mode)
ALPHA_V = 1.0
ALPHA_M = 0.25

# Audio arousal (z(energy) + z(danceability) + z(tempo))
BETA_E = 0.5
BETA_D = 0.25
BETA_T = 0.25

LABMT_LANGUAGES = {
    "en": "labMT-en-v2.csv",
    "es": "labMT-es-v2.csv",
    "fr": "labMT-fr-v2.csv",
    "de": "labMT-de-v2.csv",
    "pt": "labMT-pt-v2.csv",
    "ru": "labMT-ru-v2.csv",
    "uk": "labMT-uk-ru.csv",
    "zh": "labMT-zh-v2.csv",
    "ko": "labMT-ko-v2.csv",
    "ar": "labMT-ar-v2.csv",
}


def load_env_file(project_root: Path) -> None:
    """
    Minimal .env loader: sets env vars if not already present.
    """
    env_path = project_root / ".env"
    if not env_path.exists():
        return
    try:
        for raw in env_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            os.environ.setdefault(k, v)
    except Exception:
        return


def normalize_string(value: Any) -> str:
    """Normalize to a join key: ascii, lowercase, alnum words only."""
    if value is None or pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKD", str(value))
    text = text.encode("ascii", "ignore").decode("ascii")
    # Drop trailing metadata after " - " (common in Spotify track names)
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
    # Remove featuring/with segments
    s = re.split(r"\b(feat\.|ft\.|featuring|with)\b", s, flags=re.IGNORECASE)[0]
    # Prefer primary when multiple are present
    for sep in [";", ",", "&", " x ", " X ", " and "]:
        if sep in s:
            s = s.split(sep, 1)[0]
    # Remove parentheses qualifiers like (FIN)
    s = re.sub(r"\s*[\(\[].*?[\)\]]\s*", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def make_track_key(title: Any, artist: Any) -> str:
    return f"{normalize_string(title)}||{normalize_string(primary_artist_raw(artist))}"


def safe_ascii(s: Any) -> str:
    """Return a console-safe representation (avoid Windows cp1252 crashes)."""
    try:
        return str(s).encode("ascii", "backslashreplace").decode("ascii")
    except Exception:
        return "<unprintable>"


def get_spotify_client(requests_timeout: int = 20):
    if spotipy is None or SpotifyClientCredentials is None:  # pragma: no cover
        raise RuntimeError("spotipy is not installed. Install with: pip install spotipy")

    # Try python-dotenv; fallback to manual.
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv()
    except Exception:
        load_env_file(PROJECT_ROOT)

    client_id = os.getenv("SPOTIPY_CLIENT_ID")
    client_secret = os.getenv("SPOTIPY_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise RuntimeError(
            "Missing SPOTIPY_CLIENT_ID / SPOTIPY_CLIENT_SECRET (set env vars or add to .env)."
        )

    return spotipy.Spotify(
        client_credentials_manager=SpotifyClientCredentials(
            client_id=client_id, client_secret=client_secret
        ),
        requests_timeout=requests_timeout,
    )


def clean_title_for_search(s: Any) -> str:
    t = str(s or "").strip()
    t = re.sub(r'^[\'"]+|[\'"]+$', "", t).strip()
    t = re.sub(r"\s+", " ", t)
    # Drop trailing metadata after " - "
    if " - " in t:
        t = t.split(" - ", 1)[0].strip()
    return t


def clean_artist_for_search(s: Any) -> str:
    a = str(s or "").strip()
    a = re.sub(r'^[\'"]+|[\'"]+$', "", a).strip()
    a = re.split(r"\b(feat\.|ft\.|featuring|with)\b", a, flags=re.IGNORECASE)[0]
    # Prefer primary artist for search
    for sep in [";", ",", "&", " x ", " X ", " and "]:
        if sep in a:
            a = a.split(sep, 1)[0]
    a = re.sub(r"\s+", " ", a).strip()
    return a


def pick_best_spotify_item(items: List[dict], title: str, artist: str) -> Optional[dict]:
    """Pick best Spotify search result using simple exact-match heuristics."""
    if not items:
        return None

    nt = normalize_string(title)
    na = normalize_string(primary_artist_raw(artist))

    def score(it: dict) -> Tuple[int, int]:
        it_name = normalize_string(it.get("name"))
        it_artist = normalize_string(
            (it.get("artists") or [{}])[0].get("name") if it.get("artists") else ""
        )
        title_ok = int(it_name == nt) or int(nt != "" and nt in it_name)
        artist_ok = int(it_artist == na) or int(na != "" and na in it_artist)
        return (title_ok + artist_ok, int(it.get("popularity") or 0))

    items_sorted = sorted(items, key=lambda x: score(x), reverse=True)
    return items_sorted[0]


def spotify_search_track(
    sp,
    title: str,
    artist: str,
    limit: int = 5,
    max_retries: int = 6,
    base_sleep: float = 1.0,
) -> Optional[dict]:
    """
    Search Spotify for a track; handles 429 backoff to avoid blocking.
    """
    title_q = clean_title_for_search(title)
    artist_q = clean_artist_for_search(artist)
    if not title_q:
        return None

    queries = []
    if title_q and artist_q:
        queries.append(f"track:{title_q} artist:{artist_q}")
    queries.append(f"track:{title_q}")

    for q in queries:
        attempt = 0
        while attempt <= max_retries:
            try:
                res = sp.search(q=q, type="track", limit=limit)
                items = res.get("tracks", {}).get("items", [])
                best = pick_best_spotify_item(items, title=title, artist=artist)
                if best:
                    return best
                break  # no good match; try next query
            except SpotifyException as e:  # type: ignore
                # Respect rate limit if present
                status = getattr(e, "http_status", None)
                if status == 429:
                    retry_after = None
                    try:
                        retry_after = int(getattr(e, "headers", {}).get("Retry-After", "0"))
                    except Exception:
                        retry_after = None
                    sleep_s = retry_after if retry_after and retry_after > 0 else (base_sleep * (2**attempt))
                    time.sleep(float(min(60, max(1, sleep_s))))
                    attempt += 1
                    continue
                # Other errors: abort this query
                break
            except Exception:
                break

    return None


def load_cache(path: Path) -> Dict[str, dict]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_cache(path: Path, cache: Dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, ensure_ascii=True, indent=2), encoding="utf-8")


def load_spotify_tracks_local() -> pd.DataFrame:
    """
    Load local Spotify tracks dataset (merged from multiple Kaggle datasets if available).
    This is used as a fallback because Spotify's public Web API audio-features endpoints are restricted.
    
    Priority:
    1. spotify_tracks_merged.csv (merged from multiple Kaggle datasets)
    2. spotify_tracks.csv (original single dataset)
    """
    # Try merged dataset first (larger coverage)
    csv_path = SPOTIFY_TRACKS_MERGED_CSV if SPOTIFY_TRACKS_MERGED_CSV.exists() else SPOTIFY_TRACKS_CSV
    
    if not csv_path.exists():
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(csv_path, on_bad_lines="skip", engine="python")
    except Exception as e:
        print(f"    [WARN] Failed to load {csv_path.name}: {e}")
        return pd.DataFrame()

    if "track_id" not in df.columns:
        return pd.DataFrame()

    keep = [
        "track_id",
        "artists",
        "album_name",
        "track_name",
        "popularity",
        "duration_ms",
        "explicit",
        "danceability",
        "energy",
        "key",
        "loudness",
        "mode",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence",
        "tempo",
        "time_signature",
        "track_genre",
    ]
    keep = [c for c in keep if c in df.columns]
    df = df[keep].copy()
    df["track_id"] = df["track_id"].astype(str).str.strip()
    df = df[df["track_id"].str.len() == 22]
    df = df.drop_duplicates(subset=["track_id"], keep="first")
    return df


def fill_from_local_by_track_id(df: pd.DataFrame, spotify_local: pd.DataFrame) -> pd.DataFrame:
    if spotify_local.empty or "track_id" not in df.columns:
        return df

    merged = df.merge(
        spotify_local,
        on="track_id",
        how="left",
        suffixes=("", "_spotify_local"),
    )

    fill_cols = [
        "popularity",
        "duration_ms",
        "explicit",
        "danceability",
        "energy",
        "key",
        "loudness",
        "mode",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence",
        "tempo",
        "time_signature",
    ]

    filled_cells = 0
    for c in fill_cols:
        src = f"{c}_spotify_local"
        if src not in merged.columns:
            continue
        before = pd.to_numeric(merged.get(c), errors="coerce").notna().sum() if c in merged.columns else 0
        if c not in merged.columns:
            merged[c] = pd.NA
        merged[c] = merged[c].combine_first(merged[src])
        after = pd.to_numeric(merged[c], errors="coerce").notna().sum()
        filled_cells += max(0, after - before)
        merged = merged.drop(columns=[src])

    print(f"  Filled {filled_cells} cells from spotify_tracks.csv via track_id match.")
    return merged


def fill_track_id_and_audio_from_local_by_key(df: pd.DataFrame) -> pd.DataFrame:
    """
    Match normalized (title, primary artist) to local spotify_tracks.csv to fill:
    - track_id (if missing)
    - popularity (if missing)
    - audio feature columns (if missing)
    """
    spotify_local = load_spotify_tracks_local()
    if spotify_local.empty:
        print("  [WARN] Spotify tracks dataset not available; cannot fill audio features offline.")
        return df

    spotify_local = spotify_local.copy()
    spotify_local["k_norm"] = (
        spotify_local["track_name"].astype(str).map(normalize_string)
        + "||"
        + spotify_local["artists"].astype(str).map(primary_artist_raw).map(normalize_string)
    )

    # Prefer higher popularity for collisions
    if "popularity" in spotify_local.columns:
        spotify_local["popularity"] = pd.to_numeric(spotify_local["popularity"], errors="coerce")
        spotify_local = spotify_local.sort_values("popularity", ascending=False, na_position="last")
    spotify_best = spotify_local.drop_duplicates(subset=["k_norm"], keep="first")

    df = df.copy()
    df["k_norm"] = df.apply(lambda r: make_track_key(r.get("title"), r.get("artist")), axis=1)

    merged = df.merge(
        spotify_best,
        on="k_norm",
        how="left",
        suffixes=("", "_spotify_key"),
    )

    fill_cols = [
        "track_id",
        "track_name",
        "artists",
        "album_name",
        "popularity",
        "duration_ms",
        "explicit",
        "danceability",
        "energy",
        "key",
        "loudness",
        "mode",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence",
        "tempo",
        "time_signature",
    ]

    before_missing = pd.to_numeric(merged.get("energy"), errors="coerce").isna().sum() if "energy" in merged.columns else 0

    for c in fill_cols:
        src = f"{c}_spotify_key"
        if src not in merged.columns:
            continue
        if c not in merged.columns:
            merged[c] = pd.NA
        merged[c] = merged[c].combine_first(merged[src])
        merged = merged.drop(columns=[src])

    after_missing = pd.to_numeric(merged.get("energy"), errors="coerce").isna().sum() if "energy" in merged.columns else 0
    print(f"  Filled audio features via title+artist key match to spotify_tracks.csv: {before_missing - after_missing} tracks")

    return merged.drop(columns=["k_norm"])


def load_record_charts() -> pd.DataFrame:
    if not RECORD_CHARTS.exists():
        return pd.DataFrame(columns=["chart", "country", "continent"])
    rc = pd.read_csv(RECORD_CHARTS)
    if not {"chart", "country", "continent"}.issubset(set(rc.columns)):
        return pd.DataFrame(columns=["chart", "country", "continent"])
    rc = rc.copy()
    rc["_norm_chart"] = rc["chart"].astype(str).str.lower().str.strip()
    return rc[["_norm_chart", "country", "continent"]].drop_duplicates()


def fill_region_country_continent(row: pd.Series, rc_lookup: Dict[str, Tuple[str, str]]) -> Tuple[str, str, str]:
    """
    Returns (region, country, continent) with fallbacks.
    - region: prefers row['region'], else uses country, else heuristics by (domain, chart).
    """

    def _blank(v: Any) -> bool:
        return v is None or (isinstance(v, float) and pd.isna(v)) or (isinstance(v, str) and v.strip() == "")

    region_val = row.get("region")
    region = "" if _blank(region_val) else str(region_val).strip()
    chart = str(row.get("chart") or "").strip()
    domain = str(row.get("domain") or "").strip().lower()
    norm_chart = chart.lower().strip()

    country = ""
    continent = ""
    if norm_chart in rc_lookup:
        country, continent = rc_lookup[norm_chart]
        country = str(country or "").strip()
        continent = str(continent or "").strip()

    if not country:
        # Heuristic chart/domain -> country/continent
        if domain == "billboard.com":
            country, continent = "US", "North America"
        elif domain == "turntablecharts.com":
            country, continent = "Nigeria", "Africa"
        elif domain == "recordreport.com.ve":
            country, continent = "Venezuela", "South America"
        elif domain == "en.wikipedia.org" and "best-selling" in chart.lower():
            country, continent = "Global", "Global"
        elif domain == "research_paper":
            country, continent = "Global", "Global"

    if not region:
        region = country or ""

    return region, country, continent


def coerce_int(val: Any) -> Optional[int]:
    try:
        if pd.isna(val):
            return None
        s = str(val)
        digits = "".join(ch for ch in s if ch.isdigit())
        return int(digits) if digits else None
    except Exception:
        return None


def attach_best_scraped_chart(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach one chart appearance per track (normalized match) using the scraped catalog.
    """
    try:
        from merge_all_datasets import load_scraped_data as _load_scraped_data  # type: ignore
    except Exception:
        return df

    scraped = _load_scraped_data()
    if scraped.empty:
        return df

    scraped = scraped.copy()
    scraped["k"] = scraped.apply(lambda r: make_track_key(r.get("title"), r.get("artist")), axis=1)
    scraped["pos_int"] = scraped.get("position").map(coerce_int)
    scraped["date_dt"] = pd.to_datetime(scraped.get("date"), errors="coerce")
    scraped["has_region"] = scraped.get("region").notna() & (scraped.get("region").astype(str).str.strip() != "")
    scraped["pos_sort"] = scraped["pos_int"].fillna(9999)

    scraped = scraped.sort_values(
        by=["has_region", "pos_sort", "date_dt"],
        ascending=[False, True, False],
        na_position="last",
    )
    best = scraped.drop_duplicates(subset=["k"], keep="first")

    best = best.rename(
        columns={
            "source": "scraped_source",
            "domain": "scraped_domain",
            "chart": "scraped_chart",
            "region": "scraped_region",
            "date": "scraped_date",
            "position": "scraped_position",
            "url": "scraped_url",
            "scraped_at": "scraped_scraped_at",
        }
    )

    df = df.copy()
    df["k"] = df.apply(lambda r: make_track_key(r.get("title"), r.get("artist")), axis=1)
    df = df.merge(
        best[
            [
                "k",
                "scraped_source",
                "scraped_domain",
                "scraped_chart",
                "scraped_region",
                "scraped_date",
                "scraped_position",
                "scraped_url",
                "scraped_scraped_at",
            ]
        ],
        on="k",
        how="left",
    )
    return df


def load_all_labmt_dicts() -> Dict[str, Dict[str, float]]:
    labmt_dicts: Dict[str, Dict[str, float]] = {}
    hed = DATA_DIR / "hedonometer"
    for lang_code, filename in LABMT_LANGUAGES.items():
        path = hed / filename
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
            if "word" in df.columns and "happs" in df.columns:
                labmt_dicts[lang_code] = dict(zip(df["word"], df["happs"]))
        except Exception:
            continue
    return labmt_dicts


def detect_language(text: str) -> str:
    if not text or not isinstance(text, str) or len(text.strip()) < 10:
        return "en"
    if HAS_LANGDETECT:
        try:
            detected = detect(text)  # type: ignore
            if detected in LABMT_LANGUAGES:
                return detected
            lang_map = {"zh-cn": "zh", "zh-tw": "zh"}
            return lang_map.get(detected, "en")
        except LangDetectException:  # type: ignore
            return "en"
    return "en"


def calculate_lyric_happiness(lyrics: str, labmt_dict: Dict[str, float]) -> float:
    if not lyrics or not isinstance(lyrics, str):
        return np.nan
    words = re.findall(r"\b\w+\b", lyrics.lower())
    if not words:
        # CJK fallback: character-level
        words = [c for c in list(lyrics) if c.strip()]

    word_freqs: Dict[str, int] = {}
    word_scores: Dict[str, float] = {}

    for w in words:
        if w not in labmt_dict:
            continue
        h = labmt_dict[w]
        if FILTER_NEUTRAL and (NEUTRAL_MIN <= h <= NEUTRAL_MAX):
            continue
        word_freqs[w] = word_freqs.get(w, 0) + 1
        word_scores[w] = h

    if not word_freqs:
        return np.nan
    numerator = sum(word_freqs[w] * word_scores[w] for w in word_freqs)
    denom = sum(word_freqs.values())
    return (numerator / denom) if denom else np.nan


def normalize_to_0_1(series: pd.Series) -> pd.Series:
    """Sigmoid normalization, preserving NaNs."""
    valid = series.notna()
    out = pd.Series(np.nan, index=series.index, dtype=float)
    if not valid.any():
        return out
    x = pd.to_numeric(series[valid], errors="coerce").astype(np.float64).values
    mask = ~np.isnan(x)
    if not mask.any():
        return out
    x = np.clip(x[mask], -250, 250)
    sig = 1.0 / (1.0 + np.exp(-x))
    out.loc[series[valid].index[mask]] = sig.astype(float)
    return out


def calculate_audio_valence(df: pd.DataFrame) -> pd.Series:
    valid_valence = df["valence"].dropna()
    if len(valid_valence) == 0:
        return pd.Series(np.nan, index=df.index)
    mean = valid_valence.mean()
    std = valid_valence.std()
    z_valence = (df["valence"] - mean) / std if std and std != 0 else pd.Series(0.0, index=df.index)

    if "mode" in df.columns:
        m = df["mode"].apply(
            lambda x: 1.0 if pd.notna(x) and float(x) == 1.0 else (-1.0 if pd.notna(x) and float(x) == 0.0 else 0.0)
        )
    else:
        m = pd.Series(0.0, index=df.index)

    return ALPHA_V * z_valence + ALPHA_M * m


def calculate_audio_arousal(df: pd.DataFrame) -> pd.Series:
    def z(col: str) -> pd.Series:
        valid = df[col].dropna()
        if len(valid) == 0:
            return pd.Series(np.nan, index=df.index)
        mean = valid.mean()
        std = valid.std()
        return (df[col] - mean) / std if std and std != 0 else pd.Series(0.0, index=df.index)

    return BETA_E * z("energy") + BETA_D * z("danceability") + BETA_T * z("tempo")


def compute_happiness_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Lyrics language + labMT happiness
    labmt_dicts = load_all_labmt_dicts()
    default_dict = labmt_dicts.get("en", {})

    if "lyrics_language" not in df.columns:
        df["lyrics_language"] = pd.NA
    if "h_lyrics_raw" not in df.columns:
        df["h_lyrics_raw"] = np.nan

    processed = 0
    for idx, row in df.iterrows():
        lyrics = row.get("lyrics")
        if not isinstance(lyrics, str) or not lyrics.strip():
            continue
        lang = row.get("lyrics_language")
        if pd.isna(lang) or not str(lang).strip():
            lang = detect_language(str(lyrics))
            df.at[idx, "lyrics_language"] = lang
        labmt = labmt_dicts.get(str(lang), default_dict)
        h = calculate_lyric_happiness(str(lyrics), labmt)
        if pd.notna(h):
            df.at[idx, "h_lyrics_raw"] = h
            processed += 1

    print(f"Processed labMT lyric happiness for {processed} tracks (non-empty lyrics).")

    # Normalize lyric happiness to [0,1]
    df["h_lyrics_01"] = (pd.to_numeric(df["h_lyrics_raw"], errors="coerce") - 1) / 8
    h_mean = df["h_lyrics_01"].mean()
    h_std = df["h_lyrics_01"].std()
    if h_std and h_std != 0:
        z = (df["h_lyrics_01"] - h_mean) / h_std
        df["h_lyrics_norm"] = normalize_to_0_1(z)
    else:
        df["h_lyrics_norm"] = df["h_lyrics_01"]

    # Audio valence/arousal
    df["v_audio_raw"] = calculate_audio_valence(df)
    df["a_audio_raw"] = calculate_audio_arousal(df)
    df["v_audio_norm"] = normalize_to_0_1(df["v_audio_raw"])
    df["a_audio_norm"] = normalize_to_0_1(df["a_audio_raw"])

    # Track happiness:
    # - If both lyrics + audio exist: lambda mix
    # - If only lyrics exist: use lyrics
    # - If only audio exists: use audio valence
    # - Else: NaN
    has_lyrics = df["h_lyrics_norm"].notna()
    has_audio = df["v_audio_norm"].notna()

    df["h_track"] = np.nan
    df.loc[has_lyrics & has_audio, "h_track"] = (LAMBDA * df.loc[has_lyrics & has_audio, "h_lyrics_norm"]) + (
        (1 - LAMBDA) * df.loc[has_lyrics & has_audio, "v_audio_norm"]
    )
    df.loc[has_lyrics & ~has_audio, "h_track"] = df.loc[has_lyrics & ~has_audio, "h_lyrics_norm"]
    df.loc[~has_lyrics & has_audio, "h_track"] = df.loc[~has_lyrics & has_audio, "v_audio_norm"]

    # Catharsis score (requires arousal + track happiness)
    df["catharsis_score"] = df["a_audio_norm"] * (1 - df["h_track"])

    return df


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT))
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    parser.add_argument("--spotify_delay", type=float, default=0.35)
    parser.add_argument("--spotify_limit", type=int, default=5)
    parser.add_argument("--refresh_popularity_zeros_ones", action="store_true", default=True)
    parser.add_argument("--cache_path", type=str, default=str(CACHE_PATH))
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    cache_path = Path(args.cache_path)

    if not input_path.exists():
        print(f"Error: input not found: {input_path}", file=sys.stderr)
        return 1

    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} rows from {input_path}")

    # Drop exact duplicate rows (final_dataset_additional is supposed to be de-duped)
    before = len(df)
    df = df.drop_duplicates(keep="first")
    if len(df) != before:
        print(f"Dropped exact duplicate rows: {before} -> {len(df)}")

    # Ensure expected columns exist
    required_cols = [
        "title",
        "artist",
        "lyrics",
        "happiness_from_lyrics",
        "popularity",
        "track_id",
        "track_name",
        "artists",
        "album_name",
        "duration_ms",
        "explicit",
        "danceability",
        "energy",
        "key",
        "loudness",
        "mode",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence",
        "tempo",
        "time_signature",
    ]
    for c in required_cols:
        if c not in df.columns:
            df[c] = pd.NA

    # Keep originals for traceability
    if "popularity_orig" not in df.columns:
        df["popularity_orig"] = df["popularity"]

    # 1) Offline fill by key (no API calls)
    print("\nOffline fill from spotify_tracks.csv (title+artist key)...")
    df = fill_track_id_and_audio_from_local_by_key(df)

    # 2) Spotify enrichment for popularity/track_id (no audio-features API)
    print("\nSpotify enrichment (popularity + track_id via search)...")
    try:
        sp = get_spotify_client(requests_timeout=20)
    except Exception as e:
        print(f"[WARN] Spotify enrichment skipped: {e}", file=sys.stderr)
        sp = None

    cache: Dict[str, dict] = load_cache(cache_path)
    cache_updated = False

    if sp is not None:
        pop_num = pd.to_numeric(df["popularity"], errors="coerce")
        pop_missing = pop_num.isna()
        pop_low = (pop_num == 0) | (pop_num == 1)
        needs_pop = pop_missing | (pop_low if args.refresh_popularity_zeros_ones else False)

        # Also prioritize those missing audio features, since we want track_id for offline joins
        need_audio = pd.to_numeric(df["energy"], errors="coerce").isna()
        needs = needs_pop | need_audio

        # Only search rows with a usable title+artist
        has_title_artist = df["title"].notna() & df["artist"].notna()
        needs = needs & has_title_artist

        idxs = list(df.index[needs])
        print(f"Rows needing Spotify search: {len(idxs)} (of {len(df)})")

        matched = 0
        for i, idx in enumerate(idxs, start=1):
            title = str(df.at[idx, "title"] or "").strip()
            artist = str(df.at[idx, "artist"] or "").strip()
            if not title or not artist:
                continue

            k = make_track_key(title, artist)
            cached = cache.get(k)
            item = None
            if cached and isinstance(cached, dict) and cached.get("id"):
                item = cached
            else:
                best = spotify_search_track(sp, title=title, artist=artist, limit=int(args.spotify_limit))
                if best:
                    item = {
                        "id": best.get("id"),
                        "name": best.get("name"),
                        "artists": ";".join([a.get("name", "") for a in best.get("artists", [])]),
                        "album_name": (best.get("album") or {}).get("name"),
                        "popularity": best.get("popularity"),
                        "duration_ms": best.get("duration_ms"),
                        "explicit": best.get("explicit"),
                    }
                    cache[k] = item
                    cache_updated = True

            if item and item.get("id"):
                matched += 1
                df.at[idx, "track_id"] = item.get("id")
                # Fill Spotify text columns if missing
                if pd.isna(df.at[idx, "track_name"]) or str(df.at[idx, "track_name"]).strip() == "":
                    df.at[idx, "track_name"] = item.get("name")
                if pd.isna(df.at[idx, "artists"]) or str(df.at[idx, "artists"]).strip() == "":
                    df.at[idx, "artists"] = item.get("artists")
                if pd.isna(df.at[idx, "album_name"]) or str(df.at[idx, "album_name"]).strip() == "":
                    df.at[idx, "album_name"] = item.get("album_name")
                if pd.isna(df.at[idx, "duration_ms"]):
                    df.at[idx, "duration_ms"] = item.get("duration_ms")
                if pd.isna(df.at[idx, "explicit"]):
                    df.at[idx, "explicit"] = item.get("explicit")

                # Popularity: fill if missing or 0/1 (optional refresh)
                pop_val = pd.to_numeric(df.at[idx, "popularity"], errors="coerce")
                incoming_pop = item.get("popularity")
                if incoming_pop is not None:
                    if pd.isna(pop_val) or (args.refresh_popularity_zeros_ones and pop_val in (0, 1)):
                        df.at[idx, "popularity"] = incoming_pop

            # Gentle rate limiting
            if args.spotify_delay and args.spotify_delay > 0:
                time.sleep(float(args.spotify_delay))

            if i % 200 == 0:
                print(f"  searched {i}/{len(idxs)} (matched {matched})")

        print(f"Spotify search matched {matched}/{len(idxs)} requested rows.")

        if cache_updated:
            save_cache(cache_path, cache)
            print(f"Saved Spotify search cache: {cache_path}")

    # 3) Offline fill again by track_id (after search may have populated ids)
    print("\nOffline fill from spotify_tracks.csv (track_id join)...")
    spotify_local = load_spotify_tracks_local()
    if spotify_local.empty:
        print("  [WARN] Spotify tracks dataset not available; audio features will remain missing where not already present.")
    else:
        df = fill_from_local_by_track_id(df, spotify_local)

    # 4) Attach best chart appearance + fill region/country/continent
    print("\nAttaching chart presence (one chart per song)...")
    df = attach_best_scraped_chart(df)

    rc = load_record_charts()
    rc_lookup = {str(r["_norm_chart"]): (r.get("country", ""), r.get("continent", "")) for _, r in rc.iterrows()}

    for col in ["region", "country", "continent"]:
        if col not in df.columns:
            df[col] = pd.NA

    # Keep originals as *_orig for traceability
    for col in ["source", "domain", "chart", "region", "date", "position", "url", "scraped_at"]:
        if f"{col}_orig" not in df.columns and col in df.columns:
            df[f"{col}_orig"] = df[col]

    has_scraped = df["scraped_chart"].notna() if "scraped_chart" in df.columns else pd.Series(False, index=df.index)
    df["found_in_scraped_charts"] = has_scraped.fillna(False)

    def choose(a, b):
        return a if (pd.notna(a) and str(a).strip() != "") else b

    def _blank(v: Any) -> bool:
        return v is None or (isinstance(v, float) and pd.isna(v)) or (isinstance(v, str) and v.strip() == "")

    for idx, row in df.iterrows():
        if bool(row.get("found_in_scraped_charts")):
            df.at[idx, "source"] = choose(row.get("scraped_source"), row.get("source"))
            df.at[idx, "domain"] = choose(row.get("scraped_domain"), row.get("domain"))
            df.at[idx, "chart"] = choose(row.get("scraped_chart"), row.get("chart"))
            df.at[idx, "region"] = choose(row.get("scraped_region"), row.get("region"))
            df.at[idx, "date"] = choose(row.get("scraped_date"), row.get("date"))
            df.at[idx, "position"] = choose(row.get("scraped_position"), row.get("position"))
            df.at[idx, "url"] = choose(row.get("scraped_url"), row.get("url"))
            df.at[idx, "scraped_at"] = choose(row.get("scraped_scraped_at"), row.get("scraped_at"))

        region, country, continent = fill_region_country_continent(row=df.loc[idx], rc_lookup=rc_lookup)
        if _blank(df.at[idx, "region"]):
            df.at[idx, "region"] = region
        if _blank(df.at[idx, "country"]):
            df.at[idx, "country"] = country
        if _blank(df.at[idx, "continent"]):
            df.at[idx, "continent"] = continent

    # 5) Coerce numeric columns
    for c in [
        "energy",
        "danceability",
        "valence",
        "tempo",
        "loudness",
        "acousticness",
        "instrumentalness",
        "speechiness",
        "liveness",
        "popularity",
        "duration_ms",
        "key",
        "mode",
        "time_signature",
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # 6) Happiness scores
    print("\nComputing happiness scores (audio + labMT lyrics)...")
    df = compute_happiness_scores(df)

    # 7) Report how many audio features remain missing (expected when offline coverage is limited)
    missing_energy = int(pd.to_numeric(df["energy"], errors="coerce").isna().sum())
    print(f"\nAudio features still missing for {missing_energy} tracks (Spotify Web API audio-features is restricted; only offline fills were attempted).")

    # Drop helper columns
    df = df.drop(columns=[c for c in ["k"] if c in df.columns])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n[OK] Wrote enriched dataset to: {output_path}")
    print(f"     Input unchanged: {input_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())




