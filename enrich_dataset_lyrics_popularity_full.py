"""
End-to-end enrichment for `data/dataset_lyrics_popularity.csv`.

This script produces a NEW dataset (does not overwrite the input) by:
  1) Fetching Spotify popularity + audio features (when missing) via Spotify API.
     - Uses Spotify search by `title` + `artist` to obtain `track_id`.
     - Fetches audio features in batches via `audio_features`.
     - Stores both original and Spotify-derived popularity.
  2) Computing happiness scores using the same approach as `calculate_comprehensive_happiness.py`:
     - Audio valence/arousal (sigmoid-normalized)
     - labMT lyric happiness (language-aware where possible)
     - Track happiness (lyrics+audio) and catharsis score
  3) Attaching ONE chart appearance (if present) from the scraped chart catalog
     using normalized (title, artist) matching, and filling region/country/continent.

Then you can run:
  - `python embed/embed_lyrics.py <output_csv> lyrics --output <out>.pkl --keep_all`
  - `python pca/run_pca.py <embeddings.pkl> embedding --output_csv <pca>.pkl --output_plot <plot>.png`
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import pandas as pd

# Optional language detection
try:
    from langdetect import detect, LangDetectException

    HAS_LANGDETECT = True
except ImportError:  # pragma: no cover
    HAS_LANGDETECT = False


try:
    import spotipy
    from spotipy.oauth2 import SpotifyClientCredentials
except ImportError:  # pragma: no cover
    spotipy = None  # type: ignore[assignment]
    SpotifyClientCredentials = None  # type: ignore[assignment]


PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
SPOTIFY_TRACKS_CSV = PROJECT_ROOT / "spotify_tracks.csv"

DEFAULT_INPUT = DATA_DIR / "dataset_lyrics_popularity.csv"
DEFAULT_OUTPUT = DATA_DIR / "dataset_lyrics_popularity_enriched.csv"
RECORD_CHARTS = DATA_DIR / "record_charts.csv"

# labMT dictionaries mapping (same filenames as calculate_comprehensive_happiness.py)
LABMT_LANGUAGES: Dict[str, str] = {
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

# Happiness scoring parameters (mirrors calculate_comprehensive_happiness.py)
ALPHA_V = 0.8
ALPHA_M = 0.2
BETA_E = 1 / 3
BETA_D = 1 / 3
BETA_T = 1 / 3
LAMBDA = 0.6

FILTER_NEUTRAL = True
NEUTRAL_MIN = 4.0
NEUTRAL_MAX = 6.0


def safe_console(s: object) -> str:
    """Avoid Windows console UnicodeEncodeError by forcing ASCII-safe output."""
    try:
        return str(s).encode("ascii", "replace").decode("ascii")
    except Exception:
        return "<unprintable>"


def load_env_file(project_root: Path) -> None:
    """Manual .env loader (fallback when python-dotenv isn't installed)."""
    env_path = project_root / ".env"
    if not env_path.exists():
        return
    try:
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and value and key not in os.environ:
                os.environ[key] = value
    except Exception:
        # Best effort only
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
    return s.strip()


def make_track_key(title: Any, artist: Any) -> str:
    return f"{normalize_string(title)}||{normalize_string(primary_artist_raw(artist))}"


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


def pick_best_spotify_item(
    items: List[dict], title: str, artist: str
) -> Optional[dict]:
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
    sp, title: str, artist: str, limit: int = 5
) -> Optional[dict]:
    title_q = clean_title_for_search(title)
    artist_q = clean_artist_for_search(artist)
    if not title_q:
        return None

    queries = []
    if title_q and artist_q:
        queries.append(f"track:{title_q} artist:{artist_q}")
    queries.append(f"track:{title_q}")

    for q in queries:
        try:
            res = sp.search(q=q, type="track", limit=limit)
            items = res.get("tracks", {}).get("items", [])
            best = pick_best_spotify_item(items, title=title, artist=artist)
            if best:
                return best
        except Exception:
            continue

    return None


def chunked(seq: List[str], n: int) -> List[List[str]]:
    return [seq[i : i + n] for i in range(0, len(seq), n)]


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


def load_spotify_tracks_local() -> pd.DataFrame:
    """
    Load local `spotify_tracks.csv` (Kaggle-derived) which contains audio features + popularity.
    This is used as a fallback because Spotify's public Web API may return 403 for /audio-features.
    """
    if not SPOTIFY_TRACKS_CSV.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(SPOTIFY_TRACKS_CSV, on_bad_lines="skip", engine="python")
    except Exception:
        return pd.DataFrame()

    # Normalize expected columns
    if "track_id" not in df.columns:
        # Some copies include an unnamed first column; track_id may be second
        cols = [c for c in df.columns if "track_id" in c.lower()]
        if cols:
            df = df.rename(columns={cols[0]: "track_id"})
        else:
            return pd.DataFrame()

    # Keep only relevant columns to reduce memory
    keep = [
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
    keep = [c for c in keep if c in df.columns]
    df = df[keep].copy()

    # Ensure track_id is string
    df["track_id"] = df["track_id"].astype(str).str.strip()
    df = df[df["track_id"].str.len() == 22]
    df = df.drop_duplicates(subset=["track_id"], keep="first")
    return df


def fill_audio_from_spotify_tracks_by_key(df: pd.DataFrame) -> pd.DataFrame:
    """
    Second fallback: fill audio features by matching normalized (title, primary artist)
    against local `spotify_tracks.csv`. This helps when Spotify search track_id is not
    present in the Kaggle file.
    """
    spotify_local = load_spotify_tracks_local()
    if spotify_local.empty:
        return df

    # Build keys
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

    # Track how many rows get at least one audio feature filled
    before_missing = pd.to_numeric(df.get("energy"), errors="coerce").isna().sum() if "energy" in df.columns else 0

    for c in fill_cols:
        src = f"{c}_spotify_key"
        if src not in merged.columns:
            continue
        if c not in merged.columns:
            merged[c] = pd.NA
        merged[c] = merged[c].combine_first(merged[src])
        merged = merged.drop(columns=[src])

    after_missing = pd.to_numeric(merged.get("energy"), errors="coerce").isna().sum() if "energy" in merged.columns else 0
    print(f"  Filled additional audio features via title+artist key match to spotify_tracks.csv: {before_missing - after_missing} tracks")

    return merged.drop(columns=["k_norm"])


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
            detected = detect(text)
            if detected in LABMT_LANGUAGES:
                return detected
            lang_map = {"zh-cn": "zh", "zh-tw": "zh"}
            return lang_map.get(detected, "en")
        except LangDetectException:
            return "en"
    return "en"


def calculate_lyric_happiness(
    lyrics: str, labmt_dict: Dict[str, float]
) -> float:
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

    # Build join keys
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT))
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    parser.add_argument("--spotify_delay", type=float, default=0.35)
    parser.add_argument("--spotify_limit", type=int, default=5)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Error: input not found: {input_path}", file=sys.stderr)
        return 1

    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} rows from {input_path}")

    # Ensure expected columns exist
    required_cols = [
        "title",
        "artist",
        "lyrics",
        "happiness_from_lyrics",
        "popularity",
        "energy",
        "danceability",
        "valence",
        "tempo",
        "loudness",
        "acousticness",
        "instrumentalness",
        "speechiness",
        "liveness",
    ]
    for c in required_cols:
        if c not in df.columns:
            df[c] = pd.NA

    # Keep originals for traceability
    if "popularity_orig" not in df.columns:
        df["popularity_orig"] = df["popularity"]

    # Spotify enrichment (track id + popularity + audio features)
    print("\nSpotify enrichment...")
    try:
        sp = get_spotify_client(requests_timeout=20)
    except Exception as e:
        print(f"[WARN] Spotify enrichment skipped: {e}", file=sys.stderr)
        sp = None

    if sp is not None:
        # Ensure columns we will populate
        for col in [
            "track_id",
            "spotify_track_name",
            "spotify_artists",
            "spotify_album_name",
            "spotify_popularity",
            "duration_ms",
            "explicit",
            "key",
            "mode",
            "time_signature",
        ]:
            if col not in df.columns:
                df[col] = pd.NA

        # Ensure track_id column is object dtype (avoid pandas dtype warnings)
        df["track_id"] = df["track_id"].astype("object")

        # First pass: search to obtain track IDs (and popularity)
        track_ids: List[str] = []
        row_to_track_id: Dict[int, str] = {}
        failed = 0

        for idx, row in df.iterrows():
            title = str(row.get("title", "") or "").strip()
            artist = str(row.get("artist", "") or "").strip()
            if not title or not artist:
                failed += 1
                continue

            item = spotify_search_track(sp, title=title, artist=artist, limit=int(args.spotify_limit))
            if not item:
                failed += 1
                continue

            tid = item.get("id")
            if not tid:
                failed += 1
                continue

            row_to_track_id[idx] = tid
            track_ids.append(tid)

            # Populate lightweight track metadata (from search result)
            df.at[idx, "track_id"] = tid
            df.at[idx, "spotify_track_name"] = item.get("name")
            df.at[idx, "spotify_artists"] = ";".join([a.get("name", "") for a in item.get("artists", [])])
            df.at[idx, "spotify_album_name"] = (item.get("album") or {}).get("name")
            df.at[idx, "spotify_popularity"] = item.get("popularity")
            df.at[idx, "duration_ms"] = item.get("duration_ms")
            df.at[idx, "explicit"] = item.get("explicit")

            # We treat Spotify as source-of-truth for popularity when available.
            if pd.notna(df.at[idx, "spotify_popularity"]):
                df.at[idx, "popularity"] = df.at[idx, "spotify_popularity"]

            # Gentle rate limiting
            if args.spotify_delay and args.spotify_delay > 0:
                time.sleep(float(args.spotify_delay))

            if (idx + 1) % 100 == 0:
                print(f"  searched {idx+1}/{len(df)} (matched {len(track_ids)})")

        print(f"Spotify search matched {len(track_ids)}/{len(df)} rows; failed {failed}.")

        # Audio features fallback (local spotify_tracks.csv) because Spotify /audio-features returns 403 for client-credentials.
        print("Filling audio features from local spotify_tracks.csv (Kaggle-derived)...")
        spotify_local = load_spotify_tracks_local()
        if spotify_local.empty:
            print("  [WARN] spotify_tracks.csv not available; audio features will remain missing where not already present.")
        else:
            # Merge by track_id first
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
                before = pd.to_numeric(merged[c], errors="coerce").notna().sum() if c in merged.columns else 0
                merged[c] = merged[c].combine_first(merged[src])
                after = pd.to_numeric(merged[c], errors="coerce").notna().sum()
                filled_cells += max(0, after - before)
                merged = merged.drop(columns=[src])

            df = merged
            print(f"  Filled {filled_cells} cells from spotify_tracks.csv via track_id match.")

        # Second fallback: fill remaining missing audio features via normalized title+artist key
        df = fill_audio_from_spotify_tracks_by_key(df)

    # Attach best scraped chart appearance (if present)
    print("\nAttaching chart presence (one chart per song)...")
    df = attach_best_scraped_chart(df)

    # Fill region/country/continent from record_charts + heuristics
    rc = load_record_charts()
    rc_lookup = {
        str(r["_norm_chart"]): (r.get("country", ""), r.get("continent", ""))
        for _, r in rc.iterrows()
    }

    for col in ["region", "country", "continent"]:
        if col not in df.columns:
            df[col] = pd.NA

    # Use scraped_* if present for chart fields; keep originals as *_orig for traceability
    for col in ["source", "domain", "chart", "region", "date", "position", "url", "scraped_at"]:
        if f"{col}_orig" not in df.columns and col in df.columns:
            df[f"{col}_orig"] = df[col]

    # Prefer scraped chart assignment if available (indicates presence in scraped charts)
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

    # Coerce numeric audio columns
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
        "mode",
        "key",
        "time_signature",
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Compute happiness scores
    print("\nComputing happiness scores (audio + labMT lyrics)...")
    labmt_dicts = load_all_labmt_dicts()
    default_dict = labmt_dicts.get("en", {})

    if "lyrics_language" not in df.columns:
        df["lyrics_language"] = pd.NA
    df["h_lyrics_raw"] = pd.NA

    processed = 0
    for idx, row in df.iterrows():
        lyrics = row.get("lyrics")
        if pd.isna(lyrics) or not str(lyrics).strip():
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

    # Track happiness
    df["h_track"] = (LAMBDA * df["h_lyrics_norm"].fillna(0.0)) + ((1 - LAMBDA) * df["v_audio_norm"])
    has_lyrics = df["h_lyrics_norm"].notna()
    df.loc[has_lyrics, "h_track"] = (LAMBDA * df.loc[has_lyrics, "h_lyrics_norm"]) + (
        (1 - LAMBDA) * df.loc[has_lyrics, "v_audio_norm"]
    )

    # Catharsis score
    df["catharsis_score"] = df["a_audio_norm"] * (1 - df["h_track"])

    # Drop helper key columns we don't want in the output
    df = df.drop(columns=[c for c in ["k", "pos_int", "pos_sort", "date_dt", "has_region"] if c in df.columns])

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n[OK] Wrote enriched dataset to: {output_path}")
    print(f"     Input unchanged: {input_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


