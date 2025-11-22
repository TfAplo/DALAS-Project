"""
Build a single master dataset by merging:
  1) Charts and playlist data (from merge_all_datasets output)
  2) Chart geography mapping (from data/record_charts.csv)
  3) Research-paper datasets (Top-25 by city Excel files, and any CSVs with acoustic features)
  4) Extracted PDF tables (if they contain useful numeric features)

Outputs:
  data/master_dataset.csv

Unified columns:
  source,domain,chart,region,country,continent,date,position,title,artist,url,scraped_at,
  city,track_id,album,energy,danceability,valence,tempo,loudness,acousticness,instrumentalness,speechiness,liveness
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple
import sys
import re
import unicodedata
import pandas as pd

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"

MASTER_OUT = DATA_DIR / "master_dataset.csv"
MERGED_SONGS = DATA_DIR / "all_songs_merged.csv"  # produced by merge_all_datasets.py
RECORD_CHARTS = DATA_DIR / "record_charts.csv"
RESEARCH_DIR = DATA_DIR / "research_paper"
EXTRACTED_TABLES_DIR = DATA_DIR / "extracted_tables"

BASE_COLUMNS = [
    "source", "domain", "chart", "region", "country", "continent",
    "date", "position", "title", "artist", "url", "scraped_at",
    "city", "track_id", "album",
    "energy", "danceability", "valence", "tempo", "loudness",
    "acousticness", "instrumentalness", "speechiness", "liveness",
]


def normalize_string(value: Optional[str]) -> str:
    if value is None or pd.isna(value):
        return ""
    # Strip accents and normalize spaces/punctuation for joining
    text = unicodedata.normalize("NFKD", str(value))
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def coerce_int(val) -> Optional[int]:
    try:
        s = str(val)
        digits = "".join(ch for ch in s if ch.isdigit())
        return int(digits) if digits else None
    except Exception:
        return None


def load_base_charts() -> pd.DataFrame:
    if not MERGED_SONGS.exists():
        return pd.DataFrame()
    df = pd.read_csv(MERGED_SONGS)
    # Ensure expected columns
    for col in ["source", "domain", "chart", "region", "date", "position", "title", "artist", "url", "scraped_at"]:
        if col not in df.columns:
            df[col] = pd.NA
    # Add geo placeholders
    df["country"] = pd.NA
    df["continent"] = pd.NA
    # Coerce position
    df["position"] = df["position"].map(coerce_int)
    # Create normalized keys for title+artist join
    df["norm_title"] = df["title"].map(normalize_string)
    df["norm_artist"] = df["artist"].map(normalize_string)
    # Init feature columns
    for col in ["city", "track_id", "album", "energy", "danceability", "valence", "tempo", "loudness",
                "acousticness", "instrumentalness", "speechiness", "liveness"]:
        if col not in df.columns:
            df[col] = pd.NA
    return df


def attach_chart_geography(df: pd.DataFrame) -> pd.DataFrame:
    if not RECORD_CHARTS.exists():
        return df
    rc = pd.read_csv(RECORD_CHARTS)
    # Columns: continent, country, chart, url
    needed = {"continent", "country", "chart"}
    if not needed.issubset(set(rc.columns)):
        return df
    # Normalize chart names for a safer join
    rc["_norm_chart"] = rc["chart"].astype(str).str.lower().str.strip()
    df["_norm_chart"] = df["chart"].astype(str).str.lower().str.strip()
    df = df.merge(
        rc[["_norm_chart", "continent", "country"]].drop_duplicates(),
        on="_norm_chart",
        how="left",
    )
    df = df.drop(columns=["_norm_chart"])
    return df


def scan_research_excel() -> pd.DataFrame:
    """
    Read any Excel files under data/research_paper/**/* and try to extract track rows.
    Returns a DF with columns: title, artist, city, date, album, track_id, features...
    """
    rows: List[dict] = []
    if not RESEARCH_DIR.exists():
        return pd.DataFrame()
    for path in RESEARCH_DIR.rglob("*.xlsx"):
        city = path.parent.name  # e.g., 'atlanta' from .../Top25 Tracks/20220119/atlanta.xlsx
        try:
            df_x = pd.read_excel(path)
        except Exception:
            continue
        # Try to map common column names
        cols_lower = {c.lower(): c for c in df_x.columns}
        col_title = cols_lower.get("track", cols_lower.get("track name", cols_lower.get("title")))
        if col_title is None:
            # heuristic: first column likely title
            col_title = df_x.columns[0]
        col_artist = cols_lower.get("artist", cols_lower.get("artists", cols_lower.get("performer")))
        # Optional fields
        col_album = cols_lower.get("album", cols_lower.get("album name"))
        col_date = cols_lower.get("date")
        col_tid = cols_lower.get("track_id", cols_lower.get("id"))
        # Feature columns if present
        feature_cols = {}
        for k in ["energy", "danceability", "valence", "tempo", "loudness",
                  "acousticness", "instrumentalness", "speechiness", "liveness"]:
            if k in cols_lower:
                feature_cols[k] = cols_lower[k]

        for _, r in df_x.iterrows():
            rows.append({
                "title": r.get(col_title),
                "artist": r.get(col_artist),
                "album": r.get(col_album),
                "date": r.get(col_date),
                "city": city,
                "track_id": r.get(col_tid),
                **{k: r.get(v) for k, v in feature_cols.items()},
            })
    return pd.DataFrame(rows)


def scan_research_csvs() -> pd.DataFrame:
    """
    Read any CSVs in research_paper dir that include acoustic features.
    """
    rows: List[pd.DataFrame] = []
    if not RESEARCH_DIR.exists():
        return pd.DataFrame()
    for path in RESEARCH_DIR.rglob("*.csv"):
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        # Only keep if there is at least one known feature column
        if not set(df.columns).intersection({"energy", "danceability", "valence", "tempo", "loudness",
                                             "acousticness", "instrumentalness", "speechiness", "liveness"}):
            continue
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def scan_extracted_tables() -> pd.DataFrame:
    """
    Include any extracted PDF tables that have per-track or per-day features.
    """
    if not EXTRACTED_TABLES_DIR.exists():
        return pd.DataFrame()
    parts = []
    for path in EXTRACTED_TABLES_DIR.glob("*.csv"):
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        parts.append(df)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def attach_track_features(base: pd.DataFrame, features_like: pd.DataFrame) -> pd.DataFrame:
    """
    Join features onto base by normalized title+artist when possible.
    """
    if features_like.empty:
        return base
    f = features_like.copy()
    # try to detect track & artist columns
    candidate_title = None
    for c in f.columns:
        if c.lower() in {"title", "track", "track name", "song", "name"}:
            candidate_title = c
            break
    candidate_artist = None
    for c in f.columns:
        if c.lower() in {"artist", "artists", "performer"}:
            candidate_artist = c
            break
    if candidate_title is None or candidate_artist is None:
        return base

    f["norm_title"] = f[candidate_title].map(normalize_string)
    f["norm_artist"] = f[candidate_artist].map(normalize_string)
    feature_cols = [c for c in f.columns if c.lower() in {
        "track_id", "album", "energy", "danceability", "valence", "tempo", "loudness",
        "acousticness", "instrumentalness", "speechiness", "liveness"
    }]
    feature_cols = list(dict.fromkeys(feature_cols))  # unique preserve order

    merged = base.merge(
        f[["norm_title", "norm_artist"] + feature_cols],
        on=["norm_title", "norm_artist"],
        how="left",
        suffixes=("", "_feat"),
    )
    # Fill base columns where feature columns exist
    for col in feature_cols:
        if col in merged.columns and col in BASE_COLUMNS:
            merged[col] = merged[col].combine_first(merged[col])
        elif col in merged.columns and col + "_feat" in merged.columns:
            merged[col] = merged[col].combine_first(merged[col + "_feat"])
    # Drop helper columns used for merge suffixes if present
    drop_cols = [c for c in merged.columns if c.endswith("_feat")]
    merged = merged.drop(columns=drop_cols, errors="ignore")
    return merged


def build_master() -> int:
    base = load_base_charts()
    if base.empty:
        print("No base merged chart data found. Run merge_all_datasets.py first.", file=sys.stderr)
        return 0
    base = attach_chart_geography(base)

    # Attach features from research Excel/CSVs and extracted tables, if matchable
    research_excel = scan_research_excel()
    if not research_excel.empty:
        base = attach_track_features(base, research_excel)
    research_csvs = scan_research_csvs()
    if not research_csvs.empty:
        base = attach_track_features(base, research_csvs)

    # Extracted tables are often aggregated; include only if they have recognizable columns
    extracted = scan_extracted_tables()
    if not extracted.empty:
        base = attach_track_features(base, extracted)

    # Finalize columns
    for col in BASE_COLUMNS:
        if col not in base.columns:
            base[col] = pd.NA
    base = base[BASE_COLUMNS]

    # Deduplicate
    dedup_keys = ["domain", "chart", "date", "position", "title", "artist"]
    base = base.drop_duplicates(subset=dedup_keys, keep="first")

    MASTER_OUT.parent.mkdir(parents=True, exist_ok=True)
    base.to_csv(MASTER_OUT, index=False)
    print(f"Wrote {len(base)} rows to {MASTER_OUT}")
    return len(base)


if __name__ == "__main__":
    raise SystemExit(build_master())


