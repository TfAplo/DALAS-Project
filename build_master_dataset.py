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
# Historically, this script expected an intermediate all_songs_merged.csv
# produced by an older version of the merge pipeline. In the current setup,
# the canonical, richest dataset is data/final_dataset.csv, which already
# contains Spotify features and our happiness scores. We therefore prefer to
# build the master dataset from final_dataset.csv when available, and fall
# back to all_songs_merged.csv only if needed.
MERGED_SONGS = DATA_DIR / "all_songs_merged.csv"
FINAL_DATASET = DATA_DIR / "final_dataset.csv"
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
    """
    Load the base chart rows that will feed into the master dataset.

    Preference order:
      1) data/final_dataset.csv  (richest, with Spotify features)
      2) data/all_songs_merged.csv (legacy intermediate, if still present)
    """
    # Preferred: build from final_dataset.csv
    if FINAL_DATASET.exists():
        df = pd.read_csv(FINAL_DATASET)
        # Ensure chart/meta columns exist
        for col in ["source", "domain", "chart", "region", "date", "position", "title", "artist", "url", "scraped_at"]:
            if col not in df.columns:
                df[col] = pd.NA
        # Geo placeholders (to be filled by attach_chart_geography)
        df["country"] = pd.NA
        df["continent"] = pd.NA
        # Coerce position
        df["position"] = df["position"].map(coerce_int)
        # Normalized keys for potential feature joins
        df["norm_title"] = df["title"].map(normalize_string)
        df["norm_artist"] = df["artist"].map(normalize_string)
        # Initialize / map feature columns expected in BASE_COLUMNS
        if "album" not in df.columns:
            # Map from album_name if present
            df["album"] = df.get("album_name", pd.NA)
        for col in ["city", "track_id"]:
            if col not in df.columns:
                df[col] = pd.NA
        for col in ["energy", "danceability", "valence", "tempo", "loudness",
                    "acousticness", "instrumentalness", "speechiness", "liveness"]:
            if col not in df.columns:
                df[col] = pd.NA
        return df

    # Fallback: legacy all_songs_merged.csv if final_dataset is not available
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


def load_extra_official_charts() -> pd.DataFrame:
    """
    Load additional chart rows from data/official_charts/*.csv
    that may not yet be present in final_dataset.csv.

    These files are produced by scraping a wide variety of official
    and semi-official charts (see scraping/scrape_official_sites.py).
    We normalize them to the base chart schema so they can be appended
    as chart-only rows (Spotify/acoustic features may remain NaN).
    """
    official_dir = DATA_DIR / "official_charts"
    if not official_dir.exists():
        return pd.DataFrame()

    parts: list[pd.DataFrame] = []
    for csv_file in sorted(official_dir.glob("*.csv")):
        try:
            df = pd.read_csv(csv_file)
        except Exception:
            continue
        if df.empty:
            continue

        # Normalize column names
        cols_lower = {c.lower(): c for c in df.columns}

        def col(*names: str) -> Optional[str]:
            for n in names:
                if n in cols_lower:
                    return cols_lower[n]
            return None

        c_domain = col("domain", "source_domain")
        c_chart = col("chart")
        c_region = col("region")
        c_date = col("date")
        c_pos = col("position", "rank")
        c_title = col("title", "song", "track")
        c_artist = col("artist", "artists", "performer")
        c_url = col("url")
        c_scraped = col("scraped_at")

        norm = pd.DataFrame()
        norm["source"] = "official_site"
        norm["domain"] = df[c_domain] if c_domain else ""
        norm["chart"] = df[c_chart] if c_chart else csv_file.stem
        norm["region"] = df[c_region] if c_region else pd.NA
        norm["date"] = df[c_date] if c_date else pd.NA
        norm["position"] = df[c_pos] if c_pos else pd.NA
        norm["title"] = df[c_title] if c_title else ""
        norm["artist"] = df[c_artist] if c_artist else ""
        norm["url"] = df[c_url] if c_url else pd.NA
        norm["scraped_at"] = df[c_scraped] if c_scraped else pd.NA

        # Coerce position
        norm["position"] = norm["position"].map(coerce_int)
        # Normalized keys
        norm["norm_title"] = norm["title"].map(normalize_string)
        norm["norm_artist"] = norm["artist"].map(normalize_string)
        # Geo + feature placeholders
        norm["country"] = pd.NA
        norm["continent"] = pd.NA
        for colname in ["city", "track_id", "album",
                        "energy", "danceability", "valence", "tempo", "loudness",
                        "acousticness", "instrumentalness", "speechiness", "liveness"]:
            if colname not in norm.columns:
                norm[colname] = pd.NA

        parts.append(norm)

    if not parts:
        return pd.DataFrame()

    extra = pd.concat(parts, ignore_index=True)
    return extra


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

    # Optionally append extra chart rows from official_charts that are not yet present.
    extra = load_extra_official_charts()
    if not extra.empty:
        # Identify rows in 'extra' that are not already in 'base' by chart identity.
        dedup_keys = ["domain", "chart", "date", "position", "title", "artist"]
        for col in dedup_keys:
            if col not in base.columns:
                base[col] = pd.NA
            if col not in extra.columns:
                extra[col] = pd.NA

        base_key = base[dedup_keys].astype(str).agg("||".join, axis=1)
        extra_key = extra[dedup_keys].astype(str).agg("||".join, axis=1)
        mask_new = ~extra_key.isin(base_key)
        extra_new = extra.loc[mask_new].copy()

        if not extra_new.empty:
            print(f"Appending {len(extra_new)} new chart-only rows from official_charts to master base.")
            base = pd.concat([base, extra_new], ignore_index=True)

    # Attach geography (applies to both original and extra rows)
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


