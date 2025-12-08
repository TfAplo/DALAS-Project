"""
Augment data/master_dataset.csv with Spotify features and happiness scores.

This script:
  1. Loads data/master_dataset.csv and data/final_dataset.csv.
  2. Copies over Spotify audio/metadata features and precomputed happiness
     scores from final_dataset wherever we can match rows.
  3. First matches on full chart identity:
       (domain, chart, date, position, title, artist)
  4. Then, for any remaining rows that still lack core audio features, it
     attempts a more relaxed match on normalized (title, artist) using
     the same cleaning logic as merge_all_datasets.py.
  5. Writes the updated master_dataset.csv in-place, after saving a backup.
"""

from __future__ import annotations

from pathlib import Path
from typing import List
import sys

import numpy as np
import pandas as pd

try:
    # Optional, for relaxed normalized matching
    from merge_all_datasets import clean_text, primary_artist
except ImportError:  # pragma: no cover - optional dependency
    clean_text = None  # type: ignore[assignment]
    primary_artist = None  # type: ignore[assignment]


PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MASTER_PATH = DATA_DIR / "master_dataset.csv"
FINAL_PATH = DATA_DIR / "final_dataset.csv"

DEDUP_KEYS: List[str] = ["domain", "chart", "date", "position", "title", "artist"]

SPOTIFY_FEATURE_COLS: List[str] = [
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
    "track_genre",
    "playlistname",
]

HAPPINESS_COLS: List[str] = [
    "v_audio_raw",
    "a_audio_raw",
    "v_audio_norm",
    "a_audio_norm",
    "lyrics",
    "lyrics_language",
    "h_lyrics_raw",
    "h_lyrics_01",
    "h_lyrics_norm",
    "h_track",
    "catharsis_score",
    "happiness_from_lyrics",
]


def main() -> int:
    if not MASTER_PATH.exists():
        print(f"Error: {MASTER_PATH} not found. Run build_master_dataset.py first.", file=sys.stderr)
        return 1
    if not FINAL_PATH.exists():
        print(
            f"Error: {FINAL_PATH} not found. Run merge_all_datasets.py and "
            "calculate_comprehensive_happiness.py first.",
            file=sys.stderr,
        )
        return 1

    master = pd.read_csv(MASTER_PATH)
    final = pd.read_csv(FINAL_PATH)

    # Determine which columns we can actually pull from final_dataset
    avail_feature_cols = [c for c in SPOTIFY_FEATURE_COLS if c in final.columns]
    avail_happy_cols = [c for c in HAPPINESS_COLS if c in final.columns]
    cols_to_pull = avail_feature_cols + avail_happy_cols

    if not cols_to_pull:
        print(
            "Error: final_dataset.csv does not contain any expected Spotify feature or "
            "happiness columns.",
            file=sys.stderr,
        )
        print("Make sure you've run calculate_comprehensive_happiness.py.", file=sys.stderr)
        return 1

    # Ensure key columns exist
    missing_in_final = [k for k in DEDUP_KEYS if k not in final.columns]
    missing_in_master = [k for k in DEDUP_KEYS if k not in master.columns]
    if missing_in_final:
        print(f"Error: final_dataset.csv is missing key columns: {missing_in_final}", file=sys.stderr)
        return 1
    if missing_in_master:
        print(f"Error: master_dataset.csv is missing key columns: {missing_in_master}", file=sys.stderr)
        return 1

    print(f"Loaded master_dataset with {len(master)} rows.")
    print(f"Loaded final_dataset with {len(final)} rows.")

    # 1) Exact chart-identity match
    final_subset = final[DEDUP_KEYS + cols_to_pull].copy()
    merged = master.merge(
        final_subset,
        on=DEDUP_KEYS,
        how="left",
        suffixes=("", "_from_final"),
    )

    attached_counts = {}

    for col in cols_to_pull:
        src_col = f"{col}_from_final"
        if src_col not in merged.columns:
            # Column only exists in master (or not at all); nothing to fill from final
            continue

        before_non_null = merged[col].notna().sum() if col in merged.columns else 0

        if col not in merged.columns:
            merged[col] = merged[src_col]
        else:
            merged[col] = merged[col].combine_first(merged[src_col])

        after_non_null = merged[col].notna().sum()
        attached_counts[col] = after_non_null - before_non_null

        merged = merged.drop(columns=[src_col])

    print("Step 1: attached columns from final_dataset via exact chart key:")
    for col, inc in sorted(attached_counts.items()):
        if inc > 0:
            print(f"  {col}: +{inc} filled values")

    # 2) Relaxed normalized (title, artist) match for rows still missing core audio features
    core_audio_cols = [c for c in ["energy", "danceability", "valence", "tempo"] if c in merged.columns]
    if core_audio_cols and clean_text is not None and primary_artist is not None:
        # Rows with all core audio cols missing
        mask_missing_core = merged[core_audio_cols].isna().all(axis=1)
        n_missing = int(mask_missing_core.sum())
        if n_missing:
            print(
                f"Step 2: attempting relaxed title/artist match for {n_missing} "
                "rows without core audio features..."
            )

            subset = merged.loc[mask_missing_core].copy()
            subset["_orig_index"] = subset.index

            subset["norm_title"] = subset["title"].apply(clean_text)
            subset["norm_artist"] = subset["artist"].apply(primary_artist)

            # Build Spotify track-level lookup if we have track_name/artists
            if "track_name" in final.columns and "artists" in final.columns:
                spotify = final.copy()
                spotify["norm_title"] = spotify["track_name"].apply(clean_text)
                spotify["norm_artist"] = spotify["artists"].apply(primary_artist)

                norm_feature_cols = [c for c in cols_to_pull if c in spotify.columns]
                agg_dict = {c: "first" for c in norm_feature_cols}
                spotify_lookup = (
                    spotify.groupby(["norm_title", "norm_artist"], as_index=False)
                    .agg(agg_dict)
                )

                subset = subset.merge(
                    spotify_lookup,
                    on=["norm_title", "norm_artist"],
                    how="left",
                    suffixes=("", "_spotify"),
                )

                relaxed_counts = {}

                for col in norm_feature_cols:
                    src_col = f"{col}_spotify"
                    if src_col not in subset.columns:
                        continue

                    # Ensure merged has this column
                    if col not in merged.columns:
                        merged[col] = np.nan

                    before_non_null = merged[col].notna().sum()

                    # Series indexed by original indices
                    new_series = pd.Series(
                        subset[src_col].values,
                        index=subset["_orig_index"].values,
                    )
                    merged[col] = merged[col].combine_first(new_series)

                    after_non_null = merged[col].notna().sum()
                    relaxed_counts[col] = after_non_null - before_non_null

                print("Step 2: additional values filled via normalized matching:")
                for col, inc in sorted(relaxed_counts.items()):
                    if inc > 0:
                        print(f"  {col}: +{inc} filled values")
            else:
                print("  Warning: final_dataset is missing track_name/artists; skipping relaxed matching.")
    else:
        if clean_text is None or primary_artist is None:
            print("Step 2: merge_all_datasets.clean_text/primary_artist not available; skipping relaxed matching.")
        else:
            print("Step 2: no rows missing core audio features; relaxed match not needed.")

    # Reorder columns: keep original master columns first, then any new ones
    original_cols = list(master.columns)
    new_cols = [c for c in merged.columns if c not in original_cols]
    merged = merged[original_cols + new_cols]

    # Backup original and write result
    backup_path = MASTER_PATH.with_suffix(".backup_before_happiness.csv")
    master.to_csv(backup_path, index=False)
    print(f"Backed up original master_dataset to {backup_path}")

    merged.to_csv(MASTER_PATH, index=False)
    print(f"Wrote augmented master_dataset with {len(new_cols)} extra columns to {MASTER_PATH}")
    print("New columns:", new_cols)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


