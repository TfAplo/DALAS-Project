"""
Basic EDA for the project problem statement:

  "Predict song popularity from audio and lyric features by modeling happiness,
   arousal, and cathartic intensity as interpretable emotional predictors of
   chart success."

This script:
- Loads `data/final.csv` (skipping high-dimensional embedding columns)
- Constructs an interpretable audio arousal proxy from Spotify audio features
- Computes Pearson + Spearman correlations (and saves them)
- Produces a correlation heatmap + a few key scatter/regression plots
- Writes a short markdown report with "interesting facts"

Outputs are written to `analysis_output/`.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).parent
DATA_PATH = PROJECT_ROOT / "data" / "final.csv"
OUT_DIR = PROJECT_ROOT / "analysis_output"


USECOLS = [
    # Identifiers / chart context (for optional group summaries)
    "source",
    "chart",
    "region",
    "date",
    "position",
    "track_genre",
    "lyrics_language",
    # Target
    "popularity",
    # Interpretable emotional predictors (computed previously)
    "h_track",
    "catharsis_score",
    # Lyric sentiment model score (already present in dataset)
    "happiness_from_lyrics",
    # Audio features (interpretable)
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


def zscore(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    mu = x.mean()
    sigma = x.std()
    if pd.isna(sigma) or sigma == 0:
        return pd.Series(0.0, index=x.index)
    return (x - mu) / sigma


def sigmoid(x: pd.Series) -> pd.Series:
    arr = pd.to_numeric(x, errors="coerce").astype(float).values
    arr = np.clip(arr, -250, 250)
    out = 1.0 / (1.0 + np.exp(-arr))
    return pd.Series(out, index=x.index, dtype=float)


def coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def main() -> int:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing dataset: {DATA_PATH}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH, usecols=USECOLS, low_memory=False)
    print(f"Loaded rows: {len(df):,}")

    # --- Coerce types ---
    numeric_cols = [
        "popularity",
        "h_track",
        "catharsis_score",
        "happiness_from_lyrics",
        "duration_ms",
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
        "position",
    ]
    df = coerce_numeric(df, numeric_cols)

    # explicit is sometimes bool/str; cast to 0/1 for analysis
    if "explicit" in df.columns:
        df["explicit_int"] = (
            df["explicit"]
            .map(lambda v: 1 if str(v).strip().lower() in ("true", "1", "yes") else (0 if str(v).strip().lower() in ("false", "0", "no") else np.nan))
            .astype("float")
        )
    else:
        df["explicit_int"] = np.nan

    # --- Interpretable audio constructs ---
    # Arousal proxy: z(energy), z(danceability), z(tempo)
    df["z_energy"] = zscore(df["energy"])
    df["z_danceability"] = zscore(df["danceability"])
    df["z_tempo"] = zscore(df["tempo"])
    df["arousal_z"] = (df["z_energy"] + df["z_danceability"] + df["z_tempo"]) / 3.0
    df["arousal_norm"] = sigmoid(df["arousal_z"])

    # Audio valence proxy: z(valence) with mode (+1 major, -1 minor)
    df["z_valence"] = zscore(df["valence"])
    mode_signed = df["mode"].map(lambda m: 1.0 if m == 1 else (-1.0 if m == 0 else 0.0))
    df["mode_signed"] = mode_signed
    df["audio_valence_raw"] = (0.8 * df["z_valence"]) + (0.2 * df["mode_signed"])
    df["audio_valence_norm"] = sigmoid(df["audio_valence_raw"])

    # A “reconstructed” catharsis (for sanity checking): arousal × (1 - happiness)
    # Uses our arousal_norm, and the existing h_track (already 0-1)
    df["catharsis_recon"] = df["arousal_norm"] * (1.0 - pd.to_numeric(df["h_track"], errors="coerce"))

    # --- Core correlation table ---
    core_features = [
        "popularity",
        # Emotional predictors
        "h_track",
        "arousal_z",
        "catharsis_score",
        # Lyric happiness model score already present in dataset (may differ from labMT-based components)
        "happiness_from_lyrics",
        # Supporting interpretable features
        "audio_valence_raw",
        "energy",
        "danceability",
        "tempo",
        "valence",
        "loudness",
        "acousticness",
        "instrumentalness",
        "speechiness",
        "liveness",
        "duration_ms",
        "explicit_int",
        # Optional “chart success” proxy in the scraped data (lower is better)
        "position",
    ]
    core_features = [c for c in core_features if c in df.columns]
    corr_df = df[core_features].copy()

    # Drop rows missing popularity for correlation and modeling
    corr_df = corr_df.dropna(subset=["popularity"])

    pearson = corr_df.corr(method="pearson")
    spearman = corr_df.corr(method="spearman")

    pearson_path = OUT_DIR / "final_popularity_correlations_pearson.csv"
    spearman_path = OUT_DIR / "final_popularity_correlations_spearman.csv"
    pearson.to_csv(pearson_path)
    spearman.to_csv(spearman_path)

    # Extract popularity correlations for ranking
    pop_corr = pd.DataFrame(
        {
            "feature": [c for c in core_features if c != "popularity"],
            "pearson_r": [pearson.loc[c, "popularity"] for c in core_features if c != "popularity"],
            "spearman_rho": [spearman.loc[c, "popularity"] for c in core_features if c != "popularity"],
            "n_nonnull": [int(corr_df[["popularity", c]].dropna().shape[0]) for c in core_features if c != "popularity"],
        }
    ).sort_values(by="spearman_rho", key=lambda s: s.abs(), ascending=False)
    pop_corr_path = OUT_DIR / "final_popularity_correlations_ranked.csv"
    pop_corr.to_csv(pop_corr_path, index=False)

    # --- Popularity vs chart rank (position), within-chart ---
    # Mixing multiple charts/dates makes raw correlation hard to interpret, so we also
    # compute within-chart Spearman correlations as a sanity check.
    pos_rows = df.dropna(subset=["popularity", "position", "chart"]).copy()
    pos_rows["chart"] = pos_rows["chart"].astype(str)
    pos_rows["source"] = pos_rows["source"].astype(str) if "source" in pos_rows.columns else ""

    by_chart = []
    for (src, chart), g in pos_rows.groupby(["source", "chart"], dropna=False):
        g2 = g[["popularity", "position"]].dropna()
        if len(g2) < 20:
            continue
        rho = g2.corr(method="spearman").iloc[0, 1]
        by_chart.append(
            {
                "source": src,
                "chart": chart,
                "n": int(len(g2)),
                # Lower position is better, so negative rho is the expected direction.
                "spearman_rho_popularity_vs_position": float(rho),
            }
        )
    by_chart_df = pd.DataFrame(by_chart).sort_values(
        by=["spearman_rho_popularity_vs_position"], ascending=True
    )
    by_chart_path = OUT_DIR / "final_popularity_vs_chart_position_by_chart.csv"
    by_chart_df.to_csv(by_chart_path, index=False)

    # --- Plots ---
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid")

    # Heatmap
    heatmap_features = [
        "popularity",
        "h_track",
        "arousal_z",
        "catharsis_score",
        "happiness_from_lyrics",
        "audio_valence_raw",
        "energy",
        "danceability",
        "tempo",
        "valence",
        "loudness",
        "acousticness",
        "speechiness",
        "instrumentalness",
        "duration_ms",
        "explicit_int",
        "position",
    ]
    heatmap_features = [c for c in heatmap_features if c in corr_df.columns]
    heat_corr = corr_df[heatmap_features].corr(method="spearman")

    plt.figure(figsize=(14, 11))
    sns.heatmap(
        heat_corr,
        cmap="vlag",
        center=0,
        annot=True,
        fmt=".2f",
        square=True,
        cbar_kws={"label": "Spearman ρ"},
    )
    plt.title("Feature Correlations (Spearman) — Focus on Popularity & Emotional Predictors")
    plt.tight_layout()
    heatmap_path = OUT_DIR / "final_feature_correlation_heatmap.png"
    plt.savefig(heatmap_path, dpi=200)
    plt.close()

    # Scatter/regression plots for the key emotional predictors
    plot_df = corr_df.copy()
    # Remove extreme outliers in loudness/tempo if any (rare) to avoid squashing plots
    plot_df = plot_df.replace([np.inf, -np.inf], np.nan)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, x, title in [
        (axes[0], "h_track", "Popularity vs Track Happiness (h_track)"),
        (axes[1], "arousal_z", "Popularity vs Audio Arousal (z-score composite)"),
        (axes[2], "catharsis_score", "Popularity vs Catharsis (catharsis_score)"),
    ]:
        if x not in plot_df.columns:
            ax.axis("off")
            continue
        sns.regplot(
            data=plot_df,
            x=x,
            y="popularity",
            scatter_kws={"alpha": 0.35, "s": 20, "edgecolor": "none"},
            line_kws={"color": "crimson"},
            ax=ax,
        )
        ax.set_title(title)
    plt.tight_layout()
    key_scatter_path = OUT_DIR / "final_popularity_vs_emotional_predictors.png"
    plt.savefig(key_scatter_path, dpi=200)
    plt.close()

    # Emotional space map: happiness vs arousal, colored by popularity
    if {"h_track", "arousal_z", "popularity"}.issubset(plot_df.columns):
        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            data=plot_df,
            x="h_track",
            y="arousal_z",
            hue="popularity",
            palette="viridis",
            alpha=0.75,
            s=35,
            edgecolor="none",
        )
        plt.title("Emotional Space (Happiness vs Arousal), colored by Popularity")
        plt.xlabel("Track Happiness (h_track)")
        plt.ylabel("Audio Arousal (z-score composite)")
        plt.legend(title="Popularity", bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()
        emotion_space_path = OUT_DIR / "final_emotional_space_popularity.png"
        plt.savefig(emotion_space_path, dpi=200)
        plt.close()
    else:
        emotion_space_path = None

    # --- Simple baseline models (interpretability-first) ---
    model_lines: List[str] = []
    try:
        from sklearn.linear_model import Ridge
        from sklearn.model_selection import KFold, cross_val_score
        from sklearn.preprocessing import StandardScaler

        model_df = corr_df[["popularity", "h_track", "arousal_z"]].dropna()
        if len(model_df) >= 50:
            X_base = model_df[["h_track", "arousal_z"]].copy()
            # Standardize base features
            scaler = StandardScaler()
            Xs = scaler.fit_transform(X_base.values)
            Xs = pd.DataFrame(Xs, columns=["h_track_z", "arousal_z_z"], index=model_df.index)
            Xs["interaction"] = Xs["h_track_z"] * Xs["arousal_z_z"]

            y = model_df["popularity"].values
            X = Xs[["h_track_z", "arousal_z_z", "interaction"]].values

            ridge = Ridge(alpha=1.0, random_state=0)
            cv = KFold(n_splits=5, shuffle=True, random_state=0)
            scores = cross_val_score(ridge, X, y, cv=cv, scoring="r2")
            ridge.fit(X, y)

            coefs = ridge.coef_
            model_lines.append("### Baseline model (interpretability-first)")
            model_lines.append("")
            model_lines.append("We fit a simple ridge regression to predict Spotify popularity using:")
            model_lines.append("- `h_track` (happiness)")
            model_lines.append("- `arousal_z` (audio arousal proxy)")
            model_lines.append("- `h_track × arousal_z` interaction (captures cathartic intensity)")
            model_lines.append("")
            model_lines.append(f"- **5-fold CV R^2**: mean={scores.mean():.3f}, std={scores.std():.3f} (n={len(model_df)})")
            model_lines.append("- **Standardized coefficients** (larger magnitude = stronger association):")
            model_lines.append(f"  - happiness (h_track_z): {coefs[0]:+.3f}")
            model_lines.append(f"  - arousal (arousal_z_z): {coefs[1]:+.3f}")
            model_lines.append(f"  - interaction (h×a): {coefs[2]:+.3f}")
        else:
            model_lines.append("### Baseline model (interpretability-first)")
            model_lines.append("")
            model_lines.append("Not enough complete rows to fit the baseline model reliably.")
    except Exception as e:
        model_lines.append("### Baseline model (interpretability-first)")
        model_lines.append("")
        model_lines.append(f"Skipped (missing optional deps or runtime issue): {e}")

    # --- “Interesting facts” summary ---
    q25 = float(corr_df["popularity"].quantile(0.25))
    q75 = float(corr_df["popularity"].quantile(0.75))
    low = corr_df[corr_df["popularity"] <= q25].copy()
    high = corr_df[corr_df["popularity"] >= q75].copy()

    def _mean(col: str, frame: pd.DataFrame) -> Optional[float]:
        if col not in frame.columns:
            return None
        v = pd.to_numeric(frame[col], errors="coerce").dropna()
        return float(v.mean()) if len(v) else None

    facts = []
    for feat in [
        "h_track",
        "happiness_from_lyrics",
        "arousal_z",
        "catharsis_score",
        "energy",
        "loudness",
        "valence",
        "acousticness",
    ]:
        lo = _mean(feat, low)
        hi = _mean(feat, high)
        if lo is None or hi is None:
            continue
        facts.append((feat, lo, hi, hi - lo))

    # --- Report ---
    report_path = OUT_DIR / "final_basic_eda_report.md"
    with report_path.open("w", encoding="utf-8") as f:
        # Relative paths (portable in the repo / report)
        pearson_rel = pearson_path.relative_to(PROJECT_ROOT).as_posix()
        spearman_rel = spearman_path.relative_to(PROJECT_ROOT).as_posix()
        ranked_rel = pop_corr_path.relative_to(PROJECT_ROOT).as_posix()
        heatmap_rel = heatmap_path.relative_to(PROJECT_ROOT).as_posix()
        key_scatter_rel = key_scatter_path.relative_to(PROJECT_ROOT).as_posix()

        f.write("# Basic EDA: Popularity vs Emotional Predictors (final.csv)\n\n")
        f.write("## Dataset snapshot\n\n")
        f.write(f"- Rows loaded: **{len(df):,}**\n")
        f.write(f"- Rows used for correlation/modeling (non-null popularity): **{len(corr_df):,}**\n")
        f.write("\n")

        missing_pop = int(df["popularity"].isna().sum()) if "popularity" in df.columns else 0
        f.write(f"- Missing popularity: **{missing_pop:,}**\n")

        # Core correlations with popularity
        f.write("\n## Correlations with popularity\n\n")
        f.write("Ranked by absolute **Spearman ρ** (robust to non-linear monotonic trends).\n\n")
        f.write(f"- Saved full matrices:\n")
        f.write(f"  - `{pearson_rel}`\n")
        f.write(f"  - `{spearman_rel}`\n")
        f.write(f"- Saved ranked popularity correlations:\n")
        f.write(f"  - `{ranked_rel}`\n\n")

        # Top 10 correlations (absolute Spearman)
        top10 = pop_corr.head(10).copy()
        f.write("### Top 10 features by |Spearman ρ| with popularity\n\n")
        f.write("| feature | Spearman ρ | Pearson r | n |\n")
        f.write("|---|---:|---:|---:|\n")
        for _, r in top10.iterrows():
            f.write(f"| {r['feature']} | {r['spearman_rho']:+.3f} | {r['pearson_r']:+.3f} | {int(r['n_nonnull'])} |\n")
        f.write("\n")

        f.write("## Plots\n\n")
        f.write(f"- Correlation heatmap (Spearman): `{heatmap_rel}`\n")
        f.write(f"- Popularity vs emotional predictors: `{key_scatter_rel}`\n")
        if emotion_space_path is not None:
            emotion_rel = emotion_space_path.relative_to(PROJECT_ROOT).as_posix()
            f.write(f"- Emotional space map: `{emotion_rel}`\n")
        f.write("\n")

        f.write("## Interesting facts (top vs bottom popularity quartile)\n\n")
        f.write(f"- Popularity quartiles: Q1={q25:.1f}, Q3={q75:.1f}\n")
        f.write(f"- Bottom quartile n={len(low):,}, top quartile n={len(high):,}\n\n")
        f.write("| feature | bottom 25% mean | top 25% mean | Δ (top-bottom) |\n")
        f.write("|---|---:|---:|---:|\n")
        for feat, lo, hi, d in facts:
            f.write(f"| {feat} | {lo:.3f} | {hi:.3f} | {d:+.3f} |\n")
        f.write("\n")

        f.write("\n".join(model_lines))
        f.write("\n")

        f.write("\n## Notes / caveats\n\n")
        f.write(
            "- Popularity is Spotify popularity (0–100), which is related to—but not identical to—chart rank.\n"
            "- Chart `position` is only comparable within the same chart/date; treat any global correlation cautiously.\n"
            "- `catharsis_score` is derived from arousal and happiness, so it is expected to correlate with both.\n"
        )

        # Exclude NaNs (can happen if a chart has constant position or too little variance)
        by_chart_valid = by_chart_df.dropna(subset=["spearman_rho_popularity_vs_position"])
        if len(by_chart_valid) > 0:
            f.write("\n\n## Popularity vs chart position (within-chart sanity check)\n\n")
            med = float(by_chart_valid["spearman_rho_popularity_vs_position"].median())
            neg_pct = float((by_chart_valid["spearman_rho_popularity_vs_position"] < 0).mean() * 100.0)
            f.write(
                f"We computed within-chart Spearman correlations between Spotify popularity and chart position "
                f"(n≥20 per chart). Lower position means better rank, so **negative** correlation is expected.\n\n"
            )
            f.write(f"- Charts analyzed: **{len(by_chart_valid)}**\n")
            f.write(f"- Median ρ(popularity, position): **{med:+.3f}**\n")
            f.write(f"- Charts with negative ρ: **{neg_pct:.1f}%**\n")
            f.write(f"- Full table: `{by_chart_path.relative_to(PROJECT_ROOT).as_posix()}`\n\n")
            f.write("### Strongest alignments (most negative correlations)\n\n")
            f.write("| source | chart | n | Spearman ρ |\n")
            f.write("|---|---|---:|---:|\n")
            for _, r in by_chart_valid.sort_values(by="spearman_rho_popularity_vs_position", ascending=True).head(10).iterrows():
                f.write(
                    f"| {r['source']} | {r['chart']} | {int(r['n'])} | {r['spearman_rho_popularity_vs_position']:+.3f} |\n"
                )

    print("\n[OK] Wrote outputs:")
    print(f"  - {pearson_path}")
    print(f"  - {spearman_path}")
    print(f"  - {pop_corr_path}")
    print(f"  - {heatmap_path}")
    print(f"  - {key_scatter_path}")
    if emotion_space_path is not None:
        print(f"  - {emotion_space_path}")
    print(f"  - {report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


