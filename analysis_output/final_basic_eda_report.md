# Basic EDA: Popularity vs Emotional Predictors (final.csv)

## Dataset snapshot

- Rows loaded: **1,550**
- Rows used for correlation/modeling (non-null popularity): **1,550**

- Missing popularity: **0**

## Correlations with popularity

Ranked by absolute **Spearman ρ** (robust to non-linear monotonic trends).

- Saved full matrices:
  - `analysis_output/final_popularity_correlations_pearson.csv`
  - `analysis_output/final_popularity_correlations_spearman.csv`
- Saved ranked popularity correlations:
  - `analysis_output/final_popularity_correlations_ranked.csv`

### Top 10 features by |Spearman ρ| with popularity

| feature | Spearman ρ | Pearson r | n |
|---|---:|---:|---:|
| danceability | +0.228 | +0.219 | 1550 |
| instrumentalness | -0.193 | -0.217 | 1550 |
| position | +0.173 | +0.029 | 589 |
| loudness | +0.145 | +0.140 | 1550 |
| explicit_int | +0.139 | +0.115 | 1550 |
| valence | +0.095 | +0.070 | 1550 |
| audio_valence_raw | +0.090 | +0.062 | 1550 |
| arousal_z | +0.086 | +0.092 | 1550 |
| liveness | -0.066 | -0.062 | 1550 |
| catharsis_score | +0.041 | +0.033 | 1550 |

## Plots

- Correlation heatmap (Spearman): `analysis_output/final_feature_correlation_heatmap.png`
- Popularity vs emotional predictors: `analysis_output/final_popularity_vs_emotional_predictors.png`
- Emotional space map: `analysis_output/final_emotional_space_popularity.png`

## Interesting facts (top vs bottom popularity quartile)

- Popularity quartiles: Q1=45.0, Q3=76.0
- Bottom quartile n=394, top quartile n=418

| feature | bottom 25% mean | top 25% mean | Δ (top-bottom) |
|---|---:|---:|---:|
| h_track | 0.511 | 0.514 | +0.003 |
| happiness_from_lyrics | 0.466 | 0.480 | +0.013 |
| arousal_z | -0.051 | 0.080 | +0.131 |
| catharsis_score | 0.239 | 0.250 | +0.011 |
| energy | 0.674 | 0.658 | -0.016 |
| loudness | -7.603 | -6.184 | +1.419 |
| valence | 0.473 | 0.513 | +0.040 |
| acousticness | 0.268 | 0.226 | -0.042 |

### Baseline model (interpretability-first)

We fit a simple ridge regression to predict Spotify popularity using:
- `h_track` (happiness)
- `arousal_z` (audio arousal proxy)
- `h_track × arousal_z` interaction (captures cathartic intensity)

- **5-fold CV R^2**: mean=-0.001, std=0.012 (n=1550)
- **Standardized coefficients** (larger magnitude = stronger association):
  - happiness (h_track_z): +0.147
  - arousal (arousal_z_z): +2.093
  - interaction (h×a): +0.871

## Notes / caveats

- Popularity is Spotify popularity (0–100), which is related to—but not identical to—chart rank.
- Chart `position` is only comparable within the same chart/date; treat any global correlation cautiously.
- `catharsis_score` is derived from arousal and happiness, so it is expected to correlate with both.


## Popularity vs chart position (within-chart sanity check)

We computed within-chart Spearman correlations between Spotify popularity and chart position (n≥20 per chart). Lower position means better rank, so **negative** correlation is expected.

- Charts analyzed: **3**
- Median ρ(popularity, position): **+0.144**
- Charts with negative ρ: **33.3%**
- Full table: `analysis_output/final_popularity_vs_chart_position_by_chart.csv`

### Strongest alignments (most negative correlations)

| source | chart | n | Spearman ρ |
|---|---|---:|---:|
| billboard | hot-100 | 348 | -0.286 |
| official_site | List of best-selling singles | 38 | +0.144 |
| official_site | Greatest of All Time Hot 100 Singles | 25 | +0.294 |
