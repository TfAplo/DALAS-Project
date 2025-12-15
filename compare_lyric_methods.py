"""
Comparative Analysis of Lyric Happiness Scoring Methods

This script compares two approaches to calculating lyric happiness:
1. BERT-based sentiment analysis (happiness_from_lyrics)
2. labMT/Hedonometer word-level scoring (h_lyrics_norm)

It performs:
- Correlation analysis
- Agreement/disagreement identification
- Language-specific comparisons
- Distribution analysis
- Statistical validation
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr

DATA_DIR = Path("data")
FINAL_DATASET_PATH = DATA_DIR / "final_dataset.csv"
DATASET_LYRICS_PATH = DATA_DIR / "dataset_lyrics.csv"
OUTPUT_DIR = Path("analysis_output")
OUTPUT_DIR.mkdir(exist_ok=True)


def load_and_merge_data() -> pd.DataFrame:
    """Load both datasets and merge happiness scores."""
    print("=" * 80)
    print("Loading and Merging Data")
    print("=" * 80)
    
    # Load main dataset
    if not FINAL_DATASET_PATH.exists():
        raise FileNotFoundError(f"{FINAL_DATASET_PATH} not found")
    
    print(f"\n1. Loading {FINAL_DATASET_PATH}...")
    df = pd.read_csv(FINAL_DATASET_PATH)
    print(f"   [OK] Loaded {len(df)} tracks")
    
    # Load lyrics dataset with BERT scores
    if DATASET_LYRICS_PATH.exists():
        print(f"\n2. Loading {DATASET_LYRICS_PATH}...")
        lyrics_df = pd.read_csv(DATASET_LYRICS_PATH)
        print(f"   [OK] Loaded {len(lyrics_df)} tracks with lyrics")
        
        # Merge BERT scores if not already present
        if 'happiness_from_lyrics' in lyrics_df.columns:
            # Use 'title' and 'artist' as merge keys (both files have these)
            merge_cols = ['title', 'artist', 'happiness_from_lyrics']
            if 'lyrics' in lyrics_df.columns:
                merge_cols.append('lyrics')
            
            lyrics_merge = lyrics_df[merge_cols].dropna(subset=['title', 'artist'])
            
            # Merge, keeping existing data where available
            df = df.merge(
                lyrics_merge,
                on=['title', 'artist'],
                how='left',
                suffixes=('', '_from_file')
            )
            
            # Fill missing BERT scores
            if 'happiness_from_lyrics_from_file' in df.columns:
                df['happiness_from_lyrics'] = df['happiness_from_lyrics'].fillna(
                    df['happiness_from_lyrics_from_file']
                )
                df = df.drop(columns=['happiness_from_lyrics_from_file'])
            
            # Fill missing lyrics
            if 'lyrics_from_file' in df.columns:
                df['lyrics'] = df['lyrics'].fillna(df['lyrics_from_file'])
                df = df.drop(columns=['lyrics_from_file'])
            
            print(f"   [OK] Merged BERT scores: {df['happiness_from_lyrics'].notna().sum()} tracks")
        else:
            print("   [WARN] No 'happiness_from_lyrics' column found in lyrics dataset")
    else:
        print(f"\n2. [WARN] {DATASET_LYRICS_PATH} not found, skipping BERT score merge")
    
    # Filter to tracks with both scores
    has_both = df['happiness_from_lyrics'].notna() & df['h_lyrics_norm'].notna()
    print(f"\n3. Tracks with both scores: {has_both.sum()}")
    print(f"   Tracks with only BERT score: {(df['happiness_from_lyrics'].notna() & df['h_lyrics_norm'].isna()).sum()}")
    print(f"   Tracks with only labMT score: {(df['happiness_from_lyrics'].isna() & df['h_lyrics_norm'].notna()).sum()}")
    
    return df


def calculate_correlations(df: pd.DataFrame) -> dict:
    """Calculate correlation statistics between the two methods."""
    print("\n" + "=" * 80)
    print("Correlation Analysis")
    print("=" * 80)
    
    # Filter to tracks with both scores
    both_scores = df[df['happiness_from_lyrics'].notna() & df['h_lyrics_norm'].notna()].copy()
    
    if len(both_scores) == 0:
        print("   [WARN] No tracks with both scores available")
        return {}
    
    bert = both_scores['happiness_from_lyrics']
    labmt = both_scores['h_lyrics_norm']
    
    # Pearson correlation
    pearson_r, pearson_p = pearsonr(bert, labmt)
    
    # Spearman correlation (rank-based, less sensitive to outliers)
    spearman_r, spearman_p = spearmanr(bert, labmt)
    
    # Mean absolute difference
    mad = np.mean(np.abs(bert - labmt))
    
    # Root mean square difference
    rmsd = np.sqrt(np.mean((bert - labmt) ** 2))
    
    # Agreement within 0.1 (10% of scale)
    agreement_01 = np.mean(np.abs(bert - labmt) <= 0.1)
    
    # Agreement within 0.2 (20% of scale)
    agreement_02 = np.mean(np.abs(bert - labmt) <= 0.2)
    
    results = {
        'n_tracks': len(both_scores),
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
        'spearman_r': spearman_r,
        'spearman_p': spearman_p,
        'mean_abs_diff': mad,
        'rmsd': rmsd,
        'agreement_10pct': agreement_01,
        'agreement_20pct': agreement_02,
        'bert_mean': bert.mean(),
        'bert_std': bert.std(),
        'labmt_mean': labmt.mean(),
        'labmt_std': labmt.std(),
    }
    
    print(f"\nSample size: {results['n_tracks']} tracks")
    print(f"\nCorrelation Statistics:")
    print(f"  Pearson r:  {results['pearson_r']:.4f} (p={results['pearson_p']:.4f})")
    print(f"  Spearman rho: {results['spearman_r']:.4f} (p={results['spearman_p']:.4f})")
    print(f"\nAgreement Metrics:")
    print(f"  Mean absolute difference: {results['mean_abs_diff']:.4f}")
    print(f"  Root mean square difference: {results['rmsd']:.4f}")
    print(f"  Agreement within 0.1: {results['agreement_10pct']:.1%}")
    print(f"  Agreement within 0.2: {results['agreement_20pct']:.1%}")
    print(f"\nDistribution Statistics:")
    print(f"  BERT mean: {results['bert_mean']:.4f} (std={results['bert_std']:.4f})")
    print(f"  labMT mean: {results['labmt_mean']:.4f} (std={results['labmt_std']:.4f})")
    
    return results


def analyze_disagreements(df: pd.DataFrame) -> pd.DataFrame:
    """Identify tracks where methods disagree significantly."""
    print("\n" + "=" * 80)
    print("Disagreement Analysis")
    print("=" * 80)
    
    both_scores = df[df['happiness_from_lyrics'].notna() & df['h_lyrics_norm'].notna()].copy()
    
    if len(both_scores) == 0:
        print("   [WARN] No tracks with both scores available")
        return pd.DataFrame()
    
    # Calculate disagreement
    both_scores['disagreement'] = np.abs(
        both_scores['happiness_from_lyrics'] - both_scores['h_lyrics_norm']
    )
    
    # Identify high-disagreement tracks (>0.3 difference)
    high_disagreement = both_scores[both_scores['disagreement'] > 0.3].copy()
    high_disagreement = high_disagreement.sort_values('disagreement', ascending=False)
    
    print(f"\nTracks with high disagreement (>0.3): {len(high_disagreement)}")
    
    if len(high_disagreement) > 0:
        print("\nTop 10 tracks with highest disagreement:")
        display_cols = ['track_name', 'artists', 'happiness_from_lyrics', 'h_lyrics_norm', 
                       'disagreement', 'lyrics_language', 'track_genre']
        available_cols = [col for col in display_cols if col in high_disagreement.columns]
        
        top_disagreements = high_disagreement.head(10)[available_cols]
        for idx, row in top_disagreements.iterrows():
            track_name = str(row['track_name']).encode('ascii', 'replace').decode('ascii')
            artists = str(row['artists']).encode('ascii', 'replace').decode('ascii')
            lang = row.get('lyrics_language', 'N/A')
            genre = row.get('track_genre', 'N/A')
            print(f"  {track_name} - {artists}")
            print(f"    BERT: {row['happiness_from_lyrics']:.3f}, labMT: {row['h_lyrics_norm']:.3f}, "
                  f"diff: {row['disagreement']:.3f} | Lang: {lang}, Genre: {genre}")
    
    return high_disagreement


def analyze_by_language(df: pd.DataFrame) -> pd.DataFrame:
    """Compare methods by language."""
    print("\n" + "=" * 80)
    print("Language-Specific Analysis")
    print("=" * 80)
    
    both_scores = df[df['happiness_from_lyrics'].notna() & df['h_lyrics_norm'].notna()].copy()
    
    if len(both_scores) == 0 or 'lyrics_language' not in both_scores.columns:
        print("   [WARN] No language data available")
        return pd.DataFrame()
    
    # Filter to languages with at least 5 tracks
    lang_counts = both_scores['lyrics_language'].value_counts()
    valid_langs = lang_counts[lang_counts >= 5].index
    
    if len(valid_langs) == 0:
        print("   [WARN] No languages with sufficient data (>=5 tracks)")
        return pd.DataFrame()
    
    lang_results = []
    
    for lang in valid_langs:
        lang_data = both_scores[both_scores['lyrics_language'] == lang]
        bert = lang_data['happiness_from_lyrics']
        labmt = lang_data['h_lyrics_norm']
        
        if len(lang_data) < 3:
            continue
        
        pearson_r, pearson_p = pearsonr(bert, labmt)
        mad = np.mean(np.abs(bert - labmt))
        
        lang_results.append({
            'language': lang,
            'n_tracks': len(lang_data),
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'mean_abs_diff': mad,
            'bert_mean': bert.mean(),
            'labmt_mean': labmt.mean(),
        })
    
    lang_df = pd.DataFrame(lang_results).sort_values('n_tracks', ascending=False)
    
    print(f"\nLanguage-specific correlations (n >= 5):")
    print(f"{'Language':<12} {'N':<6} {'Pearson r':<12} {'p-value':<12} {'MAD':<8}")
    print("-" * 60)
    for _, row in lang_df.iterrows():
        print(f"{row['language']:<12} {row['n_tracks']:<6} {row['pearson_r']:>10.4f}  "
              f"{row['pearson_p']:>10.4f}  {row['mean_abs_diff']:>6.4f}")
    
    return lang_df


def create_visualizations(df: pd.DataFrame, output_dir: Path):
    """Create visualization plots."""
    print("\n" + "=" * 80)
    print("Creating Visualizations")
    print("=" * 80)
    
    both_scores = df[df['happiness_from_lyrics'].notna() & df['h_lyrics_norm'].notna()].copy()
    
    if len(both_scores) == 0:
        print("   [WARN] No tracks with both scores available")
        return
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    
    # 1. Scatter plot
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(both_scores['happiness_from_lyrics'], both_scores['h_lyrics_norm'], 
               alpha=0.5, s=20)
    ax.plot([0, 1], [0, 1], 'r--', label='Perfect agreement')
    ax.set_xlabel('BERT-based Happiness Score', fontsize=12)
    ax.set_ylabel('labMT-based Happiness Score', fontsize=12)
    ax.set_title('Comparison of Lyric Happiness Scoring Methods', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add correlation text
    pearson_r, _ = pearsonr(both_scores['happiness_from_lyrics'], both_scores['h_lyrics_norm'])
    ax.text(0.05, 0.95, f'Pearson r = {pearson_r:.3f}', 
            transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'happiness_methods_scatter.png', dpi=300, bbox_inches='tight')
    print(f"   [OK] Saved scatter plot to {output_dir / 'happiness_methods_scatter.png'}")
    plt.close()
    
    # 2. Distribution comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.hist(both_scores['happiness_from_lyrics'], bins=30, alpha=0.7, label='BERT', color='blue')
    ax1.set_xlabel('Happiness Score', fontsize=11)
    ax1.set_ylabel('Frequency', fontsize=11)
    ax1.set_title('BERT-based Distribution', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.hist(both_scores['h_lyrics_norm'], bins=30, alpha=0.7, label='labMT', color='green')
    ax2.set_xlabel('Happiness Score', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.set_title('labMT-based Distribution', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'happiness_methods_distributions.png', dpi=300, bbox_inches='tight')
    print(f"   [OK] Saved distribution plot to {output_dir / 'happiness_methods_distributions.png'}")
    plt.close()
    
    # 3. Difference distribution
    both_scores['difference'] = both_scores['happiness_from_lyrics'] - both_scores['h_lyrics_norm']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(both_scores['difference'], bins=40, alpha=0.7, color='purple', edgecolor='black')
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='No difference')
    ax.set_xlabel('Difference (BERT - labMT)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Score Differences', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'happiness_methods_difference.png', dpi=300, bbox_inches='tight')
    print(f"   [OK] Saved difference plot to {output_dir / 'happiness_methods_difference.png'}")
    plt.close()


def generate_report(df: pd.DataFrame, correlations: dict, disagreements: pd.DataFrame, 
                    lang_analysis: pd.DataFrame, output_dir: Path):
    """Generate a comprehensive text report."""
    report_path = output_dir / 'lyric_methods_comparison_report.txt'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("Comparative Analysis: BERT vs labMT Lyric Happiness Scoring\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("SUMMARY\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total tracks analyzed: {len(df)}\n")
        f.write(f"Tracks with both scores: {correlations.get('n_tracks', 0)}\n")
        f.write(f"Tracks with only BERT score: {(df['happiness_from_lyrics'].notna() & df['h_lyrics_norm'].isna()).sum()}\n")
        f.write(f"Tracks with only labMT score: {(df['happiness_from_lyrics'].isna() & df['h_lyrics_norm'].notna()).sum()}\n\n")
        
        f.write("CORRELATION ANALYSIS\n")
        f.write("-" * 80 + "\n")
        if correlations:
            f.write(f"Pearson correlation: r = {correlations['pearson_r']:.4f} (p = {correlations['pearson_p']:.4f})\n")
            f.write(f"Spearman correlation: rho = {correlations['spearman_r']:.4f} (p = {correlations['spearman_p']:.4f})\n")
            f.write(f"Mean absolute difference: {correlations['mean_abs_diff']:.4f}\n")
            f.write(f"Root mean square difference: {correlations['rmsd']:.4f}\n")
            f.write(f"Agreement within 0.1: {correlations['agreement_10pct']:.1%}\n")
            f.write(f"Agreement within 0.2: {correlations['agreement_20pct']:.1%}\n\n")
        
        f.write("LANGUAGE-SPECIFIC ANALYSIS\n")
        f.write("-" * 80 + "\n")
        if not lang_analysis.empty:
            for _, row in lang_analysis.iterrows():
                f.write(f"{row['language']}: n={row['n_tracks']}, r={row['pearson_r']:.3f}, "
                       f"MAD={row['mean_abs_diff']:.3f}\n")
        f.write("\n")
        
        f.write("HIGH DISAGREEMENT TRACKS (>0.3 difference)\n")
        f.write("-" * 80 + "\n")
        if not disagreements.empty:
            f.write(f"Found {len(disagreements)} tracks with high disagreement\n\n")
            for idx, row in disagreements.head(20).iterrows():
                track_name = str(row['track_name']).encode('ascii', 'replace').decode('ascii')
                artists = str(row['artists']).encode('ascii', 'replace').decode('ascii')
                f.write(f"{track_name} - {artists}\n")
                f.write(f"  BERT: {row['happiness_from_lyrics']:.3f}, labMT: {row['h_lyrics_norm']:.3f}, "
                       f"diff: {row['disagreement']:.3f}\n")
                if 'lyrics_language' in row:
                    f.write(f"  Language: {row['lyrics_language']}\n")
                f.write("\n")
        else:
            f.write("No high-disagreement tracks found.\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("METHODOLOGICAL NOTES\n")
        f.write("=" * 80 + "\n\n")
        f.write("BERT-based approach (happiness_from_lyrics):\n")
        f.write("- Context-aware neural network model\n")
        f.write("- Trained on multilingual review/rating data\n")
        f.write("- May capture complex linguistic structures better\n")
        f.write("- Potential biases from training data\n\n")
        f.write("labMT-based approach (h_lyrics_norm):\n")
        f.write("- Human-validated word-level happiness scores\n")
        f.write("- Interpretable and transparent\n")
        f.write("- Language-specific dictionaries\n")
        f.write("- May miss contextual nuances\n\n")
    
    print(f"   [OK] Saved report to {report_path}")


def main():
    """Main analysis pipeline."""
    print("=" * 80)
    print("Comparative Analysis of Lyric Happiness Scoring Methods")
    print("=" * 80)
    
    # Load and merge data
    df = load_and_merge_data()
    
    # Calculate correlations
    correlations = calculate_correlations(df)
    
    # Analyze disagreements
    disagreements = analyze_disagreements(df)
    
    # Language-specific analysis
    lang_analysis = analyze_by_language(df)
    
    # Create visualizations
    create_visualizations(df, OUTPUT_DIR)
    
    # Generate report
    generate_report(df, correlations, disagreements, lang_analysis, OUTPUT_DIR)
    
    # Save merged dataset with both scores
    output_path = OUTPUT_DIR / 'dataset_with_both_scores.csv'
    df.to_csv(output_path, index=False)
    print(f"\n   [OK] Saved merged dataset to {output_path}")
    
    print("\n" + "=" * 80)
    print("Analysis Complete!")
    print("=" * 80)
    print(f"\nOutput files saved to: {OUTPUT_DIR}")
    print("  - happiness_methods_scatter.png")
    print("  - happiness_methods_distributions.png")
    print("  - happiness_methods_difference.png")
    print("  - lyric_methods_comparison_report.txt")
    print("  - dataset_with_both_scores.csv")


if __name__ == "__main__":
    main()

