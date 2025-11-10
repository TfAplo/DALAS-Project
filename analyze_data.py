"""Analyze all collected data for the research project."""
import pandas as pd
import os
from pathlib import Path

print("=" * 80)
print("DATA COLLECTION ANALYSIS")
print("=" * 80)

# 1. CSV Files Analysis
print("\n1. CSV FILES")
print("-" * 80)

csv_files = {
    'songs_latest.csv': 'data/songs_latest.csv',
    'top_songs.csv': 'data/top_songs.csv',
    'chart_official_sites.csv': 'data/chart_official_sites.csv',
    'record_charts.csv': 'data/record_charts.csv',
    'map_clusters_extracted.csv': 'data/map_clusters_extracted.csv'
}

for name, path in csv_files.items():
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(f"\n{name}:")
        print(f"  Rows: {len(df):,}")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Missing values:")
        missing = df.isnull().sum()
        for col, count in missing.items():
            if count > 0:
                print(f"    {col}: {count} ({count/len(df)*100:.1f}%)")
        if missing.sum() == 0:
            print("    None")
        print(f"  Duplicates: {df.duplicated().sum()}")
        if name in ['songs_latest.csv', 'top_songs.csv']:
            print(f"  Unique songs: {df['title'].nunique() if 'title' in df.columns else 'N/A'}")
            print(f"  Unique artists: {df['artist'].nunique() if 'artist' in df.columns else 'N/A'}")
            if 'source' in df.columns:
                print(f"  Sources: {df['source'].value_counts().to_dict()}")
            if 'region' in df.columns:
                print(f"  Regions: {df['region'].nunique()} unique regions")

# 2. Research Paper Data - Top 25 Tracks
print("\n\n2. RESEARCH PAPER DATA - TOP 25 TRACKS")
print("-" * 80)
top25_path = Path('data/research_paper/Top25 Tracks/20220119')
if top25_path.exists():
    city_files = [f for f in os.listdir(top25_path) if f.endswith('.xlsx')]
    print(f"Total city files: {len(city_files)}")
    if city_files:
        sample = pd.read_excel(top25_path / city_files[0])
        print(f"Sample file ({city_files[0]}):")
        print(f"  Rows: {len(sample):,}")
        print(f"  Columns: {list(sample.columns)}")
        print(f"  Missing values: {sample.isnull().sum().sum()}")
        print(f"  Date range: {sample['DATE'].min() if 'DATE' in sample.columns else 'N/A'} to {sample['DATE'].max() if 'DATE' in sample.columns else 'N/A'}")
        
        # Check all files for consistency
        total_rows = 0
        for f in city_files[:5]:  # Sample first 5
            try:
                df = pd.read_excel(top25_path / f)
                total_rows += len(df)
            except:
                pass
        print(f"  Estimated total rows (5 sample files): {total_rows:,}")
        print(f"  Estimated total rows (all files): ~{len(sample) * len(city_files):,}")

# 3. Research Paper Data - Spotify Acoustic Features
print("\n\n3. RESEARCH PAPER DATA - SPOTIFY ACOUSTIC FEATURES")
print("-" * 80)
features_path = Path('data/research_paper/Spotify Acoustic Features of Tracks/Spotify Acoustic Features of Tracks')
if features_path.exists():
    feature_files = [f for f in os.listdir(features_path) if f.endswith('.xlsx')]
    print(f"Total city files: {len(feature_files)}")
    if feature_files:
        sample = pd.read_excel(features_path / feature_files[0])
        print(f"Sample file ({feature_files[0]}):")
        print(f"  Rows: {len(sample):,}")
        print(f"  Columns: {list(sample.columns)}")
        print(f"  Missing values:")
        missing = sample.isnull().sum()
        for col, count in missing.items():
            if count > 0:
                print(f"    {col}: {count} ({count/len(sample)*100:.1f}%)")
        
        # Check numeric columns for outliers
        numeric_cols = sample.select_dtypes(include=['float64', 'int64']).columns
        if len(numeric_cols) > 0:
            print(f"\n  Numeric feature statistics (sample):")
            for col in numeric_cols[:3]:  # First 3 numeric columns
                if col != 'Unnamed: 0':
                    stats = sample[col].describe()
                    print(f"    {col}:")
                    print(f"      Mean: {stats['mean']:.3f}, Std: {stats['std']:.3f}")
                    print(f"      Min: {stats['min']:.3f}, Max: {stats['max']:.3f}")

# 4. Average Daily Data
print("\n\n4. AVERAGE DAILY DATA")
print("-" * 80)
avg_path = Path('data/research_paper/Average daily of all cities/Average daily of all cities')
if avg_path.exists():
    avg_files = [f for f in os.listdir(avg_path) if f.endswith('.xlsx')]
    print(f"Total files: {len(avg_files)}")
    print(f"Files: {avg_files}")
    if avg_files:
        sample = pd.read_excel(avg_path / avg_files[0])
        print(f"Sample file ({avg_files[0]}):")
        print(f"  Rows: {len(sample):,}")
        print(f"  Columns: {list(sample.columns)}")
        print(f"  Missing values: {sample.isnull().sum().sum()}")

# 5. Extracted Tables
print("\n\n5. EXTRACTED TABLES FROM PDF")
print("-" * 80)
extracted_path = Path('data/extracted_tables')
if extracted_path.exists():
    table_files = [f for f in os.listdir(extracted_path) if f.endswith('.csv')]
    print(f"Total table files: {len(table_files)}")
    for f in table_files:
        df = pd.read_csv(extracted_path / f)
        print(f"\n  {f}:")
        print(f"    Rows: {len(df):,}")
        print(f"    Columns: {list(df.columns)}")
        print(f"    Missing values: {df.isnull().sum().sum()}")

# 6. Summary Statistics
print("\n\n6. DATA SUMMARY")
print("-" * 80)
print("Data Sources:")
print("  1. Billboard charts (via billboard.py)")
print("  2. Apple Music RSS feeds")
print("  3. Spotify Charts CSV")
print("  4. Research paper dataset (106 cities)")
print("  5. Chart metadata from Wikipedia")
print("  6. Geographic clustering data")

total_csv_rows = sum(len(pd.read_csv(path)) for path in csv_files.values() if os.path.exists(path))
print(f"\nTotal CSV rows: {total_csv_rows:,}")

print("\n" + "=" * 80)

