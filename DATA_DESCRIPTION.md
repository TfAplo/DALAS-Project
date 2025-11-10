# Data Description and State Analysis

## 1. Relevance of Data to the Problem

### Data Sources

The dataset comprises multiple sources collected for analyzing music chart patterns, acoustic features, and geographic clustering:

#### **Primary Chart Data Sources:**

1. **Billboard Charts** (`songs_latest.csv`)
   - Source: Billboard.com via `billboard.py` library
   - Volume: 100 rows
   - Dimensionality: 10 columns (source, domain, chart, region, date, position, title, artist, url, scraped_at)
   - Content: Billboard Hot 100 chart data from November 8, 2025
   - Coverage: 98 unique songs, 63 unique artists

2. **TurnTable Charts** (`top_songs.csv`)
   - Source: turntablecharts.com (Nigerian music charts)
   - Volume: 30 rows
   - Dimensionality: 7 columns (domain, chart, url, position, title, artist, scraped_at)
   - Content: Top 100 songs and albums from Nigerian charts
   - Coverage: 19 unique songs, 12 unique artists

3. **Chart Metadata** (`chart_official_sites.csv`, `record_charts.csv`)
   - Source: Wikipedia scraping
   - Volume: 118 chart entries, 131 regional chart mappings
   - Dimensionality: 3-4 columns per file
   - Content: Official chart websites and regional chart classifications
   - Coverage: Global coverage across 6 continents

#### **Research Paper Dataset:**

4. **Top 25 Tracks by City** (`data/research_paper/Top25 Tracks/20220119/`)
   - Source: Research paper "Music Charts for Approximating Everyday Emotions: A Dataset of Daily Charts with Music Features from 106 Cities" (2022)
   - Volume: ~433,350 rows total (107 city files × ~4,050 rows per city)
   - Dimensionality: 5 columns per file (DATE, ID, SONG, ARTIST, ALBUM)
   - Time Period: July 28, 2021 to January 19, 2022
   - Coverage: 107 cities across multiple continents
   - Geographic Distribution:
     - North America: 34 cities
     - Europe: 26 cities
     - East Asia: 14 cities
     - South America: 8 cities
     - Africa: 6 cities
     - Southeast Asia: 4 cities
     - South Asia: 4 cities
     - Central America: 3 cities
     - Australia/New Zealand: 3 cities
     - Western Asia: 3 cities
     - Central Asia: 1 city

5. **Spotify Acoustic Features** (`data/research_paper/Spotify Acoustic Features of Tracks/`)
   - Source: Spotify API acoustic feature extraction
   - Volume: ~14,976 rows total (104 city files × ~144 rows per city)
   - Dimensionality: 6 columns (Unnamed: 0, danceability, energy, loundiness, valence, tempo)
   - Features: Audio characteristics for tracks from each city
   - Coverage: 104 cities with acoustic feature data

6. **Average Daily Acoustic Features** (`data/research_paper/Average daily of all cities/`)
   - Source: Aggregated daily averages from research paper
   - Volume: 10 files (one per feature type)
   - Dimensionality: 103 cities × ~176 daily observations
   - Features: 
     - Raw features: danceability, energy, loudness, tempo, valence
     - Normalized features: normalization_danceability, normalization_energy, normalization_loudness, normalization_tempo, normalization_valence
   - Time Period: July 28, 2021 to January 19, 2022 (176 days)

#### **Geographic and Clustering Data:**

7. **Map Clusters** (`map_clusters_extracted.csv`)
   - Source: Extracted from research paper visualization
   - Volume: 133 rows
   - Dimensionality: 5 columns (x, y, cluster, lon, lat)
   - Content: Geographic coordinates and cluster assignments for cities
   - Clusters: 3 main clusters identified in the research paper

8. **Extracted Tables from PDF** (`data/extracted_tables/`)
   - Source: Tables extracted from research paper PDF
   - Volume: 5 tables
   - Content: Regional distributions, feature descriptions, cluster assignments

### Total Data Volume

- **CSV files**: 512 rows across 5 files
- **Research paper track data**: ~433,350 rows (107 cities)
- **Acoustic features**: ~14,976 rows (104 cities)
- **Daily aggregated features**: ~1,030 rows × 176 days = ~181,280 data points
- **Total estimated records**: ~630,000+ data points

### Dimensionality Summary

- **Chart data**: 10-7 dimensions per record
- **Track data**: 5 dimensions (date, ID, song, artist, album)
- **Acoustic features**: 6 dimensions (5 numeric features + track name)
- **Geographic data**: 5 dimensions (coordinates + cluster)
- **Temporal dimension**: 176 days of daily observations
- **Geographic dimension**: 106-107 cities globally

---

## 2. State of the Data

### Missing Values

#### **CSV Files:**

1. **`songs_latest.csv`**:
   - `region`: 100% missing (100/100 rows) - Billboard Hot 100 doesn't specify region
   - `url`: 100% missing (100/100 rows) - URLs not captured in Billboard scraping
   - **Impact**: Low - region is expected to be null for US-focused charts, URLs are optional

2. **`top_songs.csv`**:
   - No missing values
   - **Status**: Complete

3. **`chart_official_sites.csv`**:
   - `official_site`: 72.9% missing (86/118 rows) - Many charts don't have official websites listed
   - **Impact**: Medium - Missing official site URLs limit direct data access

4. **`record_charts.csv`**:
   - `url`: 3.1% missing (4/131 rows) - Minor missing Wikipedia URLs
   - **Impact**: Low - Only 4 records affected

5. **`map_clusters_extracted.csv`**:
   - No missing values
   - **Status**: Complete

#### **Research Paper Data:**

6. **Top 25 Tracks Files**:
   - No missing values across all 107 city files
   - **Status**: Complete
   - **Note**: DATE format includes position indicator (e.g., "2021-07-28 : 1") which may require parsing

7. **Spotify Acoustic Features**:
   - **Missing values per city file**: ~14.6% (21/144 rows on average)
   - Affected features: danceability, energy, loundiness, valence, tempo
   - **Pattern**: Missing values appear to be consistent across all features for the same tracks
   - **Possible causes**: 
     - Tracks not found in Spotify database
     - API errors during feature extraction
     - Invalid track IDs
   - **Impact**: Medium - Approximately 1 in 7 tracks missing acoustic features

8. **Average Daily Data**:
   - No missing values in the aggregated files
   - **Status**: Complete

### Outliers

#### **Numeric Feature Analysis (Spotify Acoustic Features):**

Based on sample analysis of acoustic features:

1. **Danceability**:
   - Range: 0.183 to 0.956 (expected: 0.0 to 1.0)
   - Mean: 0.699, Std: 0.149
   - **Status**: Normal distribution, no extreme outliers

2. **Energy**:
   - Range: 0.018 to 0.944 (expected: 0.0 to 1.0)
   - Mean: 0.623, Std: 0.165
   - **Status**: Normal distribution, no extreme outliers

3. **Loudness** (note: column name is "loundiness" - likely typo):
   - Range: -28.266 to -1.992 dB (expected: typically -60 to 0 dB)
   - Mean: -7.113, Std: 3.084
   - **Potential outlier**: -28.266 dB is unusually quiet but within physical limits
   - **Status**: Generally normal, one potential outlier

4. **Valence**:
   - Range: 0.0719 to 0.7070 (expected: 0.0 to 1.0)
   - Mean: Not calculated in sample, but appears normal
   - **Status**: Normal distribution

5. **Tempo**:
   - Range: 79.461 to 159.933 BPM (expected: typically 60-200 BPM)
   - **Status**: Normal range, no outliers

#### **Geographic Data:**

- **Coordinates**: All longitude/latitude values appear within valid ranges
- **Cluster assignments**: 3 clusters identified (cluster_1_grey, cluster_2_green, cluster_3)
- **Status**: No geographic outliers detected

### Noise and Data Quality Issues

1. **Column Name Typo**: 
   - "loundiness" instead of "loudness" in acoustic features files
   - **Impact**: Low - cosmetic issue, but should be standardized

2. **DATE Format Inconsistency**:
   - Top 25 Tracks files use format "YYYY-MM-DD : position" which combines date and position
   - **Impact**: Medium - Requires parsing to separate date and position

3. **Inconsistent Naming**:
   - Some city files have non-ASCII characters (e.g., Chinese, Japanese characters)
   - **Impact**: Low - May cause issues with file system compatibility but data is readable

4. **Missing Acoustic Features**:
   - ~14.6% of tracks missing all acoustic features
   - **Impact**: Medium - May require imputation or exclusion in analysis

5. **URL Completeness**:
   - Many chart official sites missing URLs
   - **Impact**: Low - URLs are supplementary metadata

### Data Merging Strategy

The data comes from multiple sources that have been collected but **not yet fully merged**. Here's how they could be integrated:

#### **Potential Merge Keys:**

1. **Song Title + Artist**:
   - Primary key for merging chart data with acoustic features
   - **Challenge**: Name variations, featuring artists, special characters
   - **Status**: Not yet merged

2. **City Name**:
   - Key for merging geographic data with city-specific track data
   - **Challenge**: Name variations (e.g., "new_york_city" vs "New York")
   - **Status**: Not yet merged

3. **Date**:
   - Key for temporal alignment across datasets
   - **Challenge**: Different date formats and time zones
   - **Status**: Not yet merged

4. **Track ID**:
   - Research paper includes ID field that may link to Spotify
   - **Status**: Not yet verified or merged

#### **Current Data Organization:**

- **Separate files**: Data is currently stored in separate files by source and type
- **No unified schema**: Each source has its own schema
- **No cross-references**: No explicit foreign keys or relationships defined
- **Geographic clustering**: Map clusters extracted separately, not linked to track data

#### **Recommended Merging Approach:**

1. **Normalize city names**: Create a mapping table for city name variations
2. **Fuzzy matching**: Use fuzzy string matching for song/artist names
3. **Temporal alignment**: Standardize date formats and align by date
4. **Feature enrichment**: Merge acoustic features to chart data by track identification
5. **Geographic enrichment**: Link city data to cluster assignments

### Data Completeness Summary

| Dataset | Completeness | Missing Data | Notes |
|---------|--------------|--------------|-------|
| Billboard charts | 80% | URLs, regions | Expected for US chart |
| TurnTable charts | 100% | None | Complete |
| Chart metadata | 73% | Official site URLs | Many charts lack official sites |
| Top 25 Tracks | 100% | None | Complete, 107 cities |
| Acoustic features | 85.4% | ~14.6% tracks | Missing features for some tracks |
| Daily averages | 100% | None | Complete aggregation |
| Geographic data | 100% | None | Complete coordinates |

### Recommendations for Data Cleaning

1. **Standardize column names**: Fix "loundiness" → "loudness"
2. **Parse DATE fields**: Separate date and position in Top 25 Tracks
3. **Handle missing acoustic features**: 
   - Investigate why features are missing
   - Consider imputation or exclusion strategies
4. **Create merge keys**: Develop standardized identifiers for songs, artists, and cities
5. **Validate geographic data**: Ensure all coordinates are within valid ranges
6. **Normalize city names**: Create canonical city name mapping

---

## 3. Possible Biases in the Data

The dataset contains several potential biases that should be acknowledged and considered when interpreting results:

### Geographic Bias

1. **Regional Overrepresentation**:
   - **North America**: 34 cities (31.8% of total) - heavily overrepresented relative to population
   - **Europe**: 26 cities (24.3% of total) - overrepresented
   - **Africa**: Only 6 cities (5.6% of total) - severely underrepresented despite having ~17% of global population
   - **South Asia**: Only 4 cities (3.7% of total) - underrepresented despite large population centers
   - **Impact**: Results may not generalize to underrepresented regions, particularly developing countries and rural areas

2. **Urban Bias**:
   - All data points are from major metropolitan areas
   - No representation of rural or smaller urban areas
   - **Impact**: Findings may not reflect music preferences in non-urban settings

3. **Economic Development Bias**:
   - Cities selected likely prioritize economically developed regions with robust music industries
   - **Impact**: May skew toward commercial music markets, missing local/indigenous music scenes

### Temporal Bias

1. **Limited Time Period**:
   - Research paper data: July 28, 2021 to January 19, 2022 (176 days)
   - Billboard data: Single snapshot from November 8, 2025
   - **Impact**: 
     - May not capture seasonal variations in music preferences
     - COVID-19 pandemic effects may still influence 2021-2022 data
     - Single Billboard snapshot cannot show temporal trends

2. **Historical Context**:
   - Data collected during/post-pandemic period may reflect unusual listening patterns
   - **Impact**: May not represent "normal" music consumption patterns

### Platform and Data Source Bias

1. **Streaming Platform Bias**:
   - **Spotify**: Dominates acoustic features data, but has specific user demographics
     - Spotify users tend to be younger, more tech-savvy, and from higher-income backgrounds
     - Spotify availability varies by region (not available in all countries)
   - **Billboard**: Reflects US market primarily, based on sales, radio play, and streaming
   - **Apple Music**: Different user base than Spotify (often older, iOS users)
   - **Impact**: Platform-specific user demographics may not represent general population preferences

2. **Chart Methodology Bias**:
   - Charts favor mainstream, commercially successful music
   - Independent, local, or niche genres may be underrepresented
   - **Impact**: May miss significant portions of local music culture

3. **Missing Acoustic Features Bias**:
   - ~14.6% of tracks missing acoustic features
   - May systematically exclude:
     - Older tracks not in Spotify database
     - Local/regional artists not on Spotify
     - Tracks with metadata issues
   - **Impact**: Missing features may correlate with certain genres, languages, or regions

### Cultural and Linguistic Bias

1. **English Language Dominance**:
   - Billboard charts are primarily English-language
   - Research paper dataset may favor English-language tracks due to Spotify's catalog
   - **Impact**: Non-English music may be underrepresented despite local popularity

2. **Western Music Industry Bias**:
   - Major labels and Western music industry structures dominate charts
   - Local music industries may be underrepresented
   - **Impact**: May not capture authentic local music preferences

3. **Genre Bias**:
   - Charts typically favor pop, hip-hop, and commercial genres
   - Classical, jazz, traditional/folk, and niche genres underrepresented
   - **Impact**: May not reflect full spectrum of musical preferences

### Selection and Sampling Bias

1. **City Selection Bias**:
   - Cities chosen by researchers may not be representative
   - Selection criteria not explicitly documented
   - **Impact**: May favor cities with better data availability or research interest

2. **Top 25 Limitation**:
   - Only top 25 tracks per city analyzed
   - **Impact**: 
     - Misses long-tail preferences
     - May overemphasize mainstream hits
     - Local favorites outside top 25 excluded

3. **Chart Availability Bias**:
   - Some regions have more comprehensive chart data than others
   - **Impact**: Better-documented regions may appear more prominent

### Demographic Bias

1. **Age Bias**:
   - Streaming platforms skew toward younger demographics
   - Older generations may prefer different platforms or consumption methods
   - **Impact**: May not represent preferences of older demographics

2. **Socioeconomic Bias**:
   - Streaming services require internet access and subscription fees
   - **Impact**: May underrepresent lower-income populations

3. **Gender Bias**:
   - Music industry historically male-dominated
   - Chart data may reflect industry biases rather than listener preferences
   - **Impact**: Female artists may be underrepresented

### Data Collection Bias

1. **Scraping Limitations**:
   - Billboard scraping may miss certain chart types or historical data
   - Rate limiting may affect data completeness
   - **Impact**: Incomplete coverage of available data

2. **API Restrictions**:
   - Spotify API limitations may affect feature extraction
   - Some tracks may be region-restricted or unavailable
   - **Impact**: Systematic exclusion of certain tracks

3. **Wikipedia Source Bias**:
   - Chart metadata from Wikipedia may reflect English-language bias
   - Some regional charts may be less documented
   - **Impact**: Incomplete or biased chart metadata

### Commercial and Industry Bias

1. **Mainstream Music Bias**:
   - Charts favor commercially successful, heavily promoted music
   - Independent artists may be underrepresented
   - **Impact**: May not reflect organic music discovery or local preferences

2. **Label Promotion Bias**:
   - Major label promotion affects chart positions
   - **Impact**: Chart positions may reflect marketing budgets rather than pure listener preference

3. **Radio/Streaming Algorithm Bias**:
   - Chart positions influenced by radio playlists and streaming algorithms
   - **Impact**: May create feedback loops that reinforce certain music

### Measurement Bias

1. **Acoustic Feature Limitations**:
   - Spotify's acoustic features may not capture all musical dimensions
   - Features may be biased toward Western music theory concepts
   - **Impact**: May not adequately represent non-Western musical traditions

2. **Normalization Bias**:
   - Normalized features may mask important variations
   - **Impact**: May obscure meaningful differences between regions

### Recommendations for Addressing Biases

1. **Acknowledge Limitations**: Explicitly state geographic, temporal, and platform limitations in any analysis
2. **Regional Analysis**: Conduct separate analyses by region to avoid overgeneralization
3. **Sensitivity Analysis**: Test how results change when excluding certain regions or time periods
4. **Complementary Data**: Consider supplementing with local music surveys or alternative data sources
5. **Bias Documentation**: Document known biases in methodology sections
6. **Qualitative Validation**: Use qualitative methods to validate findings in underrepresented regions
7. **Missing Data Analysis**: Investigate whether missing acoustic features correlate with specific characteristics
8. **Temporal Expansion**: Collect data across multiple years to reduce temporal bias
9. **Diverse Sources**: Incorporate data from multiple platforms and sources to reduce platform-specific bias

---

## Summary

The dataset is **comprehensive and well-structured** with minimal critical issues. The primary concerns are:
- Missing acoustic features for ~14.6% of tracks (likely due to Spotify API limitations)
- Missing metadata (URLs, official sites) which are supplementary
- Need for data integration/merging across sources

The data volume is substantial (~630,000+ records) with good geographic and temporal coverage, making it suitable for analysis of music preferences across cities and time periods.

