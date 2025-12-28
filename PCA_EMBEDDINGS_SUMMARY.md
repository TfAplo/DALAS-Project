# PCA Embeddings Summary

## Overview

This document summarizes the Principal Component Analysis (PCA) applied to song lyric embeddings in the final dataset.

## Embedding Generation

- **Model**: `all-MiniLM-L6-v2` (SentenceTransformers)
- **Original Embedding Dimensions**: 384
- **Total Songs**: 1,550
- **Embedding Method**: Lyrics were cleaned (removed section tags, normalized whitespace) and then embedded using the SentenceTransformer model

## PCA Results

- **Original Dimensions**: 384
- **Components Kept**: 213
- **Variance Explained**: 95.00%
- **Reduction**: From 384 dimensions to 213 dimensions (44.5% reduction)

## Why 213 Components?

The number of components was automatically selected to explain 95% of the variance in the embedding space. This threshold was chosen because:

1. **Information Preservation**: 95% variance retention ensures that the vast majority of semantic information in the embeddings is preserved
2. **Dimensionality Reduction**: Reducing from 384 to 213 dimensions (44.5% reduction) significantly reduces computational complexity while maintaining most of the information
3. **Balance**: This provides a good balance between information retention and dimensionality reduction for downstream tasks

## Component Distribution

The first few principal components explain the most variance:

- **PC1**: 9.25% of variance
- **PC2**: 5.39% of variance
- **PC3**: 3.98% of variance
- **PC4**: 3.52% of variance
- **PC5**: 2.73% of variance

The variance explained per component decreases gradually, which is typical for PCA on high-dimensional embeddings.

## Usage in Dataset

The PCA components are stored in the dataset as columns:

- `embedding_pc_1` through `embedding_pc_213`
- These can be used directly in machine learning pipelines without needing to store the full 384-dimensional embeddings
- The original `embedding` column (full 384-dim vectors) is also preserved in the dataset

## Notes

- The PCA was fit on all 1,550 songs to ensure consistent transformation
- The same PCA transformation should be applied to any new songs added in the future
- The pipeline normalizes features separately, so these PCA components are not normalized (as requested)
