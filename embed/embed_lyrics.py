import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import argparse
import os
import re
from typing import Optional

class LyricsEmbedder:
    def __init__(self, model_name='all-MiniLM-L6-v2', batch_size=32):
        """
        Initializes the embedding model.
        
        Args:
            model_name (str): The name of the SentenceTransformer model.
            batch_size (int): Number of rows to process at once (affects RAM/Speed).
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = 'cuda' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu'
        
        print(f"Loading model: {model_name} on {self.device}...")
        self.model = SentenceTransformer(model_name, device=self.device)

    @staticmethod
    def clean_lyrics_text(text: Optional[str]) -> str:
        """
        Clean lyrics text for embedding.

        - Remove section tags like [Intro], [Chorus], etc.
        - Normalize newlines / carriage returns to spaces
        - Remove common artifact tokens like literal "\\n", "\\r", "/r"
        - Collapse whitespace
        """
        if not isinstance(text, str):
            return ""

        t = text

        # Normalize control characters and common literal artifacts
        t = t.replace("\r", " ").replace("\n", " ")
        t = t.replace("\\r", " ").replace("\\n", " ")
        t = t.replace("/r", " ")

        # Remove bracketed section labels (e.g., [Intro], [Verse 1], [Chorus])
        t = re.sub(r"\[[^\]]+\]", " ", t)

        # Remove some common non-lyric artifacts seen in scraped sources
        t = re.sub(r"\bYou might also like\b", " ", t, flags=re.IGNORECASE)
        t = re.sub(r"\bEmbed\b", " ", t, flags=re.IGNORECASE)

        # Collapse whitespace
        t = re.sub(r"\s+", " ", t).strip()

        return t

    def process_csv(self, input_path, output_path, column_name):
        """
        Loads CSV, embeds the specific column, and saves the result.
        """
        # 1. Validation
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"The input file '{input_path}' was not found.")

        # 2. Load Data
        print(f"Reading {input_path}...")
        try:
            df = pd.read_csv(input_path)
        except Exception as e:
            raise ValueError(f"Failed to read CSV. Error: {e}")

        if column_name not in df.columns:
            raise ValueError(f"Column '{column_name}' not found. Available columns: {list(df.columns)}")

        # 3. Pre-processing (Basic + lyric cleaning)
        # Handle NaN/Empty values to prevent crashes
        print(f"Cleaning data in column '{column_name}'...")
        original_count = len(df)
        df = df.dropna(subset=[column_name])  # Drop rows with no lyrics
        df = df[df[column_name].astype(str).str.strip() != ""]  # Drop rows with empty strings

        # Create a cleaned column for embedding (keeps original text intact)
        cleaned_col = f"{column_name}_clean"
        print(f"Applying lyric cleaning -> '{cleaned_col}'...")
        df[cleaned_col] = df[column_name].apply(self.clean_lyrics_text)
        df = df[df[cleaned_col].str.strip() != ""]
        
        if len(df) < original_count:
            print(f"Dropped {original_count - len(df)} rows due to missing/empty data.")

        # 4. Embedding
        print(f"Embedding {len(df)} rows (Batch Size: {self.batch_size})...")
        
        # We use the model's encode method which handles batching internally, 
        # but we wrap it to show progress if needed or handle complex logic.
        lyrics_list = df[cleaned_col].tolist()
        
        embeddings = self.model.encode(
            lyrics_list, 
            batch_size=self.batch_size, 
            show_progress_bar=True,
            convert_to_numpy=True
        )

        # 5. Storing Results
        # There are two common ways to store embeddings in a CSV:
        # A. As a single column of stringified lists (easier for simple viewing)
        # B. As separate columns (dim_0, dim_1...) (better for SQL/CSV strict formats)
        
        # Here we verify the user preference or default to a single column for portability
        df['embedding'] = list(embeddings)
        df['embedding_model'] = self.model_name
        df['embedding_source_column'] = column_name
        df['embedding_cleaned_column'] = cleaned_col

        # 6. Save
        print(f"Saving results to {output_path}...")
        # We recommend pickle for DataFrames with arrays/lists to preserve dtypes, 
        # but CSV is requested.
        if output_path.endswith('.pkl'):
            df.to_pickle(output_path)
        else:
            df.to_csv(output_path, index=False)
        
        print("Done.")
        return df

# --- Command Line Interface ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embed a specific column of a CSV file using Sentence Transformers.")
    
    # Required Arguments
    parser.add_argument('input_file', type=str, help="Path to the input CSV file.")
    parser.add_argument('column_name', type=str, help="Name of the column containing the text/lyrics.")
    
    # Optional Arguments
    parser.add_argument('--output', type=str, default='output_with_embeddings.csv', help="Path for the output file.")
    parser.add_argument('--model', type=str, default='all-MiniLM-L6-v2', help="HuggingFace model name.")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for embedding.")

    args = parser.parse_args()

    embedder = LyricsEmbedder(model_name=args.model, batch_size=args.batch_size)
    embedder.process_csv(args.input_file, args.output, args.column_name)