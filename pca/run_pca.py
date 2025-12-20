import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import argparse
import ast  # Needed to convert string representations of lists back to lists
import os

class PCAAnalyzer:
    def __init__(self, n_components=2):
        """
        Initializes the PCA model.
        
        Args:
            n_components (int): Number of dimensions to reduce to (usually 2 or 3).
        """
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        self.fitted = False

    def load_and_parse_csv(self, file_path, embedding_col):
        """
        Loads CSV and converts the embedding column from strings back to numpy arrays.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Input file not found: {file_path}")
            
        print(f"Loading {file_path}...")
        if file_path.endswith('.pkl'):
            df = pd.read_pickle(file_path)
        else:
            df = pd.read_csv(file_path)
        
        if embedding_col not in df.columns:
            raise ValueError(f"Column '{embedding_col}' not found.")

        # If embeddings are stringified (CSV), parse them back into lists.
        if isinstance(df[embedding_col].iloc[0], str):
            print(f"Parsing stringified embeddings in '{embedding_col}'...")
            df[embedding_col] = df[embedding_col].apply(ast.literal_eval)
        
        # Convert column of lists/arrays into a 2D Numpy array for Sklearn
        feature_matrix = np.vstack(df[embedding_col].values)
        
        return df, feature_matrix

    def run_analysis(self, feature_matrix):
        """
        Fits PCA and transforms the data.
        """
        print(f"Running PCA to reduce to {self.n_components} dimensions...")
        projected_data = self.pca.fit_transform(feature_matrix)
        self.fitted = True
        
        # Calculate how much information was preserved
        variance_ratio = self.pca.explained_variance_ratio_
        total_variance = sum(variance_ratio) * 100
        print(f"Explained Variance Ratio: {variance_ratio}")
        print(f"Total Information Preserved: {total_variance:.2f}%")
        
        return projected_data

    def plot_2d(self, df, x_col, y_col, label_col=None, output_image='pca_plot.png'):
        """
        Generates a 2D scatter plot of the results.
        """
        if self.n_components != 2:
            print("Skipping 2D plot (n_components is not 2).")
            return

        plt.figure(figsize=(12, 8))
        plt.scatter(df[x_col], df[y_col], alpha=0.6, c='teal', edgecolors='k', s=60)
        
        if label_col and label_col in df.columns:
            # Annotate a random sample of points to avoid clutter
            sample_df = df.sample(min(20, len(df)))
            for idx, row in sample_df.iterrows():
                plt.annotate(
                    str(row[label_col])[:15], # Truncate long labels
                    (row[x_col], row[y_col]),
                    xytext=(5, 5), textcoords='offset points', fontsize=8
                )

        plt.title(f'PCA Projection (Preserved Variance: {sum(self.pca.explained_variance_ratio_)*100:.1f}%)')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.grid(True, linestyle='--', alpha=0.5)
        
        print(f"Saving plot to {output_image}...")
        plt.savefig(output_image)
        plt.close()

# --- Command Line Interface ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PCA on a column of embeddings in a CSV.")
    
    # Required Arguments
    parser.add_argument('input_file', type=str, help="Path to CSV containing embeddings.")
    parser.add_argument('embedding_col', type=str, help="Name of the column with embedding vectors.")
    
    # Optional Arguments
    parser.add_argument('--output_csv', type=str, default='pca_results.csv', help="Where to save the data with PCA columns.")
    parser.add_argument('--output_plot', type=str, default='pca_plot.png', help="Where to save the visualization.")
    parser.add_argument('--label_col', type=str, default=None, help="Column name to use for plot labels (e.g., Song Title).")
    parser.add_argument('--components', type=int, default=2, help="Number of PCA components.")

    args = parser.parse_args()

    # 1. Initialize
    analyzer = PCAAnalyzer(n_components=args.components)

    # 2. Load & Parse
    df, matrix = analyzer.load_and_parse_csv(args.input_file, args.embedding_col)

    # 3. Run PCA
    pca_results = analyzer.run_analysis(matrix)

    # 4. Save Data
    # Dynamically create column names like 'pc_1', 'pc_2', etc.
    cols = [f'pc_{i+1}' for i in range(args.components)]
    pca_df = pd.DataFrame(pca_results, columns=cols, index=df.index)
    
    # Combine original data with PCA results
    final_df = pd.concat([df, pca_df], axis=1)
    
    print(f"Saving data to {args.output_csv}...")
    if args.output_csv.endswith('.pkl'):
        final_df.to_pickle(args.output_csv)
    else:
        final_df.to_csv(args.output_csv, index=False)

    # 5. Plot (if 2D)
    if args.components == 2:
        analyzer.plot_2d(final_df, 'pc_1', 'pc_2', label_col=args.label_col, output_image=args.output_plot)