"""
合并 Word2Vec 嵌入向量部分文件
类似于 merger2.py，但用于 Word2Vec 嵌入
"""
import pandas as pd
import os
from natsort import natsorted

# Get project root directory
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
data_dir = os.path.join(project_root, 'data')

folder = os.path.join(data_dir, "embeddings", "word2vec_embeddings_part")

# Check if folder exists
if not os.path.exists(folder):
    print(f"Error: Folder not found: {folder}")
    print("Please run Word2Vec.py first to generate embeddings.")
    exit(1)

# List all CSV files
files = [f for f in os.listdir(folder) if f.endswith(".csv")]

if len(files) == 0:
    print(f"Error: No CSV files found in {folder}")
    exit(1)

# Sort them naturally (so part_10 comes after part_9)
files = natsorted(files)

print(f"Found {len(files)} embedding parts.")

dfs = []
for f in files:
    path = os.path.join(folder, f)
    print("Loading:", path)
    df = pd.read_csv(path, header=0)
    dfs.append(df)

# Concatenate all embeddings
final = pd.concat(dfs, ignore_index=True)
print("Final shape:", final.shape)

# Save final merged embeddings
output_path = os.path.join(data_dir, "embeddings", "word2vec_embeddings_all.csv")
final.to_csv(output_path, index=False)
print(f"Saved as: {output_path}")


