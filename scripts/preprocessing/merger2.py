import pandas as pd
import os
from natsort import natsorted

# Get project root directory
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
data_dir = os.path.join(project_root, 'data')

folder = os.path.join(data_dir, "embeddings", "bert_embeddings_part")

# List all CSV files
files = [f for f in os.listdir(folder) if f.endswith(".csv")]

# Sort them naturally (so part_10 comes after part_9)
files = natsorted(files)

print(f"Found {len(files)} embedding parts.")

dfs = []
for f in files:
    path = os.path.join(folder, f)
    print("Loading:", path)
    df = pd.read_csv(path, header=0)   # read normally
    dfs.append(df)

# Concatenate all embeddings
final = pd.concat(dfs, ignore_index=True)
print("Final shape:", final.shape)

# Save final merged embeddings
output_path = os.path.join(data_dir, "embeddings", "bert_embeddings_all.csv")
final.to_csv(output_path, index=False)
print(f"Saved as: {output_path}")
