import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import numpy as np
import os

# Get project root directory
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
data_dir = os.path.join(project_root, 'data')

df = pd.read_csv(os.path.join(data_dir, "processed", "headlines_preprocessed.csv"), sep='|')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased").to(device)
model.eval()

batch_size = 32
texts = df["headline_clean"].astype(str).tolist()

save_every = 1000
current_embeddings = []
saved_files = []
total_processed = 0
chunk_id = 0

for start in tqdm(range(0, len(texts), batch_size), desc="Processing BERT"):
    batch = texts[start:start + batch_size]

    encoded = tokenizer(
        batch,
        padding=True,
        truncation=True,
        max_length=32,
        return_tensors="pt"
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        outputs = model(**encoded)
        cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()

    current_embeddings.append(cls_emb)
    total_processed += len(batch)

    # Save chunk every 1000 items
    if total_processed >= save_every:
        combined = np.vstack(current_embeddings)
        embeddings_dir = os.path.join(data_dir, "embeddings", "bert_embeddings_part")
        os.makedirs(embeddings_dir, exist_ok=True)
        filename = os.path.join(embeddings_dir, f"bert_embeddings_part_{chunk_id}.csv")
        pd.DataFrame(combined).to_csv(filename, index=False)

        print(f"\nSaved: {filename}  (shape: {combined.shape})\n")

        saved_files.append(filename)
        current_embeddings = []
        total_processed = 0
        chunk_id += 1

# Save final leftover
if current_embeddings:
    combined = np.vstack(current_embeddings)
    embeddings_dir = os.path.join(data_dir, "embeddings", "bert_embeddings_part")
    os.makedirs(embeddings_dir, exist_ok=True)
    filename = os.path.join(embeddings_dir, f"bert_embeddings_part_{chunk_id}.csv")
    pd.DataFrame(combined).to_csv(filename, index=False)

    print(f"\nSaved final part: {filename}  (shape: {combined.shape})\n")
    saved_files.append(filename)

print("\nAll CSV parts saved:")
for f in saved_files:
    print(" -", f)
