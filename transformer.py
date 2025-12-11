import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import numpy as np

# Load the CSV
df = pd.read_csv("headlines_preprocessed_dual.csv", sep='|')

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load BERT
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased").to(device)
model.eval()

batch_size = 32
texts = df["headline_bert"].astype(str).tolist()

all_embeddings = []

for start in tqdm(range(0, len(texts), batch_size), desc="Processing BERT"):
    batch = texts[start:start + batch_size]

    encoded = tokenizer(
        batch,
        padding=True,
        truncation=True,
        max_length=64,
        return_tensors="pt"
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        outputs = model(**encoded)
        cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()  # CLS token
        all_embeddings.append(cls_emb)

# Combine all embeddings
all_embeddings = np.vstack(all_embeddings)

# Add embeddings as new columns
embedding_cols = [f"bert_dim_{i}" for i in range(all_embeddings.shape[1])]
emb_df = pd.DataFrame(all_embeddings, columns=embedding_cols)

# Concatenate with original data
final_df = pd.concat([df, emb_df], axis=1)

# Save new CSV
final_df.to_csv("corrected.csv", index=False, sep='|')
print("Saved.")
