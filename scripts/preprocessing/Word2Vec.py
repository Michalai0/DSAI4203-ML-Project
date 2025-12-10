import pandas as pd
import numpy as np
import os
from gensim.models import Word2Vec
from tqdm import tqdm

# Get project root directory
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
data_dir = os.path.join(project_root, 'data')

# Load data
df = pd.read_csv(os.path.join(data_dir, "raw", "news_w_description_subset.csv"), sep='|')
print(f"Loaded {len(df)} records")

# Combine headline and description
def combine_text(row):
    headline = str(row['headline_clean']) if pd.notna(row['headline_clean']) else ""
    description = str(row['short_description_clean']) if pd.notna(row['short_description_clean']) else ""
    combined = f"{headline} {description}".strip()
    return combined

df['combined_text'] = df.apply(combine_text, axis=1)

# Tokenize texts (split by whitespace, already cleaned)
def tokenize(text):
    if not isinstance(text, str):
        return []
    # Split by whitespace and filter empty strings
    tokens = [w.strip() for w in text.split() if w.strip()]
    return tokens

print("Tokenizing texts...")
tokenized_texts = [tokenize(text) for text in tqdm(df['combined_text'], desc="Tokenizing")]

# Filter out empty texts
valid_indices = [i for i, tokens in enumerate(tokenized_texts) if len(tokens) > 0]
tokenized_texts = [tokenized_texts[i] for i in valid_indices]
df_filtered = df.iloc[valid_indices].reset_index(drop=True)

print(f"Valid texts after filtering: {len(tokenized_texts)}")

# Train Word2Vec model
print("\nTraining Word2Vec model...")
word2vec_model = Word2Vec(
    sentences=tokenized_texts,
    vector_size=300,  # Word2Vec typically uses 300 dimensions
    window=5,
    min_count=2,  # Ignore words that appear less than 2 times
    workers=4,
    sg=0,  # 0 for CBOW, 1 for skip-gram
    epochs=10
)

print(f"Word2Vec model trained. Vocabulary size: {len(word2vec_model.wv)}")

# Generate document embeddings by averaging word vectors
def get_document_embedding(tokens, model, vector_size=300):
    """Get document embedding by averaging word vectors"""
    vectors = []
    for word in tokens:
        if word in model.wv:
            vectors.append(model.wv[word])
    
    if len(vectors) == 0:
        # Return zero vector if no words found in vocabulary
        return np.zeros(vector_size)
    
    return np.mean(vectors, axis=0)

print("\nGenerating document embeddings...")
embeddings = []
for tokens in tqdm(tokenized_texts, desc="Generating embeddings"):
    emb = get_document_embedding(tokens, word2vec_model, vector_size=300)
    embeddings.append(emb)

embeddings = np.array(embeddings)
print(f"Embeddings shape: {embeddings.shape}")

# Save in chunks (similar to BERT.py)
save_every = 1000
embeddings_dir = os.path.join(data_dir, "embeddings", "word2vec_embeddings_part")
os.makedirs(embeddings_dir, exist_ok=True)

saved_files = []
chunk_id = 0

for start in tqdm(range(0, len(embeddings), save_every), desc="Saving chunks"):
    end = min(start + save_every, len(embeddings))
    chunk = embeddings[start:end]
    
    filename = os.path.join(embeddings_dir, f"word2vec_embeddings_part_{chunk_id}.csv")
    pd.DataFrame(chunk).to_csv(filename, index=False)
    
    print(f"\nSaved: {filename}  (shape: {chunk.shape})\n")
    saved_files.append(filename)
    chunk_id += 1

print("\nAll CSV parts saved:")
for f in saved_files:
    print(" -", f)

print(f"\nTotal embeddings generated: {len(embeddings)}")
print(f"Embedding dimension: {embeddings.shape[1]}")
