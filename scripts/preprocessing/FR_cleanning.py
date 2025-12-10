import pandas as pd
import json
import re
import os
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Get project root directory
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
data_dir = os.path.join(project_root, 'data')

# Load dataset
data = []
with open(os.path.join(data_dir, "raw", "News_Category_Dataset_v3.json"), "r") as f:
    for line in f:
        data.append(json.loads(line))

df = pd.DataFrame(data)
print("Original dataset size:", len(df))

# Preprocessing tools
stop = set(stopwords.words("english"))
lemm = WordNetLemmatizer()

def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    # Normalize quotes and lowercase
    text = text.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    text = re.sub(r'"+', '"', text)
    text = re.sub(r"'+", "'", text)
    text = text.lower()
    
    # Remove URLs and emails
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove punctuation
    text = text.replace('"', '').replace("'", "")
    tokens = text.split()
    tokens = [w.strip('.,!?()[]{}:;') for w in tokens]
    
    # Remove stopwords and lemmatize
    tokens = [lemm.lemmatize(w) for w in tokens if w not in stop]
    
    return " ".join(tokens)

# Apply to headlines only
df['headline_clean'] = df['headline'].apply(clean_text)

# Save a formatted CSV for your teammates
df_to_save = df[['category', 'headline_clean']]
processed_dir = os.path.join(data_dir, "processed")
os.makedirs(processed_dir, exist_ok=True)
df_to_save.to_csv(os.path.join(processed_dir, "headlines_preprocessed.csv"), index=False, sep='|', quoting=1)

print("Preprocessing done. CSV saved as 'headlines_preprocessed.csv'")
