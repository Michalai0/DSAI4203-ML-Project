import pandas as pd
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Load CSV
df = pd.read_csv("headlines_features_ver3.csv", sep='|', encoding='utf-8')

# Assign dominant LDA topic
lda_columns = [f"topic_{i}" for i in range(8)]
df['dominant_topic_num'] = df[lda_columns].idxmax(axis=1).str.replace('topic_', '').astype(int)

# Stopwords for later analysis
stop = set(stopwords.words("english"))

# Crosstab: dominant topic vs category
topic_by_category = pd.crosstab(df['dominant_topic_num'], df['category'], normalize='columns') * 100

# Save normalized percentages for report
topic_by_category.to_csv("lda_topic_by_category_percent.csv", sep='|', encoding='utf-8')

# Visualize heatmap
plt.figure(figsize=(14,7))
sns.heatmap(topic_by_category, annot=True, fmt=".1f", cmap="Blues")
plt.title("Normalized LDA Topic Distribution per Category (%)")
plt.xlabel("Category")
plt.ylabel("Dominant LDA Topic")
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# Filter out dominant category (Politics / U.S. NEWS)
minority_df = df[df['category'] != "U.S. NEWS"]

# Crosstab and normalize
minority_topic_pct = pd.crosstab(minority_df['dominant_topic_num'], minority_df['category'], normalize='columns') * 100

# Heatmap for minority categories
plt.figure(figsize=(12,6))
sns.heatmap(minority_topic_pct, annot=True, fmt=".1f", cmap="Greens")
plt.title("Normalized LDA Topics for Minority Categories (%)")
plt.xlabel("Category")
plt.ylabel("Dominant LDA Topic")
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

for cat in df['category'].unique():
    words = " ".join(df[df['category']==cat]['headline_clean']).split()
    top_words = Counter([w.lower() for w in words if w.lower() not in stop]).most_common(10)
    print(f"Category: {cat}")
    print("Top words:", top_words)
    print()

import numpy as np

def entropy(prob_array):
    # Add small epsilon to avoid log(0)
    prob_array = np.array(prob_array) + 1e-10
    return -np.sum(prob_array * np.log2(prob_array))

category_entropy = {}
for cat in df['category'].unique():
    probs = df[df['category']==cat][lda_columns].mean(axis=0)
    category_entropy[cat] = entropy(probs)

# Print sorted by entropy
sorted_entropy = dict(sorted(category_entropy.items(), key=lambda x: x[1], reverse=True))
print("Category entropy (topic diversity):")
for cat, ent in sorted_entropy.items():
    print(f"{cat}: {ent:.2f}")
