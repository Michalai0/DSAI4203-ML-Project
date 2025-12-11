import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt

# Load the normalized LDA topic-by-category CSV
topic_by_category = pd.read_csv("lda_topic_by_category_percent.csv", sep='|', index_col=0)

# Compute cosine similarity between categories
similarity_matrix = cosine_similarity(topic_by_category.T)  # transpose: columns = category vectors

# Visualize heatmap
plt.figure(figsize=(12,10))
sns.heatmap(
    similarity_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    xticklabels=topic_by_category.columns,
    yticklabels=topic_by_category.columns
)
plt.title("Cosine Similarity Between Categories Based on LDA Topics")
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

import numpy as np
sim_df = pd.DataFrame(similarity_matrix, index=topic_by_category.columns, columns=topic_by_category.columns)
sim_df.to_csv("category_similarity_matrix.csv", sep='|', encoding='utf-8')

sim_no_diag = sim_df.copy()
np.fill_diagonal(sim_no_diag.values, 0)

sim_pairs = sim_no_diag.unstack()
sim_pairs = sim_pairs.sort_values(ascending=False)
sim_pairs = sim_pairs[~sim_pairs.index.duplicated(keep='first')]

top3_pairs = sim_pairs.head(3)
print("Top 3 most similar category pairs based on LDA topic distribution:")
for (cat1, cat2), sim in top3_pairs.items():
    print(f"{cat1} - {cat2}: similarity = {sim:.2f}")
