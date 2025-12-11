import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load CSV
df = pd.read_csv("headlines_features_ver3.csv", sep='|', encoding='utf-8')

# 2. Assign dominant LDA topic
lda_columns = [f"topic_{i}" for i in range(8)]  # topic_0 ... topic_7
df['dominant_topic'] = df[lda_columns].idxmax(axis=1)  # e.g., 'topic_0', 'topic_1', ...

# Optional: convert to integer topic index
df['dominant_topic_num'] = df['dominant_topic'].str.replace('topic_', '').astype(int)

# 3. Cross-tabulation: dominant topic vs original category
topic_category_crosstab = pd.crosstab(df['dominant_topic_num'], df['category'])

# 4. Convert to percentages (row-wise)
topic_category_pct = topic_category_crosstab.div(topic_category_crosstab.sum(axis=1), axis=0) * 100

# 5. Plot heatmap
plt.figure(figsize=(14,8))
sns.heatmap(
    topic_category_pct,
    annot=True,          # show numbers
    fmt=".1f",           # number format
    cmap="Blues",
    annot_kws={"size":12}  # font size
)
plt.title("LDA Dominant Topic vs Original Category (%)")
plt.xlabel("Original Category")
plt.ylabel("Dominant LDA Topic")
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.show()

# 6. Optional: print table for report
print("Cross-tab (percentages):")
print(topic_category_pct)

topic_category_pct.to_csv("lda_topic_vs_category_percent.csv", sep='|', encoding='utf-8')
print("Saved cross-tab percentages to lda_topic_vs_category_percent.csv")
