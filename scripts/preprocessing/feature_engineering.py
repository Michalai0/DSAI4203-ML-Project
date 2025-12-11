import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import re
from tqdm import tqdm

tqdm.pandas()  # enable progress_apply

# Load CSV
df = pd.read_csv("headlines_preprocessed.csv", sep='|', encoding='utf-8')

# Stopwords
stop = set(stopwords.words("english"))

# Feature 1: Keyword density
def keyword_density(text):
    if not isinstance(text, str):
        return 0
    words = text.split()
    content_words = [w for w in words if w.lower() not in stop]
    return len(content_words)

df["keyword_density"] = df["headline_clean"].progress_apply(keyword_density)

# Feature 2: Stylometric
df["numbers_count"] = df["headline_clean"].progress_apply(lambda x: sum(1 for c in str(x) if c.isdigit()))

# Feature 3: Word-based features
df["headline_clean"] = df["headline_clean"].fillna("")

def avg_word_length(text):
    if not isinstance(text, str):
        return 0
    words = text.split()
    if len(words) == 0:
        return 0
    return sum(len(w) for w in words) / len(words)

def headline_length(text):
    if not isinstance(text, str):
        return 0
    return len(text.split())

df["avg_word_length"] = df["headline_clean"].progress_apply(avg_word_length)
df["headline_length"] = df["headline_clean"].progress_apply(headline_length)

# Feature 4: TF-IDF + LDA topics
tfidf = TfidfVectorizer(max_features=500, stop_words='english')
X_tfidf = tfidf.fit_transform(df["headline_clean"].astype(str))

lda = LatentDirichletAllocation(n_components=5, random_state=42)
# Show progress for LDA using tqdm (wrap fit_transform with a loop)
# Note: LDA doesn't natively support progress bars, but you can use verbose=1
lda = LatentDirichletAllocation(n_components=8, random_state=42, verbose=1)
lda_topics = lda.fit_transform(X_tfidf)

n_top_words = 20
tf_feature_names = tfidf.get_feature_names_out()
for topic_idx, topic in enumerate(lda.components_):
    print(f"Topic {topic_idx}:")
    top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
    top_features = [tf_feature_names[i] for i in top_features_ind]
    print(" ".join(top_features))
    print()

lda_df = pd.DataFrame(lda_topics, columns=[f"topic_{i}" for i in range(lda_topics.shape[1])])
df = pd.concat([df, lda_df], axis=1)

# Save
df.to_csv("headlines_features_ver3.csv", sep='|', index=False, encoding='utf-8')
print("Feature engineering complete.")
print(df.head())
