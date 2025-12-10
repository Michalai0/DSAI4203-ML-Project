import pandas as pd
import matplotlib.pyplot as plt
import os

# Get project root directory
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
data_dir = os.path.join(project_root, 'data')

# Load original labelled data
df = pd.read_csv(os.path.join(data_dir, "processed", "headlines_preprocessed.csv"), sep="|")

# Count entries per category
category_counts = df['category'].value_counts()
print(category_counts)

# Visualise distribution
plt.figure(figsize=(10,6))
category_counts.plot(kind='bar')
plt.title("Number of Entries per Category")
plt.xlabel("Category")
plt.ylabel("Count")
plt.tight_layout()
plt.show()
