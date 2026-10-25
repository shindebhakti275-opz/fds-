import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['class'] = [iris.target_names[i] for i in iris.target]

sample_df = df.sample(n=20, random_state=42)

print("Min Numeric:\n", df.min(numeric_only=True))
print("\nMax Numeric:\n", df.max(numeric_only=True))
print("\nClass Value Counts:\n", df['class'].value_counts())
