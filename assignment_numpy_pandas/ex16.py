import pandas as pd

# Create a DataFrame with some scores
df_scores = pd.DataFrame({"name": ["a", "b", "c"], "score": [40, 75, 55]})

# Filter rows where score is greater than 50
filtered = df_scores[df_scores["score"] > 50]
print("Filtered DataFrame:\n", filtered)