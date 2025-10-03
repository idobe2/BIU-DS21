import numpy as np
import pandas as pd

rng = np.random.default_rng(1)

# Create random DataFrame and convert to NumPy array
df_rand = pd.DataFrame(rng.normal(size=(8, 6)))
X_df = df_rand.to_numpy()
Ud, Sd, Vtd = np.linalg.svd(X_df, full_matrices=False)
k = 2  # number of singular values to keep
X_rank_k = (Ud[:, :k] * Sd[:k]) @ Vtd[:k, :]
print("Original shape:", X_df.shape, "\nLow-rank approx shape:", X_rank_k.shape)