import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

# Select rows from a DataFrame based on a condition on the row means compared to the overall mean
df = pd.DataFrame(rng.random((4, 6)))
overall_mean = df.to_numpy().mean()
row_means = df.mean(axis=1)
print("Filtered DataFrame:\n", df[row_means > overall_mean])