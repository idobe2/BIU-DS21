import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

# Calculate and print the mean of each column in a DataFrame
arr = rng.random((5, 3))
df = pd.DataFrame(arr, columns=["A", "B", "C"])
print("DataFrame Means:\n", df.mean())