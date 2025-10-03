import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

# Create a DataFrame with a date column and random values
n = 7
dates = np.arange(np.datetime64("2025-01-01"), np.datetime64("2025-01-01") + n)
df = pd.DataFrame({"value": rng.random(n)})
df["date"] = dates
print("DataFrame:\n", df)