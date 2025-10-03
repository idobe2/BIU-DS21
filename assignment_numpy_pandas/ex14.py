import numpy as np
import pandas as pd

# Create a DataFrame with some NaN values and drop them
df_na = pd.DataFrame({"x": [1.0, np.nan, 3.0], "y": [4.0, 5.0, np.nan]})
print("DataFrame with NaN values:\n", df_na)
print("Cleaned DataFrame:\n", df_na.dropna())