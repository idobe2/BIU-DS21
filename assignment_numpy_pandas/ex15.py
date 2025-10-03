import numpy as np
import pandas as pd

# Create a DataFrame with some NaN values
df_na = pd.DataFrame({"x": [1.0, np.nan, 3.0], "y": [4.0, 5.0, np.nan]})

# Fill NaN values with the mean of their respective columns
df_fill = df_na.copy()
for col in df_fill.columns:
    df_fill[col] = df_fill[col].fillna(df_fill[col].mean())
print("DataFrame with NaN values:\n", df_na)
print("DataFrame after filling NaN values:\n", df_fill)