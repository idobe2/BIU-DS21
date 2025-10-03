import pandas as pd

# Create a DataFrame from a dictionary
df = pd.DataFrame({"A": [1, 3, 5], "B": [2, 4, 6]})

# Add a new column 'sum' which is the sum of columns 'A' and 'B'
df["sum"] = df["A"] + df["B"]
print("DataFrame:\n", df)