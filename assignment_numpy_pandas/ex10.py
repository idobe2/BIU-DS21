import pandas as pd

# Read a CSV file into a DataFrame
df = pd.read_csv("examples/username.csv")
print("DataFrame:\n", df.head())