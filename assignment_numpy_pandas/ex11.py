import pandas as pd

# Create a DataFrame from a dictionary
df_people = pd.DataFrame({
    "name": ["Noa", "Daniel", "Maya", "Yoav", "Tamar"],
    "age": [23, 29, 21, 27, 25]
})

# Calculate basic statistics for the 'age' column
basic_stats = df_people["age"].agg(["mean", "std", "min", "max"])
print("Basic Statistics:\n", basic_stats)
