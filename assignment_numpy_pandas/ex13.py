import pandas as pd

# Create a DataFrame from a dictionary
df_people = pd.DataFrame({
    "name": ["Noa", "Daniel", "Maya", "Yoav", "Tamar"],
    "age": [23, 29, 21, 27, 25]
})
print("DataFrame:\n", df_people)

# Sort the DataFrame by age in ascending order
sorted_df = df_people.sort_values("age", ascending=True)
print("Sorted DataFrame:\n", sorted_df)