import pandas as pd

# Create a DataFrame from a dictionary
df_people = pd.DataFrame({
    "name": ["Noa", "Daniel", "Maya", "Yoav", "Tamar"],
    "age": [23, 29, 21, 27, 25]
})
print("DataFrame:\n", df_people)