import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

# Create a random DataFrame and convert it back to a NumPy array
arr = rng.normal(size=(3, 4))
df_from_arr = pd.DataFrame(arr, columns=list("ABCD"))
back_to_np = df_from_arr.to_numpy()
print("DataFrame:\n", df_from_arr, "\nNumPy Array:\n", back_to_np)