import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

# Create a DataFrame with custom row indices and random values
mat = rng.random((3, 5))
df = pd.DataFrame(mat, index=[f"row_{i}" for i in range(1, 4)])
print("DataFrame:\n", df)