import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

# Calculating variance for a NumPy array with ddof=1 (standard sample variance) 
df_var = pd.DataFrame(rng.random((6, 4)))
print("DataFrame:\n", df_var.var(ddof=1))