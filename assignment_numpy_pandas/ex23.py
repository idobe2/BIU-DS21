import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

# Centering a NumPy array by subtracting the mean of each column
X = rng.normal(size=(20, 2))
X_centered = X - X.mean(axis=0)
print("mean before:", X.mean(axis=0), "\nmean after:", X_centered.mean(axis=0))