import numpy as np

rng = np.random.default_rng(42)

# Create an array of 15 random floats between 0 and 1 and compute max, min, and mean
rand_arr = rng.random(15)
print("max =", rand_arr.max(), "\nmin =", rand_arr.min(), "\nmean =", rand_arr.mean())