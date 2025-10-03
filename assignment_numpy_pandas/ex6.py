import numpy as np

# Create a NumPy array containing integers from 1 to 20, replace odd numbers with -1
arr = np.arange(1, 21)
arr[arr % 2 == 1] = -1
print("Array:", arr)