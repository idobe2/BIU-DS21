import numpy as np

# יצירת מטריצה 3x3 (דירוג נמוך בכוונה)
M = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [5, 7, 9]  # שורה זו היא סכום שתי הקודמות
])

# ביצוע SVD
U, S, Vh = np.linalg.svd(M)

print("--- Q29: SVD Results ---")
print("Matrix U (Left Singular Vectors):\n", U)
print("\nSingular Values (S):\n", S)
print("\nMatrix Vh (Right Singular Vectors Transposed):\n", Vh)