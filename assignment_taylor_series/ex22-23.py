import numpy as np
import matplotlib.pyplot as plt


# ----- Q22: Taylor series of cos^3(x) up to order 6 -----
def taylor_cos3_order6(x):
    return 1.0 - 1.5 * x**2 + (7.0 / 8.0) * x**4 - (61.0 / 240.0) * x**6


x0 = 0.5  # radians
true_val = np.cos(x0) ** 3
approx_val = taylor_cos3_order6(x0)
print(f"Q22 at x={x0}:")
print(f"  true cos^3(x)   = {true_val:.8f}")
print(f"  Taylor (order 6) = {approx_val:.8f}")
print(f"  abs error        = {abs(true_val-approx_val):.8e}")

# ----- Q23: plot true function vs. Taylor approximation -----
x_vals = np.linspace(-2 * np.pi, 2 * np.pi, 400)
y_true = np.cos(x_vals) ** 3
y_taylor = taylor_cos3_order6(x_vals)

plt.figure(figsize=(10, 6))
plt.plot(x_vals, y_true, "b-", label="cos^3(x) - true", linewidth=3)
plt.plot(x_vals, y_taylor, "r--", label="Taylor up to order 6", linewidth=2)
plt.title("cos^3(x) vs. Taylor Approximation (Order 6)")
plt.xlabel("x (radians)")
plt.ylabel("value")
plt.legend()
plt.grid(True)
plt.ylim(-1.5, 1.5)
plt.show()
