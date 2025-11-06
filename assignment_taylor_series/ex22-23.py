import numpy as np
import matplotlib.pyplot as plt
import math

def approx_cos_order6(x):
    """
    Q22: Calculates the Taylor approximation of cos(x) up to order 6.
    Based on the series: 1 - x^2/2! + x^4/4! - x^6/6!
    """
    term1 = 1.0
    term2 = -(x**2) / math.factorial(2)
    term3 = (x**4) / math.factorial(4)
    term4 = -(x**6) / math.factorial(6)
    
    return term1 + term2 + term3

# --- פתרון לשאלה 23: הצגת הגרף ---

# ציר X: ערכים בין -2*pi ל- 2*pi
x_vals = np.linspace(-2 * np.pi, 2 * np.pi, 400)

# חישוב הערך האמיתי
true_cos = np.cos(x_vals)

# חישוב הקירוב
taylor_approx = approx_cos_order6(x_vals)

# שרטוט הגרף (בדומה לגרף הדוגמה במצגת [cite: 786-798])
plt.figure(figsize=(10, 6))
plt.plot(x_vals, true_cos, 'b-', label='cos(x) - האמיתי', linewidth=3)
plt.plot(x_vals, taylor_approx, 'r--', 
         label='קירוב טיילור (סדר 6)', linewidth=2)
plt.title("Q23: cos(x) vs. Taylor Approximation (Order 6)")
plt.xlabel("x (radians)")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.ylim(-1.5, 1.5) # הגבלת ציר Y
plt.show()