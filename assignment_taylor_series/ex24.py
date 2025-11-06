import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq

# הגדרות האות
FS = 500  # קצב דגימה
T = 2     # משך האות (2 שניות)
N = int(FS * T)
t = np.linspace(0.0, T, N, endpoint=False)

# יצירת האות מ-3 תדרים (בדומה למצגת )
y1 = 2 * np.sin(2 * np.pi * 5 * t)
y2 = 1 * np.sin(2 * np.pi * 15 * t)
y3 = 0.5 * np.sin(2 * np.pi * 25 * t)
y_combined = y1 + y2 + y3

# חישוב FFT (לפי המאמר )
freqs = rfftfreq(N, d=1/FS)
amp = 2.0/N * np.abs(rfft(y_combined))

# הצגת הספקטרום
plt.figure(figsize=(10, 4))
plt.plot(freqs, amp)
plt.title("Q24: Spectrum of Combined Signal (5Hz, 15Hz, 25Hz)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.xlim(0, 50) # התמקדות בתדרים הרלוונטיים
plt.show()