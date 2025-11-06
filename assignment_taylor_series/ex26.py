import numpy as np
import matplotlib.pyplot as plt


# קוד זה מבוסס ישירות על עמוד 17 במצגת "פורייה" 
FS = 1000
N = 1000
t = np.linspace(0.0, 1.0, N, endpoint=False)

# 1. יצירת אות (50Hz) עם רעש בתדר גבוה
signal = np.sin(2 * np.pi * 50 * t)
noise = 0.5 * np.sin(2 * np.pi * 120 * t) + 0.3 * np.random.randn(N)
y_noisy = signal + noise

# 2. התמרת פורייה (נשתמש ב-FFT רגיל כפי שמופיע במצגת)
Y = np.fft.fft(y_noisy)
freqs = np.fft.fftfreq(N, d=1/FS)

# 3. סינון: איפוס כל התדרים מעל 100Hz
Y_filtered = Y.copy()
Y_filtered[np.abs(freqs) > 100] = 0

# 4. התמרה הפוכה חזרה למרחב הזמן
y_clean = np.fft.ifft(Y_filtered).real

# הצגת התוצאות
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(t, y_noisy)
plt.title("Q26: Original Signal with Noise")
plt.xlim(0, 0.2) # הצגת 0.2 שניות

plt.subplot(2, 1, 2)
plt.plot(t, y_clean, 'r')
plt.title("Q26: Filtered Signal (Noise Removed)")
plt.xlim(0, 0.2)

plt.tight_layout()
plt.show()