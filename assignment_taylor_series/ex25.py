import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import get_window
from scipy.fft import rfft, rfftfreq


FS = 1000
N = 1000
t = np.linspace(0.0, 1.0, N, endpoint=False)
freqs = rfftfreq(N, d=1/FS)

# 1. אות עם דליפה (תדר 10.5Hz)
f_leak = 10.5 
y_leak = np.sin(2 * np.pi * f_leak * t)
amp_leak = 2.0/N * np.abs(rfft(y_leak))

# 2. אותו אות עם חלון (הפתרון לדליפה )
window = get_window('hamming', N)
y_windowed = y_leak * window
amp_windowed = 2.0/np.sum(window) * np.abs(rfft(y_windowed))

# הצגת הגרפים
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(freqs, amp_leak)
plt.title("Q25: Spectral Leakage (10.5Hz Signal)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.xlim(0, 40)
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(freqs, amp_windowed)
plt.title("Q25: Leakage Solved with Hamming Window")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.xlim(0, 40)
plt.grid(True)

plt.tight_layout()
plt.show()