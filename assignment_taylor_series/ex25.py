import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import get_window
from scipy.fft import rfft, rfftfreq

FS = 1000
N = 1000
t = np.linspace(0.0, 1.0, N, endpoint=False)
freqs = rfftfreq(N, d=1/FS)

# 10.5 Hz sine wave
f_leak = 10.5
y_leak = np.sin(2 * np.pi * f_leak * t)
amp_leak = 2.0 / N * np.abs(rfft(y_leak))

# Hamming window reduces sidelobes (leakage)
window = get_window("hamming", N)
y_windowed = y_leak * window
amp_windowed = 2.0 / np.sum(window) * np.abs(rfft(y_windowed))  # amplitude corrected

# same Y limits for fair visual comparison
ymax = max(amp_leak.max(), amp_windowed.max()) * 1.05

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(freqs, amp_leak)
plt.axvline(f_leak, color="k", linestyle=":", linewidth=1)
plt.title("Spectral Leakage (10.5 Hz, rectangular)")
plt.xlabel("Frequency (Hz)"); plt.ylabel("Amplitude")
plt.xlim(0, 40); plt.ylim(0, ymax); plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(freqs, amp_windowed)
plt.axvline(f_leak, color='k', linestyle=':', linewidth=1)
plt.title("Hamming window (reduced leakage)")
plt.xlabel("Frequency (Hz)"); plt.ylabel("Amplitude")
plt.xlim(0, 40); plt.ylim(0, ymax); plt.grid(True)

plt.tight_layout()
plt.show()
