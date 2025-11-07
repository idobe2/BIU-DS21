import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

FS = 1000
N = 1000
t = np.linspace(0.0, 1.0, N, endpoint=False)

signal = np.sin(2 * np.pi * 50 * t)
noise = 0.5 * np.sin(2 * np.pi * 120 * t) + 0.3 * np.random.randn(N)
y_noisy = signal + noise

# Real FFT (positive freqs only)
Y = np.fft.rfft(y_noisy)
freqs = np.fft.rfftfreq(N, d=1/FS)

# Low-pass mask up to 100 Hz
mask = freqs <= 100
Y_filtered = Y * mask

# Back to time (real)
y_clean = np.fft.irfft(Y_filtered, n=N)

plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(t, y_noisy)
plt.title("Original Signal with Noise")
plt.xlim(0, 0.2)

plt.subplot(2, 1, 2)
plt.plot(t, y_clean, 'r')
plt.title("Filtered Signal")
plt.xlim(0, 0.2)

plt.tight_layout()
plt.show()
