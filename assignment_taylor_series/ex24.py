import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq

# Signal settings
FS = 500  # sampling rate [Hz]
T = 2.0  # duration [s]
N = int(FS * T)
t = np.linspace(0.0, T, N, endpoint=False)

# Three-tone signal: 5 Hz (amp=2), 15 Hz (amp=1), 25 Hz (amp=0.5)
y = (
    2.0 * np.sin(2 * np.pi * 5 * t)
    + 1.0 * np.sin(2 * np.pi * 15 * t)
    + 0.5 * np.sin(2 * np.pi * 25 * t)
)

# Single-sided FFT with amplitude normalization
freqs = rfftfreq(N, d=1 / FS)
amp = 2.0 / N * np.abs(rfft(y))

target = [5, 15, 25]

# Plot spectrum
plt.figure(figsize=(10, 4))
plt.plot(freqs, amp)
for f0 in target:
    plt.axvline(f0, color="k", linestyle=":", linewidth=0.8)  # visual guide
plt.title("Spectrum of Combined Signal (5, 15, 25 Hz)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.xlim(0, 30)
plt.show()
