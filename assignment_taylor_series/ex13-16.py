import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq
from scipy.signal import get_window

# Sampling settings
sampling_rate = 1000  # Hz
d = 1.0  # seconds
N = int(sampling_rate * d)  # samples
t = np.linspace(0.0, d, N, endpoint=False)

# Q13: 10 Hz sine and its FFT
f_signal = 10  # Hz
y_10hz = np.sin(2 * np.pi * f_signal * t)

freqs = rfftfreq(N, d=1 / sampling_rate)
Y_10hz = rfft(y_10hz)
amp_10hz = 2.0 / N * np.abs(Y_10hz)  # single-sided amplitude

# Q14: add noise and compute FFT
noise = 1.5 * np.random.normal(size=N)
y_noisy = y_10hz + noise

Y_noisy = rfft(y_noisy)
amp_noisy = 2.0 / N * np.abs(Y_noisy)

# Q15: apply Hamming window and FFT
window = get_window("hamming", N)
y_windowed = y_noisy * window
Y_windowed = rfft(y_windowed)
amp_windowed = 2.0 / np.sum(window) * np.abs(Y_windowed)  # (2/N)/CG where CG=sum(w)/N

# ----- plots for Q13–Q15 -----
plt.figure(figsize=(12, 10))

plt.subplot(3, 1, 1)
plt.plot(freqs, amp_10hz)
plt.title("Spectrum of 10 Hz Signal")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.xlim(0, 20)
plt.ylim(0, 1.2)

plt.subplot(3, 1, 2)
plt.plot(freqs, amp_noisy)
plt.title("Spectrum with Noise")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.xlim(0, 20)
plt.ylim(0, 1.2)

plt.subplot(3, 1, 3)
plt.plot(freqs, amp_windowed)
plt.title("Spectrum with Noise + Hamming Window")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.xlim(0, 20)
plt.ylim(0, 1.2)

plt.tight_layout()
plt.show()

# Q16: two close tones (50 & 55 Hz)
# long window (1.0 s) vs short (0.1 s)
f1, f2 = 50, 55
fs_q16 = 1000

# Long window: 1.0s
N_long = 1000
t_long = np.linspace(0.0, 1.0, N_long, endpoint=False)
y_long = np.sin(2 * np.pi * f1 * t_long) + np.sin(2 * np.pi * f2 * t_long)
freqs_long = rfftfreq(N_long, d=1 / fs_q16)
Y_long = rfft(y_long)
amp_long = 2.0 / N_long * np.abs(Y_long)

# Short window: 0.1s
N_short = 100
t_short = np.linspace(0.0, 0.1, N_short, endpoint=False)
y_short = np.sin(2 * np.pi * f1 * t_short) + np.sin(2 * np.pi * f2 * t_short)
freqs_short = rfftfreq(N_short, d=1 / fs_q16)
Y_short = rfft(y_short)
amp_short = 2.0 / N_short * np.abs(Y_short)

# ----- plots for Q16 -----
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(freqs_long, amp_long)
plt.title("Long Window (1.0 s)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.xlim(40, 70)
plt.ylim(0, 1.2)

plt.subplot(1, 2, 2)
plt.plot(freqs_short, amp_short)
plt.title("Short Window (0.1 s)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.xlim(40, 70)
plt.ylim(0, 1.2)

plt.tight_layout()
plt.show()
