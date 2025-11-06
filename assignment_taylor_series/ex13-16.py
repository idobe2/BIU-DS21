import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq
from scipy.signal import get_window

sampling_rate = 1000        # Sampling frequency: 1000Hz
d = 1.0                     # Duration: 1 second
N = int(sampling_rate * d)  # Number of samples
t = np.linspace(0.0, d, N, endpoint=False) # Time vector

# Solution 13
f_signal = 10   # Signal frequency
y_10hz = np.sin(2 * np.pi * f_signal * t)

freqs = rfftfreq(N, d=1/sampling_rate)

# Calculate FFT
Y_10hz = rfft(y_10hz)

# Normalize amplitude
amp_10hz = 2.0/N * np.abs(Y_10hz)


# Solution 14
noise = 1.5 * np.random.normal(size=N)  # Add noise to signal
y_noisy = y_10hz + noise

# Calculate FFT of noisy signal
Y_noisy = rfft(y_noisy)
amp_noisy = 2.0/N * np.abs(Y_noisy)


# Solution 15
window = get_window('hamming', N)
y_windowed = y_noisy * window   # Apply window to noisy signal

# Calculate FFT of windowed signal
Y_windowed = rfft(y_windowed)

# Normalize amplitude
amp_windowed = 2.0/np.sum(window) * np.abs(Y_windowed)

# Plotting results for questions 13-15
plt.figure(figsize=(12, 10))

# Spectrum of 10Hz signal
plt.subplot(3, 1, 1)
plt.plot(freqs, amp_10hz)
plt.title("Spectrum of 10Hz Signal")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.xlim(0, 50)
plt.ylim(0, 1.2)

# Spectrum of noisy signal
plt.subplot(3, 1, 2)
plt.plot(freqs, amp_noisy)
plt.title("Spectrum with Noise")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.xlim(0, 50)
plt.ylim(0, 1.2)

# Spectrum of windowed noisy signal
plt.subplot(3, 1, 3)
plt.plot(freqs, amp_windowed)
plt.title("Spectrum with Noise + Hamming Window")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.xlim(0, 50)
plt.ylim(0, 1.2)

plt.tight_layout()
plt.show()


# Solution 16
f1 = 50
f2 = 55
fs_q16 = 1000 # sampling rate

# Long window - 1.0s
N_long = 1000
t_long = np.linspace(0.0, 1.0, N_long, endpoint=False)
y_long = np.sin(2 * np.pi * f1 * t_long) + np.sin(2 * np.pi * f2 * t_long)

freqs_long = rfftfreq(N_long, d=1/fs_q16)
Y_long = rfft(y_long)
amp_long = 2.0/N_long * np.abs(Y_long)

# Short window - 0.1s
N_short = 100
t_short = np.linspace(0.0, 0.1, N_short, endpoint=False)
y_short = np.sin(2 * np.pi * f1 * t_short) + np.sin(2 * np.pi * f2 * t_short)

freqs_short = rfftfreq(N_short, d=1/fs_q16)
Y_short = rfft(y_short)
amp_short = 2.0/N_short * np.abs(Y_short)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(freqs_long, amp_long)
plt.title("Long Window (1.0s)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.xlim(40, 70)
plt.ylim(0, 1.2)

plt.subplot(1, 2, 2)
plt.plot(freqs_short, amp_short)
plt.title("Short Window (0.1s)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.xlim(40, 70)
plt.ylim(0, 1.2)

plt.tight_layout()
plt.show()