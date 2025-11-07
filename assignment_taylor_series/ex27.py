import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

FS = 1000
T = 2.0
t = np.linspace(0, T, int(T*FS), endpoint=False)

# time-varying signal (linear chirp: 10 -> 100 Hz over T seconds)
y = signal.chirp(t, f0=10, f1=100, t1=T, method='linear')

# STFT (spectrogram) with explicit params
nperseg = 256           # window length (~0.256 s)
noverlap = 128          # 50% overlap
f, tt, Sxx = signal.spectrogram(
    y, FS, window='hann', nperseg=nperseg, noverlap=noverlap,
    detrend=False, mode='psd'   # power spectral density
)

# convert to dB for better contrast
Sxx_db = 10 * np.log10(Sxx + 1e-12)

plt.figure(figsize=(10, 6))
plt.pcolormesh(tt, f, Sxx_db, shading='gouraud')
plt.title("Spectrogram (STFT) of a Linear Chirp")
plt.ylabel("Frequency (Hz)")
plt.xlabel("Time (s)")
plt.ylim(0, 150)
plt.colorbar(label="Power (dB)")
plt.tight_layout()
plt.show()
