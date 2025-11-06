import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

FS = 1000
T = 2
t = np.linspace(0, T, T * FS, endpoint=False)

# יצירת אות "צ'ירפ" (תדר עולה)
# אות זה משתנה בזמן מ-10Hz ל-100Hz
y_chirp = signal.chirp(t, f0=10, f1=100, t1=T, method='linear')

# חישוב הספקטרוגרמה (STFT) 
f, t_spec, Sxx = signal.spectrogram(y_chirp, FS)

# הצגת הספקטרוגרמה
plt.figure(figsize=(10, 6))
plt.pcolormesh(t_spec, f, Sxx, shading='gouraud')
plt.title("Q27: Spectrogram (STFT) of a Chirp Signal")
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (sec)')
plt.ylim(0, 150) # הצגת התדרים הרלוונטיים
plt.show()