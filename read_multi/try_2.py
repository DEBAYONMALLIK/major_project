import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft
import pandas as pd


   




# ---- Replace these with your actual data arrays ----
# Example placeholders (you already have these)
# time = data['time'].values
# signal = data['magnitude'].values

# Sampling interval and rate
filename="major_project/read_multi/F0010CH1.CSV"
df = pd.read_csv(filename, header=None, sep=',',
                     usecols=[3, 4],  #  zero indexed
                     names=['time', 'magnitude'])
time=df['time'].values
signal=df['magnitude'].values

dt = 1e-4         # given in seconds
fs = 1 / dt       # 10,000 Hz sampling rate

# FFT computation
n = len(signal)
fourier = fft(signal)
fourier_mag = np.abs(fourier[:n//2]) / (n/2)  # magnitude for positive half
freqs = np.fft.fftfreq(n, d=dt)[:n//2]        # frequency axis in Hz

# Ignore DC (0 Hz) and find dominant component
idx = np.argmax(fourier_mag[1:]) + 1
dominant_freq = freqs[idx]
print(f"Dominant frequency component = {dominant_freq:.2f} Hz")

# Plot time domain
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(time, signal, color='blue', alpha=0.7)
plt.xlabel("Time (s)")
plt.ylabel("Magnitude (V)")
plt.title("Time-Domain Signal")

# Plot frequency domain
plt.subplot(1,2,2)
plt.plot(freqs, fourier_mag, color='purple')
plt.axvline(dominant_freq, color='r', linestyle='--',
            label=f"Dominant = {dominant_freq:.2f} Hz")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.title("Frequency-Domain Spectrum (FFT)")
plt.xlim(0, 500)  # adjust depending on expected range
plt.legend()
plt.tight_layout()
plt.show()
