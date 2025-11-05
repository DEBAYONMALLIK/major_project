import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pywt
from scipy.signal import savgol_filter

from wavelet_func import wavelet_denoising

def read_oscilloscope_data(filename):
    df=pd.read_csv(filename,header=None,sep=',',
                   usecols=[3,4],
                   names=['time','magnitude'])
    return df["time"].values,df["magnitude"].values





filename = "major_project/wavelt_denoising/F0050CH1.CSV"
time, magnitude = read_oscilloscope_data(filename)


denoised_magnitude = wavelet_denoising(magnitude)
plt.figure(figsize=(12, 8))

# plot of original signal and wavelet denoised signal
plt.plot(time, magnitude, 'b-', linewidth=1, alpha=0.7, label='Original Signal')
plt.plot(time, denoised_magnitude, 'r-', linewidth=1.5, label='Wavelet Denoised')

plt.xlabel('Time (s)')
plt.ylabel('Magnitude (V)')
plt.title('Signal with Wavelet Denoising')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"Original signal STD: {np.std(magnitude):.6f}")
print(f"Denoised signal STD: {np.std(denoised_magnitude):.6f}")
print(f"Noise reduction: {((np.std(magnitude) - np.std(denoised_magnitude)) / np.std(magnitude) * 100):.1f}%")



