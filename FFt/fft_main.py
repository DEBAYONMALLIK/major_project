import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fft_func import get_harmonic_magnitudes
from fft_func import get_harmonic_magnitudes_new
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pywt

from wavelt_denoising.wavelet_func import wavelet_denoising


def read_oscilloscope_data(filename):
    df=pd.read_csv(filename,header=None,sep=',',
                   usecols=[3,4],
                   names=['time','magnitude'])
    return df["time"].values,df["magnitude"].values


filename = "major_project/FFt/unhealthy_1.CSV"  
time, magnitude = read_oscilloscope_data(filename)




denoised_magnitude = wavelet_denoising(magnitude)



harmonic_mags, fundamental_freq = get_harmonic_magnitudes_new(denoised_magnitude, time, num_harmonics=31)

print("\n" + "="*50)
print("PERFORMING FFT ANALYSIS")
print("="*50)
print(fundamental_freq,max(harmonic_mags))

# Now you can plot exactly like your second code:
plt.figure(figsize=(10, 6))
plt.plot(range(1, 32), harmonic_mags, 'bo-', linewidth=2, markersize=6)
plt.xlabel('Harmonic Number')
plt.ylabel('Magnitude')
plt.title('Harmonic Analysis (0-30 Harmonics)')
plt.grid(True, alpha=0.3)
plt.xticks(range(1, 32, 2))
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
bars = plt.bar(range(1, 32), harmonic_mags, alpha=0.7, color='skyblue')
plt.xlabel('Harmonic Number')
plt.ylabel('Magnitude')
plt.title('Harmonic Magnitudes Bar Chart')
plt.grid(True, alpha=0.3)

# Add values on bars
for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.4f}', ha='center', va='bottom', fontsize=8)
plt.show()