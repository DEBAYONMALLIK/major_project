import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pywt
from scipy.signal import savgol_filter

def read_oscilloscope_data(filename):
    df=pd.read_csv(filename,header=None,sep=',',
                   usecols=[3,4],
                   names=['time','magnitude'])
    return df["time"].values,df["magnitude"].values



def wavelet_denoising(signal,wavelet="db4",level=4): # CHANGE(VARY) WAVELET,LEVEL

    coeffs=pywt.wavedec(signal, wavelet, level=level)
    
    sigma=np.median(np.abs(coeffs[-level]))/0.6745
    threshold=sigma * np.sqrt(2* np.log(len(signal)))

    coeffs_thresholded = [coeffs[0]]  

    for i in range(1,len(coeffs)):
        coeffs_thresholded.append(pywt.threshold(coeffs[i],threshold,mode='soft'))

    denoised=pywt.waverec(coeffs_thresholded, wavelet)    

    if len(denoised) != len(signal):
        denoised = denoised[:len(signal)] if len(denoised) > len(signal) else np.pad(
        denoised, (0, len(signal) - len(denoised)), 'edge')
    return denoised    



filename = "major_project/wavelt_denoising/unhealthy_1.CSV"  
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