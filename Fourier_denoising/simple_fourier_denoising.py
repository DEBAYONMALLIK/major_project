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

def simple_fourier_denoising(signal, threshold_ratio=0.3, freq_cutoff_ratio=0.1):
    """
    Fourier Transform Thresholding Denoising
    
    Parameters:
    - threshold_ratio: Ratio of max amplitude to use as threshold (0.01-0.1)
    - freq_cutoff_ratio: Ratio of Nyquist frequency to cut off high frequencies (0.05-0.3)
    """
    # Perform FFT
    fft_signal = np.fft.fft(signal)
    frequencies = np.fft.fftfreq(len(signal))
    
    # Calculate amplitudes
    amplitudes = np.abs(fft_signal)
    
    # Method 1: Amplitude thresholding
    threshold = threshold_ratio * np.max(amplitudes)
    
    # Create mask for frequencies to keep
    mask_amplitude = amplitudes > threshold
    
    # Method 2: Frequency cutoff (remove very high frequencies)
    nyquist_idx = int(len(signal) * freq_cutoff_ratio)
    mask_frequency = np.abs(frequencies) < frequencies[nyquist_idx]
    
    # Combine both methods
    combined_mask = mask_amplitude & mask_frequency
    
    # Apply the mask
    fft_denoised = fft_signal.copy()
    fft_denoised[~combined_mask] = 0
    
    # Inverse FFT
    denoised_signal = np.real(np.fft.ifft(fft_denoised))
    
    return denoised_signal, fft_signal, fft_denoised, frequencies, combined_mask







filename ="major_project/Fourier_denoising/F0050CH1.CSV"
time, magnitude = read_oscilloscope_data(filename)

denoised_magnitude, fft_orig, fft_denoised, freqs, mask = simple_fourier_denoising(
    magnitude, 
    threshold_ratio=0.05, 
    freq_cutoff_ratio=0.1
)

plt.figure(figsize=(12, 8))

# plot of original signal and wavelet denoised signal
plt.plot(time, magnitude, 'b-', linewidth=1, alpha=0.7, label='Original Signal')
plt.plot(time, denoised_magnitude, 'r-', linewidth=1.5, label='Fourier Denoised')

plt.xlabel('Time (s)')
plt.ylabel('Magnitude (V)')
plt.title('Signal with Fourier transform Denoising')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"Original signal STD: {np.std(magnitude):.6f}")
print(f"Denoised signal STD: {np.std(denoised_magnitude):.6f}")
print(f"Noise reduction: {((np.std(magnitude) - np.std(denoised_magnitude)) / np.std(magnitude) * 100):.1f}%")