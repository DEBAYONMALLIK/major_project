

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft
from sklearn.linear_model import LinearRegression
import pywt
from scipy.signal import savgol_filter


def savgol_denoising(signal, window_length=20, polyorder=3):
    """
    Savitzky-Golay filter denoising
    - window_length: should be odd integer, larger = more smoothing
    - polyorder: polynomial order, typically 2-4
    """
    # Ensure window_length is odd
    if window_length % 2 == 0:
        window_length += 1
    
    # Apply Savitzky-Golay filter
    denoised = savgol_filter(signal, window_length, polyorder)
    return denoised


def simple_analysis(file_list):
    """
    Follow your seniors' simple approach with wavelet denoising
    """
    harmonics = []
    input_frequencies = []
    
    for filename, input_freq in file_list:
        try:
            print(f"Processing: {filename} (Input: {input_freq}Hz)")
            
            # Read data - skip metadata rows and use correct columns
            data = pd.read_csv(filename, skiprows=17, header=None, usecols=[3, 4])
            data.columns = ['time', 'magnitude']
            
            # Apply wavelet denoising instead of rolling mean
            magnitude_data = data['magnitude'].values
            
            # Remove any NaN values before denoising
            if np.any(np.isnan(magnitude_data)):
                magnitude_data = np.nan_to_num(magnitude_data)
            poly=4
            window=303
            # Apply wavelet denoising
            denoised_signal =  savgol_denoising(magnitude_data, window_length=window, polyorder=poly)
            # Optional: Apply Savitzky-Golay filter for additional smoothing
            # if len(denoised_signal) > 11:  # Ensure enough points for Savitzky-Golay
            #     smoothed_signal = savgol_filter(denoised_signal, window_length=11, polyorder=3)
            # else:
            smoothed_signal = denoised_signal
            
            # Simple FFT like your seniors
            fourier = fft(smoothed_signal)
            fourier_mag = np.abs(fourier[0:30]) / (len(smoothed_signal) / 2)
            
            # Find predominant harmonic (2-29) like your seniors
            harmonic, content = 0, 0
            for j in range(2, 30):
                if fourier_mag[j] > content:
                    content = fourier_mag[j]
                    harmonic = j
            
            harmonics.append(harmonic)
            input_frequencies.append(input_freq)
            
            print(f"  → Input: {input_freq}Hz, Predominant Harmonic: {harmonic}")
            
            # Optional: Plot like your seniors
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            plt.plot(data['time'], data['magnitude'], color='yellow', alpha=0.7, label='Original')
            plt.plot(data['time'], smoothed_signal, color='red', linewidth=2, label='Wavelet Denoised')
            plt.title(f"Time Domain - Frequency = {input_freq} Hz")
            plt.xlabel("Time")
            plt.ylabel("Magnitude")
            plt.legend()
            
            plt.subplot(1, 2, 2)
            frequencies = np.fft.fftfreq(len(smoothed_signal), d=data['time'].iloc[1]-data['time'].iloc[0])[:30]
            plt.stem(frequencies[2:30], fourier_mag[2:30])
            plt.title(f"Frequency Domain - Harmonic: {harmonic}")
            plt.xlabel("Frequency")
            plt.ylabel("Magnitude")
            
            plt.tight_layout()
            plt.show()
                
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    
    return input_frequencies, harmonics

# YOUR FILE LIST WITH INPUT FREQUENCIES
file_list = [
    ("major_project/read_multi/F0010_5CH1.CSV", 10.5),   # 10 Hz
    ("major_project/read_multi/F0010CH1.CSV", 10),     # 10 Hz
    ("major_project/read_multi/F0011_5CH1.CSV", 11.5),   # 11 Hz
    ("major_project/read_multi/F0011CH1.CSV", 11),    # 11 Hz
    ("major_project/read_multi/F0012_5CH1.CSV", 12.5),   # 12 Hz
    ("major_project/read_multi/F0012CH1.CSV", 12),     # 12 Hz
    ("major_project/read_multi/F0013_5CH1.CSV", 13.5),   # 13 Hz
    ("major_project/read_multi/F0013CH1.CSV", 13),     # 13 Hz
    ("major_project/read_multi/F0020CH1.CSV", 20),   # 14 Hz
    ("major_project/read_multi/F0021_CH1.CSV", 21),    # 14 Hz
    ("major_project/read_multi/F0020CH1.CSV", 20),     # 20 Hz
    ("major_project/read_multi/F0022_5CH1.CSV", 22.5),   # 32 Hz
    ("major_project/read_multi/F0023_5CH1.CSV", 23.5),     # 35 Hz
    ("major_project/read_multi/F0037_5CH1.CSV", 37.5),     # 39 Hz
    ("major_project/read_multi/F0039_5CH1.CSV", 39.5),     # 40 Hz
    ("major_project/read_multi/F0040CH1.CSV", 40),     # 50 Hz
    ("major_project/read_multi/F0050CH1.CSV", 50),
]

# PROCESS FILES
print("Starting analysis with wavelet denoising...")
input_freqs, pred_harmonics = simple_analysis(file_list)

# PLOT RESULTS
plt.figure(figsize=(10, 6))
plt.scatter(input_freqs, pred_harmonics, s=100, alpha=0.7, edgecolor='black')
plt.xlabel("Input Frequency (Hz)")
plt.ylabel("Predominant Harmonic")
plt.title("Input Frequency vs Predominant Harmonic (Wavelet Denoising)")
plt.grid(True, alpha=0.3)

# LINEAR REGRESSION (like your seniors)
if len(input_freqs) > 1:
    X = np.array(input_freqs).reshape(-1, 1)
    y = np.array(pred_harmonics)
    
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    
    plt.plot(input_freqs, y_pred, 'r-', linewidth=2, 
             label=f'Linear Fit: y = {model.coef_[0]:.3f}x + {model.intercept_:.3f}')
    plt.legend()
    
    # Calculate metrics
    from sklearn.metrics import mean_absolute_error, r2_score
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    print(f"\nLinear Regression Results:")
    print(f"MAE: {mae:.3f}")
    print(f"R²: {r2:.3f}")

plt.tight_layout()
plt.show()