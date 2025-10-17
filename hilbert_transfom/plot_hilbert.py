import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pywt
from scipy.signal import hilbert, welch
import scipy.signal as signal

def read_oscilloscope_data(filename):
    df = pd.read_csv(filename, header=None, sep=',',
                     usecols=[3, 4],
                     names=['time', 'magnitude'])
    return df['time'].values, df['magnitude'].values

def wavelet_denoising(signal_data, wavelet='db4', level=2):
    """Wavelet denoising function"""
    coeffs = pywt.wavedec(signal_data, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-level])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(signal_data)))
    
    coeffs_thresholded = [coeffs[0]]
    for i in range(1, len(coeffs)):
        coeffs_thresholded.append(pywt.threshold(coeffs[i], threshold, mode='soft'))
    
    denoised = pywt.waverec(coeffs_thresholded, wavelet)
    if len(denoised) != len(signal_data):
        denoised = denoised[:len(signal_data)] if len(denoised) > len(signal_data) else np.pad(
            denoised, (0, len(signal_data) - len(denoised)), 'edge')
    return denoised

def hilbert_analysis(signal_data, time):
    """
    Perform comprehensive Hilbert transform analysis
    Returns: envelope, instantaneous_phase, instantaneous_frequency, instantaneous_amplitude
    """
    # Apply Hilbert transform
    analytic_signal = hilbert(signal_data)
    
    # Extract components
    envelope = np.abs(analytic_signal)  # Signal envelope
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))  # Instantaneous phase
    instantaneous_frequency = (np.diff(instantaneous_phase) / 
                              (2.0 * np.pi) * (1.0 / np.diff(time)))  # Instantaneous frequency in Hz
    
    return envelope, instantaneous_phase, instantaneous_frequency, analytic_signal

# Main code
filename = "major_project/hilbert_transfom/unhealthy_1.CSV"
time, magnitude = read_oscilloscope_data(filename)

print(f"Loaded {len(time)} data points")
print(f"Sampling rate: {1/np.mean(np.diff(time)):.1f} Hz")

# Step 1: Denoise the signal
denoised_magnitude = wavelet_denoising(magnitude)

# Step 2: Apply Hilbert transform
envelope, instantaneous_phase, instantaneous_frequency, analytic_signal = hilbert_analysis(denoised_magnitude, time)

# Create comprehensive analysis plots
fig, axes = plt.subplots(2, 2, figsize=(16, 13))

# Plot 1: Original vs Denoised signal
axes[0, 0].plot(time, magnitude, 'b-', linewidth=1, alpha=0.6, label='Original')
axes[0, 0].plot(time, denoised_magnitude, 'r-', linewidth=1.5, label='Denoised')
axes[0, 0].set_title('Original vs Denoised Signal')
axes[0, 0].set_xlabel('Time (s)')
axes[0, 0].set_ylabel('Magnitude (V)')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Signal with envelope
axes[0, 1].plot(time, denoised_magnitude, 'b-', linewidth=1.5, label='Denoised Signal')
axes[0, 1].plot(time, envelope, 'r-', linewidth=2, label='Envelope')
axes[0, 1].fill_between(time, -envelope, envelope, alpha=0.2, color='red')
axes[0, 1].set_title('Signal with Hilbert Envelope')
axes[0, 1].set_xlabel('Time (s)')
axes[0, 1].set_ylabel('Magnitude (V)')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Instantaneous frequency
axes[1, 0].plot(time[1:], instantaneous_frequency, 'g-', linewidth=1.5)
axes[1, 0].set_title('Instantaneous Frequency')
axes[1, 0].set_xlabel('Time (s)')
axes[1, 0].set_ylabel('Frequency (Hz)')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_ylim(bottom=0)  # Frequency can't be negative

# Plot 4: Instantaneous phase
axes[1, 1].plot(time, instantaneous_phase, 'purple', linewidth=1.5)
axes[1, 1].set_title('Instantaneous Phase')
axes[1, 1].set_xlabel('Time (s)')
axes[1, 1].set_ylabel('Phase (radians)')
axes[1, 1].grid(True, alpha=0.3)

# Plot 5: Analytic signal in complex plane
# axes[2, 0].plot(np.real(analytic_signal), np.imag(analytic_signal), 'b-', linewidth=1, alpha=0.7)
# axes[2, 0].set_title('Analytic Signal (Complex Plane)')
# axes[2, 0].set_xlabel('Real Part')
# axes[2, 0].set_ylabel('Imaginary Part')
# axes[2, 0].grid(True, alpha=0.3)
# axes[2, 0].set_aspect('equal')

# Plot 6: Frequency distribution
frequencies, psd = welch(denoised_magnitude, fs=1/np.mean(np.diff(time)), nperseg=1024)
# axes[2, 1].semilogy(frequencies, psd, 'b-', linewidth=1.5)
# axes[2, 1].set_title('Power Spectral Density')
# axes[2, 1].set_xlabel('Frequency (Hz)')
# axes[2, 1].set_ylabel('Power Spectral Density')
# axes[2, 1].grid(True, alpha=0.3)


plt.tight_layout(pad=10.0) # Increase padding

plt.show()

# Print analysis results
print("\n" + "="*50)
print("HILBERT TRANSFORM ANALYSIS RESULTS")
print("="*50)
print(f"Envelope statistics:")
print(f"  - Mean envelope amplitude: {np.mean(envelope):.6f} V")
print(f"  - Max envelope amplitude: {np.max(envelope):.6f} V")
print(f"  - Min envelope amplitude: {np.min(envelope):.6f} V")
print(f"  - Envelope variation: {np.std(envelope):.6f} V")

print(f"\nInstantaneous frequency statistics:")
print(f"  - Mean frequency: {np.mean(instantaneous_frequency):.1f} Hz")
print(f"  - Median frequency: {np.median(instantaneous_frequency):.1f} Hz")
print(f"  - Frequency range: {np.min(instantaneous_frequency):.1f} - {np.max(instantaneous_frequency):.1f} Hz")
print(f"  - Frequency variation: {np.std(instantaneous_frequency):.1f} Hz")
