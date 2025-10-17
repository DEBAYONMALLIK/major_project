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

def extract_comprehensive_features(time, signal, instantaneous_frequency, instantaneous_phase, envelope):
    """Extract comprehensive features for signal classification"""
    
    # Time-domain features
    time_features = {
        'mean': np.mean(signal),
        'std': np.std(signal),
        'rms': np.sqrt(np.mean(signal**2)),
        'peak_to_peak': np.ptp(signal),
        'skewness': signal.skew() if hasattr(signal, 'skew') else pd.Series(signal).skew(),
        'kurtosis': pd.Series(signal).kurtosis(),
        'crest_factor': np.max(np.abs(signal)) / np.sqrt(np.mean(signal**2)),
        'form_factor': np.sqrt(np.mean(signal**2)) / np.mean(np.abs(signal))
    }
    
    # Frequency-domain features
    freq_features = {
        'inst_freq_mean': np.mean(instantaneous_frequency),
        'inst_freq_std': np.std(instantaneous_frequency),
        'inst_freq_median': np.median(instantaneous_frequency),
        'inst_freq_range': np.ptp(instantaneous_frequency),
        #'dominant_freq': frequencies[np.argmax(psd)] if 'frequencies' in locals() else 0,
        'frequency_bandwidth': np.std(instantaneous_frequency) / np.mean(instantaneous_frequency) if np.mean(instantaneous_frequency) > 0 else 0
    }
    
    # Phase and envelope features
    phase_features = {
        'phase_slope': np.polyfit(time[:len(instantaneous_phase)], instantaneous_phase, 1)[0],
        'phase_std': np.std(np.diff(instantaneous_phase)),
        'envelope_mean': np.mean(envelope),
        'envelope_std': np.std(envelope),
        'envelope_energy': np.sum(envelope**2),
        'modulation_index': np.std(envelope) / np.mean(envelope) if np.mean(envelope) > 0 else 0
    }
    
    # Combine all features
    all_features = {**time_features, **freq_features, **phase_features}
    return all_features







# Main code
filename = "major_project/hilbert_transfom/unhealthy_1.CSV"
time, magnitude = read_oscilloscope_data(filename)

print(f"Loaded {len(time)} data points")
print(f"Sampling rate: {1/np.mean(np.diff(time)):.1f} Hz")

# Step 1: Denoise the signal
denoised_magnitude = wavelet_denoising(magnitude)

# Step 2: Apply Hilbert transform
envelope, instantaneous_phase, instantaneous_frequency, analytic_signal = hilbert_analysis(denoised_magnitude, time)



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


# Extract features
features = extract_comprehensive_features(time, denoised_magnitude, instantaneous_frequency, instantaneous_phase, envelope)

print("EXTRACTED FEATURES FOR CLASSIFICATION:")
print("="*50)
for key, value in features.items():
    print(f"{key:25}: {value:.6f}")
   