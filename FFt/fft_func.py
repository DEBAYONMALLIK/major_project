
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

def simple_fft_analysis(signal, time, num_harmonics=31):
    """
    Simple FFT analysis focusing on first N harmonics
    """
    dt = np.mean(np.diff(time))
    fs = 1 / dt
    
    n = len(signal)
    fft_vals = fft(signal)
    freqs = fftfreq(n, dt)
    
    # Positive frequencies only
    pos_mask = freqs > 0
    freqs_pos = freqs[pos_mask]
    fft_mag = np.abs(fft_vals[pos_mask]) / n * 2
    
    # Find fundamental frequency (simplified)
    fundamental_idx = np.argmax(fft_mag[(freqs_pos > 10) & (freqs_pos < 1000)])
    fundamental_freq = freqs_pos[(freqs_pos > 10) & (freqs_pos < 1000)][fundamental_idx]
    
    # Extract harmonics
    harmonic_freqs = []
    harmonic_mags = []
    
    for i in range(1, num_harmonics + 1):
        harmonic_freq = fundamental_freq * i
        idx = np.argmin(np.abs(freqs_pos - harmonic_freq))
        harmonic_freqs.append(harmonic_freq)
        harmonic_mags.append(fft_mag[idx])
    
    # # Plot
    # plt.figure(figsize=(12, 6))
    # plt.plot(freqs_pos, fft_mag, 'b-', alpha=0.7, label='FFT Spectrum')
    # plt.plot(harmonic_freqs, harmonic_mags, 'ro', label=f'First {num_harmonics} Harmonics')
    # plt.xlabel('Frequency (Hz)')
    # plt.ylabel('Magnitude')
    # plt.title(f'FFT Analysis - First {num_harmonics} Harmonics')
    # plt.legend()
    # plt.grid(True, alpha=0.3)
    # plt.xlim(0, min(5000, harmonic_freqs[-1] * 1.1))
    # plt.tight_layout()
    # plt.show()
    
    return harmonic_freqs, harmonic_mags, fundamental_freq

def get_harmonic_magnitudes(signal, time, num_harmonics=31):
    """
    Returns harmonic magnitudes
    """
    dt = np.mean(np.diff(time))
    print(dt)
    n = len(signal)
    fft_vals = fft(signal)
 
    freqs = fftfreq(n, dt)
    #nyquiest frequincy

    
    # Positive frequencies only
    pos_mask = freqs > 0
    freqs_pos = freqs[pos_mask]
    fft_mag = np.abs(fft_vals[pos_mask]) / n * 2
    
    # Find fundamental frequency
    fundamental_idx = np.argmax(fft_mag[(freqs_pos > 10) & (freqs_pos < 1000)])
    fundamental_freq = freqs_pos[(freqs_pos > 10) & (freqs_pos < 1000)][fundamental_idx]
    
    # Extract harmonic magnitudes 
    harmonic_mags = []
    for i in range(1, num_harmonics + 1):
        harmonic_freq = fundamental_freq * i
        idx = np.argmin(np.abs(freqs_pos - harmonic_freq))
        harmonic_mags.append(fft_mag[idx])
    
    return harmonic_mags, fundamental_freq

def get_harmonic_magnitudes_new(signal, time, num_harmonics=31):
    """
    Returns harmonic magnitudes
    """
    dt = np.mean(np.diff(time))
    n = len(signal)
    
    if n == 0 or dt == 0:
        print("Warning: Empty signal or time data.")
        return [], 0
        
    fft_vals = fft(signal)
    freqs = fftfreq(n, dt)
    
    # Positive frequencies only
    pos_mask = freqs > 0
    freqs_pos = freqs[pos_mask]
    fft_mag = np.abs(fft_vals[pos_mask]) / n * 2
    
    if len(freqs_pos) == 0:
        print("Warning: No positive frequencies found.")
        return [], 0

    # --- SAFER FUNDAMENTAL FREQUENCY DETECTION ---
    
    # Define a minimum frequency to search from. 
    # This filters out DC (0 Hz) and very low-frequency drift. 
    # 5 Hz or 10 Hz is a common choice.
    MIN_FREQ_CUTOFF = 10 
    
    search_mask = (freqs_pos > MIN_FREQ_CUTOFF)
    
    # Check if we have any frequencies left to search
    if not np.any(search_mask):
        print(f"Warning: No frequencies found above {MIN_FREQ_CUTOFF} Hz.")
        # As a fallback, just take the first frequency
        fundamental_freq = freqs_pos[0]
    else:
        # Apply the mask
        search_freqs = freqs_pos[search_mask]
        search_mags = fft_mag[search_mask]
        
        # Find the index of the max peak *in the search area*
        fundamental_idx_in_search = np.argmax(search_mags)
        
        # Get the corresponding frequency
        fundamental_freq = search_freqs[fundamental_idx_in_search]
    
    # --- END OF CHANGES ---

    # Extract harmonic magnitudes 
    harmonic_mags = []
    for i in range(1, num_harmonics + 1):
        harmonic_freq = fundamental_freq * i
        
        # Find the *closest* frequency bin in our FFT
        idx = np.argmin(np.abs(freqs_pos - harmonic_freq))
        
        # Check if the closest bin is "close enough"
        # (e.g., within half a bin width of the target)
        if i == 1 or np.abs(freqs_pos[idx] - harmonic_freq) < (fundamental_freq / 2.0):
             harmonic_mags.append(fft_mag[idx])
        else:
            # If the harmonic is missing or too far from a bin, append 0
             harmonic_mags.append(0.0)
    
    return harmonic_mags, fundamental_freq

