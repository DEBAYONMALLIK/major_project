import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft
import pywt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# -------------------------
# Wavelet denoising (same as yours)
# -------------------------
def wavelet_denoising(signal, wavelet="db4", level=6):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    # estimate noise sigma from last detail coeffs (robust)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745 if len(coeffs) > 1 else np.std(signal)
    threshold = sigma * np.sqrt(2 * np.log(len(signal)))
    coeffs_thresholded = [coeffs[0]]
    for i in range(1, len(coeffs)):
        coeffs_thresholded.append(pywt.threshold(coeffs[i], threshold, mode='soft'))
    denoised = pywt.waverec(coeffs_thresholded, wavelet)
    if len(denoised) != len(signal):
        denoised = denoised[:len(signal)] if len(denoised) > len(signal) else np.pad(
            denoised, (0, len(signal) - len(denoised)), 'edge')
    return denoised

# -------------------------
# Main analysis
# -------------------------
def simple_analysis_convert_to_hz(file_list,
                                 wavelet="db5",
                                 wavelet_level=6,
                                 min_search_hz=1.0,
                                 max_search_hz=None):
    """
    For each (filename, input_freq):
      - read time & magnitude
      - denoise (wavelet)
      - compute FFT and frequency axis (Hz)
      - find dominant frequency (Hz) between min_search_hz and max_search_hz (if provided)
      - compute harmonic ratio = detected_freq / input_freq
    Returns lists: input_freqs, detected_freqs_hz, harmonic_ratios, harmonic_rounded
    """
    input_frequencies = []
    detected_freqs_hz = []
    harmonic_ratios = []
    harmonic_rounded = []

    for filename, input_freq in file_list:
        try:
            print(f"Processing: {filename} (Input: {input_freq} Hz)")
            data = pd.read_csv(filename, skiprows=17, header=None, usecols=[3,4])
            data.columns = ['time', 'magnitude']

            t = data['time'].values.astype(float)
            x = data['magnitude'].values.astype(float)

            # ensure monotonic time; otherwise construct uniform time vector
            if len(t) < 2 or not np.all(np.diff(t) > 0):
                dt_est = np.median(np.diff(t)) if len(t)>1 else 1.0
                t = np.arange(len(x)) * dt_est

            dt = np.median(np.diff(t))
            if dt <= 0:
                raise ValueError("Non-positive time step detected.")
            fs = 1.0 / dt
            nyquist = fs / 2.0

            # limit search band
            if max_search_hz is None:
                search_high = nyquist
            else:
                search_high = min(max_search_hz, nyquist)

            # preprocess
            x = np.nan_to_num(x)
            x = x - np.mean(x)

            # wavelet denoise
            denoised = wavelet_denoising(x, wavelet=wavelet, level=wavelet_level)

            # FFT (use positive half only)
            n = len(denoised)
            fourier = fft(denoised)
            half = n // 2
            fourier_pos = fourier[:half]
            freqs = np.fft.fftfreq(n, d=dt)[:half]  # frequency axis for positive half

            # magnitude and normalization (optional)
            fourier_mag = np.abs(fourier_pos) / (n / 2)

            # choose search indices between min_search_hz and search_high
            idx_min = np.searchsorted(freqs, min_search_hz, side='left')
            idx_max = np.searchsorted(freqs, search_high, side='right') - 1
            idx_min = max(idx_min, 1)  # skip DC
            idx_max = max(idx_max, idx_min)

            # find dominant index in search range
            # if search range too small fallback to global argmax (excluding DC)
            if idx_max <= idx_min:
                idx_global = np.argmax(fourier_mag[1:]) + 1
            else:
                local_slice = fourier_mag[idx_min:idx_max+1]
                if np.all(local_slice == 0):
                    idx_global = np.argmax(fourier_mag[1:]) + 1
                else:
                    idx_rel = np.argmax(local_slice)
                    idx_global = idx_min + idx_rel

            detected_freq = freqs[idx_global]
            harmonic_ratio = detected_freq / input_freq if input_freq != 0 else 0.0
            harmonic_round = int(np.round(harmonic_ratio)) if harmonic_ratio > 0.2 else 0

            # store results
            input_frequencies.append(input_freq)
            detected_freqs_hz.append(detected_freq)
            harmonic_ratios.append(harmonic_ratio)
            harmonic_rounded.append(harmonic_round)

            print(f"  -> detected frequency: {detected_freq:.4f} Hz | ratio: {harmonic_ratio:.4f} | rounded: {harmonic_round}")

            # optional quick plot per-file (comment/uncomment as needed)
            plt.figure(figsize=(10,3))
            plt.subplot(1,2,1)
            plt.plot(t, x, alpha=0.6, label='raw (detrended)')
            plt.plot(t, denoised, linewidth=1.2, label='denoised')
            plt.title(f"Time domain - {input_freq} Hz")
            plt.xlabel("Time (s)")
            plt.legend()
            plt.subplot(1,2,2)
            plt.plot(freqs, fourier_mag, label='FFT mag')
            plt.axvline(detected_freq, color='r', linestyle='--', label=f'detected {detected_freq:.3f} Hz')
            plt.xlim(0, min(search_high, max(60, detected_freq*3)))
            plt.xlabel("Frequency (Hz)")
            plt.legend()
            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"Error processing {filename}: {e}")
            input_frequencies.append(input_freq)
            detected_freqs_hz.append(0.0)
            harmonic_ratios.append(0.0)
            harmonic_rounded.append(0)

    return input_frequencies, detected_freqs_hz, harmonic_ratios, harmonic_rounded

# -------------------------
# Your file list (example)
# -------------------------
file_list = [
    ("major_project/read_multi/F0010_5CH1.CSV", 10.5),
    ("major_project/read_multi/F0010CH1.CSV", 10),
    ("major_project/read_multi/F0011_5CH1.CSV", 11.5),
    ("major_project/read_multi/F0011CH1.CSV", 11),
    ("major_project/read_multi/F0012_5CH1.CSV", 12.5),
    ("major_project/read_multi/F0012CH1.CSV", 12),
    ("major_project/read_multi/F0013_5CH1.CSV", 13.5),
    ("major_project/read_multi/F0013CH1.CSV", 13),
    ("major_project/read_multi/F0020CH1.CSV", 20),
    ("major_project/read_multi/F0021_CH1.CSV", 21),
    ("major_project/read_multi/F0022_5CH1.CSV", 22.5),
    ("major_project/read_multi/F0023_5CH1.CSV", 23.5),
    ("major_project/read_multi/F0037_5CH1.CSV", 37.5),
    ("major_project/read_multi/F0039_5CH1.CSV", 39.5),
    ("major_project/read_multi/F0040CH1.CSV", 40),
    ("major_project/read_multi/F0050CH1.CSV", 50),
]

# -------------------------
# Run analysis & plot summary
# -------------------------
if __name__ == "__main__":
    in_freqs, detected_hz, ratio, rounded = simple_analysis_convert_to_hz(
        file_list,
        wavelet="db5",
        wavelet_level=6,
        min_search_hz=1.0,
        max_search_hz=60.0  # limit search to 60 Hz if you like
    )

    # Convert to numpy arrays
    in_freqs = np.array(in_freqs)
    detected_hz = np.array(detected_hz)
    ratio = np.array(ratio)
    rounded = np.array(rounded)

    # 1) Plot: Input Frequency vs Detected Frequency (Hz)
    plt.figure(figsize=(7,5))
    plt.scatter(in_freqs, detected_hz, s=90, edgecolor='k', alpha=0.8)
    plt.xlabel("Input Frequency (Hz)")
    plt.ylabel("Detected Dominant Frequency (Hz)")
    plt.title("Input Frequency vs Detected Frequency (Hz)")
    # linear regression on Hz (meaningful)
    valid = detected_hz > 0
    if valid.sum() > 1:
        lr = LinearRegression().fit(in_freqs[valid].reshape(-1,1), detected_hz[valid])
        pred = lr.predict(in_freqs[valid].reshape(-1,1))
        plt.plot(in_freqs[valid], pred, 'r-', label=f'fit: y={lr.coef_[0]:.3f}x+{lr.intercept_:.3f}')
        plt.legend()
        print("\nRegression (Input Hz -> Detected Hz):")
        print(f"  slope = {lr.coef_[0]:.4f}, intercept = {lr.intercept_:.4f}")
        print(f"  MAE = {mean_absolute_error(detected_hz[valid], pred):.4f}, R2 = {r2_score(detected_hz[valid], pred):.4f}")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.show()

    # 2) Plot: Input Frequency vs Harmonic Ratio (float)
    plt.figure(figsize=(7,5))
    plt.scatter(in_freqs, ratio, s=90, edgecolor='k', alpha=0.8)
    plt.xlabel("Input Frequency (Hz)")
    plt.ylabel("Harmonic Ratio (detected_freq / input_freq)")
    plt.title("Input Frequency vs Harmonic Ratio (float)")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.show()

    # 3) Plot: Input Frequency vs Rounded Harmonic (integer)
    plt.figure(figsize=(7,5))
    plt.scatter(in_freqs, rounded, s=90, edgecolor='k', alpha=0.8)
    plt.xlabel("Input Frequency (Hz)")
    plt.ylabel("Rounded Harmonic (integer)")
    plt.title("Input Frequency vs Rounded Harmonic")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.show()

    # Print a neat table
    print("\nSummary (InputHz, DetectedHz, Ratio, RoundedHarmonic):")
    for a,b,c,d in zip(in_freqs, detected_hz, ratio, rounded):
        print(f"{a:6.2f} Hz  -> {b:7.3f} Hz  | ratio: {c:5.3f} | rounded: {d:d}")
