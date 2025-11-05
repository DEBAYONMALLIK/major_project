import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft
import pandas as pd

# -------------------------------
# FILE LIST (input freq in Hz)
# -------------------------------
file_list = [
    ("major_project/read_multi/F0010_5CH1.CSV", 10.5),
    ("major_project/read_multi/F0010CH1.CSV", 10),
    ("major_project/read_multi/F0011_5CH1.CSV", 11.5),
    ("major_project/read_multi/F0011CH1.CSV", 11),
    ("major_project/read_multi/F0012_5CH1.CSV", 12.5),
    ("major_project/read_multi/F0012CH1.CSV", 12),
    ("major_project/read_multi/F0013_5CH1.CSV", 13.5),
    ("major_project/read_multi/F0013CH1.CSV", 13),
    ("major_project/read_multi/F0020_5CH1.CSV", 20.5),
    ("major_project/read_multi/F0020CH1.CSV", 20),
    ("major_project/read_multi/F0021_5CH1.CSV", 21.5),
    ("major_project/read_multi/F0021CH1.CSV", 21),
    ("major_project/read_multi/F0022_5CH1.CSV", 22.5),
    ("major_project/read_multi/F0022CH1.CSV", 22),
    ("major_project/read_multi/F0023_5CH1.CSV", 23.5),
    ("major_project/read_multi/F0023CH1.CSV", 23),
    ("major_project/read_multi/F0024_5CH1.CSV", 24.5),
    ("major_project/read_multi/F0024CH1.CSV", 24),
    ("major_project/read_multi/F0025_5CH1.CSV", 25.5),
    ("major_project/read_multi/F0025CH1.CSV", 25),
    ("major_project/read_multi/F0026_5CH1.CSV", 26.5),
    ("major_project/read_multi/F0026CH1.CSV", 26),
    ("major_project/read_multi/F0027_5CH1.CSV", 27.5),
    ("major_project/read_multi/F0027CH1.CSV", 27),
    ("major_project/read_multi/F0028_5CH1.CSV", 28.5),
    ("major_project/read_multi/F0028CH1.CSV", 28),
    ("major_project/read_multi/F0029_5CH1.CSV", 29.5),
    ("major_project/read_multi/F0029CH1.CSV", 29),
    ("major_project/read_multi/F0030_5CH1.CSV", 30.5),
    ("major_project/read_multi/F0030CH1.CSV", 30),
    ("major_project/read_multi/F0031_5CH1.CSV", 31.5),
    ("major_project/read_multi/F0031CH1.CSV", 31),
    ("major_project/read_multi/F0032_5CH1.CSV", 32.5),
    ("major_project/read_multi/F0032CH1.CSV", 32),
    ("major_project/read_multi/F0033_5CH1.CSV", 33.5),
    ("major_project/read_multi/F0033CH1.CSV", 33),
    ("major_project/read_multi/F0034_5CH1.CSV", 34.5),
    ("major_project/read_multi/F0034CH1.CSV", 34),
    ("major_project/read_multi/F0035_5CH1.CSV", 35.5),
    ("major_project/read_multi/F0035CH1.CSV", 35),
    ("major_project/read_multi/F0036_5CH1.CSV", 36.5),
    ("major_project/read_multi/F0036CH1.CSV", 36),
    ("major_project/read_multi/F0037_5CH1.CSV", 37.5),
    ("major_project/read_multi/F0037CH1.CSV", 37),
    ("major_project/read_multi/F0038_5CH1.CSV", 38.5),
    ("major_project/read_multi/F0038CH1.CSV", 38),
    ("major_project/read_multi/F0039_5CH1.CSV", 39.5),
    ("major_project/read_multi/F0039CH1.CSV", 39),
    ("major_project/read_multi/F0040CH1.CSV", 40),
    ("major_project/read_multi/F0050CH1.CSV", 50),
]

# -------------------------------
# PARAMETERS
# -------------------------------
dt = 1e-4        # Sampling interval (s)
fs = 1 / dt      # Sampling rate (Hz)
detected_freqs = []
input_freqs = []

# -------------------------------
# PROCESS EACH FILE
# -------------------------------
for filename, input_freq in file_list:
    try:
        print(f"Processing: {filename}  (Input: {input_freq} Hz)")

        # Read CSV
        df = pd.read_csv(filename, header=None, sep=',', usecols=[3, 4], names=['time', 'magnitude'])
        time = df['time'].values
        signal = df['magnitude'].values

        # Remove NaN or invalid values
        signal = np.nan_to_num(signal)
        signal = signal - np.mean(signal)  # Detrend

        # FFT
        n = len(signal)
        fourier = fft(signal)
        fourier_mag = np.abs(fourier[:n//2]) / (n/2)
        freqs = np.fft.fftfreq(n, d=dt)[:n//2]

        # Ignore DC (0 Hz)
        idx = np.argmax(fourier_mag[1:]) + 1
        dominant_freq = freqs[idx]

        detected_freqs.append(dominant_freq)
        input_freqs.append(input_freq)

        print(f"  → Dominant frequency = {dominant_freq:.2f} Hz")

        # Optional plot per file (uncomment if needed)
        # plt.figure(figsize=(10,4))
        # plt.plot(freqs, fourier_mag)
        # plt.axvline(dominant_freq, color='r', linestyle='--', label=f'{dominant_freq:.2f} Hz')
        # plt.title(f'{filename} (Input {input_freq} Hz)')
        # plt.xlabel("Frequency (Hz)")
        # plt.ylabel("Magnitude")
        # plt.xlim(0, 200)
        # plt.legend()
        # plt.tight_layout()
        # plt.show()

    except Exception as e:
        print(f"Error processing {filename}: {e}")

# -------------------------------
# PLOT FINAL RESULT
# -------------------------------
plt.figure(figsize=(8,6))
plt.scatter(input_freqs, detected_freqs, color='purple', s=100, edgecolor='black', alpha=0.8)
plt.xlabel("Input Frequency (Hz)")
plt.ylabel("Detected Dominant Frequency (Hz)")
plt.title("Input vs Detected Dominant Frequency (FFT Analysis)")
plt.grid(True, alpha=0.3)

# Add regression line
from sklearn.linear_model import LinearRegression
X = np.array(input_freqs).reshape(-1, 1)
y = np.array(detected_freqs)
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)
plt.plot(input_freqs, y_pred, 'r--', label=f"Fit: y={model.coef_[0]:.3f}x+{model.intercept_:.3f}")
plt.legend()

plt.tight_layout()
plt.show()

# -------------------------------
# PRINT SUMMARY TABLE
# -------------------------------
print("\nSummary:")
print(f"{'File':45s} {'Input(Hz)':>10s} {'Detected(Hz)':>15s}")
print("-"*70)
for (fname, f_in), f_out in zip(file_list, detected_freqs):
    print(f"{fname:45s} {f_in:10.2f} {f_out:15.3f}")
