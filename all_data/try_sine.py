import os
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft
import pandas as pd

# ------------------------------------------------
# Auto-generate file_list from folder structure
# (keeps your original FFT logic unchanged)

# ------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) # get the name of parent direct(all_data)

SINUS_DIR = os.path.join(ROOT_DIR, "SINUSOIDAL")  # get the path of SINUSOIDAL direct

def parse_freq_from_folder(name):
    """
    Parse the leading number from folder name (supports '10', '10.5', '10_5').
    Returns float or None.
    """
    m = re.search(r"([0-9]+(?:[._][0-9]+)?)", name)

    if not m:
        return None
    s = m.group(1).replace('_', '.')
    try:
        return float(s)
    except:
        return None

file_list = [] # (file_location,freq)


if os.path.isdir(SINUS_DIR):
    for entry in sorted(os.listdir(SINUS_DIR)):
        subfolder = os.path.join(SINUS_DIR, entry)
        if not os.path.isdir(subfolder):
            continue
        freq = parse_freq_from_folder(entry)
        if freq is None:
            # skip folders that don't start with a number
            continue
        # collect CSV files inside the folder
        for fname in sorted(os.listdir(subfolder)):
            # consider only .csv files (your original reader uses pd.read_csv)
            if fname.lower().endswith('.csv'):
                fullpath = os.path.join(subfolder, fname)
                
                file_list.append((fullpath, freq))
else:
    raise SystemExit(f"SINUSOIDAL directory not found at expected location: {SINUS_DIR}")

print(f"Discovered {len(file_list)} CSV files across subfolders of SINUSOIDAL.")

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

        # Read CSV (keeps your original read logic)
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

        # Ignore DC (0 Hz) and converting the relative order.
         
        idx = np.argmax(fourier_mag[1:]) + 1
        dominant_freq = freqs[idx]

        detected_freqs.append(dominant_freq)
        input_freqs.append(input_freq)

        print(f"  → Dominant frequency = {dominant_freq:.2f} Hz")

    except Exception as e:
        print(f"Error processing {filename}: {e}")

# ---------------------------------
# PLOT FINAL RESULT
# ---------------------------------
plt.figure(figsize=(8,6))
plt.scatter(input_freqs, detected_freqs, color='purple', s=100, edgecolor='black', alpha=0.8)
plt.xlabel("Input Frequency (Hz)")
plt.ylabel("Detected Dominant Frequency (Hz)")
plt.title("Input vs Detected Dominant Frequency (FFT Analysis)")
plt.grid(True, alpha=0.3)

# Add regression line
from sklearn.linear_model import LinearRegression
if len(input_freqs) > 1:
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
print(f"{'File':70s} {'Input(Hz)':>10s} {'Detected(Hz)':>15s}")
print("-"*100)
for (fname, f_in), f_out in zip(file_list, detected_freqs):
    print(f"{fname:70s} {f_in:10.2f} {f_out:15.3f}")
