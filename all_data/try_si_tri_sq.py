import os
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft
import pandas as pd
from sklearn.linear_model import LinearRegression

# ------------------------------------------------
# Auto-generate file_list from three folder trees
# ------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
FOLDERS_TO_SCAN = [  "TRIANGULAR"
                    , "SQUARE"
                      ,"SINUSOIDAL"
                   ]

def parse_freq_from_folder(name):
    """
    Parse the leading number from folder name (supports '10', '10.5', '10_5', etc.).
    """
    m = re.search(r"([0-9]+(?:[._][0-9]+)?)", name)
    if not m:
        return None
    s = m.group(1).replace('_', '.')
    try:
        return float(s)
    except:
        return None

file_list = []  # [(path, freq, waveform_type)]
for folder_name in FOLDERS_TO_SCAN:
    folder_path = os.path.join(ROOT_DIR, folder_name)
    if not os.path.isdir(folder_path):
        print(f"⚠️  Warning: folder not found (skipping): {folder_path}")
        continue

    # Walk subfolders inside folder_path
    for entry in sorted(os.listdir(folder_path)):
        subfolder = os.path.join(folder_path, entry)
        if not os.path.isdir(subfolder):
            continue
        freq = parse_freq_from_folder(entry)
        if freq is None:
            continue
        for fname in sorted(os.listdir(subfolder)):
            if fname.lower().endswith('.csv'):
                fullpath = os.path.join(subfolder, fname)
                file_list.append((fullpath, freq, folder_name))

print(f"✅ Discovered {len(file_list)} CSV files across {FOLDERS_TO_SCAN}.")

# -------------------------------
# PARAMETERS
# -------------------------------
dt = 1e-4  # Sampling interval (s)
fs = 1 / dt
detected_freqs = []
input_freqs = []
waveform_types = []

# -------------------------------
# PROCESS EACH FILE (same logic)
# -------------------------------
for filename, input_freq, wtype in file_list:
    try:
        print(f"Processing: {filename}  (Input: {input_freq} Hz, Type: {wtype})")

        df = pd.read_csv(filename, header=None, sep=',', usecols=[3, 4], names=['time', 'magnitude'])
        signal = np.nan_to_num(df['magnitude'].values)
        signal = signal - np.mean(signal)

        n = len(signal)
        fourier = fft(signal)
        fourier_mag = np.abs(fourier[:n//2]) / (n/2)
        freqs = np.fft.fftfreq(n, d=dt)[:n//2]

        idx = np.argmax(fourier_mag[1:]) + 1
        dominant_freq = freqs[idx]

        detected_freqs.append(dominant_freq)
        input_freqs.append(input_freq)
        waveform_types.append(wtype)

        print(f"  → Dominant frequency = {dominant_freq:.2f} Hz")

    except Exception as e:
        print(f"❌ Error processing {filename}: {e}")

# -------------------------------
# PLOT FINAL RESULT (with colors)
# -------------------------------
plt.figure(figsize=(9, 7))

# Assign color for each waveform type
color_map = {
    "SINUSOIDAL": "blue",
    "SQUARE": "green",
    "TRIANGULAR": "orange"
}

for wtype in FOLDERS_TO_SCAN:
    xs = [f for (f, t) in zip(input_freqs, waveform_types) if t == wtype]
    ys = [f for (f, t) in zip(detected_freqs, waveform_types) if t == wtype]
    if len(xs) > 0:
        plt.scatter(xs, ys,
                    color=color_map.get(wtype, "gray"),
                    s=100,
                    alpha=0.8,
                    edgecolor='black',
                    label=wtype)

plt.xlabel("Input Frequency (Hz)")
plt.ylabel("Detected Dominant Frequency (Hz)")
plt.title("Input vs Detected Dominant Frequency (FFT Analysis)")
plt.grid(True, alpha=0.3)

# Add regression line for all combined data
if len(input_freqs) > 1:
    X = np.array(input_freqs).reshape(-1, 1)
    y = np.array(detected_freqs)
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    plt.plot(input_freqs, y_pred, 'r--',
             label=f"Fit: y={model.coef_[0]:.3f}x + {model.intercept_:.3f}")

plt.legend()
plt.tight_layout()
plt.show()

# -------------------------------
# PRINT SUMMARY TABLE
# -------------------------------
print("\nSummary:")
print(f"{'File':100s} {'Waveform':>12s} {'Input(Hz)':>10s} {'Detected(Hz)':>15s}")
print("-" * 140)
for (fname, f_in, wtype), f_out in zip(file_list, detected_freqs):
    print(f"{fname:100s} {wtype:>12s} {f_in:10.2f} {f_out:15.3f}")
