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


# Main code
filename = "major_project/Savitzky_Gola_denoise/F0050CH1.CSV"  
time, magnitude = read_oscilloscope_data(filename)

print(f"Loaded {len(time)} data points")
print(f"Sampling rate: {1/np.mean(np.diff(time)):.1f} Hz")
print(f"Original signal STD: {np.std(magnitude):.6f} V")




# --- Define parameters to test ---
# (polynomial order, window length)

# vary polynominal and window size
# p(y) = a0+xa1+x^2a2+.....

poly=4
window=303

plt.figure(figsize=(15, 8))
plt.plot(time, magnitude, 'k-', alpha=0.3, label='Original Signal')


if window <= poly:
    print(f"Skipping params poly={poly}, window={window} (window must be > poly)")
    
    # Apply the filter
denoised_savgol = savgol_denoising(magnitude, window_length=window, polyorder=poly)
    
    # Plot the result
plt.plot(time, denoised_savgol, linewidth=2, label=f'Poly={poly}, Window={window}')

plt.title('Comparing Savitzky-Golay Parameters')
plt.xlabel('Time (s)')
plt.ylabel('Magnitude (V)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()