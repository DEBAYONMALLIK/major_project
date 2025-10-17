import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
def read_oscilloscope_data(filename):
    """
    Reads time and magnitude data from a CSV file.
    Assumes the file is comma-separated with time in the 4th column (index 3)
    and magnitude in the 5th column (index 4), with no header.
    """
    df = pd.read_csv(filename, header=None, sep=',',
                     usecols=[3, 4],
                     names=['time', 'magnitude'])
    return df['time'].values, df['magnitude'].values

def plot_rolling_skewness(filename, window_size=200):
    """
    Reads data, calculates rolling skewness, and plots both the original signal
    and its rolling skewness.

    Args:
        filename (str): The name of the CSV file containing the oscilloscope data.
        window_size (int): The number of data points to include in each rolling window.
    """
    # 1. Read the data using your function
    time, magnitude = read_oscilloscope_data(filename)
    
    # 2. Convert data to a pandas Series for rolling calculations
    data_series = pd.Series(magnitude, index=time)
    
    # 3. Calculate the rolling skewness. A minimum of 3 periods is required.
    # 'min_periods' ensures the calculation doesn't start until enough data is available.
    rolling_skew = data_series.rolling(window=window_size, min_periods=3).skew()
    
    # 4. Create the plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Plot the original signal
    ax1.plot(time, magnitude, label='Original Signal', color='blue', alpha=0.6)
    ax1.set_title(f'Original Oscilloscope Signal')
    ax1.set_ylabel('Magnitude')
    ax1.grid(True)
    
    # Plot the rolling skewness
    ax2.plot(rolling_skew.index, rolling_skew.values, label='Rolling Skewness', color='red')
    ax2.set_title(f'Rolling Skewness (Window Size: {window_size})')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Skewness')
    ax2.grid(True)
    
    # Add a horizontal line at y=0 for reference, indicating perfect symmetry
    ax2.axhline(0, color='gray', linestyle='--', linewidth=1)
    
    # Adjust layout and show the plots
    plt.tight_layout()
    plt.show()

# --- Example Usage ---
# Create a dummy CSV file for demonstration.
# This simulates a noisy sine wave being read by your function.
# Replace this with your actual CSV file name.
dummy_filename = "unhealthy_1.CSV"
if not os.path.exists(dummy_filename):
    time_points = np.linspace(0, 10, 2000)
    # Generate a noisy sine wave
    noisy_magnitude = np.sin(2 * np.pi * time_points) + np.random.normal(0, 0.5, 2000)
    dummy_df = pd.DataFrame({'col1': 0, 'col2': 0, 'col3': 0, 'time': time_points, 'magnitude': noisy_magnitude})
    dummy_df.to_csv(dummy_filename, header=False, index=False)

# Call the function with your data file
# Make sure to replace "your_data.csv" with the actual file name.
# You can also adjust the window_size to see how it affects the plot.
plot_rolling_skewness(dummy_filename, window_size=100)
