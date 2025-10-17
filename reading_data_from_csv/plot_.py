import pandas as pd
import matplotlib.pyplot as plt

def read_oscilloscope_data(filename):
   
    df = pd.read_csv(filename, header=None, sep=',',
                     usecols=[3, 4],  #  zero indexed
                     names=['time', 'magnitude'])
    return df['time'].values, df['magnitude'].values



filename = "major_project/reading_data_from_csv/unhealthy_1.CSV" 

time, magnitude = read_oscilloscope_data(filename)


plt.figure(figsize=(12, 6))
plt.plot(time, magnitude, 'g-', linewidth=1.5)
plt.xlabel('Time (s)')
plt.ylabel('Magnitude (V)')
plt.title('Magnitude vs Time')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"Loaded {len(time)} data points")
print(f"Time range: {time.min():.6f} to {time.max():.6f} s")
print(f"Magnitude range: {magnitude.min():.6f} to {magnitude.max():.6f} V")