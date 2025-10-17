import pandas as pd
import matplotlib.pyplot as plt
import os
def read_multiple_file(direct_name):
   
    for file in direct_name:
        df = pd.read_csv(f"major_project/read_multiple_files_csv/all_files/{file}", header=None, sep=',',
                     usecols=[3, 4],  #  zero indexed
                     names=['time', 'magnitude'])
        time=df['time'].values
        magnitude=df['magnitude'].values
        plt.figure(figsize=(12, 6))
        plt.plot(time, magnitude, 'g-', linewidth=1.5)
        plt.xlabel('Time (s)')
        plt.ylabel('Magnitude (V)')
        plt.title('Magnitude vs Time')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()



a=os.listdir("major_project/read_multiple_files_csv/all_files")
read_multiple_file(a)
