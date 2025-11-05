import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pywt
from scipy.signal import savgol_filter

def wavelet_denoising(signal,wavelet="db5",level=6): # CHANGE(VARY) WAVELET,LEVEL

    coeffs=pywt.wavedec(signal, wavelet, level=level)
    
    sigma=np.median(np.abs(coeffs[-level]))/0.6745
    threshold=sigma * np.sqrt(2* np.log(len(signal)))

    coeffs_thresholded = [coeffs[0]]  

    for i in range(1,len(coeffs)):
        coeffs_thresholded.append(pywt.threshold(coeffs[i],threshold,mode='soft'))

    denoised=pywt.waverec(coeffs_thresholded, wavelet)    

    if len(denoised) != len(signal):
        denoised = denoised[:len(signal)] if len(denoised) > len(signal) else np.pad(
        denoised, (0, len(signal) - len(denoised)), 'edge')
    return denoised    

