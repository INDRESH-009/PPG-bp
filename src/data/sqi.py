import numpy as np
from scipy.signal import find_peaks

def basic_sqi(ppg, fs, amp_range=(0.1,3.0), flat_std_min=0.05, hr_range=(40,180)):
    amp = float(ppg.max()-ppg.min())
    if not (amp_range[0] <= amp <= amp_range[1]): return False
    if np.std(ppg) < flat_std_min: return False
    # peak distance >= 0.3 s
    peaks,_ = find_peaks(ppg, distance=int(0.3*fs))
    hr = (60.0*len(peaks))/ (len(ppg)/fs) if len(ppg)>0 else 0
    if not (hr_range[0] <= hr <= hr_range[1]): return False
    return True
