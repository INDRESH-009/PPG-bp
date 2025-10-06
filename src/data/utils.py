import numpy as np
from scipy.signal import butter, filtfilt

def bandpass(x, fs, lo=0.5, hi=8.0, order=3):
    ny = 0.5*fs
    b,a = butter(order, [lo/ny, hi/ny], btype="band")
    return filtfilt(b,a,x).astype(np.float32)

def detrend_poly(x, order=1):
    t = np.arange(len(x))
    coefs = np.polyfit(t, x, order)
    trend = np.polyval(coefs, t)
    return (x - trend).astype(np.float32)

def normalize(x, mode="zscore"):
    if mode=="zscore":
        mu = x.mean()
        sd = x.std() + 1e-8
        return ((x-mu)/sd).astype(np.float32)
    if mode=="minmax":
        mn, mx = x.min(), x.max()
        rng = (mx-mn) if (mx>mn) else 1.0
        return ((x-mn)/rng).astype(np.float32)
    return x.astype(np.float32)
