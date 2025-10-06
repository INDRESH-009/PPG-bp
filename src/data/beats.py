import numpy as np
from scipy.signal import find_peaks

def find_ppg_peaks(x, fs, min_dist_ms=300, prominence=0.1):
    min_dist = int((min_dist_ms/1000.0)*fs)
    peaks, props = find_peaks(x, distance=max(1, min_dist), prominence=prominence)
    return peaks, props

def beat_bounds_from_peaks(peaks, L):
    # simple mid-point partitioning: boundaries halfway between consecutive peaks
    if len(peaks) < 2:
        return []
    bounds = []
    mids = ((peaks[1:] + peaks[:-1]) // 2).astype(int)
    # first beat: [start..mid0], last beat: [midN-1..end]
    start = 0
    for m in mids:
        bounds.append((start, m))
        start = m
    bounds.append((start, L))
    return bounds

def resample_to_length(x, target_len):
    xp = np.linspace(0, 1, num=len(x), dtype=np.float32)
    xq = np.interp(np.linspace(0,1,target_len, dtype=np.float32), xp, x).astype(np.float32)
    return xq

def derive_vel_acc(x, fs):
    dx = np.gradient(x)*fs
    ddx = np.gradient(dx)*fs
    return dx.astype(np.float32), ddx.astype(np.float32)

def beats_from_window(x, fs, target_len=256, max_beats=16, min_beats=6,
                      min_dist_ms=300, prominence=0.1, use_vel_acc=True):
    """
    Returns:
      beats: [B,C,T] with B<=max_beats, C=1 or 3, T=target_len
      mask:  [B] 1 for valid beats, 0 for pad
    """
    L = len(x)
    pk, _ = find_ppg_peaks(x, fs, min_dist_ms=min_dist_ms, prominence=prominence)
    bounds = beat_bounds_from_peaks(pk, L)
    if len(bounds) == 0:
        return None, None
    # assemble beats
    beat_list = []
    for (a,b) in bounds:
        seg = x[a:b]
        if len(seg) < 8:  # too short
            continue
        seg = resample_to_length(seg, target_len)
        if use_vel_acc:
            dx, ddx = derive_vel_acc(seg, fs)  # derivative on resampled seg is fine
            beat = np.stack([seg, dx, ddx], axis=0)  # [3,T]
        else:
            beat = seg[None, ...]  # [1,T]
        beat_list.append(beat.astype(np.float32))

    if len(beat_list) < (min_beats if min_beats>0 else 1):
        return None, None

    # trim/pad to max_beats
    if len(beat_list) > max_beats:
        # keep the central beats to avoid partials at edges
        start = (len(beat_list) - max_beats)//2
        beat_list = beat_list[start:start+max_beats]

    B = len(beat_list)
    C, T = beat_list[0].shape
    beats = np.zeros((max_beats, C, T), dtype=np.float32)
    mask  = np.zeros((max_beats,), dtype=np.float32)
    beats[:B] = np.stack(beat_list, axis=0)
    mask[:B]  = 1.0
    return beats, mask
