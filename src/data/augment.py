import numpy as np

def _rand(a,b): return np.random.uniform(a,b)

def time_warp(x, pct=0.05):
    if pct<=0: return x
    L = len(x)
    factor = 1.0 + _rand(-pct, pct)      # e.g., 0.95..1.05
    new_L = max(8, int(round(L*factor)))
    xp = np.linspace(0, 1, L)
    xq = np.interp(np.linspace(0,1,new_L), xp, x)
    # back to original length by resampling
    return np.interp(np.linspace(0,1,L), np.linspace(0,1,new_L), xq).astype(np.float32)

def amp_scale(x, pct=0.1):
    if pct<=0: return x
    s = 1.0 + _rand(-pct, pct)
    return (x*s).astype(np.float32)

def jitter(x, std=0.005):
    if std<=0: return x
    return (x + np.random.normal(0, std, size=x.shape).astype(np.float32)).astype(np.float32)

def random_crop_pad(x, fs, crop_seconds=0.5):
    if crop_seconds<=0: return x
    L = len(x)
    chop = int(crop_seconds*fs)
    chop = min(max(chop, 0), L//4)  # limit
    left = np.random.randint(0, chop+1)
    right = chop - left
    y = x[left: L-right] if right>0 else x[left:]
    # pad back to L
    if len(y)<L:
        pad_left = np.random.randint(0, L-len(y)+1)
        pad_right = L-len(y)-pad_left
        y = np.pad(y, (pad_left, pad_right), mode="edge")
    return y.astype(np.float32)

def time_mask(x, max_frac=0.08):
    if max_frac <= 0: 
        return x
    L = len(x)
    w = int(L * np.random.uniform(0.0, max_frac))
    if w <= 0 or w >= L:
        return x
    s = np.random.randint(0, L - w + 1)
    y = x.copy()
    # linear interpolation across the masked region
    left_idx = max(0, s-1)
    right_idx = min(L-1, s+w)
    y[s:s+w] = np.interp(np.arange(s, s+w),
                         [left_idx, right_idx],
                         [x[left_idx], x[right_idx]]).astype(np.float32)
    return y

def apply_augs(x, fs, cfg):
    if not cfg.get("enable", True):
        return x.astype(np.float32)
    x = time_warp(x, cfg.get("time_warp_pct", 0.05))
    x = amp_scale(x, cfg.get("amp_scale_pct", 0.10))
    x = jitter(x, cfg.get("jitter_std", 0.005))
    if cfg.get("time_mask", {}).get("enable", False):
        x = time_mask(x, cfg.get("time_mask", {}).get("max_frac", 0.08))
    # keep crop off for SBP stability (crop_seconds set to 0.0)
    return x.astype(np.float32)

