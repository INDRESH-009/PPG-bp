import numpy as np, torch
from torch.utils.data import Dataset
from .utils import bandpass, detrend_poly, normalize
from .sqi import basic_sqi
from .augment import apply_augs
from .beats import beats_from_window

class PulseBeatsDataset(Dataset):
    def __init__(self, df, fs=125, band=(0.5,8.0), norm="zscore",
                 use_sqi=True, sqi_cfg=None, aug_cfg=None, train=True,
                 beats_cfg=None):
        self.df = df.reset_index(drop=True)
        self.fs = fs
        self.band = band
        self.norm = norm
        self.use_sqi = use_sqi
        self.sqi_cfg = sqi_cfg or {}
        self.aug_cfg = aug_cfg or {}
        self.train = train
        self.beats_cfg = beats_cfg or {
            "target_len": 256, "max_beats": 16, "min_beats": 6,
            "peak_distance_ms": 300, "prominence": 0.1, "use_vel_acc": True
        }

    def __len__(self): return len(self.df)

    def _load_npz(self, path):
        with np.load(path) as d:
            x = d["PPG_Record_F"].astype(np.float32)
            y = np.array([float(d["SegSBP"].reshape(-1)[0]),
                          float(d["SegDBP"].reshape(-1)[0])], dtype=np.float32)
        return x, y

    def _proc_signal(self, x):
        if self.band is not None:
            x = bandpass(x, self.fs, self.band[0], self.band[1])
        x = detrend_poly(x, order=1)
        x = normalize(x, self.norm)
        return x

    def __getitem__(self, i):
        row = self.df.iloc[i]
        x, y = self._load_npz(row["path"])
        x = self._proc_signal(x)

        if self.use_sqi:
            ok = basic_sqi(x, self.fs,
                           amp_range=self.sqi_cfg.get("amp_range",(0.1,3.0)),
                           flat_std_min=self.sqi_cfg.get("flat_std_min",0.05),
                           hr_range=self.sqi_cfg.get("hr_range",(40,180)))
            if not ok and self.train:
                # try next window to avoid train stalling
                j = (i+1) % len(self.df)
                row2 = self.df.iloc[j]
                x, y = self._load_npz(row2["path"])
                x = self._proc_signal(x)

        if self.train:
            x = apply_augs(x, self.fs, self.aug_cfg)

        beats, mask = beats_from_window(
            x, self.fs,
            target_len=self.beats_cfg.get("target_len",256),
            max_beats=self.beats_cfg.get("max_beats",16),
            min_beats=self.beats_cfg.get("min_beats",6),
            min_dist_ms=self.beats_cfg.get("peak_distance_ms",300),
            prominence=self.beats_cfg.get("prominence",0.1),
            use_vel_acc=self.beats_cfg.get("use_vel_acc",True)
        )

        # For validation/test, if beats extraction fails, fallback to a dummy to keep batch shape,
        # but mark mask=0 (the model head will globally pool; here we ensure at least 1 beat).
        if beats is None or mask is None:
            beats = np.zeros((self.beats_cfg.get("max_beats",16),
                              3 if self.beats_cfg.get("use_vel_acc",True) else 1,
                              self.beats_cfg.get("target_len",256)), dtype=np.float32)
            mask = np.zeros((self.beats_cfg.get("max_beats",16)), dtype=np.float32)

        x_beats = torch.tensor(beats, dtype=torch.float32)  # [B,C,T]
        mask = torch.tensor(mask, dtype=torch.float32)      # [B]
        y = torch.tensor(y, dtype=torch.float32)            # [2]
        pid = row["pid"]
        return x_beats, mask, y, pid
