import numpy as np, torch
from torch.utils.data import Dataset
from .utils import bandpass, detrend_poly, normalize
from .sqi import basic_sqi
from .augment import apply_augs

class PulseDataset(Dataset):
    def __init__(self, df, fs=100, band=(0.5,8.0), norm="zscore",
                 use_sqi=True, sqi_cfg=None, aug_cfg=None, train=True):
        self.df = df.reset_index(drop=True)
        self.fs = fs
        self.band = band
        self.norm = norm
        self.use_sqi = use_sqi
        self.sqi_cfg = sqi_cfg or {}
        self.aug_cfg = aug_cfg or {}
        self.train = train

    def __len__(self): return len(self.df)

    def _load_npz(self, path):
        with np.load(path) as d:
            x = d["PPG_Record_F"].astype(np.float32)
            y = np.array([float(d["SegSBP"].reshape(-1)[0]),
                          float(d["SegDBP"].reshape(-1)[0])], dtype=np.float32)
        return x, y

    def _proc(self, x):
        if self.band is not None:
            x = bandpass(x, self.fs, self.band[0], self.band[1])
        x = detrend_poly(x, order=1)
        x = normalize(x, self.norm)
        return x

    def __getitem__(self, i):
        row = self.df.iloc[i]
        x, y = self._load_npz(row["path"])
        x = self._proc(x)

        if self.use_sqi:
            ok = basic_sqi(x, self.fs,
                           amp_range=self.sqi_cfg.get("amp_range",(0.1,3.0)),
                           flat_std_min=self.sqi_cfg.get("flat_std_min",0.05),
                           hr_range=self.sqi_cfg.get("hr_range",(40,180)))
            # if fails SQI during train, try a neighbor; during val/test, keep as-is
            if not ok and self.train:
                j = (i+1) % len(self.df)
                row2 = self.df.iloc[j]
                x, y = self._load_npz(row2["path"])
                x = self._proc(x)

        if self.train:
            x = apply_augs(x, self.fs, self.aug_cfg)

        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)  # [1, L]
        y = torch.tensor(y, dtype=torch.float32)               # [2]
        pid = row["pid"]
        return x, y, pid
