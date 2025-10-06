import argparse, yaml, os, numpy as np, torch
from .models.tc_transformer import TCTransformer
from .data.utils import bandpass, detrend_poly, normalize

@torch.no_grad()
def predict_npz(npz_path, cfg, ckpt_path):
    with np.load(npz_path) as d:
        x = d["PPG_Record_F"].astype(np.float32)
    fs = cfg["data"]["fs"]
    lo,hi = cfg["data"]["bandpass"]
    x = bandpass(x, fs, lo, hi)
    x = detrend_poly(x, order=1)
    x = normalize(x, cfg["data"]["normalize"])
    x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [1,1,L]

    model = TCTransformer({**cfg["model"], "loss": cfg["train"]["loss"]})
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()
    y = model(x)
    if y.shape[1]>2: y=y[:,:2]
    return y.squeeze(0).numpy()  # [2]

def main(cfg_path, in_npz, ckpt=None):
    with open(cfg_path) as f: cfg=yaml.safe_load(f)
    if ckpt is None:
        ckpt = os.path.join(cfg["paths"]["runs_dir"], "fold_0", "best.ckpt")
    pred = predict_npz(in_npz, cfg, ckpt)
    print(f"Predicted SBP={pred[0]:.2f} DBP={pred[1]:.2f}")

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--in_npz", required=True)
    ap.add_argument("--ckpt", default=None)
    a=ap.parse_args()
    main(a.config, a.in_npz, a.ckpt)
