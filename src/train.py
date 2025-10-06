import os, argparse, yaml, json, numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupKFold
from .utils.seed import set_seed
from .utils.logging import tbar
from .utils.cv import make_folds, save_fold_split, save_metrics
from .data.dataset import PulseDataset
from .models.tc_transformer import TCTransformer
from .models.losses import SmoothL1Multi, HeteroscedasticLoss
from .models.metrics import mae, DictMeter

class EMA:
    def __init__(self, model, decay=0.999):
        self.decay=decay
        self.shadow={k:v.detach().clone() for k,v in model.state_dict().items()}
    def update(self, model):
        with torch.no_grad():
            for k,v in model.state_dict().items():
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0-self.decay)
    def apply_to(self, model):
        model.load_state_dict(self.shadow, strict=True)

def build_model(cfg_train, cfg_model):
    m = TCTransformer({**cfg_model, "loss": cfg_train["loss"]})
    return m

def train_one_epoch(model, loader, optimizer, criterion, device, grad_clip=1.0, ema=None):
    model.train()
    dm = DictMeter()
    for x,y,_ in tbar(loader,"train"):
        x=x.to(device); y=y.to(device)
        optimizer.zero_grad()
        out = model(x)
        pred = out if out.shape[1]==2 else out[:,:2]
        loss = criterion(out, y) if out.shape[1]>2 else criterion(pred, y)
        loss.backward()
        if grad_clip: nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if ema: ema.update(model)
        with torch.no_grad():
            m = torch.mean(torch.abs(pred - y), dim=0)  # [2]
            dm.update({"loss":loss.item(), "mae_sbp":m[0].item(), "mae_dbp":m[1].item()}, n=x.size(0))
    return dm.avg()

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    dm = DictMeter()
    preds=[]; gts=[]; pids=[]
    for x,y,pid in tbar(loader,"valid"):
        x=x.to(device); y=y.to(device)
        out = model(x)
        pred = out if out.shape[1]==2 else out[:,:2]
        loss = criterion(out, y) if out.shape[1]>2 else criterion(pred, y)
        m = torch.mean(torch.abs(pred - y), dim=0)
        dm.update({"loss":loss.item(), "mae_sbp":m[0].item(), "mae_dbp":m[1].item()}, n=x.size(0))
        preds.append(pred.cpu().numpy()); gts.append(y.cpu().numpy()); pids+=list(pid)
    preds = np.concatenate(preds,0); gts=np.concatenate(gts,0)
    return dm.avg(), preds, gts, pids

def main(cfg_path):
    with open(cfg_path) as f: cfg=yaml.safe_load(f)
    set_seed(cfg["cv"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    index = pd.read_csv(cfg["paths"]["index_csv"])
    runs_dir = cfg["paths"]["runs_dir"]; os.makedirs(runs_dir, exist_ok=True)

    # CV loop
    for k, tr_df, va_df in make_folds(index, cfg["cv"]["n_splits"], cfg["cv"]["seed"]):
        fold_dir = os.path.join(runs_dir, f"fold_{k}"); os.makedirs(fold_dir, exist_ok=True)
        save_fold_split(runs_dir, k, tr_df, va_df)

        tr_ds = PulseDataset(tr_df, fs=cfg["data"]["fs"], band=tuple(cfg["data"]["bandpass"]),
                             norm=cfg["data"]["normalize"],
                             use_sqi=cfg["data"]["sqi"]["enable"], sqi_cfg=cfg["data"]["sqi"],
                             aug_cfg=cfg["data"]["augment"], train=True)
        va_ds = PulseDataset(va_df, fs=cfg["data"]["fs"], band=tuple(cfg["data"]["bandpass"]),
                             norm=cfg["data"]["normalize"],
                             use_sqi=cfg["data"]["sqi"]["enable"], sqi_cfg=cfg["data"]["sqi"],
                             aug_cfg={**cfg["data"]["augment"],"enable":False}, train=False)

        tr_dl = DataLoader(tr_ds, batch_size=cfg["train"]["batch_size"], shuffle=True,
                           num_workers=cfg["train"]["num_workers"], pin_memory=True, drop_last=True)
        va_dl = DataLoader(va_ds, batch_size=cfg["train"]["batch_size"], shuffle=False,
                           num_workers=cfg["train"]["num_workers"], pin_memory=True)

        model = build_model(cfg["train"], cfg["model"]).to(device)
        if cfg["train"]["loss"]=="hetero":
            criterion = HeteroscedasticLoss(sbp_weight=cfg["train"]["sbp_weight"])
        else:
            criterion = SmoothL1Multi(sbp_weight=cfg["train"]["sbp_weight"])

        if cfg["train"]["optimizer"].lower()=="adamw":
            optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"],
                                          weight_decay=cfg["train"]["weight_decay"])
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"])

        if cfg["train"]["scheduler"]=="cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=cfg["train"]["epochs"]-cfg["train"]["warmup_epochs"])
        else:
            scheduler = None

        ema = EMA(model, cfg["train"]["ema"]["decay"]) if cfg["train"]["ema"]["enable"] else None
        best_val = 1e9; best_path = os.path.join(fold_dir, "best.ckpt")
        patience = cfg["train"]["early_stop_patience"]; bad=0

        for epoch in range(cfg["train"]["epochs"]):
            # warmup
            if epoch < cfg["train"]["warmup_epochs"]:
                for g in optimizer.param_groups:
                    g["lr"] = cfg["train"]["lr"] * (epoch+1)/cfg["train"]["warmup_epochs"]

            tr_metrics = train_one_epoch(model, tr_dl, optimizer, criterion, device,
                                         cfg["train"]["grad_clip"], ema)
            if scheduler and epoch >= cfg["train"]["warmup_epochs"]:
                scheduler.step()

            # validate with EMA weights if enabled
            backup=None
            if ema:
                backup = {k:v.clone() for k,v in model.state_dict().items()}
                ema.apply_to(model)
            va_metrics, preds, gts, pids = validate(model, va_dl, criterion, device)
            if ema and backup: model.load_state_dict(backup, strict=True)

            # selection metric: weighted MAE
            score = va_metrics["mae_sbp"]*cfg["train"]["sbp_weight"] + va_metrics["mae_dbp"]
            if score < best_val:
                best_val=score; bad=0
                torch.save(model.state_dict(), best_path)
                # save OOF for this fold
                oof = pd.DataFrame({"pid":pids,
                                    "sbp_true":gts[:,0], "dbp_true":gts[:,1],
                                    "sbp_pred":preds[:,0], "dbp_pred":preds[:,1]})
                oof.to_csv(os.path.join(fold_dir,"oof_predictions.csv"), index=False)
                save_metrics(runs_dir, k, {"val":va_metrics,"best_score":best_val})
            else:
                bad+=1
                if bad>=patience: break

        print(f"Fold {k} best_score={best_val:.4f} saved at {best_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    a=ap.parse_args()
    main(a.config)
