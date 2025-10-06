import os, argparse, yaml, json, numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import DataLoader
from .utils.seed import set_seed
from .utils.logging import tbar
from .utils.cv import make_folds, save_fold_split, save_metrics
from .data.dataset import PulseDataset
from .data.dataset_beats import PulseBeatsDataset
from .models.tc_transformer import TCTransformer
from .models.beatformer import BeatFormer
from .models.losses import SmoothL1Multi, HeteroscedasticLoss
from .models.metrics import DictMeter

# ==============================
# Helpers
# ==============================

class TargetScaler:
    """Per-fold z-score scaling for [SBP, DBP]."""
    def __init__(self, mean=None, std=None):
        self.mean = None if mean is None else np.array(mean, dtype=np.float32)
        self.std  = None if std  is None else np.array(std, dtype=np.float32)

    def fit(self, y_np):   # y_np: [N,2]
        self.mean = y_np.mean(axis=0).astype(np.float32)
        self.std  = (y_np.std(axis=0) + 1e-6).astype(np.float32)
        return self

    def transform(self, y_t: torch.Tensor) -> torch.Tensor:
        return (y_t - torch.tensor(self.mean, device=y_t.device)) / torch.tensor(self.std, device=y_t.device)

    def inv_transform(self, y_t: torch.Tensor) -> torch.Tensor:
        return y_t * torch.tensor(self.std, device=y_t.device) + torch.tensor(self.mean, device=y_t.device)

class EMA:
    """Float-parameter EMA (skips long/bool buffers)."""
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        for k, v in model.state_dict().items():
            if v.dtype is not None and torch.is_floating_point(v):
                self.shadow[k] = v.detach().clone()

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype is not None and torch.is_floating_point(v) and k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def apply_to(self, model):
        sd = model.state_dict()
        for k, v in self.shadow.items():
            if k in sd and torch.is_floating_point(sd[k]):
                sd[k].copy_(v)
        model.load_state_dict(sd, strict=True)

def build_model(cfg_train, cfg_model):
    name = cfg_model["name"].lower()
    if name == "beatformer":
        return BeatFormer({**cfg_model, "loss": cfg_train["loss"]})
    return TCTransformer({**cfg_model, "loss": cfg_train["loss"]})

def build_datasets(cfg, tr_df, va_df):
    use_beats = (cfg["model"]["name"].lower() == "beatformer")
    if use_beats:
        tr_ds = PulseBeatsDataset(
            tr_df, fs=cfg["data"]["fs"], band=tuple(cfg["data"]["bandpass"]),
            norm=cfg["data"]["normalize"], use_sqi=cfg["data"]["sqi"]["enable"],
            sqi_cfg=cfg["data"]["sqi"], aug_cfg=cfg["data"]["augment"], train=True,
            beats_cfg=cfg["beats"]
        )
        va_ds = PulseBeatsDataset(
            va_df, fs=cfg["data"]["fs"], band=tuple(cfg["data"]["bandpass"]),
            norm=cfg["data"]["normalize"], use_sqi=cfg["data"]["sqi"]["enable"],
            sqi_cfg=cfg["data"]["sqi"], aug_cfg={**cfg["data"]["augment"], "enable": False},
            train=False, beats_cfg=cfg["beats"]
        )
    else:
        tr_ds = PulseDataset(
            tr_df, fs=cfg["data"]["fs"], band=tuple(cfg["data"]["bandpass"]),
            norm=cfg["data"]["normalize"], use_sqi=cfg["data"]["sqi"]["enable"],
            sqi_cfg=cfg["data"]["sqi"], aug_cfg=cfg["data"]["augment"], train=True,
            channels=cfg["data"].get("channels", {"raw": True})
        )
        va_ds = PulseDataset(
            va_df, fs=cfg["data"]["fs"], band=tuple(cfg["data"]["bandpass"]),
            norm=cfg["data"]["normalize"], use_sqi=cfg["data"]["sqi"]["enable"],
            sqi_cfg=cfg["data"]["sqi"], aug_cfg={**cfg["data"]["augment"], "enable": False},
            train=False, channels=cfg["data"].get("channels", {"raw": True})
        )
    return use_beats, tr_ds, va_ds

# ==============================
# Train / Validate
# ==============================

def train_one_epoch(model, loader, optimizer, criterion, device,
                    grad_clip=1.0, ema=None, scaler: TargetScaler=None, use_beats=False):
    model.train()
    dm = DictMeter()

    for batch in tbar(loader, "train"):
        if use_beats:
            beats, mask, y, _ = batch
            beats, mask, y = beats.to(device), mask.to(device), y.to(device)
            out = model(beats, mask)  # [B,2] or [B,4]
        else:
            x, y, _ = batch
            x, y = x.to(device), y.to(device)
            out = model(x)

        pred = out if out.shape[1] == 2 else out[:, :2]
        y_scaled = scaler.transform(y) if scaler is not None else y

        optimizer.zero_grad()
        loss = criterion(out, y_scaled) if out.shape[1] > 2 else criterion(pred, y_scaled)
        loss.backward()
        if grad_clip:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if ema:
            ema.update(model)

        with torch.no_grad():
            # report MAE in real units
            pred_real = scaler.inv_transform(pred) if scaler is not None else pred
            m = torch.mean(torch.abs(pred_real - y), dim=0)
            dm.update({
                "loss": loss.item(),
                "mae_sbp": m[0].item(),
                "mae_dbp": m[1].item()
            }, n=y.size(0))

    return dm.avg()

@torch.no_grad()
def validate(model, loader, criterion, device, scaler: TargetScaler=None, use_beats=False):
    model.eval()
    dm = DictMeter()
    preds, gts, pids = [], [], []

    for batch in tbar(loader, "valid"):
        if use_beats:
            beats, mask, y, pid = batch
            beats, mask, y = beats.to(device), mask.to(device), y.to(device)
            out = model(beats, mask)
        else:
            x, y, pid = batch
            x, y = x.to(device), y.to(device)
            out = model(x)

        pred = out if out.shape[1] == 2 else out[:, :2]
        y_scaled = scaler.transform(y) if scaler is not None else y
        loss = criterion(out, y_scaled) if out.shape[1] > 2 else criterion(pred, y_scaled)

        pred_real = scaler.inv_transform(pred) if scaler is not None else pred
        m = torch.mean(torch.abs(pred_real - y), dim=0)
        dm.update({
            "loss": loss.item(),
            "mae_sbp": m[0].item(),
            "mae_dbp": m[1].item()
        }, n=y.size(0))

        preds.append(pred_real.cpu().numpy())
        gts.append(y.cpu().numpy())
        pids += list(pid)

    preds = np.concatenate(preds, 0)
    gts   = np.concatenate(gts, 0)
    return dm.avg(), preds, gts, pids

def log_epoch(fold_dir, epoch, tr_metrics, va_metrics):
    """Append epoch metrics to CSV and print formatted line."""
    log_path = os.path.join(fold_dir, "train_log.csv")
    row = {
        "epoch": epoch,
        "train_loss": tr_metrics["loss"],
        "train_mae_sbp": tr_metrics["mae_sbp"],
        "train_mae_dbp": tr_metrics["mae_dbp"],
        "val_loss": va_metrics["loss"],
        "val_mae_sbp": va_metrics["mae_sbp"],
        "val_mae_dbp": va_metrics["mae_dbp"],
    }
    header = not os.path.exists(log_path)
    pd.DataFrame([row]).to_csv(log_path, mode="a", index=False, header=header)
    print(f"Epoch {epoch:03d} | "
          f"Train SBP {tr_metrics['mae_sbp']:.3f}  DBP {tr_metrics['mae_dbp']:.3f} | "
          f"Val SBP {va_metrics['mae_sbp']:.3f}  DBP {va_metrics['mae_dbp']:.3f}")

# ==============================
# Main
# ==============================

def main(cfg_path):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["cv"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    index = pd.read_csv(cfg["paths"]["index_csv"])
    runs_dir = cfg["paths"]["runs_dir"]
    os.makedirs(runs_dir, exist_ok=True)

    # Cross-validated training
    for k, tr_df, va_df in make_folds(index, cfg["cv"]["n_splits"], cfg["cv"]["seed"]):
        fold_dir = os.path.join(runs_dir, f"fold_{k}")
        os.makedirs(fold_dir, exist_ok=True)
        save_fold_split(runs_dir, k, tr_df, va_df)

        # Per-fold target scaler (fit on TRAIN labels only)
        y_tr = tr_df[["sbp", "dbp"]].values.astype(np.float32)
        scaler = TargetScaler().fit(y_tr)

        use_beats, tr_ds, va_ds = build_datasets(cfg, tr_df, va_df)

        tr_dl = DataLoader(tr_ds, batch_size=cfg["train"]["batch_size"], shuffle=True,
                           num_workers=cfg["train"]["num_workers"], pin_memory=True, drop_last=True)
        va_dl = DataLoader(va_ds, batch_size=cfg["train"]["batch_size"], shuffle=False,
                           num_workers=cfg["train"]["num_workers"], pin_memory=True)

        model = build_model(cfg["train"], cfg["model"]).to(device)
        criterion = (HeteroscedasticLoss if cfg["train"]["loss"] == "hetero" else SmoothL1Multi)(
            sbp_weight=cfg["train"]["sbp_weight"]
        )

        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"],
                                      weight_decay=cfg["train"]["weight_decay"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg["train"]["epochs"] - cfg["train"]["warmup_epochs"]
        ) if cfg["train"]["scheduler"] == "cosine" else None

        ema = EMA(model, cfg["train"]["ema"]["decay"]) if cfg["train"]["ema"]["enable"] else None
        best_val = 1e9
        best_path = os.path.join(fold_dir, "best.ckpt")
        patience = cfg["train"]["early_stop_patience"]
        bad = 0

        for epoch in range(cfg["train"]["epochs"]):
            # warmup
            if epoch < cfg["train"]["warmup_epochs"]:
                for g in optimizer.param_groups:
                    g["lr"] = cfg["train"]["lr"] * (epoch + 1) / cfg["train"]["warmup_epochs"]

            tr_metrics = train_one_epoch(
                model, tr_dl, optimizer, criterion, device,
                cfg["train"]["grad_clip"], ema, scaler, use_beats
            )

            if scheduler and epoch >= cfg["train"]["warmup_epochs"]:
                scheduler.step()

            # Validate with EMA weights if enabled
            backup = None
            if ema:
                backup = {k: v.clone() for k, v in model.state_dict().items()}
                ema.apply_to(model)

            va_metrics, preds, gts, pids = validate(
                model, va_dl, criterion, device, scaler, use_beats
            )

            if ema and backup:
                model.load_state_dict(backup, strict=True)

            # Log
            log_epoch(fold_dir, epoch, tr_metrics, va_metrics)

            # Early-stopping score (SBP-weighted)
            score = va_metrics["mae_sbp"] * cfg["train"]["sbp_weight"] + va_metrics["mae_dbp"]
            if score < best_val:
                best_val = score
                bad = 0
                torch.save(model.state_dict(), best_path)

                # Save OOF preds for this fold
                oof = pd.DataFrame({
                    "pid": pids,
                    "sbp_true": gts[:, 0], "dbp_true": gts[:, 1],
                    "sbp_pred": preds[:, 0], "dbp_pred": preds[:, 1]
                })
                oof.to_csv(os.path.join(fold_dir, "oof_predictions.csv"), index=False)
                save_metrics(runs_dir, k, {"val": va_metrics, "best_score": best_val})
            else:
                bad += 1
                if bad >= patience:
                    break

        print(f"Fold {k}: best_score={best_val:.4f} saved at {best_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    a = ap.parse_args()
    main(a.config)
