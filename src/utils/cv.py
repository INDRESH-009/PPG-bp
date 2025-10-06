import os, json, pandas as pd
from sklearn.model_selection import GroupKFold

def make_folds(df, n_splits=5, seed=42):
    # GroupKFold is deterministic; seed kept for interface parity
    gkf = GroupKFold(n_splits=n_splits)
    for k,(tr,va) in enumerate(gkf.split(df, groups=df["pid"])):
        yield k, df.iloc[tr].reset_index(drop=True), df.iloc[va].reset_index(drop=True)

def save_fold_split(out_dir, k, tr_df, va_df):
    d = os.path.join(out_dir, f"fold_{k}")
    os.makedirs(d, exist_ok=True)
    tr_df.to_csv(os.path.join(d,"train.csv"), index=False)
    va_df.to_csv(os.path.join(d,"val.csv"), index=False)

def save_metrics(out_dir, k, metrics):
    d = os.path.join(out_dir, f"fold_{k}")
    with open(os.path.join(d,"val_metrics.json"),"w") as f:
        json.dump(metrics, f, indent=2)
