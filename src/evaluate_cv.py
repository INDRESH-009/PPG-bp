
import argparse, yaml, os, numpy as np, pandas as pd
from .utils.plots import load_oof_csvs, bland_altman, save_json

def summarize(oof):
    sbp_mae = np.mean(np.abs(oof["sbp_pred"]-oof["sbp_true"]))
    dbp_mae = np.mean(np.abs(oof["dbp_pred"]-oof["dbp_true"]))
    sbp_rmse = np.sqrt(np.mean((oof["sbp_pred"]-oof["sbp_true"])**2))
    dbp_rmse = np.sqrt(np.mean((oof["dbp_pred"]-oof["dbp_true"])**2))

    # per-subject MAE
    ps = oof.groupby("pid").apply(lambda g: pd.Series({
        "sbp_mae": np.mean(np.abs(g["sbp_pred"]-g["sbp_true"])),
        "dbp_mae": np.mean(np.abs(g["dbp_pred"]-g["dbp_true"]))
    })).reset_index()

    within = {}
    for mm in [5,10,15]:
        within[f"sbp_within_{mm}"] = float(np.mean(np.abs(oof["sbp_pred"]-oof["sbp_true"])<=mm))
        within[f"dbp_within_{mm}"] = float(np.mean(np.abs(oof["dbp_pred"]-oof["dbp_true"])<=mm))

    return {
        "overall": {"sbp_mae":float(sbp_mae),"dbp_mae":float(dbp_mae),
                    "sbp_rmse":float(sbp_rmse),"dbp_rmse":float(dbp_rmse)},
        "within": within,
        "per_subject_count": int(ps.shape[0])
    }, ps

def main(cfg_path):
    with open(cfg_path) as f: cfg=yaml.safe_load(f)
    runs = cfg["paths"]["runs_dir"]
    oof = load_oof_csvs(runs)
    summary, ps = summarize(oof)
    print(summary)
    os.makedirs(runs, exist_ok=True)
    oof.to_csv(os.path.join(runs,"oof_all.csv"), index=False)
    ps.to_csv(os.path.join(runs,"per_subject_mae.csv"), index=False)
    save_json(summary, os.path.join(runs,"cv_summary.json"))

    # plots
    bland_altman(oof["sbp_true"].values, oof["sbp_pred"].values,
                 "SBP Bland–Altman", os.path.join(runs,"sbp_ba.png"))
    bland_altman(oof["dbp_true"].values, oof["dbp_pred"].values,
                 "DBP Bland–Altman", os.path.join(runs,"dbp_ba.png"))

if __name__ == "__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    a=ap.parse_args()
    main(a.config)
