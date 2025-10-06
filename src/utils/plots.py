import numpy as np, matplotlib.pyplot as plt, json, os, pandas as pd

def bland_altman(y_true, y_pred, title, save_path):
    diff = y_pred - y_true
    mean = (y_pred + y_true)/2
    b = np.mean(diff); sd = np.std(diff)
    lo, hi = b - 1.96*sd, b + 1.96*sd
    plt.figure(figsize=(6,4))
    plt.scatter(mean, diff, s=6, alpha=0.5)
    plt.axhline(b, color='r'); plt.axhline(lo, color='g'); plt.axhline(hi, color='g')
    plt.title(f"{title} (bias={b:.2f}, LOA={lo:.2f}..{hi:.2f})")
    plt.xlabel("Mean"); plt.ylabel("Diff (pred - true)")
    plt.tight_layout(); plt.savefig(save_path); plt.close()

def per_subject_box(mae_df, title, save_path):
    ax = mae_df.boxplot(figsize=(6,4))
    ax.set_title(title); ax.set_ylabel("MAE (mmHg)")
    plt.tight_layout(); plt.savefig(save_path); plt.close()

def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path,"w") as f: json.dump(obj,f,indent=2)

def load_oof_csvs(runs_dir):
    all_rows=[]
    for fold in sorted(os.listdir(runs_dir)):
        p = os.path.join(runs_dir, fold, "oof_predictions.csv")
        if os.path.isfile(p):
            df = pd.read_csv(p)
            df["fold"]=fold
            all_rows.append(df)
    return pd.concat(all_rows, ignore_index=True)
