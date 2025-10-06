import argparse, random, numpy as np, pandas as pd, matplotlib.pyplot as plt

def main(index_csv, n):
    df = pd.read_csv(index_csv)
    print(df.describe())
    picks = df.sample(min(n, len(df)), random_state=0)
    fig, axes = plt.subplots(len(picks), 1, figsize=(10, 2.2*len(picks)))
    if len(picks)==1: axes=[axes]
    for ax, (_, r) in zip(axes, picks.iterrows()):
        with np.load(r["path"]) as d:
            x = d["PPG_Record_F"].astype(np.float32)
        ax.plot(x)
        ax.set_title(f'pid={r.pid} {r.win_id} SBP={r.sbp:.1f} DBP={r.dbp:.1f}')
        ax.set_ylabel("amp")
    axes[-1].set_xlabel("samples")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="data/index.csv")
    ap.add_argument("--n", type=int, default=8)
    a=ap.parse_args()
    main(a.index, a.n)
