import argparse, os, glob, numpy as np, pandas as pd

def probe_npz(p):
    try:
        with np.load(p) as d:
            sbp = float(d["SegSBP"].reshape(-1)[0])
            dbp = float(d["SegDBP"].reshape(-1)[0])
            x = d["PPG_Record_F"].astype(np.float32)
            L = int(x.shape[0])
        return sbp, dbp, L, True
    except Exception as e:
        return None, None, None, False

def main(data_root, out_csv):
    rows = []
    pid_dirs = sorted([d for d in glob.glob(os.path.join(data_root, "*")) if os.path.isdir(d)])
    for pid_dir in pid_dirs:
        pid = os.path.basename(pid_dir)
        for p in sorted(glob.glob(os.path.join(pid_dir, "win*.npz"))):
            sbp, dbp, L, ok = probe_npz(p)
            if ok:
                win = os.path.splitext(os.path.basename(p))[0]
                rows.append((pid, win, os.path.abspath(p), sbp, dbp, L))
    df = pd.DataFrame(rows, columns=["pid","win_id","path","sbp","dbp","length"])
    assert len(df)>0, "No valid npz files found."
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Wrote {len(df)} rows to {out_csv}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out", default="data/index.csv")
    a = ap.parse_args()
    main(a.data_root, a.out)
