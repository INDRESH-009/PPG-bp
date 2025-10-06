# ppg2bp

End-to-end PPG→BP estimation (SBP & DBP) from 10-s PPG windows with strict subject-wise CV.

## Quickstart
1) Build index:
python scripts/prepare_index.py --data_root "/ABS/PATH/DataRoot" --out data/index.csv
2) (Optional) Sanity plots:
python scripts/sanity_check.py --index data/index.csv --n 12
3) Train 5-fold CV:
python -m src.train --config configs/default.yaml
4) Aggregate + plots:
python -m src.evaluate_cv --config configs/default.yaml
5) Single-file inference:
python -m src.infer --config configs/default.yaml --in_npz "/path/pid123/win42.npz"