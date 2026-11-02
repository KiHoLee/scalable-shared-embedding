"""Revision: SNR gap against the full codebook outside the main operating point,
on the Table II channel. Output: data/rev_gap_curves.csv and rev_gap_db.csv.
The gap at the SER of the proposal is read by log-SER interpolation of the
full-codebook curve."""
import math

import numpy as np

from exp_rev_robust import load_or_train, run
from sse_lib import DATA, write_csv

GRID = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
CASES = [("flat", 4, "sse_P4.pt", "full_V65536.pt", {}, {}),
         ("U2", 2, "rev_sse_U2.pt", "rev_full_U2.pt", {}, {}),
         ("T4", 4, "sse_P4.pt", "full_V65536.pt",
          {"taps": 4, "inter": True, "rule": "wmd"}, {"taps": 4, "rule": "wmd"}),
         ("T8", 4, "sse_P4.pt", "full_V65536.pt",
          {"taps": 8, "inter": True, "rule": "wmd"}, {"taps": 8, "rule": "wmd"})]


def main():
    rows, gaps = [], []
    for name, users, fs, ff, ks, kf in CASES:
        s = load_or_train("sse", users, fs)
        f = load_or_train("full", users, ff)
        a, b = [], []
        for snr in GRID:
            a.append(float(run("sse", s, users, float(snr), 1_000_000, 100_000, 777, **ks).mean()))
            b.append(float(run("full", f, users, float(snr), 200_000, 128, 777, **kf).mean()))
            rows.append((name, snr, a[-1], b[-1]))
            print(rows[-1], flush=True)
            write_csv(DATA / "rev_gap_curves.csv", ["case", "snr_db", "ser_sse", "ser_full"], rows)
        lb = np.log(np.array(b))
        for snr, sa in zip(GRID, a):
            if sa >= b[0] or sa <= b[-1]:
                continue
            at = float(np.interp(-math.log(sa), -lb, GRID))
            gaps.append((name, snr, sa, snr - at))
            write_csv(DATA / "rev_gap_db.csv", ["case", "snr_db", "ser_sse", "gap_db"], gaps)
    print(gaps, flush=True)


if __name__ == "__main__":
    main()
