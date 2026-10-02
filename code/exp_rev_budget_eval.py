"""Final evaluation of the long-budget (2e5-iteration) models of
exp_rev_budget.py for training seeds 1, 2 and 3, and their mean.

Loads data/rev_{scheme}_long_md{tag}.pt (tag "" for seed 1, "_s2",
"_s3"), evaluates the SER over SNR_GRID with 2e6 frames per point (the
full-codebook variants by minimum distance, SSE by correlation), and
writes

  data/rev_budget_final_{scheme}{tag}.csv   per seed (snr, ser, ser_md)
  data/rev_budget_final_mean.csv            3-seed mean per scheme
  data/rev_budget_curve_{scheme}_mean.csv   3-seed mean of the logged
                                            training curves (same
                                            iteration grid in all seeds)

Table I long-budget rows and Fig. 3 use the *_mean files.
"""
import csv

import torch

from exp_rev_budget import (SNR_GRID, V, HierCodebook, evaluator)
from sse_lib import DATA, DEVICE, P_MAX, SSE, FullCodebook, write_csv

SEEDS = [1, 2, 3]
SCHEMES = ["sse", "full", "sampled", "hier"]
FRAMES = 2_000_000


def tag(seed: int) -> str:
    return "" if seed == 1 else f"_s{seed}"


def load(scheme: str, seed: int):
    if scheme == "sse":
        m = SSE(P=P_MAX)
    elif scheme == "hier":
        m = HierCodebook(V)
    else:
        m = FullCodebook(V)
    m.load_state_dict(torch.load(DATA / f"rev_{scheme}_long_md{tag(seed)}.pt",
                                 map_location="cpu"))
    m.to(DEVICE).eval()
    m.calibrate_power()
    return m


def read_rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def main():
    mean_rows = []
    for sch in SCHEMES:
        per_seed = []
        for seed in SEEDS:
            m = load(sch, seed)
            ev = evaluator(sch, m)
            ser = ev(SNR_GRID, FRAMES)
            if sch != "sse":
                m.md = True
                ser_md = ev(SNR_GRID, FRAMES)
                m.md = False
            else:
                ser_md = ser
            write_csv(DATA / f"rev_budget_final_{sch}{tag(seed)}.csv",
                      ["snr_db", f"ser_{sch}", f"ser_{sch}_md"],
                      list(zip(SNR_GRID, ser, ser_md)))
            per_seed.append(ser_md)
            print(f"[final] {sch} seed {seed} 10dB={ser_md[5]:.4f} "
                  f"20dB={ser_md[10]:.4f}", flush=True)
            del m
            torch.cuda.empty_cache() if DEVICE.type == "cuda" else None
        for i, snr in enumerate(SNR_GRID):
            mean_rows.append((sch, snr, sum(s[i] for s in per_seed) / len(SEEDS),
                              min(s[i] for s in per_seed), max(s[i] for s in per_seed)))
        # mean of the logged curves (identical iteration grid per seed)
        curves = [read_rows(DATA / f"rev_budget_curve_{sch}{tag(s)}.csv")
                  for s in SEEDS]
        assert all(len(c) == len(curves[0]) for c in curves)
        out = []
        for rows in zip(*curves):
            assert len({r["iter"] for r in rows}) == 1
            col = "ser10_md" if "ser10_md" in rows[0] else "ser10"
            out.append((int(rows[0]["iter"]),
                        sum(float(r["time_s"]) for r in rows) / len(rows),
                        float(rows[0]["macs"]),
                        sum(float(r["ser10"]) for r in rows) / len(rows),
                        sum(float(r[col]) for r in rows) / len(rows),
                        sch))
        write_csv(DATA / f"rev_budget_curve_{sch}_mean.csv",
                  ["iter", "time_s", "macs", "ser10", "ser10_md", "scheme"], out)
    write_csv(DATA / "rev_budget_final_mean.csv",
              ["scheme", "snr_db", "ser_mean", "ser_min", "ser_max"], mean_rows)


if __name__ == "__main__":
    main()
