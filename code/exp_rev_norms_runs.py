"""Revision R1.4: masked-norm spread over the runs beyond the reported model.

No training. For each run, ||b_i . mu_u|| relative to its per-user mean, as in
exp_rev_analysis (A1) and exp_rev_full_long_norms.py, and the coefficient of
variation per user with the population standard deviation (the definition behind
the letter's 0.8-1.6 percent for the reported model) and the sample one. Also the
SER of correlation and minimum-distance decoding at 10 and 20 dB on the same
Monte Carlo draws, 1e6 frames, and the relative gap, since the norm spread acts on
the SER only through that gap. Groups:
  reported      sse_P4.pt (reference)
  other_4k_16_4 the 15 other 4,000-iteration runs of (16,4): rev_sse_seed1..5 and
                rev_factor_16_4_s1..10
  long_16_4     the ten long-budget runs rev_sse_long_md{,_s2,...,_s10}.pt
  runs_256_2    the ten runs rev_factor_256_2_s1..10
The rows with model ALL_min and ALL_max hold the minimum and maximum of each
column over the group.

Output: data/rev_norms_runs.csv
"""
import torch

import exp_rev_analysis as A
from exp_rev_factor_mech import CHUNK, FRAMES, load_run
from sse_lib import DATA, write_csv

GROUPS = {"reported": [(16, 4, "sse_P4.pt")],
          "other_4k_16_4": [(16, 4, f"rev_sse_seed{s}.pt") for s in range(1, 6)]
          + [(16, 4, f"rev_factor_16_4_s{s}.pt") for s in range(1, 11)],
          "long_16_4": [(16, 4, "rev_sse_long_md.pt")]
          + [(16, 4, f"rev_sse_long_md_s{s}.pt") for s in range(2, 11)],
          "runs_256_2": [(256, 2, f"rev_factor_256_2_s{s}.pt") for s in range(1, 11)]}
COLS = ["rel_min", "rel_max", "max_dev", "cv_pop_min", "cv_pop_max", "cv_sample_max",
        "ser_corr10", "ser_md10", "gap10_pct", "ser_corr20", "ser_md20", "gap20_pct"]


@torch.no_grad()
def one_run(vu, P, name):
    m = load_run(vu, P, name)
    n = (m.unit_codebook()[None] * m.masks()[:, None, :]).double().norm(dim=-1)
    rel = n / n.mean(1, keepdim=True)
    cv_p = n.std(1, unbiased=False) / n.mean(1)
    cv_s = n.std(1) / n.mean(1)
    out = [float(rel.min()), float(rel.max()), float((rel - 1).abs().max()),
           float(cv_p.min()), float(cv_p.max()), float(cv_s.max())]
    for snr in (10.0, 20.0):
        c = A.ser(m, snr, frames=FRAMES, chunk=CHUNK, rule="corr")
        d = A.ser(m, snr, frames=FRAMES, chunk=CHUNK, rule="md")
        out += [c, d, 100.0 * (c - d) / d]
    return out


def main():
    rows = []
    for group, runs in GROUPS.items():
        vals = []
        for vu, P, name in runs:
            v = one_run(vu, P, name)
            vals.append(v)
            rows.append([group, vu, P, name] + v)
            print(rows[-1], flush=True)
            write_csv(DATA / "rev_norms_runs.csv", ["group", "vu", "P", "model"] + COLS, rows)
        rows.append([group, runs[0][0], runs[0][1], "ALL_min"] + [min(c) for c in zip(*vals)])
        rows.append([group, runs[0][0], runs[0][1], "ALL_max"] + [max(c) for c in zip(*vals)])
        write_csv(DATA / "rev_norms_runs.csv", ["group", "vu", "P", "model"] + COLS, rows)


if __name__ == "__main__":
    main()
