"""Revision R1.2: SNR gap of the proposal to the full codebook with minimum
distance decoding, against the vocabulary, at 10 and 16 dB, flat Rayleigh channel
of sse_lib. The gap at SNR s is s minus the SNR at which the full-codebook curve
reaches the SER of the proposal at s (log-SER interpolation on a 1 dB grid).

Budgets compared. These are unequal on purpose: the full codebook gets the budget
at which it is close to convergence, so the gap is not inflated by an
under-converged baseline.
  budget "A", V = 256, 1296, 4096, 10000, 65536:
    proposal: Vu = V^(1/4), P = 4, six runs at 4,000 iterations, constant learning
      rate, correlation decoding. For V < 65,536, seeds 1-6 of the protocol of
      exp_rev_factor_seeds.py (see exp_rev_oma_sweep.load_or_train). For V = 65,536,
      the six released runs sse_P4.pt (best of six) and rev_sse_seed1-5.pt.
    full codebook: one run, seed 1, 40,000 iterations, cosine decay, minimum
      distance decoding (data/rev_gapV_full_V*.pt).
    gap_db uses the mean SER of the six runs; gap_min and gap_max are per-run gaps.
  budget "B", V = 65,536 only: three-run means at 200,000 iterations, cosine decay,
    both schemes, read from rev_budget_final_mean.csv (2 dB grid, 2e6 frames), so
    no new simulation. Range: the lowest and highest of the proposal's runs.

Monte Carlo: 1e6 frames per point. rev_gap_curves.csv and rev_gap_db.csv of
exp_rev_gap_curves.py cover the channels at V = 65,536 and are not repeated here.

Usage: python exp_rev_gap_vs_V.py [full|sse|gap]   (default: all three, in turn)
Output: data/rev_gap_vs_V.csv, and the raw points data/rev_gap_vs_V_full.csv and
data/rev_gap_vs_V_sse.csv
"""
from __future__ import annotations

import csv
import math
import sys
import time

import numpy as np
import torch

from exp_rev_oma_sweep import load_or_train
from sse_lib import (DATA, DEVICE, LR, P_MAX, SSE, TRAIN_BATCH, TRAIN_SNR_DB,
                     FullCodebook, eval_ser_full, eval_ser_sse, set_seed, write_csv)

VS = (256, 1296, 4096, 10000, 65536)
SNRS = (10.0, 16.0)
GRID = [float(s) for s in range(6, 17)]
FRAMES = 1_000_000
SSE_ITERS = 4000
FULL_ITERS = 40000
FULL_CSV = DATA / "rev_gap_vs_V_full.csv"
SSE_CSV = DATA / "rev_gap_vs_V_sse.csv"
RELEASED = [("reported", "sse_P4.pt")] + [(f"seed{s}", f"rev_sse_seed{s}.pt") for s in range(1, 6)]


def read(path):
    if not path.exists():
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def train_full_cosine(V: int, iters: int = FULL_ITERS, seed: int = 1) -> FullCodebook:
    path = DATA / f"rev_gapV_full_V{V}.pt"
    m = FullCodebook(V)
    if path.exists():
        m.load_state_dict(torch.load(path, map_location="cpu"))
        return m.to(DEVICE).eval()
    set_seed(seed)
    m = FullCodebook(V).to(DEVICE)
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
    ce = torch.nn.CrossEntropyLoss()
    t0 = time.time()
    for it in range(1, iters + 1):
        idx = torch.randint(V, (TRAIN_BATCH, m.users), device=DEVICE)
        snr = torch.empty(TRAIN_BATCH).uniform_(*TRAIN_SNR_DB)
        m.calibrate_power(8192)
        loss = ce((m(idx, snr) * m.logit_scale.exp()).reshape(-1, V), idx.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        if it % 5000 == 0:
            print(f"[train] full V={V} it={it} {time.time() - t0:.0f}s", flush=True)
    m.calibrate_power()
    torch.save({k: v.cpu() for k, v in m.state_dict().items()}, path)
    return m.eval()


def stage_full():
    rows = [(int(r["V"]), int(r["iters"]), float(r["snr_db"]), float(r["ser"]))
            for r in read(FULL_CSV)]
    done = {(r[0], r[2]) for r in rows}
    for V in VS:
        m = train_full_cosine(V)
        m.md = True
        for snr in GRID:
            if (V, snr) in done:
                continue
            t0 = time.time()
            set_seed(int(10 * snr))         # the channel draws of the evaluation
            # eval_ser_full bounds the chunk itself from V = 4,096 on
            s = eval_ser_full(m, [snr], frames=FRAMES, chunk=min(20000, max(512, (1 << 23) // V)))[0]
            rows.append((V, FULL_ITERS, snr, s))
            print(rows[-1], f"{time.time() - t0:.0f}s", flush=True)
            write_csv(FULL_CSV, ["V", "iters", "snr_db", "ser"], rows)
        del m
        torch.cuda.empty_cache()


def stage_sse():
    rows = []
    for V in VS:
        vu = round(V ** (1 / P_MAX))
        assert vu ** P_MAX == V
        for k in range(6):
            if V == 65536:
                run, f = RELEASED[k]
                m = SSE(P=P_MAX)
                m.load_state_dict(torch.load(DATA / f, map_location="cpu"))
                m.to(DEVICE).eval()
            else:
                run = f"seed{k + 1}"
                m = load_or_train(vu, P_MAX, k + 1)
            set_seed(k + 1)                 # the channel draws of the evaluation
            s = eval_ser_sse(m, list(SNRS), frames=FRAMES)
            rows.append((V, vu, P_MAX, SSE_ITERS, run, s[0], s[1]))
            print(rows[-1], flush=True)
            write_csv(SSE_CSV, ["V", "vu", "P", "iters", "run", "ser10", "ser16"], rows)
            del m


def gap(ser: float, snr: float, grid, curve) -> float:
    """SNR gap of `ser` at `snr` to a curve (grid, SER), NaN outside the grid."""
    curve = np.asarray(curve)
    if not curve[-1] <= ser <= curve[0]:
        return float("nan")
    return snr - float(np.interp(-math.log(ser), -np.log(curve), grid))


def stage_gap():
    full = read(FULL_CSV)
    sse = read(SSE_CSV)
    rows = []
    for V in VS:
        cur = sorted((float(r["snr_db"]), float(r["ser"])) for r in full if int(r["V"]) == V)
        grid, ser = zip(*cur)
        runs = [r for r in sse if int(r["V"]) == V]
        assert len(grid) == len(GRID) and len(runs) == 6
        for snr in SNRS:
            col = f"ser{int(snr)}"
            v = [float(r[col]) for r in runs]
            mean = sum(v) / len(v)
            g = [gap(x, snr, grid, ser) for x in v]
            rows.append(("A", V, int(snr), SSE_ITERS, 6, FULL_ITERS, 1,
                         mean, min(v), max(v), ser[grid.index(snr)],
                         gap(mean, snr, grid, ser), min(g), max(g)))
    tab = read(DATA / "rev_budget_final_mean.csv")
    pick = lambda sch: sorted((float(r["snr_db"]), r) for r in tab if r["scheme"] == sch)
    fs, ss = pick("full"), pick("sse")
    grid = [x[0] for x in fs]
    ser = [float(x[1]["ser_mean"]) for x in fs]
    for snr in SNRS:
        r = dict(ss)[snr]
        mean, lo, hi = float(r["ser_mean"]), float(r["ser_min"]), float(r["ser_max"])
        rows.append(("B", 65536, int(snr), 200000, 10, 200000, 10, mean, lo, hi,
                     ser[grid.index(snr)], gap(mean, snr, grid, ser),
                     gap(lo, snr, grid, ser), gap(hi, snr, grid, ser)))
    for r in rows:
        print(r, flush=True)
    write_csv(DATA / "rev_gap_vs_V.csv",
              ["budget", "V", "snr_db", "sse_iters", "sse_runs", "full_iters", "full_runs",
               "ser_sse_mean", "ser_sse_min", "ser_sse_max", "ser_full",
               "gap_db", "gap_min", "gap_max"], rows)


if __name__ == "__main__":
    which = sys.argv[1:] or ["full", "sse", "gap"]
    for w in which:
        {"full": stage_full, "sse": stage_sse, "gap": stage_gap}[w]()
