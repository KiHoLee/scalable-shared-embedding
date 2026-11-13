"""Earlier design, mitigation: train the Proposed SSE through
the complex channel simulator of rev_channel.py with imperfect channel
estimates (NMSE drawn per batch from the given value) and, optionally,
frequency-selective fading, then evaluate it under the same conditions.
Same optimizer, batch, iterations, SNR range and loss as the main model.

Output: data/rev_csitrain.csv, models data/rev_sse_csi<nmse>.pt
"""
from __future__ import annotations

import argparse

import torch

from sse_lib import (DATA, DEVICE, D, U, VU, P_MAX, SSE, TRAIN_ITERS,
                     TRAIN_BATCH, TRAIN_SNR_DB, LR, set_seed, write_csv)
import rev_channel as rc
from exp_rev_robust import run


def train_channel(model, nmse, iters=TRAIN_ITERS, batch=TRAIN_BATCH, seed=1):
    set_seed(seed)
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    ce = torch.nn.CrossEntropyLoss()
    for it in range(1, iters + 1):
        dg = torch.randint(VU, (batch, U, P_MAX), device=DEVICE)
        snr = torch.empty(batch).uniform_(*TRAIN_SNR_DB)
        model.calibrate_power(8192)
        x = model.tx_frame(dg).reshape(batch, D) / model.c
        r, w = rc.channel(x, snr, U, nmse=nmse)
        sc = rc.sse_scores(model, r, w, "corr") * model.logit_scale.exp()
        loss = ce(sc.reshape(-1, VU), dg.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    model.calibrate_power()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nmse", default="0.05,0.1")
    ap.add_argument("--frames", type=int, default=1_000_000)
    a = ap.parse_args()
    rows = []
    for nm in [float(v) for v in a.nmse.split(",")]:
        m = SSE(P=P_MAX)
        path = DATA / f"rev_sse_csi{nm:g}.pt"
        if path.exists():
            m.load_state_dict(torch.load(path, map_location="cpu"))
            m.to(DEVICE)
        else:
            train_channel(m, nm)
            torch.save(m.state_dict(), path)
        m.eval()
        for snr in (10.0, 20.0):
            s = run("sse", m, U, snr, a.frames, 100_000, 777, nmse=nm)
            rows.append((nm, snr, float(s.mean()), float(s.max())))
            print("[csitrain]", rows[-1], flush=True)
    write_csv(DATA / "rev_csitrain.csv",
              ["nmse", "snr_db", "ser_SSEcsi", "worst_SSEcsi"], rows)


if __name__ == "__main__":
    main()
