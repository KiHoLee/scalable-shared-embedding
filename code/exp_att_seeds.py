"""Seed-averaged Full-attention scaling curve.

The Full-attention points in the vocabulary sweep are each a single
training run, and the attention receiver has the largest run-to-run
convergence variance of the trained schemes, which renders as a
jagged curve. Monte Carlo noise is negligible (>=2e5 frames per
evaluation). This script trains S=5 seeds per vocabulary size,
averages the evaluated SER at 10 and 16 dB, stores the per-seed raw
values in data/scaling_att_seeds.csv, and rewrites the
ser10_FullATT / ser16_FullATT columns of data/scaling.csv with the
seed means. Seed 1 reuses the cached models from exp_scaling_dense.
"""
from __future__ import annotations

import time

import torch

from sse_lib import (DATA, DEVICE, P_MAX, U, TRAIN_SNR_DB, set_seed,
                     write_csv)
from exp_sota import FullAttention, eval_ser_att

VUS = (2, 3, 4, 5, 6, 8, 10, 12, 16)
SEEDS = (1, 2, 3, 4, 5)
EVAL_FRAMES = 200_000


def att_path(V: int, seed: int):
    if seed == 1:
        return DATA / ("full_att.pt" if V == 16 ** P_MAX
                       else f"full_att_V{V}.pt")
    return DATA / f"full_att_V{V}_s{seed}.pt"


def get_or_train(V: int, seed: int, iters: int = 4000, batch: int = 256,
                 lr: float = 3e-3):
    path = att_path(V, seed)
    model = FullAttention(V)
    if path.exists():
        model.load_state_dict(torch.load(path, map_location="cpu"))
        return model.to(DEVICE)
    set_seed(seed)
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ce = torch.nn.CrossEntropyLoss()
    t0 = time.time()
    for _ in range(iters):
        idx = torch.randint(V, (batch, U), device=DEVICE)
        snr = torch.empty(batch).uniform_(*TRAIN_SNR_DB)
        model.calibrate_power(8192)
        scores = model(idx, snr) * model.logit_scale.exp()
        loss = ce(scores.reshape(-1, V), idx.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    model.calibrate_power()
    torch.save(model.state_dict(), path)
    print(f"[train] att V={V} seed={seed}: {time.time()-t0:.1f}s",
          flush=True)
    return model


def main():
    rows = []
    means = {}
    for vu in VUS:
        V = vu ** P_MAX
        s10, s16 = [], []
        for seed in SEEDS:
            m = get_or_train(V, seed)
            a, b = eval_ser_att(m, [10.0, 16.0], frames=EVAL_FRAMES)
            s10.append(a); s16.append(b)
            del m
            print(f"[eval] V={V} seed={seed}: {a:.5f} / {b:.5f}",
                  flush=True)
        m10 = sum(s10) / len(s10)
        m16 = sum(s16) / len(s16)
        sd10 = (sum((x - m10) ** 2 for x in s10) / (len(s10) - 1)) ** 0.5
        means[V] = (m10, m16)
        rows.append((V, m10, sd10, m16, *s10, *s16))
        print(f"[seedavg] V={V}: mean10={m10:.5f} sd10={sd10:.5f}",
              flush=True)
    write_csv(DATA / "scaling_att_seeds.csv",
              ["V", "mean10", "sd10", "mean16",
               *[f"s10_seed{s}" for s in SEEDS],
               *[f"s16_seed{s}" for s in SEEDS]], rows)

    # rewrite the FullATT columns of scaling.csv with the seed means
    lines = (DATA / "scaling.csv").read_text().strip().split("\n")
    hdr = lines[0].split(",")
    i10 = hdr.index("ser10_FullATT")
    i16 = hdr.index("ser16_FullATT")
    out = [lines[0]]
    for ln in lines[1:]:
        f = ln.split(",")
        V = int(float(f[0]))
        f[i10] = f"{means[V][0]:.10g}"
        f[i16] = f"{means[V][1]:.10g}"
        out.append(",".join(f))
    (DATA / "scaling.csv").write_text("\n".join(out) + "\n")
    print("[csv] scaling.csv FullATT columns updated with seed means")


if __name__ == "__main__":
    main()
