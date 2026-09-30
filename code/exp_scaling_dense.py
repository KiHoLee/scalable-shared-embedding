"""Dense vocabulary-scaling sweep with the full comparison set.

The original sweep used four grid points (Vu in {2,4,8,16}) and two
schemes, which renders as an angular two-curve polyline. This script
densifies the grid to Vu in {2,3,4,5,6,8,10,12,16}, i.e.
V = Vu^4 in {16, 81, 256, 625, 1296, 4096, 10000, 20736, 65536}, and
evaluates all compared schemes at every V:

  Proposed SSE    trained per V (cached as sse_Vu*_P4.pt)
  Full codebook   trained per V (cached as full_V*.pt)
  Full attention  trained per V (cached as full_att_V*.pt)
  ToDMA           OMP over a V-column random dictionary, no training
  OMA             semi-analytic with b = log2(V) bits per user

Identical training budget and evaluation protocol throughout.
Overwrites data/scaling.csv.
"""
from __future__ import annotations

import json
import math
import time

import torch

from sse_lib import (DATA, DEVICE, P_MAX, SEED, U, set_seed, eval_ser_sse,
                     eval_ser_full, oma_ser, write_csv, TRAIN_SNR_DB)
from exp_main import get_or_train_sse, get_or_train_full, EVAL_FRAMES
from exp_sota import FullAttention, eval_ser_att, todma_ser

VUS = (2, 3, 4, 5, 6, 8, 10, 12, 16)


def get_or_train_att(V: int, iters: int = 4000, batch: int = 256,
                     lr: float = 3e-3, seed: int = SEED):
    path = DATA / ("full_att.pt" if V == 16 ** P_MAX else f"full_att_V{V}.pt")
    model = FullAttention(V)
    if path.exists():
        model.load_state_dict(torch.load(path, map_location="cpu"))
        return model.to(DEVICE)
    set_seed(seed)
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ce = torch.nn.CrossEntropyLoss()
    t0 = time.time()
    for it in range(1, iters + 1):
        idx = torch.randint(V, (batch, U), device=DEVICE)
        snr = torch.empty(batch).uniform_(*TRAIN_SNR_DB)
        model.calibrate_power(8192)
        scores = model(idx, snr) * model.logit_scale.exp()
        loss = ce(scores.reshape(-1, V), idx.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    model.calibrate_power()
    torch.save(model.state_dict(), path)
    print(f"[train] att V={V}: {time.time()-t0:.1f}s", flush=True)
    return model


def main():
    set_seed(SEED)
    rows = []
    for vu in VUS:
        Vp = vu ** P_MAX
        m_sse, _ = get_or_train_sse(P_MAX, f"sse Vu={vu}", vu=vu)
        m_full, _ = get_or_train_full(Vp)
        m_att = get_or_train_att(Vp)
        s10_sse = eval_ser_sse(m_sse, [10.0], frames=EVAL_FRAMES)[0]
        s16_sse = eval_ser_sse(m_sse, [16.0], frames=EVAL_FRAMES)[0]
        s10_full = eval_ser_full(m_full, [10.0], frames=EVAL_FRAMES // 4)[0]
        s16_full = eval_ser_full(m_full, [16.0], frames=EVAL_FRAMES // 4)[0]
        s10_att = eval_ser_att(m_att, [10.0], frames=EVAL_FRAMES // 4)[0]
        s16_att = eval_ser_att(m_att, [16.0], frames=EVAL_FRAMES // 4)[0]
        s10_td, s16_td = todma_ser([10.0, 16.0], frames=100_000, V=Vp)
        b = math.log2(Vp)
        s10_oma, s16_oma = oma_ser([10.0, 16.0], bits=b)
        rows.append((Vp, s10_sse, s16_sse, m_sse.n_params(),
                     s10_full, s16_full, m_full.n_params(),
                     s10_att, s16_att, s10_td, s16_td, s10_oma, s16_oma))
        print("[scaling-dense]", rows[-1], flush=True)
    write_csv(DATA / "scaling.csv",
              ["V", "ser10_SSE", "ser16_SSE", "par_SSE",
               "ser10_Full", "ser16_Full", "par_Full",
               "ser10_FullATT", "ser16_FullATT",
               "ser10_ToDMA", "ser16_ToDMA",
               "ser10_OMA", "ser16_OMA"], rows)


if __name__ == "__main__":
    main()
