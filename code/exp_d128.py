"""Exploratory: d=128 operating point (2x channel uses per token, half rate).

Convention matches the ToDMA x2 case already in the letter: per-dimension
SNR is kept at gamma, i.e. sigma^2 = 1/(dim * gamma) with dim = 128.  This
is implemented by overriding sse_lib.D before any sigma computation
(snr_to_sigma2 reads the module global at call time).

Schemes trained fresh at d=128 (same budget: 4,000 iters, batch 256,
Adam 3e-3, train seed 1, eval seed 777, 2M frames):
  - SSE  P=4, L=32, Vu=16  (512 + 128 = 640 trained values)
  - Full codebook V=65,536 (65,536*128 + 4*128 = 8,389,120 values)
  - Single-user SSE bound  (users=1)
OMA d=128: 32 dims for 16 bits per user -> repetition-2 coherent combining,
SER = oma_ser at snr + 10*log10(2).  ToDMA d=128 = ser_ToDMA2 column of
data/sota_snr.csv (already measured).
"""
import json
import math

import torch

import sse_lib
sse_lib.D = 128          # per-dim SNR convention at dim=128 (ToDMA x2 parity)

from sse_lib import (DATA, DEVICE, SSE, FullCodebook, train_sse, train_full,
                     eval_ser_sse, eval_ser_full, oma_ser)

SNRS = [0.0, 5.0, 10.0, 15.0, 20.0]
D128 = 128


def get(name, build, train):
    path = DATA / f"{name}.pt"
    model = build()
    if path.exists():
        model.load_state_dict(torch.load(path, map_location=DEVICE))
        model.to(DEVICE)
    else:
        train(model)
        model.calibrate_power()
        torch.save(model.state_dict(), path)
    return model


def main():
    sse = get("sse_P4_d128", lambda: SSE(P=4, d=D128).to(DEVICE),
              lambda m: train_sse(m))
    single = get("sse_single_d128", lambda: SSE(P=4, d=D128, users=1).to(DEVICE),
                 lambda m: train_sse(m))
    full = get("full_V65536_d128",
               lambda: FullCodebook(65536, d=D128).to(DEVICE),
               lambda m: train_full(m))

    print("params:", sse.n_params(), single.n_params(), full.n_params(),
          flush=True)
    ser_sse = eval_ser_sse(sse, SNRS)
    print("SSE d=128:", [f"{v:.5f}" for v in ser_sse], flush=True)
    ser_single = eval_ser_sse(single, SNRS)
    print("Single d=128:", [f"{v:.5f}" for v in ser_single], flush=True)
    ser_oma = oma_ser([s + 10 * math.log10(2) for s in SNRS])
    print("OMA d=128:", [f"{v:.5f}" for v in ser_oma], flush=True)
    ser_full = eval_ser_full(full, SNRS)
    print("Full d=128:", [f"{v:.5f}" for v in ser_full], flush=True)

    with open(DATA / "d128_snr.json", "w") as f:
        json.dump({"snr": SNRS, "ser_SSE": ser_sse, "ser_Full": ser_full,
                   "ser_OMA": ser_oma, "ser_Single": ser_single,
                   "note": "d=128, sigma^2=1/(128*gamma); ToDMA d=128 = "
                           "ser_ToDMA2 in sota_snr.csv"}, f, indent=1)
    print("saved data/d128_snr.json", flush=True)


if __name__ == "__main__":
    main()
