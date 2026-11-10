"""Revision R2.6 / Fig. R3: OMA at unit frame energy over the nine vocabularies
of the sweep, against the proposal.

Convention. Section II fixes unit frame energy and gamma = 1/(d sigma^2) for all
schemes. OMA gives each of the U users d/U = 16 real dimensions, so a user holds
frame energy 1/U. A token of b = log2 V bits, each bit sent as one sign on 16/b of these dimensions, spends energy
1/(U b) per bit, and matched filtering over the dimensions of the bit gives the
per-bit SNR h^2 gamma 16/b. Hence

    SER = E_h[1 - (1 - Q(h sqrt(gamma 16/b)))^b],  h^2 ~ Exp(1).

This is the rule of the manuscript for U = 2 (each bit sent twice) and of
exp_d128.py. exp_scaling_dense.py line 75 instead calls oma_ser at the per-bit
SNR gamma for every V, so its OMA spends only b/16 of the frame energy (one
quarter at V = 16). Those logged columns of scaling.csv are kept here as
ser*_logged for comparison only; scaling.csv itself is left as it is.

Two bit counts are reported, and they differ only where V is not a power of two:
  ser10, ser16            real b = log2 V, the same real-valued b as the logged
                          data (primary column, plotted in Fig. R3)
  ser10_ceil, ser16_ceil  integer B = ceil(log2 V) bits at the same frame energy
ser*_mc is a dimension-level Monte Carlo of the integer case: 16 dimensions are
split over B bits as evenly as possible, each bit carries energy 1/(U B), and
the decision is a matched filter over its dimensions. It also measures the frame
energy (energy_mc). At V = 16, 256, 4096, 65536 it is literal repetition (4, 2,
1 and 1 repeats with 16 mod B = 4 extra dimensions at V = 4096).

Ratios are OMA over the proposal:
  ratio*_best   better of P = 2 and P = 4, single runs of scaling_sse_P.csv (the
                proposal curve of Fig. R3)
  ratio*_ten    better of the two ten-run means, at 4,000 iterations each. The
                runs are rev_factor_*.pt of exp_rev_factor_seeds.py and
                exp_rev_factor_sweep.py where they exist and are trained here by
                the same protocol otherwise (rev_sweep_*.pt, V = 16). All are
                evaluated at 10 and 16 dB over 1e6 frames
                (data/rev_oma_sweep_runs.csv).

Usage: python exp_rev_oma_sweep.py [--tenrun]   (--tenrun trains and evaluates
the runs first, then the table is rewritten).
Output: data/rev_oma_sweep.csv, data/rev_oma_sweep_runs.csv
"""
from __future__ import annotations

import argparse
import csv
import math

import torch

from sse_lib import (D, DATA, DEVICE, SSE, U, eval_ser_sse, oma_ser, rayleigh_gain,
                     set_seed, train_sse, write_csv)

VS = (16, 81, 256, 625, 1296, 4096, 10000, 20736, 65536)
SNRS = [10.0, 16.0]
DIMS = D // U                     # real dimensions per user
FRAMES = 1_000_000
SEEDS = range(1, 11)
RUNS = DATA / "rev_oma_sweep_runs.csv"


def oma_unit_energy(bits: float, snrs=SNRS):
    """SER of OMA at unit frame energy with `bits` bits per token."""
    g = 10.0 * math.log10(DIMS / bits)
    return oma_ser([s + g for s in snrs], bits=bits)


@torch.no_grad()
def oma_mc(bits: int, snr_db: float, frames: int = FRAMES, chunk: int = 100_000,
           seed: int = 2026):
    """Dimension-level Monte Carlo of integer-bit OMA at unit frame energy.
    Returns (SER, mean frame energy)."""
    rep = [DIMS // bits + (i < DIMS % bits) for i in range(bits)]
    A = torch.zeros(bits, DIMS, device=DEVICE)      # bit-to-dimension amplitudes
    k = 0
    for i, r in enumerate(rep):
        A[i, k:k + r] = math.sqrt(1.0 / (U * bits) / r)
        k += r
    sigma = math.sqrt(1.0 / (D * 10.0 ** (snr_db / 10.0)))
    torch.manual_seed(seed + int(10 * snr_db) + bits)
    err = tot = 0
    energy = 0.0
    for n0 in range(0, frames, chunk):
        n = min(chunk, frames - n0)
        b = torch.randint(0, 2, (n, U, bits), device=DEVICE) * 2.0 - 1.0
        x = b @ A                                   # (n,U,DIMS)
        energy += float(x.pow(2).sum(dim=(1, 2)).sum())
        h = rayleigh_gain((n, U, 1))
        y = h * x + sigma * torch.randn(n, U, DIMS, device=DEVICE)
        z = y @ A.T                                 # matched filter per bit
        err += int(((z * b) < 0).any(dim=2).sum())
        tot += n * U
    return err / tot, energy / frames


def load_or_train(vu: int, P: int, seed: int) -> SSE:
    """A 4,000-iteration proposal run, by the protocol of exp_rev_factor_seeds.py."""
    m = SSE(P=P, vu=vu)
    for name in (f"rev_factor_{vu}_{P}_s{seed}.pt", f"rev_sweep_{vu}_{P}_s{seed}.pt"):
        if (DATA / name).exists():
            m.load_state_dict(torch.load(DATA / name, map_location="cpu"))
            return m.to(DEVICE).eval()
    set_seed(seed)
    m = SSE(P=P, vu=vu)
    train_sse(m, seed=seed)
    m.to(DEVICE).eval()
    torch.save({k: v.cpu() for k, v in m.state_dict().items()}, DATA / name)
    return m


def read(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def ten_runs():
    rows = [(int(r["V"]), int(r["vu"]), int(r["P"]), int(r["seed"]),
             float(r["ser10"]), float(r["ser16"])) for r in read(RUNS)] if RUNS.exists() else []
    done = {(r[1], r[2], r[3]) for r in rows}
    for V in VS:
        for P in (2, 4):
            vu = round(V ** (1 / P))
            assert vu ** P == V
            for seed in SEEDS:
                if (vu, P, seed) in done:
                    continue
                m = load_or_train(vu, P, seed)
                set_seed(seed)              # the channel draws of the evaluation
                s = eval_ser_sse(m, SNRS, frames=FRAMES)
                rows.append((V, vu, P, seed, s[0], s[1]))
                print(rows[-1], flush=True)
                write_csv(RUNS, ["V", "vu", "P", "seed", "ser10", "ser16"], rows)
                del m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tenrun", action="store_true")
    a = ap.parse_args()
    if a.tenrun:
        ten_runs()

    logged = {int(float(r["V"])): r for r in read(DATA / "scaling.csv")}
    best = {int(float(r["V"])): r for r in read(DATA / "scaling_sse_P.csv")}
    ten = {}
    if RUNS.exists():
        for r in read(RUNS):
            ten.setdefault((int(r["V"]), int(r["P"])), []).append((float(r["ser10"]), float(r["ser16"])))

    def tenmean(V, P, i):
        v = ten.get((V, P), [])
        return sum(x[i] for x in v) / len(v) if len(v) == len(SEEDS) else float("nan")

    rows = []
    for V in VS:
        b = math.log2(V)
        B = math.ceil(b - 1e-9)
        s = oma_unit_energy(b)
        sc = oma_unit_energy(B)
        m10, en = oma_mc(B, 10.0)
        m16, _ = oma_mc(B, 16.0)
        b10 = min(float(best[V]["ser10_P2"]), float(best[V]["ser10_P4"]))
        b16 = min(float(best[V]["ser16_P2"]), float(best[V]["ser16_P4"]))
        t = [tenmean(V, 2, 0), tenmean(V, 4, 0), tenmean(V, 2, 1), tenmean(V, 4, 1)]
        t10, t16 = min(t[0], t[1]), min(t[2], t[3])
        rows.append((V, b, b / DIMS, s[0], s[1],
                     float(logged[V]["ser10_OMA"]), float(logged[V]["ser16_OMA"]),
                     b10, b16, s[0] / b10, s[1] / b16,
                     float(logged[V]["ser10_OMA"]) / b10,
                     sc[0], sc[1], sc[0] / b10, sc[1] / b16, m10, m16, en,
                     t[0], t[1], t[2], t[3], s[0] / t10, s[1] / t16,
                     sc[0] / t10, sc[1] / t16))
        print(rows[-1], flush=True)
    write_csv(DATA / "rev_oma_sweep.csv",
              ["V", "bits", "energy_logged", "ser10", "ser16",
               "ser10_logged", "ser16_logged", "best10", "best16",
               "ratio10_best", "ratio16_best", "ratio10_logged",
               "ser10_ceil", "ser16_ceil", "ratio10_ceil_best", "ratio16_ceil_best",
               "ser10_mc", "ser16_mc", "energy_mc",
               "ten10_P2", "ten10_P4", "ten16_P2", "ten16_P4",
               "ratio10_ten", "ratio16_ten", "ratio10_ceil_ten", "ratio16_ceil_ten"], rows)


if __name__ == "__main__":
    main()
