"""Numerical check of the closed-form scaling bound (Remark 1 of the
earlier design):

    SER_u <= 8 P ln(e P (Vu-1)/2) / (kappa_u d gamma),
    kappa_u = min_{i!=j} ||(b_i - b_j) . mu_u||^2 / c^2,

against the intermediate integral form x^{-1} int min{1, M Q(sqrt t)} dt
and the Monte Carlo SER of the trained models (rev_scaleP.csv,
rev_factor.csv). Writes data/rev_scalingbound.csv.
"""
from __future__ import annotations

import csv
import math

import numpy as np
import torch

from sse_lib import DATA, D, SSE, write_csv

MODELS = [(16, 1, "rev_sse_Vu16_P1.pt"), (16, 2, "sse_P2.pt"),
          (16, 4, "sse_P4.pt"), (4, 8, "rev_sse_Vu4_P8.pt"),
          (256, 2, "sse_Vu256_P2.pt")]


def kappas(vu, P, name):
    m = SSE(P=P, vu=vu)
    m.load_state_dict(torch.load(DATA / name, map_location="cpu"))
    Bn, mu, c = m.unit_codebook(), m.masks(), float(m.c)
    out = []
    for u in range(m.users):
        cand = Bn * mu[u]
        dd = torch.cdist(cand, cand) + torch.eye(vu) * 1e9
        out.append(float(dd.min() ** 2) / c ** 2)
    return out


def qfun(x):
    from math import erfc
    return 0.5 * np.vectorize(erfc)(np.asarray(x) / math.sqrt(2))


def integral_form(M, x, n=400_000, tmax=200.0):
    t = (np.arange(n) + 0.5) * (tmax / n)
    return float(np.minimum(1.0, M * qfun(np.sqrt(t))).sum() * (tmax / n) / x)


def measured():
    meas = {}
    for r in csv.DictReader(open(DATA / "rev_scaleP.csv")):
        meas[(16, int(r["P"]), float(r["snr_db"]))] = float(r["ser"])
    for r in csv.DictReader(open(DATA / "rev_factor.csv")):
        meas[(int(r["Vu"]), int(r["P"]), 10.0)] = float(r["ser10"])
        meas[(int(r["Vu"]), int(r["P"]), 20.0)] = float(r["ser20"])
    return meas


def main():
    meas = measured()
    rows = []
    for vu, P, name in MODELS:
        ks = kappas(vu, P, name)
        M = P * (vu - 1)
        for snr in (10.0, 20.0):
            g = 10 ** (snr / 10)
            closed = np.mean([8 * P * math.log(math.e * M / 2) / (k * D * g)
                              for k in ks])
            integ = np.mean([integral_form(M, (k / P) * D * g / 4) for k in ks])
            ms = meas.get((vu, P, snr), float("nan"))
            rows.append((vu, P, snr, float(np.mean(ks)), ms, integ, closed))
            print(rows[-1], flush=True)
    write_csv(DATA / "rev_scalingbound.csv",
              ["Vu", "P", "snr_db", "kappa_mean", "ser_mc", "integral",
               "closed_form"], rows)


if __name__ == "__main__":
    main()
