"""Revision R2.1/R2.2/R2.6: a fixed packing kappa_u under rho_u = 0.

Builds, inside the SSE class, the binary hypercube with disjoint masks: each
user owns b = log2 Vu of the L = d/P coordinates of a period, the shared unit
codeword b_i repeats the b-bit pattern of index i on all U blocks, and the
L - U b remaining coordinates stay unused. This fits whenever U log2 Vu <= L,
that is U log2 V <= d, and needs no divisibility of L/U by b.

Reports the mask overlap, the largest cross product between candidates of two
users, rho_u by the closed form (8) of exp_rev_analysis.rho_closed_form, and
kappa_u against the design value 4/(U log2 Vu). Monte Carlo SER at 10 and 20 dB
through the letter's real model (eval_ser_sse), against the closed form
E_h[1 - (1 - Q(h sqrt(gamma d/(U b P))))^(b P)]. At (Vu,P) = (16,4) this design
is the binary OMA reference, so it bounds a scheme that needs no training and
says nothing about what training reaches.

(16,1), (16,2), (16,4), (4,8), (256,2) are the requested points, (2,16) and the
non-divisible (8,4) are added checks, and (16,8) is listed to record that the
design does not fit there (U log2 V = 128 > d = 64).

Output: data/rev_hypercube.csv
"""
import math

import torch

from exp_rev_analysis import rho_closed_form
from sse_lib import (D, DATA, DEVICE, SSE, U, eval_ser_sse, oma_ser, set_seed,
                     write_csv)

PAIRS = [(16, 1), (16, 2), (16, 4), (4, 8), (256, 2), (2, 16), (8, 4), (16, 8)]
SNRS = [10.0, 20.0]
FRAMES = 1_000_000


def build(vu, P, users=U, d=D):
    """Returns the SSE model of the hypercube at (vu, P), or None if it does
    not fit in a period."""
    b = int(round(math.log2(vu)))
    assert 2 ** b == vu
    L = d // P
    if users * b > L:
        return None
    bits = torch.tensor([[2.0 * ((i >> k) & 1) - 1.0 for k in range(b)]
                         for i in range(vu)], dtype=torch.float64)
    B = torch.zeros(vu, L, dtype=torch.float64)
    W = torch.zeros(users, L, dtype=torch.float64)
    for u in range(users):
        B[:, u * b:(u + 1) * b] = bits
        W[u, u * b:(u + 1) * b] = 1.0
    m = SSE(P=P, vu=vu, d=d, users=users).double()
    m.B.data.copy_(B)
    m.W.data.copy_(W)
    m.calibrate_power(4096)     # every frame has the same energy here
    return m


@torch.no_grad()
def geometry(m):
    Bn, mu, c = m.unit_codebook(), m.masks(), float(m.c)
    cand = Bn[None] * mu[:, None]                       # (U,Vu,L)
    kap, cross = [], 0.0
    for u in range(m.users):
        d2 = torch.cdist(cand[u], cand[u]) ** 2 / c ** 2
        kap.append(float(d2[~torch.eye(m.vu, dtype=torch.bool)].min()))
        for v in range(m.users):
            if v != u:
                cross = max(cross, float((cand[u] @ cand[v].T).abs().max()))
    G = (mu @ mu.T) / m.L
    off = float((G - torch.eye(m.users, dtype=G.dtype)).abs().max())
    s = torch.linalg.svdvals(cand[0] - cand[0].mean(0))
    rank = int((s ** 2 / (s ** 2).sum() > 1e-12).sum())
    return kap, cross, off, rank, c ** 2


def main():
    rows = []
    hdr = ["Vu", "P", "L", "b", "L_over_U", "U_b", "fits", "rank_user0", "c2",
           "kappa_min", "kappa_max", "kappa_design", "rho_max", "max_cross",
           "mask_offdiag", "ser10", "ser20", "ser10_closed", "ser20_closed"]
    for vu, P in PAIRS:
        set_seed(1)
        b = int(round(math.log2(vu)))
        L = D // P
        design = 4.0 / (U * b)
        m = build(vu, P)
        if m is None:
            nan = float("nan")
            rows.append((vu, P, L, b, L / U, U * b, 0, 0, nan, nan, nan, design,
                         nan, nan, nan, nan, nan, nan, nan))
        else:
            kap, cross, off, rank, c2 = geometry(m)
            rho = rho_closed_form(m)
            ser = eval_ser_sse(m.float().to(DEVICE), SNRS, frames=FRAMES)
            shift = 10 * math.log10(D / (U * b * P))
            cl = oma_ser([s + shift for s in SNRS], bits=b * P)
            rows.append((vu, P, L, b, L / U, U * b, 1, rank, float(c2),
                         min(kap), max(kap), design, max(rho), cross, off,
                         ser[0], ser[1], cl[0], cl[1]))
        print(dict(zip(hdr, rows[-1])), flush=True)
        write_csv(DATA / "rev_hypercube.csv", hdr, rows)


if __name__ == "__main__":
    main()
