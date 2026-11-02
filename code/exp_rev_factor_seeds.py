"""Revision R2.6/R2.2: run-to-run spread of the factorization comparison.

Trains every (Vu, P) pair of the selection-rule discussion over ten seeds at
the 4,000-iteration budget and reports SER, kappa_u, the largest off-diagonal
entry of the mask Gram matrix, and bound (7). The rule and the "perform alike"
statement otherwise rest on single runs against a best-of-six model.

Output: data/rev_factor_seeds.csv. Models: data/rev_factor_{Vu}_{P}_s{seed}.pt
"""
import argparse

import torch

from sse_lib import sse_union_bound
from sse_lib import DATA, DEVICE, SSE, eval_ser_sse, set_seed, train_sse, write_csv

PAIRS = [(16, 4), (256, 2), (4, 8), (2, 16), (36, 2), (6, 4), (64, 2), (8, 4)]


def kappa(m):
    Bn, mu, c = m.unit_codebook().detach(), m.masks().detach(), float(m.c)
    out = []
    for u in range(m.users):
        cand = Bn * mu[u]
        dd = torch.cdist(cand, cand) + torch.eye(m.vu, device=cand.device) * 1e9
        out.append(float(dd.min() ** 2) / c ** 2)
    return sum(out) / len(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--frames", type=int, default=1_000_000)
    a = ap.parse_args()
    rows = []
    for vu, P in PAIRS:
        for seed in range(1, a.seeds + 1):
            set_seed(seed)
            m = SSE(P=P, vu=vu)
            train_sse(m, seed=seed)
            m.to(DEVICE).eval()
            torch.save({k: v.cpu() for k, v in m.state_dict().items()},
                       DATA / f"rev_factor_{vu}_{P}_s{seed}.pt")
            s = eval_ser_sse(m, [10.0, 20.0], frames=a.frames)
            W = m.masks().detach()
            L = W.shape[1]
            off = float((W @ W.T - L * torch.eye(m.users, device=W.device)).abs().max()) / L
            ub = list(sse_union_bound(m, [10.0, 20.0]))
            rows.append((vu, P, seed, s[0], s[1], kappa(m), off, ub[0], ub[1]))
            print(rows[-1], flush=True)
            write_csv(DATA / "rev_factor_seeds.csv",
                      ["vu", "P", "seed", "ser10", "ser20", "kappa_mean",
                       "mask_offdiag_rel", "ub10", "ub20"], rows)


if __name__ == "__main__":
    main()
