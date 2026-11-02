"""Revision R2.6: ten-run sweep of the factorizations behind the selection rule.

Same procedure as exp_rev_factor_seeds.py, for the pairs of Fig. R3 that file
does not cover: (Vu, P) = (3,4), (9,2), (4,4), (16,2), (5,4), (25,2), (10,4),
(100,2), (12,4), (144,2), that is V = 81, 256, 625, 10,000 and 20,736 with
P = 2 and P = 4. U = 4, d = 64, 4,000 iterations, seeds 1-10, 1e6 frames per
SNR. Each row holds SER, kappa_u, the largest off-diagonal entry of the mask
Gram matrix and bound (7) (Monte Carlo form of sse_lib.sse_union_bound). With
rev_factor_seeds.csv this gives the ten-run means of the P = 2 against P = 4
comparison. A model file that already exists is loaded, not retrained, and the
evaluation is reseeded, so an interrupted run resumes without changing a row.

Output: data/rev_factor_sweep.csv. Models: data/rev_factor_{Vu}_{P}_s{seed}.pt
"""
import argparse

import torch

from exp_rev_factor_seeds import kappa
from sse_lib import (DATA, DEVICE, SSE, eval_ser_sse, set_seed, sse_union_bound,
                     train_sse, write_csv)

PAIRS = [(3, 4), (9, 2), (4, 4), (16, 2), (5, 4), (25, 2), (10, 4), (100, 2),
         (12, 4), (144, 2)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--frames", type=int, default=1_000_000)
    a = ap.parse_args()
    rows = []
    for vu, P in PAIRS:
        for seed in range(1, a.seeds + 1):
            path = DATA / f"rev_factor_{vu}_{P}_s{seed}.pt"
            set_seed(seed)
            m = SSE(P=P, vu=vu)
            if path.exists():
                m.load_state_dict(torch.load(path, map_location="cpu"))
            else:
                train_sse(m, seed=seed)
                torch.save({k: v.cpu() for k, v in m.state_dict().items()}, path)
            m.to(DEVICE).eval()
            set_seed(seed)
            s = eval_ser_sse(m, [10.0, 20.0], frames=a.frames, chunk=50_000)
            W = m.masks().detach()
            L = W.shape[1]
            off = float((W @ W.T - L * torch.eye(m.users, device=W.device)).abs().max()) / L
            ub = list(sse_union_bound(m, [10.0, 20.0]))
            rows.append((vu, P, seed, s[0], s[1], kappa(m), off, ub[0], ub[1]))
            print(rows[-1], flush=True)
            write_csv(DATA / "rev_factor_sweep.csv",
                      ["vu", "P", "seed", "ser10", "ser20", "kappa_mean",
                       "mask_offdiag_rel", "ub10", "ub20"], rows)


if __name__ == "__main__":
    main()
