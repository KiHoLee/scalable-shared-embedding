"""Revision R2.1: the growth law of Proposition 2 at exactly fixed kappa_u.

Reuses one trained unit codebook, its masks and c in frames of P periods with
L = 16 fixed (d = 16 P), so that kappa_u is the same at every P, and holds the
frame SNR d*gamma constant. Decoding by (5). P doubles from 1 to 64.
Monte Carlo SER against bound (7) at rho_u = 0 and the closed form (9).
Output: data/rev_fixedkappa.csv
"""
import math

import numpy as np
import torch

from sse_lib import DATA, DEVICE, SSE, write_csv

MODELS = ["sse_P4.pt"] + [f"rev_sse_seed{i}.pt" for i in range(1, 6)]
PS = [1, 2, 4, 8, 16, 32, 64]
TAU = -np.log1p(-(np.arange(200000) + 0.5) / 200000)


def qfun(x):
    return 0.5 * np.vectorize(math.erfc)(np.asarray(x) / math.sqrt(2))


@torch.no_grad()
def main():
    rows = []
    for name in MODELS:
        m = SSE(P=4, vu=16)
        m.load_state_dict(torch.load(DATA / name, map_location="cpu"))
        m.to(DEVICE).eval()
        B, mu, c = m.unit_codebook().detach(), m.masks().detach(), float(m.c)
        cand = B[None] * mu[:, None]
        U, vu, L = cand.shape
        ks = []
        for u in range(U):
            dd = torch.cdist(cand[u], cand[u]) + torch.eye(vu, device=DEVICE) * 1e9
            ks.append(float(dd.min() ** 2) / c ** 2)
        for dg in (6400.0, 64000.0):
            s2 = 1 / dg
            for P in PS:
                M = P * (vu - 1)
                err, tot, bs = torch.zeros(U, device=DEVICE), 0, max(2000, 40000 // P)
                target = 2_000_000
                while tot < target:
                    tau = torch.distributions.Exponential(1.0).sample((bs, U)).to(DEVICE)
                    dig = torch.randint(vu, (bs, U, P), device=DEVICE)
                    x = torch.stack([cand[u][dig[:, u]] for u in range(U)], 1).sum(1) / (c * math.sqrt(P))
                    for u in range(U):
                        n = torch.randn(bs, P, L, device=DEVICE) * math.sqrt(s2) / tau[:, u].sqrt()[:, None, None]
                        sc = (x + n) @ cand[u].T
                        err[u] += (sc.argmax(-1) != dig[:, u]).any(1).float().sum()
                    tot += bs
                ser = float(err.sum() / (tot * U))
                c9 = float(np.mean([8 * P * math.log(math.e * M / 2) / (k * dg) for k in ks]))
                ub = []
                for k in ks:
                    q = np.minimum(1, (vu - 1) * qfun(np.sqrt((k / P) / (4 * s2 / TAU))))
                    ub.append(np.mean(1 - (1 - q) ** P))
                rows.append((name, P, 4 * P, dg, float(np.mean(ks)), ser, float(np.mean(ub)), c9,
                             P * math.log(math.e * M / 2)))
                print(rows[-1], flush=True)
                write_csv(DATA / "rev_fixedkappa.csv",
                          ["model", "P", "V_bits", "d_gamma", "kappa_mean", "ser",
                           "ub7_rho0", "closed_form", "law"], rows)


if __name__ == "__main__":
    main()
