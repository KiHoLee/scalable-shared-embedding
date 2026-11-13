"""Earlier design, analysis: why the proposal fails as users are added.

For the trained proposal at U = 2, 4, 8, 16 users, once with the fixed
frame d = 64 (rev_sse_U{U}.pt, sse_P4.pt for U = 4) and once with the
frame growing as d = 16U (rev_sse_fixedload_U{U}.pt), compute per user

  rho_u        the closed-form interference variance (8), exactly as in
               exp_rev_analysis.rho_closed_form, relative to sigma^2 at
               10 dB, and
  kappa_u / P  the minimum squared distance of the user's own masked
               candidates per period, min_{i!=j} ||(b_i-b_j).mu_u||^2
               / (c^2 P),

and the OMA reference distance per user at the same rate: 16 bits on
d/U real dimensions with unit frame energy, minimum squared distance
4 * (1/d) * (amplitude factor of the PAM level spacing).

Output: data/rev_users_rho.csv (one row per configuration, means over
users). The values show whether the masks still separate the users
(small rho_u) and how much room each user's candidates keep (kappa_u/P).
"""
import math

import torch

from exp_rev_analysis import rho_closed_form
from sse_lib import DATA, SSE, write_csv

SNR_REF_DB = 10.0


def oma_dmin2(users: int, d: int, bits: int = 16) -> float:
    """Minimum squared distance between two OMA symbol vectors of one user
    that differ in one bit, at unit frame energy. Each user owns d/U real
    dimensions with energy 1/U, so each dimension carries energy 1/d."""
    n = d // users
    if n >= bits:                       # one sign per bit, repeated n/bits times
        rep = n // bits
        return 4.0 * rep / d
    bps = bits // n                     # M-PAM with M = 2^bps, unit power
    M = 2 ** bps
    a2 = 3.0 / (M * M - 1)              # squared half spacing at unit power
    return 4.0 * a2 / d


@torch.no_grad()
def kappa_over_p(model: SSE) -> list:
    Bn, mu, c = model.unit_codebook(), model.masks(), float(model.c)
    out = []
    for u in range(model.users):
        cand = Bn * mu[u]
        dd = torch.cdist(cand, cand) + torch.eye(model.vu) * 1e9
        out.append(float(dd.min() ** 2) / (c ** 2 * model.P))
    return out


def main():
    rows = []
    for kind in ("fixed_d64", "fixed_load"):
        for users in (2, 4, 8, 16):
            d = 64 if kind == "fixed_d64" else 16 * users
            sig2 = 1.0 / (d * 10.0 ** (SNR_REF_DB / 10.0))   # sigma^2 = 1/(d gamma)
            m = SSE(P=4, vu=16, d=d, users=users)
            if kind == "fixed_d64":
                name = "sse_P4.pt" if users == 4 else f"rev_sse_U{users}.pt"
            else:
                name = f"rev_sse_fixedload_U{users}.pt"
            m.load_state_dict(torch.load(DATA / name, map_location="cpu"))
            m.calibrate_power()
            rho = rho_closed_form(m)
            kap = kappa_over_p(m)
            oma = oma_dmin2(users, d)
            rows.append((kind, users, d, name,
                         sum(rho) / len(rho), sum(rho) / len(rho) / sig2,
                         sum(kap) / len(kap), min(kap), oma,
                         sum(kap) / len(kap) / oma))
            print(f"[{kind}] U={users} d={d} rho={rows[-1][4]:.3g} "
                  f"rho/sigma2@10dB={rows[-1][5]:.3g} kappa/P={rows[-1][6]:.3g} "
                  f"OMA dmin2={oma:.3g} ratio={rows[-1][9]:.3g}", flush=True)
    write_csv(DATA / "rev_users_rho.csv",
              ["frame", "users", "d", "model", "rho_mean", "rho_over_sigma2_10dB",
               "kappa_over_P_mean", "kappa_over_P_min", "oma_dmin2",
               "kappa_over_P_rel_oma"], rows)


if __name__ == "__main__":
    main()
