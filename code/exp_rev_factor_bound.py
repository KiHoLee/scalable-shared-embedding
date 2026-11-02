"""Revision R2.2/R2.6: load and budget checks of the factorizations that fail.

Trains (Vu, P) at other loads and at a longer budget with train_sse, the
procedure of exp_rev_factor_seeds.py, to test whether the failing runs of (4,8)
and (2,16) at U = 4, d = 64 follow from L against U, from the load, or from the
budget:
  (2,16), (4,8)  U = 2, d = 32,   4,000 iterations, seeds 1-10 (full load)
  (2,16)         U = 2, d = 64,   4,000 iterations, seeds 1-10 (half load)
  (2,16), (4,8)  U = 4, d = 128,  4,000 iterations, seeds 1-10 (half load)
  (4,8)          U = 4, d = 64,  20,000 iterations, seeds 1-5
  (2,16)         U = 4, d = 64,  20,000 iterations, seeds 1-2
Noise follows the d of each row (sse_lib.D is set to it). 1e6 frames per SNR.
SER, worst user, kappa_u, largest mask cosine, bound (7) and the training mode as
defined in exp_rev_factor_mech.py; trained = "yes" if no user has SER >= 0.9 at
10 dB and the SER does not exceed (7) by more than three standard errors. A model
file that already exists is loaded, so an interrupted run resumes.

Output: data/rev_factor_bound.csv.
Models: data/rev_factor_bound_{Vu}_{P}_U{U}_d{d}_it{iters}_s{seed}.pt
"""
import torch

import exp_rev_analysis as A
import sse_lib
from exp_rev_factor_mech import (FAILED_USER, SNRS, bound7_exact, classify, kappa_users,
                                 mask_cos, per_user_ser, ser_se, yesno)
from sse_lib import DATA, DEVICE, SSE, set_seed, train_sse, write_csv

#          vu  P  U   d  iters  seeds
CONFIGS = [(2, 16, 2, 32, 4000, 10),
           (4, 8, 2, 32, 4000, 10),
           (2, 16, 2, 64, 4000, 10),
           (2, 16, 4, 128, 4000, 10),
           (4, 8, 4, 128, 4000, 10),
           (4, 8, 4, 64, 20000, 5),
           (2, 16, 4, 64, 20000, 2)]
HEADER = ["vu", "P", "U", "d", "L", "iters", "seed", "ser10", "ser20", "worst_user10",
          "worst_user20", "maskcos_max", "kappa_mean", "kappa_min", "ub10", "ub20", "z10",
          "z20", "exceeds10", "exceeds20", "mode", "trained"]


def main():
    rows = []
    for vu, P, users, d, iters, seeds in CONFIGS:
        sse_lib.D = d                  # snr_to_sigma2 reads the module-level D
        for seed in range(1, seeds + 1):
            path = DATA / f"rev_factor_bound_{vu}_{P}_U{users}_d{d}_it{iters}_s{seed}.pt"
            set_seed(seed)
            m = SSE(P=P, vu=vu, d=d, users=users)
            if path.exists():
                m.load_state_dict(torch.load(path, map_location="cpu"))
            else:
                train_sse(m, iters=iters, seed=seed)
                torch.save({k: v.cpu() for k, v in m.state_dict().items()}, path)
            m.to(DEVICE).eval()
            ks = kappa_users(m)
            cos_max = mask_cos(m)
            pu = [per_user_ser(m, s) for s in SNRS]
            ser = [sum(p) / len(p) for p in pu]
            se = [ser_se(p) for p in pu]
            ub = bound7_exact(m, ks, A.rho_closed_form(m))
            z, ex = zip(*(yesno(ser[i], se[i], ub[i]) for i in range(2)))
            ok = max(pu[0]) < FAILED_USER and ex[0] == "no" and ex[1] == "no"
            rows.append((vu, P, users, d, m.L, iters, seed, ser[0], ser[1], max(pu[0]), max(pu[1]),
                         cos_max, sum(ks) / len(ks), min(ks), ub[0], ub[1], z[0], z[1], ex[0],
                         ex[1], classify(cos_max, ks), "yes" if ok else "no"))
            print(rows[-1], flush=True)
            write_csv(DATA / "rev_factor_bound.csv", HEADER, rows)


if __name__ == "__main__":
    main()
