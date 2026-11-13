"""Earlier design: what closed form (9) can and cannot decide between the
full codebook and the proposal at V = 65,536, U = 4, d = 64, 20 dB.

For each trained run computes per user:
  kappa_u = min_{i!=j} ||(b_i - b_j) . mu_u||^2 / c^2 (per period; chunked over
      all pairs for the full codebook) and delta_u^2 = kappa_u/P;
  closed form (9), 8 P ln(eM/2) / (kappa_u d gamma), with M = P (Vu-1);
  the union bound with the actual pair distances at rho_u = 0, with no Chernoff
      or exponential relaxation,
          E_i E_h min{1, sum_{j!=i} Q(h d_ij / (2 sigma))},
      where d_ij is the distance between whole frames at unit energy. i runs over
      N_CODEWORDS codewords drawn per user with a fixed seed, and the gain
      expectation uses a log-spaced quadrature. pair_union_se is the standard
      error over the drawn codewords;
  median_d2, the median frame-level squared distance to the competitors, averaged
      over the drawn codewords;
  the Monte Carlo SER at 20 dB (exp_rev_robust.run, FRAMES frames, flat fading on
      the complex channel), with minimum-distance decoding for the full codebook
      and correlation decoding (5) for the proposal.

Runs: full codebook after 4,000 iterations (full_V65536.pt) and 2e5 iterations
(rev_full_long_md.pt, the first long run; rev_full_long.pt, an earlier run at the
same budget), sampled softmax at the long budget (rev_sampled_long_md.pt); the
proposal as reported (sse_P4.pt), its five other 4,000-iteration seeds, five
20,000-iteration seeds, and three 2e5-iteration seeds. Checkpoints for the other
two long full-codebook runs are not stored, so kappa_u exists for these only.

Rows: one per (model, user), with user = 0..3 and "mean". One further row,
scheme "full_at_proposal_kappa", gives (9) at P = 1, M = V-1 if all V codewords
reached the kappa_u of sse_P4.pt.

Output: data/rev_full_kappa.csv
"""
import math

import numpy as np
import torch

from exp_rev_robust import run
from exp_rev_users_rho import kappa_over_p
from sse_lib import D, DATA, DEVICE, SSE, FullCodebook, write_csv

V = 65536
SNR_DB = 20.0
N_CODEWORDS = 400
FRAMES = 1_000_000
SEED = 2026
FULL = [("full", 4000, "full_V65536.pt"),
        ("full", 200000, "rev_full_long_md.pt"),
        ("full", 200000, "rev_full_long.pt"),
        ("full_sampled", 200000, "rev_sampled_long_md.pt")]
PROP = [("sse", 4000, "sse_P4.pt")] + \
       [("sse", 4000, f"rev_sse_seed{i}.pt") for i in range(1, 6)] + \
       [("sse", 20000, f"rev_sse_20k_seed{i}.pt") for i in range(1, 6)] + \
       [("sse", 200000, n) for n in ("rev_sse_long_md.pt", "rev_sse_long_md_s2.pt",
                                     "rev_sse_long_md_s3.pt")]

_LT = torch.linspace(math.log(1e-6), math.log(40.0), 240, device=DEVICE,
                     dtype=torch.float64)
_T = _LT.exp()
TAU_SQRT = _T.sqrt().float()
TAU_W = (_T * (-_T).exp() * (_LT[1] - _LT[0])).float()
TAU_W0 = 1.0 - math.exp(-1e-6)       # mass below the grid, where the min is 1


def qf(x):
    return 0.5 * torch.erfc(x / math.sqrt(2.0))


@torch.no_grad()
def pair_union(frames, gen):
    """frames: (V,d) at unit average frame energy."""
    sig = math.sqrt(1.0 / (D * 10.0 ** (SNR_DB / 10.0)))
    vals, med = [], []
    keep = torch.ones(V, dtype=torch.bool, device=frames.device)
    for i in torch.randperm(V, generator=gen)[:N_CODEWORDS].tolist():
        keep[i] = False
        dist = (frames - frames[i]).norm(dim=1)[keep]
        keep[i] = True
        s = qf(TAU_SQRT[None] * dist[:, None] / (2.0 * sig)).sum(0)
        vals.append(float((s.clamp_max(1.0) * TAU_W).sum()) + TAU_W0)
        med.append(float(dist.median() ** 2))
    return float(np.mean(vals)), float(np.std(vals) / math.sqrt(len(vals))), float(np.mean(med))


@torch.no_grad()
def full_kappa_user(cand, c, chunk=2048):
    best, pair = math.inf, None
    for i0 in range(0, V, chunk):
        dd = torch.cdist(cand[i0:i0 + chunk], cand)
        r = torch.arange(dd.shape[0], device=dd.device)
        dd[r, r + i0] = math.inf
        v, j = dd.min(1)
        k = int(v.argmin())
        if float(v[k]) < best:
            best, pair = float(v[k]), (i0 + k, int(j[k]))
    i, j = pair
    return float(((cand[i].double() - cand[j].double()) ** 2).sum()) / c ** 2


def closed9(kappa, P, M):
    return 8.0 * P * math.log(math.e * M / 2.0) / (kappa * D * 10.0 ** (SNR_DB / 10.0))


HDR = ["scheme", "model", "iters", "P", "Vu", "M", "user", "kappa", "delta2",
       "ln_eM_over_2", "closed_form9_20dB", "pair_union_20dB", "pair_union_se",
       "median_d2", "ser20_mc"]


@torch.no_grad()
def main():
    rows = []

    def emit(scheme, name, iters, P, vu, kap, pu, ser):
        M = P * (vu - 1)
        for u in range(len(kap)):
            rows.append((scheme, name, iters, P, vu, M, str(u), kap[u], kap[u] / P,
                         math.log(math.e * M / 2.0), closed9(kap[u], P, M),
                         pu[u][0], pu[u][1], pu[u][2], float(ser[u])))
        rows.append((scheme, name, iters, P, vu, M, "mean", float(np.mean(kap)),
                     float(np.mean(kap)) / P, math.log(math.e * M / 2.0),
                     float(np.mean([closed9(k, P, M) for k in kap])),
                     float(np.mean([p[0] for p in pu])),
                     float(np.sqrt(np.sum([p[1] ** 2 for p in pu])) / len(pu)),
                     float(np.mean([p[2] for p in pu])), float(ser.mean())))
        print(rows[-1], flush=True)
        write_csv(DATA / "rev_full_kappa.csv", HDR, rows)

    # proposal first: cheap, and sse_P4.pt gives the reference kappa
    ix = torch.cartesian_prod(*[torch.arange(16, device=DEVICE)] * 4)
    kap_ref = None
    for scheme, iters, name in PROP:
        m = SSE(P=4, vu=16)
        m.load_state_dict(torch.load(DATA / name, map_location="cpu"))
        kap = [k * 4 for k in kappa_over_p(m)]      # this helper runs on CPU
        m.to(DEVICE).eval()
        kap_ref = kap_ref or kap
        Bn, mu, c = m.unit_codebook(), m.masks(), float(m.c)
        gen = torch.Generator().manual_seed(SEED)
        pu = [pair_union(((Bn * mu[u])[ix] / 2.0).reshape(V, D) / c, gen)
              for u in range(m.users)]
        ser = run("sse", m, 4, SNR_DB, FRAMES, 100_000, 777, rule="corr")
        emit(scheme, name, iters, 4, 16, kap, pu, ser)
    for scheme, iters, name in FULL:
        m = FullCodebook(V)
        m.load_state_dict(torch.load(DATA / name, map_location="cpu"))
        m.to(DEVICE).eval()
        En, mu, c = m.codebook(), m.masks(), float(m.c)
        kap, pu = [], []
        gen = torch.Generator().manual_seed(SEED)
        for u in range(m.users):
            cand = En * mu[u]
            kap.append(full_kappa_user(cand, c))
            pu.append(pair_union(cand / c, gen))
        ser = run("full", m, 4, SNR_DB, FRAMES, 256, 777)
        emit(scheme, name, iters, 1, V, kap, pu, ser)
    nan = float("nan")
    rows.append(("full_at_proposal_kappa", "sse_P4.pt", 0, 1, V, V - 1, "mean",
                 float(np.mean(kap_ref)), float(np.mean(kap_ref)),
                 math.log(math.e * (V - 1) / 2.0),
                 float(np.mean([closed9(k, 1, V - 1) for k in kap_ref])),
                 nan, nan, nan, nan))
    print(rows[-1], flush=True)
    write_csv(DATA / "rev_full_kappa.csv", HDR, rows)


if __name__ == "__main__":
    main()
