"""Earlier design: bound (7) with and without the residual interference.

For the trained models (Vu,P) = (16,1), (16,2), (16,4), (4,8), (256,2), plus
(16,8) and the five other (16,4) runs as added references, computes per user:

  kappa_u, delta_u^2 = kappa_u/P, and rho_u by the closed form (8)
      (exp_rev_analysis.rho_closed_form), with rho_u/sigma^2 at every SNR;
  y = delta_u^2/(4 rho_u), and its worst-pair value, with the interference
      variance taken along each candidate difference instead of the average
      candidate direction of (8);
  the extra term (M/2) exp(-theta y/2) of the relaxed bound
      SER_u <= (9)/(1-theta) + (M/2) exp(-theta y/2), at theta = 0.02 and 0.1;
  rank_e1e3, the number of singular directions of the centred candidates
      {b_i . mu_u} holding more than 0.1 percent of the centred energy
      (RANK_ENERGY_FRACTION);
  bound (7) with rho_u and with rho_u = 0, E[1 - (1 - min{1, (Vu-1) Q(sqrt(
      delta^2/(4 (sigma^2/h^2 + rho))))})^P], and closed form (9), at 0 to 40 dB,
      with the gain expectation done by a log-spaced quadrature down to
      h^2 = exp(-60) (an Exp(1) quantile grid is too coarse at high SNR);
  corr_floor, the frame error of the correlation rule (5) with no noise and no
      interference, 1 - (1 - f/Vu)^P, where f counts the candidates a_i with
      a_i^T a_j > a_i^T a_i for some j. The masked norms are unequal, so f can be
      positive; then the SER has a floor that (7) does not capture, since (7) goes
      with the minimum distance, and removing the interference can raise the SER;
  the paired Monte Carlo SER at 10, 20, 30 and 40 dB with the actual interference
      and with it removed, under common digits, gain and noise, on the letter's
      real model with decoding by (5). The 95 percent half width of the ratio
      comes from the discordant frames. FRAMES frames per user.

Rows: one per (model, user, snr_db), with user = 0..U-1, "mean", and "worst".
"mean" uses user means, with rise and ratio_7_9 taken on the mean bounds. "worst"
takes the largest rho_u, rise, ratio and SER, and the smallest kappa_u, y and
rank over the users. rise = (7) with rho_u / (7) at rho_u = 0, minus 1. The
*_max_0_40 columns hold the largest value over 0 to 40 dB. SER columns are nan
at the other SNRs. Beyond the simulated range the floor of (7) eventually exceeds
(9), which goes to zero, so these claims are stated for 0 to 40 dB only.

Output: data/rev_scalingbound_rho.csv
"""
import math

import numpy as np
import torch
from scipy.special import erfc

from exp_rev_analysis import rho_closed_form
from exp_rev_users_rho import kappa_over_p
from sse_lib import D, DATA, DEVICE, SSE, rayleigh_gain, snr_to_sigma2, write_csv

MODELS = [(16, 1, "rev_sse_Vu16_P1.pt"), (16, 2, "sse_P2.pt"),
          (16, 4, "sse_P4.pt"), (4, 8, "rev_sse_Vu4_P8.pt"),
          (256, 2, "sse_Vu256_P2.pt"), (16, 8, "rev_sse_Vu16_P8.pt")] + \
         [(16, 4, f"rev_sse_seed{i}.pt") for i in range(1, 6)]
SNRS = list(range(0, 41, 2))
MC_SNRS = (10, 20, 30, 40)
FRAMES = 2_000_000
CHUNK = 50_000
SEED = 2026
RANK_ENERGY_FRACTION = 1e-3
THETAS = (0.02, 0.1)

_LT = np.linspace(-60.0, math.log(60.0), 400_001)
TAU = np.exp(_LT)
WT = TAU * np.exp(-TAU) * (_LT[1] - _LT[0])


def ub7(delta2, rho, vu, P, gamma):
    s2 = 1.0 / (D * gamma)
    arg = np.sqrt(delta2 / (4.0 * (s2 / TAU + rho)))
    q = np.minimum(1.0 - 1e-16, (vu - 1) * 0.5 * erfc(arg / math.sqrt(2.0)))
    return float(np.sum(WT * -np.expm1(P * np.log1p(-q))))


def closed9(kappa, vu, P, gamma):
    M = P * (vu - 1)
    return 8.0 * P * math.log(math.e * M / 2.0) / (kappa * D * gamma)


@torch.no_grad()
def candidate_stats(m):
    """rank_e1e3 per user, y at the worst pair per user, largest mask correlation."""
    Bn, mu, c = m.unit_codebook().double(), m.masks().double(), float(m.c)
    cand = Bn[None] * mu[:, None]
    s = 1.0 / (c * math.sqrt(m.P))
    ranks, ypair = [], []
    i, j = torch.triu_indices(m.vu, m.vu, 1, device=Bn.device)
    for u in range(m.users):
        cu = cand[u]
        sv = torch.linalg.svdvals(cu - cu.mean(0)) ** 2
        ranks.append(int((sv / sv.sum() > RANK_ENERGY_FRACTION).sum()))
        df = cu[i] - cu[j]
        nd = df.norm(dim=1)
        e = df / nd[:, None]
        var = torch.zeros_like(nd)
        mean = torch.zeros_like(nd)
        for v in range(m.users):
            if v != u:
                pr = s * (e @ cand[v].T)
                var += pr.var(1, unbiased=False)
                mean += pr.mean(1)
        ypair.append(float(((nd * s) ** 2 / (4.0 * (var + mean ** 2))).min()))
    mun = mu / mu.norm(dim=1, keepdim=True)
    G = mun @ mun.T
    corr = float((G - torch.eye(m.users, dtype=G.dtype, device=G.device)).abs().max())
    return ranks, ypair, corr


@torch.no_grad()
def paired_mc(m, u, snr_db):
    """Errors with interference, without it, and the two discordant counts."""
    torch.manual_seed(SEED + 1000 * u + int(10 * snr_db))
    Bn, mu = m.unit_codebook(), m.masks()
    cand = Bn[None] * mu[:, None]
    scale = 1.0 / (float(m.c) * math.sqrt(m.P))
    sig = float(snr_to_sigma2(float(snr_db))) ** 0.5
    others = [v for v in range(m.users) if v != u]
    C = cand[u].T
    e1 = e0 = d10 = d01 = tot = 0
    for n0 in range(0, FRAMES, CHUNK):
        n = min(CHUNK, FRAMES - n0)
        dg = torch.randint(m.vu, (n, m.P), device=DEVICE)
        oth = torch.randint(m.vu, (n, len(others), m.P), device=DEVICE)
        h = rayleigh_gain(n)
        nz = torch.randn(n, m.P, m.L, device=DEVICE) * (sig / h)[:, None, None]
        sg = cand[u][dg] * scale
        it = sum(cand[v][oth[:, t]] for t, v in enumerate(others)) * scale
        a = (((sg + it + nz) @ C).argmax(-1) != dg).any(1)
        b = (((sg + nz) @ C).argmax(-1) != dg).any(1)
        e1 += int(a.sum()); e0 += int(b.sum())
        d10 += int((a & ~b).sum()); d01 += int((b & ~a).sum())
        tot += n
    return e1, e0, d10, d01, tot


@torch.no_grad()
def corr_floor(m):
    """Noiseless, interference-free frame error of the correlation rule, per user."""
    cand = m.unit_codebook()[None] * m.masks()[:, None]
    out = []
    for u in range(m.users):
        wrong = (cand[u] @ cand[u].T).argmax(1) != torch.arange(m.vu, device=cand.device)
        out.append(1.0 - (1.0 - float(wrong.float().mean())) ** m.P)
    return out


def ratio_ci(e1, e0, d10, d01):
    return e1 / max(e0, 1), 1.96 * math.sqrt(d10 + d01) / max(e0, 1)


HDR = ["model", "Vu", "P", "L", "L_over_U", "user", "snr_db", "kappa", "delta2",
       "rho", "rho_over_sigma2", "rho_over_sigma2_20dB", "y", "y_worst_pair",
       "extra_theta0p02", "extra_theta0p1", "rank_e1e3", "mask_maxcorr", "corr_floor",
       "ub7_rho", "ub7_rho0", "closed_form9", "rise", "ratio_7_9",
       "rise_max_0_40", "ratio_7_9_max_0_40", "ser_interf", "ser_no_interf",
       "ser_ratio", "ser_ratio_ci95"]


def main():
    rows = []
    for vu, P, name in MODELS:
        m = SSE(P=P, vu=vu)
        m.load_state_dict(torch.load(DATA / name, map_location="cpu"))
        kap = [k * P for k in kappa_over_p(m)]      # this helper runs on CPU
        m.to(DEVICE).eval()
        U, L, M = m.users, m.L, P * (vu - 1)
        rho = rho_closed_form(m)
        ranks, ypair, corr = candidate_stats(m)
        floor = corr_floor(m)
        y = [k / P / (4.0 * r) for k, r in zip(kap, rho)]
        ex = [[M / 2.0 * math.exp(-t * yy / 2.0) for yy in y] for t in THETAS]

        a = np.array([[ub7(kap[u] / P, rho[u], vu, P, 10 ** (s / 10)) for s in SNRS]
                      for u in range(U)])                    # (U,S)
        b = np.array([[ub7(kap[u] / P, 0.0, vu, P, 10 ** (s / 10)) for s in SNRS]
                      for u in range(U)])
        c9 = np.array([[closed9(kap[u], vu, P, 10 ** (s / 10)) for s in SNRS]
                       for u in range(U)])
        mc = {s: [paired_mc(m, u, s) for u in range(U)] for s in MC_SNRS}
        print(name, "kappa", np.round(kap, 4), "rho", np.array2string(np.array(rho), precision=3),
              "y", np.round(y, 0), "rank", ranks, flush=True)

        def add(user, kk, rr, yy, yp, e0, e1, rk, fl, A, B, C, serfn):
            rise = A / B - 1.0
            ratio = A / C
            for i, s in enumerate(SNRS):
                ser = serfn(s) if s in MC_SNRS else (float("nan"),) * 4
                s2 = 1.0 / (D * 10 ** (s / 10))
                rows.append((name, vu, P, L, L / U, user, s, kk, kk / P, rr, rr / s2,
                             rr * D * 100.0, yy, yp, e0, e1, rk, corr, fl,
                             float(A[i]), float(B[i]), float(C[i]), float(rise[i]),
                             float(ratio[i]), float(rise.max()), float(ratio.max()),
                             *ser))

        tot = lambda s, k: sum(x[k] for x in mc[s])
        for u in range(U):
            def f(s, u=u):
                e1, e0, d10, d01, n = mc[s][u]
                return (e1 / n, e0 / n, *ratio_ci(e1, e0, d10, d01))
            add(str(u), kap[u], rho[u], y[u], ypair[u], ex[0][u], ex[1][u],
                float(ranks[u]), floor[u], a[u], b[u], c9[u], f)

        def fmean(s):
            e1, e0, d10, d01, n = (tot(s, k) for k in range(5))
            return (e1 / n, e0 / n, *ratio_ci(e1, e0, d10, d01))
        add("mean", float(np.mean(kap)), float(np.mean(rho)), float(np.mean(y)),
            float(np.mean(ypair)), float(np.mean(ex[0])), float(np.mean(ex[1])),
            float(np.mean(ranks)), float(np.mean(floor)), a.mean(0), b.mean(0), c9.mean(0),
            fmean)

        def fworst(s):
            per = [(x[0] / x[4], x[1] / x[4], *ratio_ci(*x[:4])) for x in mc[s]]
            iw = int(np.argmax([p[2] for p in per]))
            return (max(p[0] for p in per), max(p[1] for p in per),
                    per[iw][2], per[iw][3])
        # "worst": per-user extremes of the ratios, via a per-user rise and ratio
        rises, ratios = a / b - 1.0, a / c9
        for i, s in enumerate(SNRS):
            ser = fworst(s) if s in MC_SNRS else (float("nan"),) * 4
            s2 = 1.0 / (D * 10 ** (s / 10))
            rows.append((name, vu, P, L, L / U, "worst", s, min(kap), min(kap) / P,
                         max(rho), max(rho) / s2, max(rho) * D * 100.0, min(y),
                         min(ypair), max(ex[0]), max(ex[1]), float(min(ranks)), corr, max(floor),
                         float(a[:, i].max()), float(b[:, i].max()),
                         float(c9[:, i].max()), float(rises[:, i].max()),
                         float(ratios[:, i].max()), float(rises.max()),
                         float(ratios.max()), *ser))
        write_csv(DATA / "rev_scalingbound_rho.csv", HDR, rows)
        w20 = rows[-len(SNRS) + SNRS.index(20)]
        print(f"[{name}] rho/s2@20 max {w20[11]:.4f} y min {w20[12]:.0f} ypair min {w20[13]:.0f} "
              f"rise max {w20[24]:.5f} (7)/(9) max {w20[25]:.4f} floor {max(floor):.5f} mask corr {corr:.4f} "
              f"SER20 {fmean(20)[0]:.5f}/{fmean(20)[1]:.5f} ratio {fmean(20)[2]:.4f}"
              f"+-{fmean(20)[3]:.4f} SER10 ratio {fmean(10)[2]:.4f}", flush=True)


if __name__ == "__main__":
    main()
