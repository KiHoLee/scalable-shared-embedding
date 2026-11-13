"""Earlier design: mechanism of every run of the factorization study.

No training. Reads the 80 runs listed in data/rev_factor_seeds.csv and the six
4,000-iteration runs sse_P4.pt, rev_sse_seed1..5 (U = 4, d = 64). Per run:
  maskcos_max        largest absolute cosine between two masks
  rho_mean/max_s2_20 mean and largest rho_u of (8) over the users, in units of
                     sigma^2 at 20 dB (closed form, exp_rev_analysis)
  kappa_mean/min/max kappa_u of Proposition 2 over the users
  active_dims        participation ratio (sum mu^2)^2 / sum mu^4 per user, mean
                     and per user
  top4_share_min/mean  mask energy on the four largest dimensions, where L = 16
  mode               the three training outcomes:
                       highcos        maskcos_max >= 0.1
                       concentrated   otherwise, a user with kappa_u < 0.01
                       lowcos         otherwise
  n_lowcos, n_concentrated, n_highcos  counts over the runs of the pair
  ser10/20, worst_user10/20, se10/20   Monte Carlo SER at 1e6 frames per SNR over
                     all users, worst user, standard error (users independent)
  ub10/20            bound (7) without Monte Carlo noise: rho_u in closed form and
                     the Rayleigh expectation on a 1e6-point quantile grid
  ub10/20_lib        the same bound through sse_lib.sse_union_bound (seeded)
  z10/20, exceeds10/20  (SER - bound)/se, "yes" if this exceeds 3; a run that sits
                     on the bound within Monte Carlo error is therefore "no"
  closed10/20, exceeds_closed10/20  closed form (9), mean over users, same test
  trained            "yes" if no user has SER >= 0.9 at 10 dB and exceeds10/20 = no
  gauss20            SER at 20 dB with the interference replaced by Gaussian noise
                     of variance rho_u, for every run; the failing (4,8) and (2,16)
                     runs are those with trained = no
  gauss20_ratio      ser20 / gauss20

Fixed seeds: all Monte Carlo streams use 777 + 10 * SNR, as exp_rev_analysis.

Output: data/rev_factor_mech.csv
"""
import csv
import math

import numpy as np
import torch

import exp_rev_analysis as A
from sse_lib import (DATA, DEVICE, SSE, snr_to_sigma2, sse_union_bound, write_csv)

FRAMES = 1_000_000
CHUNK = 50_000
HIGH_COS = 0.1
CONC_KAPPA = 0.01
FAILED_USER = 0.9
Z_EXCEED = 3.0
SNRS = (10.0, 20.0)


def load_run(vu, P, name):
    m = SSE(P=P, vu=vu)
    m.load_state_dict(torch.load(DATA / name, map_location="cpu"))
    return m.to(DEVICE).eval()


@torch.no_grad()
def per_user_ser(m, snr, frames=FRAMES, chunk=CHUNK, seed=777):
    """Frame error rate per user, correlation decoder (5)."""
    torch.manual_seed(seed + int(10 * snr))
    err = torch.zeros(m.users, dtype=torch.float64)
    tot = 0
    for n0 in range(0, frames, chunk):
        n = min(chunk, frames - n0)
        dg = torch.randint(m.vu, (n, m.users, m.P), device=DEVICE)
        r = A.period_obs(m, dg, snr)
        err += (A.decide(m, r) != dg).any(2).double().sum(0).cpu()
        tot += n
    return (err / tot).tolist()


def ser_se(pu, frames=FRAMES):
    """Standard error of the user-averaged SER, users taken as independent."""
    return math.sqrt(sum(p * (1 - p) for p in pu)) / len(pu) / math.sqrt(frames)


@torch.no_grad()
def mask_cos(m):
    Wn = m.masks().detach()
    Wn = Wn / Wn.norm(dim=1, keepdim=True)
    return float((Wn @ Wn.T - torch.eye(m.users, device=Wn.device)).abs().max())


@torch.no_grad()
def kappa_users(m):
    """min_{i!=j} ||(b_i - b_j) . mu_u||^2 / c^2 per user, in double precision."""
    Bn, mu, c = m.unit_codebook().double(), m.masks().double(), float(m.c)
    out = []
    for u in range(m.users):
        cand = Bn * mu[u]
        dd = torch.cdist(cand, cand, compute_mode="donot_use_mm_for_euclid_dist")
        dd = dd + torch.eye(m.vu, device=cand.device, dtype=dd.dtype) * 1e9
        out.append(float(dd.min() ** 2) / c ** 2)
    return out


def active_dims(m):
    w2 = m.masks().detach().pow(2)
    return (w2.sum(1) ** 2 / w2.pow(2).sum(1)).tolist()


def classify(cos_max, kappas):
    if cos_max >= HIGH_COS:
        return "highcos"
    if min(kappas) < CONC_KAPPA:
        return "concentrated"
    return "lowcos"


def bound7_exact(m, kappas, rho, snrs=SNRS, n_grid=1_000_000):
    """Bound (7) as sse_union_bound, with rho_u in closed form and the expectation
    over h^2 ~ Exp(1) on a midpoint quantile grid, so that it has no sampling noise."""
    tau = torch.from_numpy(-np.log1p(-(np.arange(n_grid) + 0.5) / n_grid))
    out = []
    for s in snrs:
        sig2 = float(snr_to_sigma2(torch.tensor(float(s))))
        pe = []
        for u in range(m.users):
            eff = sig2 / tau + rho[u]
            q = 0.5 * torch.erfc(torch.sqrt(kappas[u] / m.P / (4.0 * eff)) / math.sqrt(2.0))
            p_digit = torch.clamp((m.vu - 1) * q, max=1.0)
            pe.append(float((1.0 - (1.0 - p_digit) ** m.P).mean()))
        out.append(sum(pe) / len(pe))
    return out


def closed_form9(m, kappas, snrs=SNRS):
    """Closed form (9), mean over users; very large where a kappa_u is near zero."""
    M = m.P * (m.vu - 1)
    return [sum(8 * m.P * math.log(math.e * M / 2) / (k * m.d * 10 ** (s / 10))
                for k in kappas) / len(kappas) for s in snrs]


def yesno(ser, se, bound):
    z = (ser - bound) / se if se > 0 else float("inf")
    return z, ("yes" if z > Z_EXCEED else "no")


HEADER = ["group", "vu", "P", "seed", "L", "model", "maskcos_max", "rho_mean_s2_20",
          "rho_max_s2_20", "kappa_mean", "kappa_min", "kappa_max", "active_dims_mean",
          "active_dims_users", "top4_share_min", "top4_share_mean", "mode",
          "n_lowcos", "n_concentrated", "n_highcos", "ser10", "ser20",
          "worst_user10", "worst_user20", "se10", "se20", "ub10", "ub20", "ub10_lib",
          "ub20_lib", "z10", "z20", "exceeds10", "exceeds20", "closed10", "closed20",
          "exceeds_closed10", "exceeds_closed20", "trained", "gauss20", "gauss20_ratio"]


def runs():
    yield "six", 16, 4, 0, "sse_P4.pt"
    for s in range(1, 6):
        yield "six", 16, 4, s, f"rev_sse_seed{s}.pt"
    for r in csv.DictReader(open(DATA / "rev_factor_seeds.csv")):
        vu, P, s = int(r["vu"]), int(r["P"]), int(r["seed"])
        yield "factor", vu, P, s, f"rev_factor_{vu}_{P}_s{s}.pt"


@torch.no_grad()
def one_run(group, vu, P, seed, name):
    m = load_run(vu, P, name)
    cos_max = mask_cos(m)
    rho = A.rho_closed_form(m)
    s2_20 = float(snr_to_sigma2(torch.tensor(20.0)))
    ks = kappa_users(m)
    ad = active_dims(m)
    if m.L == 16:
        w2 = m.masks().detach().pow(2)
        top = (w2.topk(4, dim=1).values.sum(1) / w2.sum(1)).tolist()
        top_min, top_mean = min(top), sum(top) / len(top)
    else:
        top_min, top_mean = "", ""
    mode = classify(cos_max, ks)
    pu = [per_user_ser(m, s) for s in SNRS]
    ser = [sum(p) / len(p) for p in pu]
    se = [ser_se(p) for p in pu]
    ub = bound7_exact(m, ks, rho)
    torch.manual_seed(777)
    ub_lib = list(sse_union_bound(m, list(SNRS)))
    cf = closed_form9(m, ks)
    z, ex = zip(*(yesno(ser[i], se[i], ub[i]) for i in range(2)))
    ex_cf = [yesno(ser[i], se[i], cf[i])[1] for i in range(2)]
    trained = "yes" if max(pu[0]) < FAILED_USER and ex[0] == "no" and ex[1] == "no" else "no"
    g20 = A.ser(m, 20.0, frames=FRAMES, chunk=CHUNK, gauss=True, rho=rho)
    return dict(group=group, vu=vu, P=P, seed=seed, L=m.L, model=name, maskcos_max=cos_max,
                rho_mean_s2_20=sum(rho) / len(rho) / s2_20, rho_max_s2_20=max(rho) / s2_20,
                kappa_mean=sum(ks) / len(ks), kappa_min=min(ks), kappa_max=max(ks),
                active_dims_mean=sum(ad) / len(ad),
                active_dims_users=" ".join(f"{x:.2f}" for x in ad),
                top4_share_min=top_min, top4_share_mean=top_mean, mode=mode,
                ser10=ser[0], ser20=ser[1], worst_user10=max(pu[0]), worst_user20=max(pu[1]),
                se10=se[0], se20=se[1], ub10=ub[0], ub20=ub[1], ub10_lib=ub_lib[0],
                ub20_lib=ub_lib[1], z10=z[0], z20=z[1], exceeds10=ex[0], exceeds20=ex[1],
                closed10=cf[0], closed20=cf[1], exceeds_closed10=ex_cf[0],
                exceeds_closed20=ex_cf[1], trained=trained, gauss20=g20,
                gauss20_ratio=ser[1] / g20)


def write_rows(rows):
    for r in rows:
        same = [x["mode"] for x in rows
                if (x["group"], x["vu"], x["P"]) == (r["group"], r["vu"], r["P"])]
        for mode in ("lowcos", "concentrated", "highcos"):
            r[f"n_{mode}"] = same.count(mode)
    write_csv(DATA / "rev_factor_mech.csv", HEADER, [[r[k] for k in HEADER] for r in rows])


def main():
    rows = []
    for run in runs():
        rows.append(one_run(*run))
        r = rows[-1]
        print(r["group"], r["vu"], r["P"], r["seed"], r["mode"], round(r["maskcos_max"], 4),
              round(r["rho_max_s2_20"], 3), round(r["ser10"], 5), round(r["ser20"], 5),
              round(r["ub10"], 5), round(r["ub20"], 5), r["exceeds10"], r["exceeds20"],
              round(r["gauss20"], 5), r["trained"], flush=True)
        write_rows(rows)


if __name__ == "__main__":
    main()
