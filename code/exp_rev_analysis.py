"""Revision experiments on the analysis (R1.4, R2.1, R2.2, R2.3, R2.6).

On the trained main model (data/sse_P4.pt, Vu=16, P=4, V=65,536):
  A1  masked-candidate norms ||b_i . mu_u|| (all Vu x U)   -> rev_norms.csv
  A2  decoder comparison: correlation (letter), normalized
      correlation, exact minimum distance                    -> rev_decoders.csv
  A3  Gaussianity of the projected residual interference:
      excess kurtosis, Kolmogorov-Smirnov distance, tail
      ratios, and the SER obtained when the actual
      interference is replaced by isotropic Gaussian noise
      of variance rho_u (the surrogate the bound assumes)     -> rev_decoders.csv (ser_gauss)
  A4  independence of the digit errors across periods at
      fixed gain: measured frame error against the product
      form, and the pairwise error correlation                -> rev_indep.csv
  A5  rho_u in closed form from (B, mu, c) against its
      Monte Carlo estimate                                    -> rev_rho.json
Scaling and factorization studies (models trained here, same budget):
  A6  isotropy of the interference: variance along the
      difference directions (b_i - b_j) . mu_u relative to
      rho_u (mean, max, min ratio per user)                  -> rev_isotropy.json
  S1  fixed Vu=16, P in {1,2,4,8}: measured SER and bound    -> rev_scaleP.csv
  S2  fixed V=65,536, (Vu,P) in {(256,2),(16,4),(4,8),(2,16)}
      plus the full codebook (P=1)                          -> rev_factor.csv
"""
from __future__ import annotations

import json
import math

import numpy as np
import torch

from sse_lib import (DATA, DEVICE, SSE, train_sse, eval_ser_sse,
                     sse_union_bound, rayleigh_gain, snr_to_sigma2,
                     set_seed, write_csv)

SNR_GRID = list(range(0, 21, 2))
FR = 1_000_000


def load_sse(P, vu, name):
    m = SSE(P=P, vu=vu)
    p = DATA / name
    if p.exists():
        m.load_state_dict(torch.load(p, map_location="cpu"))
        return m.to(DEVICE).eval()
    set_seed(1)
    train_sse(m)
    torch.save(m.state_dict(), p)
    return m.to(DEVICE).eval()


@torch.no_grad()
def period_obs(model, digits, snr_db, h=None, gauss=False, rho=None):
    """Equalized per-period observations (N,U,P,L) under the letter's
    real model; with gauss=True the actual multiuser interference is
    replaced by isotropic Gaussian noise of per-direction variance rho_u."""
    N = digits.shape[0]
    Bn, m = model.unit_codebook(), model.masks()
    c, P = model.c, model.P
    x = Bn[digits] / math.sqrt(P) * m[None, :, None, :] / c        # (N,U,P,L)
    y = x.sum(1)                                                   # (N,P,L)
    if h is None:
        h = rayleigh_gain((N, model.users))
    sig = float(snr_to_sigma2(torch.tensor(float(snr_db)))) ** 0.5
    n = torch.randn(N, model.users, P, model.L, device=DEVICE) * sig
    if not gauss:
        r = y[:, None] + n / h[:, :, None, None]
    else:
        g = torch.randn_like(n) * torch.tensor(rho, device=DEVICE).sqrt()[None, :, None, None]
        r = x + g + n / h[:, :, None, None]
    return r


def decide(model, r, rule="corr"):
    cand = model.unit_codebook()[None] * model.masks()[:, None, :]
    corr = torch.einsum("nupl,uvl->nupv", r, cand)
    if rule == "corr":
        return corr.argmax(-1)
    nrm2 = cand.pow(2).sum(-1)
    if rule == "ncorr":
        return (corr / nrm2.sqrt()[None, :, None, :]).argmax(-1)
    a = 1.0 / (float(model.c) * math.sqrt(model.P))
    return (2 * a * corr - a * a * nrm2[None, :, None, :]).argmax(-1)


@torch.no_grad()
def ser(model, snr_db, frames=FR, chunk=100_000, seed=777, rule="corr",
        gauss=False, rho=None, h_fixed=None, per_digit=False):
    torch.manual_seed(seed + int(10 * snr_db))
    err = tot = 0
    dig_err = torch.zeros(model.P, dtype=torch.float64)
    pair = torch.zeros(model.P, model.P, dtype=torch.float64)
    for n0 in range(0, frames, chunk):
        n = min(chunk, frames - n0)
        dg = torch.randint(model.vu, (n, model.users, model.P), device=DEVICE)
        h = None if h_fixed is None else torch.full((n, model.users), h_fixed,
                                                    device=DEVICE)
        r = period_obs(model, dg, snr_db, h=h, gauss=gauss, rho=rho)
        wrong = decide(model, r, rule) != dg                       # (n,U,P)
        err += int(wrong.any(2).sum()); tot += n * model.users
        if per_digit:
            wf = wrong.double().reshape(-1, model.P)
            dig_err += wf.sum(0).cpu()
            pair += (wf.T @ wf).cpu()
    if per_digit:
        return err / tot, dig_err / tot, pair / tot
    return err / tot


@torch.no_grad()
def rho_closed_form(model):
    """Per-direction variance of the projected interference, computed
    exactly from the trained parameters: for user u and unit candidate
    direction n_k, the interference is (1/(c sqrt P)) sum_{v!=u} b_{j_v} .
    mu_v with independent uniform j_v, so its projection has second
    moment sum_v Var_v + (sum_v mean_v)^2, averaged over k."""
    Bn, m = model.unit_codebook().double(), model.masks().double()
    s = 1.0 / (float(model.c) * math.sqrt(model.P))
    out = []
    for u in range(model.users):
        cand = Bn * m[u]
        nk = cand / cand.norm(dim=1, keepdim=True)                 # (Vu,L)
        mean = torch.zeros(model.vu, dtype=torch.float64, device=Bn.device)
        var = torch.zeros_like(mean)
        for v in range(model.users):
            if v == u:
                continue
            proj = s * (nk @ (Bn * m[v]).T)                        # (k, j)
            mean += proj.mean(1)
            var += proj.var(1, unbiased=False)
        out.append(float((var + mean ** 2).mean()))
    return out


@torch.no_grad()
def rho_mc(model, n=400_000):
    Bn, m = model.unit_codebook(), model.masks()
    dg = torch.randint(model.vu, (n, model.users, model.P), device=DEVICE)
    x = Bn[dg] / math.sqrt(model.P) * m[None, :, None, :] / model.c
    y = x.sum(1)
    rho, stats = [], []
    for u in range(model.users):
        iu = y - x[:, u]
        cand = Bn * m[u]
        nk = cand / cand.norm(dim=1, keepdim=True)
        z = torch.einsum("npl,vl->npv", iu, nk)                    # (n,P,Vu)
        rho.append(float(z.pow(2).mean()))
        zc = (z - z.mean(0, keepdim=True)) / z.std(0, keepdim=True)
        zc = zc.reshape(-1).double().cpu().numpy()
        kurt = float(np.mean(zc ** 4) - 3.0)
        zs = np.sort(zc[::7])
        from math import erf
        cdf = 0.5 * (1 + np.vectorize(erf)(zs / math.sqrt(2)))
        emp = np.arange(1, len(zs) + 1) / len(zs)
        ks = float(np.max(np.abs(emp - cdf)))
        t2 = float(np.mean(np.abs(zc) > 2)); t3 = float(np.mean(np.abs(zc) > 3))
        stats.append({"user": u, "kurt_excess": kurt, "ks": ks,
                      "tail2": t2, "tail2_gauss": 0.0455,
                      "tail3": t3, "tail3_gauss": 0.0027})
    return rho, stats


@torch.no_grad()
def isotropy(model):
    """Second-moment matrix of the interference of one period over uniform
    digits, then the variance along every candidate direction (-> rho_u)
    and along every difference direction, which the pairwise error
    probability of Proposition 1 actually sees."""
    Bn, m = model.unit_codebook().double(), model.masks().double()
    s = 1.0 / (float(model.c) * math.sqrt(model.P))
    out = []
    for u in range(model.users):
        M = torch.zeros(model.L, model.L, dtype=torch.float64, device=Bn.device)
        mean = torch.zeros(model.L, dtype=torch.float64, device=Bn.device)
        for v in range(model.users):
            if v == u:
                continue
            X = s * (Bn * m[v])
            mv = X.mean(0)
            M += X.T @ X / model.vu - torch.outer(mv, mv)
            mean += mv
        M += torch.outer(mean, mean)
        cand = Bn * m[u]
        nk = cand / cand.norm(dim=1, keepdim=True)
        rho = float(torch.einsum("kl,lm,km->k", nk, M, nk).mean())
        ratios = []
        for i in range(model.vu):
            for j in range(model.vu):
                if i != j:
                    d = cand[i] - cand[j]
                    d = d / d.norm()
                    ratios.append(float(d @ M @ d) / rho)
        out.append({"user": u, "rho": rho, "ratio_mean": sum(ratios) / len(ratios),
                    "ratio_max": max(ratios), "ratio_min": min(ratios)})
    return out


def main():
    set_seed(1)
    sse = load_sse(4, 16, "sse_P4.pt")

    # A6 isotropy of the interference along difference directions
    iso = isotropy(sse)
    json.dump(iso, open(DATA / "rev_isotropy.json", "w"), indent=1)
    print("[A6]", iso, flush=True)

    # A1 masked norms
    with torch.no_grad():
        cand = sse.unit_codebook()[None] * sse.masks()[:, None, :]
        nrm = cand.norm(dim=-1).cpu()                              # (U,Vu)
    rows = [(u, i, float(nrm[u, i]), float(nrm[u, i] / nrm[u].mean()))
            for u in range(sse.users) for i in range(sse.vu)]
    write_csv(DATA / "rev_norms.csv", ["user", "i", "norm", "norm_rel"], rows)
    rel = nrm / nrm.mean(1, keepdim=True)
    print("[A1] norm rel min/max", float(rel.min()), float(rel.max()),
          "cv per user", (nrm.std(1) / nrm.mean(1)).tolist(), flush=True)

    # A5 rho
    rho_cf = rho_closed_form(sse)
    rho_m, gstats = rho_mc(sse)
    json.dump({"rho_closed_form": rho_cf, "rho_mc": rho_m,
               "gauss_stats": gstats},
              open(DATA / "rev_rho.json", "w"), indent=1)
    print("[A5] rho cf", rho_cf, "mc", rho_m, flush=True)
    print("[A3] stats", gstats, flush=True)

    # A2 decoders, A3 Gaussian surrogate
    rows = []
    for s in SNR_GRID:
        row = [s] + [ser(sse, s, rule=r) for r in ("corr", "ncorr", "md")]
        row.append(ser(sse, s, gauss=True, rho=rho_cf))
        rows.append(row)
        print("[A2/A3]", row, flush=True)
    write_csv(DATA / "rev_decoders.csv",
              ["snr_db", "ser_corr", "ser_ncorr", "ser_md", "ser_gauss"], rows)

    # A4 independence at fixed gain h = 1 (conditioning of the proof)
    rows = []
    for s in SNR_GRID:
        fe, pd, pr = ser(sse, s, h_fixed=1.0, per_digit=True)
        prod = 1.0 - float(torch.prod(1.0 - pd))
        pdn = pd.numpy(); prn = pr.numpy()
        cor = []
        for a in range(sse.P):
            for b in range(a + 1, sse.P):
                cov = prn[a, b] - pdn[a] * pdn[b]
                den = math.sqrt(pdn[a] * (1 - pdn[a]) * pdn[b] * (1 - pdn[b]))
                cor.append(cov / den if den > 0 else 0.0)
        rows.append((s, fe, prod, max(cor), min(cor)))
        print("[A4]", rows[-1], flush=True)
    write_csv(DATA / "rev_indep.csv",
              ["snr_db", "frame_meas", "frame_prod", "corr_max", "corr_min"], rows)

    # S1 fixed Vu = 16, growing P
    rows = []
    for P, name in ((1, "rev_sse_Vu16_P1.pt"), (2, "sse_P2.pt"),
                    (4, "sse_P4.pt"), (8, "rev_sse_Vu16_P8.pt")):
        m = load_sse(P, 16, name)
        meas = eval_ser_sse(m, SNR_GRID, frames=FR)
        ub = sse_union_bound(m, SNR_GRID)
        for s, a, b in zip(SNR_GRID, meas, ub):
            rows.append((P, 16 ** P, s, a, b))
        print("[S1] P", P, meas[5], meas[-1], ub[5], ub[-1], flush=True)
    write_csv(DATA / "rev_scaleP.csv", ["P", "V", "snr_db", "ser", "ub"], rows)

    # S2 factorizations at V = 65,536
    rows = []
    for vu, P, name in ((256, 2, "sse_Vu256_P2.pt"), (16, 4, "sse_P4.pt"),
                        (4, 8, "rev_sse_Vu4_P8.pt"), (2, 16, "rev_sse_Vu2_P16.pt")):
        m = load_sse(P, vu, name)
        meas = eval_ser_sse(m, [10.0, 20.0], frames=FR)
        ub = sse_union_bound(m, [10.0, 20.0])
        rows.append((vu, P, m.n_params(), meas[0], meas[1], ub[0], ub[1]))
        print("[S2]", rows[-1], flush=True)
    write_csv(DATA / "rev_factor.csv",
              ["Vu", "P", "params", "ser10", "ser20", "ub10", "ub20"], rows)


if __name__ == "__main__":
    main()
