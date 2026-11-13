"""Gaussian treatment of the interference in Proposition 1 under mask
overlap, and the exact token sum behind the fixed-gain bound.

Setting of both parts: U = 4 users, d = 64 real dimensions, the BERT
codebook whitened under the source statistics (pwhite, exp_pm.codebook),
independent tokens drawn with the token frequencies p_v, flat fading with
exact gains, and weighted minimum-distance decoding (4) over all
V = 30,522 tokens. With a flat gain known at the receiver the weights of
(4) are equal on all dimensions, so the simulated receiver scores
2 a r.(e_w o m_u) - a^2 ||e_w o m_u||^2 with a = 1/c, which gives the
decisions of exp_pm.Masks.scores at half the cost. c is the root mean frame
energy over 65,536 frames drawn with p_v (seed 0) for the fixed mask sets,
and the stored c for the trained entry.

part overlap -> data/pm/pm_boundov.csv
  Mask sets (U x d, ||m_u||^2 = d) and their mean cross-energy iota (7):
    trained_L16  trained flat entry white_U4_L16_flat_s1 (iota about 0.004),
                 the masks of pm_bound2.csv and pm_bound3.csv
    comb         L = 4, user u weights pattern position u by 2 (iota = 0)
    mix_i0.5, mix_i1, mix_i2
                 L = 4, the comb mixed with full overlap at unit norm: user u
                 has weight a on pattern position u and b > 0 on the three
                 others, a^2 + 3 b^2 = 4, b set for iota = 0.5, 1, 2 (b = 1
                 would be full_equal)
    pair_opp, pair_same
                 L = 2, the masks of users 0 and 1 overlap on position 0 and
                 those of users 2 and 3 on position 1, each with weight
                 sqrt(2); the two users of a pair have opposite (as the
                 trained L = 2 entry) or equal signs (iota = 2)
    full_alt, full_equal
                 L = 1, full overlap with one sign per user, (+,+,-,-) or
                 (+,+,+,+) (iota = 3)
    full_rand    L = 64, full overlap with an independent random sign per user
                 and dimension (CPU generator seed 0, iota = 3)
  (a) rows test = pep_nn, pep_pw: the pairwise term (6) against Monte Carlo of
      the same event xi^T (z + n) < -||xi||^2 / (2c) at the fixed gain
      h = 1 and SNR 0, 5, 10, 15, 20 dB. For every user u, 128 tokens v are
      drawn with p_v (CPU generator seed 100 + u); pep_nn pairs every v with
      its 16 nearest tokens w in the masked distance ||xi||, pep_pw with 4
      tokens w drawn with p_w. The interference z is sampled from 65,536
      frames of the other users' tokens (CUDA generator seed 200 + u), shared
      by all pairs of the user, and the noise is integrated in closed form:
      pep_mc = mean_z Q((||xi||^2/(2c) + xi^T z) / (sigma ||xi||)).
      pep_count draws the noise as well and counts the event. pep_gauss_emp
      is the Gaussian tail with the empirical mean and variance of
      xi^T z, so ratio_tail = pep_gauss_emp / pep_mc tests the Gaussian
      shape alone and ratio = pep_eq6 / pep_mc tests (6) with its analytic
      moments nu_u and Sigma_u (ratios of means over the pairs; ratio_med,
      ratio_q10 and ratio_q90 are per-pair quantiles of pep_eq6 / pep_mc over
      the pairs whose Monte Carlo standard error is below 10 percent).
      var_ratio is the median ratio of the empirical to the analytic variance
      of xi^T z, intf_share the mean share of the interference in the
      variance of (6), skew and exkurt the median skewness and excess kurtosis
      of xi^T z.
  (b) rows test = ser: the clipped SER and distortion bounds (5) on the 600
      tokens drawn with p_v of exp_pm.bound_eval (seed 5), at the fixed unit
      gain and averaged over Rayleigh fading (400 gain nodes), against
      simulations of 200,000 frames per point (seeds 700 + SNR at the fixed
      gain and 900 + SNR under Rayleigh fading, as exp_pm.py parts bound3 and
      bound2). Column prop2 inserts the closed form (8) of Proposition 2 with
      the masks' iota_u into the same clipped SER sum on the same tokens.
part exact -> data/pm/pm_bound_exact.csv, data/pm/pm_bound_exact.json
  The fixed-gain (h = 1) SER and distortion bounds of pm_bound3.csv
  (trained_L16) and of the comb, SNR 0 to 14 dB in steps of 2. The sum over v
  is taken on the 600 tokens of pm_bound3.csv (seed 5), once with
  exp_pm.bound_eval (v_set 600_seed5_bound_eval) and once with the batched
  float64 evaluation of this script (600_seed5); on 5,000 tokens drawn with
  p_v (seed 6); and exactly over every token with p_v > 0, weighted by p_v
  (all_support). v_set 600_redraw gives the mean and the 5 and 95 percent
  points of the 600-token estimate over 2,000 redraws (seeds 1000-2999), and
  frac_below_mc the share of redraws below the simulated SER. Simulation at
  the fixed gain: 4,000,000 frames from 10 dB on, 1,000,000 at 6 and 8 dB,
  200,000 below (seed 7000 + SNR). The JSON holds the share of the exact
  bound carried by tokens with p_v < 1/600 and the rank of the seed-5 draw
  among the redraws.

Run on one GPU (code/ directory); on a shared GPU a step that runs out of
memory waits and repeats with the same seeds:
  python exp_pm_boundov.py --part overlap
  python exp_pm_boundov.py --part exact
  --smoke runs a few masks, tokens and frames and writes *_smoke files.
"""
from __future__ import annotations

import argparse
import json
import math
import time

import torch

import rev_channel as rc
from sse_lib import DEVICE, D, write_csv
import exp_pm as pm

DATA = pm.DATA                      # data/pm/
U = 4
SQ2 = math.sqrt(2.0)


def qfun(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * torch.erfc(x / SQ2)


# ----------------------------------------------------------------------
# shared GPU: hold a memory pool and repeat a step that runs out of memory
# (every step is seeded, so a repeated step gives the same result)
# ----------------------------------------------------------------------
def reserve(mb: int = 768, wait: int = 20, tries: int = 540) -> None:
    if DEVICE.type != "cuda":
        return
    for k in range(tries):
        try:
            x = torch.empty(mb * 2 ** 20, dtype=torch.uint8, device=DEVICE)
            del x                                   # stays in this process's cache
            return
        except torch.OutOfMemoryError:
            print(f"[wait] GPU memory full, try {k + 1}", flush=True)
            time.sleep(wait)
    raise RuntimeError("no GPU memory")


def retry(fn, *args, tries: int = 360, wait: int = 20, **kw):
    for k in range(tries):
        try:
            return fn(*args, **kw)
        except torch.OutOfMemoryError:
            print(f"[wait] out of memory in {fn.__name__}, try {k + 1}", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"{fn.__name__}: no GPU memory")


def sigma2(snr_db: float) -> float:
    """Noise variance per real dimension, sigma^2 = 1/(d SNR)."""
    return 1.0 / (D * 10 ** (snr_db / 10))


def moments(E: torch.Tensor, pd: torch.Tensor):
    """Mean e_bar and covariance C of the codebook under p_v."""
    mu = (pd[:, None] * E).sum(0)
    C = (E * pd[:, None]).T @ E - torch.outer(mu, mu)
    return mu, C


# ----------------------------------------------------------------------
# mask sets
# ----------------------------------------------------------------------
def periodic(pattern: torch.Tensor) -> torch.Tensor:
    """(U, L) pattern -> (U, d) mask, the pattern repeated d/L times."""
    return pattern.repeat(1, D // pattern.shape[1])


def mix_pattern(iota: float) -> torch.Tensor:
    """L = U: user u has weight a on pattern position u and b on the U-1
    others, a^2 + (U-1) b^2 = U; b^2 in [0, 1] found by bisection for the mean
    cross-energy iota (monotone in b^2 on this interval)."""
    def build(b2):
        P = torch.full((U, U), math.sqrt(b2))
        P.fill_diagonal_(math.sqrt(U - (U - 1) * b2))
        return P
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if sum(pm.cross_energy(periodic(build(mid)))) / U < iota:
            lo = mid
        else:
            hi = mid
    return build(0.5 * (lo + hi))


@torch.no_grad()
def frame_c(E, pd, M, n=65536, seed=0) -> float:
    """c as exp_pm.Masks.calibrate: root mean frame energy over n frames."""
    torch.manual_seed(seed)
    idx = torch.multinomial(pd, n * U, replacement=True).view(n, U)
    return float((E[idx] * M[None]).sum(1).pow(2).sum(1).mean().sqrt())


def c_exact(E, pd, M) -> float:
    """c^2 = sum_i (C_ii sum_u m_ui^2 + e_bar_i^2 (sum_u m_ui)^2)."""
    mu, C = moments(E.double(), pd.double())
    Md = M.double()
    return float((torch.diagonal(C) * Md.pow(2).sum(0) + mu.pow(2) * Md.sum(0).pow(2)).sum().sqrt())


def load_trained():
    sd = torch.load(pm.MODELS / "white_U4_L16_flat_s1.pt", map_location="cpu")
    m = pm.Masks(sd["E"], 16, U)
    m.load_state_dict(sd)
    return m.to(DEVICE).eval()


def mask_sets(E, pd, names=None):
    s2 = math.sqrt(2.0)
    comb = torch.zeros(U, U)
    comb.fill_diagonal_(2.0)
    g = torch.Generator().manual_seed(0)
    spec = {
        "trained_L16": None,
        "comb": (periodic(comb), U),
        "mix_i0.5": (periodic(mix_pattern(0.5)), U),
        "mix_i1": (periodic(mix_pattern(1.0)), U),
        "mix_i2": (periodic(mix_pattern(2.0)), U),
        "pair_opp": (periodic(torch.tensor([[s2, 0.], [-s2, 0.], [0., s2], [0., -s2]])), 2),
        "pair_same": (periodic(torch.tensor([[s2, 0.], [s2, 0.], [0., s2], [0., s2]])), 2),
        "full_alt": (periodic(torch.tensor([[1.], [1.], [-1.], [-1.]])), 1),
        "full_rand": ((torch.randint(0, 2, (U, D), generator=g) * 2 - 1).float(), D),
        "full_equal": (torch.ones(U, D), 1),
    }
    out = {}
    for name, val in spec.items():
        if names is not None and name not in names:
            continue
        if val is None:
            m = load_trained()
            M, L, c = m.masks().detach(), 16, float(m.c)
            dE = float((m.E - E).abs().max())
            print(f"[masks] trained_L16 stored codebook vs pwhite: max diff {dE:.2e}", flush=True)
        else:
            M, L = val[0].to(DEVICE), val[1]
            c = frame_c(E, pd, M)
        assert torch.allclose(M.pow(2).sum(1), torch.full((U,), float(D), device=M.device), rtol=1e-3)
        iota = pm.cross_energy(M.cpu())
        out[name] = dict(M=M, L=L, c=c, iota=iota)
        print(f"[masks] {name} L {L} c {c:.4f} (exact {c_exact(E, pd, M):.4f}) iota "
              f"{[round(x, 4) for x in iota]}", flush=True)
    return out


class _Fixed:
    """Mask set in the form exp_pm.bound_eval reads (masks(), c)."""

    def __init__(self, M, c):
        self._M, self.c = M, torch.tensor(c)

    def masks(self):
        return self._M


# ----------------------------------------------------------------------
# simulation
# ----------------------------------------------------------------------
@torch.no_grad()
def simulate(E, pd, Wn, M, c, snr, gain, frames, seed, chunk=1024):
    """SER and distortion of rule (4) for all users, flat fading with exact
    gains (gain = 'rayleigh', rev_channel.channel) or the fixed gain h = 1."""
    torch.manual_seed(seed)
    a = 1.0 / c
    cand = E[None] * M[:, None, :] * a                       # (U,V,d)
    cn2 = cand.pow(2).sum(-1)                                # (U,V)
    sig = math.sqrt(sigma2(snr))
    err, sem = 0, 0.0
    for k in range(0, frames, chunk):
        n = min(chunk, frames - k)
        idx = torch.multinomial(pd, n * U, replacement=True).view(n, U)
        x = (E[idx] * M[None]).sum(1) * a
        if gain == "fixed":
            r = x[:, None, :] + sig * torch.randn(n, U, D, device=DEVICE)
        else:
            r, _ = rc.channel(x, snr, U, **pm.COND["flat"])
        dec = torch.stack([(2 * r[:, u] @ cand[u].T - cn2[u][None]).argmax(1)
                           for u in range(U)], 1)
        err += int((dec != idx).sum())
        sem += float((1 - (Wn[idx] * Wn[dec]).sum(-1)).sum())
    return err / (frames * U), err, sem / (frames * U)


# ----------------------------------------------------------------------
# part overlap (a): pairwise test of (6)
# ----------------------------------------------------------------------
@torch.no_grad()
def pep_test(E, pd, M, c, snrs, n_v, n_nb, n_pw, K, kchunk=2048):
    """Per-pair arrays (concatenated over users) of the pairwise term (6), its
    Gaussian form with empirical moments, and Monte Carlo of the event."""
    mu, C = moments(E, pd)
    S = len(snrs)
    keys = ["eq6", "emp", "mc", "se", "count", "share"]
    res = {k: [] for k in keys}
    res.update(label=[], var_ratio=[], skew=[], exkurt=[])
    pdc = pd.cpu()
    for u in range(U):
        others = [k for k in range(U) if k != u]
        nu = sum(M[k] * mu for k in others) / c
        Sig = sum(M[k][:, None] * C * M[k][None, :] for k in others) / c ** 2
        g = torch.Generator().manual_seed(100 + u)
        vs = torch.multinomial(pdc, n_v, replacement=True, generator=g).tolist()
        Em = E * M[u]
        a = (Em * Em).sum(1)
        vi, wi, lab = [], [], []
        for v in vs:
            d2 = a[v] + a - 2 * Em @ Em[v]
            d2[v] = float("inf")
            d2[d2 <= 1e-9] = float("inf")
            nn = torch.topk(d2, n_nb, largest=False).indices.tolist()
            fin = torch.isfinite(d2).cpu()
            cand = torch.multinomial(pdc, 16 * n_pw, replacement=True, generator=g).tolist()
            pw = [w for w in cand if bool(fin[w])][:n_pw]
            vi += [v] * (len(nn) + len(pw))
            wi += nn + pw
            lab += ["nn"] * len(nn) + ["pw"] * len(pw)
        vi = torch.tensor(vi, device=DEVICE)
        wi = torch.tensor(wi, device=DEVICE)
        Dl = (E[vi] - E[wi]) * M[u]                                   # (P,d)
        n2 = (Dl * Dl).sum(1).double()
        thr = n2 / (2 * c)
        m_a = (Dl @ nu).double()
        v_a = ((Dl @ Sig) * Dl).sum(1).double().clamp_min(0)
        Dl64 = Dl.double()
        P = Dl.shape[0]
        sd = [math.sqrt(sigma2(s)) * n2.sqrt() for s in snrs]
        s1 = torch.zeros(P, dtype=torch.float64, device=DEVICE)
        s2_, s3, s4 = s1.clone(), s1.clone(), s1.clone()
        sQ = torch.zeros(S, P, dtype=torch.float64, device=DEVICE)
        sQ2, cnt = sQ.clone(), sQ.clone()
        gk = torch.Generator(device=DEVICE).manual_seed(200 + u)
        for k0 in range(0, K, kchunk):
            kk = min(kchunk, K - k0)
            s = torch.multinomial(pd, kk * (U - 1), replacement=True, generator=gk).view(kk, U - 1)
            z = sum(E[s[:, j]] * M[others[j]] for j in range(U - 1)) / c        # (kk,d)
            t = Dl64 @ z.double().T                                              # (P,kk)
            t2 = t * t
            s1 += t.sum(1); s2_ += t2.sum(1); s3 += (t2 * t).sum(1); s4 += (t2 * t2).sum(1)
            for j in range(S):
                q = qfun((thr[:, None] + t) / sd[j][:, None])
                sQ[j] += q.sum(1)
                sQ2[j] += (q * q).sum(1)
                nz = torch.randn(t.shape, generator=gk, device=DEVICE, dtype=torch.float64)
                cnt[j] += ((t + nz * sd[j][:, None]) < -thr[:, None]).sum(1)
        mt = s1 / K
        vt = (s2_ / K - mt * mt).clamp_min(0)
        m3 = s3 / K - 3 * mt * s2_ / K + 2 * mt ** 3
        m4 = s4 / K - 4 * mt * s3 / K + 6 * mt * mt * s2_ / K - 3 * mt ** 4
        okv = vt > 1e-14
        nanv = torch.full_like(vt, float("nan"))
        res["skew"].append(torch.where(okv, m3 / vt.clamp_min(1e-300) ** 1.5, nanv))
        res["exkurt"].append(torch.where(okv, m4 / vt.clamp_min(1e-300) ** 2 - 3, nanv))
        res["var_ratio"].append(torch.where(v_a > 1e-14, vt / v_a.clamp_min(1e-300), nanv))
        eq6, emp, mc, se, ct, sh = [], [], [], [], [], []
        for j, snr in enumerate(snrs):
            s2v = sigma2(snr)
            eq6.append(qfun((thr + m_a) / torch.sqrt(s2v * n2 + v_a)))
            emp.append(qfun((thr + mt) / torch.sqrt(s2v * n2 + vt)))
            p = sQ[j] / K
            mc.append(p)
            se.append(((sQ2[j] / K - p * p).clamp_min(0) / K).sqrt())
            ct.append(cnt[j] / K)
            sh.append(v_a / (s2v * n2 + v_a))
        for k, arr in zip(keys, (eq6, emp, mc, se, ct, sh)):
            res[k].append(torch.stack(arr))                                     # (S,P)
        res["label"] += lab
    for k in keys:
        res[k] = torch.cat(res[k], 1).cpu()
    for k in ("var_ratio", "skew", "exkurt"):
        res[k] = torch.cat(res[k]).cpu()
    return res


def pep_rows(name, info, snrs, res, n_v, n_nb, n_pw, K):
    rows = []
    lab = res["label"]
    for st in ("nn", "pw"):
        sel = torch.tensor([x == st for x in lab])
        if not bool(sel.any()):
            continue
        for j, snr in enumerate(snrs):
            eq6, emp, mc = res["eq6"][j, sel], res["emp"][j, sel], res["mc"][j, sel]
            se, ct, sh = res["se"][j, sel], res["count"][j, sel], res["share"][j, sel]
            n = int(sel.sum())
            m_mc = float(mc.mean())
            rsv = (mc > 0) & (se < 0.1 * mc)
            r = (eq6[rsv] / mc[rsv]) if bool(rsv.any()) else torch.tensor([float("nan")])
            qs = torch.quantile(r.double(), torch.tensor([0.1, 0.5, 0.9], dtype=torch.float64)).tolist() \
                if r.numel() > 1 else [float(r[0])] * 3
            rows.append([f"pep_{st}", name, info["L"], sum(info["iota"]) / U, info["c"], "fixed", snr,
                         n, float(eq6.mean()), float(emp.mean()), m_mc,
                         float(se.pow(2).sum().sqrt() / n), float(ct.mean()),
                         float(eq6.mean()) / m_mc if m_mc > 0 else float("nan"),
                         float(emp.mean()) / m_mc if m_mc > 0 else float("nan"),
                         qs[1], qs[0], qs[2], int(rsv.sum()),
                         float(torch.nanmedian(res["var_ratio"][sel])),
                         float(sh.mean()),
                         float(torch.nanmedian(res["skew"][sel])),
                         float(torch.nanmedian(res["exkurt"][sel])),
                         "", "", "", "", "", ""])
            print(f"[pep] {name} {st} {snr:>2} dB eq6 {rows[-1][8]:.3e} emp {rows[-1][9]:.3e} "
                  f"mc {m_mc:.3e} (se {rows[-1][11]:.1e}) ratio {rows[-1][13]:.3f} tail "
                  f"{rows[-1][14]:.3f} med {qs[1]:.3f} [{qs[0]:.3f},{qs[2]:.3f}] n_res {int(rsv.sum())}/{n} "
                  f"share {rows[-1][20]:.3f}", flush=True)
    return rows


# ----------------------------------------------------------------------
# part overlap (b): Proposition 2 closed form in the clipped sum
# ----------------------------------------------------------------------
@torch.no_grad()
def prop2_bound(E, pd, iotas, snrs, n_v, n_h=400, seed=5, fixed_gain=False):
    """sum_v p_v E_h[min(1, sum_w Q(||e_v - e_w|| / (2 sqrt(U s2/h^2 + iota_u/d))))]
    on the tokens of exp_pm.bound_eval, mean over the users."""
    uq = (torch.arange(n_h, device=DEVICE, dtype=torch.float32) + 0.5) / n_h
    h2 = torch.ones(1, device=DEVICE) if fixed_gain else -torch.log1p(-uq)
    g = torch.Generator(device="cpu").manual_seed(seed)
    vs = torch.multinomial(pd.cpu(), n_v, replacement=True, generator=g).tolist()
    keys = sorted({round(x, 6) for x in iotas})
    acc = {k: torch.zeros(len(snrs), device=DEVICE) for k in keys}
    for v in vs:
        dist = (E[v][None] - E).norm(dim=1)
        dist = dist[dist > 0]
        for k in keys:
            for j, snr in enumerate(snrs):
                den = 2 * torch.sqrt(U * sigma2(snr) / h2 + k / D)               # (H,)
                q = qfun(dist[None] / den[:, None])
                acc[k][j] += q.sum(1).clamp(max=1.0).mean() / n_v
    return torch.stack([acc[round(x, 6)] for x in iotas]).mean(0)


def part_overlap(a):
    t0 = time.time()
    src = pm.Source("bert")
    pdc = src.dist()
    reserve()
    E = pm.codebook(src.W, D, "pwhite", pd=pdc).to(DEVICE)
    pd = pdc.to(DEVICE)
    Wn = src.Wn.to(DEVICE)
    if a.smoke:
        names = ["trained_L16", "comb", "full_alt", "full_equal"]
        snrs, n_v, n_nb, n_pw, K, nv_b, frames = [0, 10], 4, 4, 2, 4096, 8, 4096
    else:
        names = None
        snrs, n_v, n_nb, n_pw, K, nv_b, frames = [0, 5, 10, 15, 20], 128, 16, 4, 65536, 600, 200_000
    sets = retry(mask_sets, E, pd, names)
    rows = []
    for name, info in sets.items():
        res = retry(pep_test, E, pd, info["M"], info["c"], snrs, n_v, n_nb, n_pw, K)
        rows += pep_rows(name, info, snrs, res, n_v, n_nb, n_pw, K)
        print(f"[time] {name} pep {(time.time() - t0) / 60:.1f} min", flush=True)
        m = _Fixed(info["M"], info["c"])
        for gain in ("fixed", "rayleigh"):
            fg = gain == "fixed"
            ub_ser, ub_sem = retry(pm.bound_eval, m, E, pd, Wn, U, snrs, nv_b, fixed_gain=fg)
            p2 = retry(prop2_bound, E, pd, info["iota"], snrs, nv_b, fixed_gain=fg)
            print(f"[time] {name} {gain} bounds {(time.time() - t0) / 60:.1f} min", flush=True)
            for j, snr in enumerate(snrs):
                seed = (700 if fg else 900) + snr
                ser, nerr, sem = retry(simulate, E, pd, Wn, info["M"], info["c"], snr, gain, frames, seed)
                ub = float(ub_ser[:, j].mean())
                se = math.sqrt(max(ser * (1 - ser), 0.0) / (frames * U))
                rows.append(["ser", name, info["L"], sum(info["iota"]) / U, info["c"], gain, snr,
                             frames, ub, "", ser, se, "", ub / ser if ser > 0 else float("nan"), "",
                             "", "", "", "", "", "", "", "",
                             nerr, float(ub_sem[:, j].mean()), sem,
                             float(ub_sem[:, j].mean()) / sem if sem > 0 else float("nan"),
                             float(p2[j]), float(p2[j]) / ser if ser > 0 else float("nan")])
                print(f"[ser] {name} {gain} {snr:>2} dB ub {ub:.4g} sim {ser:.4g} ({nerr} err) "
                      f"ratio {rows[-1][13]:.3f} | ub_sem {rows[-1][24]:.4g} sem {sem:.4g} | "
                      f"prop2 {float(p2[j]):.4g}", flush=True)
        print(f"[time] {name} done {(time.time() - t0) / 60:.1f} min", flush=True)
    hdr = ["test", "mask", "pattern_len", "iota", "c", "gain", "snr_db", "n",
           "approx", "pep_gauss_emp", "sim", "sim_se", "pep_count", "ratio", "ratio_tail",
           "ratio_med", "ratio_q10", "ratio_q90", "n_resolved", "var_ratio", "intf_share",
           "skew", "exkurt", "errors", "ub_sem", "mc_sem", "ratio_sem", "prop2", "ratio_prop2"]
    write_csv(DATA / f"pm_boundov{'_smoke' if a.smoke else ''}.csv", hdr, rows)


# ----------------------------------------------------------------------
# part exact
# ----------------------------------------------------------------------
@torch.no_grad()
def token_bounds(E, pd, Wn, M, c, toks, snrs, batch=128):
    """Clipped fixed-gain (h = 1) sums of (5)-(6) for every token v in toks,
    as exp_pm.bound_eval but batched in float64: ser, sem of shape
    (U, len(snrs), len(toks))."""
    Ed, Md = E.double(), M.double()
    mu, C = moments(Ed, pd.double())
    ac = 1.0 / c
    s2s = [sigma2(s) for s in snrs]
    pre = []
    for u in range(U):
        others = [k for k in range(U) if k != u]
        Em = Ed * Md[u]
        nu = ac * sum(Md[k] * mu for k in others)
        A = ac * ac * sum((Md[u] * Md[k])[:, None] * C * (Md[u] * Md[k])[None, :] for k in others)
        EA = Ed @ A
        pre.append((Em, (Em * Em).sum(1), Ed @ (Md[u] * nu), EA, (EA * Ed).sum(1)))
    toks = torch.as_tensor(toks, device=DEVICE)
    out_ser = torch.zeros(U, len(snrs), len(toks), dtype=torch.float64, device=DEVICE)
    out_sem = torch.zeros_like(out_ser)
    n_dup = 0
    for i0 in range(0, len(toks), batch):
        vb = toks[i0:i0 + batch]
        B = len(vb)
        ar = torch.arange(B, device=DEVICE)
        semw = (1 - Wn[vb] @ Wn.T).clamp_min(0).double()                       # (B,V)
        for u in range(U):
            Em, a, b, EA, q = pre[u]
            n2 = a[vb][:, None] + a[None] - 2 * Em[vb] @ Em.T                    # ||xi||^2
            ok = n2 > 1e-10
            ok[ar, vb] = False
            n_dup += int((~ok).sum()) - B
            n2 = n2.clamp_min(1e-10)
            mean = b[vb][:, None] - b[None]                                      # xi^T nu_u
            var = (q[vb][:, None] + q[None] - 2 * EA[vb] @ Ed.T).clamp_min(0)    # xi^T Sigma_u xi
            num = ac * n2 / 2 + mean
            sw = torch.where(ok, semw, torch.zeros_like(semw))
            cap = sw.max(1).values
            for j, s2 in enumerate(s2s):
                qq = torch.where(ok, qfun(num / torch.sqrt(n2 * s2 + var)), torch.zeros_like(n2))
                out_ser[u, j, i0:i0 + B] = qq.sum(1).clamp(max=1.0)
                out_sem[u, j, i0:i0 + B] = torch.minimum((qq * sw).sum(1), cap)
    if n_dup:
        print(f"[exact] {n_dup} (v, w != v) pairs with ||Delta||^2 <= 1e-10 left out, as bound_eval",
              flush=True)
    return out_ser, out_sem


def part_exact(a):
    t0 = time.time()
    src = pm.Source("bert")
    pdc = src.dist()
    reserve()
    E = pm.codebook(src.W, D, "pwhite", pd=pdc).to(DEVICE)
    pd = pdc.to(DEVICE)
    Wn = src.Wn.to(DEVICE)
    if a.smoke:
        snrs, n_big, n_redraw = [10, 12], 300, 20
        frames = lambda s: 8192
    else:
        snrs, n_big, n_redraw = [0, 2, 4, 6, 8, 10, 12, 14], 5000, 2000
        frames = lambda s: 4_000_000 if s >= 10 else (1_000_000 if s >= 6 else 200_000)
    sets = retry(mask_sets, E, pd, ["trained_L16", "comb"])
    support = torch.nonzero(pdc > 0).flatten()
    if a.smoke:
        support = support[:1500]
    pos = torch.full((len(pdc),), -1, dtype=torch.long)
    pos[support] = torch.arange(len(support))
    g5 = torch.Generator(device="cpu").manual_seed(5)
    v600 = torch.multinomial(pdc, 600, replacement=True, generator=g5)
    g6 = torch.Generator(device="cpu").manual_seed(6)
    v5k = torch.multinomial(pdc, n_big, replacement=True, generator=g6)
    print(f"[exact] support tokens {len(support)}, p_v mass {float(pdc[support].sum()):.6f}", flush=True)
    rows, info_js = [], {}
    for name, info in sets.items():
        M, c = info["M"], info["c"]
        m = _Fixed(M, c)
        be_ser, be_sem = retry(pm.bound_eval, m, E, pd, Wn, U, snrs, 600, fixed_gain=True)
        print(f"[time] {name} bound_eval {(time.time() - t0) / 60:.1f} min", flush=True)
        tb = {}
        for key, toks in (("600_seed5", v600), (f"{n_big}_seed6", v5k), ("all_support", support)):
            s_, d_ = retry(token_bounds, E, pd, Wn, M, c, toks.tolist(), snrs)
            tb[key] = (s_.mean(0).cpu(), d_.mean(0).cpu())                     # user mean (S, n)
            print(f"[time] {name} {key} {(time.time() - t0) / 60:.1f} min", flush=True)
        w_all = pdc[support].double()
        w_all = w_all / w_all.sum()
        ser_all, sem_all = tb["all_support"]
        exact_ser = (ser_all * w_all[None]).sum(1)
        exact_sem = (sem_all * w_all[None]).sum(1)
        # redraws of the 600-token estimate from the per-token values (support only)
        red = []
        for b in range(n_redraw):
            gb = torch.Generator(device="cpu").manual_seed(1000 + b)
            vv = torch.multinomial(pdc, 600, replacement=True, generator=gb)
            ii = pos[vv]
            ii = ii[ii >= 0]
            red.append(ser_all[:, ii].mean(1))
        red = torch.stack(red)                                                   # (R,S)
        js = dict(c=c, iota=info["iota"], n_support=len(support), snr={})
        for j, snr in enumerate(snrs):
            fr = frames(snr)
            ser, nerr, sem = retry(simulate, E, pd, Wn, M, c, snr, "fixed", fr, 7000 + snr)
            se = math.sqrt(max(ser * (1 - ser), 0.0) / (fr * U))
            common = [ser, se, nerr, sem, fr]

            def row(vset, n, ub, ubs, q05="", q95="", fb=""):
                rows.append([name, snr, vset, n, ub, ubs, q05, q95, fb] + common
                            + [ub / ser if ser > 0 else float("nan")])

            row("600_seed5_bound_eval", 600, float(be_ser[:, j].mean()), float(be_sem[:, j].mean()))
            row("600_seed5", 600, float(tb["600_seed5"][0][j].mean()), float(tb["600_seed5"][1][j].mean()))
            row(f"{n_big}_seed6", n_big, float(tb[f"{n_big}_seed6"][0][j].mean()),
                float(tb[f"{n_big}_seed6"][1][j].mean()))
            row("all_support", len(support), float(exact_ser[j]), float(exact_sem[j]))
            rj = red[:, j].double()
            row("600_redraw", 600, float(rj.mean()), "", float(rj.quantile(0.05)), float(rj.quantile(0.95)),
                float((rj < ser).double().mean()))
            contrib = ser_all[j] * w_all
            rare = pdc[support] < 1.0 / 600
            top = contrib.sort(descending=True).values
            js["snr"][str(snr)] = dict(
                exact_ser=float(exact_ser[j]), exact_sem=float(exact_sem[j]),
                seed5_ser=float(tb["600_seed5"][0][j].mean()),
                seed5_rank_among_redraws=float((rj < tb["600_seed5"][0][j].mean()).double().mean()),
                redraws_below_exact=float((rj < exact_ser[j]).double().mean()),
                redraws_below_mc=float((rj < ser).double().mean()),
                share_from_pv_below_1_over_600=float(contrib[rare].sum() / contrib.sum()),
                pmass_pv_below_1_over_600=float(pdc[support][rare].sum()),
                share_top20_tokens=float(top[:20].sum() / contrib.sum()),
                mc_ser=ser, mc_se=se, mc_errors=nerr, mc_frames=fr)
            print(f"[exact] {name} {snr:>2} dB bound_eval {rows[-5][4]:.4g} 600 {rows[-4][4]:.4g} "
                  f"{n_big} {rows[-3][4]:.4g} all {rows[-2][4]:.4g} redraw [{rows[-1][6]:.3g},{rows[-1][7]:.3g}] "
                  f"| sim {ser:.4g} +- {se:.1g} ({nerr} err) | sem all {rows[-2][5]:.4g} sim {sem:.4g}",
                  flush=True)
        info_js[name] = js
    hdr = ["mask", "snr_db", "v_set", "n_tokens", "ub_ser", "ub_sem", "ub_ser_q05", "ub_ser_q95",
           "frac_below_mc", "mc_ser", "mc_ser_se", "mc_errors", "mc_sem", "mc_frames", "ratio_ser"]
    sfx = "_smoke" if a.smoke else ""
    write_csv(DATA / f"pm_bound_exact{sfx}.csv", hdr, rows)
    json.dump(info_js, open(DATA / f"pm_bound_exact{sfx}.json", "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["overlap", "exact"])
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False        # evaluation in full fp32
    torch.set_grad_enabled(False)
    t0 = time.time()
    {"overlap": part_overlap, "exact": part_exact}[a.part](a)
    print("[done]", a.part, round((time.time() - t0) / 60, 1), "min", flush=True)


if __name__ == "__main__":
    main()
