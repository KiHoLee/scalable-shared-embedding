"""Revision (reported model below OMA under flat fading): mechanism.

Question. The reported model sse_P4.pt has kappa_u = 0.252 on average, almost
the value 0.25 of binary OMA, yet its SER is below OMA at every SNR of Fig. 2.
Why? No training here.

Codes compared, both at frame energy 1 (Sec. II, gamma = 1/(d sigma^2)):
  SSE  for user u the 16 masked digit candidates of one period at the transmit
       scale actually used, x_i = b_i . mu_u / (c sqrt(P)) in R^L, L = 16.
       A frame carries P = 4 such digits, all with the same gain h.
  OMA  the user owns d/U = 16 dimensions, 4 per period, and sends one bit per
       dimension with amplitude 1/sqrt(d). Per period that is the binary 4-cube
       (16 points, energy 4/d = 1/16 per digit, d_OMA^2 = 4/d). Its 16 bits with
       one gain are exactly 4 cube digits with one gain.
Instantaneous SNR s = h^2 gamma; OMA's bit error is then Q(sqrt(s)) and a pair
of points at squared distance d2 has pairwise error Q(sqrt(s d2/d_OMA^2)).

Per (run, user), geometry:
  E_avg_rel, E_min_rel, E_max_rel   energy per digit over 1/16 (OMA)
  centroid_rel                      energy of the mean point over 1/16
  eff_dim                           participation ratio of the eigenvalues of
                                    the centred 16 candidates (cube: 4)
  eig_shares                        five largest eigenvalue shares
  kappa                             P d_min^2 (Proposition 2; OMA 0.25)
  d2min_rel                         d_min^2 / d_OMA^2 (= kappa / 0.25)
  d2min_over_E                      d_min^2 / E_avg (cube: 1)
  nn_own                            neighbours per point within 1.05 d_min^2
  n_le_<t>                          neighbours per point within t d_OMA^2
                                    (cube: 4 at t = 1, 10 at 2, 14 at 3)
  neff_s10, neff_s20                union-bound digit error over Q(sqrt(s)) at
                                    s = 10 and 20 dB, the effective number of
                                    OMA-distance neighbours (cube: 4.0x)
  gain_db_1e-1/1e-2/1e-3            fixed-gain SNR gain over the cube at a digit
                                    error of 1e-1, 1e-2, 1e-3 (positive: SSE needs
                                    less), from the AWGN curves below
  area_iso, area_ub                 A = int_0^inf P_frame(s) ds, so that
                                    SER * gamma -> A at high SNR under Rayleigh
                                    fading; from the Monte Carlo AWGN curve and
                                    from the union bound
  area_ratio_iso, area_ratio_ub     A over the cube's A (high-SNR SER ratio)
Per (run, user, SNR in 0:2:20 dB):
  ser_meas    per-user SER of the trained model as in the letter (all users
              superposed, correlation decoder (5)), 1e6 frames, seed 777+10 SNR
  ser_iso     the user's code alone (no interference), minimum-distance decoder,
              Rayleigh gain per frame, 1e6 frames
  dig_iso     digit error rate of that run
  ub          full-spectrum union bound 1 - (1 - min(1, (1/Vu) sum_{i!=j}
              Q(.)))^P over h^2 ~ Exp(1) on a 1e6-point quantile grid
  ser_oma     closed form E_h[1 - (1 - Q(h sqrt(gamma)))^16] (sse_lib.oma_ser)
  ser_oma_cube  Monte Carlo of 4 cube digits with one gain (same routine as
              ser_iso), checks ser_oma
  dig_oma     cube digit error rate
  ub_oma      the same union bound with the cube spectrum
  ratio_meas_oma, ratio_iso_oma   ser_meas / ser_oma, ser_iso / ser_oma
User "all" rows hold the mean over the users; the row run = OMA holds the cube.

Runs: sse_P4.pt (reported), rev_sse_seed1..5.pt (the other five of the six),
rev_factor_16_4_s1..s10.pt (ten further runs).

Outputs: data/rev_spectrum.csv, data/rev_spectrum_pairs.csv (every pair
distance of every code, the full distance spectrum),
data/rev_spectrum_awgn.csv (fixed-gain digit error curves).
"""
import math

import numpy as np
import torch

import exp_rev_analysis as A
from sse_lib import D, DATA, DEVICE, SSE, oma_ser, write_csv

SNRS = list(range(0, 21, 2))
FRAMES = 1_000_000
CHUNK = 250_000
S_DB = np.round(np.arange(-10.0, 20.001, 0.25), 2)
AWGN_N = 1_000_000
TARGETS = (1e-1, 1e-2, 1e-3)
NBR_T = (1.0, 1.05, 1.1, 1.25, 1.5, 2.0, 3.0)
NGRID = 1_000_000
D2_OMA = 4.0 / D
E_OMA = 4.0 / D            # energy per digit (4 bits of energy 1/D)
VU, P = 16, 4


def runs():
    yield "reported", "sse_P4.pt"
    for s in range(1, 6):
        yield f"seed{s}", f"rev_sse_seed{s}.pt"
    for s in range(1, 11):
        yield f"factor_s{s}", f"rev_factor_16_4_s{s}.pt"


def load(name):
    m = SSE(P=P, vu=VU)
    m.load_state_dict(torch.load(DATA / name, map_location="cpu"))
    return m.to(DEVICE).eval()


@torch.no_grad()
def codes(m):
    Bn, mu, c = m.unit_codebook().double(), m.masks().double(), float(m.c)
    return [Bn * mu[u] / (c * math.sqrt(m.P)) for u in range(m.users)]


def cube():
    bits = torch.tensor([[2.0 * ((i >> k) & 1) - 1.0 for k in range(4)]
                         for i in range(VU)], dtype=torch.float64, device=DEVICE)
    return bits / math.sqrt(D)


def qfun(x):
    return 0.5 * torch.erfc(x / math.sqrt(2.0))


def rel_pairs(X):
    d2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    off = ~torch.eye(X.shape[0], dtype=torch.bool, device=X.device)
    return d2, off, d2[off] / D2_OMA                 # ordered pairs (240,)


def ub_digit(rel, s):
    """(1/Vu) sum_{i != j} Q(sqrt(s rel_ij)), capped at 1; s is a tensor."""
    out = torch.empty_like(s)
    for k0 in range(0, len(s), 20_000):
        ss = s[k0:k0 + 20_000]
        out[k0:k0 + 20_000] = qfun(torch.sqrt(ss[:, None] * rel[None, :])).sum(1) / VU
    return out.clamp(max=1.0)


def ub_rayleigh(rel, snrs):
    tau = torch.from_numpy(-np.log1p(-(np.arange(NGRID) + 0.5) / NGRID)).to(DEVICE)
    out = []
    for g in snrs:
        p = ub_digit(rel, tau * 10 ** (g / 10))
        out.append(float((1.0 - (1.0 - p) ** P).mean()))
    return out


def area_ub(rel):
    s = torch.linspace(0.0, 400.0, 400_001, dtype=torch.float64, device=DEVICE)
    pf = 1.0 - (1.0 - ub_digit(rel, s)) ** P
    return float(torch.trapezoid(pf, s))


def area_curve(s_db, pd):
    """A = int_0^inf (1 - (1 - p_d(s))^P) ds on the grid, P_frame = 1 below it."""
    s = 10 ** (np.asarray(s_db) / 10)
    pf = 1.0 - (1.0 - np.asarray(pd)) ** P
    return float(s[0] * 1.0 + np.trapezoid(pf, s))


@torch.no_grad()
def md_decide(X32, nrm2, r):
    return (2 * r @ X32.T - nrm2).argmax(-1)


@torch.no_grad()
def awgn_digit(X, s_db, n=AWGN_N, seed=1234):
    """Digit error of the isolated code at a fixed instantaneous SNR s."""
    X32 = X.float()
    nrm2 = X32.pow(2).sum(1)
    out = []
    for k, sd in enumerate(s_db):
        torch.manual_seed(seed + k)
        sig = math.sqrt(1.0 / (D * 10 ** (sd / 10)))
        err = 0
        for n0 in range(0, n, CHUNK):
            b = min(CHUNK, n - n0)
            dg = torch.randint(VU, (b,), device=DEVICE)
            r = X32[dg] + sig * torch.randn(b, X32.shape[1], device=DEVICE)
            err += int((md_decide(X32, nrm2, r) != dg).sum())
        out.append(err / n)
    return out


@torch.no_grad()
def rayleigh_iso(X, snrs, frames=FRAMES, seed=777):
    """Frame and digit error of the isolated code, one Rayleigh gain per frame."""
    X32 = X.float()
    nrm2 = X32.pow(2).sum(1)
    fr, dg_out = [], []
    for g in snrs:
        torch.manual_seed(seed + int(10 * g))
        ef = ed = 0
        for n0 in range(0, frames, CHUNK):
            b = min(CHUNK, frames - n0)
            dg = torch.randint(VU, (b, P), device=DEVICE)
            tau = -torch.log(torch.rand(b, device=DEVICE).clamp_min(1e-12))
            sig = torch.sqrt(1.0 / (D * 10 ** (g / 10) * tau))
            r = X32[dg] + sig[:, None, None] * torch.randn(b, P, X32.shape[1], device=DEVICE)
            wrong = md_decide(X32, nrm2, r) != dg
            ef += int(wrong.any(1).sum()); ed += int(wrong.sum())
        fr.append(ef / frames); dg_out.append(ed / (frames * P))
    return fr, dg_out


@torch.no_grad()
def meas_per_user(m, snrs, frames=1_000_000, chunk=100_000, seed=777):
    """Per-user SER of the trained model as in the letter (exp_rev_factor_mech)."""
    out = []
    for g in snrs:
        torch.manual_seed(seed + int(10 * g))
        err = torch.zeros(m.users, dtype=torch.float64)
        for n0 in range(0, frames, chunk):
            b = min(chunk, frames - n0)
            dg = torch.randint(m.vu, (b, m.users, m.P), device=DEVICE)
            r = A.period_obs(m, dg, g)
            err += (A.decide(m, r) != dg).any(2).double().sum(0).cpu()
        out.append((err / frames).tolist())
    return out                                    # [snr][user]


def crossing(s_db, pd, target):
    """s (dB) where the digit error falls to target, by log-linear interpolation."""
    lp = np.log10(np.maximum(np.asarray(pd, dtype=float), 1e-12))
    t = math.log10(target)
    for k in range(len(lp) - 1):
        if lp[k] >= t > lp[k + 1]:
            return s_db[k] + (s_db[k + 1] - s_db[k]) * (lp[k] - t) / (lp[k] - lp[k + 1])
    return float("nan")


@torch.no_grad()
def geometry(X):
    d2, off, rel = rel_pairs(X)
    E = X.pow(2).sum(1)
    cen = X.mean(0)
    Xc = X - cen
    lam = torch.linalg.eigvalsh(Xc.T @ Xc / X.shape[0]).flip(0).clamp_min(0)
    dmin2 = float(d2[off].min())
    g = dict(E_avg_rel=float(E.mean()) / E_OMA, E_min_rel=float(E.min()) / E_OMA,
             E_max_rel=float(E.max()) / E_OMA, centroid_rel=float(cen.pow(2).sum()) / E_OMA,
             eff_dim=float(lam.sum() ** 2 / lam.pow(2).sum()),
             eig_shares=" ".join(f"{float(v):.4f}" for v in (lam / lam.sum())[:5]),
             kappa=P * dmin2, d2min_rel=dmin2 / D2_OMA, d2min_over_E=dmin2 / float(E.mean()),
             nn_own=float(((d2 <= 1.05 * dmin2) & off).sum(1).double().mean()))
    relm = d2 / D2_OMA
    for t in NBR_T:
        g[f"n_le_{t:g}"] = float(((relm <= t * (1 + 1e-9)) & off).sum(1).double().mean())
    for sd in (10.0, 20.0):
        s = torch.tensor([10 ** (sd / 10)], dtype=torch.float64, device=DEVICE)
        g[f"neff_s{int(sd)}"] = float(ub_digit(rel, s)[0]) / float(qfun(torch.sqrt(s))[0])
    return g, d2, rel


GEO = ["E_avg_rel", "E_min_rel", "E_max_rel", "centroid_rel", "eff_dim", "eig_shares",
       "kappa", "d2min_rel", "d2min_over_E", "nn_own"] + \
      [f"n_le_{t:g}" for t in NBR_T] + \
      ["neff_s10", "neff_s20", "gain_db_1e-1", "gain_db_1e-2", "gain_db_1e-3",
       "area_iso", "area_ub", "area_ratio_iso", "area_ratio_ub"]
SERC = ["ser_meas", "ser_iso", "dig_iso", "ub", "ser_oma", "ser_oma_cube", "dig_oma",
        "ub_oma", "ratio_meas_oma", "ratio_iso_oma"]
HEADER = ["run", "model", "user", "snr_db"] + GEO + SERC


def main():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rows, pairs, awgn = [], [], []

    # ---------------- OMA reference: the binary 4-cube --------------------
    C = cube()
    g_oma, d2c, rel_c = geometry(C)
    pd_cube_cf = [1 - (1 - 0.5 * math.erfc(math.sqrt(10 ** (sd / 10)) / math.sqrt(2))) ** 4
                  for sd in S_DB]
    pd_cube_mc = awgn_digit(C, S_DB)
    cross_c = {t: crossing(S_DB, pd_cube_cf, t) for t in TARGETS}
    A_cube_iso = area_curve(S_DB, pd_cube_cf)
    A_cube_ub = area_ub(rel_c)
    ser_oma = oma_ser(SNRS)
    ser_cube, dig_cube = rayleigh_iso(C, SNRS)
    ub_oma = ub_rayleigh(rel_c, SNRS)
    g_oma.update({"gain_db_1e-1": 0.0, "gain_db_1e-2": 0.0, "gain_db_1e-3": 0.0,
                  "area_iso": A_cube_iso, "area_ub": A_cube_ub,
                  "area_ratio_iso": 1.0, "area_ratio_ub": 1.0})
    for k, g in enumerate(SNRS):
        rows.append(["OMA", "binary 4-cube", "cube", g] + [g_oma[c] for c in GEO] +
                    [ser_oma[k], ser_cube[k], dig_cube[k], ub_oma[k], ser_oma[k],
                     ser_cube[k], dig_cube[k], ub_oma[k], 1.0, ser_cube[k] / ser_oma[k]])
    for i in range(VU):
        for j in range(i + 1, VU):
            pairs.append(["OMA", "binary 4-cube", "cube", i, j, float(d2c[i, j]),
                          float(d2c[i, j]) / D2_OMA])
    for sd, a, b in zip(S_DB, pd_cube_cf, pd_cube_mc):
        awgn.append(["OMA", "binary 4-cube", "cube", float(sd), b, a])
    print("[OMA] A_iso", A_cube_iso, "A_ub", A_cube_ub, "ser", ser_oma[5], ser_oma[10],
          "cube MC", ser_cube[5], ser_cube[10], "ub", ub_oma[5], ub_oma[10],
          "cross", cross_c, flush=True)

    # ---------------- trained runs -----------------------------------------
    for run, name in runs():
        m = load(name)
        Xs = codes(m)
        meas = meas_per_user(m, SNRS)
        per_user = []
        for u, X in enumerate(Xs):
            g, d2, rel = geometry(X)
            pd = awgn_digit(X, S_DB)
            for t in TARGETS:
                g[f"gain_db_{t:.0e}".replace("e-0", "e-")] = cross_c[t] - crossing(S_DB, pd, t)
            g["area_iso"] = area_curve(S_DB, pd)
            g["area_ub"] = area_ub(rel)
            g["area_ratio_iso"] = g["area_iso"] / A_cube_iso
            g["area_ratio_ub"] = g["area_ub"] / A_cube_ub
            fr, dg = rayleigh_iso(X, SNRS)
            ub = ub_rayleigh(rel, SNRS)
            ser_u = [meas[k][u] for k in range(len(SNRS))]
            per_user.append((g, ser_u, fr, dg, ub))
            for k, s in enumerate(SNRS):
                rows.append([run, name, u, s] + [g[c] for c in GEO] +
                            [ser_u[k], fr[k], dg[k], ub[k], ser_oma[k], ser_cube[k],
                             dig_cube[k], ub_oma[k], ser_u[k] / ser_oma[k], fr[k] / ser_oma[k]])
            for i in range(VU):
                for j in range(i + 1, VU):
                    pairs.append([run, name, u, i, j, float(d2[i, j]),
                                  float(d2[i, j]) / D2_OMA])
            for sd, b, a in zip(S_DB, pd, pd_cube_cf):
                awgn.append([run, name, u, float(sd), b, a])
            print(f"[{run}] u{u} eff_dim {g['eff_dim']:.2f} d2min_rel {g['d2min_rel']:.3f} "
                  f"n_le_1.1 {g['n_le_1.1']:.2f} n_le_1.25 {g['n_le_1.25']:.2f} "
                  f"n_le_2 {g['n_le_2']:.2f} neff20 {g['neff_s20']:.2f} "
                  f"gain1e-2 {g['gain_db_1e-2']:.3f} dB Aratio {g['area_ratio_iso']:.3f} "
                  f"ser10 meas {ser_u[5]:.4f} iso {fr[5]:.4f} ub {ub[5]:.4f} "
                  f"ser20 meas {ser_u[10]:.5f} iso {fr[10]:.5f}", flush=True)
        # mean over users
        gm = {}
        for c in GEO:
            vals = [pu[0][c] for pu in per_user]
            gm[c] = "" if isinstance(vals[0], str) else sum(vals) / len(vals)
        for k, s in enumerate(SNRS):
            sm = sum(pu[1][k] for pu in per_user) / len(per_user)
            si = sum(pu[2][k] for pu in per_user) / len(per_user)
            di = sum(pu[3][k] for pu in per_user) / len(per_user)
            ub = sum(pu[4][k] for pu in per_user) / len(per_user)
            rows.append([run, name, "all", s] + [gm[c] for c in GEO] +
                        [sm, si, di, ub, ser_oma[k], ser_cube[k], dig_cube[k], ub_oma[k],
                         sm / ser_oma[k], si / ser_oma[k]])
        r10 = rows[-6]; r20 = rows[-1]
        print(f"[{run}] ALL ser10 meas {r10[-10]:.4f} iso {r10[-9]:.4f} oma {ser_oma[5]:.4f} | "
              f"ser20 meas {r20[-10]:.5f} iso {r20[-9]:.5f} oma {ser_oma[10]:.5f}", flush=True)
        write_csv(DATA / "rev_spectrum.csv", HEADER, rows)
        write_csv(DATA / "rev_spectrum_pairs.csv",
                  ["run", "model", "user", "i", "j", "d2", "d2_rel"], pairs)
        write_csv(DATA / "rev_spectrum_awgn.csv",
                  ["run", "model", "user", "s_db", "pd_mc", "pd_cube_closed"], awgn)


if __name__ == "__main__":
    main()
