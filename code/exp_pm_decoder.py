"""Masked candidate norms and decoding rules for the fixed-codebook design.

Setting of every part: BERT base uncased, codebook pwhite (exp_pm.codebook,
eq. (1)), d = 64 real dimensions = K = 32 subcarriers, U = 4 users, flat
Rayleigh fading, one token per user and frame, decisions over all
V = 30,522 tokens. Mask entries:
  white_U4_L16_flat_s1  trained static entry, L = 16 (exp_pm.py --part channel)
  white_U4_L64_flat_s1  trained full-length entry, L = 64, same run
  bertw_U4_L64_flat_s1  trained full-length entry, L = 64 (exp_pm.py --part
                        period), norms only
  comb                  untrained pattern without overlap, L = U
                        (exp_pm.fixed_pattern "comb")
  pair                  untrained pattern on whole subcarriers, L = 2U
                        (exp_pm.fixed_pattern "pair")
The trained entries keep their stored normalization c; for comb and pair
c is the exact expectation of the frame energy under p_v (c = 2).

part norms : masked candidate norms ||e_v (.) m_u|| per user (and pooled),
             weighted by the token frequencies p_v (tokens with p_v > 0) and
             uniform over all V tokens: mean, mean square, coefficient of
             variation, min, 1st/5th/50th/95th/99th percentiles, max, in
             absolute terms and relative to the mean of the same weighting.
             The 1/c of the frame is not applied (columns c_stored, c_exact
             give it). Output data/pm/pm_norms.csv. No random numbers.
part decoders: SER (and semantic distortion 1 - cos of the BERT input
             embeddings) at 10 and 20 dB under receiver NMSE 0, 0.01, 0.05,
             0.1 (rev_channel.channel, taps = 1: g_hat = g + e,
             e ~ CN(0, NMSE) common to all subcarriers) for the entries
             white_U4_L16_flat_s1, white_U4_L64_flat_s1, comb and pair.
             40,000 frames per point; tokens and channel seeded with
             11 + 10 SNR as exp_pm_ext.ser, so all decoders and all NMSE
             values see the same tokens, gains and noise, and rule (4) at
             NMSE 0 reproduces data/pm/pm_figs.csv (g1, static_L16).
             Output data/pm/pm_decoder.csv (one row per entry, SNR, NMSE and
             decoder: SER, per-user SER, distortion, and the fraction of
             decisions that differ from rule (4)).
  Decoders (r = equalized observation of user u, s_v = (e_v (.) m_u)/c,
  w = |g_hat|^2 on both real dimensions of a subcarrier):
    rule4    : argmin_v sum_i w_i (r_i - s_{v,i})^2  (eq. (4), exp_pm.Masks.scores)
    md       : argmin_v ||r - s_v||^2  (unweighted minimum distance)
    corr     : argmax_v r . (e_v (.) m_u)  (correlator of the earlier design)
    ncorr    : argmax_v r . (e_v (.) m_u) / ||e_v (.) m_u||  (normalized correlation)
    mismatch : Gaussian likelihood given the estimate (formula below), with
               the other users' signal as Gaussian interference of power q_i
               on dimension i
    mismatch_q0: the same likelihood with q = 0, i.e. without the other
               users' signal that the residual gain error moves into the
               dimensions of user u (ablation)
  Mismatch-aware rule. With g ~ CN(0,1) and e ~ CN(0,eps), g | g_hat ~
  CN(alpha g_hat, beta), alpha = 1/(1+eps), beta = eps/(1+eps). Then
  r = (g/g_hat)(s_v + i_u) + n/g_hat, and the rule takes
    r | v ~ N(alpha s_v, Sigma_v),  Sigma_v = Delta + b (s s^T + Js (Js)^T),
    b = beta/(2|g_hat|^2),  Delta = diag(delta_i),
    delta_i = sigma^2/|g_hat|^2 + alpha^2 q_i + b (q_i + q_{i'}),
  sigma^2 = 1/(d SNR) the noise variance per real dimension, i' the
  partner dimension of the same subcarrier, J the 90-degree rotation of
  every subcarrier ((Js)_{2k} = -s_{2k+1}, (Js)_{2k+1} = s_{2k}), and
  q_i = c^-2 sum_{k != u} m_{k,i}^2 E_p[e_{v,i}^2] (q = 0 for "mismatch_q0").
  With W = Delta^-1, P = s^T W r, Q = (Js)^T W r, G11 = s^T W s,
  G22 = (Js)^T W Js, G12 = s^T W Js, h1 = P - alpha G11, h2 = Q - alpha G12,
  Lam = (1 + b G11)(1 + b G22) - b^2 G12^2, the decision maximizes
  -(r - alpha s)^T Sigma_v^-1 (r - alpha s) - log det Sigma_v:
    v_hat = argmax_v 2 alpha P - alpha^2 G11
            + b [(1 + b G22) h1^2 - 2 b G12 h1 h2 + (1 + b G11) h2^2] / Lam
            - log Lam.
  The first two terms are the weighted distance of rule (4) with the gain
  shrunk by alpha; the third credits the energy that the residual gain
  error rotates or scales out of s_v, and log Lam charges each candidate
  for the error variance its own energy draws. With eps = 0 and q = 0 the
  rule is rule (4); with eps = 0 it differs from rule (4) only through
  q_i in the weights (not at all for comb and pair). Under flat fading with q = 0 it reduces to
    (|g_hat|^2/sigma^2) (kappa (p^2 + p'^2) + 2 alpha p - alpha^2 S)/(1 + kappa S)
    - 2 log(1 + kappa S),  kappa = beta/(2 sigma^2),
  with p = s^T r, p' = (Js)^T r, S = ||s||^2. --smoke checks this closed
  form, and the general metric against the explicit 64 x 64 Gaussian
  log-likelihood on a few frames and tokens.

Run:        CUDA_VISIBLE_DEVICES=1 python exp_pm_decoder.py --part norms
            CUDA_VISIBLE_DEVICES=1 python exp_pm_decoder.py --part decoders
            (--part all runs both; --smoke: 2,048 frames, outputs *_smoke.csv)
"""
from __future__ import annotations

import argparse
import math
import time

import torch

import rev_channel as rc
from sse_lib import DEVICE, D, write_csv
import exp_pm as pm
import exp_pm_ext as pmx

U = 4
DATA = pm.DATA                                   # data/pm/
SNRS = (10, 20)
NMSES = (0.0, 0.01, 0.05, 0.1)
DECODERS = ("rule4", "md", "corr", "ncorr", "mismatch", "mismatch_q0")


# ----------------------------------------------------------------------
# entries
# ----------------------------------------------------------------------
def exact_c(E: torch.Tensor, M: torch.Tensor, pd: torch.Tensor) -> float:
    """sqrt of E||sum_u e_{v_u} (.) m_u||^2 for independent tokens drawn
    from pd: sum_i [s2_i sum_u m_ui^2 + mu_i^2 ((sum_u m_ui)^2 - sum_u m_ui^2)]."""
    E, M, pd = E.double(), M.double(), pd.double().to(E.device)
    mu = pd @ E
    s2 = pd @ E.pow(2)
    m2 = M.pow(2).sum(0)
    return float((s2 * m2 + mu.pow(2) * (M.sum(0).pow(2) - m2)).sum().sqrt())


def load_saved(name: str) -> pm.Masks:
    sd = torch.load(pm.MODELS / f"{name}.pt", map_location="cpu")
    m = pm.Masks(sd["E"], sd["Wm"].shape[1], U)
    m.load_state_dict(sd)
    return m.to(DEVICE).eval()


def fixed(E: torch.Tensor, kind: str, pd: torch.Tensor) -> pm.Masks:
    m = pm.fixed_pattern(E.cpu(), U, kind)
    m.c.fill_(exact_c(E.cpu(), m.masks().detach(), pd.cpu()))
    return m.to(DEVICE).eval()


def entries(E0, pdc, names):
    out = {}
    for nm in names:
        if nm in ("comb", "pair"):
            out[nm] = fixed(E0, nm, pdc)
            print(f"[entry] {nm} L={out[nm].L} c={float(out[nm].c):.6f} (exact)", flush=True)
        else:
            m = load_saved(nm)
            dev = float((m.E.cpu() - E0).abs().max())
            print(f"[entry] {nm} L={m.L} c={float(m.c):.6f} codebook max|diff|={dev:.2e}", flush=True)
            assert dev < 1e-5, "stored codebook differs from pwhite"
            out[nm] = m
    return out


# ----------------------------------------------------------------------
# part norms
# ----------------------------------------------------------------------
QS = (0.01, 0.05, 0.50, 0.95, 0.99)


def wstats(x: torch.Tensor, w: torch.Tensor) -> list[float]:
    keep = w > 0
    x, w = x[keep].double(), w[keep].double()
    w = w / w.sum()
    mean = float((w * x).sum())
    msq = float((w * x * x).sum())
    sd = math.sqrt(max(msq - mean * mean, 0.0))
    o = torch.argsort(x)
    xs, cw = x[o], torch.cumsum(w[o], 0)
    qs = [float(xs[min(int(torch.searchsorted(cw, torch.tensor(t, dtype=cw.dtype))), len(xs) - 1)])
          for t in QS]
    absv = [float(xs[0])] + qs + [float(xs[-1])]
    return [mean, msq, sd / mean] + absv + [v / mean for v in absv]


NORM_HDR = (["entry", "L", "user", "weighting", "n_tokens", "c_stored", "c_exact", "mean", "mean_sq", "cv"]
            + [f"{k}" for k in ("min", "p01", "p05", "p50", "p95", "p99", "max")]
            + [f"{k}_rel" for k in ("min", "p01", "p05", "p50", "p95", "p99", "max")])


@torch.no_grad()
def part_norms(a):
    src = pm.Source("bert")
    pdc = src.dist()
    E0 = pm.codebook(src.W, D, "pwhite", pd=pdc)
    names = ["white_U4_L16_flat_s1", "white_U4_L64_flat_s1", "bertw_U4_L64_flat_s1", "comb", "pair"]
    rows = []
    V = E0.shape[0]
    mu_e = pdc.double() @ E0.double()
    print(f"[codebook] V={V} support={int((pdc > 0).sum())} ||E_p[e]||={float(mu_e.norm()):.4f}", flush=True)
    for nm, m in entries(E0, pdc, names).items():
        M = m.masks().detach().cpu()
        E = m.E.cpu()
        nrm = (E[None] * M[:, None, :]).norm(dim=-1)                 # (U,V)
        c_st, c_ex = float(m.c), exact_c(E, M, pdc)
        groups = [(str(u), nrm[u], pdc, torch.full((V,), 1.0 / V)) for u in range(U)]
        groups.append(("all", nrm.reshape(-1), pdc.repeat(U) / U, torch.full((U * V,), 1.0 / (U * V))))
        for ul, x, wp, wu in groups:
            for wl, w in (("p_v", wp), ("uniform", wu)):
                st = wstats(x, w)
                rows.append([nm, m.L, ul, wl, int((w > 0).sum()), c_st, c_ex] + st)
                print(f"[norms] {nm} u={ul} {wl}: mean {st[0]:.4f} cv {st[2]:.4f} "
                      f"rel min {st[10]:.3f} p01 {st[11]:.3f} p99 {st[15]:.3f} max {st[16]:.3f}", flush=True)
    write_csv(DATA / f"pm_norms{a.tag}.csv", NORM_HDR, rows)


# ----------------------------------------------------------------------
# part decoders
# ----------------------------------------------------------------------
def swap_pairs(x: torch.Tensor) -> torch.Tensor:
    """exchange the two real dimensions of every subcarrier (last axis)."""
    return x.view(*x.shape[:-1], -1, 2).flip(-1).reshape(x.shape)


def rot_t(x: torch.Tensor) -> torch.Tensor:
    """J^T x: (J^T x)_{2k} = x_{2k+1}, (J^T x)_{2k+1} = -x_{2k}."""
    y = torch.empty_like(x)
    y[..., 0::2] = x[..., 1::2]
    y[..., 1::2] = -x[..., 0::2]
    return y


def mismatch_metric(r, gh2, cand, cand_sq, cand_pr, q, sig2, eps):
    """Mismatch-aware log-likelihood (module docstring) for one user.
    r (n,d), gh2 = |g_hat|^2 (n,), cand (V,d) scaled candidates s_v,
    cand_sq = cand^2, cand_pr (V,K) = s_{2k} s_{2k+1}, q (d,) interference
    power. Returns (n,V) in float64."""
    alpha, beta = 1.0 / (1.0 + eps), eps / (1.0 + eps)
    b = (beta / (2.0 * gh2))[:, None]                                  # (n,1)
    delta = sig2 / gh2[:, None] + alpha ** 2 * q[None] + b * (q + swap_pairs(q))[None]
    W = 1.0 / delta                                                    # (n,d)
    Wr = W * r
    P = (Wr @ cand.T).double()
    Q = (rot_t(Wr) @ cand.T).double()
    G11 = (W @ cand_sq.T).double()
    G22 = (swap_pairs(W) @ cand_sq.T).double()
    G12 = ((W[:, 1::2] - W[:, 0::2]) @ cand_pr.T).double()
    b = b.double()
    h1 = P - alpha * G11
    h2 = Q - alpha * G12
    A11, A22 = 1.0 + b * G11, 1.0 + b * G22
    lam = A11 * A22 - (b * G12) ** 2
    quad = (A22 * h1 * h1 - 2.0 * b * G12 * h1 * h2 + A11 * h2 * h2) / lam
    return 2.0 * alpha * P - alpha ** 2 * G11 + b * quad - torch.log(lam)


def mismatch_flat_q0(r, gh2, cand, S, sig2, eps):
    """closed form of mismatch_metric for flat fading and q = 0 (self-check)."""
    alpha, beta = 1.0 / (1.0 + eps), eps / (1.0 + eps)
    kap = beta / (2.0 * sig2)
    p = (r @ cand.T).double()
    pp = (rot_t(r) @ cand.T).double()
    S = S.double()[None]
    om = (gh2.double() / sig2)[:, None]
    return om * (kap * (p * p + pp * pp) + 2 * alpha * p - alpha ** 2 * S) / (1 + kap * S) \
        - 2.0 * torch.log1p(kap * S)


def brute_check(r, gh2, cand, q, sig2, eps):
    """explicit Gaussian log-likelihood with the full covariance Sigma_v
    against mismatch_metric + r^T W r + log det Delta (float64)."""
    alpha, beta = 1.0 / (1.0 + eps), eps / (1.0 + eps)
    r, gh2, cand, q = r.double(), gh2.double(), cand.double(), q.double()
    met = mismatch_metric(r.float(), gh2.float(), cand.float(), cand.float().pow(2),
                          (cand[:, 0::2] * cand[:, 1::2]).float(), q.float(), sig2, eps)
    worst = 0.0
    for n in range(r.shape[0]):
        b = beta / (2.0 * float(gh2[n]))
        delta = sig2 / gh2[n] + alpha ** 2 * q + b * (q + swap_pairs(q))
        const = float((r[n] ** 2 / delta).sum() + torch.log(delta).sum())
        for v in range(cand.shape[0]):
            s = cand[v]
            js = torch.empty_like(s)
            js[0::2], js[1::2] = -s[1::2], s[0::2]
            Sig = torch.diag(delta) + b * (torch.outer(s, s) + torch.outer(js, js))
            x = r[n] - alpha * s
            ll = -float(x @ torch.linalg.solve(Sig, x)) - float(torch.linalg.slogdet(Sig)[1])
            worst = max(worst, abs(ll + const - float(met[n, v])) / max(1.0, abs(ll)))
    print(f"[check] explicit likelihood vs mismatch_metric: max rel. diff {worst:.2e}", flush=True)


@torch.no_grad()
def run_decoders(m, pd, Wn, snr, eps, frames, seed=11, chunk=512, sub=256, check=False):
    M = m.masks().detach()
    ac = 1.0 / float(m.c)
    E = m.E
    cand = E[None] * M[:, None, :] * ac                                # (U,V,d)
    cand_sq = cand.pow(2)
    cand_pr = cand[..., 0::2] * cand[..., 1::2]                        # (U,V,K)
    S = cand_sq.sum(-1)                                                # (U,V)
    Sn = S.sqrt()
    s2e = pd @ E.pow(2)                                                # (d,)
    m2 = M.pow(2)
    q_int = ac * ac * (m2.sum(0, keepdim=True) - m2) * s2e[None]       # (U,d)
    q0 = torch.zeros(D, device=DEVICE)
    sig2 = 1.0 / (D * 10 ** (snr / 10))
    cond = dict(taps=1, nmse=eps, offsets_db=None)
    err = {k: torch.zeros(U, device=DEVICE) for k in DECODERS}
    sem = {k: torch.zeros(U, device=DEVICE) for k in DECODERS}
    diff = {k: 0 for k in DECODERS}
    chk = [0, 0, 0]
    torch.manual_seed(seed + int(10 * snr))
    for k0 in range(0, frames, chunk):
        n = min(chunk, frames - k0)
        idx = pmx.sample(pd, n, U, 0.0)
        x = m.tx(idx) / m.c
        r, w = rc.channel(x, snr, U, **cond)
        dec = {"rule4": m.scores(r, w).argmax(-1)}                     # (n,U)
        for k in DECODERS[1:]:
            dec[k] = torch.empty_like(idx)
        for u in range(U):
            for s0 in range(0, n, sub):                                # bounds GPU memory only
                sl = slice(s0, min(s0 + sub, n))
                ru, gh2 = r[sl, u], w[sl, u, 0]
                P = ru @ cand[u].T                                     # (n',V)
                dec["corr"][sl, u] = P.argmax(1)
                dec["ncorr"][sl, u] = (P / Sn[u][None]).argmax(1)
                dec["md"][sl, u] = (2 * P - S[u][None]).argmax(1)
                del P
                dec["mismatch"][sl, u] = mismatch_metric(
                    ru, gh2, cand[u], cand_sq[u], cand_pr[u], q_int[u], sig2, eps).argmax(1)
                mm = mismatch_metric(ru, gh2, cand[u], cand_sq[u], cand_pr[u], q0, sig2, eps).argmax(1)
                dec["mismatch_q0"][sl, u] = mm
                if check:
                    cf = mismatch_flat_q0(ru, gh2, cand[u], S[u], sig2, eps).argmax(1)
                    chk[0] += int((cf == mm).sum()); chk[1] += ru.shape[0]
                    if k0 == 0 and s0 == 0:
                        brute_check(ru[:3], gh2[:3], cand[u][:40], q_int[u], sig2, eps)
        if check:
            chk[2] += int((dec["md"] == dec["rule4"]).sum())
        for k in DECODERS:
            wrong = dec[k] != idx
            err[k] += wrong.float().sum(0)
            sem[k] += (1 - (Wn[idx] * Wn[dec[k]]).sum(-1)).sum(0)
            diff[k] += int((dec[k] != dec["rule4"]).sum())
    if check:
        print(f"[check] closed form agrees with mismatch_metric on {chk[0]}/{chk[1]} decisions; "
              f"md agrees with rule4 on {chk[2]}/{chk[1]}", flush=True)
    out = {}
    for k in DECODERS:
        out[k] = dict(ser=float(err[k].sum()) / (frames * U), ser_u=[float(v) / frames for v in err[k]],
                      sem=float(sem[k].sum()) / (frames * U), diff=diff[k] / (frames * U))
    return out


DEC_HDR = ["entry", "L", "mean_iota", "snr_db", "nmse", "decoder", "frames", "ser",
           "ser_u0", "ser_u1", "ser_u2", "ser_u3", "sem_dist", "frac_diff_rule4"]


def part_decoders(a):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    src = pm.Source("bert")
    pdc = src.dist()
    E0 = pm.codebook(src.W, D, "pwhite", pd=pdc)
    pd = pdc.to(DEVICE)
    Wn = src.Wn.to(DEVICE)
    names = ["white_U4_L16_flat_s1", "comb"] if a.smoke else \
        ["white_U4_L16_flat_s1", "white_U4_L64_flat_s1", "comb", "pair"]
    snrs, nmses = ((20,), (0.0, 0.05)) if a.smoke else (SNRS, NMSES)
    frames = 2048 if a.smoke else a.frames
    rows = []
    for nm, m in entries(E0, pdc, names).items():
        io = sum(pm.cross_energy(m.masks().detach().cpu())) / U
        for snr in snrs:
            for eps in nmses:
                t = time.time()
                res = run_decoders(m, pd, Wn, snr, eps, frames, check=a.smoke)
                for k in DECODERS:
                    v = res[k]
                    rows.append([nm, m.L, io, snr, eps, k, frames, v["ser"]] + v["ser_u"]
                                + [v["sem"], v["diff"]])
                print(f"[dec] {nm} {snr} dB nmse {eps}: "
                      + " ".join(f"{k} {res[k]['ser']:.4f}" for k in DECODERS)
                      + f" ({time.time() - t:.0f} s)", flush=True)
    write_csv(DATA / f"pm_decoder{a.tag}.csv", DEC_HDR, rows)


def part_all(a):
    part_norms(a)
    part_decoders(a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["norms", "decoders", "all"])
    ap.add_argument("--frames", type=int, default=40000)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    a.tag = "_smoke" if a.smoke else ""
    t0 = time.time()
    {"norms": part_norms, "decoders": part_decoders, "all": part_all}[a.part](a)
    print("[done]", a.part, round((time.time() - t0) / 60, 1), "min", flush=True)


if __name__ == "__main__":
    main()
