"""Common complex-baseband channel simulator for the revision experiments.

Every scheme produces a real transmit frame x in R^d with unit average
energy. Real dimensions (2k, 2k+1) form complex subcarrier k (the
real-imaginary stacking declared in the letter), optionally after a
fixed interleaver that spreads each period across the band. User u sees

    z_k = g_{u,k} x_k + w_k,     w_k ~ CN(0, 2 sigma_u^2),

with sigma_u^2 = 1/(d gamma_u). Flat fading uses one g_u ~ CN(0,1) for
all subcarriers; frequency-selective fading uses T equal-power taps,
g_{u,k} = sum_t a_t exp(-j 2 pi k t / K). The receiver holds the
estimate g_hat = g + e, e ~ CN(0, nmse) (one e per tap-domain draw for
flat fading, independent per subcarrier otherwise), zero-forces
z_k / g_hat_k, and restacks to a real vector r in R^d. It also returns
per-dimension reliability weights |g_hat_k|^2.

With perfect CSI and flat fading this reproduces the letter's model
exactly: the per-real-dimension noise after equalization has variance
sigma^2/|g|^2 with |g|^2 ~ Exp(1).
"""
from __future__ import annotations

import math

import torch

from sse_lib import BITS, DEVICE


def interleaver(d: int, P: int) -> torch.Tensor:
    """perm[j] = physical position of logical dim j (period-major logical
    order). Logical dim p*L + l goes to physical l*P + p."""
    L = d // P
    j = torch.arange(d)
    p, l = j // L, j % L
    return l * P + p


def channel(x: torch.Tensor, snr_db: float, users: int, *, taps: int = 1,
            nmse: float = 0.0, offsets_db=None, perm=None):
    """x: (N, d) real frames, unit average energy. Returns r (N,U,d) and
    w (N,U,d) in LOGICAL dimension order."""
    N, d = x.shape
    K = d // 2
    if perm is not None:
        xp = torch.empty_like(x)
        xp[:, perm.to(x.device)] = x
    else:
        xp = x
    xc = torch.complex(xp[:, 0::2], xp[:, 1::2])                  # (N,K)
    off = torch.zeros(users) if offsets_db is None else torch.tensor(
        offsets_db, dtype=torch.float32)
    snr_t = torch.as_tensor(snr_db, dtype=torch.float32).reshape(-1, 1)
    gam = 10.0 ** ((snr_t + off[None, :]) / 10.0)                 # (N or 1,U)
    sig = torch.sqrt(1.0 / (d * gam)).to(x.device)                 # (N or 1,U)

    def cn(*shape):
        return torch.complex(torch.randn(*shape, device=x.device),
                             torch.randn(*shape, device=x.device)) / math.sqrt(2)

    if taps == 1:
        g = cn(N, users, 1).expand(N, users, K)
        e = cn(N, users, 1).expand(N, users, K) * math.sqrt(nmse)
    else:
        a = cn(N, users, taps) / math.sqrt(taps)
        k = torch.arange(K, device=x.device)
        t = torch.arange(taps, device=x.device)
        F = torch.exp(-2j * math.pi * torch.outer(t, k).float() / K)  # (T,K)
        g = a @ F.to(a.dtype)
        e = cn(N, users, K) * math.sqrt(nmse)
    gh = g + e
    w = cn(N, users, K) * math.sqrt(2.0) * sig[:, :, None]
    z = g * xc[:, None, :] + w
    q = z / gh
    r = torch.stack([q.real, q.imag], dim=-1).reshape(N, users, d)
    wt = (gh.abs() ** 2).repeat_interleave(2, dim=-1)
    if perm is not None:
        pm = perm.to(x.device)
        r = r[:, :, pm]
        wt = wt[:, :, pm]
    return r, wt


# ----------------------------------------------------------------------
# decoders
# ----------------------------------------------------------------------
def sse_scores(model, r, w, rule: str = "corr"):
    """r, w: (N,U,d). Returns digit scores (N,U,P,Vu).
    corr : r^T (b_i . mu_u)                     (letter's rule)
    ncorr: r^T (b_i . mu_u) / ||b_i . mu_u||    (normalized correlation)
    md   : -||r - a (b_i . mu_u)||^2, a = 1/(c sqrt(P))  (minimum distance)
    wmd  : weighted minimum distance with weights |g_hat|^2"""
    N, Uu, d = r.shape
    P, L = model.P, model.L
    r = r.view(N, Uu, P, L)
    cand = model.unit_codebook()[None] * model.masks()[:, None, :]    # (U,Vu,L)
    cand = cand[:Uu]
    corr = torch.einsum("nupl,uvl->nupv", r, cand)
    if rule == "corr":
        return corr
    nrm2 = cand.pow(2).sum(-1)                                        # (U,Vu)
    if rule == "ncorr":
        return corr / nrm2.sqrt()[None, :, None, :]
    a = 1.0 / (float(model.c) * math.sqrt(P))
    if rule == "md":
        return 2 * a * corr - a * a * nrm2[None, :, None, :]
    if rule == "wmd":
        wv = w.view(N, Uu, P, L)
        wc = torch.einsum("nupl,uvl->nupv", wv * r, cand)
        wn = torch.einsum("nupl,uvl->nupv", wv, cand.pow(2))
        return 2 * a * wc - a * a * wn
    raise ValueError(rule)


def full_scores(model, r, w, rule: str = "corr", chunk_v: int = 65536):
    cand = model.codebook()[None] * model.masks()[:, None, :]         # (U,V,d)
    if rule == "corr":
        return torch.einsum("nud,uvd->nuv", r, cand)
    a = 1.0 / float(model.c)
    wc = torch.einsum("nud,uvd->nuv", w * r, cand)
    wn = torch.einsum("nud,uvd->nuv", w, cand.pow(2))
    return 2 * a * wc - a * a * wn


# ----------------------------------------------------------------------
# OMA with M-PAM per real dimension, same channel simulator
# ----------------------------------------------------------------------
def oma_layout(users: int, d: int, bits: int = BITS):
    """dims per user n = d/U. Returns (n_sym, M, rep): n_sym symbols of
    M-PAM, each repeated rep times, n_sym*rep = n, n_sym*log2 M = bits."""
    n = d // users
    if n >= bits:
        return bits, 2, n // bits
    bps = bits // n
    return n, 2 ** bps, 1


def pam_levels(M: int, device):
    lv = torch.arange(M, device=device, dtype=torch.float32) * 2 - (M - 1)
    return lv / math.sqrt((M * M - 1) / 3.0)          # unit average energy


def oma_frames(sym: torch.Tensor, users: int, d: int, M: int, rep: int):
    """sym: (N,U,n_sym) integer PAM symbols -> (N,d) real frames with unit
    average energy (each user owns a contiguous block of d/U dims)."""
    lv = pam_levels(M, sym.device)
    s = lv[sym].repeat_interleave(rep, dim=-1)        # (N,U,d/U)
    return s.reshape(sym.shape[0], d) / math.sqrt(d)


def oma_detect(r, w, users: int, d: int, M: int, rep: int):
    N = r.shape[0]
    n = d // users
    blk = torch.stack([r[:, u, u * n:(u + 1) * n] for u in range(users)], 1)
    wb = torch.stack([w[:, u, u * n:(u + 1) * n] for u in range(users)], 1)
    blk = blk.view(N, users, -1, rep)
    wb = wb.view(N, users, -1, rep)
    est = (blk * wb).sum(-1) / wb.sum(-1).clamp_min(1e-12) * math.sqrt(d)
    lv = pam_levels(M, r.device)
    return (est[..., None] - lv).abs().argmin(-1)
