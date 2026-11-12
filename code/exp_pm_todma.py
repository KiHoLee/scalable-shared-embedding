"""Token-domain multiple access (ToDMA) reference for the channel figure (Fig. 2).

ToDMA [qiao2025todma] maps every token to a codeword of a shared orthonormal
DFT codebook whose length equals the codebook size, detects the active
tokens at a 256-antenna base station by projection and an energy threshold,
assigns them to the devices with the CSI as an anchor, and fills collisions
by masked-token prediction. In the downlink frame of the letter (d = 64 real
dimensions, V = 30,522 BERT tokens) an orthonormal codebook cannot exist and
every receiver sees all tokens through its own single channel, so the
reference is adapted as in the original submission:
  - every token has a fixed unit-norm random Gaussian codeword of length d,
    unrelated to its meaning (exp_pm.codebook kind "rand", seed 0, the
    codewords of the random-codebook reference); the frame is the plain sum
    of the U codewords, scaled to unit average energy under the token
    frequencies p_v, without user masks;
  - each receiver zero-forces its channel (rev_channel.channel) and detects U
    tokens by orthogonal matching pursuit with signed correlation (every
    coefficient is known to be positive) and a weighted least-squares refit
    after every pick. Dimension k is weighted by 1/(sigma^2/|g_hat_k|^2 + I0),
    the inverse of its noise plus the interference of the U-1 other
    codewords, I0 = (U-1)/(c^2 d). This weighting gave the lowest SER of the
    tested ones at every SNR and channel (part detector): pure noise
    whitening (|g_hat|^2) leaves a floor of 0.35 to 0.41 under 4 and 8 taps
    even without noise, because it suppresses faded subcarriers that carry
    the frame exactly after zero forcing, and uniform weights lose at low
    SNR;
  - a genie counts user u correct whenever its own token is among the U
    detected tokens, which replaces the CSI-anchored assignment and the
    masked-token prediction and favors ToDMA.
SER = missed own tokens / (U frames), tokens drawn i.i.d. with p_v, U = 4.

Parts (--part):
  fig       flat, fs4, fs8, 0-20 dB in steps of 2, 200,000 frames per point
            (data/pm/pm_todma.csv; columns as pm_figs_class.csv)
  detector  detector variants at 6, 10, 20 dB and without noise (300 dB):
            proj (one weighted correlation, top U, as the projection of
            [qiao2025todma]), omp_abs (absolute correlation), omp (signed),
            each with the weights ipn (used), wh (|g_hat|^2) or zf (uniform),
            100,000 frames (data/pm/pm_todma_detector.csv)
  energy    average frame energy on the AG News test streams with c fixed
            under p_v (data/pm/pm_todma_energy.json)
  all       the three parts in one process

Run from the code folder on a GPU:
    python exp_pm_todma.py --part all
"""
from __future__ import annotations

import argparse
import json
import math
import time

import torch

import exp_pm as pm
import rev_channel as rc
from sse_lib import DEVICE, D, write_csv

U = 4
FRAMES = 200_000
CHUNK = 1024
SNRS = list(range(0, 21, 2))
CONDS = ("flat", "fs4", "fs8")
HDR = ["graph", "channel", "scheme", "snr_db", "frames", "ser"]


def setup():
    src = pm.Source("bert")
    pd = src.dist()
    A = pm.codebook(src.W, D, "rand", seed=0)            # (V, d), unit-norm rows
    return src, A.to(DEVICE), pd.to(DEVICE)


@torch.no_grad()
def calibrate(A, pd, n=65536, seed=0):
    """c with c^2 = E||sum_u a_{s_u}||^2 under p_v (unit average frame energy)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    idx = torch.multinomial(pd.cpu(), n * U, replacement=True, generator=g).view(n, U).to(DEVICE)
    return float(A[idx].sum(1).pow(2).sum(1).mean().sqrt())


@torch.no_grad()
def detect(r, wt, A, mode="omp", k=U):
    """r, wt: (n, d) zero-forced observation and weights |g_hat|^2 of one
    receiver; A: (V, d). Returns the k detected token ids (n, k)."""
    AT = A.T                                                    # (d, V)
    nrm = (wt @ AT.pow(2)).clamp_min(1e-12).sqrt()              # (n, V) whitened atom norms
    if mode == "proj":
        return ((wt * r) @ AT / nrm).topk(k, dim=1).indices
    n = r.shape[0]
    sup = torch.zeros(n, k, dtype=torch.long, device=r.device)
    R = r.clone()
    eye = torch.eye(k, device=r.device)
    for t in range(k):
        c = (wt * R) @ AT / nrm
        if mode == "omp_abs":
            c = c.abs()
        if t > 0:
            c.scatter_(1, sup[:, :t], -float("inf"))
        sup[:, t] = c.argmax(1)
        As = A[sup[:, :t + 1]]                                  # (n, t+1, d)
        Aw = As * wt[:, None, :]
        G = torch.einsum("nkd,njd->nkj", Aw, As) + 1e-6 * eye[:t + 1, :t + 1]
        b = (Aw * r[:, None, :]).sum(2, keepdim=True)
        beta = torch.linalg.solve(G, b)
        R = r - (As * beta).sum(1)
    return sup


def weights(g2, snr, c, kind="ipn"):
    """Per-dimension detector weights from g2 = |g_hat|^2 (n, U, d)."""
    if kind == "zf":
        return torch.ones_like(g2)
    if kind == "wh":
        return g2
    sig2 = 1.0 / (D * 10 ** (snr / 10))                   # noise per real dimension
    i0 = (U - 1) / (c * c * D)                             # interference of U-1 codewords
    return 1.0 / (sig2 / g2.clamp_min(1e-12) + i0)


@torch.no_grad()
def ser(A, c, pd, snr, cond, frames, seed, mode="omp", wkind="ipn"):
    torch.manual_seed(seed + int(10 * snr))
    err = 0
    for k0 in range(0, frames, CHUNK):
        n = min(CHUNK, frames - k0)
        idx = torch.multinomial(pd, n * U, replacement=True).view(n, U)
        x = A[idx].sum(1) / c
        r, g2 = rc.channel(x, snr, U, **pm.COND[cond])
        wt = weights(g2, snr, c, wkind)
        for u in range(U):
            sup = detect(r[:, u], wt[:, u], A, mode)
            err += int((~(sup == idx[:, u:u + 1]).any(1)).sum())
    return err / (frames * U)


def part_fig(A, c, pd):
    rows = []
    for cn in CONDS:
        for snr in SNRS:
            t0 = time.time()
            v = ser(A, c, pd, snr, cn, FRAMES, seed=71)
            rows.append(["g1", cn, "todma", snr, FRAMES, v])
            print("[fig]", cn, snr, f"{v:.4e}", f"{time.time() - t0:.1f}s", flush=True)
            write_csv(pm.DATA / "pm_todma.csv", HDR, rows)


def part_detector(A, c, pd):
    rows = []
    variants = [("proj", "ipn"), ("omp_abs", "ipn"), ("omp", "ipn"), ("omp", "wh"), ("omp", "zf")]
    for cn in CONDS:
        for snr in (6, 10, 20, 300):
            for mode, wk in variants:
                v = ser(A, c, pd, snr, cn, 100_000, seed=73, mode=mode, wkind=wk)
                rows.append([cn, mode, wk, snr, 100_000, v])
                print("[detector]", cn, snr, mode, wk, f"{v:.4e}", flush=True)
                write_csv(pm.DATA / "pm_todma_detector.csv",
                          ["channel", "detector", "weights", "snr_db", "frames", "ser"], rows)


def part_energy(src, A, c):
    ids, _ = src.streams(U)
    ids = ids.to(DEVICE)
    e = A[ids].sum(1).pow(2).sum(1) / (c * c)
    out = {"c": c, "c2": c * c, "frames": int(ids.shape[0]),
           "mean_energy_test_streams": float(e.mean()),
           "same_token_share_test_streams": float((ids[:, :, None] == ids[:, None, :])
                                                   .triu(1).any(-1).any(-1).float().mean())}
    (pm.DATA / "pm_todma_energy.json").write_text(json.dumps(out, indent=1))
    print("[energy]", out, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all", choices=["fig", "detector", "energy", "all"])
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False             # evaluation in full fp32
    src, A, pd = setup()
    c = calibrate(A, pd)
    print("[setup]", "V", A.shape[0], "d", A.shape[1], "c", f"{c:.4f}", "device", DEVICE, flush=True)
    if a.part in ("energy", "all"):
        part_energy(src, A, c)
    if a.part in ("detector", "all"):
        part_detector(A, c, pd)
    if a.part in ("fig", "all"):
        part_fig(A, c, pd)


if __name__ == "__main__":
    main()
