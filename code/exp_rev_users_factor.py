"""Revision R1.6: other factorizations at U = 8, d = 64 (load 2 bits per dimension).

Trains (Vu,P) = (256,2), (4,8), (2,16) over seeds 1-6 at the 4,000-iteration
budget, plus (256,2) with seed 1 at ten times that budget, under the letter's real
model (train_sse). Checkpoints: data/rev_users_factor_{Vu}_{P}_s{seed}[_10x].pt,
reused if present. Evaluates each on the flat complex channel of rev_channel.py
with correlation decoding (5), FRAMES frames at 10 and 20 dB (mean and worst
user), and reports kappa_u, delta_u^2 = kappa_u/P, and the largest mask overlap.
References on the same channel: OMA with four-level PAM (exp_rev_robust.run), and
the existing checkpoints rev_sse_U8.pt (the U = 8 row of the sweep, Vu=16, P=4),
rev_sse_U8_P4.pt and rev_sse_U8_P2.pt (no generating script of their own).

The ideal packing: PACK_RESTARTS restarts of an annealed soft-min repulsion of N
unit points in L/U dimensions, N = 256 in 4 (P = 2) and N = 16 in 2 (P = 4). Since the masks
are disjoint and the codewords have equal norm, delta^2 = chord^2/(U P), which is
compared with 4 a^2/d of four-level OMA. The value is a numerical lower bound on
the best packing. The best (256,4) configuration is stored in
data/rev_users_factor_pack256_4.pt. Each packing is also evaluated as a design
inside the SSE class (same points in each user's block, disjoint masks), through
the same channel and decoder, to test that the limit belongs to training and not
to the frame.

Rows by kind: oma, reference, trained, packing_design. The column chord2 is nan
except for packing_design.

Output: data/rev_users_factor.csv
"""
import copy

import numpy as np
import torch

from exp_rev_robust import run
from exp_rev_users_rho import kappa_over_p, oma_dmin2
from sse_lib import D, DATA, DEVICE, SSE, set_seed, train_sse, write_csv
import rev_channel as rc

USERS = 8
PAIRS = [(256, 2), (4, 8), (2, 16)]
SEEDS = range(1, 7)
LONG_ITERS = 40_000
FRAMES = 1_000_000
CHUNK = 20_000
SNRS = (10.0, 20.0)
PACK_RESTARTS = 12
PACK_ITERS = 12_000
HDR = ["kind", "name", "Vu", "P", "seed", "iters", "ser10", "worst10", "ser20",
       "worst20", "kappa_mean", "delta2_mean", "delta2_min", "mask_offdiag",
       "chord2"]


@torch.no_grad()
def ser_complex(m, snr, frames=FRAMES, chunk=CHUNK, seed=777):
    """Per-user SER on the flat complex channel with correlation decoding."""
    torch.manual_seed(seed + int(10 * snr))
    U = m.users
    err = torch.zeros(U)
    tot = 0
    for n0 in range(0, frames, chunk):
        n = min(chunk, frames - n0)
        dg = torch.randint(m.vu, (n, U, m.P), device=DEVICE)
        x = m.tx_frame(dg).reshape(n, m.d) / m.c
        r, w = rc.channel(x, snr, U)
        err += (rc.sse_scores(m, r, w, "corr").argmax(-1) != dg).any(-1).float().sum(0).cpu()
        tot += n
    return err / tot


@torch.no_grad()
def describe(m):
    """(kappa mean, delta^2 mean, delta^2 min, largest mask overlap)."""
    dl = kappa_over_p(copy.deepcopy(m).cpu())       # this helper runs on CPU
    W = m.masks()
    G = W @ W.T / m.L
    off = float((G - torch.eye(m.users, device=G.device)).abs().max())
    return float(np.mean(dl)) * m.P, float(np.mean(dl)), float(min(dl)), off


def result_row(kind, name, m, seed, iters, chord2):
    s = [ser_complex(m, x) for x in SNRS]
    k, dm, dn, off = describe(m)
    return (kind, name, m.vu, m.P, seed, iters, float(s[0].mean()), float(s[0].max()),
            float(s[1].mean()), float(s[1].max()), k, dm, dn, off, chord2)


def load_or_train(vu, P, seed, iters):
    tag = "_10x" if iters > 4000 else ""
    path = DATA / f"rev_users_factor_{vu}_{P}_s{seed}{tag}.pt"
    set_seed(seed)
    m = SSE(P=P, vu=vu, d=D, users=USERS)
    if path.exists():
        m.load_state_dict(torch.load(path, map_location="cpu"))
        return m.to(DEVICE).eval(), path.name
    train_sse(m, iters=iters, seed=seed)
    torch.save({k: v.cpu() for k, v in m.state_dict().items()}, path)
    return m.to(DEVICE).eval(), path.name


def pack(N, n):
    """Best unit-norm configuration over the restarts; returns (chord^2, points)."""
    best, pts = -1.0, None
    for s in range(PACK_RESTARTS):
        torch.manual_seed(s)
        X = torch.randn(N, n, device=DEVICE).requires_grad_(True)
        opt = torch.optim.Adam([X], lr=0.01)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, PACK_ITERS, eta_min=1e-4)
        for i in range(PACK_ITERS):
            Y = torch.nn.functional.normalize(X, dim=1)
            d2 = 2.0 - 2.0 * Y @ Y.T + 4.0 * torch.eye(N, device=DEVICE)
            T = 0.02 * 0.05 ** (i / PACK_ITERS)        # 0.02 down to 0.001
            loss = T * torch.logsumexp(-d2.flatten() / T, 0)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        Y = torch.nn.functional.normalize(X.detach().double(), dim=1)
        d2 = 2.0 - 2.0 * Y @ Y.T + 4.0 * torch.eye(N, device=DEVICE, dtype=Y.dtype)
        v = float(d2.min())
        print(f"[pack {N},{n}] restart {s} chord2 {v:.5f}", flush=True)
        if v > best:
            best, pts = v, Y.cpu()
    return best, pts


def design(points, P):
    """SSE with the same points in each user's block and disjoint masks."""
    N, n = points.shape
    L = D // P
    assert n * USERS == L
    m = SSE(P=P, vu=N, d=D, users=USERS)
    W = torch.zeros(USERS, L)
    for u in range(USERS):
        W[u, u * n:(u + 1) * n] = 1.0
    m.B.data.copy_(points.float().repeat(1, USERS))
    m.W.data.copy_(W)
    m.calibrate_power()
    return m.to(DEVICE).eval()


def main():
    nan = float("nan")
    rows = []

    def add(row):
        rows.append(row)
        print(row, flush=True)
        write_csv(DATA / "rev_users_factor.csv", HDR, rows)

    oma = [run("oma", None, USERS, s, FRAMES, 100_000, 777) for s in SNRS]
    add(("oma", "oma_4pam", 0, 0, 0, 0, float(oma[0].mean()), float(oma[0].max()),
         float(oma[1].mean()), float(oma[1].max()), nan, oma_dmin2(USERS, D),
         oma_dmin2(USERS, D), 0.0, nan))

    for name, vu, P in (("rev_sse_U8.pt", 16, 4), ("rev_sse_U8_P4.pt", 16, 4),
                        ("rev_sse_U8_P2.pt", 256, 2)):
        m = SSE(P=P, vu=vu, d=D, users=USERS)
        m.load_state_dict(torch.load(DATA / name, map_location="cpu"))
        add(result_row("reference", name, m.to(DEVICE).eval(), 0, 4000, nan))

    for vu, P in PAIRS:
        for seed in SEEDS:
            m, name = load_or_train(vu, P, seed, 4000)
            add(result_row("trained", name, m, seed, 4000, nan))
    m, name = load_or_train(256, 2, 1, LONG_ITERS)
    add(result_row("trained", name, m, 1, LONG_ITERS, nan))

    # ideal packing, numerically for (256,4) and (16,2)
    ch256, p256 = pack(256, 4)
    torch.save(p256, DATA / "rev_users_factor_pack256_4.pt")
    ch16, p16 = pack(16, 2)
    for ch, pts, P, label in ((ch256, p256, 2, "pack_256_in_4"), (ch16, p16, 4, "pack_16_in_2")):
        m = design(pts, P)
        add(result_row("packing_design", label, m, 0, 0, ch))
        print(f"[{label}] chord2 {ch:.5f} delta2 formula {ch / (USERS * P):.6f} "
              f"OMA {oma_dmin2(USERS, D):.6f}", flush=True)


if __name__ == "__main__":
    main()
