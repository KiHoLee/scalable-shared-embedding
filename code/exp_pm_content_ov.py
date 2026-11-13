"""Adapted masks with a partial overlap chosen for the shared-token probability rho.

Question. The two-level adapted masks of Figs. 2 and 3 (exp_pm_ext.adaptive_masks:
sqrt(U) on J/U blocks and zero elsewhere) do not overlap, so under the receiver
of eq. (4) their SER does not change with rho (pm_content_fs.csv, rows
adapted_std). The static entries trained for rho place small real weights of
matching sign on the other users' positions. This script gives the adapted
masks the same freedom with one real number per entry: user u keeps the level
a on the blocks that its two-level mask weights and places the level b = t a,
of the same sign, on every other block, with a^2 + (U - 1) b^2 = U, so that
||m_u||^2 = d for every t. t = 0 is the two-level mask and t = 1 full overlap
with one common mask of equal weights. The update still broadcasts only the
user index of each block; t belongs to the entry (load, channel class, rho),
as the patterns of a static entry do, so it adds no bits per update.

Normalization. Every dimension carries the level a for one user and b for the
U - 1 others, so sum_u m_{u,i}^2 = U and sum_u m_{u,i} = a + (U - 1) b on every
dimension whatever the block assignment, and c depends on rho and t only. It is
calibrated on 65,536 frames drawn at rho with seed 0 on the comb assignment
(exp_pm_content_fs.adapt_c does the same at t = 0).

Receiver. Eq. (4): user u decodes its own token by weighted minimum distance
over all V tokens with its own mask m_u (known from the broadcast user indices
and t) and the weights |g|^2.

Grid: channel classes fs4 and fs8 (exact gains), blocks of B = 2 subcarriers
(J = 16), SNR 5, 10 and 15 dB, rho in RHOS, t in T_GRID. Every (channel, SNR,
rho, t) runs on two frame sets: a validation set (seed 21 + int(10 snr) + 100 T,
batches of 512) on which the best t of every (channel, SNR, rho) is chosen, and
the test set of exp_pm_content_fs.py part main (seed 11 + int(10 snr), batches
of 512, the same draws of tokens, gains and noise) on which every t is
reported. At t = 0 and 10 dB the test set reproduces the adapted_std rows of
pm_content_fs.csv (check rows in the JSON).

Outputs: data/pm/pm_content_ov.csv (columns channel, snr_db, rho, t, split,
frames, tokens, errors, ter, ser_sem, ser_sem_shared, ser_sem_indep,
cross_energy, c) and data/pm/pm_content_ov.json (best t per channel, SNR and rho
chosen on validation with its test SER against t = 0, checks, minutes). With
--smoke (2,048 frames, fs4, 10 dB, rho in {0, 0.9}, t in {0, 0.3}) everything
goes to *_smoke files.

Run from the code folder on a GPU:
    python exp_pm_content_ov.py
    python exp_pm_content_ov.py --smoke
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time

import torch

import exp_pm as pm
import exp_pm_content as pc
import exp_pm_content_fs as cf
import exp_pm_ext as px
from sse_lib import DEVICE, D, write_csv

U = cf.U
B = cf.B
RHOS = (0.0, 0.25, 0.5, 0.75, 0.9, 1.0)
T_GRID = (0.0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.3, 0.5, 1.0)   # cross-energy 0 to 3
SNRS = (5, 10, 15)
FRAMES = 40_000
VAL_SEED, TEST_SEED = 21, cf.TEST_SEED
HDR = ["channel", "snr_db", "rho", "t", "split", "frames", "tokens", "errors", "ter", "ser_sem",
       "ser_sem_shared", "ser_sem_indep", "cross_energy", "c"]


def levels(t: float):
    """Levels a (own blocks) and b = t a (other blocks) with a^2 + (U - 1) b^2 = U."""
    a = math.sqrt(U / (1.0 + (U - 1) * t * t))
    return a, t * a


def overlap_masks(M2: torch.Tensor, t: float) -> torch.Tensor:
    """Masks with the partial overlap t from the two-level masks M2 (n,U,D)."""
    a, b = levels(t)
    return torch.where(M2 > 0, torch.full_like(M2, a), torch.full_like(M2, b))


def iota(t: float) -> float:
    """Mean cross-energy (7) of the masks with overlap t (the same for every
    user and every block assignment)."""
    a, b = levels(t)
    own = a * a * (U - 1) * b * b                  # dimensions that the user's own blocks hold
    oth = b * b * (a * a + (U - 2) * b * b)        # every other dimension
    return (own + (U - 1) * oth) / U


@torch.no_grad()
def calib_c(ctx, rho: float, t: float, n: int = 65536) -> float:
    """c for the overlap t at rho (comb assignment, seed 0)."""
    comb = torch.zeros(1, U, D, device=DEVICE)
    for u in range(U):
        comb[0, u, u::U] = math.sqrt(U)
    Mt = overlap_masks(comb, t)[0]
    torch.manual_seed(0)
    idx, _ = pc.draw_all(ctx.pd, n, U, rho)
    return float(((ctx.E[idx] * Mt[None]).sum(1)).pow(2).sum(1).mean().sqrt())


@torch.no_grad()
def run(ctx, cs: dict, rho: float, snr: float, taps: int, frames: int, seed: int, ts=T_GRID):
    """All t of ts on the same frames (the draws of exp_pm_content_fs.run_adapted)."""
    torch.backends.cuda.matmul.allow_tf32 = False
    E = ctx.E
    E2 = E * E
    K = D // 2
    sig = math.sqrt(1.0 / (D * 10 ** (snr / 10)))
    c0 = cf.adapt_c(ctx)                           # the constant c of the two-level masks
    accs = {t: cf.Acc(ctx) for t in ts}
    torch.manual_seed(seed)
    for n, _ in cf.chunks(frames, cf.BATCH):
        idx, flags = pc.draw_all(ctx.pd, n, U, rho)
        g = px.gains(n, taps, DEVICE)
        M2 = px.adaptive_masks(g, B, sig * sig, c0, False)            # (n,U,D) two levels
        w = torch.complex(torch.randn(n, U, K, device=DEVICE), torch.randn(n, U, K, device=DEVICE)) * sig
        wt = (g.abs() ** 2).repeat_interleave(2, dim=-1)
        for t in ts:
            M = M2 if t == 0.0 else overlap_masks(M2, t)
            c = c0 if t == 0.0 else cs[t]
            x = (E[idx] * M).sum(1) / c
            xc = torch.complex(x[:, 0::2], x[:, 1::2])
            q = (g * xc[:, None, :] + w) / g
            r = torch.stack([q.real, q.imag], -1).reshape(n, U, D)
            a_ = 1.0 / c
            sc = 2 * a_ * torch.einsum("nud,vd->nuv", wt * r * M, E) \
                - a_ * a_ * torch.einsum("nud,vd->nuv", wt * M * M, E2)
            accs[t].add(idx, sc.argmax(-1), flags)
            del sc
    return accs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    tag = "_smoke" if a.smoke else ""
    chans = cf.CHANNELS[:1] if a.smoke else cf.CHANNELS
    snrs = (10,) if a.smoke else SNRS
    rhos = (0.0, 0.9) if a.smoke else RHOS
    ts = (0.0, 0.3) if a.smoke else T_GRID
    frames = 2048 if a.smoke else FRAMES
    ctx = cf.Ctx()
    print(f"[setup] V {ctx.V} device {DEVICE} c0 {cf.adapt_c(ctx):.6f} t={time.time() - t0:.0f}s", flush=True)
    rows, js = [], dict(setup=dict(users=U, d=D, block_subcarriers=B, J=D // (2 * B), rhos=list(rhos),
                                   t_grid=list(ts), snrs=list(snrs), frames=frames,
                                   val_seed="21 + int(10 snr) + 100 T", test_seed="11 + int(10 snr)",
                                   batch=cf.BATCH, levels={f"{t:g}": levels(t) for t in ts},
                                   cross_energy={f"{t:g}": iota(t) for t in ts}),
                        c={}, best={}, checks={})
    path = pm.DATA / f"pm_content_ov{tag}.csv"
    jpath = pm.DATA / f"pm_content_ov{tag}.json"
    for rho in rhos:
        cs = {t: calib_c(ctx, rho, t) for t in ts if t > 0}
        js["c"][f"{rho:g}"] = {f"{t:g}": v for t, v in cs.items()}
        for cn, taps in chans:
            for snr in snrs:
                res = {}
                for split, seed in (("val", VAL_SEED + int(10 * snr) + 100 * taps), ("test", TEST_SEED + int(10 * snr))):
                    accs = run(ctx, cs, rho, snr, taps, frames, seed, ts)
                    for t, acc in accs.items():
                        r = acc.row(cn, "ov", snr, rho, iota(t))
                        c = cf.adapt_c(ctx) if t == 0.0 else cs[t]
                        row = [cn, snr, rho, t, split, r[4], r[5], r[6], r[7], r[8], r[9], r[10], iota(t), c]
                        rows.append(row)
                        res[(split, t)] = row
                        print(f"[ov] {cn} {snr} dB rho {rho:g} t {t:g} {split} errors {r[6]} ter {r[7]:.5f} "
                              f"ser {r[8]:.5f} | {time.time() - t0:.0f}s", flush=True)
                    write_csv(path, HDR, rows)
                tb = min(ts, key=lambda t: res[("val", t)][9])
                key = f"{cn}|{snr}|{rho:g}"
                js["best"][key] = dict(t=tb, ser_test=res[("test", tb)][9], ser_test_t0=res[("test", 0.0)][9],
                                       ratio=res[("test", 0.0)][9] / res[("test", tb)][9] if res[("test", tb)][9] else None,
                                       ser_val=res[("val", tb)][9], ser_val_t0=res[("val", 0.0)][9])
                print(f"[best] {key} t {tb:g} test ser {res[('test', tb)][9]:.5f} vs t=0 {res[('test', 0.0)][9]:.5f}",
                      flush=True)
                if snr == 10 and not a.smoke:      # reproduction of pm_content_fs.csv adapted_std
                    with open(pm.DATA / "pm_content_fs.csv") as f:
                        ref = [x for x in csv.DictReader(f) if x["channel"] == cn and x["scheme"] == "adapted_std"
                               and int(x["snr_db"]) == 10 and abs(float(x["rho"]) - rho) < 1e-9]
                    if ref:
                        js["checks"][key] = dict(released_errors=int(ref[0]["errors"]),
                                                 here_errors=res[("test", 0.0)][7],
                                                 equal=int(ref[0]["errors"]) == res[("test", 0.0)][7])
                js["minutes"] = round((time.time() - t0) / 60, 1)
                with open(jpath, "w") as f:
                    json.dump(js, f, indent=1)
    print("[done]", round((time.time() - t0) / 60, 1), "min", flush=True)


if __name__ == "__main__":
    main()
