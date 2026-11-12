"""Semantic error rate of the adapted masks of Fig. 4 against the mask size J.

Metric. The source alphabet is the BERT base uncased vocabulary of V = 30,522
tokens with input token embeddings w_v in R^768. The semantic similarity of
two tokens is the cosine of their embeddings,
    zeta(v, w) = w_v^T w_w / (||w_v|| ||w_w||),
and the semantic error rate (SER) of user u is
    SER_u = E[1 - zeta(s_u, s_hat_u)],
the expectation over the sent token s_u ~ p_v, the fading gains and the
noise, with s_hat_u the decoded token. A correct token gives zeta = 1 and
costs 0, a wrong one costs 1 - zeta in [0, 2]. The token error rate is
TER_u = Pr[s_hat_u != s_u]. Over N frames of U users the estimates are
    ser_sem = (1/(N U)) sum_n sum_u [1 - zeta(s_{n,u}, s_hat_{n,u})],
    ter     = errors / tokens,  errors = #{(n,u): s_hat_{n,u} != s_{n,u}},
    tokens  = N U.
The cosine is the rule of exp_pm.Evaluator.score: row-normalized table Wn,
decoded ids outside [0, V) mapped to [UNK] before the cosine. The cost of a
correctly decoded token is set to exactly 0 (the float32 self-cosine of a
row of Wn is 1 only up to rounding).

Parts (--part):
  adapted  masks adapted to known subcarrier gains (exp_pm_ext.unit_masks,
           the scheme "adapted" of exp_pm_ext.py --part sig): J blocks of
           64/J real dimensions, J in {1, 2, 4, 8, 16, 32, 64}; with J >= U
           the users take turns and the masks do not overlap, with J < U
           each user sets its level on its best block and the masks overlap;
           J = 16 are the blocks of B = 2
           subcarriers of Fig. 2. Channels with T = 4 and 8 taps (classes
           fs4, fs8 of exp_pm.COND), U = 4, 10 dB for every J, and 0 and
           20 dB for J = 16 and 32. 40,000 frames per point; a point with
           fewer than 100 token errors is run again with 400,000 frames from
           the same seed (its first 39,936 frames are the same frames).

Seeds. The constant c of the adapted masks is drawn with torch.manual_seed(0)
on 65,536 frames, as in exp_pm_ext.py --part sig and exp_pm_chanhi.py --part
hi. Each point uses torch.manual_seed(31 + int(10 snr) + 100 T + int(7 ud/2)),
ud = 64/J, with the chunks and draws of exp_pm_ext.run_units, so the 10 dB
ter equals pm_sig.csv (scheme adapted), and J = 16 with 400,000 frames
equals pm_figs_hi.csv (adapted_B2, same seed through run_adapt with B = 2).
pm_adapt.csv and graph g1 of pm_figs.csv drew c without a seed, so their
40,000-frame values can differ from the first pass by a few errors.
The semantic accumulation draws no random numbers.

No mask is trained here: the adapted masks follow from the gains in every
frame, so nothing is saved under pm_models.

Outputs in data/pm/: pm_semfig4_adapted.csv (smoke: pm_semfig4_adapted_smoke.csv)
with columns scheme, channel, taps, snr_db, values (J), unit_dims (64/J),
seed, frames, errors, tokens, ter, ser_sem.

Run from the code folder on a GPU:
    python exp_pm_semfig4a.py --part adapted --smoke
    python exp_pm_semfig4a.py --part adapted
"""
from __future__ import annotations

import argparse
import csv
import math
import time

import torch

import exp_pm as pm
import exp_pm_ext as px
from sse_lib import DEVICE, D, write_csv

U = px.U
FRAMES = 40_000
FRAMES_HI = 400_000
MIN_ERR = 100
JS = (64, 32, 16, 8, 4, 2, 1)
CLASSES = ((4, "fs4"), (8, "fs8"))
HDR = ["scheme", "channel", "taps", "snr_db", "values", "unit_dims", "seed",
       "frames", "errors", "tokens", "ter", "ser_sem"]


def setup():
    src = pm.Source("bert")
    pdc = src.dist()
    E = pm.codebook(src.W, D, "pwhite", pd=pdc).to(DEVICE)
    pd = pdc.to(DEVICE)
    Wn = src.Wn.to(DEVICE)
    comb = torch.zeros(U, D, device=DEVICE)
    for u in range(U):
        comb[u, u::4] = 2.0
    torch.manual_seed(0)                       # fixed draw for the constant c
    idx0 = px.sample(pd, 65536, U, 0.0)
    c = float(((E[idx0] * comb[None]).sum(1)).pow(2).sum(1).mean().sqrt())
    return src, E, pd, Wn, c


def seed_of(snr, taps, ud, seed=31):
    return seed + int(10 * snr) + 100 * taps + int(7 * ud / 2)


@torch.no_grad()
def run_units_sem(E, pd, Wn, unk, c, snr, taps, ud, frames, seed=31):
    """exp_pm_ext.run_units with the semantic cost accumulated next to the
    token errors; same seed, chunks and random draws. Returns (errors,
    tokens, ser_sem)."""
    torch.manual_seed(seed_of(snr, taps, ud, seed))
    K = D // 2
    V = E.shape[0]
    sig = math.sqrt(1.0 / (D * 10 ** (snr / 10)))
    err = 0
    sem = torch.zeros((), dtype=torch.float64, device=DEVICE)
    for k0 in range(0, frames, 384):
        n = min(384, frames - k0)
        idx = px.sample(pd, n, U, 0.0)
        g = px.gains(n, taps, DEVICE)
        M = px.unit_masks(g, ud, sig * sig)
        x = (E[idx] * M).sum(1) / c
        xc = torch.complex(x[:, 0::2], x[:, 1::2])
        w = torch.complex(torch.randn(n, U, K, device=DEVICE), torch.randn(n, U, K, device=DEVICE)) * sig
        q = (g * xc[:, None, :] + w) / g
        r = torch.stack([q.real, q.imag], -1).reshape(n, U, D)
        wt = (g.abs() ** 2).repeat_interleave(2, dim=-1)
        a_ = 1.0 / c
        sc = 2 * a_ * torch.einsum("nud,vd->nuv", wt * r * M, E) \
            - a_ * a_ * torch.einsum("nud,vd->nuv", wt * M * M, E * E)
        dec = sc.argmax(-1)
        dec = torch.where((dec < 0) | (dec >= V), torch.full_like(dec, unk), dec)
        wrong = dec != idx
        err += int(wrong.sum())
        cos = (Wn[idx] * Wn[dec]).sum(-1)
        sem += torch.where(wrong, 1.0 - cos, torch.zeros_like(cos)).double().sum()
    tokens = frames * U
    return err, tokens, float(sem) / tokens


def point(ctx, snr, taps, cn, J, frames, frames_hi, min_err, first):
    src, E, pd, Wn, c = ctx
    ud = D // J
    sd = seed_of(snr, taps, ud)
    e, t, s = run_units_sem(E, pd, Wn, src.unk, c, snr, taps, ud, frames)
    first[(cn, snr, J)] = (frames, e, t, e / t, s)
    print(f"[pass] {cn} {snr} dB J={J} frames {frames} errors {e} ter {e / t:.4e} ser_sem {s:.4e}",
          flush=True)
    fr = frames
    if e < min_err:
        fr = frames_hi
        e, t, s = run_units_sem(E, pd, Wn, src.unk, c, snr, taps, ud, fr)
        print(f"[pass] {cn} {snr} dB J={J} frames {fr} errors {e} ter {e / t:.4e} ser_sem {s:.4e}",
              flush=True)
    return ["adapted", cn, taps, snr, J, ud, sd, fr, e, t, e / t, s]


def read_rows(name):
    p = pm.DATA / name
    if not p.exists():
        return []
    with open(p) as f:
        return list(csv.DictReader(f))


def repro(rows, first):
    """TER against the released data of the same scheme."""
    sig = {(r["channel"], int(r["values"])): float(r["ser"])
           for r in read_rows("pm_sig.csv") if r["scheme"] == "adapted"}
    hi = {(r["channel"], int(r["snr_db"])): (int(r["frames"]), float(r["ser"]))
          for r in read_rows("pm_figs_hi.csv") if r["scheme"] == "adapted_B2"}
    ad = {(f"fs{int(r['taps'])}", int(r["unit_subcarriers"]), int(r["snr_db"])): float(r["ser"])
          for r in read_rows("pm_adapt.csv") if r["waterfill"] == "0"}
    g1 = {(r["channel"], int(r["snr_db"])): float(r["ser"])
          for r in read_rows("pm_figs.csv") if r["graph"] == "g1" and r["scheme"] == "adapted_B2"}
    for (cn, snr, J), (fr, e, t, ter, s) in sorted(first.items()):
        ref = []
        if snr == 10 and (cn, J) in sig:
            ref.append(("pm_sig", sig[(cn, J)]))
        B = (D // J) // 2
        if (cn, B, snr) in ad:
            ref.append(("pm_adapt", ad[(cn, B, snr)]))
        if J == 16 and (cn, snr) in g1:
            ref.append(("pm_figs_g1", g1[(cn, snr)]))
        for nm, v in ref:
            print(f"[repro] {cn} {snr} dB J={J} frames {fr}: ter {ter:.6g} {nm} {v:.6g} "
                  f"diff {ter - v:+.3g} ({(ter - v) * t:+.0f} errors)", flush=True)
    for r in rows:
        cn, snr, J, fr, ter = r[1], r[3], r[4], r[7], r[10]
        if J == 16 and (cn, snr) in hi and hi[(cn, snr)][0] == fr:
            v = hi[(cn, snr)][1]
            print(f"[repro] {cn} {snr} dB J=16 frames {fr}: ter {ter:.6g} pm_figs_hi {v:.6g} "
                  f"diff {ter - v:+.3g} ({(ter - v) * fr * U:+.0f} errors)", flush=True)


def part_adapted(a):
    ctx = setup()
    print("[semfig4a] c", ctx[4], flush=True)
    if a.smoke:
        frames, frames_hi, min_err = 2048, 8192, 100
        plan = [(10, 4, "fs4", J) for J in (16, 1)] + [(20, 8, "fs8", 32)]
    else:
        frames, frames_hi, min_err = FRAMES, FRAMES_HI, MIN_ERR
        plan = [(10, taps, cn, J) for taps, cn in CLASSES for J in JS]
        plan += [(snr, taps, cn, J) for snr in (0, 20) for taps, cn in CLASSES for J in (16, 32)]
    for snr, taps, cn, J in plan:
        assert pm.COND[cn]["taps"] == taps
    rows, first = [], {}
    for snr, taps, cn, J in plan:
        t0 = time.time()
        rows.append(point(ctx, snr, taps, cn, J, frames, frames_hi, min_err, first))
        print("[row]", " ".join(f"{v:.4e}" if isinstance(v, float) else str(v) for v in rows[-1]),
              f"({time.time() - t0:.1f} s)", flush=True)
    if a.smoke:
        # the copied loop must give the token errors of exp_pm_ext.run_units
        src, E, pd, Wn, c = ctx
        for snr, taps, cn, J in plan:
            v = px.run_units(E, pd, c, snr, taps, D // J, frames=frames)
            mine = first[(cn, snr, J)][3]
            print(f"[check] {cn} {snr} dB J={J}: run_units {v:.6g} here {mine:.6g}", flush=True)
            assert v == mine, "copied loop differs from exp_pm_ext.run_units"
    repro(rows, first)
    write_csv(pm.DATA / f"pm_semfig4_adapted{'_smoke' if a.smoke else ''}.csv", HDR, rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="adapted", choices=["adapted"])
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    {"adapted": part_adapted}[a.part](a)
    print("[done]", a.part, round((time.time() - t0) / 60, 1), "min", flush=True)


if __name__ == "__main__":
    main()
