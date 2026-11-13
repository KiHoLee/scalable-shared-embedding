"""Adapted masks with a partial overlap chosen for shared content: the points
added to Figs. 2 to 4 at the representative rho = 0.9.

Masks (as exp_pm_content_ov.py, generalized to every mask size J). The
two-level adapted mask of user u (exp_pm_ext.unit_masks on units of ud = 64/J
real dimensions; with J >= U the users take turns and the masks do not
overlap, with J < U each user sets its level on its best unit) has weight on
the fraction f = 1/min(J, U) of the dimensions. The mask with overlap t keeps
the level a on those dimensions and places b = t a, of the same sign, on all
others, with f a^2 + (1 - f) b^2 = 1, so ||m_u||^2 = d for every t. t = 0 is
the two-level mask; with J = 1 every t gives full overlap. t is one real
number of the entry and adds no bits to an update.

Normalization. c is calibrated for every (channel, J, t, rho) on 65,536 frames
(seed 0) drawn at rho with the gains of the class at 10 dB, the SNR at which
the units are chosen in the calibration. The one exception is t = 0 with
J >= U, whose masks never overlap: there c is the constant of the two-level
masks (exp_pm_content_fs.adapt_c), which the frame energy does not depend on.

Receiver: eq. (4), user u decodes its own token with its own mask, weights
|g|^2 (as exp_pm_semfig4a.run_units_sem and exp_pm_semfig2.run_adapted).

Parts (--part):
  jsweep  Figs. 3 and 4: rho = 0.9, 10 dB, fs4 and fs8, J in {1, 2, 4, 8, 16,
          32, 64}. The best t of T_GRID is chosen on validation frames (seed 41
          + int(10 snr) + 100 T + int(7 ud/2), chunks of 384, 40,000 frames).
          The test frames are those of exp_pm_semfig4a.py (seed 31 + int(10
          snr) + 100 T + int(7 ud/2), chunks of 384, 40,000 frames, 400,000
          when the best t has fewer than 100 token errors); the tokens are
          drawn at rho with the same random calls (exp_pm_content.draw_all), so
          the channel and noise draws equal those of the points of Fig. 3 at
          rho = 0. Rows for the best t and for t = 0.
  snr     Fig. 2: rho = 0.9, J = 16 (blocks of B = 2 subcarriers), fs4 and fs8,
          SNR 0 to 20 dB in steps of 2. The best t is chosen on validation
          frames (seed 41 + int(10 snr) + 100 T + 14, chunks of 384) from
          T_GRID with 40,000 frames below 12 dB and from T_FINE with 400,000
          frames at 12 and 14 dB and 2,000,000 from 16 dB. The test frames use
          the seed of exp_pm_semfig2.run_adapted (31 + int(10 snr) + 100 T +
          14) in chunks of 384, 200,000 frames below 12 dB and 2,000,000 from
          12 dB, extended to the frame count that gives about 80 token errors
          at the validation TER of the best t (at least one error assumed, at
          most MAX_FRAMES). Rows for the best t and, on the same frames when
          the test needs no more than 2,000,000 frames, for t = 0; the
          two-level masks do not change with rho (pm_content_fs.csv), so
          Fig. 2 compares the longer points with its own two-level curve.

  codedref  the coded OMA reference lines of Fig. 3: RM(2,5) in Gray 16-QAM
          with ML over the V valid ids (exp_pm_semfig2.run_coded, comb
          placement) under flat fading (one gain per user, taps = 1) at 10 dB,
          tokens drawn i.i.d. with p_v, 200,000 frames, seed 9500 + 100 + 10;
          coded OMA decodes each user on its own subcarriers, so its SER does
          not depend on rho. The 8-tap reference on the adapted blocks is the
          coded_il_adapted row of pm_semfig2.csv. Output
          data/pm/pm_content_ov_codedref.csv.

--rho sets the shared-token probability (default 0.9); with another value the
outputs carry the suffix _rho<rho> (Fig. 3 also uses --part jsweep --rho 0.5).

Outputs: data/pm/pm_content_ov_j.csv and data/pm/pm_content_ov_snr.csv
(columns channel, snr_db, rho, J, t, role (best or t0), split, frames,
tokens, errors, ter, ser_sem, ser_sem_shared, ser_sem_indep, cross_energy,
c), data/pm/pm_content_ov_fig.json (validation tables, chosen t, c, minutes).
--smoke runs fs4 at J = 16 and 10 dB on 2,048 frames and writes *_smoke files.

Run from the code folder on a GPU:
    python exp_pm_content_ovfig.py --part jsweep
    python exp_pm_content_ovfig.py --part snr
"""
from __future__ import annotations

import argparse
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
RHO = 0.9
T_GRID = (0.0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.3, 0.5, 1.0)
T_FINE = (0.0, 0.0125, 0.025, 0.0375, 0.05, 0.075)
JS = (1, 2, 4, 8, 16, 32, 64)
SNRS = tuple(range(0, 21, 2))
MAX_FRAMES = 120_000_000
CHUNK = 384
HDR = ["channel", "snr_db", "rho", "J", "t", "role", "split", "frames", "tokens", "errors", "ter",
       "ser_sem", "ser_sem_shared", "ser_sem_indep", "cross_energy", "c"]


def frac(J: int) -> float:
    """Share of the dimensions that a two-level mask weights."""
    return 1.0 / min(J, U)


def overlap_from(M2: torch.Tensor, t: float, f: float) -> torch.Tensor:
    a = math.sqrt(1.0 / (f + (1.0 - f) * t * t))
    return torch.where(M2 > 0, torch.full_like(M2, a), torch.full_like(M2, t * a))


@torch.no_grad()
def calib_c(ctx, rho, t, taps, ud, n=65536, snr=10.0):
    J = D // ud
    if t == 0.0 and J >= U:
        return cf.adapt_c(ctx)
    torch.manual_seed(0)
    sig2 = 1.0 / (D * 10 ** (snr / 10))
    tot = 0.0
    for k in range(0, n, 4096):
        m = min(4096, n - k)
        idx, _ = pc.draw_all(ctx.pd, m, U, rho)
        g = px.gains(m, taps, DEVICE)
        M2 = px.unit_masks(g, ud, sig2)
        M = M2 if t == 0.0 else overlap_from(M2, t, frac(J))
        tot += float(((ctx.E[idx] * M).sum(1)).pow(2).sum())
    return math.sqrt(tot / n)


def cross_energy(M: torch.Tensor) -> float:
    M2 = M.pow(2)
    return float((M2 * (M2.sum(1, keepdim=True) - M2)).mean(-1).mean(1).sum())


@torch.no_grad()
def run(ctx, rho, snr, taps, ud, frames, seed, ts, cs):
    """All t of ts on the same frames; returns {t: (Acc, mean cross-energy)}."""
    torch.backends.cuda.matmul.allow_tf32 = False
    E = ctx.E
    E2 = E * E
    K = D // 2
    J = D // ud
    sig = math.sqrt(1.0 / (D * 10 ** (snr / 10)))
    accs = {t: cf.Acc(ctx) for t in ts}
    ce = {t: 0.0 for t in ts}
    torch.manual_seed(seed)
    for k0 in range(0, frames, CHUNK):
        n = min(CHUNK, frames - k0)
        idx, flags = pc.draw_all(ctx.pd, n, U, rho)
        g = px.gains(n, taps, DEVICE)
        M2 = px.unit_masks(g, ud, sig * sig)
        w = torch.complex(torch.randn(n, U, K, device=DEVICE), torch.randn(n, U, K, device=DEVICE)) * sig
        wt = (g.abs() ** 2).repeat_interleave(2, dim=-1)
        for t in ts:
            M = M2 if t == 0.0 else overlap_from(M2, t, frac(J))
            c = cs[t]
            x = (E[idx] * M).sum(1) / c
            xc = torch.complex(x[:, 0::2], x[:, 1::2])
            q = (g * xc[:, None, :] + w) / g
            r = torch.stack([q.real, q.imag], -1).reshape(n, U, D)
            a_ = 1.0 / c
            sc = 2 * a_ * torch.einsum("nud,vd->nuv", wt * r * M, E) \
                - a_ * a_ * torch.einsum("nud,vd->nuv", wt * M * M, E2)
            accs[t].add(idx, sc.argmax(-1), flags)
            ce[t] += cross_energy(M)                  # summed over the frames of the chunk
            del sc
    return {t: (accs[t], ce[t] / frames) for t in ts}


def row(cn, snr, J, t, role, split, acc, ce, c):
    r = acc.row(cn, "ov", snr, RHO, ce)
    return [cn, snr, RHO, J, t, role, split, r[4], r[5], r[6], r[7], r[8], r[9], r[10], ce, c]


def choose(res):
    """t with the lowest validation SER (ties: the smaller t)."""
    return min(sorted(res), key=lambda t: res[t][0].row("", "", 0, 0, 0)[8])


def part_jsweep(ctx, a, js, t0):
    rows, jn = [], {}
    path = pm.DATA / f"pm_content_ov_j{a.tag}.csv"
    for cn, taps in a.chans:
        for J in a.js:
            ud = D // J
            snr = 10
            ts = (0.0,) if J == 1 else a.tgrid           # every t is full overlap with J = 1
            cs = {t: calib_c(ctx, RHO, t, taps, ud) for t in ts}
            val = run(ctx, RHO, snr, taps, ud, a.frames, 41 + int(10 * snr) + 100 * taps + int(7 * ud / 2), ts, cs)
            tb = choose(val)
            for t in ts:
                rows.append(row(cn, snr, J, t, "val", "val", val[t][0], val[t][1], cs[t]))
            seed = 31 + int(10 * snr) + 100 * taps + int(7 * ud / 2)
            fr = a.frames
            test = run(ctx, RHO, snr, taps, ud, fr, seed, sorted({0.0, tb}), cs)
            if test[tb][0].row("", "", 0, 0, 0)[6] < 100 and not a.smoke:
                fr = 400_000
                test = run(ctx, RHO, snr, taps, ud, fr, seed, sorted({0.0, tb}), cs)
            for t in sorted({0.0, tb}):
                role = "best" if t == tb else "t0"
                rows.append(row(cn, snr, J, t, role, "test", test[t][0], test[t][1], cs[t]))
                if t == tb and tb == 0.0:
                    rows.append(row(cn, snr, J, t, "t0", "test", test[t][0], test[t][1], cs[t]))
            jn[f"{cn}|J{J}"] = dict(t=tb, c={f"{t:g}": cs[t] for t in ts}, test_frames=fr,
                                    val={f"{t:g}": val[t][0].row("", "", 0, 0, 0)[8] for t in ts})
            b, z = test[tb][0].row("", "", 0, 0, 0), test[0.0][0].row("", "", 0, 0, 0)
            print(f"[jsweep] {cn} J={J} t {tb:g} ser {b[8]:.5f} (errors {b[6]}) vs t=0 {z[8]:.5f} "
                  f"(errors {z[6]}) frames {fr} | {time.time() - t0:.0f}s", flush=True)
            write_csv(path, HDR, rows)
    return jn


def part_snr(ctx, a, t0):
    rows, jn = [], {}
    path = pm.DATA / f"pm_content_ov_snr{a.tag}.csv"
    J, ud = 16, 4
    for cn, taps in a.chans:
        for snr in a.snrs:
            ts = a.tgrid if snr < 12 else T_FINE
            vfr = a.frames if snr < 12 else (400_000 if snr < 16 else 2_000_000)
            if a.smoke:
                vfr = a.frames
            cs = {t: calib_c(ctx, RHO, t, taps, ud) for t in ts}
            val = run(ctx, RHO, snr, taps, ud, vfr, 41 + int(10 * snr) + 100 * taps + int(7 * ud / 2), ts, cs)
            tb = choose(val)
            for t in ts:
                rows.append(row(cn, snr, J, t, "val", "val", val[t][0], val[t][1], cs[t]))
            vb = val[tb][0].row("", "", 0, 0, 0)
            ter_est = max(vb[6], 1) / vb[5]                        # at least one error, conservative
            base = 200_000 if snr < 12 else 2_000_000
            fr = base if a.smoke else int(min(MAX_FRAMES, max(base, math.ceil(80 / (U * ter_est)))))
            if a.smoke:
                fr = a.frames
            seed = 31 + int(10 * snr) + 100 * taps + 14
            tl = sorted({0.0, tb}) if fr <= 2_000_000 else [tb]
            test = run(ctx, RHO, snr, taps, ud, fr, seed, tl, cs)
            for t in tl:
                role = "best" if t == tb else "t0"
                rows.append(row(cn, snr, J, t, role, "test", test[t][0], test[t][1], cs[t]))
                if t == tb and tb == 0.0:
                    rows.append(row(cn, snr, J, t, "t0", "test", test[t][0], test[t][1], cs[t]))
            jn[f"{cn}|{snr}"] = dict(t=tb, val_frames=vfr, test_frames=fr, c={f"{t:g}": cs[t] for t in ts},
                                     val={f"{t:g}": val[t][0].row("", "", 0, 0, 0)[8] for t in ts})
            b = test[tb][0].row("", "", 0, 0, 0)
            z = test[0.0][0].row("", "", 0, 0, 0) if 0.0 in test else [None] * 9
            print(f"[snr] {cn} {snr} dB t {tb:g} ser {b[8]:.3e} (errors {b[6]}) vs t=0 {z[8]} "
                  f"(errors {z[6]}) frames {fr} | {time.time() - t0:.0f}s", flush=True)
            write_csv(path, HDR, rows)
    return jn


def part_codedref(a, t0):
    import exp_pm_semfig2 as sf
    ctx = sf.Ctx()
    A, Mml, _ = sf.coded_table(ctx)
    fr = 2048 if a.smoke else 200_000
    acc, _ = sf.run_coded(ctx, A, Mml, "comb", 1, 10, fr, fr)
    row = acc.row("flat", "coded_il_comb", 10)
    print(f"[codedref] flat 10 dB frames {row[3]} errors {row[5]} ter {row[6]:.5f} ser_sem {row[7]:.5f} "
          f"| {time.time() - t0:.0f}s", flush=True)
    if not a.smoke:
        write_csv(pm.DATA / "pm_content_ov_codedref.csv", sf.HDR, [row])
    print("[done] codedref", round((time.time() - t0) / 60, 1), "min", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["jsweep", "snr", "codedref"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--rho", type=float, default=0.9)
    a = ap.parse_args()
    global RHO
    RHO = a.rho
    t0 = time.time()
    a.tag = ("" if RHO == 0.9 else f"_rho{RHO:g}") + ("_smoke" if a.smoke else "")
    a.chans = cf.CHANNELS[:1] if a.smoke else cf.CHANNELS
    a.js = (16,) if a.smoke else JS
    a.snrs = (10,) if a.smoke else SNRS
    a.tgrid = (0.0, 0.05) if a.smoke else T_GRID
    a.frames = 2048 if a.smoke else 40_000
    if a.part == "codedref":
        part_codedref(a, t0)
        return
    ctx = cf.Ctx()
    print(f"[setup] V {ctx.V} device {DEVICE} t={time.time() - t0:.0f}s", flush=True)
    jpath = pm.DATA / f"pm_content_ov_fig{'_smoke' if a.smoke else ''}.json"
    js = json.load(open(jpath)) if jpath.exists() else {}
    key = a.part if RHO == 0.9 else f"{a.part}_rho{RHO:g}"
    js[key] = part_jsweep(ctx, a, None, t0) if a.part == "jsweep" else part_snr(ctx, a, t0)
    js[f"minutes_{key}"] = round((time.time() - t0) / 60, 1)
    js["setup" if RHO == 0.9 else f"setup_rho{RHO:g}"] = dict(rho=RHO, t_grid=list(T_GRID), t_fine=list(T_FINE), js=list(JS), snrs=list(SNRS),
                       chunk=CHUNK, max_frames=MAX_FRAMES, receiver="eq. (4)")
    with open(jpath, "w") as f:
        json.dump(js, f, indent=1)
    print("[done]", a.part, round((time.time() - t0) / 60, 1), "min", flush=True)


if __name__ == "__main__":
    main()
