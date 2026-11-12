"""Semantic error rate (SER) of the static masks of Figs. 3 and 4 at 10 dB.

Metrics. In every frame user u receives one token s_u and its receiver
decides s_hat_u. Every point accumulates over all frames and the U = 4 users
  errors   token errors, the number of decisions with s_hat_u != s_u
  tokens   decided tokens, frames x U
  ter      token error rate errors / tokens, the estimate of
           TER = Pr[s_hat_u != s_u]
  ser_sem  semantic error rate, the mean over all decided tokens of
           1 - zeta(s_u, s_hat_u), the estimate of
               SER = E[1 - zeta(s_u, s_hat_u)],
           zeta(v, w) = w_v^T w_w / (||w_v|| ||w_w||)
           with w_v the input token embedding of token v in BERT base uncased
           (row v of the 30,522 x 768 table; the rows of exp_pm.Source("bert").Wn
           are w_v / ||w_v||). A correct token costs 0. A decided id outside
           [0, V) is replaced by [UNK] before the cosine, as in
           exp_pm.Evaluator.score; with minimum-distance decoding over the
           whole vocabulary no such id occurs here.
Both rates are averaged over the users.

Common setup (as exp_pm_ext.py): U = 4, d = 64, BERT codebook whitened under
the source statistics (exp_pm.codebook kind "pwhite", rotation seed 0),
tokens drawn with p_v (exp_pm.Source("bert").dist(), first 40,000 AG News
training articles), all users receive the same token with probability rho
(exp_pm_ext.sample), flat Rayleigh fading with exact gains
(exp_pm.COND["flat"]), 10 dB, weighted minimum-distance decoding over all
30,522 tokens. Every point uses the 40,000 frames of exp_pm_ext.ser (chunks of
512 frames, generator seed 11 + 10 SNR = 111 on test frames, 21 + 10 SNR = 121
on validation frames), so the token errors are those of exp_pm_ext.ser on the
same masks. Training: exp_pm_ext.train_rho (3000 iterations, batch 256, SNR
fixed at 10 dB, training seed s).

Parts (--part):
  content  Fig. 3. For rho in {0, 0.25, 0.5, 0.75, 0.9, 1}: masks without
           overlap (comb pattern L = U, weight 2 on dimensions u, u + 4, ...),
           full overlap (all ones) and masks trained at 10 dB (L = 16, seed 1,
           the trained_at_snr masks of exp_pm_ext.py --part content_snr), each
           with ter and ser_sem in total and split into frames with a shared
           token (all four users receive the same token) and frames with
           independent tokens, the mean cross-energy (1/U) sum_u iota_u
           (exp_pm.cross_energy) and the constant c. The fixed masks are
           calibrated on 65,536 frames drawn with seed 0 at that rho; the
           trained masks keep the c of the last calibration in train_rho.
           Output data/pm/pm_semfig3.csv.
  static   Fig. 4. Static entries trained at 10 dB for rho in {0, 0.25, 0.5,
           0.75, 0.9, 1} and pattern length L in {1, 2, 4, 8, 16, 32, 64}
           (J = L values per mask), training seeds 1, 2, 3 (as
           exp_pm_ext.py --part sig10 and --part sig10rho), each scored on
           validation frames (seed 21, columns *_val) and test frames
           (seed 11) with ter and ser_sem, the test-frame split by frame kind,
           the mean cross-energy and c. A run whose signs cancel the frame of
           a shared token (c = 0, ZeroDivisionError in training) is kept as a
           failed run with nan values. Output data/pm/pm_semfig4_static.csv.

Trained masks (state dict of exp_pm.Masks, codebook included) go to
data/pm/pm_models/sem_fig3_rho{rho}_L16_s1.pt and
data/pm/pm_models/sem_fig4_rho{rho}_L{L}_s{s}.pt; reload with
    sd = torch.load(path); m = exp_pm.Masks(sd["E"], L, 4); m.load_state_dict(sd)
The column model names the file. With --smoke (50 iterations, 2,048 frames,
rho in {0, 1}, L in {16, 1}, seed 1) the CSVs and the masks go to
data/pm/pm_models_smoke/.

Run from the code folder on a GPU:
    python exp_pm_semfig34.py --part content     # data/pm/pm_semfig3.csv
    python exp_pm_semfig34.py --part static      # data/pm/pm_semfig4_static.csv
    python exp_pm_semfig34.py --part static --smoke
"""
from __future__ import annotations

import argparse
import time

import torch

import exp_pm as pm
import exp_pm_ext as px
import rev_channel as rc
from sse_lib import DEVICE, D, write_csv

U = px.U
SNR = 10
RHOS = (0.0, 0.25, 0.5, 0.75, 0.9, 1.0)
LS = (64, 32, 16, 8, 4, 2, 1)
SEEDS = (1, 2, 3)
TEST_SEED, VAL_SEED = 11, 21
FRAMES = 40_000
NAN = float("nan")
KEYS = ("errors", "tokens", "ter", "ser_sem")


def setup():
    src = pm.Source("bert")
    pdc = src.dist()
    E = pm.codebook(src.W, D, "pwhite", pd=pdc).to(DEVICE)
    return src, E, pdc.to(DEVICE), src.Wn.to(DEVICE)


@torch.no_grad()
def evaluate(m, pd, Wn, unk, rho, snr=SNR, frames=FRAMES, seed=TEST_SEED):
    """exp_pm_ext.ser (same seed, chunks, token draws and channel draws) with
    the semantic cost 1 - zeta(s, s_hat) accumulated next to the token
    errors, in total and split by frame kind (shared: all users receive the
    same token, as exp_pm_ext.ser_split)."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(seed + int(10 * snr))
    V = Wn.shape[0]
    acc = {"shared": [0, 0, 0.0], "indep": [0, 0, 0.0]}      # errors, tokens, semantic cost
    for k in range(0, frames, 512):
        n = min(512, frames - k)
        idx = px.sample(pd, n, m.users, rho)
        x = m.tx(idx) / m.c
        r, w = rc.channel(x, snr, m.users, **pm.COND["flat"])
        dec = m.scores(r, w).argmax(-1)
        dec = dec.masked_fill((dec < 0) | (dec >= V), unk)
        wrong = dec != idx
        cost = (1.0 - (Wn[idx] * Wn[dec]).sum(-1)).masked_fill(~wrong, 0.0)
        same = (idx == idx[:, :1]).all(1)
        for key, sel in (("shared", same), ("indep", ~same)):
            a = acc[key]
            a[0] += int(wrong[sel].sum())
            a[1] += int(sel.sum()) * m.users
            a[2] += float(cost[sel].double().sum())
    out = {}
    tot = [acc["shared"][j] + acc["indep"][j] for j in range(3)]
    for tag, (e, t, s) in (("", tot), ("_shared", acc["shared"]), ("_indep", acc["indep"])):
        out[f"errors{tag}"], out[f"tokens{tag}"] = e, t
        out[f"ter{tag}"] = e / t if t else NAN
        out[f"ser_sem{tag}"] = s / t if t else NAN
    return out


def train(E, pd, rho, L, s, iters):
    """exp_pm_ext.train_rho at 10 dB; None when the signs of the users cancel
    the frame of a shared token (c = 0)."""
    try:
        return px.train_rho(E, L, pd, rho, iters, seed=s, snr_range=(float(SNR), float(SNR)))
    except ZeroDivisionError:
        torch.backends.cuda.matmul.allow_tf32 = False
        return None


def iota(m):
    return sum(pm.cross_energy(m.masks().detach().cpu())) / U


def part_content(a, out_dir, models, tag):
    src, E, pd, Wn = setup()
    rhos = (0.0, 1.0) if a.smoke else RHOS
    comb = torch.zeros(U, D, device=DEVICE)
    for u in range(U):
        comb[u, u::4] = 2.0
    hdr = ["scheme", "snr_db", "rho", "frames"]
    for t in ("", "_shared", "_indep"):
        hdr += [f"{k}{t}" for k in KEYS]
    hdr += ["cross_energy", "c", "model"]
    rows = []
    for rho in rhos:
        ms = {}
        for name, M in (("no_overlap", comb), ("full_overlap", torch.ones(U, D, device=DEVICE))):
            m = px.FreeMasks(E, M).to(DEVICE)
            torch.manual_seed(0)                        # fixed draw for the constant c
            px.calibrate_rho(m, pd, rho)
            ms[name] = (m.eval(), "")
        m = train(E, pd, rho, 16, 1, a.iters)
        fname = f"sem_fig3_rho{rho:g}_L16_s1"
        if m is not None:
            torch.save(m.state_dict(), models / f"{fname}.pt")
            ms["trained_at_snr"] = (m, fname)
        else:
            print("[content] trained", rho, "failed: signal cancelled (c = 0)", flush=True)
        for name, (m, fn) in ms.items():
            ev = evaluate(m, pd, Wn, src.unk, rho, frames=a.frames)
            io = iota(m)
            row = [name, SNR, rho, a.frames]
            for t in ("", "_shared", "_indep"):
                row += [ev[f"{k}{t}"] for k in KEYS]
            rows.append(row + [io, float(m.c), fn])
            print("[content]", rho, name, f"ter {ev['ter']:.5f} ser_sem {ev['ser_sem']:.5f} "
                  f"shared {ev['ter_shared']:.4f}/{ev['ser_sem_shared']:.4f} "
                  f"indep {ev['ter_indep']:.4f}/{ev['ser_sem_indep']:.4f} iota {io:.4f}", flush=True)
    write_csv(out_dir / f"pm_semfig3{tag}.csv", hdr, rows)


def part_static(a, out_dir, models, tag):
    src, E, pd, Wn = setup()
    rhos, Ls, seeds = ((0.0, 1.0), (16, 1), (1,)) if a.smoke else (RHOS, LS, SEEDS)
    hdr = ["scheme", "channel", "rho", "values", "seed", "frames"] \
        + [f"{k}_val" for k in KEYS] + list(KEYS) \
        + ["ter_shared", "ser_sem_shared", "ter_indep", "ser_sem_indep", "cross_energy", "c", "model"]
    rows = []
    for rho in rhos:
        name = "static_at10dB" if rho == 0 else f"shared_rho{rho:g}_at10dB"
        for L in Ls:
            for s in seeds:
                m = train(E, pd, rho, L, s, a.iters)
                if m is None:
                    rows.append([name, "flat", rho, L, s, a.frames] + [NAN] * (len(hdr) - 7) + [""])
                    print("[static]", name, L, s, "failed: signal cancelled (c = 0)", flush=True)
                    continue
                fname = f"sem_fig4_rho{rho:g}_L{L}_s{s}"
                torch.save(m.state_dict(), models / f"{fname}.pt")
                ev_v = evaluate(m, pd, Wn, src.unk, rho, frames=a.frames, seed=VAL_SEED)
                ev = evaluate(m, pd, Wn, src.unk, rho, frames=a.frames, seed=TEST_SEED)
                io = iota(m)
                rows.append([name, "flat", rho, L, s, a.frames] + [ev_v[k] for k in KEYS]
                            + [ev[k] for k in KEYS]
                            + [ev["ter_shared"], ev["ser_sem_shared"], ev["ter_indep"], ev["ser_sem_indep"],
                               io, float(m.c), fname])
                print("[static]", name, L, s, f"val ter {ev_v['ter']:.5f} ser_sem {ev_v['ser_sem']:.5f} "
                      f"test ter {ev['ter']:.5f} ser_sem {ev['ser_sem']:.5f} iota {io:.4f}", flush=True)
    write_csv(out_dir / f"pm_semfig4_static{tag}.csv", hdr, rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["content", "static"])
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--frames", type=int, default=FRAMES)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    out_dir, models, tag = pm.DATA, pm.MODELS, ""
    if a.smoke:                                   # keep smoke outputs apart from the real ones
        a.iters, a.frames = 50, 2048
        out_dir = models = pm.DATA / "pm_models_smoke"
        models.mkdir(exist_ok=True)
        tag = "_smoke"
    t0 = time.time()
    {"content": part_content, "static": part_static}[a.part](a, out_dir, models, tag)
    print("[done]", a.part, round((time.time() - t0) / 60, 1), "min", flush=True)


if __name__ == "__main__":
    main()
