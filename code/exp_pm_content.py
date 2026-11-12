"""Shared content with the untrained comb pattern and a receiver that tests
which users received its token.

Setting of every part: flat Rayleigh fading with exact gains, U = 4 users,
d = 64 real dimensions, the BERT codebook whitened under the source
statistics (pwhite, as exp_pm_ext.part_content_snr), tokens drawn with the
source statistics p_v, weighted minimum-distance decoding over all
V = 30,522 tokens. The transmitter always uses the comb with L = U: user u
weights the real dimensions u, u + U, u + 2U, ... with sqrt(U) (= 2 at
U = 4), so ||m_u||^2 = d and the masks of all users add up to the same
weight on every dimension. When all users share a token, the comb therefore
sends that token with full overlap by itself. The comb masks do not
overlap, so the likelihood of independent tokens factorizes over the
position sets of the masks (the positions where each mask has weight).

part test  : all users share one token with probability rho, independent
             tokens otherwise (exp_pm_ext.sample). Receiver u computes, for
             every token v and the position set j of every mask, the
             log-likelihood
             S_j(v) = |g_u|^2 / (2 sigma^2) (2a r_j.(m_j e_v) - a^2 ||m_j e_v||^2),
             a = 1/c, sigma^2 = 1/(d SNR). It tests H1 (one token shared by
             all users, scored on all d dimensions with the summed masks:
             log sum_v p_v exp(sum_j S_j(v))) against H0 (independent tokens
             on the position set of its mask and on those of the other masks:
             sum_j log sum_v p_v exp(S_j(v))), with the kind prior rho
             clipped to [1e-3, 1 - 1e-3] (test_true_prior) or an assumed
             prior 0.5 (test_prior05). Under the chosen kind it decodes its
             token by weighted minimum distance on all dimensions with the
             summed masks (H1) or on the position set of its mask (H0).
             References: a genie receiver told the kind (genie) and the comb
             with its standard decoder (standard_comb, exp_pm.Masks.scores).
             Grid rho in {0, 0.25, 0.5, 0.75, 0.9, 1}, SNR {0, 10, 20} dB,
             40,000 frames per point, all receivers on the same frames.
             Output data/pm/pm_content_test.csv with columns receiver,
             snr_db, rho, ser, ser_shared, ser_indep, miss_rate,
             false_alarm_rate.
part pairs : users (0, 1) and (2, 3) each share one token with probability
             rho, independently per pair and per frame; otherwise the two
             users of a pair draw independent tokens. 10 dB,
             rho in {0.5, 0.75, 0.9, 1}. Schemes:
               standard_comb      comb, standard decoder
               test_general8      comb, receiver u tests the 2^(U-1) = 8
                                  hypotheses of which other users share its
                                  token (structure unknown; the users outside
                                  its group are taken as independent; uniform
                                  prior 1/8, no knowledge of rho)
               test_pair2         comb, receiver u tests the 2 hypotheses of
                                  its own pair (structure known), prior rho
                                  clipped to [1e-3, 1 - 1e-3]
               test_pair2_prior05 the same with the assumed prior 0.5
               genie_pair         comb, receiver told whether its pair shares
               trained_L16        one mask per user, L = 16, trained at 10 dB
                                  on the pair statistics (train_groups, the
                                  pair version of exp_pm_ext.train_rho),
                                  seeds 1, 2, 3; the run with the lowest SER
                                  on the validation frames is selected
                                  (column selected = 1)
             After each test the token is decoded by weighted minimum distance
             on the positions of the chosen group (the summed masks of its
             users). ser_shared / ser_indep split the tokens by whether the
             user's pair shares; miss_rate and false_alarm_rate give the
             fraction of shared and of independent tokens on which the
             receiver picks a wrong hypothesis (for test_pair2: misses and
             false alarms of the pair test). Outputs
             data/pm/pm_content_pairs.csv (columns scheme, snr_db, rho, seed,
             selected, ser_val, ser, ser_shared, ser_indep, miss_rate,
             false_alarm_rate, hypotheses, cross_energy_in_pairs,
             cross_energy_across_pairs) and data/pm/pm_content_pairs.json
             (cross-energy matrix iota_uk = (1/d) sum_i m_{u,i}^2 m_{k,i}^2 of
             every trained mask, per-user totals as exp_pm.cross_energy,
             stored patterns, c, and the receiver cost in hypotheses,
             exponential sums over V and correlation MACs per token).
part all   : test, then pairs.

Seeds: the constant c of the comb is calibrated as exp_pm_ext.calibrate_rho
(65,536 frames of the statistics served) after torch.manual_seed(0); test
frames and channels use torch.manual_seed(11 + int(10 snr)) and batches of
512 frames, as exp_pm_ext.ser, so part test draws the frames of
exp_pm_ext.ser_split (pm_content_snr.csv); validation frames use
21 + int(10 snr); training uses sse_lib.set_seed(s), s = 1, 2, 3, with
3,000 steps of 256 frames, Adam 1e-2, TF32 during training only. Shared and
independent tokens are split by the drawn frame kind (an independent frame
whose tokens coincide by chance counts as independent, a difference below
1e-5 of the frames from exp_pm_ext.ser_split).

Run from the code folder (GPU via sse_lib.DEVICE):
    python exp_pm_content.py --part test     # data/pm/pm_content_test.csv
    python exp_pm_content.py --part pairs    # data/pm/pm_content_pairs.{csv,json}
    python exp_pm_content.py --part test --smoke   # short check, *_smoke files
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import time

import torch

import exp_pm as pm
import rev_channel as rc
from exp_pm_ext import FreeMasks
from sse_lib import DEVICE, D, set_seed, write_csv

DATA = pm.DATA                     # data/pm/
U = 4
PAIRS = ((0, 1), (2, 3))
PRIOR_CLIP = 1e-3
BATCH = 512                        # evaluation batch of exp_pm_ext.ser


# ----------------------------------------------------------------------
# masks, samplers, calibration
# ----------------------------------------------------------------------
def comb_masks(users: int = U, d: int = D) -> torch.Tensor:
    """Untrained comb with L = U: m_u = sqrt(U) on dims u, u+U, ..."""
    M = torch.zeros(users, d, device=DEVICE)
    for u in range(users):
        M[u, u::users] = math.sqrt(users)
    return M


def draw_all(pd, n, users, rho):
    """exp_pm_ext.sample (same random draws in the same order) that also
    returns the drawn frame kind as flags (n,1): all users share."""
    common = torch.multinomial(pd, n, replacement=True)
    indep = torch.multinomial(pd, n * users, replacement=True).view(n, users)
    same = torch.rand(n, device=pd.device) < rho
    idx = torch.where(same[:, None], common[:, None].expand(n, users), indep)
    return idx, same[:, None]


def draw_groups(pd, n, users, rho, groups=PAIRS):
    """Every group shares one token with probability rho, independently per
    group and frame (the token of its first user); flags (n,G)."""
    idx = torch.multinomial(pd, n * users, replacement=True).view(n, users)
    flags = torch.stack([torch.rand(n, device=pd.device) < rho for _ in groups], 1)
    for g, grp in enumerate(groups):
        for k in grp[1:]:
            idx[:, k] = torch.where(flags[:, g], idx[:, grp[0]], idx[:, k])
    return idx, flags


@torch.no_grad()
def calibrate(m, pd, draw, rho, n=65536):
    """c^2 = mean frame energy under the statistics served
    (exp_pm_ext.calibrate_rho with the sampler as a parameter)."""
    idx, _ = draw(pd, n, m.users, rho)
    m.c.fill_(float(m.tx(idx).pow(2).sum(1).mean().sqrt()))


def clip_prior(p: float) -> float:
    return min(max(p, PRIOR_CLIP), 1.0 - PRIOR_CLIP)


def pair_cross_energy(M: torch.Tensor) -> list[list[float]]:
    """iota_uk = (1/d) sum_i m_{u,i}^2 m_{k,i}^2 (u != k, zero diagonal);
    row sums equal exp_pm.cross_energy."""
    E2 = M.pow(2)
    C = (E2[:, None, :] * E2[None, :, :]).mean(-1)
    C.fill_diagonal_(0.0)
    return [[float(v) for v in row] for row in C]


# ----------------------------------------------------------------------
# receivers
# ----------------------------------------------------------------------
class PositionScores:
    """Weighted minimum-distance metrics on the position set of every mask
    (the positions where it has weight) for masks without overlap:
      md_j(v) = 2a sum_{i in j} w_i r_i m_{j,i} e_{v,i} - a^2 sum_{i in j} w_i m_{j,i}^2 e_{v,i}^2,
    a = 1/c. md_u equals row u of exp_pm.Masks.scores; md_j / (2 sigma^2)
    is the log-likelihood of token v on the position set of mask j up to a
    constant."""

    def __init__(self, E, M, c):
        sup = [torch.nonzero(M[j]).flatten() for j in range(M.shape[0])]
        cover = torch.zeros(M.shape[1], device=M.device)
        for s in sup:
            cover[s] += 1
        assert bool((cover <= 1).all()), "position sets must be disjoint"
        self.sup = sup
        self.cand = [E[:, s] * M[j, s][None] for j, s in enumerate(sup)]     # (V,|s|)
        self.cand2 = [cj.pow(2) for cj in self.cand]
        self.a = 1.0 / c

    def md(self, r, w):
        """r, w: (n,d) of one receiver -> (n,U,V)."""
        a = self.a
        return torch.stack([2 * a * (w[:, s] * r[:, s]) @ cj.T - a * a * w[:, s] @ c2.T
                            for s, cj, c2 in zip(self.sup, self.cand, self.cand2)], 1)


def subsets(others):
    """All subsets of the other users as (bitmask, list), bit i <-> others[i]."""
    out = []
    for b in range(2 ** len(others)):
        out.append((b, [others[i] for i in range(len(others)) if (b >> i) & 1]))
    return out


@torch.no_grad()
def eval_comb(m, pd, logp, draw, groups, rho, snr, frames, seed, names):
    """All comb receivers in `names` on the same frames. Receiver u belongs
    to group groups[gi[u]]; flags[:, g] tells whether group g shares.
      standard : position set of its mask
      genie    : position sets of the group if the group shares, else that
                 of its mask
      test_*   : MAP test between 'group shares' and 'independent' with the
                 kind prior of the name (true -> clipped rho, 05 -> 0.5)
      general  : 2^(U-1) hypotheses of which other users share u's token,
                 uniform prior, users outside the hypothesized group taken
                 as independent
    Returns {name: [ser, ser_shared, ser_indep, miss_rate, false_alarm_rate]}
    and the agreement of the decoder on the position set of its mask with
    Masks.scores on the first batch."""
    torch.backends.cuda.matmul.allow_tf32 = False
    users = m.users
    M = m.masks().detach()
    ps = PositionScores(m.E, M, float(m.c))
    gi = {k: g for g, grp in enumerate(groups) for k in grp}
    sig2 = 1.0 / (D * 10 ** (snr / 10))
    s2 = 2.0 * sig2
    prior = {"test_true": clip_prior(rho), "test_05": 0.5}
    # counters: err_sh, n_sh, err_in, n_in, hyp_err_sh, hyp_err_in
    cnt = {k: [0] * 6 for k in names}
    agree = None
    torch.manual_seed(seed + int(10 * snr))
    for k0 in range(0, frames, BATCH):
        n = min(BATCH, frames - k0)
        idx, flags = draw(pd, n, users, rho)
        x = m.tx(idx) / m.c
        r, w = rc.channel(x, snr, users, **pm.COND["flat"])
        if agree is None:
            ref = m.scores(r, w).argmax(-1)                                  # (n,U)
        for u in range(users):
            md = ps.md(r[:, u], w[:, u])                                     # (n,U,V)
            lse1 = torch.logsumexp(logp + md / s2, -1)                       # (n,U)
            tru = idx[:, u]
            dec_own = md[:, u].argmax(-1)
            if agree is None and u == 0:
                agree = (int((dec_own == ref[:, 0]).sum()), n)
            if u in gi:
                grp = list(groups[gi[u]])
                share = flags[:, gi[u]]
            else:
                grp, share = [u], torch.zeros(n, dtype=torch.bool, device=DEVICE)
            md_g = md[:, grp].sum(1)
            dec_g = md_g.argmax(-1)
            llr = torch.logsumexp(logp + md_g / s2, -1) - lse1[:, grp].sum(1)
            for nm in names:
                hyp_err = None
                if nm == "standard":
                    dec = dec_own
                elif nm == "genie":
                    dec = torch.where(share, dec_g, dec_own)
                elif nm in prior:
                    p = prior[nm]
                    dsh = llr > math.log((1 - p) / p)
                    dec = torch.where(dsh, dec_g, dec_own)
                    hyp_err = dsh != share
                elif nm == "general":
                    others = [k for k in range(users) if k != u]
                    ll, dd = [], []
                    for b, S in subsets(others):
                        G = [u] + S
                        rest = [k for k in others if k not in S]
                        mg = md[:, G].sum(1)
                        ll.append(torch.logsumexp(logp + mg / s2, -1) + lse1[:, rest].sum(1))
                        dd.append(mg.argmax(-1))
                    bh = torch.stack(ll, 1).argmax(1)                         # uniform prior
                    dec = torch.stack(dd, 1).gather(1, bh[:, None])[:, 0]
                    b_true = sum(1 << others.index(k) for k in grp if k != u)
                    hyp_err = bh != torch.where(share, torch.full_like(bh, b_true),
                                                torch.zeros_like(bh))
                else:
                    raise ValueError(nm)
                err = dec != tru
                c_ = cnt[nm]
                c_[0] += int((err & share).sum()); c_[1] += int(share.sum())
                c_[2] += int((err & ~share).sum()); c_[3] += int((~share).sum())
                if hyp_err is not None:
                    c_[4] += int((hyp_err & share).sum()); c_[5] += int((hyp_err & ~share).sum())
    res = {}
    nan = float("nan")
    for nm, (e_s, n_s, e_i, n_i, h_s, h_i) in cnt.items():
        tested = nm in prior or nm == "general"
        res[nm] = [(e_s + e_i) / (n_s + n_i),
                   e_s / n_s if n_s else nan, e_i / n_i if n_i else nan,
                   h_s / n_s if (tested and n_s) else nan,
                   h_i / n_i if (tested and n_i) else nan]
    return res, agree


@torch.no_grad()
def ser_groups(m, pd, draw, groups, rho, snr, frames, seed):
    """Standard decoder (Masks.scores) on the frames of eval_comb (same seed,
    batches and draws); SER in total and split by whether the user's group
    shares."""
    torch.backends.cuda.matmul.allow_tf32 = False
    users = m.users
    gi = torch.tensor([next(g for g, grp in enumerate(groups) if k in grp) for k in range(users)],
                      device=DEVICE)
    torch.manual_seed(seed + int(10 * snr))
    e_s = n_s = e_i = n_i = 0
    for k0 in range(0, frames, BATCH):
        n = min(BATCH, frames - k0)
        idx, flags = draw(pd, n, users, rho)
        x = m.tx(idx) / m.c
        r, w = rc.channel(x, snr, users, **pm.COND["flat"])
        wrong = m.scores(r, w).argmax(-1) != idx                          # (n,U)
        sh = flags[:, gi]                                                   # (n,U)
        e_s += int((wrong & sh).sum()); n_s += int(sh.sum())
        e_i += int((wrong & ~sh).sum()); n_i += int((~sh).sum())
    nan = float("nan")
    return (e_s + e_i) / (n_s + n_i), (e_s / n_s if n_s else nan), (e_i / n_i if n_i else nan)


def train_groups(E, L, pd, rho, iters, seed=1, snr_range=(10.0, 10.0), users=U,
                 groups=PAIRS, batch=256):
    """exp_pm_ext.train_rho with the group sampler (and users as a
    parameter): Adam 1e-2, c recalibrated on 8,192 frames every step,
    cross-entropy over all V tokens, TF32 during training only."""
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(seed)
    m = pm.Masks(E, L, users).to(DEVICE)
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    ce = torch.nn.CrossEntropyLoss()
    draw = lambda p, n, uu, r: draw_groups(p, n, uu, r, groups)   # noqa: E731
    for _ in range(iters):
        idx, _ = draw(pd, batch, users, rho)
        snr = torch.empty(batch).uniform_(*snr_range)
        calibrate(m, pd, draw, rho, 8192)
        x = m.tx(idx) / m.c
        r, w = rc.channel(x, snr, users, **pm.COND["flat"])
        sc = m.scores(r, w) * m.logit_scale.exp()
        loss = ce(sc.reshape(-1, sc.shape[-1]), idx.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    torch.backends.cuda.matmul.allow_tf32 = False
    calibrate(m, pd, draw, rho)
    return m.eval()


def setup():
    src = pm.Source("bert")
    pdc = src.dist()
    E = pm.codebook(src.W, D, "pwhite", pd=pdc).to(DEVICE)
    pd = pdc.to(DEVICE)
    logp = torch.log(pd.clamp_min(1e-30))
    return E, pd, logp


# ----------------------------------------------------------------------
# parts
# ----------------------------------------------------------------------
def part_test(a):
    E, pd, logp = setup()
    rhos = (0.0, 0.25, 0.5, 0.75, 0.9, 1.0)
    snrs = (10,) if a.smoke else (0, 10, 20)
    names = {"test_true": "test_true_prior", "test_05": "test_prior05",
             "genie": "genie", "standard": "standard_comb"}
    rows = []
    for rho in rhos:
        m = FreeMasks(E, comb_masks()).to(DEVICE)
        torch.manual_seed(0)                                  # fixed draw for c
        calibrate(m, pd, draw_all, rho)
        for snr in snrs:
            t = time.time()
            res, agree = eval_comb(m, pd, logp, draw_all, (tuple(range(U)),), rho, snr,
                                   a.frames, 11, list(names))
            for nm, out in names.items():
                rows.append([out, snr, rho] + res[nm])
            print(f"[test] rho {rho} snr {snr} c {float(m.c):.4f} "
                  + " ".join(f"{names[k]} {v[0]:.4f}" for k, v in res.items())
                  + f" | miss {res['test_true'][3]:.4f} fa {res['test_true'][4]:.4f}"
                  + f" | prior0.5 miss {res['test_05'][3]:.4f} fa {res['test_05'][4]:.4f}"
                  + f" | own-decoder check {agree[0]}/{agree[1]} | {time.time() - t:.0f} s",
                  flush=True)
    write_csv(DATA / f"pm_content_test{a.suffix}.csv",
              ["receiver", "snr_db", "rho", "ser", "ser_shared", "ser_indep",
               "miss_rate", "false_alarm_rate"], rows)


def part_pairs(a):
    E, pd, logp = setup()
    snr = 10
    rhos = (0.75, 1.0) if a.smoke else (0.5, 0.75, 0.9, 1.0)
    seeds = (1, 2) if a.smoke else (1, 2, 3)
    V = E.shape[0]
    draw = lambda p, n, uu, r: draw_groups(p, n, uu, r, PAIRS)    # noqa: E731
    names = {"standard": ("standard_comb", 1), "general": ("test_general8", 2 ** (U - 1)),
             "test_true": ("test_pair2", 2), "test_05": ("test_pair2_prior05", 2),
             "genie": ("genie_pair", 1)}
    in_pair = [(i, j) for grp in PAIRS for i in grp for j in grp if i < j]
    across = [(i, j) for i in range(U) for j in range(i + 1, U) if (i, j) not in in_pair]
    nan = float("nan")
    rows = []
    info = {"groups": [list(g) for g in PAIRS], "snr_db": snr, "frames": a.frames,
            "test_seed": 11 + 10 * snr, "val_seed": 21 + 10 * snr, "iters": a.iters,
            "train_seeds": list(seeds),
            "cost_per_token": {
                "note": ("flat fading: the energy terms ||m_j e_v||^2 are precomputed, so the MACs "
                         "count the correlations r_j.(m_j e_v); exp_sums counts sums of V exponentials"),
                "standard_comb": dict(hypotheses=1, exp_sums=0, macs=V * D // U),
                "genie_pair": dict(hypotheses=1, exp_sums=0, macs=2 * V * D // U),
                "test_pair2": dict(hypotheses=2, exp_sums=3, macs=2 * V * D // U),
                "test_general8": dict(hypotheses=2 ** (U - 1), exp_sums=2 ** (U - 1) + U - 1,
                                      macs=V * D),
                "test_all_share (part test)": dict(hypotheses=2, exp_sums=U + 1, macs=V * D),
                "trained_L16": dict(hypotheses=1, exp_sums=0, macs=V * D)}}
    for rho in rhos:
        key = f"rho{rho}"
        info[key] = {}
        m = FreeMasks(E, comb_masks()).to(DEVICE)
        torch.manual_seed(0)
        calibrate(m, pd, draw, rho)
        t = time.time()
        res, agree = eval_comb(m, pd, logp, draw, PAIRS, rho, snr, a.frames, 11, list(names))
        info[key]["comb_c"] = float(m.c)
        for nm, (out, hyp) in names.items():
            rows.append([out, snr, rho, 0, "", nan] + res[nm] + [hyp, 0.0, 0.0])
        print(f"[pairs] rho {rho} c {float(m.c):.4f} "
              + " ".join(f"{names[k][0]} {v[0]:.4f}" for k, v in res.items())
              + f" | pair2 miss {res['test_true'][3]:.4f} fa {res['test_true'][4]:.4f}"
              + f" | general8 hyp-err shared {res['general'][3]:.4f} indep {res['general'][4]:.4f}"
              + f" | own-decoder check {agree[0]}/{agree[1]} | {time.time() - t:.0f} s", flush=True)
        trained = []
        for s in seeds:
            t = time.time()
            mt = train_groups(E, 16, pd, rho, a.iters, seed=s)
            v_val = ser_groups(mt, pd, draw, PAIRS, rho, snr, a.frames, 21)[0]
            tot, s_sh, s_in = ser_groups(mt, pd, draw, PAIRS, rho, snr, a.frames, 11)
            Mk = mt.masks().detach().cpu()
            C = pair_cross_energy(Mk)
            ce_in = sum(C[i][j] for i, j in in_pair) / len(in_pair)
            ce_x = max(C[i][j] for i, j in across)
            info[key][f"trained_s{s}"] = dict(
                c=float(mt.c), ser_val=v_val, ser=tot, ser_shared=s_sh, ser_indep=s_in,
                cross_energy_matrix=C, cross_energy_user=pm.cross_energy(Mk),
                pattern=[[round(float(z), 4) for z in row] for row in Mk[:, :16]])
            trained.append([s, v_val, tot, s_sh, s_in, ce_in, ce_x])
            print(f"[pairs] rho {rho} trained_L16 seed {s} val {v_val:.4f} test {tot:.4f} "
                  f"shared {s_sh:.4f} indep {s_in:.4f} | cross-energy in pairs "
                  f"{[round(C[i][j], 3) for i, j in in_pair]} across max {ce_x:.4f} | "
                  f"{time.time() - t:.0f} s", flush=True)
        best = min(trained, key=lambda z: z[1])[0]
        info[key]["selected_seed"] = best
        for s, v_val, tot, s_sh, s_in, ce_in, ce_x in trained:
            rows.append(["trained_L16", snr, rho, s, int(s == best), v_val, tot, s_sh, s_in,
                         nan, nan, 1, ce_in, ce_x])
    write_csv(DATA / f"pm_content_pairs{a.suffix}.csv",
              ["scheme", "snr_db", "rho", "seed", "selected", "ser_val", "ser", "ser_shared",
               "ser_indep", "miss_rate", "false_alarm_rate", "hypotheses",
               "cross_energy_in_pairs", "cross_energy_across_pairs"], rows)
    path = DATA / f"pm_content_pairs{a.suffix}.json"
    json.dump(info, open(path, "w"), indent=1)
    print("[json]", path)


def part_all(a):
    for f in (part_test, part_pairs):
        t = time.time()
        f(a)
        print("[part-done]", f.__name__, round((time.time() - t) / 60, 1), "min", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["test", "pairs", "all"])
    ap.add_argument("--frames", type=int, default=40000)
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--smoke", action="store_true",
                    help="2,048 frames, 50 training steps, fewer points, *_smoke outputs")
    a = ap.parse_args()
    a.suffix = ""
    if a.smoke:
        a.frames, a.iters, a.suffix = 2048, 50, "_smoke"
    t0 = time.time()
    {"test": part_test, "pairs": part_pairs, "all": part_all}[a.part](a)
    print("[done]", a.part, round((time.time() - t0) / 60, 1), "min", flush=True)


if __name__ == "__main__":
    main()
