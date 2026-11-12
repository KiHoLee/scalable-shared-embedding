"""Shared content under frequency-selective fading: masks adapted to the
channel, the untrained comb and static entries trained for the content,
with the standard receiver and with a receiver that tests the frame kind.

Question. The SER-versus-J figure of the manuscript (fig4_pm.pdf, data
pm_semfig4_static.csv and pm_semfig4_adapted.csv) sweeps the probability rho
of shared content only for the static entries; its adapted masks were
evaluated on independent content (rho = 0) alone. This script measures what
shared content does to masks adapted to the channel.

Setting (as exp_pm_ext.py, exp_pm_content.py and exp_pm_semfig2.py): U = 4
users, d = 64 real dimensions = K = 32 subcarriers, the BERT input-embedding
table whitened under the source statistics (exp_pm.codebook "pwhite"),
tokens drawn with the source statistics p_v; with probability rho every user
receives the same token in a frame, otherwise independent tokens
(exp_pm_ext.sample, drawn here by exp_pm_content.draw_all, which makes the
same random draws and also returns the drawn frame kind). Channel classes
fs4 and fs8 of exp_pm.COND (4 and 8 equal-power taps, exact gains at the
transmitter and the receivers), weighted minimum-distance decoding over all
V = 30,522 tokens with the per-dimension weights |g|^2 of eq. (4).

Metrics of every point (the users' tokens of all frames):
  errors, ter     token errors on the raw decoded ids, TER = errors / tokens
  ser_sem         semantic error rate, mean over tokens of
                  1 - zeta(s, s_hat), zeta the cosine of the BERT input
                  embeddings (exp_pm.Evaluator rule: ids outside [0, V) map to
                  [UNK]; a correct token costs 0)
  ser_sem_shared  the same over the tokens of frames drawn as shared
  ser_sem_indep   ... and of frames drawn as independent (a frame of
                  independent tokens that coincide by chance counts as
                  independent)
  cross_energy    mean cross-energy (1/U) sum_u iota_u (exp_pm.cross_energy);
                  for the adapted masks averaged over the frames
  kind_miss       receivers that test the kind: fraction of the tokens of
                  shared frames on which the receiver decides "independent"
  kind_false_alarm  fraction of the tokens of independent frames on which it
                  decides "shared"

Schemes (column scheme):
  adapted_std     masks adapted to the known gains on blocks of B = 2
                  subcarriers (exp_pm_ext.adaptive_masks, no water-filling,
                  formed per frame, two weight levels, sqrt(U) on J/U blocks
                  and zero elsewhere); every receiver decodes its token with
                  its own mask (exp_pm_ext.run_adapt). c from 65,536 frames
                  drawn with seed 0 on the comb
                  (exp_pm_semfig2.part_adapted). The masks of a frame do not
                  overlap, so the mean frame energy does not depend on rho; a
                  shared token is sent as e_s . sum_u m_u / c, i.e. on every
                  dimension with weight sqrt(U).
  adapted_test    the same frames. The user index of each block is broadcast,
                  so receiver u knows the block set S_j on which the mask of
                  every user j has weight. With
                    md_j(v) = 2a sum_{i in S_j} w_i r_i m_{j,i} e_{v,i}
                              - a^2 sum_{i in S_j} w_i m_{j,i}^2 e_{v,i}^2,
                  a = 1/c, w_i = |g_{u,i}|^2 (row j = u is the standard
                  metric), it tests H1 (one token shared by all users on all
                  blocks, log sum_v p_v exp(sum_j md_j(v) / (2 sigma^2)))
                  against H0 (independent tokens, sum_j log sum_v p_v
                  exp(md_j(v) / (2 sigma^2))) with the kind prior rho clipped
                  to [1e-3, 1 - 1e-3], and decodes its token by weighted
                  minimum distance on all blocks with the summed masks (H1)
                  or on the blocks that its mask weights (H0).
  adapted_genie   the same receiver told the frame kind (one kind bit per
                  frame from the transmitter, which knows it).
  comb_std        the untrained comb with L = U (weight sqrt(U) on the
                  dimensions u, u + U, ...; the no-overlap pattern of
                  pm_semfig3.csv), standard decoder exp_pm.Masks.scores; c
                  calibrated at rho (exp_pm_content.calibrate, seed 0).
  comb_test       the frame-kind test receiver of exp_pm_content.eval_comb
                  (receiver test_true_prior) with the per-subcarrier gains of
                  the class; H0 decodes on the position set of its mask
                  (exp_pm_content.PositionScores).
  comb_genie      the comb receiver told the frame kind.
  static_trained  one mask per user, L = 16, trained at 10 dB for the
                  channel class and the rho (exp_pm_ext.train_rho with the
                  channel of the class instead of flat fading: COND[fs4] or
                  COND[fs8], snr_range = (10, 10), seed 1, 3,000 steps of 256
                  frames, Adam 1e-2, TF32 during training only), standard
                  decoder; saved as
                  data/pm/pm_models/sem_fs_<channel>_rho<rho>_L16_s1.pt
                  (reload: sd = torch.load(p); m = exp_pm.Masks(sd["E"], 16, 4);
                  m.load_state_dict(sd)).
  static_class_L16  the saved static entry of Fig. 2, trained for the class
                  on 0-20 dB for independent content (white_U4_L16_<ch>_s1),
                  used unchanged (its c from independent content; the frame
                  energy at every rho is in the JSON), standard decoder.
  full_overlap    all-ones masks, c calibrated at rho (65,536 frames, seed
                  0, exp_pm_ext.calibrate_rho), standard decoder.

Grid (part main): fs4 and fs8, rho in {0, 0.25, 0.5, 0.75, 0.9, 1}, 10 dB
for every scheme and 20 dB for adapted_* and full_overlap, 40,000 frames per
point, frames and channels from torch.manual_seed(11 + int(10 snr)) in
batches of 512 frames (the seed of exp_pm_ext.ser). The receivers of one
scheme family (adapted_*, comb_*) decode the same frames.

Part hi: adapted_* at 20 dB with 400,000 frames (same seed and batches, a
batch boundary at 40,000 frames, so the first 40,000 frames are those of the
main point; the errors there are stored as a check), since at 20 dB the
40,000-frame points hold only a few token errors.

Part check (reproduction of released points):
  flat comb       comb_std / comb_test / comb_genie under flat fading at
                  10 dB against pm_content_test.csv (standard_comb,
                  test_true_prior, genie: token errors) and comb_std against
                  the no_overlap rows of pm_semfig3.csv (errors and ser_sem)
  flat overlap    full_overlap under flat fading at 10 dB against the
                  full_overlap rows of pm_semfig3.csv
  trainer         the generalized trainer with COND["flat"], rho = 0.9,
                  against the saved sem_fig3_rho0.9_L16_s1.pt (mask values,
                  c) and its pm_semfig3.csv row
  adapted rho=0   adapted_std with the seeds and batches of
                  exp_pm_semfig2.run_adapted (seed 31 + int(10 snr) + 100 T +
                  14, batches of 384; 40,000 frames at 10 dB, 400,000 at
                  20 dB) against the checkpoint errors of
                  pm_semfig2_check.csv; at 10 dB also adapted_test with
                  rho = 0
  static rho=0    static_class_L16 at 10 dB, 40,000 frames, seed 111, against
                  the checkpoint errors of pm_semfig2_check.csv
  (comb_std under the classes fs4/fs8 must not change with rho, since the comb
  does not overlap: see the main CSV.)

Outputs: data/pm/pm_content_fs.csv (columns channel, scheme, snr_db, rho,
frames, tokens, errors, ter, ser_sem, ser_sem_shared, ser_sem_indep,
cross_energy, kind_miss, kind_false_alarm), data/pm/pm_content_fs_hi.csv
(same columns), data/pm/pm_content_fs.json (sections: setup, training =
training logs of static_trained, points = per-point TER split by kind, c,
seconds and decoder agreement, energy = measured mean frame energy of every
scheme at every rho, checks, hi_checkpoint, minutes). With --smoke (2,048
frames, 50 steps, rho in {0, 0.9}, fs4) everything goes to *_smoke files and
data/pm/pm_models_smoke/.

Run from the code folder on a GPU:
    python exp_pm_content_fs.py --part main
    python exp_pm_content_fs.py --part check
    python exp_pm_content_fs.py --part hi
    python exp_pm_content_fs.py --part all --smoke
"""
from __future__ import annotations

import argparse
import copy
import csv
import fcntl
import json
import math
import tempfile
import time
from pathlib import Path

import torch

import exp_pm as pm
import exp_pm_content as pc
import exp_pm_ext as px
import rev_channel as rc
from sse_lib import DEVICE, D, set_seed, write_csv

U = 4
B = 2                                   # subcarriers per block of the adapted masks
L_STATIC = 16
RHOS = (0.0, 0.25, 0.5, 0.75, 0.9, 1.0)
CHANNELS = (("fs4", 4), ("fs8", 8))
TEST_SEED = 11
BATCH = 512
FRAMES = 40_000
FRAMES_HI = 400_000
TRAIN_SNR = 10.0
HDR = ["channel", "scheme", "snr_db", "rho", "frames", "tokens", "errors", "ter", "ser_sem",
       "ser_sem_shared", "ser_sem_indep", "cross_energy", "kind_miss", "kind_false_alarm"]
ORDER = ("adapted_std", "adapted_test", "adapted_genie", "comb_std", "comb_test", "comb_genie",
         "static_trained", "static_class_L16", "full_overlap")
NAN = float("nan")


# ----------------------------------------------------------------------
# context, accumulator, chunking, json
# ----------------------------------------------------------------------
class Ctx:
    def __init__(self):
        self.src = pm.Source("bert")
        self.pdc = self.src.dist()                         # p_v on the CPU
        self.pd = self.pdc.to(DEVICE)
        self.logp = torch.log(self.pd.clamp_min(1e-30))    # as exp_pm_content.setup
        self.E = pm.codebook(self.src.W, D, "pwhite", pd=self.pdc).to(DEVICE)
        self.Wn = self.src.Wn.to(DEVICE)                   # normalized BERT input embeddings
        self.V, self.unk = self.src.V, self.src.unk


class Acc:
    """Token errors and semantic cost of one point, split by the drawn frame
    kind (index 0 shared, 1 independent), and wrong kind decisions."""

    def __init__(self, ctx):
        self.Wn, self.V, self.unk = ctx.Wn, ctx.V, ctx.unk
        self.frames = 0
        self.tok = [0, 0]
        self.err = [0, 0]
        self.sem = [0.0, 0.0]
        self.hyp = [0, 0]
        self.tested = False

    def copy(self):
        c = copy.copy(self)
        c.tok, c.err, c.sem, c.hyp = list(self.tok), list(self.err), list(self.sem), list(self.hyp)
        return c

    def add(self, idx, dec, flags, dsh=None):
        """idx, dec (n,U) sent and decoded ids; flags (n,1) frame drawn as
        shared; dsh (n,U) receiver decided 'shared' (tested receivers)."""
        n, users = idx.shape
        self.frames += n
        sh = flags.expand(n, users)
        wrong = dec != idx
        d2 = torch.where((dec < 0) | (dec >= self.V), torch.full_like(dec, self.unk), dec)
        ws = d2 != idx
        cost = torch.zeros(n, users, dtype=torch.float64, device=idx.device)
        if bool(ws.any()):
            cost[ws] = (1.0 - (self.Wn[idx[ws]] * self.Wn[d2[ws]]).sum(-1)).double()
        for k, sel in enumerate((sh, ~sh)):
            self.tok[k] += int(sel.sum())
            self.err[k] += int((wrong & sel).sum())
            self.sem[k] += float(cost[sel].sum())
            if dsh is not None:
                self.hyp[k] += int(((dsh != sh) & sel).sum())
        if dsh is not None:
            self.tested = True

    def row(self, cn, scheme, snr, rho, ce):
        tok, err, sem = sum(self.tok), sum(self.err), sum(self.sem)

        def rate(x, t):
            return x / t if t else NAN
        return [cn, scheme, snr, rho, self.frames, tok, err, err / tok, sem / tok,
                rate(self.sem[0], self.tok[0]), rate(self.sem[1], self.tok[1]), ce,
                rate(self.hyp[0], self.tok[0]) if self.tested else NAN,
                rate(self.hyp[1], self.tok[1]) if self.tested else NAN]

    def split_ter(self):
        return [self.err[0] / self.tok[0] if self.tok[0] else NAN,
                self.err[1] / self.tok[1] if self.tok[1] else NAN]


def chunks(frames, size, ckpt=None):
    """Batch sizes of a run of `frames` frames with a batch boundary at
    `ckpt`; yields (n, True when the checkpoint is reached)."""
    k = 0
    while k < frames:
        n = min(size, frames - k)
        if ckpt is not None and k < ckpt:
            n = min(n, ckpt - k)
        k += n
        yield n, (ckpt is not None and k == ckpt)


def update_json(path: Path, key: str, value) -> None:
    """Read-modify-write of one section under a file lock (parts may run as
    separate processes)."""
    with open(Path(tempfile.gettempdir()) / f"{path.name}.lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        d = json.load(open(path)) if path.exists() else {}
        d[key] = value
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "w") as f:
            json.dump(d, f, indent=1)
        tmp.replace(path)
        fcntl.flock(lk, fcntl.LOCK_UN)


def mean_iota(M: torch.Tensor) -> float:
    return sum(pm.cross_energy(M.detach().cpu())) / M.shape[0]


# ----------------------------------------------------------------------
# masks
# ----------------------------------------------------------------------
@torch.no_grad()
def adapt_c(ctx) -> float:
    """c of the adapted masks (exp_pm_semfig2.part_adapted)."""
    comb = torch.zeros(U, D, device=DEVICE)
    for u in range(U):
        comb[u, u::4] = 2.0
    torch.manual_seed(0)
    idx0 = px.sample(ctx.pd, 65536, U, 0.0)
    return float(((ctx.E[idx0] * comb[None]).sum(1)).pow(2).sum(1).mean().sqrt())


@torch.no_grad()
def comb_model(ctx, rho):
    m = px.FreeMasks(ctx.E, pc.comb_masks()).to(DEVICE)
    torch.manual_seed(0)                                  # fixed draw for c (exp_pm_content.part_test)
    pc.calibrate(m, ctx.pd, pc.draw_all, rho)
    return m.eval()


@torch.no_grad()
def overlap_model(ctx, rho):
    m = px.FreeMasks(ctx.E, torch.ones(U, D, device=DEVICE)).to(DEVICE)
    torch.manual_seed(0)                                  # fixed draw for c (exp_pm_semfig34.part_content)
    px.calibrate_rho(m, ctx.pd, rho)
    return m.eval()


def load_saved(name, L=16):
    """exp_pm_chanhi.load."""
    sd = torch.load(pm.MODELS / f"{name}.pt", map_location="cpu")
    m = pm.Masks(sd["E"], L, U)
    m.load_state_dict(sd)
    return m.to(DEVICE).eval()


def train_cond(E, L, pd, rho, iters, cond, seed=1, snr_range=(TRAIN_SNR, TRAIN_SNR), log_every=500):
    """exp_pm_ext.train_rho with the channel class as a parameter (and the
    loss logged every log_every steps, which draws no random numbers).
    Returns (masks or None if the signs cancel a shared frame, log, seconds)."""
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(seed)
    m = pm.Masks(E, L, U).to(DEVICE)
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    ce = torch.nn.CrossEntropyLoss()
    log = []
    t0 = time.time()
    try:
        for it in range(iters):
            idx = px.sample(pd, 256, U, rho)
            snr = torch.empty(256).uniform_(*snr_range)
            px.calibrate_rho(m, pd, rho, 8192)
            x = m.tx(idx) / m.c
            r, w = rc.channel(x, snr, U, **cond)
            sc = m.scores(r, w) * m.logit_scale.exp()
            loss = ce(sc.reshape(-1, sc.shape[-1]), idx.reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
            if it == 0 or (it + 1) % log_every == 0:
                log.append([it + 1, round(float(loss.detach()), 5)])
    except ZeroDivisionError:
        torch.backends.cuda.matmul.allow_tf32 = False
        return None, log, time.time() - t0
    torch.backends.cuda.matmul.allow_tf32 = False
    px.calibrate_rho(m, pd, rho)
    return m.eval(), log, time.time() - t0


# ----------------------------------------------------------------------
# runners
# ----------------------------------------------------------------------
@torch.no_grad()
def run_std(ctx, m, rho, snr, cond, frames, seed, chunk=BATCH):
    """Standard decoder exp_pm.Masks.scores (exp_pm_ext.ser with the channel
    of the class), frames from torch.manual_seed(seed)."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(seed)
    acc = Acc(ctx)
    for n, _ in chunks(frames, chunk):
        idx, flags = pc.draw_all(ctx.pd, n, U, rho)
        x = m.tx(idx) / m.c
        r, w = rc.channel(x, snr, U, **cond)
        acc.add(idx, m.scores(r, w).argmax(-1), flags)
    return acc


@torch.no_grad()
def run_comb(ctx, m, rho, snr, cond, frames, seed, chunk=BATCH):
    """exp_pm_content.eval_comb for one group of all users (receivers
    standard, test_true, genie) with the channel of the class; comb_std uses
    exp_pm.Masks.scores, and its agreement with the decoder on the position
    set of each mask is counted on every token."""
    torch.backends.cuda.matmul.allow_tf32 = False
    users = m.users
    M = m.masks().detach()
    ps = pc.PositionScores(m.E, M, float(m.c))
    grp = list(range(users))
    sig2 = 1.0 / (D * 10 ** (snr / 10))
    s2 = 2.0 * sig2
    p = pc.clip_prior(rho)
    thr = math.log((1 - p) / p)
    accs = {k: Acc(ctx) for k in ("comb_std", "comb_test", "comb_genie")}
    agree = [0, 0]
    torch.manual_seed(seed)
    for n, _ in chunks(frames, chunk):
        idx, flags = pc.draw_all(ctx.pd, n, users, rho)
        x = m.tx(idx) / m.c
        r, w = rc.channel(x, snr, users, **cond)
        dec_std = m.scores(r, w).argmax(-1)
        share = flags[:, 0]
        dt = torch.empty_like(idx)
        dg = torch.empty_like(idx)
        dsh_all = torch.zeros_like(idx, dtype=torch.bool)
        for u in range(users):
            md = ps.md(r[:, u], w[:, u])                                     # (n,U,V)
            lse1 = torch.logsumexp(ctx.logp + md / s2, -1)                   # (n,U)
            dec_own = md[:, u].argmax(-1)
            md_g = md[:, grp].sum(1)
            dec_g = md_g.argmax(-1)
            llr = torch.logsumexp(ctx.logp + md_g / s2, -1) - lse1[:, grp].sum(1)
            dsh = llr > thr
            dt[:, u] = torch.where(dsh, dec_g, dec_own)
            dg[:, u] = torch.where(share, dec_g, dec_own)
            dsh_all[:, u] = dsh
            agree[0] += int((dec_own == dec_std[:, u]).sum())
            agree[1] += n
        accs["comb_std"].add(idx, dec_std, flags)
        accs["comb_test"].add(idx, dt, flags, dsh_all)
        accs["comb_genie"].add(idx, dg, flags)
    return accs, agree


@torch.no_grad()
def run_adapted(ctx, c, rho, snr, taps, frames, seed, chunk=BATCH, tested=True, ckpt=None):
    """exp_pm_ext.run_adapt (B = 2, no water-filling) with shared content,
    the standard receiver and, with tested, the frame-kind test receiver and
    the genie on the same frames. Returns (accs, accs at ckpt, agreement of
    the own row of the test metrics with the standard metric, mean
    cross-energy)."""
    torch.backends.cuda.matmul.allow_tf32 = False
    E = ctx.E
    E2 = E * E
    K = D // 2
    sig = math.sqrt(1.0 / (D * 10 ** (snr / 10)))
    s2 = 2.0 * sig * sig
    p = pc.clip_prior(rho)
    thr = math.log((1 - p) / p)
    a_ = 1.0 / c
    names = ("adapted_std", "adapted_test", "adapted_genie") if tested else ("adapted_std",)
    accs = {k: Acc(ctx) for k in names}
    snap, agree, ce = None, [0, 0], 0.0
    torch.manual_seed(seed)
    for n, at in chunks(frames, chunk, ckpt):
        idx, flags = pc.draw_all(ctx.pd, n, U, rho)
        g = px.gains(n, taps, DEVICE)
        M = px.adaptive_masks(g, B, sig * sig, c, False)                   # (n,U,D)
        x = (E[idx] * M).sum(1) / c
        xc = torch.complex(x[:, 0::2], x[:, 1::2])
        w = torch.complex(torch.randn(n, U, K, device=DEVICE), torch.randn(n, U, K, device=DEVICE)) * sig
        q = (g * xc[:, None, :] + w) / g
        r = torch.stack([q.real, q.imag], -1).reshape(n, U, D)
        wt = (g.abs() ** 2).repeat_interleave(2, dim=-1)
        sc = 2 * a_ * torch.einsum("nud,vd->nuv", wt * r * M, E) \
            - a_ * a_ * torch.einsum("nud,vd->nuv", wt * M * M, E * E)
        dec_std = sc.argmax(-1)
        del sc
        M2 = M.pow(2)
        ce += float((M2 * (M2.sum(1, keepdim=True) - M2)).mean(-1).mean(1).sum())
        accs["adapted_std"].add(idx, dec_std, flags)
        if tested:
            share = flags[:, 0]
            dt = torch.empty_like(idx)
            dg = torch.empty_like(idx)
            dsh_all = torch.zeros_like(idx, dtype=torch.bool)
            for u in range(U):
                wu = wt[:, u][:, None, :]                                     # (n,1,D)
                md = 2 * a_ * torch.einsum("njd,vd->njv", wu * r[:, u][:, None, :] * M, E) \
                    - a_ * a_ * torch.einsum("njd,vd->njv", wu * M * M, E2)    # (n,U,V)
                lse1 = torch.logsumexp(ctx.logp + md / s2, -1)               # (n,U)
                md_g = md.sum(1)                                             # all blocks, summed masks
                dec_g = md_g.argmax(-1)
                llr = torch.logsumexp(ctx.logp + md_g / s2, -1) - lse1.sum(1)
                dsh = llr > thr
                own = dec_std[:, u]
                agree[0] += int((md[:, u].argmax(-1) == own).sum())
                agree[1] += n
                dt[:, u] = torch.where(dsh, dec_g, own)
                dg[:, u] = torch.where(share, dec_g, own)
                dsh_all[:, u] = dsh
                del md
            accs["adapted_test"].add(idx, dt, flags, dsh_all)
            accs["adapted_genie"].add(idx, dg, flags)
        if at:
            snap = {k: v.copy() for k, v in accs.items()}
    return accs, snap, agree, ce / frames


@torch.no_grad()
def energy(ctx, rho, m=None, c=None, taps=None, n=65536, seed=3):
    """Measured mean frame energy at rho on n frames (seed 3): fixed masks m,
    or adapted masks with constant c on gains of `taps` taps at 10 dB."""
    torch.manual_seed(seed)
    idx, _ = pc.draw_all(ctx.pd, n, U, rho)
    if m is not None:
        return float((m.tx(idx) / m.c).pow(2).sum(1).mean())
    sig2 = 1.0 / (D * 10 ** (TRAIN_SNR / 10))
    tot = 0.0
    for k in range(0, n, 8192):
        g = px.gains(min(8192, n - k), taps, DEVICE)
        M = px.adaptive_masks(g, B, sig2, c, False)
        tot += float(((ctx.E[idx[k:k + 8192]] * M).sum(1) / c).pow(2).sum())
    return tot / n


# ----------------------------------------------------------------------
# parts
# ----------------------------------------------------------------------
class Out:
    def __init__(self, a, csv_name):
        self.a = a
        self.path = pm.DATA / f"{csv_name}{a.tag}.csv"
        self.rows = []
        self.points = {}

    def emit(self, cn, scheme, snr, rho, acc, ce, c, secs, agree=None):
        r = acc.row(cn, scheme, snr, rho, ce)
        self.rows.append(r)
        ts, ti = acc.split_ter()
        info = dict(ter_shared=ts, ter_indep=ti, c=c, seconds=round(secs, 1))
        if agree is not None:
            info["own_decoder_agreement"] = agree
        self.points[f"{cn}|{scheme}|{snr}|{rho:g}"] = info
        print(f"[{scheme}] {cn} {snr} dB rho {rho:g} frames {r[4]} errors {r[6]} ter {r[7]:.5f} "
              f"ser {r[8]:.5f} shared {r[9]:.5f} indep {r[10]:.5f} ce {ce:.4f} "
              f"miss {r[12]:.4f} fa {r[13]:.4f} ter_sh {ts:.5f} ter_in {ti:.5f} | {secs:.0f} s", flush=True)
        write_csv(self.path, HDR, self.rows)

    def finish(self):
        chan = {cn: i for i, (cn, _) in enumerate(CHANNELS)}
        self.rows.sort(key=lambda r: (chan[r[0]], ORDER.index(r[1]), r[2], r[3]))
        write_csv(self.path, HDR, self.rows)


def part_main(ctx, a):
    out = Out(a, "pm_content_fs")
    tlog, en = {}, {}
    jpath = pm.DATA / f"pm_content_fs{a.tag}.json"
    update_json(jpath, "setup", dict(
        users=U, d=D, V=ctx.V, block_subcarriers=B, static_L=L_STATIC, rhos=list(a.rhos),
        channels=[cn for cn, _ in a.chans], frames=a.frames, test_seed="11 + int(10 snr)",
        batch=BATCH, train=dict(iters=a.iters, batch=256, snr_db=TRAIN_SNR, seed=1, lr=1e-2),
        prior_clip=pc.PRIOR_CLIP, models=f"{a.models.name}/sem_fs_<channel>_rho<rho>_L16_s1.pt"))
    for cn, taps in a.chans:
        cond = pm.COND[cn]
        c_ad = adapt_c(ctx)
        st_cls = load_saved(f"white_U4_L16_{cn}_s1")
        ce_cls = mean_iota(st_cls.masks())
        print(f"[setup] {cn} adapted c {c_ad:.6f} static_class c {float(st_cls.c):.6f} ce {ce_cls:.4f}",
              flush=True)
        for rho in a.rhos:
            # adapted masks: standard, test and genie receivers on the same frames
            for snr in (10, 20):
                t = time.time()
                accs, _, agree, ce = run_adapted(ctx, c_ad, rho, snr, taps, a.frames,
                                                 TEST_SEED + int(10 * snr))
                for nm, acc in accs.items():
                    out.emit(cn, nm, snr, rho, acc, ce, c_ad, time.time() - t,
                             agree if nm == "adapted_test" else None)
            # full overlap
            m_fo = overlap_model(ctx, rho)
            for snr in (10, 20):
                t = time.time()
                acc = run_std(ctx, m_fo, rho, snr, cond, a.frames, TEST_SEED + int(10 * snr))
                out.emit(cn, "full_overlap", snr, rho, acc, mean_iota(m_fo.masks()), float(m_fo.c),
                         time.time() - t)
            # comb with the standard, test and genie receivers
            m_cb = comb_model(ctx, rho)
            t = time.time()
            accs, agree = run_comb(ctx, m_cb, rho, 10, cond, a.frames, TEST_SEED + 100)
            for nm, acc in accs.items():
                out.emit(cn, nm, 10, rho, acc, mean_iota(m_cb.masks()), float(m_cb.c), time.time() - t,
                         agree if nm == "comb_std" else None)
            # saved static entry of the class (independent content)
            t = time.time()
            acc = run_std(ctx, st_cls, rho, 10, cond, a.frames, TEST_SEED + 100)
            out.emit(cn, "static_class_L16", 10, rho, acc, ce_cls, float(st_cls.c), time.time() - t)
            # static entry trained for the class and rho at 10 dB
            m_tr, log, secs = train_cond(ctx.E, L_STATIC, ctx.pd, rho, a.iters, cond)
            key = f"{cn}_rho{rho:g}"
            if m_tr is None:
                tlog[key] = dict(failed="signal cancelled (c = 0)", seconds=round(secs, 1), loss=log)
                print("[static_trained]", key, "failed: signal cancelled", flush=True)
            else:
                fname = f"sem_fs_{cn}_rho{rho:g}_L16_s1"
                torch.save(m_tr.state_dict(), a.models / f"{fname}.pt")
                Mk = m_tr.masks().detach().cpu()
                ce_tr = mean_iota(Mk)
                tlog[key] = dict(model=fname, seconds=round(secs, 1), loss=log, c=float(m_tr.c),
                                 logit_scale=float(m_tr.logit_scale), cross_energy=ce_tr,
                                 cross_energy_user=pm.cross_energy(Mk),
                                 pattern=[[round(float(z), 3) for z in row] for row in Mk[:, :L_STATIC]])
                t = time.time()
                acc = run_std(ctx, m_tr, rho, 10, cond, a.frames, TEST_SEED + 100)
                out.emit(cn, "static_trained", 10, rho, acc, ce_tr, float(m_tr.c), time.time() - t)
                en[f"{cn}|static_trained|{rho:g}"] = energy(ctx, rho, m=m_tr)
            update_json(jpath, "training", tlog)
            # measured frame energies
            en[f"{cn}|adapted|{rho:g}"] = energy(ctx, rho, c=c_ad, taps=taps)
            en[f"{cn}|comb|{rho:g}"] = energy(ctx, rho, m=m_cb)
            en[f"{cn}|full_overlap|{rho:g}"] = energy(ctx, rho, m=m_fo)
            en[f"{cn}|static_class_L16|{rho:g}"] = energy(ctx, rho, m=st_cls)
            update_json(jpath, "energy", en)
            update_json(jpath, "points", out.points)
    out.finish()


def part_hi(ctx, a):
    out = Out(a, "pm_content_fs_hi")
    jpath = pm.DATA / f"pm_content_fs{a.tag}.json"
    ck = {}
    snr = 20
    for cn, taps in a.chans:
        c_ad = adapt_c(ctx)
        for rho in a.rhos:
            t = time.time()
            accs, snap, agree, ce = run_adapted(ctx, c_ad, rho, snr, taps, a.frames_hi,
                                                TEST_SEED + int(10 * snr), ckpt=a.frames)
            for nm, acc in accs.items():
                out.emit(cn, nm, snr, rho, acc, ce, c_ad, time.time() - t,
                         agree if nm == "adapted_test" else None)
                ck[f"{cn}|{nm}|{snr}|{rho:g}"] = dict(frames=snap[nm].frames, errors=sum(snap[nm].err),
                                                      ser_sem=sum(snap[nm].sem) / sum(snap[nm].tok))
            update_json(jpath, "hi_checkpoint", ck)
    out.finish()
    # compare the checkpoint with the main CSV when it exists
    main = pm.DATA / f"pm_content_fs{a.tag}.csv"
    if main.exists():
        with open(main) as f:
            ref = {(r["channel"], r["scheme"], int(r["snr_db"]), float(r["rho"])): int(r["errors"])
                   for r in csv.DictReader(f)}
        for k, v in ck.items():
            cn, nm, s, rho = k.split("|")
            rk = (cn, nm, int(s), float(rho))
            if rk in ref:
                v["main_errors"] = ref[rk]
                v["equal"] = int(ref[rk] == v["errors"])
                print(f"[hi-check] {k} checkpoint {v['errors']} main {ref[rk]} equal {v['equal']}", flush=True)
        update_json(jpath, "hi_checkpoint", ck)


def released(fname, **match):
    with open(pm.DATA / fname) as f:
        for row in csv.DictReader(f):
            if all(float(row[k]) == float(v) if isinstance(v, (int, float)) else row[k] == v
                   for k, v in match.items()):
                return row
    raise KeyError(f"{fname} {match}")


def part_check(ctx, a):
    jpath = pm.DATA / f"pm_content_fs{a.tag}.json"
    flat = pm.COND["flat"]
    res = []
    tok = 4 * a.frames

    def rec(item, ref, here, tol=0.0):
        eq = int(abs(float(ref) - float(here)) <= tol)
        res.append(dict(item=item, released=ref, here=here, equal=eq))
        print(f"[check] {item}: released {ref} here {here} equal {eq}", flush=True)
        update_json(jpath, "checks", res)

    # 1, 2: comb and full overlap under flat fading at 10 dB
    for rho in a.rhos:
        m_cb = comb_model(ctx, rho)
        accs, agree = run_comb(ctx, m_cb, rho, 10, flat, a.frames, TEST_SEED + 100)
        for nm, recv in (("comb_std", "standard_comb"), ("comb_test", "test_true_prior"), ("comb_genie", "genie")):
            r = released("pm_content_test.csv", receiver=recv, snr_db=10, rho=rho)
            rec(f"flat comb {nm} rho {rho:g} errors vs pm_content_test {recv}",
                int(round(float(r["ser"]) * tok)), sum(accs[nm].err))
        r = released("pm_semfig3.csv", scheme="no_overlap", snr_db=10, rho=rho)
        rec(f"flat comb_std rho {rho:g} errors vs pm_semfig3 no_overlap", int(r["errors"]), sum(accs["comb_std"].err))
        rec(f"flat comb_std rho {rho:g} ser_sem vs pm_semfig3 no_overlap", float(r["ser_sem"]),
            sum(accs["comb_std"].sem) / sum(accs["comb_std"].tok), 1e-9)
        rec(f"flat comb rho {rho:g} own-position decoder agrees with Masks.scores (tokens)", agree[1], agree[0])
        m_fo = overlap_model(ctx, rho)
        acc = run_std(ctx, m_fo, rho, 10, flat, a.frames, TEST_SEED + 100)
        r = released("pm_semfig3.csv", scheme="full_overlap", snr_db=10, rho=rho)
        rec(f"flat full_overlap rho {rho:g} errors vs pm_semfig3", int(r["errors"]), sum(acc.err))
        rec(f"flat full_overlap rho {rho:g} ser_sem vs pm_semfig3", float(r["ser_sem"]),
            sum(acc.sem) / sum(acc.tok), 1e-9)
    # 3: generalized trainer under flat fading against the saved Fig. 3 mask
    rho = 0.9
    m_tr, _, secs = train_cond(ctx.E, L_STATIC, ctx.pd, rho, a.iters, flat)
    sd = torch.load(pm.MODELS / "sem_fig3_rho0.9_L16_s1.pt", map_location="cpu")
    dW = float((m_tr.Wm.detach().cpu() - sd["Wm"]).abs().max())
    rec("trainer flat rho 0.9: max |Wm - saved sem_fig3_rho0.9_L16_s1| (0 = identical)", 0.0, dW, 1e-6)
    rec("trainer flat rho 0.9: c vs saved", float(sd["c"]), float(m_tr.c), 1e-6)
    acc = run_std(ctx, m_tr, rho, 10, flat, a.frames, TEST_SEED + 100)
    r = released("pm_semfig3.csv", scheme="trained_at_snr", snr_db=10, rho=rho)
    rec("trainer flat rho 0.9: errors vs pm_semfig3 trained_at_snr", int(r["errors"]), sum(acc.err))
    rec("trainer flat rho 0.9: ser_sem vs pm_semfig3 trained_at_snr", float(r["ser_sem"]),
        sum(acc.sem) / sum(acc.tok), 1e-9)
    # 4: adapted masks with the seeds of exp_pm_semfig2.run_adapted, rho = 0
    for cn, taps in a.chans:
        c_ad = adapt_c(ctx)
        for snr, frames, tested in ((10, a.frames, True), (20, a.frames_hi, False)):
            seed = 31 + int(10 * snr) + 100 * taps + 7 * B
            accs, _, agree, _ = run_adapted(ctx, c_ad, 0.0, snr, taps, frames, seed, chunk=384, tested=tested)
            r = released("pm_semfig2_check.csv", channel=cn, scheme="adapted_B2", snr_db=snr)
            rec(f"{cn} adapted_std rho 0 {snr} dB {frames} frames errors vs pm_semfig2_check ckpt "
                f"({r['ckpt_frames']} frames)", int(r["ckpt_errors"]), sum(accs["adapted_std"].err))
            if tested:
                rec(f"{cn} adapted_test rho 0 {snr} dB errors (test with prior 1e-3) vs adapted_std",
                    sum(accs["adapted_std"].err), sum(accs["adapted_test"].err))
                rec(f"{cn} adapted rho 0 {snr} dB own row of the test metric agrees with the standard "
                    f"metric (tokens)", agree[1], agree[0])
        # 5: saved static entry, rho = 0
        st = load_saved(f"white_U4_L16_{cn}_s1")
        acc = run_std(ctx, st, 0.0, 10, pm.COND[cn], a.frames, TEST_SEED + 100)
        r = released("pm_semfig2_check.csv", channel=cn, scheme="static_class_L16", snr_db=10)
        rec(f"{cn} static_class_L16 rho 0 10 dB errors vs pm_semfig2_check ckpt ({r['ckpt_frames']} frames)",
            int(r["ckpt_errors"]), sum(acc.err))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["main", "check", "hi", "all"])
    ap.add_argument("--frames", type=int, default=FRAMES)
    ap.add_argument("--frames_hi", type=int, default=FRAMES_HI)
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--smoke", action="store_true",
                    help="2,048 frames (4,096 in part hi), 50 steps, rho in {0, 0.9}, fs4, *_smoke outputs")
    a = ap.parse_args()
    a.tag, a.models, a.rhos, a.chans = "", pm.MODELS, RHOS, CHANNELS
    if a.smoke:
        a.frames, a.frames_hi, a.iters, a.tag = 2048, 4096, 50, "_smoke"
        a.rhos, a.chans = (0.0, 0.9), CHANNELS[:1]
        a.models = pm.DATA / "pm_models_smoke"
        a.models.mkdir(exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False             # evaluation in full fp32
    torch.backends.cudnn.allow_tf32 = False
    t0 = time.time()
    ctx = Ctx()
    print(f"[setup] V {ctx.V} unk {ctx.unk} device {DEVICE} t={time.time() - t0:.0f}s", flush=True)
    parts = {"main": (part_main,), "check": (part_check,), "hi": (part_hi,),
             "all": (part_check, part_main, part_hi)}[a.part]
    for f in parts:
        t = time.time()
        f(ctx, a)
        mins = round((time.time() - t) / 60, 1)
        print("[part-done]", f.__name__, mins, "min", flush=True)
        jpath = pm.DATA / f"pm_content_fs{a.tag}.json"
        d = json.load(open(jpath)).get("minutes", {}) if jpath.exists() else {}
        d[f.__name__] = mins
        update_json(jpath, "minutes", d)
    print("[done]", a.part, round((time.time() - t0) / 60, 1), "min", flush=True)


if __name__ == "__main__":
    main()
