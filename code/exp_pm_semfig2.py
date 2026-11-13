"""Semantic error rate of the channel figure (Fig. 2): SER and TER per point.

Metric. User u sends token s_u and decodes s_hat_u. The semantic error rate is

    SER_u = E[1 - zeta(s_u, s_hat_u)],   SER = (1/U) sum_u SER_u,
    zeta(v, w) = w_v^T w_w / (||w_v|| ||w_w||),

where w_v in R^768 is row v of the input token-embedding table of BERT base
uncased (exp_pm.Source("bert").Wn holds the normalized rows), the rule of
exp_pm.Evaluator: a decoded id outside [0, V) is mapped to [UNK] before the
cosine, and a token decoded correctly costs 0 (zeta(v, v) = 1). Over N
frames the estimate is

    ser_sem = (1/(N U)) sum_{n=1}^{N} sum_{u=1}^{U} 1[s_hat_{n,u} != s_{n,u}] (1 - zeta(s_{n,u}, s_hat_{n,u})),

so SER = TER (1 - E[zeta | s_hat != s]) with the token error rate
TER = Pr[s_hat_u != s_u]. TER is counted on the raw decoded ids, as in the
released CSVs (an out-of-range id is a token error even when the sent token
is [UNK]); the semantic cost uses the id after the [UNK] mapping. Every
point stores frames, tokens = U frames, errors (token errors), ter =
errors / tokens and ser_sem.

Setting: U = 4 tokens per frame drawn i.i.d. with the source statistics p_v
(exp_pm.Source("bert").dist()), d = 64 real dimensions, 4-tap (fs4) and
8-tap (fs8) fading of exp_pm.COND with exact gains, SNR 0-20 dB in steps
of 2 dB. Schemes (column scheme):
  oma               uncoded OMA, exp_pm.dec_oma (16 index bits, BPSK, one bit
                    per real dimension on the 8 adjacent subcarriers of the
                    user, detection by sign); ids >= V map to [UNK]
  coded_il_comb     RM(2,5) of the 16 index bits, code bits permuted by the
  coded_il_adapted  interleaver of data/pm/pm_oma_coded_il.json, Gray 16-QAM,
                    table scaled to unit energy per real dimension under p_v,
                    ML over the V valid ids with weights |g|^2 (exp_pm_oma.py
                    --part coded_il, code rm_il), on every U-th subcarrier
                    (comb) or on the adapted blocks of two subcarriers
  todma             exp_pm_todma.py: random Gaussian codewords (seed 0),
  todma_unmatched   plain sum scaled by c, zero forcing and weighted OMP (ipn
                    weights) detecting U tokens per receiver; the genie counts
                    user u correct when its own token is among its U detected
                    tokens. That detector assigns no detected token to a user,
                    so the semantic cost of a missed token is the mean of
                    1 - zeta(s_u, v) over the U detected tokens v (todma, the
                    cost of a token picked uniformly from the detected set).
                    todma_unmatched is a sensitivity row with the same
                    detections: the mean over the detected tokens that are
                    not the token of another user of the frame (a genie
                    assignment; at least one such token exists whenever the
                    own token is missed, since OMP picks U distinct tokens)
  static_class_L16  the static entry trained for the class (exp_pm.py --part
                    channel: white_U4_L16_fs4_s1 / white_U4_L16_fs8_s1),
                    decoded as exp_pm_ext.ser
  adapted_B2        masks adapted to the known gains on blocks of B = 2
                    subcarriers, no water-filling (exp_pm_ext.run_adapt), c
                    from 65,536 frames drawn with seed 0 as in
                    exp_pm_chanhi.py --part hi
No mask is trained here; the static entries are the saved ones.

Frames per point: coded OMA, static entry and adapted masks 200,000 below
12 dB and 2,000,000 from 12 dB on; oma and ToDMA 200,000 at every SNR.
Exception (part extra, table EXTRA): points with fewer than 20 token errors
in 2,000,000 frames, which Fig. 2 does not draw, run longer at 8 taps: the
adapted masks at 20 dB and coded OMA on the adapted blocks at 16, 18 and 20 dB
(frames in EXTRA). Their first frames are chunked as the shorter runs, so
the check rows compare the errors there with the released and the
2,000,000-frame points.

Seeds and chunking (those of the released parts, so that the first frames
of every point equal a released point):
  oma      five blocks of 40,000 frames; block b draws its tokens on the CPU
           with exp_pm_ext.sample after torch.manual_seed(5000 + snr + 100 T
           + 1000 b) and decodes with seed 777 + snr + 1000 b; block 0 is
           the point of exp_pm_ext.py --part figs (graph g1)
  coded    torch.manual_seed(9500 + 100 T + snr), chunks of 2,000 frames,
           draws per chunk tokens, gains, noise (exp_pm_oma.run_point); the
           draws do not depend on the set of codes, so the rm_il column of
           pm_oma_coded_il.csv is reproduced with rm_il alone
  todma    seed 71 + int(10 snr), chunks of 1,024 (exp_pm_todma.ser)
  static   seed 11 + int(10 snr), chunks of 512 (exp_pm_ext.ser); at 16, 18
           and 20 dB seed 13 + int(10 snr), the seed of exp_pm_chanhi.py
           --part class_hi
  adapted  seed 31 + int(10 snr) + 100 T + 14, chunks of 384
           (exp_pm_ext.run_adapt with B = 2)
T is the number of taps. Each run puts a chunk boundary at the frame count
of the released point (a shorter chunk there, as the released run ended),
records the errors at that checkpoint, and continues with full chunks.

Reproduction check (pm_semfig2_check.csv): errors at the checkpoint against
the released errors (ter times 4 frames):
  static_class_L16  pm_figs_class.csv (40,000 frames at 0-10 dB, 400,000 at
                    12 and 14 dB) and pm_figs_class_hi.csv (2,000,000 at
                    16-20 dB): equal
  adapted_B2        pm_figs_hi.csv (400,000 at 12-20 dB): equal;
                    pm_figs.csv g1 (40,000 at 0-10 dB), whose c was drawn
                    without a seed: within a few errors
  coded_il_*        pm_oma_coded_il.csv rm_il (200,000 at 0-14 dB, 2,000,000
                    at 16-20 dB): equal
  oma               pm_figs.csv g1 (40,000): equal
  todma             pm_todma.csv (200,000): equal
"Equal" holds on the same GPU type and torch version as the released runs
(torch 2.11); column exact marks it.

Parts (--part): oma, coded, todma, static, adapted (each writes
data/pm/semfig2_parts/pm_semfig2_<part>.csv and _<part>_check.csv), merge
(joins the five parts into data/pm/pm_semfig2.csv with columns channel,
scheme, snr_db, frames, tokens, errors, ter, ser_sem, and
data/pm/pm_semfig2_check.csv, then puts in the rows of part extra when its
file exists), all (the five parts, then merge), extra (the points of EXTRA
with more frames; it writes pm_semfig2_extra.csv and _extra_check.csv and
replaces their rows in data/pm/pm_semfig2.csv and pm_semfig2_check.csv).
--channel fs4 or fs8 restricts a part to one channel (files
pm_semfig2_<part>_<channel>.csv), so that the long parts can run as two
processes; merge takes the file of a part or, without it, its two channel
files. --smoke
runs every part at 10 dB with the released frame count (ToDMA 20,480
frames), prints the check and writes nothing.

Run from the code folder on a GPU:
    python exp_pm_semfig2.py --part all
    python exp_pm_semfig2.py --part extra      # after all
    python exp_pm_semfig2.py --part static     # one part; then --part merge
    python exp_pm_semfig2.py --part adapted --channel fs8
    python exp_pm_semfig2.py --part oma --smoke
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time

import torch

import exp_pm as pm
import exp_pm_chanhi as ch
import exp_pm_ext as px
import exp_pm_oma as po
import exp_pm_todma as pt
import rev_channel as rc
from sse_lib import DEVICE, D, write_csv

U = 4
SNRS = list(range(0, 21, 2))
CHANNELS = (("fs4", 4), ("fs8", 8))
LO, HI = 200_000, 2_000_000
FR_C16, FR_C18, FR_C20 = 20_000_000, 100_000_000, 1_000_000_000   # coded OMA, adapted blocks, 8 taps
# Points with fewer than 20 token errors in HI frames, run with more frames:
# (channel, scheme, snr_db) -> (frames, token errors of the HI-frame run)
EXTRA = {("fs8", "adapted_B2", 20): (40_000_000, 4),
         ("fs8", "coded_il_adapted", 16): (FR_C16, 11),
         ("fs8", "coded_il_adapted", 18): (FR_C18, 1),
         ("fs8", "coded_il_adapted", 20): (FR_C20, 0)}
PARTS = ("oma", "coded", "todma", "static", "adapted")
ORDER = ("oma", "coded_il_comb", "coded_il_adapted", "todma", "todma_unmatched",
         "static_class_L16", "adapted_B2")
PART_DIR = pm.DATA / "semfig2_parts"
HDR = ["channel", "scheme", "snr_db", "frames", "tokens", "errors", "ter", "ser_sem"]
HDR_CHK = ["channel", "scheme", "snr_db", "ref_file", "ref_frames", "ref_errors", "ref_ter",
           "ckpt_frames", "ckpt_errors", "ckpt_ter", "exact", "frames", "ter", "ser_sem"]


# ----------------------------------------------------------------------
# context, accumulator, chunking, released references
# ----------------------------------------------------------------------
class Ctx:
    def __init__(self):
        self.src = pm.Source("bert")
        self.pdc = self.src.dist()                       # p_v on the CPU
        self.pd = self.pdc.to(DEVICE)
        self.V = self.src.V
        self.unk = self.src.unk
        self.Wn = self.src.Wn.to(DEVICE)                 # (V, 768) normalized BERT rows


class Acc:
    """Frames, token errors and semantic cost of one point."""

    def __init__(self, ctx: Ctx):
        self.Wn, self.unk, self.V = ctx.Wn, ctx.unk, ctx.V
        self.frames = 0
        self.errors = 0
        self.sem = 0.0

    def add(self, idx: torch.Tensor, dec: torch.Tensor) -> None:
        """idx, dec: (n, U) sent and decoded ids on DEVICE."""
        self.frames += idx.shape[0]
        self.errors += int((dec != idx).sum())
        d2 = torch.where((dec < 0) | (dec >= self.V), torch.full_like(dec, self.unk), dec)
        wrong = d2 != idx
        if bool(wrong.any()):
            s, t = idx[wrong], d2[wrong]
            self.sem += float((1.0 - (self.Wn[s] * self.Wn[t]).sum(-1)).double().sum())

    def snap(self):
        return self.frames, self.errors

    def row(self, cn, scheme, snr):
        tok = U * self.frames
        return [cn, scheme, snr, self.frames, tok, self.errors, self.errors / tok, self.sem / tok]


def chunks(frames: int, size: int, ckpt: int, ckpt2: int | None = None):
    """Chunk sizes of a run of `frames` frames whose first `ckpt` frames are
    chunked as a released run of `ckpt` frames (full chunks, then the rest)
    and, when ckpt2 > ckpt is given, whose first ckpt2 frames are chunked as
    a run of ckpt2 frames. Yields (n, True when the checkpoint ckpt is
    reached)."""
    k = 0
    while k < frames:
        nxt = ckpt if k < ckpt else (ckpt2 if ckpt2 is not None and k < ckpt2 else None)
        n = min(size, frames - k, (nxt - k) if nxt is not None else size)
        k += n
        yield n, k == ckpt


def released(fname: str, **match):
    with open(pm.DATA / fname) as f:
        for row in csv.DictReader(f):
            if all(row[k] == str(v) for k, v in match.items()):
                return row
    raise KeyError(f"{fname} {match}")


def ref_point(scheme: str, cn: str, snr: int):
    """(file, frames, errors, ter) of the released point that the first
    frames of a run reproduce."""
    if scheme == "static_class_L16":
        f = "pm_figs_class.csv" if snr <= 14 else "pm_figs_class_hi.csv"
        r = released(f, graph="g1", channel=cn, scheme="static_class_L16", snr_db=snr)
        fr, v = int(r["frames"]), float(r["ser"])
    elif scheme == "adapted_B2":
        if snr >= 12:
            f = "pm_figs_hi.csv"
            r = released(f, graph="g1", channel=cn, scheme="adapted_B2", snr_db=snr)
            fr, v = int(r["frames"]), float(r["ser"])
        else:
            f = "pm_figs.csv"
            r = released(f, graph="g1", channel=cn, scheme="adapted_B2", snr_db=snr)
            fr, v = 40_000, float(r["ser"])
    elif scheme == "oma":
        f = "pm_figs.csv"
        r = released(f, graph="g1", channel=cn, scheme="oma", snr_db=snr)
        fr, v = 40_000, float(r["ser"])
    elif scheme.startswith("coded_il_"):
        f = "pm_oma_coded_il.csv"
        r = released(f, code="rm_il", placement=scheme[len("coded_il_"):], channel=cn, snr_db=snr)
        return f, int(r["frames"]), int(r["errors"]), float(r["ser"])
    elif scheme == "todma":
        f = "pm_todma.csv"
        r = released(f, graph="g1", channel=cn, scheme="todma", snr_db=snr)
        fr, v = int(r["frames"]), float(r["ser"])
    else:
        raise ValueError(scheme)
    return f, fr, int(round(v * U * fr)), v


def check_row(cn, scheme, snr, ref, snap, acc_row):
    f, rfr, rerr, rter = ref
    cfr, cerr = snap
    exact = int(cfr == rfr and cerr == rerr)
    row = [cn, scheme, snr, f, rfr, rerr, rter, cfr, cerr, cerr / (U * cfr), exact,
           acc_row[3], acc_row[6], acc_row[7]]
    print(f"[check] {cn} {scheme} {snr} dB released {rerr}/{rfr} ({f}) "
          f"here {cerr}/{cfr} exact {exact}", flush=True)
    return row


def frames_for(scheme: str, snr: int, smoke: bool, ref_frames: int) -> int:
    if smoke:
        return 20_480 if scheme == "todma" else ref_frames
    if scheme in ("oma", "todma"):
        return LO
    return HI if snr >= 12 else LO


# ----------------------------------------------------------------------
# schemes
# ----------------------------------------------------------------------
@torch.no_grad()
def run_oma(ctx, snr, cn, taps, frames):
    """Uncoded OMA (exp_pm.dec_oma) in blocks of 40,000 frames; block 0 is the
    point of exp_pm_ext.py --part figs."""
    acc, snap = Acc(ctx), None
    for b in range(frames // 40_000):
        torch.manual_seed(5000 + snr + 100 * taps + 1000 * b)
        ids = px.sample(ctx.pdc, 40_000, U, 0.0).cpu()
        dec = pm.dec_oma(ids, snr, pm.COND[cn], 777 + snr + 1000 * b)
        acc.add(ids.to(DEVICE), dec.to(DEVICE))
        if b == 0:
            snap = acc.snap()
    return acc, snap


def coded_table(ctx):
    """Interleaved RM(2,5) in Gray 16-QAM at unit energy per real dimension
    under p_v (exp_pm_oma.py part coded_il, code rm_il)."""
    perm = torch.tensor(json.load(open(pm.DATA / "pm_oma_coded_il.json"))["interleaver"])
    c = ((po.bits_msb(torch.arange(ctx.V), po.BITS) @ po.rm25_generator()) % 2)[:, perm]
    A = po.modulate(c, "gray4")
    p = ctx.pd.detach().double().cpu()
    e = float((p[:, None] * A.double().pow(2)).sum(0).mean() / p.sum())
    A = (A / math.sqrt(e)).to(DEVICE)
    return A, po.ml_matrix(A), e


@torch.no_grad()
def run_coded(ctx, A, Mml, place, taps, snr, frames, ckpt, ckpt2=None):
    """exp_pm_oma.run_point for the rm_il table alone.
    With ckpt2, acc.snap2 holds (frames, errors) after the first ckpt2 frames."""
    torch.manual_seed(9500 + 100 * taps + snr)
    S = po.K // U
    sig2 = 1.0 / (D * 10 ** (snr / 10))
    sig = math.sqrt(sig2)
    a = 1.0 / math.sqrt(D)
    acc, snap = Acc(ctx), None
    acc.snap2 = None
    for n, at in chunks(frames, po.CHUNK, ckpt, ckpt2):
        idx = torch.multinomial(ctx.pd, n * U, replacement=True).view(n, U)
        g = po.gains(n, U, taps)
        sub = po.subcarriers(place, g, sig2)
        w = po.noise(n, U, S, sig)
        r, wt = po.observe(g, sub, a * A[idx], w)
        acc.add(idx, po.dec_ml(r, wt, Mml))
        if at:
            snap = acc.snap()
        if ckpt2 is not None and acc.frames == ckpt2:
            acc.snap2 = acc.snap()
    return acc, snap


@torch.no_grad()
def run_todma(ctx, A, c, snr, cn, frames, ckpt):
    """exp_pm_todma.ser (weighted OMP, ipn weights) with the semantic cost of
    a missed token averaged over the U detected tokens (todma) or over the
    detected tokens that are no other user's token (todma_unmatched)."""
    torch.manual_seed(71 + int(10 * snr))
    acc, acc_u, snap = Acc(ctx), Acc(ctx), None
    Wn = ctx.Wn
    for n, at in chunks(frames, pt.CHUNK, ckpt):
        idx = torch.multinomial(ctx.pd, n * U, replacement=True).view(n, U)
        x = A[idx].sum(1) / c
        r, g2 = rc.channel(x, snr, U, **pm.COND[cn])
        wt = pt.weights(g2, snr, c, "ipn")
        for u in range(U):
            sup = pt.detect(r[:, u], wt[:, u], A, "omp")                  # (n, U) distinct ids
            miss = ~(sup == idx[:, u:u + 1]).any(1)
            m = int(miss.sum())
            acc.errors += m
            acc_u.errors += m
            if m:
                s, det = idx[miss, u], sup[miss]                            # (m,), (m, U)
                cost = 1.0 - (Wn[det] * Wn[s][:, None, :]).sum(-1)          # (m, U)
                acc.sem += float(cost.mean(1).double().sum())
                others = idx[miss][:, [k for k in range(U) if k != u]]      # (m, U-1)
                free = ~(det[:, :, None] == others[:, None, :]).any(-1)     # (m, U)
                acc_u.sem += float(((cost * free).sum(1) / free.sum(1).clamp_min(1)).double().sum())
        acc.frames += n
        acc_u.frames += n
        if at:
            snap = acc.snap()
    return acc, acc_u, snap


@torch.no_grad()
def run_static(ctx, m, snr, cn, frames, ckpt, seed):
    """exp_pm_ext.ser (independent tokens) with the semantic cost."""
    torch.manual_seed(seed + int(10 * snr))
    acc, snap = Acc(ctx), None
    for n, at in chunks(frames, 512, ckpt):
        idx = px.sample(ctx.pd, n, U, 0.0)
        x = m.tx(idx) / m.c
        r, w = rc.channel(x, snr, U, **pm.COND[cn])
        acc.add(idx, m.scores(r, w).argmax(-1))
        if at:
            snap = acc.snap()
    return acc, snap


@torch.no_grad()
def run_adapted(ctx, E, c, snr, taps, frames, ckpt, seed=31, ckpt2=None):
    """exp_pm_ext.run_adapt with B = 2, no water-filling, with the semantic cost.
    With ckpt2, acc.snap2 holds (frames, errors) after the first ckpt2 frames."""
    B = 2
    torch.manual_seed(seed + int(10 * snr) + 100 * taps + 7 * B)
    K = D // 2
    sig = math.sqrt(1.0 / (D * 10 ** (snr / 10)))
    acc, snap = Acc(ctx), None
    acc.snap2 = None
    for n, at in chunks(frames, 384, ckpt, ckpt2):
        idx = px.sample(ctx.pd, n, U, 0.0)
        g = px.gains(n, taps, DEVICE)
        M = px.adaptive_masks(g, B, sig * sig, c, False)
        x = (E[idx] * M).sum(1) / c
        xc = torch.complex(x[:, 0::2], x[:, 1::2])
        w = torch.complex(torch.randn(n, U, K, device=DEVICE), torch.randn(n, U, K, device=DEVICE)) * sig
        q = (g * xc[:, None, :] + w) / g
        r = torch.stack([q.real, q.imag], -1).reshape(n, U, D)
        wt = (g.abs() ** 2).repeat_interleave(2, dim=-1)
        a_ = 1.0 / c
        sc = 2 * a_ * torch.einsum("nud,vd->nuv", wt * r * M, E) \
            - a_ * a_ * torch.einsum("nud,vd->nuv", wt * M * M, E * E)
        acc.add(idx, sc.argmax(-1))
        if at:
            snap = acc.snap()
        if ckpt2 is not None and acc.frames == ckpt2:
            acc.snap2 = acc.snap()
    return acc, snap


# ----------------------------------------------------------------------
# parts
# ----------------------------------------------------------------------
def tag(part, a):
    return part if a.channel == "both" else f"{part}_{a.channel}"


def emit(rows, chks, part, a, t0, cn, scheme, snr, acc, snap, ref):
    r = acc.row(cn, scheme, snr)
    rows.append(r)
    print(f"[{part}] {cn} {scheme} {snr} dB frames {r[3]} errors {r[5]} ter {r[6]:.4e} "
          f"ser_sem {r[7]:.4e} t={time.time() - t0:.0f}s", flush=True)
    if snap is not None and ref is not None:
        chks.append(check_row(cn, scheme, snr, ref, snap, r))
    if not a.smoke:
        write_csv(PART_DIR / f"pm_semfig2_{tag(part, a)}.csv", HDR, rows)
        write_csv(PART_DIR / f"pm_semfig2_{tag(part, a)}_check.csv", HDR_CHK, chks)


def part_oma(ctx, a, snrs, t0):
    rows, chks = [], []
    for cn, taps in a.chans:
        for snr in snrs:
            ref = ref_point("oma", cn, snr)
            fr = 40_000 if a.smoke else LO
            acc, snap = run_oma(ctx, snr, cn, taps, fr)
            emit(rows, chks, "oma", a, t0, cn, "oma", snr, acc, snap, ref)


def part_coded(ctx, a, snrs, t0):
    A, Mml, e = coded_table(ctx)
    print(f"[coded] rm_il table energy per real dim under p_v before scaling {e:.4f}", flush=True)
    rows, chks = [], []
    for cn, taps in a.chans:
        for place in ("comb", "adapted"):
            sch = f"coded_il_{place}"
            for snr in snrs:
                ref = ref_point(sch, cn, snr)
                fr = frames_for(sch, snr, a.smoke, ref[1])
                acc, snap = run_coded(ctx, A, Mml, place, taps, snr, fr, ref[1])
                emit(rows, chks, "coded", a, t0, cn, sch, snr, acc, snap, ref)


def part_todma(ctx, a, snrs, t0):
    A = pm.codebook(ctx.src.W, D, "rand", seed=0).to(DEVICE)
    c = pt.calibrate(A, ctx.pd)
    print(f"[todma] c {c:.4f}", flush=True)
    rows, chks = [], []
    for cn, _ in a.chans:
        for snr in snrs:
            ref = ref_point("todma", cn, snr)
            fr = frames_for("todma", snr, a.smoke, ref[1])
            acc, acc_u, snap = run_todma(ctx, A, c, snr, cn, fr, ref[1])
            emit(rows, chks, "todma", a, t0, cn, "todma", snr, acc, snap, ref)
            emit(rows, chks, "todma", a, t0, cn, "todma_unmatched", snr, acc_u, None, None)


def part_static(ctx, a, snrs, t0):
    rows, chks = [], []
    for cn, _ in a.chans:
        m = ch.load(f"white_U4_L16_{cn}_s1")
        for snr in snrs:
            ref = ref_point("static_class_L16", cn, snr)
            fr = frames_for("static_class_L16", snr, a.smoke, ref[1])
            seed = 13 if snr >= 16 else 11
            acc, snap = run_static(ctx, m, snr, cn, fr, ref[1], seed)
            emit(rows, chks, "static", a, t0, cn, "static_class_L16", snr, acc, snap, ref)


def adapted_setup(ctx):
    """Codebook (1) and the constant c of the adapted masks."""
    E = pm.codebook(ctx.src.W, D, "pwhite", pd=ctx.pdc).to(DEVICE)
    comb = torch.zeros(U, D, device=DEVICE)
    for u in range(U):
        comb[u, u::4] = 2.0
    torch.manual_seed(0)                       # fixed draw for the constant c
    idx0 = px.sample(ctx.pd, 65536, U, 0.0)
    c = float(((E[idx0] * comb[None]).sum(1)).pow(2).sum(1).mean().sqrt())
    print(f"[adapted] c {c:.6f}", flush=True)
    return E, c


def part_adapted(ctx, a, snrs, t0):
    E, c = adapted_setup(ctx)
    rows, chks = [], []
    for cn, taps in a.chans:
        for snr in snrs:
            ref = ref_point("adapted_B2", cn, snr)
            fr = frames_for("adapted_B2", snr, a.smoke, ref[1])
            acc, snap = run_adapted(ctx, E, c, snr, taps, fr, ref[1])
            emit(rows, chks, "adapted", a, t0, cn, "adapted_B2", snr, acc, snap, ref)


def read_rows(path):
    with open(path) as f:
        rd = csv.reader(f)
        next(rd)
        return [r for r in rd]


def apply_extra(rows, chks):
    """Rows of part extra in place of the same (channel, scheme, snr_db) rows,
    and its check rows in place of the check rows of those points."""
    p, q = PART_DIR / "pm_semfig2_extra.csv", PART_DIR / "pm_semfig2_extra_check.csv"
    if not p.exists():
        return rows, chks
    ext = {tuple(r[:3]): r for r in read_rows(p)}
    extc = read_rows(q)
    keys = {tuple(r[:3]) for r in extc}
    rows = [ext.get(tuple(str(x) for x in r[:3]), r) for r in rows]
    chks = [r for r in chks if tuple(str(x) for x in r[:3]) not in keys] + extc
    return rows, chks


def part_extra(ctx, a, snrs, t0):
    E = A = None
    keep, keepc = [], []                       # rows of the other schemes of part extra (--only)
    if a.only and not a.smoke and (PART_DIR / "pm_semfig2_extra.csv").exists():
        keep = [r for r in read_rows(PART_DIR / "pm_semfig2_extra.csv") if r[1] != a.only]
        keepc = [r for r in read_rows(PART_DIR / "pm_semfig2_extra_check.csv") if r[1] != a.only]
    rows, chks = [], []
    for (cn, scheme, snr), (fr, hi_err) in EXTRA.items():
        if a.only and scheme != a.only:
            continue
        taps = dict(CHANNELS)[cn]
        ref = ref_point(scheme, cn, snr)
        frames = ref[1] if a.smoke else fr
        ck2 = HI if (not a.smoke and ref[1] < HI) else None
        if scheme == "adapted_B2":
            if E is None:
                E, c = adapted_setup(ctx)
            acc, snap = run_adapted(ctx, E, c, snr, taps, frames, ref[1], ckpt2=ck2)
        elif scheme == "coded_il_adapted":
            if A is None:
                A, Mml, _ = coded_table(ctx)
            acc, snap = run_coded(ctx, A, Mml, "adapted", taps, snr, frames, ref[1], ckpt2=ck2)
        else:
            raise ValueError(scheme)
        emit(rows, chks, "extra", a, t0, cn, scheme, snr, acc, snap, ref)
        if acc.snap2 is not None:   # the first HI frames against the HI-frame run
            chks.append(check_row(cn, scheme, snr, ("pm_semfig2.csv", HI, hi_err, hi_err / (U * HI)),
                                  acc.snap2, rows[-1]))
    if a.smoke:
        return
    rows, chks = keep + rows, keepc + chks
    write_csv(PART_DIR / f"pm_semfig2_{tag('extra', a)}.csv", HDR, rows)
    write_csv(PART_DIR / f"pm_semfig2_{tag('extra', a)}_check.csv", HDR_CHK, chks)
    chan = {"fs4": 0, "fs8": 1}

    def key(r):
        return chan[r[0]], ORDER.index(r[1]), int(r[2])
    all_rows, all_chks = apply_extra(read_rows(pm.DATA / "pm_semfig2.csv"),
                                     read_rows(pm.DATA / "pm_semfig2_check.csv"))
    all_chks.sort(key=key)
    write_csv(pm.DATA / "pm_semfig2.csv", HDR, all_rows)
    write_csv(pm.DATA / "pm_semfig2_check.csv", HDR_CHK, all_chks)
    print("[extra] rows replaced in pm_semfig2.csv and pm_semfig2_check.csv", flush=True)


def part_merge():
    rows, chks = [], []
    for p in PARTS:
        tags = [p] if (PART_DIR / f"pm_semfig2_{p}.csv").exists() else [f"{p}_{cn}" for cn, _ in CHANNELS]
        for t in tags:
            rows += read_rows(PART_DIR / f"pm_semfig2_{t}.csv")
            chks += read_rows(PART_DIR / f"pm_semfig2_{t}_check.csv")
            print("[merge]", t, flush=True)
    rows, chks = apply_extra(rows, chks)
    chan = {"fs4": 0, "fs8": 1}

    def key(r):
        return chan[r[0]], ORDER.index(r[1]), int(r[2])
    rows.sort(key=key)
    chks.sort(key=key)
    n_exp = len(CHANNELS) * len(ORDER) * len(SNRS)
    assert len(rows) == n_exp and len({tuple(r[:3]) for r in rows}) == n_exp, (len(rows), n_exp)
    write_csv(pm.DATA / "pm_semfig2.csv", HDR, rows)
    write_csv(pm.DATA / "pm_semfig2_check.csv", HDR_CHK, chks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=list(PARTS) + ["extra", "merge", "all"])
    ap.add_argument("--channel", default="both", choices=["both", "fs4", "fs8"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--only", default="", help="part extra: run only this scheme of EXTRA")
    a = ap.parse_args()
    a.chans = [c for c in CHANNELS if a.channel in ("both", c[0])]
    torch.backends.cuda.matmul.allow_tf32 = False             # evaluation in full fp32
    torch.backends.cudnn.allow_tf32 = False
    t0 = time.time()
    if a.part == "merge":
        part_merge()
    else:
        if not a.smoke:
            PART_DIR.mkdir(exist_ok=True)
        ctx = Ctx()
        print(f"[setup] V {ctx.V} unk {ctx.unk} device {DEVICE} t={time.time() - t0:.0f}s", flush=True)
        snrs = [10] if a.smoke else SNRS
        fns = {"oma": part_oma, "coded": part_coded, "todma": part_todma,
               "static": part_static, "adapted": part_adapted, "extra": part_extra}
        for p in (PARTS if a.part == "all" else (a.part,)):
            fns[p](ctx, a, snrs, t0)
        if a.part == "all" and not a.smoke and a.channel == "both":
            part_merge()
        if a.smoke:
            print("[smoke] no file written", flush=True)
    print("[done]", tag(a.part, a), round((time.time() - t0) / 60, 1), "min", flush=True)


if __name__ == "__main__":
    main()
