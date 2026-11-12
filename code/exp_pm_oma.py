"""Coded OMA references for the periodic-mask letter.

The OMA reference of the letter sends the 16 bits of each token index
uncoded. This script gives OMA a channel code of the same rate (16 bits or
fewer per token on the d/U real dimensions of each user) and places it on
fixed or on adapted subcarriers, so that the gain of the pretrained codeword
can be separated from the gain of a code and from the gain of choosing
strong subcarriers.

Codes (code bits of token id v; the index bits b15 ... b0, b15 = 0 for every
id since V = 30,522 < 2^15):
  none    the 16 index bits uncoded, bit k on real dimension k (as
          exp_pm.dec_oma at U = 4), BPSK, detection by sign
  spc     single parity check: the 15 bits b14 ... b0 and their parity in
          place of the always-zero top bit; BPSK; Wagner decoding, which is
          ML over the 2^15 codewords: hard decisions on the weighted
          observations, and if the parity fails the least reliable bit is
          flipped (decoded ids from V to 2^15 - 1 are out of range)
  spc_ml  the same code and modulation, ML over the V valid ids (weighted
          minimum distance against all V codewords, no prior); not part of
          the specification, added so that part coded and part sem share a
          decoder for the parity check
  rm      RM(2,5), (32,16,8): the bits b15 ... b0 times the generator with
          rows 1, x1..x5, xi xj (b15 = 0 multiplies the all-ones row); ML
          over the V valid ids, no prior
Modulation: BPSK puts one code bit on one real dimension; Gray 4-PAM puts
the code-bit pair (c_2i, c_2i+1) on real dimension i with levels 00 -> -3,
01 -> -1, 11 -> +1, 10 -> +3 (divided by sqrt(5), unit energy for equiprobable
levels), so the real dimensions (2k, 2k+1) of a
subcarrier carry Gray 16-QAM. At U = 4 a user owns 16 real dimensions (8
subcarriers): none and spc are QPSK, rm is Gray 16-QAM.
Energy: every amplitude table A (V, d/U) is scaled to unit average energy
per real dimension under the token frequencies p_v (exp_pm.Source.dist),
and the frame carries A[v_u] / sqrt(d) for each user, so the average frame
energy under p_v is one, as for the masks. BPSK tables need no scaling; the
Gray 16-QAM RM table has energy 1.0138 under p_v before scaling.
Channel: the model of rev_channel.channel with exact gains, evaluated on each
user's own subcarriers (OMA receivers ignore the others): subcarrier k is the
real pair (2k, 2k+1), z_k = g_{u,k} x_k + w_k, w_k ~ CN(0, 2 sigma^2),
sigma^2 = 1/(d gamma); flat fading uses one g_u ~ CN(0,1), T taps use
g_{u,k} = sum_t a_t exp(-j 2 pi k t / K), a_t ~ CN(0, 1/T), K = 32. The
receiver zero-forces and decodes with weights |g_{u,k}|^2 (weighted minimum
distance, the decoder of the masks).
Placements (K/U subcarriers per user):
  adjacent  user u on subcarriers uK/U, ..., (u+1)K/U - 1 (the OMA of Fig. 2)
  comb      user u on subcarriers u, u+U, u+2U, ...
  adapted   the blocks of two subcarriers that the adapted mask of user u
            weights (exp_pm_ext.adaptive_masks, B = 2, no water-filling,
            formed from the known gains), in increasing subcarrier order

Parts (--part):
  coded  U = 4, tokens drawn i.i.d. with p_v. Codes none, spc, spc_ml, rm;
         channels flat (adjacent only), fs4 and fs8 (4 and 8 taps) with the
         placements adjacent, comb and adapted; SNR 0-20 dB in steps of 2.
         200,000 frames per point; from 14 dB on, a point with fewer than 50
         token errors is extended to 1,000,000 frames (the same random
         stream continued). SER = errors / (4 frames).
         Output data/pm/pm_oma_coded.csv
         (code, placement, channel, snr_db, frames, errors, ser).
  sem    text metrics (exp_pm.Evaluator) on the AG News test streams, SNR 0,
         5, 10, 15, 20 dB: flat fading with U = 2 (rm, BPSK on 32 real
         dimensions), U = 4 (spc and spc_ml, BPSK; rm, Gray 16-QAM) and U = 8
         (spc_ml on Gray 4-PAM, 8 real dimensions); 4 and 8 taps at U = 4
         (rm adjacent, rm adapted, and the uncoded none on adapted blocks).
         The released uncoded OMA (exp_pm.dec_oma: BPSK, BPSK sent twice at
         U = 2, natural 4-PAM at U = 8, adjacent subcarriers) is decoded with
         the released seeds at every U and channel. Every scheme that can
         decode an out-of-range id (none, spc) is scored twice: with
         out-of-range ids deleted, as exp_pm.Evaluator does by mapping them to
         [UNK] (oor_rule unk), and with each out-of-range id replaced by a
         token drawn uniformly from the V valid ids (oor_rule rand).
         oor_share is the share of wrong ids that lie out of range;
         frame_energy is the average frame energy measured on the test
         streams. ML decoders over the valid ids never leave the range, so
         spc_ml and rm have one row (oor_rule unk).
         Output data/pm/pm_oma_coded_sem.csv (users, channel, placement,
         code, modulation, oor_rule, snr_db, frames, seed, frame_energy, ser,
  coded_il RM(2,5) as in part coded, with the code bits permuted before the
         Gray 16-QAM mapping by the interleaver that maximizes the minimum
         number of subcarriers on which two valid tokens differ (searched
         over 20,000 random permutations, seed 0); comb and adapted
         placements, 4 and 8 taps, 0-20 dB, 200,000 frames below 16 dB and
         2,000,000 from 16 dB on. Output data/pm/pm_oma_coded_il.csv and
         pm_oma_coded_il.json (the interleaver and its minimum).
  sem_il text metrics on the test streams under 4 and 8 taps for the
         interleaved coded OMA (every U-th subcarrier and adapted blocks), the
         adapted masks, the static entry trained for the class and uncoded
         OMA, 0/5/10/15/20 dB. Output data/pm/pm_oma_coded_il_sem.csv.
         oor_share, sem_dist, err_tok_sim, err_close, topic_acc, gtr_sim;
         the last six as exp_pm.Evaluator.score).
  all    coded, then sem

Seeds: part coded seeds torch with 9000 + 100 T + snr (T = 1 for flat) at
every point; all codes and placements of one channel and SNR share the
tokens, gains and noise (chunks of 2,000 frames, draws in the order tokens,
gains, noise). Part sem decodes with seed 777 + 10 snr, as the released
test-stream evaluations, and draws the replacement tokens of oor_rule rand
with a CPU generator seeded 5000 + 10 snr. Evaluation runs in full fp32.

Run from the code folder (GPU recommended):
    python exp_pm_oma.py --part coded      # data/pm/pm_oma_coded.csv
    python exp_pm_oma.py --part sem        # data/pm/pm_oma_coded_sem.csv
    python exp_pm_oma.py --part coded --smoke   # short check, writes nothing
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import time

import torch

import exp_pm as pm
import exp_pm_ext as px
import rev_channel as rc
from sse_lib import DEVICE, D, write_csv

K = D // 2                      # subcarriers
BITS = 16                       # index bits per token
B_ADAPT = 2                     # subcarriers per adapted block
CHUNK = 2000                    # frames per chunk in part coded


# ----------------------------------------------------------------------
# codes and amplitude tables
# ----------------------------------------------------------------------
def bits_lsb(ids: torch.Tensor, nb: int) -> torch.Tensor:
    """(..., nb), column k = bit k."""
    return torch.stack([(ids >> k) & 1 for k in range(nb)], -1)


def bits_msb(ids: torch.Tensor, nb: int) -> torch.Tensor:
    """(..., nb), column k = bit nb - 1 - k."""
    return torch.stack([(ids >> (nb - 1 - k)) & 1 for k in range(nb)], -1)


def rm25_generator() -> torch.Tensor:
    """(16, 32) generator of RM(2,5): rows 1, x1..x5, xi xj (i < j)."""
    pts = torch.tensor(list(itertools.product([0, 1], repeat=5)))      # (32,5)
    rows = [torch.ones(32, dtype=torch.long)] + [pts[:, i] for i in range(5)]
    rows += [pts[:, i] * pts[:, j] for i in range(5) for j in range(i + 1, 5)]
    return torch.stack(rows)


def code_bits(code: str, ids: torch.Tensor) -> torch.Tensor:
    if code == "none":
        return bits_lsb(ids, BITS)
    if code in ("spc", "spc_ml"):
        b = bits_msb(ids, BITS)[:, 1:]                                  # b14 ... b0
        return torch.cat([b, b.sum(-1, keepdim=True) % 2], -1)
    if code == "rm":
        return (bits_msb(ids, BITS) @ rm25_generator()) % 2
    raise ValueError(code)


def modulate(c: torch.Tensor, mod: str) -> torch.Tensor:
    if mod == "bpsk":
        return 2.0 * c.float() - 1.0
    if mod == "gray4":
        p = c.view(c.shape[0], -1, 2)
        lv = torch.tensor([-3.0, -1.0, 3.0, 1.0]) / math.sqrt(5.0)     # index 2 c_2i + c_2i+1
        return lv[2 * p[..., 0] + p[..., 1]]
    raise ValueError(mod)


def table(code: str, mod: str, pd: torch.Tensor, V: int):
    """Amplitude table (V, n) with unit average energy per real dimension
    under p_v, on DEVICE, and its energy before scaling."""
    A = modulate(code_bits(code, torch.arange(V)), mod)
    p = pd.detach().double().cpu()
    e = float((p[:, None] * A.double().pow(2)).sum(0).mean() / p.sum())
    return (A / math.sqrt(e)).to(DEVICE), e


def ml_matrix(A: torch.Tensor) -> torch.Tensor:
    """[2a A, -a^2 A^2] with a = 1/sqrt(d): the weighted minimum-distance
    metric of codeword v is [w r, w] . row v."""
    a = 1.0 / math.sqrt(D)
    return torch.cat([2 * a * A, -a * a * A * A], 1)


# ----------------------------------------------------------------------
# channel, placements, decoders
# ----------------------------------------------------------------------
def gains(n: int, users: int, taps: int) -> torch.Tensor:
    """exp_pm_ext.gains with the number of users as a parameter: (n,U,K)."""
    def cn(*shape):
        return torch.complex(torch.randn(*shape, device=DEVICE),
                             torch.randn(*shape, device=DEVICE)) / math.sqrt(2)
    if taps == 1:
        return cn(n, users, 1).expand(n, users, K).contiguous()
    a = cn(n, users, taps) / math.sqrt(taps)
    k = torch.arange(K, device=DEVICE)
    t = torch.arange(taps, device=DEVICE)
    F = torch.exp(-2j * math.pi * torch.outer(t, k).float() / K)
    return a @ F.to(a.dtype)


def subcarriers(place: str, g: torch.Tensor, sig2: float) -> torch.Tensor:
    """(n,U,K/U) subcarrier indices of every user, increasing."""
    n, users, _ = g.shape
    S = K // users
    ar = torch.arange(S, device=DEVICE)
    uu = torch.arange(users, device=DEVICE)[:, None]
    if place == "adjacent":
        return (ar[None] + S * uu).expand(n, users, S)
    if place == "comb":
        return (ar[None] * users + uu).expand(n, users, S)
    if place == "adapted":
        own = px.adaptive_masks(g, B_ADAPT, sig2, 1.0, False)[..., 0::2] > 0   # (n,U,K)
        pos = torch.sort(own.float(), dim=-1, descending=True, stable=True).indices[..., :S]
        return pos.sort(-1).values
    raise ValueError(place)


def observe(g, sub, xs, w):
    """User u's own subcarriers: xs (n,U,2S) real symbols in frame scale,
    w (n,U,S) complex noise. Returns the zero-forced observation r and the
    weights |g|^2, both (n,U,2S)."""
    gs = g.gather(2, sub)
    xc = torch.complex(xs[..., 0::2], xs[..., 1::2])
    q = (gs * xc + w) / gs
    r = torch.stack([q.real, q.imag], -1).flatten(-2)
    wt = (gs.abs() ** 2).repeat_interleave(2, -1)
    return r, wt


def noise(n: int, users: int, S: int, sig: float) -> torch.Tensor:
    return torch.complex(torch.randn(n, users, S, device=DEVICE),
                         torch.randn(n, users, S, device=DEVICE)) * sig


def dec_sign(r):
    """none: bit k = sign of real dimension k."""
    b = (r > 0).long()
    return (b << torch.arange(b.shape[-1], device=r.device)).sum(-1)


def dec_wagner(r, wt):
    """spc: hard decisions on w r; if the parity fails, flip the least
    reliable bit; the id is the 15 information bits b14 ... b0."""
    llr = r * wt
    hard = (llr > 0).long()
    bad = (hard.sum(-1) % 2) == 1
    j = llr.abs().argmin(-1, keepdim=True)
    hard = hard ^ (torch.zeros_like(hard).scatter_(-1, j, 1) * bad[..., None])
    sh = torch.arange(BITS - 2, -1, -1, device=r.device)
    return (hard[..., :BITS - 1] << sh).sum(-1)


def dec_ml(r, wt, M):
    """Weighted minimum distance over the V rows of the table behind M."""
    n, users, m = r.shape
    s = torch.cat([wt * r, wt], -1).reshape(n * users, 2 * m) @ M.T
    return s.argmax(1).view(n, users)


def decode(code, r, wt, M):
    if code == "none":
        return dec_sign(r)
    if code == "spc":
        return dec_wagner(r, wt)
    return dec_ml(r, wt, M)


# ----------------------------------------------------------------------
# part coded: SER on tokens drawn with p_v, U = 4
# ----------------------------------------------------------------------
@torch.no_grad()
def run_point(codes, tabs, pd, place, taps, snr, frames, frames_hi, seed):
    """Token errors of every code at one placement, channel and SNR. All codes
    see the same tokens, gains and noise. A code with fewer than 50 errors
    after `frames` frames continues to `frames_hi` when snr >= 14."""
    users = 4
    S = K // users
    torch.manual_seed(seed)
    sig2 = 1.0 / (D * 10 ** (snr / 10))
    sig = math.sqrt(sig2)
    a = 1.0 / math.sqrt(D)
    err = {c: 0 for c in codes}
    nfr = {c: 0 for c in codes}
    active = list(codes)
    n0 = 0
    while active and n0 < frames_hi:
        n = min(CHUNK, frames - n0) if n0 < frames else min(CHUNK, frames_hi - n0)
        idx = torch.multinomial(pd, n * users, replacement=True).view(n, users)
        g = gains(n, users, taps)
        sub = subcarriers(place, g, sig2)
        w = noise(n, users, S, sig)
        for c in active:
            A, M = tabs[c]
            r, wt = observe(g, sub, a * A[idx], w)
            err[c] += int((decode(c, r, wt, M) != idx).sum())
            nfr[c] += n
        n0 += n
        if n0 == frames:
            active = [c for c in active if snr >= 14 and err[c] < 50]
    return err, nfr


def part_coded(a):
    t0 = time.time()
    src = pm.Source("bert")
    pd = src.dist().to(DEVICE)
    V = src.V
    spec = {"none": ("none", "bpsk"), "spc": ("spc", "bpsk"),
            "spc_ml": ("spc_ml", "bpsk"), "rm": ("rm", "gray4")}
    tabs = {}
    for name, (code, mod) in spec.items():
        A, e = table(code, mod, pd, V)
        tabs[name] = (A, ml_matrix(A))
        print(f"[table] {name} {mod} energy per real dim under p_v before scaling {e:.4f}", flush=True)
    codes = list(spec)
    snrs = [10, 20] if a.smoke else list(range(0, 21, 2))
    settings = [("flat", 1, "adjacent")] + [(cn, t, pl) for cn, t in (("fs4", 4), ("fs8", 8))
                                            for pl in ("adjacent", "comb", "adapted")]
    rows = []
    for cn, taps, place in settings:
        for snr in snrs:
            err, nfr = run_point(codes, tabs, pd, place, taps, snr, a.frames, a.frames_hi,
                                 9000 + 100 * taps + snr)
            for c in codes:
                v = err[c] / (4 * nfr[c])
                rows.append([c, place, cn, snr, nfr[c], err[c], v])
            print(f"[coded] {cn} {place} {snr} dB " + " ".join(
                f"{c} {err[c] / (4 * nfr[c]):.3e} ({err[c]}/{nfr[c]})" for c in codes)
                + f" t={time.time() - t0:.0f}s", flush=True)
    hdr = ["code", "placement", "channel", "snr_db", "frames", "errors", "ser"]
    if a.smoke:
        print("[smoke] no file written", flush=True)
    else:
        write_csv(pm.DATA / "pm_oma_coded.csv", hdr, rows)


# ----------------------------------------------------------------------
# part sem: text metrics on the test streams
# ----------------------------------------------------------------------
@torch.no_grad()
def dec_streams(ids, snr, taps, place, code, A, M, seed):
    users = ids.shape[1]
    S = K // users
    chunk = 8192 // users
    sig2 = 1.0 / (D * 10 ** (snr / 10))
    sig = math.sqrt(sig2)
    a = 1.0 / math.sqrt(D)
    torch.manual_seed(seed)
    out = torch.empty_like(ids)
    for n0 in range(0, ids.shape[0], chunk):
        idx = ids[n0:n0 + chunk].to(DEVICE)
        n = idx.shape[0]
        g = gains(n, users, taps)
        sub = subcarriers(place, g, sig2)
        w = noise(n, users, S, sig)
        r, wt = observe(g, sub, a * A[idx], w)
        out[n0:n0 + n] = decode(code, r, wt, M).cpu()
    return out


def energy_table(ids, A):
    return float(A[ids.to(DEVICE)].pow(2).sum((-1, -2)).mean()) / D


def energy_oma(ids):
    """Frame energy of exp_pm.dec_oma on the streams (as exp_pm_oma_energy.py)."""
    users = ids.shape[1]
    n_sym, Mq, rep = rc.oma_layout(users, D, BITS)
    bps = int(math.log2(Mq))
    sym = torch.stack([(ids >> (bps * k)) & (Mq - 1) for k in range(n_sym)], -1)
    return float(rc.oma_frames(sym, users, D, Mq, rep).pow(2).sum(-1).mean())


def part_sem(a):
    t0 = time.time()
    src = pm.Source("bert")
    pd = src.dist()
    V = src.V
    sc = pm.Scorer()
    tabs = {}
    for code, mod in (("rm", "bpsk"), ("spc", "bpsk"), ("spc_ml", "bpsk"), ("rm", "gray4"),
                      ("spc_ml", "gray4"), ("none", "bpsk")):
        A, e = table(code, mod, pd, V)
        tabs[(code, mod)] = (A, ml_matrix(A))
        print(f"[table] {code} {mod} energy per real dim under p_v before scaling {e:.4f}", flush=True)
    oma_mod = {2: "bpsk_rep2", 4: "bpsk", 8: "pam4_natural"}
    # (users, channel, taps, [(placement, code, modulation), ...]); the released
    # uncoded OMA (exp_pm.dec_oma, adjacent) is added to every entry
    plan = [(2, "flat", 1, [("adjacent", "rm", "bpsk")]),
            (4, "flat", 1, [("adjacent", "spc", "bpsk"), ("adjacent", "spc_ml", "bpsk"),
                            ("adjacent", "rm", "gray4")]),
            (8, "flat", 1, [("adjacent", "spc_ml", "gray4")]),
            (4, "fs4", 4, [("adjacent", "rm", "gray4"), ("adapted", "rm", "gray4"),
                           ("adapted", "none", "bpsk")]),
            (4, "fs8", 8, [("adjacent", "rm", "gray4"), ("adapted", "rm", "gray4"),
                           ("adapted", "none", "bpsk")])]
    snrs = [0, 5, 10, 15, 20]
    if a.smoke:
        plan, snrs = [plan[1]], [10]
    rows = []
    evs = {}
    for users, cn, taps, schemes in plan:
        if users not in evs:
            evs[users] = pm.Evaluator(src, users, sc)
            print(f"[clean] U={users} frames {evs[users].ids.shape[0]} acc {evs[users].clean_acc:.4f}",
                  flush=True)
        ev = evs[users]
        ids = ev.ids
        nfr = ids.shape[0]
        for snr in snrs:
            seed = 777 + 10 * snr
            entries = [("adjacent", "none", oma_mod[users], None)] + \
                      [(pl, c, m, tabs[(c, m)]) for pl, c, m in schemes]
            for place, code, mod, tab in entries:
                if tab is None:
                    dec = pm.dec_oma(ids, snr, pm.COND[cn], seed)
                    e = energy_oma(ids)
                else:
                    dec = dec_streams(ids, snr, taps, place, code, tab[0], tab[1], seed)
                    e = energy_table(ids, tab[0])
                oor = (dec < 0) | (dec >= V)
                wrong = dec != ids
                share = float(oor.sum()) / max(1, int(wrong.sum()))
                rules = ["unk"] + (["rand"] if code in ("none", "spc") else [])
                for rule in rules:
                    d2 = dec
                    if rule == "rand":
                        gen = torch.Generator().manual_seed(5000 + 10 * snr)
                        d2 = dec.clone()
                        d2[oor] = torch.randint(0, V, (int(oor.sum()),), generator=gen)
                    res = ev.score(d2)
                    rows.append([users, cn, place, code, mod, rule, snr, nfr, seed, e,
                                 res[0], share] + res[1:])
                    print(f"[sem] U={users} {cn} {place} {code} {mod} {rule} {snr} dB "
                          f"ser {res[0]:.4f} oor_share {share:.3f} topic {res[4]:.4f} "
                          f"gtr {res[5]:.4f} energy {e:.4f} t={time.time() - t0:.0f}s", flush=True)
    hdr = ["users", "channel", "placement", "code", "modulation", "oor_rule", "snr_db", "frames",
           "seed", "frame_energy", "ser", "oor_share", "sem_dist", "err_tok_sim", "err_close",
           "topic_acc", "gtr_sim"]
    if a.smoke:
        print("[smoke] no file written", flush=True)
    else:
        write_csv(pm.DATA / "pm_oma_coded_sem.csv", hdr, rows)



# ----------------------------------------------------------------------
# part coded_il: RM(2,5) with a bit interleaver chosen for diversity
# ----------------------------------------------------------------------
def symbol_weights(C: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    """Number of 16-QAM symbols (subcarriers) on which each codeword of C
    (rows, 32 bits) is nonzero when bit perm[4k:4k+4] goes to symbol k."""
    return C[:, perm].view(C.shape[0], 8, 4).any(-1).sum(-1)


def search_interleaver(trials: int = 20000, seed: int = 0):
    """Bit permutation of the RM(2,5) codeword that maximizes the minimum
    number of subcarriers on which two valid tokens differ, then minimizes
    the number of codeword differences that reach this minimum. Valid ids
    lie below 2^15, so their differences are the codewords with b15 = 0."""
    G = rm25_generator()[1:]                                   # drop the all-ones row (b15)
    msg = bits_msb(torch.arange(1, 2 ** 15), 15)               # nonzero messages b14 ... b0
    C = ((msg @ G) % 2).bool().to(DEVICE)
    gen = torch.Generator().manual_seed(seed)
    cands = [torch.arange(32), torch.arange(32).view(4, 8).T.reshape(-1)]
    cands += [torch.randperm(32, generator=gen) for _ in range(trials)]
    best = None
    for p in cands:
        sw = symbol_weights(C, p.to(DEVICE))
        key = (int(sw.min()), -int((sw == sw.min()).sum()))
        if best is None or key > best[0]:
            best = (key, p.clone())
    plain = symbol_weights(C, torch.arange(32, device=DEVICE))
    return best[1], best[0][0], -best[0][1], int(plain.min()), int((plain == plain.min()).sum())


def part_coded_il(a):
    """RM(2,5) in Gray 16-QAM as in part coded, but with the code bits
    permuted before the mapping by the interleaver of search_interleaver, on
    every U-th subcarrier (comb) and on the adapted blocks, 4 and 8 taps,
    0-20 dB; 200,000 frames below 16 dB and 2,000,000 from 16 dB on, so the
    high-SNR points of Fig. 2 rest on many errors (data/pm/pm_oma_coded_il.csv)."""
    t0 = time.time()
    src = pm.Source("bert")
    pd = src.dist().to(DEVICE)
    V = src.V
    perm, mw, mult, mw0, mult0 = search_interleaver(200 if a.smoke else 20000)
    print(f"[il] interleaver {perm.tolist()} min subcarriers {mw} (count {mult}); "
          f"plain mapping {mw0} (count {mult0})", flush=True)
    tabs = {}
    for name, bits_perm in (("rm", torch.arange(32)), ("rm_il", perm)):
        c = ((bits_msb(torch.arange(V), BITS) @ rm25_generator()) % 2)[:, bits_perm]
        A = modulate(c, "gray4")
        p = pd.detach().double().cpu()
        e = float((p[:, None] * A.double().pow(2)).sum(0).mean() / p.sum())
        A = (A / math.sqrt(e)).to(DEVICE)
        tabs[name] = (A, ml_matrix(A))
    codes = list(tabs)
    snrs = [10, 20] if a.smoke else list(range(0, 21, 2))
    rows = []
    for cn, taps in (("fs4", 4), ("fs8", 8)):
        for place in ("comb", "adapted"):
            for snr in snrs:
                fr = (a.frames if snr < 16 else 10 * a.frames) if not a.smoke else a.frames
                err, nfr = run_point(codes, tabs, pd, place, taps, snr, fr, fr,
                                     9500 + 100 * taps + snr)
                for c in codes:
                    rows.append([c, place, cn, snr, nfr[c], err[c], err[c] / (4 * nfr[c])])
                print(f"[coded_il] {cn} {place} {snr} dB " + " ".join(
                    f"{c} {err[c] / (4 * nfr[c]):.3e} ({err[c]}/{nfr[c]})" for c in codes)
                    + f" t={time.time() - t0:.0f}s", flush=True)
    hdr = ["code", "placement", "channel", "snr_db", "frames", "errors", "ser"]
    if a.smoke:
        print("[smoke] no file written", flush=True)
    else:
        write_csv(pm.DATA / "pm_oma_coded_il.csv", hdr, rows)
        with open(pm.DATA / "pm_oma_coded_il.json", "w") as f:
            json.dump({"interleaver": perm.tolist(), "min_subcarriers": mw, "count": mult,
                       "plain_min_subcarriers": mw0, "plain_count": mult0}, f, indent=1)



# ----------------------------------------------------------------------
# part sem_il: text metrics of the interleaved coded OMA and of the masks
# ----------------------------------------------------------------------
@torch.no_grad()
def dec_adapted_streams(E, ids, snr, taps, c, seed):
    """Adapted masks (blocks of two subcarriers, as in Fig. 2) on the test
    streams: the decoder of exp_pm_ext.run_adapt applied to given tokens."""
    users = ids.shape[1]
    Kc = D // 2
    sig = math.sqrt(1.0 / (D * 10 ** (snr / 10)))
    torch.manual_seed(seed)
    out = torch.empty_like(ids)
    chunk = 384
    for n0 in range(0, ids.shape[0], chunk):
        idx = ids[n0:n0 + chunk].to(DEVICE)
        n = idx.shape[0]
        g = px.gains(n, taps, DEVICE)
        M = px.adaptive_masks(g, B_ADAPT, sig * sig, c, False)
        x = (E[idx] * M).sum(1) / c
        xc = torch.complex(x[:, 0::2], x[:, 1::2])
        w = torch.complex(torch.randn(n, users, Kc, device=DEVICE),
                          torch.randn(n, users, Kc, device=DEVICE)) * sig
        q = (g * xc[:, None, :] + w) / g
        r = torch.stack([q.real, q.imag], -1).reshape(n, users, D)
        wt = (g.abs() ** 2).repeat_interleave(2, dim=-1)
        a_ = 1.0 / c
        sc = 2 * a_ * torch.einsum("nud,vd->nuv", wt * r * M, E) \
            - a_ * a_ * torch.einsum("nud,vd->nuv", wt * M * M, E * E)
        out[n0:n0 + n] = sc.argmax(-1).cpu()
    return out


def part_sem_il(a):
    """Topic accuracy and GTR similarity on the AG News test streams under 4-
    and 8-tap fading, U = 4, SNR 0/5/10/15/20 dB, for the RM(2,5) coded OMA
    with the interleaver of part coded_il (data/pm/pm_oma_coded_il.json) on
    every U-th subcarrier and on the adapted blocks, the adapted masks, the
    static entry trained for the class (white_U4_L16_fs4_s1 / _fs8_s1) and
    uncoded OMA (data/pm/pm_oma_coded_il_sem.csv)."""
    t0 = time.time()
    src = pm.Source("bert")
    pdc = src.dist()
    V = src.V
    E = pm.codebook(src.W, D, "pwhite", pd=pdc).to(DEVICE)
    perm = torch.tensor(json.load(open(pm.DATA / "pm_oma_coded_il.json"))["interleaver"])
    c_bits = ((bits_msb(torch.arange(V), BITS) @ rm25_generator()) % 2)[:, perm]
    A = modulate(c_bits, "gray4")
    p = pdc.detach().double().cpu()
    A = (A / math.sqrt(float((p[:, None] * A.double().pow(2)).sum(0).mean() / p.sum()))).to(DEVICE)
    M = ml_matrix(A)
    comb = torch.zeros(4, D, device=DEVICE)
    for u in range(4):
        comb[u, u::4] = 2.0
    torch.manual_seed(0)
    idx0 = px.sample(pdc.to(DEVICE), 65536, 4, 0.0)
    c_comb = float(((E[idx0] * comb[None]).sum(1)).pow(2).sum(1).mean().sqrt())
    sc = pm.Scorer()
    ev = pm.Evaluator(src, 4, sc)
    ids = ev.ids
    rows = []
    snrs = [10] if a.smoke else [0, 5, 10, 15, 20]
    for cn, taps in (("fs4", 4), ("fs8", 8)):
        sd = torch.load(pm.MODELS / f"white_U4_L16_{cn}_s1.pt", map_location="cpu")
        st = pm.Masks(sd["E"], 16, 4)
        st.load_state_dict(sd)
        st = st.to(DEVICE).eval()
        for snr in snrs:
            seed = 777 + 10 * snr
            decs = {"oma": pm.dec_oma(ids, snr, pm.COND[cn], seed),
                    "coded_il_comb": dec_streams(ids, snr, taps, "comb", "rm", A, M, seed),
                    "coded_il_adapted": dec_streams(ids, snr, taps, "adapted", "rm", A, M, seed),
                    "static_class": pm.dec_mask(st, ids, snr, pm.COND[cn], seed),
                    "adapted_masks": dec_adapted_streams(E, ids, snr, taps, c_comb, seed)}
            for name, dec in decs.items():
                res = ev.score(dec)
                rows.append([cn, name, snr, ids.shape[0]] + list(res))
                print(f"[sem_il] {cn} {name} {snr} dB ser {res[0]:.4f} topic {res[4]:.4f} "
                      f"gtr {res[5]:.4f} t={time.time() - t0:.0f}s", flush=True)
    hdr = ["channel", "scheme", "snr_db", "frames", "ser", "sem_dist", "err_tok_sim", "err_close",
           "topic_acc", "gtr_sim"]
    if a.smoke:
        print("[smoke] no file written", flush=True)
    else:
        write_csv(pm.DATA / "pm_oma_coded_il_sem.csv", hdr, rows)


def part_all(a):
    part_coded(a)
    part_sem(a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["coded", "sem", "coded_il", "sem_il", "all"])
    ap.add_argument("--frames", type=int, default=200_000)
    ap.add_argument("--frames_hi", type=int, default=1_000_000)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.frames, a.frames_hi = 4 * CHUNK, 8 * CHUNK
    assert a.frames % CHUNK == 0 and a.frames_hi % CHUNK == 0
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    t0 = time.time()
    {"coded": part_coded, "sem": part_sem, "coded_il": part_coded_il, "sem_il": part_sem_il, "all": part_all}[a.part](a)
    print("[done]", a.part, round((time.time() - t0) / 60, 1), "min", flush=True)


if __name__ == "__main__":
    main()
