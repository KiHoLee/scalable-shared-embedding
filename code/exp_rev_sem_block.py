"""Revision experiment R1.1 / R2.5: topic accuracy on the real AG News token
streams when one gain is held over several frames.

exp_rev_semantic.py draws a new gain per frame, so token errors spread evenly
through an article. Here each user's h_u, with h_u^2 ~ Exp(1) and known at the
receiver, is held for T consecutive frames, T in {1, 16, 64}, with a random
block offset per user, or for one whole article (T = article; a mean article
has about 51 tokens). Gains stay independent across users and across blocks.
The noise, the streams and the scoring are those of exp_rev_sem_runs.py.
T = 1 uses the unmodified decoders of exp_rev_semantic.py.

Schemes: SSE (sse_P4.pt), the full codebook with minimum-distance decoding, and
OMA, at 10, 16 and 20 dB, 4 repeats, with the seed SEED_EVAL + 1000 r + 10 snr.

Metrics, as means over the repeats: ter, acc, agree and cos as in
exp_rev_sem_runs.py; art_ter_gt50, the fraction of articles with a token error
rate above 0.5; acc_rep_sd, the standard deviation of acc over the repeats; and
for T = article, the accuracy on articles with h_u^2 < 0.1 and with h_u^2 >= 0.1.

Output: data/rev_sem_block.csv
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch

import exp_rev_semantic as E
from exp_rev_sem_runs import Task, load_full, load_sse, repeat_seed
from sse_lib import DATA, DEVICE, U, write_csv

SNRS = [10, 16, 20]
TS = [1, 16, 64, "article"]
REPEATS = 4
SEED_BLOCK = 987654
SCHEMES = ("SSE", "Full", "OMA")


def block_gain(N, T, items, seed):
    """(N,U) gains on the CPU. Integer T: blocks of T frames with a random
    offset per user. T = 'article': one gain per article of that user, and the
    tail frames after the last complete article share one more gain."""
    g = torch.Generator().manual_seed(seed + SEED_BLOCK)
    h = torch.zeros(N, U)
    for u in range(U):
        if T == "article":
            idx = torch.zeros(N, dtype=torch.long)
            for j, (a, b, _) in enumerate(items[u]):
                idx[a:b] = j
            idx[items[u][-1][1]:] = len(items[u])
            nb = len(items[u]) + 1
        else:
            off = int(torch.randint(T, (1,), generator=g))
            idx = (torch.arange(N) + off) // T
            nb = int(idx.max()) + 1
        v = torch.rand(nb, generator=g).clamp_min(1e-12)
        h[:, u] = torch.sqrt(-torch.log(v))[idx]
    return h


@torch.no_grad()
def dec_sse(model, ids, snr, seed, h, chunk=50_000):
    """E.dec_sse with given gains."""
    torch.manual_seed(seed)
    out = torch.zeros_like(ids)
    dg_all = E.ids_to_digits(ids)
    for n0 in range(0, ids.shape[0], chunk):
        dg = dg_all[n0:n0 + chunk].to(DEVICE)
        sc = model(dg, torch.full((dg.shape[0],), float(snr)),
                   h=h[n0:n0 + chunk].to(DEVICE))
        out[n0:n0 + chunk] = E.digits_to_ids(sc.argmax(-1)).cpu()
    return out


@torch.no_grad()
def dec_full(model, ids, snr, seed, h):
    """E.dec_full with given gains."""
    torch.manual_seed(seed)
    chunk = max(128, (1 << 23) // model.Vc)
    out = torch.zeros_like(ids)
    for n0 in range(0, ids.shape[0], chunk):
        idx = ids[n0:n0 + chunk].to(DEVICE)
        sc = model(idx, torch.full((idx.shape[0],), float(snr)),
                   h=h[n0:n0 + chunk].to(DEVICE))
        out[n0:n0 + chunk] = sc.argmax(-1).cpu()
    return out


@torch.no_grad()
def dec_oma(ids, snr, seed, h, bits=16):
    """E.dec_oma with given gains."""
    torch.manual_seed(seed)
    b = ((ids[..., None] >> torch.arange(bits)) & 1).float() * 2 - 1
    b = b.to(DEVICE)
    sigma = math.sqrt(1.0 / (10.0 ** (snr / 10.0)))
    y = (h.to(DEVICE)[..., None] * b
         + sigma * torch.randn(ids.shape[0], U, bits, device=DEVICE))
    bh = (y > 0).long().cpu()
    return (bh << torch.arange(bits)).sum(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snrs", default=",".join(map(str, SNRS)))
    ap.add_argument("--repeats", type=int, default=REPEATS)
    ap.add_argument("--out", default=str(DATA / "rev_sem_block.csv"))
    a = ap.parse_args()
    snrs = [int(s) for s in a.snrs.split(",")]

    t = Task()
    ids, items = t.ids, t.items
    sse, full = load_sse("sse_P4.pt"), load_full()
    a0 = [x for u in range(U) for x, _, _ in items[u]]
    b0 = [x for u in range(U) for _, x, _ in items[u]]
    u0 = [u for u in range(U) for _ in items[u]]
    plain = {"SSE": lambda s, k: E.dec_sse(sse, ids, s, k),
             "Full": lambda s, k: E.dec_full(full, ids, s, k),
             "OMA": lambda s, k: E.dec_oma(ids, s, k)}
    held = {"SSE": lambda s, k, h: dec_sse(sse, ids, s, k, h),
            "Full": lambda s, k, h: dec_full(full, ids, s, k, h),
            "OMA": lambda s, k, h: dec_oma(ids, s, k, h)}

    rows = [["clean", "none", "inf", 0.0, t.clean_acc, 1.0, 1.0, 0.0, 0.0, "", ""]]
    for s in snrs:
        for nm in SCHEMES:
            for T in TS:
                res = []
                for r in range(a.repeats):
                    k = repeat_seed(r, s)
                    h = None if T == 1 else block_gain(ids.shape[0], T, items, k)
                    dec = plain[nm](s, k) if T == 1 else held[nm](s, k, h)
                    ter, ok, agr, cos = t.score(dec)
                    err = (dec != ids).float()
                    art = np.array([float(err[x:y, u].mean())
                                    for x, y, u in zip(a0, b0, u0)])
                    lo = hi = float("nan")
                    if T == "article":
                        h2 = (h[a0, u0] ** 2).numpy()
                        lo, hi = float(ok[h2 < 0.1].mean()), float(ok[h2 >= 0.1].mean())
                    res.append([ter, ok.mean(), agr, cos, (art > 0.5).mean(), lo, hi])
                res = np.array(res, dtype=float)
                m = res.mean(0)
                sd = res[:, 1].std(ddof=1) if a.repeats > 1 else float("nan")
                tail = [float(m[5]), float(m[6])] if T == "article" else ["", ""]
                rows.append([nm, T, s, float(m[0]), float(m[1]), float(m[2]),
                             float(m[3]), float(m[4]), float(sd), *tail])
                print("[block]", nm, "T", T, "snr", s,
                      "ter %.4f acc %.4f (sd %.4f) agree %.4f cos %.4f art>0.5 %.4f"
                      % (m[0], m[1], sd, m[2], m[3], m[4]), tail, flush=True)
    write_csv(Path(a.out),
              ["scheme", "T", "snr_db", "ter", "acc", "agree", "cos",
               "art_ter_gt50", "acc_rep_sd", "acc_given_h2_lt0p1",
               "acc_given_h2_ge0p1"], rows)


if __name__ == "__main__":
    main()
