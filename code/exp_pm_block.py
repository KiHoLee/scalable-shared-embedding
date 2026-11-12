"""Text metrics with a channel gain held over time (block fading).

The text metrics of exp_pm.py (pm_models_eval.csv) draw a new Rayleigh gain
in every frame, so the token errors of an article spread over it. The net
efficiency of Fig. 5 instead assumes gains that stay fixed over the frames
served by one mask update. This script grades the decoded articles when each
user's gain is held over several frames.

Setup: BERT tokens on the test streams of exp_pm.Evaluator (the first 2,000
AG News test articles, user u carries articles u, u + 4, ...; the four streams
are cut to the length of the shortest, 24,998 frames, so 1,948 articles are
graded), U = 4, d = 64, flat Rayleigh fading with exact gains at the
receiver, weighted minimum-distance decoding over all 30,522 tokens. No
training; the saved masks of exp_pm.py --part models are loaded:
  pwhite  the proposal, codebook whitened under the source statistics,
          L = 16, training runs 1-3 (bert_pwhite_U4_L16_flat_s{1,2,3})
  rand    the i.i.d. random codebook with the same masks and decoder, one
          training run (bert_rand_U4_L16_flat_s1)
  oma     16 uncoded bits per token, BPSK on 16 real dimensions per user
          (QPSK on 8 adjacent subcarriers), detected by sign; an identifier
          outside the vocabulary becomes [UNK], as in exp_pm.Evaluator

Gain held for (column hold):
  1        a new gain in every frame, through rev_channel.channel with the
           chunking of exp_pm.dec_mask and exp_pm.dec_oma; draw 0 uses the
           seeds 777 + 10 snr and reproduces pm_models_eval.csv (rows kind =
           check give the differences at 0, 10 and 20 dB)
  16, 64   blocks of 16 or 64 consecutive frames of each user's stream, with
           a random block offset per user
  article  one gain per article: every user carries its own articles and its
           gain is held over each of them; the frames after a user's last
           graded article share one more gain
Gains are CN(0, 1), independent across users and blocks, and the same for
every scheme and every SNR of one draw. SNR 0, 5, 10, 15, 20 dB; two draws
(fading and noise) per point.

Part (--part):
  block   every hold, SNR, draw and scheme above; writes both outputs

Outputs in data/pm/:
  pm_block_eval.csv      summary, one row per
      kind = clean   error-free text
      kind = run     hold, SNR, scheme, training run (seed) and draw: SER over
                     all 4 x 24,998 tokens, semantic distortion, topic
                     accuracy, GTR similarity, share of OMA's wrong tokens
                     that fall outside the vocabulary (oov_share)
      kind = mean    mean over the draws (and over the three runs for pwhite)
      kind = diff    pwhite-oma and pwhite-rand: the per-article difference
                     (pwhite averaged over its runs and draws, the reference
                     over its draws), averaged over the graded articles, with
                     the paired percentile bootstrap 95 percent interval over
                     articles (1,000 resamples, one resample matrix for all
                     rows); the SER difference is token weighted over the
                     graded articles
      kind = widen   the diff at a held gain minus the diff at hold 1, paired
                     over the same articles, with its bootstrap interval
      kind = h2bin   held gains only: articles pooled over the draws and split
                     by the mean gain power h2 of the article (bins 0-0.1,
                     0.1-0.3, 0.3-1, 1-3, above 3); share of articles, token
                     SER, topic accuracy and GTR similarity per scheme
      kind = check   hold 1, draw 0: this run minus pm_models_eval.csv
  pm_block_articles.csv  per hold, SNR, draw and graded article: user, tokens,
      h2 (mean gain power |g_u|^2 over the article's frames; at hold 1 the
      gains seen by the mask decoders), h2_oma (hold 1 only, where the OMA
      decoder draws its own gains; empty otherwise, since every scheme sees
      h2), and per scheme the number of wrong tokens (err_*), whether the
      topic is kept (top_*) and the GTR similarity (gtr_*, three decimals).
      For pwhite, err and top are summed over the three runs (top 0-3) and
      gtr is their mean.

Seeds: noise 777 + 10 snr + 1000 i_hold + 10000 draw (i_hold = 0, 1, 2, 3
for hold 1, 16, 64, article; draw 0, 1); gains 424242 + 1000 i_hold +
10000 draw (CPU generator); bootstrap numpy default_rng(0).

Run from the code folder on a GPU (about one hour with the GPU shared):
    python exp_pm_block.py --part block
    python exp_pm_block.py --part block --smoke   # first 2,048 frames, 10 dB,
                                                  # writes *_smoke.csv
"""
from __future__ import annotations

import argparse
import csv
import math
import time

import numpy as np
import torch

import exp_pm as pm
import rev_channel as rc
from sse_lib import DEVICE, D, write_csv

U = 4
HOLDS = ["1", "16", "64", "article"]
SNRS = [0, 5, 10, 15, 20]
DRAWS = 2
RUNS = (1, 2, 3)
BOOT = 1000
BOOT_SEED = 0
GAIN_SEED = 424242
H2_EDGES = [0.0, 0.1, 0.3, 1.0, 3.0, float("inf")]
H2_LABELS = ["0-0.1", "0.1-0.3", "0.3-1", "1-3", "3-inf"]
SCHEMES = [("pwhite", s) for s in RUNS] + [("rand", 1), ("oma", 0)]


def noise_seed(snr, i_hold, draw):
    return 777 + 10 * snr + 1000 * i_hold + 10000 * draw


def gain_seed(i_hold, draw):
    return GAIN_SEED + 1000 * i_hold + 10000 * draw


def load(name):
    sd = torch.load(pm.MODELS / f"{name}.pt", map_location="cpu")
    m = pm.Masks(sd["E"], sd["Wm"].shape[1], sd["Wm"].shape[0])
    m.load_state_dict(sd)
    return m.to(DEVICE).eval()


# ----------------------------------------------------------------------
# channel with given gains, and the decoders of exp_pm with either channel
# ----------------------------------------------------------------------
def channel_held(x, snr, g):
    """rev_channel.channel for flat fading with exact gains, the gains given:
    g (n,U) complex. Returns r (n,U,d) and the weights |g|^2 (n,U,d)."""
    n, d = x.shape
    K = d // 2
    xc = torch.complex(x[:, 0::2], x[:, 1::2])
    sig = math.sqrt(1.0 / (d * 10 ** (snr / 10)))
    w = torch.complex(torch.randn(n, U, K, device=x.device),
                      torch.randn(n, U, K, device=x.device)) / math.sqrt(2) * math.sqrt(2.0) * sig
    gk = g[:, :, None]
    q = (gk * xc[:, None, :] + w) / gk
    r = torch.stack([q.real, q.imag], dim=-1).reshape(n, U, d)
    wt = (gk.abs() ** 2).expand(n, U, K).repeat_interleave(2, dim=-1)
    return r, wt


def held_gains(N, hold, items, seed):
    """(N,U) CN(0,1) gains on the CPU, held over blocks of int(hold) frames
    (random offset per user) or over each article of the user (hold =
    'article'; the frames after its last graded article form one more block)."""
    gen = torch.Generator().manual_seed(seed)
    G = torch.empty(N, U, dtype=torch.complex64)
    for u in range(U):
        if hold == "article":
            blk = torch.empty(N, dtype=torch.long)
            for j, (a, b, _) in enumerate(items[u]):
                blk[a:b] = j
            blk[items[u][-1][1]:] = len(items[u])
            nb = len(items[u]) + 1
        else:
            T = int(hold)
            off = int(torch.randint(T, (1,), generator=gen))
            blk = (torch.arange(N) + off) // T
            nb = int(blk.max()) + 1
        gb = torch.complex(torch.randn(nb, generator=gen), torch.randn(nb, generator=gen)) / math.sqrt(2)
        G[:, u] = gb[blk]
    return G


@torch.no_grad()
def dec_mask(m, ids, snr, seed, G=None):
    """exp_pm.dec_mask (same chunks and draws) that also returns the gain
    power |g_u|^2 of every frame; G (N,U) holds the gains if given."""
    chunk = max(64, 3072 // U)
    torch.manual_seed(seed)
    out = torch.empty_like(ids)
    h2 = torch.empty(ids.shape, dtype=torch.float32)
    for n0 in range(0, ids.shape[0], chunk):
        idx = ids[n0:n0 + chunk].to(DEVICE)
        x = m.tx(idx) / m.c
        if G is None:
            r, w = rc.channel(x, snr, U, **pm.COND["flat"])
        else:
            r, w = channel_held(x, snr, G[n0:n0 + chunk].to(DEVICE))
        out[n0:n0 + chunk] = m.scores(r, w).argmax(-1).cpu()
        h2[n0:n0 + chunk] = w[:, :, 0].cpu()
    return out, h2


@torch.no_grad()
def dec_oma(ids, snr, seed, G=None):
    """exp_pm.dec_oma (one chunk for the whole stream) with a new gain per
    frame; with held gains G it uses the chunks of dec_mask, so that every
    scheme sees the same gains and noise draws."""
    chunk = 100_000 if G is None else max(64, 3072 // U)
    torch.manual_seed(seed)
    n_sym, M, rep = rc.oma_layout(U, D, pm.BITS)
    bps = int(math.log2(M))
    out = torch.empty_like(ids)
    h2 = torch.empty(ids.shape, dtype=torch.float32)
    for n0 in range(0, ids.shape[0], chunk):
        x_ids = ids[n0:n0 + chunk].to(DEVICE)
        sym = torch.stack([(x_ids >> (bps * k)) & (M - 1) for k in range(n_sym)], -1)
        x = rc.oma_frames(sym, U, D, M, rep)
        if G is None:
            r, w = rc.channel(x, snr, U, **pm.COND["flat"])
        else:
            r, w = channel_held(x, snr, G[n0:n0 + chunk].to(DEVICE))
        sh = rc.oma_detect(r, w, U, D, M, rep)
        val = torch.zeros_like(x_ids)
        for k in range(n_sym):
            val = val | (sh[..., k] << (bps * k))
        out[n0:n0 + x_ids.shape[0]] = val.cpu()
        h2[n0:n0 + x_ids.shape[0]] = w[:, :, 0].cpu()
    return out, h2


# ----------------------------------------------------------------------
# graded streams with per-article results
# ----------------------------------------------------------------------
class Streams:
    """exp_pm.Evaluator (U = 4) with per-article results; frames cuts the
    streams to their first frames (smoke test)."""

    def __init__(self, src, sc, frames=None):
        ev = pm.Evaluator(src, U, sc)
        self.src, self.sc, self.clean_acc = src, sc, ev.clean_acc
        keep = [b <= (frames or ev.ids.shape[0]) for u in range(U) for (_, b, _) in ev.items[u]]
        keep = torch.tensor(keep)
        n = frames or ev.ids.shape[0]
        self.ids = ev.ids[:n]
        self.items = [[t for t in ev.items[u] if t[1] <= n] for u in range(U)]
        self.labels = ev.labels[keep]
        self.ref = ev.ref[keep]
        self.N, self.A = self.ids.shape[0], int(keep.sum())
        self.art_u = torch.tensor([u for u in range(U) for _ in self.items[u]])
        self.art_a = torch.tensor([a for u in range(U) for (a, _, _) in self.items[u]])
        self.art_b = torch.tensor([b for u in range(U) for (_, b, _) in self.items[u]])
        self.tokens = (self.art_b - self.art_a).numpy()

    def per_article(self, x):
        """sum of x (N,U) over the frames of every article (A,)"""
        cs = torch.cat([torch.zeros(1, U, dtype=torch.float64), x.double().cumsum(0)])
        return (cs[self.art_b, self.art_u] - cs[self.art_a, self.art_u]).numpy()

    @torch.no_grad()
    def predict(self, texts, bs=256):
        """topic per article, batched as exp_pm.Scorer.acc"""
        pr = []
        for i in range(0, len(texts), bs):
            enc = self.sc.ctok(texts[i:i + bs], padding=True, truncation=True, max_length=128,
                               return_tensors="pt").to(DEVICE)
            pr.append(self.sc.cls(**enc).logits.argmax(-1).cpu())
        return torch.cat(pr)

    def grade(self, dec, h2):
        """the metrics of exp_pm.Evaluator.score, plus per-article results"""
        s = self.src
        bad = (dec < 0) | (dec >= s.V)
        d2 = dec.clone()
        d2[bad] = s.unk
        err = d2 != self.ids
        cos = (s.Wn[self.ids] * s.Wn[d2]).sum(-1)
        txt = s.texts(d2, self.items)
        top = (self.predict(txt) == self.labels).numpy()
        gtr = (self.sc.emb(txt) * self.ref).sum(-1).numpy()
        n_wrong = int((dec != self.ids).sum())
        return dict(ser=float(err.float().mean()), sem_dist=float(1 - cos.mean()),
                    topic_acc=float(top.mean()), gtr_sim=float(gtr.mean()),
                    oov_share=float(bad.sum()) / max(1, n_wrong),
                    err=np.rint(self.per_article(err)).astype(np.int64), top=top, gtr=gtr,
                    h2=self.per_article(h2) / np.maximum(self.tokens, 1))


# ----------------------------------------------------------------------
# summary
# ----------------------------------------------------------------------
HDR = ["kind", "hold", "snr_db", "scheme", "seed", "draw", "h2_bin", "articles", "share",
       "ser", "sem_dist", "topic_acc", "gtr_sim", "oov_share",
       "ser_lo", "ser_hi", "topic_lo", "topic_hi", "gtr_lo", "gtr_hi"]
METRICS = ["ser", "sem_dist", "topic_acc", "gtr_sim", "oov_share"]


def row(**kw):
    return [kw.get(k, "") for k in HDR]


def article_means(R, hold, snr, scheme, draws):
    """per-article errors, topic and GTR similarity averaged over the runs of
    the scheme and the draws"""
    keys = [(hold, snr, d, sc, s) for d in draws for (sc, s) in SCHEMES if sc == scheme]
    err = np.mean([R[k]["err"] for k in keys], 0)
    top = np.mean([R[k]["top"] for k in keys], 0)
    gtr = np.mean([R[k]["gtr"] for k in keys], 0)
    return err, top, gtr


def paired(st, Bi, P, Q):
    """point value and bootstrap interval of P - Q for (err, top, gtr)"""
    (ep, tp, gp), (eq, tq, gq) = P, Q
    tok = st.tokens.astype(np.float64)
    de, dt, dg = ep - eq, tp - tq, gp - gq
    out = dict(ser=float(de.sum() / tok.sum()), topic_acc=float(dt.mean()), gtr_sim=float(dg.mean()))
    bs = dict(ser=de[Bi].sum(1) / tok[Bi].sum(1), topic=dt[Bi].mean(1), gtr=dg[Bi].mean(1))
    for k, v in bs.items():
        lo, hi = np.percentile(v, [2.5, 97.5])
        out[f"{k}_lo"], out[f"{k}_hi"] = float(lo), float(hi)
    return out


def summarize(st, R, holds, snrs, draws):
    rows = [row(kind="clean", hold="-", snr_db="inf", scheme="clean", articles=st.A,
                ser=0.0, sem_dist=0.0, topic_acc=st.clean_acc, gtr_sim=1.0)]
    for (hold, snr, d, sc, s), v in R.items():
        rows.append(row(kind="run", hold=hold, snr_db=snr, scheme=sc, seed=s, draw=d, articles=st.A,
                        **{k: v[k] for k in METRICS}))
    for hold in holds:
        for snr in snrs:
            for sc in ("pwhite", "rand", "oma"):
                ks = [k for k in R if k[0] == hold and k[1] == snr and k[3] == sc]
                rows.append(row(kind="mean", hold=hold, snr_db=snr, scheme=sc, seed="mean", draw="mean",
                                articles=st.A, **{m: float(np.mean([R[k][m] for k in ks])) for m in METRICS}))
    Bi = np.random.default_rng(BOOT_SEED).integers(0, st.A, (BOOT, st.A))
    lead = {}
    for hold in holds:
        for snr in snrs:
            P = article_means(R, hold, snr, "pwhite", draws)
            for ref in ("oma", "rand"):
                Q = article_means(R, hold, snr, ref, draws)
                lead[(hold, snr, ref)] = (P, Q)
                rows.append(row(kind="diff", hold=hold, snr_db=snr, scheme=f"pwhite-{ref}", seed="mean",
                                draw="mean", articles=st.A, **paired(st, Bi, P, Q)))
                print("[diff]", hold, snr, ref, {k: round(v, 4) for k, v in paired(st, Bi, P, Q).items()},
                      flush=True)
    if "1" in holds:
        for hold in holds[1:]:
            for snr in snrs:
                for ref in ("oma", "rand"):
                    (P1, Q1), (Ph, Qh) = lead[("1", snr, ref)], lead[(hold, snr, ref)]
                    # (Ph - Qh) - (P1 - Q1): the change of the lead, paired over articles
                    Lh = tuple(ph - qh for ph, qh in zip(Ph, Qh))
                    L1 = tuple(p1 - q1 for p1, q1 in zip(P1, Q1))
                    rows.append(row(kind="widen", hold=hold, snr_db=snr, scheme=f"pwhite-{ref}", seed="mean",
                                    draw="mean", articles=st.A, **paired(st, Bi, Lh, L1)))
    tok = st.tokens.astype(np.float64)
    for hold in [h for h in holds if h != "1"]:
        for snr in snrs:
            h2 = np.concatenate([R[(hold, snr, d, "oma", 0)]["h2"] for d in draws])
            tk = np.concatenate([tok for _ in draws])
            for sc in ("pwhite", "rand", "oma"):
                runs = [s for (c, s) in SCHEMES if c == sc]
                cat = {f: np.concatenate([np.mean([R[(hold, snr, d, sc, s)][f] for s in runs], 0) for d in draws])
                       for f in ("err", "top", "gtr")}
                for j, lab in enumerate(H2_LABELS):
                    sel = (h2 >= H2_EDGES[j]) & (h2 < H2_EDGES[j + 1])
                    if not sel.any():
                        continue
                    rows.append(row(kind="h2bin", hold=hold, snr_db=snr, scheme=sc, seed="mean", draw="pooled",
                                    h2_bin=lab, articles=int(sel.sum()), share=float(sel.mean()),
                                    ser=float(cat["err"][sel].sum() / tk[sel].sum()),
                                    topic_acc=float(cat["top"][sel].mean()), gtr_sim=float(cat["gtr"][sel].mean())))
    return rows


def check_rows(R, snrs):
    """hold 1, draw 0 against pm_models_eval.csv (same seeds 777 + 10 snr)"""
    path = pm.DATA / "pm_models_eval.csv"
    if not path.exists():
        return []
    ref = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["part"] == "models" and r["source"] == "bert" and r["codebook"] in ("pwhite", "rand", "oma"):
                ref[(r["codebook"], int(r["seed"]), int(float(r["snr_db"])))] = r
    rows, worst = [], 0.0
    for snr in snrs:
        for sc, s in SCHEMES:
            k = (sc, s, snr)
            if k not in ref or ("1", snr, 0, sc, s) not in R:
                continue
            v = R[("1", snr, 0, sc, s)]
            dif = dict(ser=v["ser"] - float(ref[k]["ter"]), sem_dist=v["sem_dist"] - float(ref[k]["sem_dist"]),
                       topic_acc=v["topic_acc"] - float(ref[k]["topic_acc"]),
                       gtr_sim=v["gtr_sim"] - float(ref[k]["gtr_sim"]))
            worst = max(worst, max(abs(x) for x in dif.values()))
            rows.append(row(kind="check", hold="1", snr_db=snr, scheme=sc, seed=s, draw=0, **dif))
    print(f"[check] hold 1, draw 0 against pm_models_eval.csv: {len(rows)} rows, largest |difference| "
          f"{worst:.2e}", flush=True)
    return rows


# ----------------------------------------------------------------------
# part
# ----------------------------------------------------------------------
def retry_oom(fn, tries=60, wait=30):
    """Run fn again after a CUDA out-of-memory error on a shared GPU. The
    decoders reseed on entry and the grading is deterministic, so a retry
    repeats the same draws."""
    for k in range(tries):
        try:
            return fn()
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"[oom] retry {k + 1} in {wait} s", flush=True)
            time.sleep(wait)
    return fn()


def part_block(a):
    torch.backends.cuda.matmul.allow_tf32 = False          # evaluation in full fp32, as exp_pm
    t0 = time.time()
    holds, snrs, draws = HOLDS, SNRS, list(range(DRAWS))
    frames, tag = None, ""
    if a.smoke:
        snrs, frames, tag = [10], 2048, "_smoke"
    src = pm.Source("bert")
    sc = pm.Scorer()
    st = Streams(src, sc, frames)
    print(f"[streams] frames {st.N} graded articles {st.A} clean topic {st.clean_acc:.4f} "
          f"({time.time() - t0:.0f} s)", flush=True)
    models = {(c, s): load(f"bert_{c}_U4_L16_flat_s{s}") for (c, s) in SCHEMES if c != "oma"}
    R = {}
    for ih, hold in enumerate(HOLDS):
        if hold not in holds:
            continue
        for d in draws:
            G = None if hold == "1" else held_gains(st.N, hold, st.items, gain_seed(ih, d))
            for snr in snrs:
                seed = noise_seed(snr, ih, d)
                h2_mask = None
                for c, s in SCHEMES:
                    if c == "oma":
                        dec, h2 = retry_oom(lambda: dec_oma(st.ids, snr, seed, G))
                    else:
                        dec, h2 = retry_oom(lambda: dec_mask(models[(c, s)], st.ids, snr, seed, G))
                        if h2_mask is None:
                            h2_mask = h2
                        elif not torch.equal(h2, h2_mask):
                            print("[warn] mask decoders saw different gains", hold, snr, d, c, s, flush=True)
                    R[(hold, snr, d, c, s)] = v = retry_oom(lambda: st.grade(dec, h2))
                    print(f"[run] hold {hold} snr {snr} draw {d} {c} s{s} ser {v['ser']:.4f} "
                          f"topic {v['topic_acc']:.4f} gtr {v['gtr_sim']:.4f} oov {v['oov_share']:.3f} "
                          f"mean h2 {v['h2'].mean():.3f} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    # per-article file
    arows = []
    for hold in holds:
        for snr in snrs:
            for d in draws:
                h2 = R[(hold, snr, d, "pwhite", 1)]["h2"]
                h2o = R[(hold, snr, d, "oma", 0)]["h2"]
                cols = []
                for c in ("pwhite", "rand", "oma"):
                    ks = [(hold, snr, d, c, s) for (cc, s) in SCHEMES if cc == c]
                    cols.append((np.sum([R[k]["err"] for k in ks], 0), np.sum([R[k]["top"] for k in ks], 0),
                                 np.mean([R[k]["gtr"] for k in ks], 0)))
                for i in range(st.A):
                    r = [hold, snr, d, i, int(st.art_u[i]), int(st.tokens[i]), f"{h2[i]:.4g}",
                         f"{h2o[i]:.4g}" if hold == "1" else ""]
                    for e, t, g in cols:
                        r += [int(e[i]), int(t[i]), f"{g[i]:.3f}"]
                    arows.append(r)
    write_csv(pm.DATA / f"pm_block_articles{tag}.csv",
              ["hold", "snr_db", "draw", "article", "user", "tokens", "h2", "h2_oma",
               "err_pwhite", "top_pwhite", "gtr_pwhite", "err_rand", "top_rand", "gtr_rand",
               "err_oma", "top_oma", "gtr_oma"], arows)
    rows = summarize(st, R, holds, snrs, draws) + check_rows(R, snrs)
    write_csv(pm.DATA / f"pm_block_eval{tag}.csv", HDR, rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["block"])
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    {"block": part_block}[a.part](a)
    print("[done]", a.part, round((time.time() - t0) / 60, 1), "min", flush=True)


if __name__ == "__main__":
    main()
