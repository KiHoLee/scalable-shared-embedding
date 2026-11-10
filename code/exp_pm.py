"""Periodic user masks for semantic shared-embedding multiple access.

The source codebook is the input-embedding table of a pretrained language
model (BERT or GTR-T5), held at the transmitter and every receiver. It is
reduced to the d = 64 frame dimensions by a whitened principal projection
and a fixed random rotation, normalized to unit norm, and never trained.
Only the user masks are trained, separately for each channel class and
each load (number of users). A periodic mask stores a pattern of length L
and repeats it d/L times; L = d is the full-length mask.

Parts (--part):
  models   source codebooks BERT and GTR-T5 (pwhite with seeds 1-3, uniform
           whitening, raw projection) and a random codebook, U = 4 flat,
           L = 16 (supplementary figure fig/fig_codebook_pm.pdf)
  period   mask length L in {64,...,1} for U = 2, 4, 8 (flat fading), with
           seeds 1-3 at U = 4, load mismatch (masks of one load used at
           another), training curves
  channel  eight channel classes at U = 4: masks trained for each class
           (L = 16 and L = 64), the flat-trained L = 16 masks applied
           unchanged, a random codebook trained for each class, and every
           class-trained L = 16 mask applied in every class (cross_class)
  plen     pattern length against channel class (and U = 8 flat), with the
           isotropy statistics of the codebook
  bound    union bounds on the SER and on the semantic distortion against
           Monte Carlo, U = 4 flat, L = 16 and L = 64 (--bound_nv 400)
  bound2   the bounds conditioned on the gain and clipped (run with
           --bound_nv 600)
  bound3   the bounds at a fixed unit gain (run with --bound_nv 600)
  comb     fixed patterns without overlap (L = U) and with pairs (L = 2U)
  energy   measured frame energy of every saved mask set on the test streams
  pwhite   the source-statistics codebook alone (exploratory)
  all      models, period, channel, plen, bound in one process
  rest     channel, plen, bound in one process

Every scheme uses the channel simulator rev_channel.channel, one token per
user and frame, d = 64 real dimensions, unit average transmit energy, and
weighted minimum-distance decoding over the whole vocabulary. OMA sends the
16 bits of the token id on d/U real dimensions per user (BPSK, repetition
for U = 2, 4-PAM for U = 8); its frame energy is one at U = 2 and 4 and
above one at U = 8 (exp_pm_oma_energy.py).

Outputs in data/pm/: pm_<part>_eval.csv, pm_<part>_train.json, pm_bound*.csv,
pm_comb*.{csv,json}, pm_energy.json. Trained masks go to data/pm/pm_models/
(not released; each file stores the fixed codebook as well), smoke runs to
data/pm/pm_models_smoke/. In the period, channel and plen outputs the
codebook label "white" (model names "white_*", "bertw_*") denotes the
source-statistics codebook pwhite; in pm_models_eval.csv "white" denotes
uniform whitening and "pwhite" the source-statistics codebook.
"""
from __future__ import annotations

import argparse
import json
import math
import time

import torch

import rev_channel as rc
from sse_lib import DATA, DEVICE, D, set_seed, write_csv

N_TEXTS = 2000
BITS = 16
DATA = DATA / "pm"            # every output of this study goes to data/pm/
DATA.mkdir(exist_ok=True)
MODELS = DATA / "pm_models"
MODELS.mkdir(exist_ok=True)
COND = {
    "flat": dict(taps=1, nmse=0.0, offsets_db=None),
    "off6": dict(taps=1, nmse=0.0, offsets_db=[-6.0, -2.0, 2.0, 6.0]),
    "off12": dict(taps=1, nmse=0.0, offsets_db=[-12.0, -4.0, 4.0, 12.0]),
    "nmse01": dict(taps=1, nmse=0.01, offsets_db=None),
    "nmse05": dict(taps=1, nmse=0.05, offsets_db=None),
    "nmse10": dict(taps=1, nmse=0.1, offsets_db=None),
    "fs4": dict(taps=4, nmse=0.0, offsets_db=None),
    "fs8": dict(taps=8, nmse=0.0, offsets_db=None),
}


def cond_for(name: str, users: int) -> dict:
    c = dict(COND[name])
    if c["offsets_db"] is not None and users != 4:
        dl = c["offsets_db"][-1]
        c["offsets_db"] = [(-dl + 2 * dl * k / (users - 1)) for k in range(users)]
    return c


# ----------------------------------------------------------------------
# sources: tokenizers, embedding tables, streams, statistics
# ----------------------------------------------------------------------
class Source:
    def __init__(self, kind: str):
        from transformers import AutoTokenizer
        self.kind = kind
        if kind == "bert":
            from transformers import AutoModel
            self.tok = AutoTokenizer.from_pretrained("bert-base-uncased")
            m = AutoModel.from_pretrained("bert-base-uncased")
            W = m.embeddings.word_embeddings.weight.detach().float().cpu()
            self.unk = self.tok.unk_token_id
        elif kind == "gtr":
            from sentence_transformers import SentenceTransformer
            st = SentenceTransformer("sentence-transformers/gtr-t5-base", device="cpu")
            self.tok = st.tokenizer
            W = st[0].auto_model.shared.weight.detach().float().cpu()
            self.unk = self.tok.unk_token_id
        else:
            raise ValueError(kind)
        self.V = len(self.tok)
        self.W = W[: self.V]                                   # (V,768)
        self.Wn = torch.nn.functional.normalize(self.W, dim=1)

    def streams(self, users: int):
        from datasets import load_dataset
        ds = load_dataset("fancyzhx/ag_news", split="test")
        st = [[] for _ in range(users)]
        items = [[] for _ in range(users)]
        for i in range(N_TEXTS):
            ids = self.tok(ds[i]["text"], add_special_tokens=False)["input_ids"]
            u = i % users
            s = len(st[u])
            st[u].extend(ids)
            items[u].append((s, s + len(ids), ds[i]["label"]))
        n = min(len(s) for s in st)
        st = [s[:n] for s in st]
        items = [[t for t in iu if t[1] <= n] for iu in items]
        return torch.tensor(list(zip(*st)), dtype=torch.long), items

    def dist(self, n_items: int = 40000) -> torch.Tensor:
        from datasets import load_dataset
        ds = load_dataset("fancyzhx/ag_news", split="train")
        enc = self.tok(ds[:n_items]["text"], add_special_tokens=False)["input_ids"]
        cnt = torch.zeros(self.V)
        for ids in enc:
            cnt += torch.bincount(torch.tensor(ids, dtype=torch.long), minlength=self.V).float()
        return cnt / cnt.sum()

    def texts(self, ids: torch.Tensor, items) -> list[str]:
        ids = ids.clone()
        ids[(ids < 0) | (ids >= self.V)] = self.unk
        out = []
        for u in range(len(items)):
            for (a, b, _) in items[u]:
                out.append(self.tok.decode(ids[a:b, u].tolist(), skip_special_tokens=True))
        return out


def codebook(W: torch.Tensor, d: int, kind: str, seed: int = 0, pd=None) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    if kind == "rand":
        Z = torch.randn(W.shape[0], d, generator=g)
    elif kind == "pwhite":
        # whitening under the source statistics: mean and covariance of the
        # token embeddings weighted by the token frequencies pd
        p = pd.double()
        W64 = W.double()
        mu = (p[:, None] * W64).sum(0)
        Wc = W64 - mu
        C = (Wc * p[:, None]).T @ Wc
        lam, Vec = torch.linalg.eigh(C)
        lam, Vec = lam.flip(0)[:d], Vec.flip(1)[:, :d]
        Z = ((Wc @ Vec) / lam.sqrt()).float()
    else:
        Wc = W - W.mean(0, keepdim=True)
        _, S, Vh = torch.linalg.svd(Wc, full_matrices=False)
        Z = Wc @ Vh[:d].T
        if kind == "white":
            Z = Z / S[:d]
        elif kind != "raw":
            raise ValueError(kind)
    Q, _ = torch.linalg.qr(torch.randn(d, d, generator=g))
    Z = Z @ Q
    return Z / Z.norm(dim=1, keepdim=True).clamp_min(1e-8)


# ----------------------------------------------------------------------
# scoring with public models that never see the channel
# ----------------------------------------------------------------------
class Scorer:
    CLS = "textattack/bert-base-uncased-ag-news"

    def __init__(self):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        from sentence_transformers import SentenceTransformer
        self.ctok = AutoTokenizer.from_pretrained(self.CLS)
        self.cls = AutoModelForSequenceClassification.from_pretrained(self.CLS).to(DEVICE).eval()
        self.gtr = SentenceTransformer("sentence-transformers/gtr-t5-base", device=str(DEVICE))
        self.ref = {}

    @torch.no_grad()
    def acc(self, texts, labels, bs=256):
        pr = []
        for i in range(0, len(texts), bs):
            enc = self.ctok(texts[i:i + bs], padding=True, truncation=True, max_length=128,
                            return_tensors="pt").to(DEVICE)
            pr.append(self.cls(**enc).logits.argmax(-1).cpu())
        return float((torch.cat(pr) == labels).float().mean())

    @torch.no_grad()
    def emb(self, texts):
        return self.gtr.encode(texts, batch_size=256, convert_to_tensor=True,
                               normalize_embeddings=True, show_progress_bar=False).float().cpu()


# ----------------------------------------------------------------------
# masks
# ----------------------------------------------------------------------
class Masks(torch.nn.Module):
    """Fixed unit-norm codebook E (V,d); trained periodic masks of length L."""

    def __init__(self, E: torch.Tensor, L: int, users: int):
        super().__init__()
        V, d = E.shape
        assert d % L == 0
        self.register_buffer("E", E.clone())
        self.L, self.P, self.d, self.users = L, d // L, d, users
        self.Wm = torch.nn.Parameter(torch.randn(users, L) / math.sqrt(L))
        self.logit_scale = torch.nn.Parameter(torch.tensor(2.0))
        self.register_buffer("c", torch.tensor(1.0))

    def masks(self) -> torch.Tensor:
        mu = self.Wm / self.Wm.norm(dim=1, keepdim=True).clamp_min(1e-8) * math.sqrt(self.L)
        return mu.repeat(1, self.P)

    def tx(self, idx):
        return (self.E[idx] * self.masks()[: idx.shape[1]][None]).sum(1)

    @torch.no_grad()
    def calibrate(self, pd, n=65536, users=None):
        users = users or self.users
        idx = torch.multinomial(pd, n * users, replacement=True).view(n, users).to(self.E.device)
        self.c.fill_(float(self.tx(idx).pow(2).sum(1).mean().sqrt()))

    def scores(self, r, w):
        m = self.masks()[: r.shape[1]]
        cand = self.E[None] * m[:, None, :]
        a = 1.0 / float(self.c)
        wc = torch.einsum("nud,uvd->nuv", w * r, cand)
        wn = torch.einsum("nud,uvd->nuv", w, cand.pow(2))
        return 2 * a * wc - a * a * wn


@torch.no_grad()
def quick_ter(model, pd, cond, users, snr=10.0, n=4096, seed=4242):
    tf = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    g = torch.Generator(device="cpu").manual_seed(seed)
    idx = torch.multinomial(pd.cpu(), n * users, replacement=True, generator=g).view(n, users).to(DEVICE)
    torch.manual_seed(seed)
    err = 0
    for k in range(0, n, 512):
        x = model.tx(idx[k:k + 512]) / model.c
        r, w = rc.channel(x, snr, users, **cond)
        err += int((model.scores(r, w).argmax(-1) != idx[k:k + 512]).sum())
    torch.backends.cuda.matmul.allow_tf32 = tf
    return err / (n * users)


def train_masks(E, L, users, pd, cond, iters=3000, seed=1, lr=1e-2, batch=256, curve=False):
    torch.backends.cuda.matmul.allow_tf32 = True      # training only; evaluation runs in full fp32
    set_seed(seed)
    m = Masks(E, L, users).to(DEVICE)
    pdd = pd.to(DEVICE)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    ce = torch.nn.CrossEntropyLoss()
    log = []
    t0 = time.time()
    for it in range(1, iters + 1):
        idx = torch.multinomial(pdd, batch * users, replacement=True).view(batch, users)
        snr = torch.empty(batch).uniform_(0.0, 20.0)
        m.calibrate(pdd, 8192)
        x = m.tx(idx) / m.c
        r, w = rc.channel(x, snr, users, **cond)
        sc = m.scores(r, w) * m.logit_scale.exp()
        loss = ce(sc.reshape(-1, sc.shape[-1]), idx.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if curve and (it in (1, 50, 100, 200) or it % 250 == 0):
            m.calibrate(pdd, 16384)
            log.append((it, round(time.time() - t0, 1), float(loss.detach()),
                        quick_ter(m, pdd, cond, users)))
    torch.backends.cuda.matmul.allow_tf32 = False
    m.calibrate(pdd)
    m.eval()
    return m, log, time.time() - t0


# ----------------------------------------------------------------------
# decoders on the real streams
# ----------------------------------------------------------------------
@torch.no_grad()
def dec_mask(model, ids, snr, cond, seed, chunk=None):
    users = ids.shape[1]
    chunk = chunk or max(64, 3072 // users)
    torch.manual_seed(seed)
    out = torch.empty_like(ids)
    for n0 in range(0, ids.shape[0], chunk):
        idx = ids[n0:n0 + chunk].to(DEVICE)
        x = model.tx(idx) / model.c
        r, w = rc.channel(x, snr, users, **cond)
        out[n0:n0 + chunk] = model.scores(r, w).argmax(-1).cpu()
    return out


@torch.no_grad()
def dec_oma(ids, snr, cond, seed, chunk=100_000):
    users = ids.shape[1]
    torch.manual_seed(seed)
    n_sym, M, rep = rc.oma_layout(users, D, BITS)
    bps = int(math.log2(M))
    out = torch.empty_like(ids)
    for n0 in range(0, ids.shape[0], chunk):
        x_ids = ids[n0:n0 + chunk].to(DEVICE)
        sym = torch.stack([(x_ids >> (bps * k)) & (M - 1) for k in range(n_sym)], -1)
        x = rc.oma_frames(sym, users, D, M, rep)
        r, w = rc.channel(x, snr, users, **cond)
        sh = rc.oma_detect(r, w, users, D, M, rep)
        val = torch.zeros_like(x_ids)
        for k in range(n_sym):
            val = val | (sh[..., k] << (bps * k))
        out[n0:n0 + x_ids.shape[0]] = val.cpu()
    return out


class Evaluator:
    def __init__(self, src: Source, users: int, scorer: Scorer):
        self.src, self.users, self.sc = src, users, scorer
        self.ids, self.items = src.streams(users)
        self.labels = torch.tensor([t[2] for u in range(users) for t in self.items[u]])
        key = (src.kind, users)
        if key not in scorer.ref:
            scorer.ref[key] = scorer.emb(src.texts(self.ids, self.items))
        self.ref = scorer.ref[key]
        self.clean_acc = scorer.acc(src.texts(self.ids, self.items), self.labels)

    def score(self, dec):
        s = self.src
        d2 = dec.clone()
        d2[(d2 < 0) | (d2 >= s.V)] = s.unk
        err = d2 != self.ids
        cos = (s.Wn[self.ids] * s.Wn[d2]).sum(-1)
        txt = s.texts(d2, self.items)
        acc = self.sc.acc(txt, self.labels)
        gsim = float((self.sc.emb(txt) * self.ref).sum(-1).mean())
        e_cos = float(cos[err].mean()) if err.any() else float("nan")
        e_close = float((cos[err] > 0.4).float().mean()) if err.any() else float("nan")
        return [float(err.float().mean()), float(1 - cos.mean()), e_cos, e_close, acc, gsim]


HDR = ["part", "source", "codebook", "users", "L", "train_cond", "eval_cond", "seed",
       "snr_db", "ter", "sem_dist", "err_tok_sim", "err_close", "topic_acc", "gtr_sim"]


def log_row(rows, row):
    rows.append(row)
    print("[eval]", " ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in row), flush=True)


# ----------------------------------------------------------------------
# parts
# ----------------------------------------------------------------------
def part_period(a):
    src = Source("bert")
    pd = src.dist()
    E = codebook(src.W, D, a.book, pd=pd)
    sc = Scorer()
    rows, tlog = [], {}
    grid = [0, 5, 10, 15, 20]
    plan = []
    spec = ((4, [64, 32, 16, 8, 4, 2, 1], [1, 2, 3]),
            (8, [64, 32, 16, 8, 4, 2], [1]),
            (2, [64, 32, 16, 8, 4, 2, 1], [1]))
    if a.smoke:
        spec, grid = ((4, [64, 16], [1]), (8, [16], [1])), [10]
    for users, Ls, seeds in spec:
        for L in Ls:
            for s in seeds:
                plan.append((users, L, s))
    trained = {}
    for users, L, s in plan:
        name = f"bertw_U{users}_L{L}_flat_s{s}"
        m, lg, tt = train_masks(E, L, users, pd, COND["flat"], a.iters, seed=s,
                                curve=(users == 4 and s == 1))
        torch.save(m.state_dict(), MODELS / f"{name}.pt")
        tlog[name] = dict(users=users, L=L, seed=s, train_s=round(tt, 1), curve=lg,
                          ter10=quick_ter(m, pd.to(DEVICE), COND["flat"], users))
        print("[train]", name, tlog[name]["ter10"], flush=True)
        trained[(users, L, s)] = m
    for users in sorted({sp[0] for sp in spec}):
        ev = Evaluator(src, users, sc)
        print(f"[clean] U={users} acc {ev.clean_acc:.4f}", flush=True)
        for snr in grid:
            sd = 777 + int(10 * snr)
            log_row(rows, ["period", "bert", "oma", users, 0, "-", "flat", 0, snr]
                    + ev.score(dec_oma(ev.ids, snr, COND["flat"], sd)))
            for (uu, L, s), m in trained.items():
                if uu != users:
                    continue
                log_row(rows, ["period", "bert", "white", users, L, "flat", "flat", s, snr]
                        + ev.score(dec_mask(m, ev.ids, snr, COND["flat"], sd)))
            # load mismatch: masks trained for another load serve this load
            for (uu, L, s), m in trained.items():
                if s != 1 or L not in (64, 16) or uu == users or uu < users:
                    continue
                m.calibrate(pd.to(DEVICE), users=users)
                log_row(rows, ["period", "bert", "white", users, L, f"U{uu}", "flat", s, snr]
                        + ev.score(dec_mask(m, ev.ids, snr, COND["flat"], sd)))
                m.calibrate(pd.to(DEVICE))
    write_csv(DATA / f"pm_period_eval{a.tag}.csv", HDR, rows)
    json.dump(tlog, open(DATA / f"pm_period_train{a.tag}.json", "w"), indent=1)


def part_channel(a):
    src = Source("bert")
    pd = src.dist()
    Ew = codebook(src.W, D, a.book, pd=pd)
    Er = codebook(src.W, D, "rand")
    sc = Scorer()
    users = 4
    rows, tlog, models = [], {}, {}
    conds = ["flat", "fs4"] if a.smoke else list(COND)
    for cn in conds:
        for tag, E, L in (("white", Ew, 16), ("white", Ew, 64), ("rand", Er, 16)):
            name = f"{tag}_U4_L{L}_{cn}_s1"
            m, lg, tt = train_masks(E, L, users, pd, COND[cn], a.iters, seed=1)
            torch.save(m.state_dict(), MODELS / f"{name}.pt")
            tlog[name] = dict(L=L, cond=cn, train_s=round(tt, 1),
                              ter10=quick_ter(m, pd.to(DEVICE), COND[cn], users))
            print("[train]", name, tlog[name]["ter10"], flush=True)
            models[(tag, L, cn)] = m
    ev = Evaluator(src, users, sc)
    for cn in conds:
        for snr in ([10] if a.smoke else [0, 5, 10, 15, 20]):
            sd = 777 + int(10 * snr)
            log_row(rows, ["channel", "bert", "oma", users, 0, "-", cn, 0, snr]
                    + ev.score(dec_oma(ev.ids, snr, COND[cn], sd)))
            for (tag, L, tc), m in models.items():
                if tc == cn or (tc == "flat" and tag == "white" and L == 16):
                    log_row(rows, ["channel", "bert", tag, users, L, tc, cn, 1, snr]
                            + ev.score(dec_mask(m, ev.ids, snr, COND[cn], sd)))
    # cross-class check: every class-trained L=16 mask applied in every class at 10 dB
    for cn in conds:
        for tc in conds:
            m = models[("white", 16, tc)]
            log_row(rows, ["cross_class", "bert", "white", users, 16, tc, cn, 1, 10]
                    + ev.score(dec_mask(m, ev.ids, 10, COND[cn], 877)))
    write_csv(DATA / f"pm_channel_eval{a.tag}.csv", HDR, rows)
    json.dump(tlog, open(DATA / f"pm_channel_train{a.tag}.json", "w"), indent=1)


def part_models(a):
    sc = Scorer()
    rows, tlog = [], {}
    users = 4
    for kind in (("gtr",) if a.smoke else ("bert", "gtr")):
        src = Source(kind)
        pd = src.dist()
        ev = Evaluator(src, users, sc)
        print(f"[clean] {kind} V={src.V} acc {ev.clean_acc:.4f}", flush=True)
        books = {"white": codebook(src.W, D, "white"), "pwhite": codebook(src.W, D, "pwhite", pd=pd),
                 "raw": codebook(src.W, D, "raw"), "rand": codebook(src.W, D, "rand")}
        ms = {}
        for bk, E in books.items():
            for s in ((1, 2, 3) if (bk == a.book and not a.smoke) else (1,)):
                name = f"{kind}_{bk}_U4_L16_flat_s{s}"
                m, lg, tt = train_masks(E, 16, users, pd, COND["flat"], a.iters, seed=s)
                torch.save(m.state_dict(), MODELS / f"{name}.pt")
                tlog[name] = dict(train_s=round(tt, 1), ter10=quick_ter(m, pd.to(DEVICE), COND["flat"], users))
                print("[train]", name, tlog[name]["ter10"], flush=True)
                ms[(bk, s)] = m
        for snr in ([10] if a.smoke else range(0, 21, 2)):
            sd = 777 + int(10 * snr)
            log_row(rows, ["models", kind, "oma", users, 0, "-", "flat", 0, snr]
                    + ev.score(dec_oma(ev.ids, snr, COND["flat"], sd)))
            for (bk, s), m in ms.items():
                log_row(rows, ["models", kind, bk, users, 16, "flat", "flat", s, snr]
                        + ev.score(dec_mask(m, ev.ids, snr, COND["flat"], sd)))
        tlog[f"{kind}_clean_acc"] = ev.clean_acc
        tlog[f"{kind}_V"] = src.V
    write_csv(DATA / f"pm_models_eval{a.tag}.csv", HDR, rows)
    json.dump(tlog, open(DATA / f"pm_models_train{a.tag}.json", "w"), indent=1)


# ---------------- union bounds -------------------------------------------
def part_bound(a):
    """Gaussian-interference union bounds for user u under flat Rayleigh
    fading and weighted minimum-distance decoding:
      P(v->w | h) ~ Q( (a||D||^2/2 + D.mu) / sqrt(||D||^2 s2/h^2 + D^T K D) ),
    D = (e_v - e_w).m_u, a = 1/c, mu and K the mean and covariance of the
    other users' masked signal. SER <= sum_v p(v) sum_w E_h[P];
    semantic distortion E[1 - s(v, v_hat)] <= sum_v p(v) sum_w (1 - s_vw) E_h[P].
    """
    src = Source("bert")
    pdc = src.dist()
    E = codebook(src.W, D, a.book, pd=pdc).to(DEVICE)
    pd = pdc.to(DEVICE)
    users = 4
    Wn = src.Wn.to(DEVICE)
    mu_e = (pd[:, None] * E).sum(0)
    Ce = (E * pd[:, None]).T @ E - torch.outer(mu_e, mu_e)          # (d,d)
    # Gauss-Laguerre nodes for h^2 ~ Exp(1)
    import numpy as np
    xn, wn = np.polynomial.laguerre.laggauss(48)
    xg = torch.tensor(xn, dtype=torch.float32, device=DEVICE)
    wg = torch.tensor(wn, dtype=torch.float32, device=DEVICE)
    rows = []
    n_v = a.bound_nv
    g = torch.Generator(device="cpu").manual_seed(5)
    vs_all = torch.multinomial(pd.cpu(), n_v, replacement=True, generator=g).to(DEVICE)
    snrs = [10, 20] if a.smoke else list(range(0, 21, 2))
    for L in (16, 64):
        with torch.enable_grad():
            m, _, _ = train_masks(E.cpu(), L, users, pd.cpu(), COND["flat"], a.iters, seed=1)
        torch.set_grad_enabled(False)
        M = m.masks().detach()
        ac = 1.0 / float(m.c)
        ub_ser = torch.zeros(users, len(snrs), device=DEVICE)
        ub_sem = torch.zeros(users, len(snrs), device=DEVICE)
        for u in range(users):
            others = [k for k in range(users) if k != u]
            mu_I = ac * sum(M[k] * mu_e for k in others)                  # (d,)
            for v in vs_all.tolist():
                Dl = (E[v][None] - E) * M[u][None]                         # (V,d)
                Dl[v] = 0
                n2 = (Dl * Dl).sum(1)                                       # (V,)
                mean = Dl @ mu_I
                var_i = torch.zeros_like(n2)
                for k in others:
                    Z = Dl * M[k][None]
                    var_i += ac * ac * ((Z @ Ce) * Z).sum(1)
                semw = (1 - (Wn[v][None] * Wn).sum(1)).clamp_min(0)
                ok = n2 > 0
                for j, snr in enumerate(snrs):
                    s2 = 1.0 / (D * 10 ** (snr / 10))
                    num = (ac * n2 / 2 + mean)[ok]                          # (V',)
                    den = torch.sqrt(n2[ok][None] * s2 / xg[:, None] + var_i[ok][None])
                    q = 0.5 * torch.erfc(num[None] / den / math.sqrt(2))   # (G,V')
                    pe = (wg[:, None] * q).sum(0)
                    ub_ser[u, j] += pe.sum() / n_v
                    ub_sem[u, j] += (pe * semw[ok]).sum() / n_v
        # Monte Carlo on source-distribution tokens
        frames = a.bound_mc
        for j, snr in enumerate(snrs):
            torch.manual_seed(900 + snr)
            err = torch.zeros(users, device=DEVICE)
            sem = torch.zeros(users, device=DEVICE)
            for k in range(0, frames, 2048):
                n = min(2048, frames - k)
                idx = torch.multinomial(pd, n * users, replacement=True).view(n, users)
                x = m.tx(idx) / m.c
                r, w = rc.channel(x, snr, users, **COND["flat"])
                dec = m.scores(r, w).argmax(-1)
                e = dec != idx
                err += e.float().sum(0)
                sem += (1 - (Wn[idx] * Wn[dec]).sum(-1)).sum(0)
            for u in range(users):
                rows.append([L, u, snr, float(ub_ser[u, j]), float(err[u] / frames),
                             float(ub_sem[u, j]), float(sem[u] / frames)])
            print("[bound]", L, snr, f"ub_ser {float(ub_ser[:, j].mean()):.4f} ser "
                  f"{float(err.mean() / frames):.4f} ub_sem {float(ub_sem[:, j].mean()):.4f} "
                  f"sem {float(sem.mean() / frames):.4f}", flush=True)
    write_csv(DATA / f"pm_bound{a.tag}.csv", ["L", "user", "snr_db", "ub_ser", "mc_ser", "ub_sem", "mc_sem"], rows)



def cross_energy(M: torch.Tensor, pair: bool = False) -> list[float]:
    """iota_u = (1/d) sum_i m_{u,i}^2 sum_{k!=u} m_{k,i}^2; with pair=True the
    other users' energy on the partner dimension of the same subcarrier
    (2l <-> 2l+1) is counted as well."""
    E2 = M.pow(2)
    tot = E2.sum(0, keepdim=True) - E2
    out = (E2 * tot).mean(1)
    if pair:
        sw = E2.view(E2.shape[0], -1, 2).flip(-1).reshape(E2.shape)
        tsw = sw.sum(0, keepdim=True) - sw
        out = (E2 * tsw).mean(1)
    return [float(x) for x in out]


def part_plen(a):
    """Pattern length L against channel class at U = 4 (and U = 8 flat),
    with the masks' cross-energies and codebook isotropy statistics."""
    src = Source("bert")
    pd = src.dist()
    books = {k: codebook(src.W, D, k, pd=pd) for k in ("white", "pwhite", "raw", "rand")}
    stats = {}
    for k, E in books.items():
        mu = (pd[:, None] * E).sum(0)
        C = (E * pd[:, None]).T @ E - torch.outer(mu, mu)
        ev = torch.linalg.eigvalsh(C)
        stats[k] = dict(mean_norm=float(mu.norm()), eig_max_over_mean=float(ev.max() / ev.mean()),
                        eig_min_over_mean=float(ev.min() / ev.mean()), trace=float(ev.sum()))
        print("[stats]", k, stats[k], flush=True)
    E = books[a.book]
    sc = Scorer()
    rows, tlog = [], {"isotropy": stats}
    ev4 = Evaluator(src, 4, sc)
    conds = ["flat", "off12", "nmse05", "fs4"] if not a.smoke else ["flat"]
    Ls = [64, 16, 8, 4, 2] if not a.smoke else [8]
    for cn in conds:
        for L in Ls:
            name = f"white_U4_L{L}_{cn}_s1"
            m, lg, tt = train_masks(E, L, 4, pd, COND[cn], a.iters, seed=1)
            M = m.masks().detach().cpu()
            tlog[name] = dict(L=L, cond=cn, iota=cross_energy(M), iota_pair=cross_energy(M, True),
                              ter10=quick_ter(m, pd.to(DEVICE), COND[cn], 4, snr=10.0, n=8192),
                              ter20=quick_ter(m, pd.to(DEVICE), COND[cn], 4, snr=20.0, n=8192),
                              pattern=[[round(float(x), 4) for x in row] for row in M[:, :L]])
            print("[train]", name, tlog[name]["ter10"], tlog[name]["ter20"],
                  [round(x, 4) for x in tlog[name]["iota"]], flush=True)
            for snr in (10, 20):
                log_row(rows, ["plen", "bert", "white", 4, L, cn, cn, 1, snr]
                        + ev4.score(dec_mask(m, ev4.ids, snr, COND[cn], 777 + 10 * snr)))
    write_csv(DATA / f"pm_plen_eval{a.tag}.csv", HDR, rows)
    json.dump(tlog, open(DATA / f"pm_plen_train{a.tag}.json", "w"), indent=1)



def part_pwhite(a):
    """Quick comparison of whitening under uniform tokens (white) and under
    the source statistics (pwhite), U = 4 flat, L = 16."""
    src = Source(a.source)
    pd = src.dist()
    sc = Scorer()
    ev = Evaluator(src, 4, sc)
    rows, tlog = [], {}
    for k in ("white", "pwhite"):
        E = codebook(src.W, D, k, pd=pd)
        mu = (pd[:, None] * E).sum(0)
        C = (E * pd[:, None]).T @ E - torch.outer(mu, mu)
        evl = torch.linalg.eigvalsh(C)
        m, lg, tt = train_masks(E, 16, 4, pd, COND["flat"], a.iters, seed=1)
        tlog[k] = dict(mean_norm=float(mu.norm()), eig_max_over_mean=float(evl.max() / evl.mean()),
                       ter10=quick_ter(m, pd.to(DEVICE), COND["flat"], 4))
        print("[stats]", k, tlog[k], flush=True)
        for snr in (0, 5, 10, 15, 20):
            log_row(rows, ["pwhite", a.source, k, 4, 16, "flat", "flat", 1, snr]
                    + ev.score(dec_mask(m, ev.ids, snr, COND["flat"], 777 + 10 * snr)))
    write_csv(DATA / f"pm_pwhite_eval_{a.source}{a.tag}.csv", HDR, rows)
    json.dump(tlog, open(DATA / f"pm_pwhite_{a.source}{a.tag}.json", "w"), indent=1)




def fixed_pattern(E, users, kind):
    """Untrained periodic patterns predicted by the analysis.
    comb : L = U, user u carries all of its weight on pattern position u
    pair : L = 2U, user u carries its weight on positions 2u and 2u+1,
           i.e. on whole subcarriers"""
    L = users if kind == "comb" else 2 * users
    m = Masks(E, L, users)
    W = torch.zeros(users, L)
    for u in range(users):
        if kind == "comb":
            W[u, u] = 1.0
        else:
            W[u, 2 * u] = W[u, 2 * u + 1] = 1.0
    m.Wm.data.copy_(W)
    return m


def part_comb(a):
    src = Source("bert")
    pd = src.dist()
    E = codebook(src.W, D, a.book, pd=pd)
    sc = Scorer()
    rows, tlog = [], {}
    for users in (4, 8):
        ev = Evaluator(src, users, sc)
        conds = list(COND) if users == 4 else ["flat", "nmse05"]
        for kind in ("comb", "pair"):
            m = fixed_pattern(E, users, kind).to(DEVICE)
            m.calibrate(pd.to(DEVICE))
            m.eval()
            M = m.masks().detach().cpu()
            tlog[f"{kind}_U{users}"] = dict(iota=cross_energy(M), iota_pair=cross_energy(M, True))
            for cn in conds:
                cnd = cond_for(cn, users)
                for snr in (10, 20):
                    log_row(rows, ["comb", "bert", a.book, users, m.L, kind, cn, 0, snr]
                            + ev.score(dec_mask(m, ev.ids, snr, cnd, 777 + 10 * snr)))
    write_csv(DATA / f"pm_comb_eval{a.tag}.csv", HDR, rows)
    json.dump(tlog, open(DATA / f"pm_comb{a.tag}.json", "w"), indent=1)



@torch.no_grad()
def part_energy(a):
    """Measured average frame energy on the real test streams (U = 4) for
    the saved models; c was calibrated on the training-token frequencies."""
    out = {}
    for kind in ("bert", "gtr"):
        src = Source(kind)
        ids, _ = src.streams(4)
        names = sorted(f.stem for f in MODELS.glob(f"{kind}_*_U4_L16_flat_s*.pt"))
        if kind == "bert":
            names += [f.stem for f in sorted(MODELS.glob("white_U4_L*_*_s1.pt"))]
        for nm in names:
            sd = torch.load(MODELS / f"{nm}.pt", map_location="cpu")
            L = sd["Wm"].shape[1]
            m = Masks(sd["E"], L, 4)
            m.load_state_dict(sd)
            m.to(DEVICE).eval()
            e = 0.0
            for k in range(0, ids.shape[0], 4096):
                y = m.tx(ids[k:k + 4096].to(DEVICE)) / m.c
                e += float(y.pow(2).sum())
            out[nm] = e / ids.shape[0]
            print("[energy]", nm, round(out[nm], 4), flush=True)
    json.dump(out, open(DATA / f"pm_energy{a.tag}.json", "w"), indent=1)



@torch.no_grad()
def bound_eval(m, E, pd, Wn, users, snrs, n_v, n_h=400, seed=5, fixed_gain=False):
    """Union bounds conditioned on the gain and clipped before averaging:
      SER_u <= sum_v p_v E_h[min(1, sum_w P(v->w|h))]
      D_u   <= sum_v p_v E_h[min(max_w(1-rho_vw), sum_w (1-rho_vw) P(v->w|h))]
    with P(v->w|h) the Gaussian-interference pairwise probability of (5).
    h^2 ~ Exp(1) on n_h equiprobable quantile nodes. Tokens v are drawn
    from p_v, so the sum over v is a sample mean."""
    M = m.masks().detach()
    ac = 1.0 / float(m.c)
    mu_e = (pd[:, None] * E).sum(0)
    Ce = (E * pd[:, None]).T @ E - torch.outer(mu_e, mu_e)
    uq = (torch.arange(n_h, device=E.device, dtype=torch.float32) + 0.5) / n_h
    h2 = -torch.log1p(-uq)                                           # (H,)
    if fixed_gain:
        h2 = torch.ones(1, device=E.device)
    g = torch.Generator(device="cpu").manual_seed(seed)
    vs = torch.multinomial(pd.cpu(), n_v, replacement=True, generator=g).tolist()
    ub_ser = torch.zeros(users, len(snrs), device=E.device)
    ub_sem = torch.zeros(users, len(snrs), device=E.device)
    for u in range(users):
        others = [k for k in range(users) if k != u]
        mu_I = ac * sum(M[k] * mu_e for k in others)
        for v in vs:
            Dl = (E[v][None] - E) * M[u][None]
            n2 = (Dl * Dl).sum(1)
            ok = n2 > 0
            Dl, n2 = Dl[ok], n2[ok]
            mean = Dl @ mu_I
            var_i = torch.zeros_like(n2)
            for k in others:
                Z = Dl * M[k][None]
                var_i += ac * ac * ((Z @ Ce) * Z).sum(1)
            semw = (1 - (Wn[v][None] * Wn).sum(1))[ok].clamp_min(0)
            cap = float(semw.max())
            num = ac * n2 / 2 + mean
            for j, snr in enumerate(snrs):
                s2 = 1.0 / (D * 10 ** (snr / 10))
                den = torch.sqrt(n2[None] * s2 / h2[:, None] + var_i[None])
                q = 0.5 * torch.erfc(num[None] / den / math.sqrt(2))      # (H,V')
                ub_ser[u, j] += q.sum(1).clamp(max=1.0).mean() / n_v
                ub_sem[u, j] += (q * semw[None]).sum(1).clamp(max=cap).mean() / n_v
    return ub_ser, ub_sem


def part_bound2(a):
    src = Source("bert")
    pdc = src.dist()
    E = codebook(src.W, D, a.book, pd=pdc).to(DEVICE)
    pd = pdc.to(DEVICE)
    Wn = src.Wn.to(DEVICE)
    users = 4
    snrs = list(range(0, 21, 2))
    rows = []
    for L in (16, 64):
        sd = torch.load(MODELS / f"white_U4_L{L}_flat_s1.pt", map_location="cpu")
        m = Masks(sd["E"], L, users)
        m.load_state_dict(sd)
        m.to(DEVICE).eval()
        ub_ser, ub_sem = bound_eval(m, E, pd, Wn, users, snrs, a.bound_nv)
        for j, snr in enumerate(snrs):
            torch.manual_seed(900 + snr)
            err = torch.zeros(users, device=DEVICE)
            sem = torch.zeros(users, device=DEVICE)
            frames = a.bound_mc
            for k in range(0, frames, 2048):
                n = min(2048, frames - k)
                idx = torch.multinomial(pd, n * users, replacement=True).view(n, users)
                x = m.tx(idx) / m.c
                r, w = rc.channel(x, snr, users, **COND["flat"])
                dec = m.scores(r, w).argmax(-1)
                err += (dec != idx).float().sum(0)
                sem += (1 - (Wn[idx] * Wn[dec]).sum(-1)).sum(0)
            for u in range(users):
                rows.append([L, u, snr, float(ub_ser[u, j]), float(err[u] / frames),
                             float(ub_sem[u, j]), float(sem[u] / frames)])
            print("[bound2]", L, snr, f"ub_ser {float(ub_ser[:, j].mean()):.4f} ser "
                  f"{float(err.mean() / frames):.4f} ub_sem {float(ub_sem[:, j].mean()):.4f} "
                  f"sem {float(sem.mean() / frames):.4f}", flush=True)
    write_csv(DATA / f"pm_bound2{a.tag}.csv", ["L", "user", "snr_db", "ub_ser", "mc_ser", "ub_sem", "mc_sem"], rows)



def part_bound3(a):
    """Fixed unit gain (no fading), where the clipping of the union bound is
    inactive at moderate SNR: SER and semantic-distortion bounds against
    Monte Carlo, L = 16."""
    src = Source("bert")
    pdc = src.dist()
    E = codebook(src.W, D, a.book, pd=pdc).to(DEVICE)
    pd = pdc.to(DEVICE)
    Wn = src.Wn.to(DEVICE)
    users = 4
    snrs = [0, 2, 4, 6, 8, 10, 12]
    rows = []
    sd = torch.load(MODELS / "white_U4_L16_flat_s1.pt", map_location="cpu")
    m = Masks(sd["E"], 16, users)
    m.load_state_dict(sd)
    m.to(DEVICE).eval()
    ub_ser, ub_sem = bound_eval(m, E, pd, Wn, users, snrs, a.bound_nv, fixed_gain=True)
    with torch.no_grad():
        for j, snr in enumerate(snrs):
            torch.manual_seed(700 + snr)
            sig = math.sqrt(1.0 / (D * 10 ** (snr / 10)))
            err = sem = 0.0
            frames = a.bound_mc
            for k in range(0, frames, 2048):
                n = min(2048, frames - k)
                idx = torch.multinomial(pd, n * users, replacement=True).view(n, users)
                x = m.tx(idx) / m.c
                r = x[:, None, :] + sig * torch.randn(n, users, D, device=DEVICE)
                w = torch.ones_like(r)
                dec = m.scores(r, w).argmax(-1)
                err += float((dec != idx).float().sum())
                sem += float((1 - (Wn[idx] * Wn[dec]).sum(-1)).sum())
            mc_ser, mc_sem = err / (frames * users), sem / (frames * users)
            rows.append([snr, float(ub_ser[:, j].mean()), mc_ser, float(ub_sem[:, j].mean()), mc_sem])
            print("[bound3]", snr, f"ub_ser {rows[-1][1]:.4f} ser {mc_ser:.4f} ub_sem {rows[-1][3]:.4f} "
                  f"sem {mc_sem:.4f}", flush=True)
    write_csv(DATA / f"pm_bound3{a.tag}.csv", ["snr_db", "ub_ser", "mc_ser", "ub_sem", "mc_sem"], rows)


def part_all(a):
    for f in (part_models, part_period, part_channel, part_plen, part_bound):
        t = time.time()
        f(a)
        print("[part-done]", f.__name__, round((time.time() - t) / 60, 1), "min", flush=True)



def part_rest(a):
    for f in (part_channel, part_plen, part_bound):
        t = time.time()
        f(a)
        print("[part-done]", f.__name__, round((time.time() - t) / 60, 1), "min", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True)
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--bound_nv", type=int, default=400)
    ap.add_argument("--bound_mc", type=int, default=200_000)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--source", default="bert")
    ap.add_argument("--book", default="pwhite")
    a = ap.parse_args()
    if a.smoke:                  # keep smoke-test models apart from the real ones
        global MODELS
        MODELS = DATA / "pm_models_smoke"
        MODELS.mkdir(exist_ok=True)
    t0 = time.time()
    {"period": part_period, "channel": part_channel, "models": part_models,
     "bound": part_bound, "plen": part_plen, "pwhite": part_pwhite,
     "all": part_all, "comb": part_comb, "rest": part_rest,
     "energy": part_energy, "bound2": part_bound2,
     "bound3": part_bound3}[a.part](a)
    print("[done]", a.part, round((time.time() - t0) / 60, 1), "min", flush=True)


if __name__ == "__main__":
    main()
