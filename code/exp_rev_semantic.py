"""Earlier design: task-level and semantic metrics
on the real AG News token streams, and the measured transmit energy.

Stream construction is identical to exp_real.py (2,000 AG News test
items, bert-base-uncased WordPiece ids, four separate user streams,
one token per user per frame, base-16 digits of the raw id). Each
scheme decodes the streams; the decoded ids are reassembled into the
news items, detokenized, and scored by two frozen public models that
never see the channel:
  * topic classification accuracy with a BERT classifier fine-tuned on
    AG News (textattack/bert-base-uncased-ag-news),
  * semantic similarity, the cosine between sentence embeddings of the
    transmitted and received item (all-MiniLM-L6-v2, mean pooling).
Out-of-vocabulary decoded ids map to [UNK]. For ToDMA the genie-aided
association gives no decoded id on a miss, so a missed token becomes
[UNK] (an erasure, which favors ToDMA).

Also measured on the real stream: the average transmit frame energy of
SSE, the full codebook, and ToDMA under the uniform-index normalization,
and the SSE/full token error rate when c is recalibrated on the real
stream (energy-matched check).

Outputs: data/rev_semantic.csv, data/rev_energy.json
"""
from __future__ import annotations

import json
import math

import torch

from sse_lib import (DATA, DEVICE, D, U, VU, P_MAX, SSE, FullCodebook,
                     rayleigh_gain, snr_to_sigma2, set_seed, write_csv)
from exp_sota import make_dictionary, omp_support

SNR_GRID = list(range(0, 21, 2))
N_TEXTS = 2000
REPEATS = 4
SEED_EVAL = 777
V = VU ** P_MAX
UNK = 100
CLS_NAME = "textattack/bert-base-uncased-ag-news"
SIM_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_streams():
    from datasets import load_dataset
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    ds = load_dataset("fancyzhx/ag_news", split="test")
    streams = [[] for _ in range(U)]
    items = [[] for _ in range(U)]       # (start, end, label)
    for i in range(N_TEXTS):
        ids = tok(ds[i]["text"], add_special_tokens=False)["input_ids"]
        u = i % U
        s = len(streams[u])
        streams[u].extend(ids)
        items[u].append((s, s + len(ids), ds[i]["label"]))
    n = min(len(s) for s in streams)
    streams = [s[:n] for s in streams]
    items = [[t for t in iu if t[1] <= n] for iu in items]
    return tok, torch.tensor(list(zip(*streams)), dtype=torch.long), items


def ids_to_digits(ids):
    d, x = [], ids.clone()
    for _ in range(P_MAX):
        d.append(x % VU); x = x // VU
    return torch.stack(d[::-1], dim=-1)


def digits_to_ids(dg):
    out = torch.zeros(dg.shape[:-1], dtype=torch.long, device=dg.device)
    for p in range(P_MAX):
        out = out * VU + dg[..., p]
    return out


# ----------------------------------------------------------------------
# decoders returning decoded ids (N,U)
# ----------------------------------------------------------------------
@torch.no_grad()
def dec_sse(model, ids_all, snr, seed, c=None, chunk=50_000):
    torch.manual_seed(seed)
    c0 = model.c.clone()
    if c is not None:
        model.c.fill_(c)
    out = torch.zeros_like(ids_all)
    dg_all = ids_to_digits(ids_all)
    for n0 in range(0, ids_all.shape[0], chunk):
        dg = dg_all[n0:n0 + chunk].to(DEVICE)
        sc = model(dg, torch.full((dg.shape[0],), float(snr)))
        out[n0:n0 + chunk] = digits_to_ids(sc.argmax(-1)).cpu()
    model.c.copy_(c0)
    return out


@torch.no_grad()
def dec_full(model, ids_all, snr, seed, c=None):
    torch.manual_seed(seed)
    c0 = model.c.clone()
    if c is not None:
        model.c.fill_(c)
    chunk = max(128, (1 << 23) // model.Vc)
    out = torch.zeros_like(ids_all)
    for n0 in range(0, ids_all.shape[0], chunk):
        idx = ids_all[n0:n0 + chunk].to(DEVICE)
        sc = model(idx, torch.full((idx.shape[0],), float(snr)))
        out[n0:n0 + chunk] = sc.argmax(-1).cpu()
    model.c.copy_(c0)
    return out


@torch.no_grad()
def dec_oma(ids_all, snr, seed, bits=16):
    torch.manual_seed(seed)
    N = ids_all.shape[0]
    b = ((ids_all[..., None] >> torch.arange(bits)) & 1).float() * 2 - 1
    b = b.to(DEVICE)
    sigma = math.sqrt(1.0 / (10.0 ** (snr / 10.0)))
    h = rayleigh_gain((N, U, 1))
    y = h * b + sigma * torch.randn(N, U, bits, device=DEVICE)
    bh = (y > 0).long().cpu()
    return (bh << torch.arange(bits)).sum(-1)


_A = None


@torch.no_grad()
def dec_todma(ids_all, snr, seed, chunk=1024):
    global _A
    if _A is None:
        _A = make_dictionary(D, V)
    torch.manual_seed(seed)
    sigma = math.sqrt(float(snr_to_sigma2(torch.tensor(float(snr)))))
    out = torch.full_like(ids_all, UNK)
    for n0 in range(0, ids_all.shape[0], chunk):
        tok = ids_all[n0:n0 + chunk].to(DEVICE)
        n = tok.shape[0]
        x = _A.T[tok].sum(dim=1) / math.sqrt(U)
        h = rayleigh_gain((n, U))
        for u in range(U):
            y = h[:, u:u + 1] * x + sigma * torch.randn(n, D, device=DEVICE)
            sup = omp_support(y / h[:, u:u + 1].clamp_min(1e-6), _A)
            hit = (sup == tok[:, u:u + 1]).any(dim=1)
            out[n0:n0 + n, u] = torch.where(hit, tok[:, u], UNK).cpu()
    return out


# ----------------------------------------------------------------------
# semantic scoring
# ----------------------------------------------------------------------
class Scorer:
    def __init__(self, tok):
        from transformers import (AutoTokenizer, AutoModel,
                                  AutoModelForSequenceClassification)
        self.tok = tok
        self.ctok = AutoTokenizer.from_pretrained(CLS_NAME)
        self.cls = AutoModelForSequenceClassification.from_pretrained(
            CLS_NAME).to(DEVICE).eval()
        self.stok = AutoTokenizer.from_pretrained(SIM_NAME)
        self.sim = AutoModel.from_pretrained(SIM_NAME).to(DEVICE).eval()

    def texts(self, dec_ids, items):
        dec_ids = dec_ids.clone()
        dec_ids[(dec_ids < 0) | (dec_ids >= self.tok.vocab_size)] = UNK
        out = []
        for u in range(U):
            for (a, b, _) in items[u]:
                out.append(self.tok.decode(dec_ids[a:b, u].tolist()))
        return out

    @torch.no_grad()
    def predict(self, texts, bs=256):
        pr = []
        for i in range(0, len(texts), bs):
            enc = self.ctok(texts[i:i + bs], padding=True, truncation=True,
                            max_length=128, return_tensors="pt").to(DEVICE)
            pr.append(self.cls(**enc).logits.argmax(-1).cpu())
        return torch.cat(pr)

    @torch.no_grad()
    def embed(self, texts, bs=256):
        em = []
        for i in range(0, len(texts), bs):
            enc = self.stok(texts[i:i + bs], padding=True, truncation=True,
                            max_length=128, return_tensors="pt").to(DEVICE)
            hs = self.sim(**enc).last_hidden_state
            m = enc["attention_mask"][..., None].float()
            e = (hs * m).sum(1) / m.sum(1)
            em.append(torch.nn.functional.normalize(e, dim=-1).cpu())
        return torch.cat(em)


def main():
    set_seed(SEED_EVAL)
    tok, ids_all, items = load_streams()
    labels = torch.tensor([t[2] for u in range(U) for t in items[u]])
    sc = Scorer(tok)
    clean_txt = sc.texts(ids_all, items)
    clean_pred = sc.predict(clean_txt)
    clean_emb = sc.embed(clean_txt)
    clean_acc = float((clean_pred == labels).float().mean())
    print("[clean] acc", clean_acc, "items", len(clean_txt), flush=True)

    sse = SSE(P=P_MAX)
    sse.load_state_dict(torch.load(DATA / "sse_P4.pt", map_location="cpu"))
    sse.to(DEVICE).eval()
    full = FullCodebook(V)
    full.load_state_dict(torch.load(DATA / f"full_V{V}.pt", map_location="cpu"))
    full.to(DEVICE).eval()
    full.md = True          # full codebook at its best: minimum distance

    # ---------------- measured transmit energy on the real stream -------
    with torch.no_grad():
        dg = ids_to_digits(ids_all).to(DEVICE)
        e_sse = float((sse.tx_frame(dg) / sse.c).pow(2).sum(dim=(1, 2)).mean())
        e_full = 0.0
        for n0 in range(0, ids_all.shape[0], 4096):
            y = full.tx_frame(ids_all[n0:n0 + 4096].to(DEVICE)) / full.c
            e_full += float(y.pow(2).sum())
        e_full /= ids_all.shape[0]
        A = make_dictionary(D, V)
        e_todma = float((A.T[ids_all.to(DEVICE)].sum(1) / math.sqrt(U))
                        .pow(2).sum(1).mean())
        # per-frame energy spread of SSE on the real stream
        ef = (sse.tx_frame(dg) / sse.c).pow(2).sum(dim=(1, 2))
    c_real_sse = float(sse.c) * math.sqrt(e_sse)
    c_real_full = float(full.c) * math.sqrt(e_full)
    energy = {"sse": e_sse, "full": e_full, "todma": e_todma, "oma": 1.0,
              "sse_frame_min": float(ef.min()), "sse_frame_max": float(ef.max()),
              "sse_frame_std": float(ef.std()),
              "c_uniform_sse": float(sse.c), "c_real_sse": c_real_sse,
              "c_uniform_full": float(full.c), "c_real_full": c_real_full}
    print("[energy]", energy, flush=True)

    decoders = {
        "SSE": lambda s, k: dec_sse(sse, ids_all, s, k),
        "Full": lambda s, k: dec_full(full, ids_all, s, k),
        "ToDMA": lambda s, k: dec_todma(ids_all, s, k),
        "OMA": lambda s, k: dec_oma(ids_all, s, k),
        "SSEem": lambda s, k: dec_sse(sse, ids_all, s, k, c=c_real_sse),
        "Fullem": lambda s, k: dec_full(full, ids_all, s, k, c=c_real_full),
    }
    names = list(decoders)
    rows = []
    for s in SNR_GRID:
        row = [s]
        for nm in names:
            ter = acc = agr = cos = 0.0
            for r in range(REPEATS):
                dec = decoders[nm](s, SEED_EVAL + 1000 * r + int(10 * s))
                ter += float((dec != ids_all).float().mean())
                if nm.endswith("em"):
                    continue
                txt = sc.texts(dec, items)
                pr = sc.predict(txt)
                acc += float((pr == labels).float().mean())
                agr += float((pr == clean_pred).float().mean())
                cos += float((sc.embed(txt) * clean_emb).sum(-1).mean())
            row += [ter / REPEATS, acc / REPEATS, agr / REPEATS, cos / REPEATS]
        rows.append(row)
        print("[sem]", [f"{v:.4f}" if isinstance(v, float) else v for v in row],
              flush=True)
    hdr = ["snr_db"]
    for nm in names:
        hdr += [f"ter_{nm}", f"acc_{nm}", f"agree_{nm}", f"cos_{nm}"]
    write_csv(DATA / "rev_semantic.csv", hdr, rows)
    energy["clean_acc"] = clean_acc
    energy["n_items"] = len(clean_txt)
    energy["repeats"] = REPEATS
    (DATA / "rev_energy.json").write_text(json.dumps(energy, indent=1))


if __name__ == "__main__":
    main()
