"""Symbol error rate of the untrained comb pattern against the vocabulary size.

Nothing is trained. Every user sends one token per frame on the comb pattern
without overlap (L = U = 4: user u puts weight 2 on the real dimensions
u, u + 4, u + 8, ..., so ||m_u||^2 = d = 64), the codebook is the pwhite
codebook of exp_pm.codebook (whitening under the token frequencies p_v,
top 64 eigenpairs, fixed rotation with generator seed 0, unit-norm rows),
and every receiver applies the weighted minimum-distance decoder of
exp_pm.Masks.scores over all V candidates of that codebook.

Part vocab (the only part):
  bert  BERT base uncased token table restricted to the V most frequent
        tokens, V = 1,024, 4,096, 16,384 and 30,522 (the whole table), with
        p_v renormalized over the kept tokens and the codebook rebuilt from
        the kept rows. Tokens are ranked by p_v with a stable sort, so ties
        keep the order of the token ids; the kept tokens stay in token-id
        order. V = 30,522 is therefore the codebook and the token draws of
        the letter and repeats the comb points of exp_pm_ext.py --part figs
        (graph g2, no_overlap, rho = 0) up to the draw of c.
  gpt2  GPT-2 token table (wte, V = 50,257) with p_v counted, as in
        exp_pm.Source.dist, on the first 40,000 AG News training articles
        under the GPT-2 tokenizer without special tokens. The model is read
        from the local Hugging Face cache only; the row is skipped (with a
        message) when GPT-2 is not cached.
For every vocabulary: flat Rayleigh fading with exact gains (COND["flat"]),
U = 4, SNR 0, 5, 10, 15, 20 dB, 40,000 frames per point with i.i.d. tokens
drawn with p_v (exp_pm_ext.ser), and the source entropy
H = -sum_v p_v log2 p_v in bits per token.

Seeds: codebook rotation generator seed 0; constant c of the comb drawn
from 65,536 frames after torch.manual_seed(0) (as in exp_pm_ext.py --part
sig); test frames seeded 11 + 10 snr (exp_pm_ext.ser), so every vocabulary
sees the same channel and noise draws.

Output: data/pm/pm_vocab.csv with columns
  source, V, n_support (tokens with p_v > 0), mass_in_full (share of the
  token occurrences of the full table that the kept tokens carry),
  entropy_bits, log2_V, c, snr_db, frames, seed, errors, ser
(--smoke: data/pm/pm_vocab_smoke.csv, 2,048 frames at 10 dB, V = 1,024 and
GPT-2 only).

Run from the code folder (GPU recommended):
    python exp_pm_vocab.py --part vocab --offline
--offline sets HF_HUB_OFFLINE and HF_DATASETS_OFFLINE so that the models
and AG News are read from the local cache and nothing is downloaded.
"""
from __future__ import annotations

import argparse
import math
import os
import time

import torch

import exp_pm as pm
import exp_pm_ext as px
from sse_lib import DEVICE, D, write_csv

USERS = 4
SNRS = (0, 5, 10, 15, 20)
FRAMES = 40_000
BERT_V = (1024, 4096, 16384, 30522)
HDR = ["source", "V", "n_support", "mass_in_full", "entropy_bits", "log2_V", "c",
       "snr_db", "frames", "seed", "errors", "ser"]


def comb_masks(users: int, d: int = D) -> torch.Tensor:
    """Comb pattern without overlap, L = U: weight sqrt(U) on the real
    dimensions u, u + U, ... of user u, so ||m_u||^2 = d."""
    M = torch.zeros(users, d)
    for u in range(users):
        M[u, u::users] = math.sqrt(users)
    return M


def entropy_bits(p: torch.Tensor) -> float:
    p = p.double()
    p = p[p > 0]
    return float(-(p * torch.log2(p)).sum())


def gpt2_source(n_items: int = 40000):
    """GPT-2 token table (V, 768) and p_v from the first n_items AG News
    training articles, without special tokens; None if not cached."""
    from transformers import AutoModel, AutoTokenizer
    try:
        tok = AutoTokenizer.from_pretrained("gpt2", local_files_only=True)
        model = AutoModel.from_pretrained("gpt2", local_files_only=True)
    except OSError as e:                     # not in the local cache
        print("[gpt2] not in the local Hugging Face cache, skipped:", e, flush=True)
        return None
    W = model.get_input_embeddings().weight.detach().float().cpu()
    V = len(tok)
    assert W.shape[0] == V, (W.shape, V)
    from datasets import load_dataset
    ds = load_dataset("fancyzhx/ag_news", split="train")
    enc = tok(ds[:n_items]["text"], add_special_tokens=False)["input_ids"]
    cnt = torch.zeros(V)
    for ids in enc:
        cnt += torch.bincount(torch.tensor(ids, dtype=torch.long), minlength=V).float()
    return W, cnt / cnt.sum()


@torch.no_grad()
def run_vocab(name, W, p, mass_in_full, snrs, frames, rows):
    """Comb SER at every SNR for the pwhite codebook of the table W (V,768)
    under the token frequencies p (V,), sum p = 1."""
    E = pm.codebook(W, D, "pwhite", pd=p).to(DEVICE)
    pd = p.to(DEVICE)
    m = px.FreeMasks(E, comb_masks(USERS).to(DEVICE)).to(DEVICE).eval()
    torch.manual_seed(0)                                  # fixed draw for c
    px.calibrate_rho(m, pd, 0.0)
    V = W.shape[0]
    H = entropy_bits(p)
    n_sup = int((p > 0).sum())
    c = float(m.c)
    print(f"[vocab] {name} V={V} support={n_sup} mass={mass_in_full:.4f} "
          f"H={H:.3f} bits log2V={math.log2(V):.3f} c={c:.4f}", flush=True)
    for snr in snrs:
        v = px.ser(m, pd, 0.0, snr, pm.COND["flat"], frames=frames)
        errors = int(round(v * frames * USERS))
        rows.append([name, V, n_sup, mass_in_full, H, math.log2(V), c,
                     snr, frames, 11 + int(10 * snr), errors, v])
        print(f"[vocab] {name} V={V} snr={snr} ser={v:.5f} ({errors} errors)", flush=True)


def part_vocab(a):
    torch.backends.cuda.matmul.allow_tf32 = False         # evaluation in full fp32
    snrs = (10,) if a.smoke else SNRS
    frames = 2048 if a.smoke else a.frames
    bert_v = (1024,) if a.smoke else BERT_V
    rows = []
    src = pm.Source("bert")
    p_full = src.dist()
    order = torch.sort(p_full, descending=True, stable=True).indices
    for V in bert_v:
        keep = order[:V].sort().values                    # token-id order
        mass = float(p_full[keep].sum())
        p = p_full[keep] / p_full[keep].sum() if V < p_full.numel() else p_full
        run_vocab("bert", src.W[keep], p, mass, snrs, frames, rows)
    g = gpt2_source()
    if g is not None:
        run_vocab("gpt2", g[0], g[1], 1.0, snrs, frames, rows)
    write_csv(pm.DATA / f"pm_vocab{'_smoke' if a.smoke else ''}.csv", HDR, rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="vocab", choices=["vocab"])
    ap.add_argument("--frames", type=int, default=FRAMES)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--offline", action="store_true",
                    help="read models and data from the local Hugging Face cache only")
    a = ap.parse_args()
    if a.offline:                       # before transformers/datasets are imported
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
    t0 = time.time()
    {"vocab": part_vocab}[a.part](a)
    print("[done]", a.part, round((time.time() - t0) / 60, 1), "min", flush=True)


if __name__ == "__main__":
    main()
