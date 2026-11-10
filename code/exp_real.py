"""Real-data stage: transmission of real BERT WordPiece token streams.

AG News test headlines are tokenized with the bert-base-uncased
WordPiece tokenizer (vocabulary size 30,522). Four users carry four
separate headline streams. Each frame transmits one token per user,
mapped to base-16 digits of the raw token identifier (the 16-bit digit
space 16^4 = 65,536 covers the vocabulary). The models trained in
exp_main.py on uniform indices are reused unchanged, so this stage
tests distribution mismatch on real data.

Outputs:
  real_ter.csv   : token error rate vs SNR for SSE, Full, OMA
  real_stats.json: stream statistics and headline recovery rates
"""
from __future__ import annotations

import json
import math

import torch

from sse_lib import (DATA, DEVICE, SSE, FullCodebook, P_MAX, VU,
                     rayleigh_gain, set_seed, write_csv)

SNR_GRID = list(range(0, 21, 2))
N_TEXTS = 2000
REPEATS = 24
REC_RUNS = 8
SEED_EVAL = 777


def load_streams():
    from datasets import load_dataset
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    ds = load_dataset("fancyzhx/ag_news", split="test")
    texts = [ds[i]["text"] for i in range(N_TEXTS)]
    streams = [[] for _ in range(4)]
    bounds = [[] for _ in range(4)]     # (start, end) per headline
    for i, t in enumerate(texts):
        ids = tok(t, add_special_tokens=False)["input_ids"]
        u = i % 4
        s = len(streams[u])
        streams[u].extend(ids)
        bounds[u].append((s, s + len(ids)))
    n = min(len(s) for s in streams)
    streams = [s[:n] for s in streams]
    bounds = [[(a, b) for (a, b) in bu if b <= n] for bu in bounds]
    return streams, bounds, tok.vocab_size


def ids_to_digits(ids: torch.Tensor) -> torch.Tensor:
    """(N,U) token ids -> (N,U,P) base-16 digits, most significant first."""
    d = []
    x = ids.clone()
    for _ in range(P_MAX):
        d.append(x % VU)
        x = x // VU
    return torch.stack(d[::-1], dim=-1)


@torch.no_grad()
def wrong_sse(model: SSE, digits_all, snr_db, gen_seed, chunk=50_000):
    """One channel realization: per-frame per-user wrong mask (N,U)."""
    torch.manual_seed(gen_seed)
    N = digits_all.shape[0]
    wrong = torch.zeros(N, 4, dtype=torch.bool)
    for n0 in range(0, N, chunk):
        dg = digits_all[n0:n0 + chunk].to(DEVICE)
        snr = torch.full((dg.shape[0],), float(snr_db))
        sc = model(dg, snr)
        wrong[n0:n0 + chunk] = (sc.argmax(-1) != dg).any(dim=2).cpu()
    return wrong


@torch.no_grad()
def wrong_full(model: FullCodebook, ids_all, snr_db, gen_seed):
    torch.manual_seed(gen_seed)
    N = ids_all.shape[0]
    chunk = max(512, (1 << 23) // model.Vc)
    wrong = torch.zeros(N, 4, dtype=torch.bool)
    for n0 in range(0, N, chunk):
        idx = ids_all[n0:n0 + chunk].to(DEVICE)
        snr = torch.full((idx.shape[0],), float(snr_db))
        sc = model(idx, snr)
        wrong[n0:n0 + chunk] = (sc.argmax(-1) != idx).cpu()
    return wrong


@torch.no_grad()
def wrong_oma(ids_all, snr_db, gen_seed, bits=16):
    """BPSK per real dimension on the actual token bits."""
    torch.manual_seed(gen_seed)
    N, Uu = ids_all.shape
    b = ((ids_all[..., None] >> torch.arange(bits)) & 1).float() * 2 - 1
    b = b.to(DEVICE)
    sigma = math.sqrt(1.0 / (10.0 ** (snr_db / 10.0)))
    h = rayleigh_gain((N, Uu, 1))
    y = h * b + sigma * torch.randn(N, Uu, bits, device=DEVICE)
    return ((y * b) < 0).any(dim=2).cpu()


def headline_recovery(wrong: torch.Tensor, bounds) -> tuple[int, int]:
    ok = tot = 0
    for u in range(4):
        wu = wrong[:, u]
        for (a, b) in bounds[u]:
            tot += 1
            ok += int(not bool(wu[a:b].any()))
    return ok, tot


def main():
    set_seed(SEED_EVAL)
    streams, bounds, vocab = load_streams()
    ids_all = torch.tensor(list(zip(*streams)), dtype=torch.long)   # (N,4)
    digits_all = ids_to_digits(ids_all)
    N = ids_all.shape[0]
    assert int(ids_all.max()) < VU ** P_MAX

    sse = SSE(P=P_MAX)
    sse.load_state_dict(torch.load(DATA / "sse_P4.pt", map_location="cpu"))
    sse.to(DEVICE).eval()
    full = FullCodebook(VU ** P_MAX)
    full.load_state_dict(torch.load(DATA / f"full_V{VU**P_MAX}.pt",
                                    map_location="cpu"))
    full.to(DEVICE).eval()

    schemes = {
        "SSE": lambda s, k: wrong_sse(sse, digits_all, s, k),
        "Full": lambda s, k: wrong_full(full, ids_all, s, k),
        "OMA": lambda s, k: wrong_oma(ids_all, s, k),
    }

    rows = []
    for s in SNR_GRID:
        ter = {}
        for name, fn in schemes.items():
            e = 0
            for r in range(REPEATS):
                e += int(fn(s, SEED_EVAL + 1000 * r + int(10 * s)).sum())
            ter[name] = e / (N * 4 * REPEATS)
        rows.append((s, ter["SSE"], ter["Full"], ter["OMA"]))
        print("[real]", rows[-1])
    write_csv(DATA / "real_ter.csv",
              ["snr_db", "ter_SSE", "ter_Full", "ter_OMA"], rows)

    rec_rates = {}
    for s in (10, 16):
        rec_rates[str(s)] = {}
        for name, fn in schemes.items():
            ok = tot = 0
            for r in range(REC_RUNS):
                w = fn(s, SEED_EVAL + 5000 * r + int(10 * s))
                o, t = headline_recovery(w, bounds)
                ok += o; tot += t
            rec_rates[str(s)][name] = ok / tot

    stats = {
        "vocab_size": vocab,
        "n_texts": N_TEXTS,
        "frames": N,
        "repeats": REPEATS,
        "decisions_per_point": N * 4 * REPEATS,
        "distinct_tokens": int(torch.unique(ids_all).numel()),
        "max_token_id": int(ids_all.max()),
        "headlines_scored": sum(len(b) for b in bounds),
        "recovery": rec_rates,
    }
    (DATA / "real_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
