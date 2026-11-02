"""Revision experiment R1.1 / R2.5: topic accuracy on the real AG News token
streams for all six 4,000-iteration SSE runs, not only the reported one.

The streams, the channel and the scoring are those of exp_rev_semantic.py,
with its decoders and its Scorer used unchanged: 1,948 complete news items in
four disjoint user streams, one token per user per frame, a new gain h_u per
frame and per user with h_u^2 ~ Exp(1) known at the receiver, and sigma^2 =
1/(d snr). The received text is scored by two frozen public models that never
see the channel, textattack/bert-base-uncased-ag-news (revision fe417ad) and
sentence-transformers/all-MiniLM-L6-v2 (revision 1110a24).

Schemes: sse_P4.pt (the reported model, best of six), rev_sse_seed1..5.pt, the
full codebook with minimum-distance decoding, OMA, and ToDMA with OMP and the
genie-aided association (a miss becomes [UNK], which favors ToDMA).
SSE_mean6 is the mean over the six SSE runs, item by item and repeat by repeat.

Each scheme is decoded 8 times at 6, 10, 16 and 20 dB, with the seed
SEED_EVAL + 1000 r + 10 snr of exp_rev_semantic.py. Repeats 0-3 therefore use
the seeds of data/rev_semantic.csv, which was computed on another machine. The
full codebook and ToDMA, drawn in small chunks, reproduce that file exactly on
u24; SSE and OMA, drawn in one large tensor, reproduce it only in distribution.

Metrics: token error rate (ter), topic accuracy against the dataset label
(acc), agreement with the label the classifier gives the error-free text
(agree), and the mean cosine between the sentence embeddings of the
transmitted and received items (cos). acc_rep_sd is the standard deviation of
acc over the repeats. In the SSE_mean6 rows acc_run_min and acc_run_max give
the range over the six runs.

Pairs: paired percentile bootstrap of the accuracy difference model - ref,
ref in {OMA, Full}. Each item contributes its mean difference over the
repeats, and items are resampled (10,000 draws, fixed seed, one draw matrix
for all comparisons). With gains independent across frames, the items are
independent given the streams, so this interval covers the sampling of items
and the channel. It does not cover the training seed: SSE_mean6 treats the six
runs as fixed. rep_se = rep_sd / sqrt(repeats) is the channel-only check.

Model files: local snapshots under $SSE_HF_SNAPSHOTS if set,
else the public Hugging Face names.

Outputs: data/rev_sem_runs.csv, data/rev_sem_runs_pairs.csv
"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import numpy as np
import torch

import exp_rev_semantic as E
from sse_lib import DATA, DEVICE, ROOT, U, SSE, FullCodebook, set_seed, write_csv

SNRS = [6, 10, 16, 20]
REPEATS = 8
BOOT = 10_000
SEED_BOOT = 1
RUNS = {"reported": "sse_P4.pt", "seed1": "rev_sse_seed1.pt",
        "seed2": "rev_sse_seed2.pt", "seed3": "rev_sse_seed3.pt",
        "seed4": "rev_sse_seed4.pt", "seed5": "rev_sse_seed5.pt"}
MEAN6 = "SSE_mean6"
REFS = ("OMA", "Full")


def local_model(name: str) -> str:
    """Local snapshot of a Hugging Face model if there is one, else its name."""
    root = Path(os.environ.get("SSE_HF_SNAPSHOTS",
                               ROOT / "hf"))
    snaps = sorted((root / f"models--{name.replace('/', '--')}" / "snapshots").glob("*"))
    return str(snaps[0]) if snaps else name


E.CLS_NAME = local_model(E.CLS_NAME)
E.SIM_NAME = local_model(E.SIM_NAME)


def repeat_seed(r: int, snr) -> int:
    """Decoding seed of exp_rev_semantic.py."""
    return E.SEED_EVAL + 1000 * r + int(10 * snr)


class Task:
    """The streams, the scorer, and the error-free reference of one evaluation."""

    def __init__(self):
        set_seed(E.SEED_EVAL)
        self.tok, self.ids, self.items = E.load_streams()
        self.labels = torch.tensor([t[2] for u in range(U) for t in self.items[u]])
        self.sc = E.Scorer(self.tok)
        txt = self.sc.texts(self.ids, self.items)
        self.clean_pred = self.sc.predict(txt)
        self.clean_emb = self.sc.embed(txt)
        self.clean_acc = float((self.clean_pred == self.labels).float().mean())
        print("[clean] acc", self.clean_acc, "items", len(txt), "frames",
              self.ids.shape[0], flush=True)

    def score(self, dec):
        """Returns TER, per-item correct flags, agreement and mean cosine."""
        txt = self.sc.texts(dec, self.items)
        pr = self.sc.predict(txt)
        return (float((dec != self.ids).float().mean()),
                (pr == self.labels).numpy(),
                float((pr == self.clean_pred).float().mean()),
                float((self.sc.embed(txt) * self.clean_emb).sum(-1).mean()))


def load_sse(name: str) -> SSE:
    m = SSE(P=E.P_MAX)
    m.load_state_dict(torch.load(DATA / name, map_location="cpu"))
    return m.to(DEVICE).eval()


def load_full() -> FullCodebook:
    m = FullCodebook(E.V)
    m.load_state_dict(torch.load(DATA / f"full_V{E.V}.pt", map_location="cpu"))
    m.to(DEVICE).eval()
    m.md = True             # full codebook at its best: minimum distance
    return m


def row(model, snr, m, sd_acc, run_min="", run_max=""):
    return [model, snr, float(m[0]), float(m[1]), float(m[2]), float(m[3]),
            float(sd_acc), run_min, run_max]


def pair_rows(model, ref, snr, a, b, idx):
    """a, b: (R,N) per-repeat per-item correct flags of model and ref."""
    di = a.mean(0) - b.mean(0)
    bs = di[idx].mean(1)
    rd = a.mean(1) - b.mean(1)
    sd = rd.std(ddof=1) if len(rd) > 1 else float("nan")
    return [model, ref, snr, float(di.mean()), float(np.quantile(bs, .025)),
            float(np.quantile(bs, .975)), float(bs.std()), float(sd),
            float(sd / math.sqrt(len(rd))), int((di != 0).sum())]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snrs", default=",".join(map(str, SNRS)))
    ap.add_argument("--repeats", type=int, default=REPEATS)
    ap.add_argument("--boot", type=int, default=BOOT)
    ap.add_argument("--out", default=str(DATA / "rev_sem_runs"),
                    help="output prefix; adds .csv and _pairs.csv")
    a = ap.parse_args()
    snrs = [int(s) for s in a.snrs.split(",")]

    t = Task()
    ids = t.ids
    full = load_full()
    decoders = {"OMA": lambda s, k: E.dec_oma(ids, s, k),
                "Full": lambda s, k: E.dec_full(full, ids, s, k),
                "ToDMA": lambda s, k: E.dec_todma(ids, s, k)}
    for nm, file in RUNS.items():
        decoders[nm] = (lambda m: lambda s, k: E.dec_sse(m, ids, s, k))(load_sse(file))

    rows = [["clean", "inf", 0.0, t.clean_acc, 1.0, 1.0, 0.0, "", ""]]
    C, M = {}, {}
    for nm, dec in decoders.items():
        for s in snrs:
            res = [t.score(dec(s, repeat_seed(r, s))) for r in range(a.repeats)]
            C[nm, s] = np.stack([x[1] for x in res])       # (R,N)
            rep = np.array([[x[0], x[1].mean(), x[2], x[3]] for x in res])  # (R,4)
            M[nm, s] = rep
            sd = rep[:, 1].std(ddof=1) if a.repeats > 1 else float("nan")
            rows.append(row(nm, s, rep.mean(0), sd))
            print("[sem]", nm, s, "ter %.4f acc %.4f agree %.4f cos %.4f sd %.4f"
                  % (*rep.mean(0), sd), flush=True)
    for s in snrs:
        rep = np.stack([M[nm, s] for nm in RUNS]).mean(0)   # (R,4)
        per_run = [M[nm, s][:, 1].mean() for nm in RUNS]
        sd = rep[:, 1].std(ddof=1) if a.repeats > 1 else float("nan")
        rows.append(row(MEAN6, s, rep.mean(0), sd, float(min(per_run)), float(max(per_run))))
        C[MEAN6, s] = np.stack([C[nm, s] for nm in RUNS]).mean(0)
        print("[sem]", MEAN6, s, "ter %.4f acc %.4f agree %.4f cos %.4f" % tuple(rep.mean(0)),
              "runs %.4f..%.4f" % (min(per_run), max(per_run)), flush=True)
    write_csv(Path(a.out + ".csv"),
              ["model", "snr_db", "ter", "acc", "agree", "cos", "acc_rep_sd",
               "acc_run_min", "acc_run_max"], rows)

    rng = np.random.default_rng(SEED_BOOT)
    idx = rng.integers(len(t.labels), size=(a.boot, len(t.labels)))
    pr = []
    for nm in [*RUNS, MEAN6, "ToDMA", "Full"]:
        for ref in REFS:
            if nm == ref:
                continue
            for s in snrs:
                pr.append(pair_rows(nm, ref, s, C[nm, s].astype(float),
                                    C[ref, s].astype(float), idx))
    write_csv(Path(a.out + "_pairs.csv"),
              ["model", "ref", "snr_db", "diff", "ci_lo", "ci_hi", "item_se",
               "rep_sd", "rep_se", "discordant_items"], pr)


if __name__ == "__main__":
    main()
