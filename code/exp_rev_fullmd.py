"""Revision: re-evaluation of the full codebook with exact minimum-distance
(MD) decoding. The trained full codebooks (data/full_V*.pt, the same
4,000-iteration models as the original letter) have unequal masked
codeword norms, so the correlation rule is not minimum-distance for
them. MD decoding uses the known scale 1/c and the precomputable norms.

Outputs:
  rev_norms_full.json   : masked-norm statistics of the full codebook
  rev_snr_fullmd.csv    : SER vs SNR at V = 65,536, corr and MD
  rev_scaling_fullmd.csv: SER at 10 and 16 dB vs V, corr and MD
"""
from __future__ import annotations

import argparse
import json

import torch

from sse_lib import DATA, DEVICE, FullCodebook, eval_ser_full, write_csv

SNR_GRID = list(range(0, 21, 2))
V_LIST = [16, 81, 256, 625, 1296, 4096, 10000, 20736, 65536]


def load(V):
    m = FullCodebook(V)
    m.load_state_dict(torch.load(DATA / f"full_V{V}.pt", map_location="cpu"))
    return m.to(DEVICE).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=500_000)
    ap.add_argument("--part", default="norms,snr,scaling")
    a = ap.parse_args()
    if "norms" in a.part:
        m = load(65536)
        with torch.no_grad():
            cand = m.codebook()[None] * m.masks()[:, None, :]
            n = cand.norm(dim=-1)
            rel = n / n.mean(1, keepdim=True)
        st = {"rel_min": float(rel.min()), "rel_max": float(rel.max()),
              "cv": (n.std(1) / n.mean(1)).tolist()}
        json.dump(st, open(DATA / "rev_norms_full.json", "w"), indent=1)
        print("[norms]", st, flush=True)
    if "snr" in a.part:
        m = load(65536)
        m.md = False
        corr = eval_ser_full(m, SNR_GRID, frames=a.frames)
        m.md = True
        md = eval_ser_full(m, SNR_GRID, frames=a.frames)
        write_csv(DATA / "rev_snr_fullmd.csv", ["snr_db", "ser_corr", "ser_md"],
                  list(zip(SNR_GRID, corr, md)))
        print("[snr]", md, flush=True)
    if "scaling" in a.part:
        rows = []
        for V in V_LIST:
            m = load(V)
            m.md = True
            md = eval_ser_full(m, [10.0, 16.0], frames=a.frames)
            rows.append((V, md[0], md[1]))
            print("[scaling]", rows[-1], flush=True)
        write_csv(DATA / "rev_scaling_fullmd.csv",
                  ["V", "ser10_md", "ser16_md"], rows)


if __name__ == "__main__":
    main()
