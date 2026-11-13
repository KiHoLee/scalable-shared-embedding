"""Earlier design, analysis: masked-norm spread of the full codebook
after the long budget.

Compares the masked codeword norms of the 4,000-iteration full codebook
(data/full_V65536.pt) with those of the first long-budget run
(data/rev_full_long_md.pt, 2e5 iterations, written by exp_rev_budget.py),
each relative to its per-user mean, exactly as exp_rev_fullmd.py does for
rev_norms_full.json. Both checkpoints are large and not stored in the
repository; exp_main.py and exp_rev_budget.py retrain them.

Output: data/rev_norms_full_long.json
"""
import json

import torch

from sse_lib import DATA, FullCodebook


def stats(name):
    m = FullCodebook(65536)
    m.load_state_dict(torch.load(DATA / name, map_location="cpu"))
    with torch.no_grad():
        n = (m.codebook()[None] * m.masks()[:, None, :]).norm(dim=-1)
        rel = n / n.mean(1, keepdim=True)
    return {"rel_min": float(rel.min()), "rel_max": float(rel.max()),
            "max_dev": float((rel - 1).abs().max()),
            "cv": (n.std(1) / n.mean(1)).tolist()}


def main():
    out = {"iter_4000": stats("full_V65536.pt"),
           "iter_200000_seed1": stats("rev_full_long_md.pt")}
    (DATA / "rev_norms_full_long.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
