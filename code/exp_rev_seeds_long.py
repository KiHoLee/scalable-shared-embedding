"""Revision: seed spread of the Proposed SSE with a 20,000-iteration
cosine-decay budget (seeds 1-5), evaluated over 5e5 frames at 10 and
20 dB. Usage: python exp_rev_seeds_long.py <seed>. Output rows are
appended to data/rev_seeds_sse_20k.csv.
"""
import sys
import torch
import exp_rev_budget as B
from sse_lib import DATA, eval_ser_sse

seed = int(sys.argv[1])
m, _ = B.train("sse", 20000, seed=seed)
torch.save(m.state_dict(), DATA / f"rev_sse_20k_seed{seed}.pt")
s = eval_ser_sse(m, [10.0, 20.0], frames=500_000)
with open(DATA / "rev_seeds_sse_20k.csv", "a") as f:
    f.write(f"{seed},{s[0]:.10g},{s[1]:.10g}\n")
print("[seed20k]", seed, s, flush=True)
