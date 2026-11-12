"""Spread of codeword differences over the subcarriers, for the comparison
with the bit interleaver of coded OMA (data/pm/pm_oma_coded_il.json).

Codebook (1): exp_pm.codebook(W, 64, "pwhite", pd=p_v), the BERT base
uncased input embeddings whitened under the token frequencies p_v of the
first 40,000 AG News training articles and rotated by the fixed random
orthogonal matrix R (seed 0), as in every experiment of the letter.

For every token v with p_v > 0 the nearest other token w of the codebook
(Euclidean distance over all V = 30,522 rows) gives the difference
Delta = e_v - e_w. Real dimensions 2k and 2k+1 (0-based) form subcarrier k,
k = 0..31, so a_k = Delta_{2k}^2 + Delta_{2k+1}^2 is the energy of the
difference on subcarrier k. The effective number of subcarriers is the
participation ratio (sum_k a_k)^2 / sum_k a_k^2, which equals n when the
energy is spread evenly over n subcarriers and one when it sits on one.
It is computed over all 32 subcarriers and over the eight subcarriers
k = u, u + 4, ..., u + 28 that coded OMA on interleaved subcarriers gives
user u (U = 4), averaged over u. Means are weighted by p_v (column
"pv") and taken over all support tokens alike ("uniform"). For reference,
the same ratio for a difference with independent Gaussian entries is
reported as "gauss_ref" (100,000 draws, CPU generator seed 1).

Coded OMA: the minimum number of a user's eight subcarriers on which two
RM(2,5) codewords differ is 4 with the interleaver and 2 without
(pm_oma_coded_il.json, min_subcarriers and plain_min_subcarriers).

Output: data/pm/pm_spread.json. Run on one GPU from code/:
  python exp_pm_spread.py
"""
from __future__ import annotations

import json

import torch

import exp_pm as pm
from sse_lib import DEVICE


def pratio(a: torch.Tensor) -> torch.Tensor:
    """Participation ratio along the last axis."""
    return a.sum(-1) ** 2 / (a ** 2).sum(-1).clamp_min(1e-30)


def main():
    src = pm.Source("bert")
    pdc = src.dist()
    E = pm.codebook(src.W, 64, "pwhite", pd=pdc).to(DEVICE)       # (V, 64), unit norm
    pd = pdc.to(DEVICE)
    sup = torch.nonzero(pd > 0).squeeze(1)
    nn = torch.empty_like(sup)
    sq = (E * E).sum(1)
    for i in range(0, len(sup), 2048):
        v = sup[i:i + 2048]
        d2 = sq[v, None] - 2 * E[v] @ E.T + sq[None]
        d2[torch.arange(len(v), device=DEVICE), v] = float("inf")
        nn[i:i + 2048] = d2.argmin(1)
    Dl = E[sup] - E[nn]                                            # (S, 64)
    a = Dl.view(-1, 32, 2).pow(2).sum(-1)                          # (S, 32) energy per subcarrier
    n32 = pratio(a)
    n8 = torch.stack([pratio(a[:, u::4]) for u in range(4)], 1).mean(1)
    w = pd[sup] / pd[sup].sum()
    g = torch.Generator(device="cpu").manual_seed(1)
    Z = torch.randn(100_000, 64, generator=g).view(-1, 32, 2).pow(2).sum(-1)
    il = json.load(open(pm.DATA / "pm_oma_coded_il.json"))
    out = {
        "codebook": "pwhite BERT base uncased, seed 0, d = 64",
        "support_tokens": int(len(sup)),
        "nn_distance_pv": float((w * Dl.norm(dim=1)).sum()),
        "eff_subcarriers_of_32_pv": float((w * n32).sum()),
        "eff_subcarriers_of_32_uniform": float(n32.mean()),
        "eff_subcarriers_of_user8_pv": float((w * n8).sum()),
        "eff_subcarriers_of_user8_uniform": float(n8.mean()),
        "eff_user8_q05_q50_q95_uniform": [float(q) for q in torch.quantile(n8.float().cpu(), torch.tensor([0.05, 0.5, 0.95]))],
        "gauss_ref_of_32": float(pratio(Z).mean()),
        "gauss_ref_of_user8": float(torch.stack([pratio(Z[:, u::4]) for u in range(4)], 1).mean()),
        "coded_oma_min_subcarriers_interleaved": il["min_subcarriers"],
        "coded_oma_min_subcarriers_plain": il["plain_min_subcarriers"],
    }
    (pm.DATA / "pm_spread.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
