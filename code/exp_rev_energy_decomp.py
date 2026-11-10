"""Revision R1.3/R1.6: decomposition of the frame energy on the real streams and
the largest cosine between masks. No training.

Output: data/rev_energy_decomp.json
  sse/full/todma: mean, min, max, std of the frame energy on the AG News streams
  and on uniform indices, per-user masked energy, and the part from inner
  products between users (same token or digit, and distinct).
  masks: largest normalized inner product between two masks per trained model.
"""
import itertools
import json
import math

import torch

from exp_rev_semantic import digits_to_ids, ids_to_digits, load_streams  # noqa: F401
from exp_sota import make_dictionary
from sse_lib import D, DATA, DEVICE, P_MAX, SSE, U, VU, FullCodebook

V = VU ** P_MAX


def stats(t):
    return {"mean": float(t.mean()), "min": float(t.min()), "max": float(t.max()), "std": float(t.std())}


@torch.no_grad()
def main():
    _, ids, _ = load_streams()
    ids = ids.to(DEVICE)
    out = {"frames": ids.shape[0]}
    same = torch.tensor([float((ids[:, a] == ids[:, b]).float().mean())
                         for a, b in itertools.combinations(range(U), 2)])
    out["same_token_pair_fraction"] = float(same.mean())
    # proposal
    m = SSE(P=P_MAX)
    m.load_state_dict(torch.load(DATA / "sse_P4.pt", map_location="cpu"))
    m.to(DEVICE).eval()
    B, Mk, c = m.unit_codebook(), m.masks(), float(m.c)

    def sse_parts(dg):
        xx = (B[dg] / math.sqrt(P_MAX)) * Mk[None, :, None, :]
        self_e = xx.pow(2).sum((1, 2, 3)) / c ** 2
        tot = xx.sum(1).pow(2).sum((1, 2)) / c ** 2
        return tot, self_e, tot - self_e
    dg = ids_to_digits(ids)
    t, s, x = sse_parts(dg)
    out["sse_real"] = {"frame": stats(t), "self": stats(s), "cross": stats(x)}
    tu, su, xu = sse_parts(torch.randint(VU, (2_000_000, U, P_MAX), device=DEVICE))
    out["sse_uniform"] = {"frame": stats(tu), "self": stats(su), "cross": stats(xu)}
    # full codebook
    f = FullCodebook(V)
    f.load_state_dict(torch.load(DATA / f"full_V{V}.pt", map_location="cpu"))
    f.to(DEVICE).eval()
    fc = float(f.c)

    def full_e(idx):
        es = []
        for k in range(0, idx.shape[0], 8192):
            y = f.tx_frame(idx[k:k + 8192])
            es.append(y.pow(2).sum(tuple(range(1, y.dim()))) / fc ** 2)
        return torch.cat(es)
    out["full_real"] = {"frame": stats(full_e(ids))}
    out["full_uniform"] = {"frame": stats(full_e(torch.randint(V, (200_000, U), device=DEVICE)))}
    # ToDMA
    A = make_dictionary(D, V).to(DEVICE)
    cols = A.T[ids]
    tot = (cols.sum(1) / math.sqrt(U)).pow(2).sum(1)
    s_c, d_c = 0.0, 0.0
    for a, b in itertools.combinations(range(U), 2):
        ip = (cols[:, a] * cols[:, b]).sum(-1)
        eq = ids[:, a] == ids[:, b]
        s_c += float(ip[eq].sum())
        d_c += float(ip[~eq].sum())
    N = ids.shape[0]
    out["todma_real"] = {"frame": stats(tot), "same_pairs": 2 / U * s_c / N, "distinct_pairs": 2 / U * d_c / N}
    # masks
    res = {}
    names = {"d64_U2": ("rev_sse_U2.pt", 2, 64), "d64_U4": ("sse_P4.pt", 4, 64),
             "d64_U8": ("rev_sse_U8.pt", 8, 64), "d64_U16": ("rev_sse_U16.pt", 16, 64)}
    for u in (2, 4, 8, 16):
        names[f"load_U{u}"] = (f"rev_sse_fixedload_U{u}.pt", u, 16 * u)
    for k, (fn, u, d) in names.items():
        x = SSE(P=4, vu=16, d=d, users=u)
        x.load_state_dict(torch.load(DATA / fn, map_location="cpu"))
        W = x.masks().detach()
        Wn = W / W.norm(dim=1, keepdim=True)
        g = (Wn @ Wn.T - torch.eye(u)).abs().max()
        res[k] = float(g)
    out["mask_max_abs_cos"] = res
    (DATA / "rev_energy_decomp.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1), flush=True)


if __name__ == "__main__":
    main()
