"""Revision analysis R1.6: how much the residual interference matters as users are added.

For the trained proposal at U = 2, 4, 8, 16 with the fixed frame d = 64, on the
flat complex channel of rev_channel.py, evaluate the SER on paired frames with and
without the signals of the other users in the received frames, for the correlation
rule and for minimum-distance decoding, without noise and at 10, 20, and 40 dB.
Also write the largest absolute mask cosine and delta_u^2/(4 rho_u) of each model.

Output: data/rev_users_interf.csv
"""
import math

import torch

import rev_channel as rc
from exp_rev_analysis import rho_closed_form
from sse_lib import DATA, SSE, write_csv

torch.set_num_threads(16)
FRAMES = 1_000_000
CHUNK = 20_000
MODELS = ((2, "rev_sse_U2.pt"), (4, "sse_P4.pt"), (8, "rev_sse_U8.pt"), (16, "rev_sse_U16.pt"))


def main():
    rows = []
    for users, name in MODELS:
        m = SSE(P=4, vu=16, d=64, users=users)
        m.load_state_dict(torch.load(DATA / name, map_location="cpu"))
        m.calibrate_power()
        mk = m.masks()
        nm = mk / mk.norm(dim=1, keepdim=True)
        maxcos = float((nm @ nm.T - torch.eye(users)).abs().max())
        rho = torch.tensor(rho_closed_form(m))
        b, c = m.unit_codebook(), float(m.c)
        d2 = []
        for u in range(users):
            cand = b * mk[u]
            dist = torch.cdist(cand, cand) + torch.eye(16) * 1e9
            d2.append(float(dist.min() ** 2) / (c ** 2 * 4))
        d2 = torch.tensor(d2)
        ratio = float((d2.mean() / (4 * rho.mean())))
        for snr in (None, 10.0, 20.0, 40.0):
            torch.manual_seed(777)
            err = {k: torch.zeros(users) for k in ("corr_all", "corr_noint", "md_all", "md_noint")}
            for _ in range(FRAMES // CHUNK):
                dg = torch.randint(16, (CHUNK, users, 4))
                own = (b[dg] / math.sqrt(4) * mk[None, :, None, :] / c).reshape(CHUNK, users, 64)
                x = own.sum(1)
                if snr is None:
                    r = x[:, None, :].expand(CHUNK, users, 64).clone()
                    w = torch.ones_like(r)
                else:
                    r, w = rc.channel(x, snr, users)
                r_noint = r - (x[:, None, :] - own)
                for rule in ("corr", "md"):
                    err[rule + "_all"] += (rc.sse_scores(m, r, w, rule).argmax(-1) != dg).any(-1).float().sum(0)
                    err[rule + "_noint"] += (rc.sse_scores(m, r_noint, w, rule).argmax(-1) != dg).any(-1).float().sum(0)
            ser = {k: float(v.mean()) / (FRAMES // CHUNK * CHUNK) for k, v in err.items()}
            rows.append([users, name, "inf" if snr is None else snr, maxcos, ratio, FRAMES // CHUNK * CHUNK,
                         ser["corr_all"], ser["corr_noint"], ser["md_all"], ser["md_noint"]])
            print(rows[-1], flush=True)
    write_csv(DATA / "rev_users_interf.csv",
              ["users", "model", "snr_db", "maxcos", "delta2_over_4rho", "frames",
               "corr_all", "corr_noint", "md_all", "md_noint"], rows)


if __name__ == "__main__":
    main()
