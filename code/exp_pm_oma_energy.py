"""Average frame energy of the OMA reference on the test streams.

OMA sends the 16 bits of each token index uncoded on d/U real dimensions
(BPSK at U = 2 and 4, natural-mapped 4-PAM at U = 8). The levels have unit
average energy only under equiprobable symbols, and the top bit of every
index is zero (V < 2^15), so this script measures the frame energy that the
test streams actually produce. No channel, no training.

Run from the code folder:
    python exp_pm_oma_energy.py
Writes data/pm/pm_oma_energy.csv (source, users, M, rep, frame_energy,
energy_top_symbol, frac_top_bit_zero).
"""
from __future__ import annotations

import torch

import exp_pm as ep
import rev_channel as rc


def main():
    rows = []
    for kind in ("bert", "gtr"):
        src = ep.Source(kind)
        for users in (2, 4, 8):
            ids, _ = src.streams(users)
            ids = ids.long()
            n_sym, M, rep = rc.oma_layout(users, ep.D, ep.BITS)
            bps = int(M).bit_length() - 1
            sym = torch.stack([(ids >> (bps * k)) & (M - 1) for k in range(n_sym)], -1)
            x = rc.oma_frames(sym, users, ep.D, M, rep)            # (N, d)
            e = float(x.pow(2).sum(-1).mean())
            lv = rc.pam_levels(M, sym.device)
            e_top = float(lv[sym[..., -1]].pow(2).mean())
            rows.append([kind, users, M, rep, e, e_top, float((ids >> 15).eq(0).float().mean())])
            print(kind, users, M, rep, f"frame energy {e:.4f}  top-symbol energy {e_top:.4f}",
                  flush=True)
    ep.write_csv(ep.DATA / "pm_oma_energy.csv",
                 ["source", "users", "M", "rep", "frame_energy", "energy_top_symbol",
                  "frac_top_bit_zero"], rows)


if __name__ == "__main__":
    main()
