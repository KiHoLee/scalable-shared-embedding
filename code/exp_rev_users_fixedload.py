"""Earlier design, fixed load: the number of users grows with
the frame, d = 16U, so every user keeps 16 real dimensions per frame and
16 index bits, as in the main configuration (U = 4, d = 64). Proposed
SSE with Vu = 16 and P = 4 (L = 4U) is retrained per U under the main
budget; OMA keeps 16 dimensions and 16 BPSK bits per user, so its SER
equals the main-configuration value at every U (closed form).
Channel, SNR definition gamma = 1/(d sigma^2), and evaluation follow the
letter's real model, with the frame dimension set to d.

Output: data/rev_users_fixedload.csv
"""
from __future__ import annotations

import torch

import sse_lib
from sse_lib import (DATA, SSE, train_sse, eval_ser_sse, oma_ser, set_seed,
                     write_csv)


def main():
    rows = []
    oma = oma_ser([10.0, 20.0])
    for users in (2, 4, 8, 16):
        d = 16 * users
        sse_lib.D = d            # sigma^2 = 1/(d gamma) inside the library
        m = SSE(P=4, vu=16, d=d, users=users)
        path = DATA / f"rev_sse_fixedload_U{users}.pt"
        if path.exists():
            m.load_state_dict(torch.load(path, map_location="cpu"))
        else:
            set_seed(1)
            train_sse(m)
            torch.save(m.state_dict(), path)
        ser = eval_ser_sse(m, [10.0, 20.0], frames=500_000)
        rows.append((users, d, ser[0], ser[1], oma[0], oma[1]))
        print("[fixedload]", rows[-1], flush=True)
    sse_lib.D = 64
    write_csv(DATA / "rev_users_fixedload.csv",
              ["U", "d", "ser10_SSE", "ser20_SSE", "ser10_OMA", "ser20_OMA"],
              rows)


if __name__ == "__main__":
    main()
