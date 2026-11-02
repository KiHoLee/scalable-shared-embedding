"""Revision R1.6: error floor under imperfect channel estimates, flat fading.
Table II stops at 20 dB. This script holds the NMSE fixed and lets the noise
vanish, so the values above the trained range of 0-20 dB are limit cases and
not operating points (a pilot-based estimate would improve with the SNR).

Model. U = 4, V = 65,536, d = 64, flat fading of rev_channel.py with the
estimate g_hat = g + e, e ~ CN(0, nmse), one e per user and frame, NMSE in
{0.01, 0.05}. The six runs sse_P4.pt (reported model) and rev_sse_seed1..5 are
used as trained on perfect CSI, no interleaver, under correlation (corr, the
letter's rule) and minimum distance (md). OMA is plain BPSK per real dimension
as in Table II.

SNR grid: 30, 40, 60, 80 dB, plus 10 and 20 dB to tie the file to
rev_robust.csv. Seeds follow exp_rev_robust.run (777 + 10*snr).

Output: data/rev_csifloor.csv, average and worst-user SER, one row per NMSE,
SNR, model and rule.
"""
import argparse

import torch

from sse_lib import DATA, DEVICE, P_MAX, SSE, write_csv
from exp_rev_robust import run

U = 4
MODELS = [("sse_P4", "sse_P4.pt")] + [(f"seed{i}", f"rev_sse_seed{i}.pt") for i in range(1, 6)]
NMSES = (0.01, 0.05)
SNRS = (10.0, 20.0, 30.0, 40.0, 60.0, 80.0)
CHUNK = 100_000
SEED = 777


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=4_000_000)
    a = ap.parse_args()
    hdr = ["nmse", "snr_db", "model", "rule", "frames", "ser_mean", "ser_worst"]
    rows = []
    models = []
    for tag, name in MODELS:
        m = SSE(P=P_MAX, users=U)
        m.load_state_dict(torch.load(DATA / name, map_location="cpu"))
        models.append((tag, m.to(DEVICE).eval()))
    for nmse in NMSES:
        for snr in SNRS:
            v = run("oma", None, U, snr, a.frames, CHUNK, SEED, nmse=nmse)
            rows.append([nmse, snr, "OMA", "bpsk", a.frames, float(v.mean()), float(v.max())])
            print("[csi]", rows[-1], flush=True)
            for tag, m in models:
                for rule in ("corr", "md"):
                    v = run("sse", m, U, snr, a.frames, CHUNK, SEED, nmse=nmse, rule=rule)
                    rows.append([nmse, snr, tag, rule, a.frames, float(v.mean()), float(v.max())])
                    print("[csi]", rows[-1], flush=True)
            write_csv(DATA / "rev_csifloor.csv", hdr, rows)


if __name__ == "__main__":
    main()
