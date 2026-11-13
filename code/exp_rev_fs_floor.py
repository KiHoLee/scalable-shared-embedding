"""Earlier design: error floor of the proposal under frequency-selective fading.
Table II stops at 20 dB. This script extends the frequency-selective rows in
SNR and over the training runs.

Model. U = 4, V = 65,536, d = 64, channel of rev_channel.py, perfect CSI, T = 4
or 8 equal-power taps over 32 subcarriers. The six runs sse_P4.pt (reported
model) and rev_sse_seed1..5 are used as trained on flat fading with the
interleaver, under weighted minimum distance (wmd, weights |g_k|^2) and
unweighted minimum distance (md).
Variant `zeroed` applies to sse_P4.pt only: the mask entries off each user's
four largest dimensions are set to zero, without retraining, and c is
recalibrated. For sse_P4.pt these supports cover all 16 dimensions of a
period, which does not hold for the other five runs. OMA is plain BPSK per real dimension on contiguous blocks, as in
Table II. Reference rows at T = 1 (flat, no interleaver, correlation and md)
compare the reported model with its zeroed variant.

SNR grid: 10, 15, 20, 22, 24, 25, 30, 40 dB. Seeds follow exp_rev_robust.run
(777 + 10*snr).

Output: data/rev_fs_floor.csv, one row per taps, model, variant, rule and SNR,
with average, worst-user and per-user SER and the OMA row of the same point.
data/rev_fs_floor_crossover.csv lists, per taps, model, variant, rule and metric
(average or worst user), every crossing with OMA on the grid: kind `up` means the
proposal goes above OMA as the SNR grows, `down` below it, with log10 SER
interpolated linearly between grid points. With no crossing, kind is
`always_above` or `always_below` and snr_db is empty.
"""
import argparse
import copy
import math

import torch

from sse_lib import DATA, DEVICE, P_MAX, SSE, set_seed, write_csv
from exp_rev_robust import run

U = 4
MODELS = [("sse_P4", "sse_P4.pt")] + [(f"seed{i}", f"rev_sse_seed{i}.pt") for i in range(1, 6)]
SNRS = (10.0, 15.0, 20.0, 22.0, 24.0, 25.0, 30.0, 40.0)
TAPS = (4, 8)
CHUNK = 100_000
SEED = 777


def load(name):
    m = SSE(P=P_MAX, users=U)
    m.load_state_dict(torch.load(DATA / name, map_location="cpu"))
    return m.to(DEVICE).eval()


def zero_off_support(m):
    """Copy of m with each mask kept only on its four largest entries."""
    n = copy.deepcopy(m)
    M = m.masks().detach()
    top = (M ** 2).topk(4, dim=1).indices
    assert len(set(top.reshape(-1).tolist())) == 16, "supports must cover all 16 dimensions"
    W = torch.zeros_like(n.W.data)
    for u in range(U):
        W[u, top[u]] = M[u, top[u]]
    n.W.data = W
    set_seed(1)
    n.calibrate_power()
    off = [float(1 - (M[u, top[u]] ** 2).sum() / (M[u] ** 2).sum()) for u in range(U)]
    largest = [float((M[u] ** 2).index_fill(0, top[u], 0).max()) for u in range(U)]
    print("[zeroed] supports", top.sort(1).values.tolist(),
          "off-support energy", [f"{x:.4f}" for x in off],
          "largest squared off-support entry", [f"{x:.4f}" for x in largest],
          f"c {float(m.c):.4f} -> {float(n.c):.4f}", flush=True)
    return n


def crossings(snrs, ser, oma):
    """All sign changes of log10(ser/oma) on the grid, as (kind, snr) pairs."""
    d = [math.log10(s) - math.log10(o) for s, o in zip(ser, oma)]
    out = []
    for k in range(1, len(snrs)):
        if (d[k - 1] < 0) != (d[k] < 0):
            t = -d[k - 1] / (d[k] - d[k - 1])
            out.append(("up" if d[k] >= 0 else "down", snrs[k - 1] + t * (snrs[k] - snrs[k - 1])))
    return out or [("always_above" if d[0] >= 0 else "always_below", "")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=4_000_000)
    ap.add_argument("--frames_zeroed", type=int, default=50_000_000,
                    help="zeroed variant at 30 and 40 dB, where errors are rare")
    a = ap.parse_args()
    hdr = ["taps", "model", "variant", "rule", "snr_db", "frames", "ser_mean", "ser_worst",
           "ser_u0", "ser_u1", "ser_u2", "ser_u3", "oma_mean", "oma_worst"]
    rows = []
    oma = {}
    for T in (1, 4, 8):
        for snr in SNRS:
            oma[(T, snr)] = run("oma", None, U, snr, a.frames, CHUNK, SEED, taps=T)
            v = oma[(T, snr)]
            print("[oma]", T, snr, f"{float(v.mean()):.5f}/{float(v.max()):.5f}", flush=True)

    def add(T, model, variant, rule, snr, fr, v):
        o = oma[(T, snr)]
        rows.append([T, model, variant, rule, snr, fr, float(v.mean()), float(v.max())]
                    + [float(x) for x in v] + [float(o.mean()), float(o.max())])
        print("[fs]", T, model, variant, rule, snr, f"{float(v.mean()):.5f}/{float(v.max()):.5f}", flush=True)
        write_csv(DATA / "rev_fs_floor.csv", hdr, rows)

    for T in TAPS:
        for snr in SNRS:
            o = oma[(T, snr)]
            rows.append([T, "OMA", "plain", "bpsk", snr, a.frames, float(o.mean()), float(o.max())]
                        + [float(x) for x in o] + [float(o.mean()), float(o.max())])
    for tag, name in MODELS:
        m = load(name)
        for T in TAPS:
            for rule in ("wmd", "md"):
                for snr in SNRS:
                    add(T, tag, "trained", rule, snr, a.frames,
                        run("sse", m, U, snr, a.frames, CHUNK, SEED, taps=T, inter=True, rule=rule))
    m = load("sse_P4.pt")
    z = zero_off_support(m)
    for T in TAPS:
        for rule in ("wmd", "md"):
            for snr in SNRS:
                fr = a.frames_zeroed if snr >= 30 else a.frames
                add(T, "sse_P4", "zeroed", rule, snr, fr,
                    run("sse", z, U, snr, fr, CHUNK, SEED, taps=T, inter=True, rule=rule))
    for variant, x in (("trained", m), ("zeroed", z)):
        for rule in ("corr", "md"):
            for snr in SNRS:
                add(1, "sse_P4", variant, rule, snr, a.frames,
                    run("sse", x, U, snr, a.frames, CHUNK, SEED, taps=1, rule=rule))

    cross = []
    keys = sorted({(r[0], r[1], r[2], r[3]) for r in rows if r[1] != "OMA" and r[0] != 1})
    for T, model, variant, rule in keys:
        sel = sorted((r for r in rows if r[:4] == [T, model, variant, rule]), key=lambda r: r[4])
        s = [r[4] for r in sel]
        for metric, i_s, i_o in (("mean", 6, 12), ("worst", 7, 13)):
            for kind, x in crossings(s, [max(r[i_s], 1e-9) for r in sel], [r[i_o] for r in sel]):
                cross.append([T, model, variant, rule, metric, kind, x])
                print("[cross]", cross[-1], flush=True)
    write_csv(DATA / "rev_fs_floor_crossover.csv",
              ["taps", "model", "variant", "rule", "metric", "kind", "snr_db"], cross)


if __name__ == "__main__":
    main()
