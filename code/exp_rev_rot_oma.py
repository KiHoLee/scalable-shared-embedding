"""Revision R1.6: OMA with rotated digits under flat and frequency-selective
fading, with no trained values. It answers whether the lead of the proposal
over plain OMA in Table II is specific to the proposal.

Model. U = 4, d = 64, 16 bits per user per frame, channel of rev_channel.py,
perfect CSI, unit average frame energy. Each user splits its 16 bits into four
4-bit digits. A digit b in {+-1}^4 is sent as R b / sqrt(d) on four real
dimensions, R a fixed 4x4 orthogonal matrix, and is decoded on its own by
16-candidate weighted minimum distance with weights |g_k|^2. A frame error is
any wrong digit. Flat fading uses T = 1; frequency-selective fading uses T = 4
or 8 equal-power taps over 32 subcarriers.

Matrices (column `matrix`):
  C4  cos((2k-1)(2l-1)pi/16)/sqrt(2), k,l = 1..4. All four coordinates of every
      codeword difference are nonzero, minimum product distance 0.354.
      The rotation the letter quotes.
  H4  the 4x4 Hadamard matrix / 2. A difference can fall on 1 coordinate.
  RND the Q factor of torch.randn(4,4) with generator seed 5.
  K4  a second orthogonal matrix, a = 0.37174803,
      b = 0.60150095. Kept only to identify its values. The difference
      (2,2,2,2) falls on 2 coordinates, so its minimum product distance is 0.
  I   no rotation (binary signs), control.

Layouts (column `layout`), physical real dimensions of digit p of user u,
j = 0..3, and subcarrier = dimension // 2:
  comb8   (u+4j)*4+p, subcarriers 2u+p//2+8j, EIGHT APART. Same as logical dims
          p*16+u+4j through rc.interleaver. Uncorrelated fading for T = 4, 8.
  block2  16u+p+4j, subcarriers 8u+p//2+2j, two apart inside the user's own
          block of eight adjacent subcarriers.
  nointer p*16+u+4j, the logical dims of comb8 without the interleaver,
          subcarriers 8p+u//2+2j, two apart in the p-th block of eight.
  contig  16u+4p+j, two adjacent subcarriers, the bit layout of plain OMA.

Reference rows: plainOMA (BPSK per real dimension, contiguous, as in Table II)
and SSE_P4 (reported model, as in Table II: correlation for T = 1, interleaver
and weighted minimum distance for T = 4, 8).

Column dp_min is the minimum product distance over differences in {-2,0,2}^4 and
L_min the smallest number of nonzero coordinates. Seeds follow exp_rev_robust.run
(777 + 10*snr), and frames is the number of frames per row.

Which variant gives which values (T = 4, comb8, average SER at 10 / 20 / 30 dB):
  C4  0.160 / 0.00043 / 1.1e-7    the review's 0.160 / 0.0004
  K4  0.147 / 0.00094 / 8.3e-6    the review's 0.146 / 0.0010 (1e6 frames)
  H4  0.155 / 0.0039  / 2.6e-4
  RND 0.174 / 0.00091 / 9.0e-7
K4 leads at 10 dB, but its slope is that of two coordinates, against four for C4.
The review's 0.183 / 0.005 for a Hadamard matrix is another layout, the trained
mask supports of sse_P4.pt, and is not computed here.

Output: data/rev_rot_oma.csv
"""
import argparse
import itertools
import math

import torch

from sse_lib import DATA, DEVICE, D, VU, P_MAX, SSE, write_csv
from exp_rev_robust import ids_to_digits, load_or_train, run
import rev_channel as rc

U = 4
V = VU ** P_MAX
BITS4 = torch.tensor([[(i >> k) & 1 for k in range(4)] for i in range(16)],
                     dtype=torch.float32) * 2 - 1                    # (16,4)
A4, B4 = 0.37174803, 0.60150095
TAPS = (1, 4, 8)
SNRS = (10.0, 20.0, 30.0)


def rotations():
    C4 = torch.tensor([[math.cos((2 * k - 1) * (2 * l - 1) * math.pi / 16) / math.sqrt(2)
                        for l in range(1, 5)] for k in range(1, 5)],
                      dtype=torch.float32)
    H4 = 0.5 * torch.tensor([[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]],
                            dtype=torch.float32)
    K4 = torch.tensor([[-A4, -B4, -B4, -A4], [-B4, -A4, A4, B4],
                       [-B4, A4, A4, -B4], [-A4, B4, -B4, A4]], dtype=torch.float32)
    g = torch.Generator().manual_seed(5)
    Q, _ = torch.linalg.qr(torch.randn(4, 4, generator=g))
    return {"C4": C4, "H4": H4, "RND": Q, "K4": K4, "I": torch.eye(4)}


def layouts():
    """Physical real dimensions, (U, P, 4), one tensor per layout."""
    r = range(4)
    return {
        "comb8": torch.tensor([[[(u + 4 * j) * 4 + p for j in r] for p in r] for u in r]),
        "block2": torch.tensor([[[16 * u + p + 4 * j for j in r] for p in r] for u in r]),
        "nointer": torch.tensor([[[p * 16 + u + 4 * j for j in r] for p in r] for u in r]),
        "contig": torch.tensor([[[16 * u + 4 * p + j for j in r] for p in r] for u in r]),
    }


def diversity(R):
    """(minimum product distance, smallest nonzero count) over {-2,0,2}^4."""
    best, lmin = float("inf"), 4
    for d in itertools.product((-2.0, 0.0, 2.0), repeat=4):
        if any(d):
            x = R.double() @ torch.tensor(d, dtype=torch.float64)
            nz = int((x.abs() > 1e-5).sum())
            lmin = min(lmin, nz)
            best = min(best, float(x.abs().prod()) if nz == 4 else 0.0)
    return best, lmin


@torch.no_grad()
def run_rot(R, phys, snr, frames, taps=1, chunk=100_000, seed=777):
    """Per-user SER (U,) of rotated OMA. R (4,4), phys (U,P,4)."""
    torch.manual_seed(seed + int(10 * snr))
    cand = (BITS4 @ R.T).to(DEVICE) / math.sqrt(D)                    # (16,4)
    idx = phys.to(DEVICE)
    err = torch.zeros(U, dtype=torch.float64)
    tot = 0
    for n0 in range(0, frames, chunk):
        n = min(chunk, frames - n0)
        ids = torch.randint(V, (n, U), device=DEVICE)
        dg = ids_to_digits(ids)                                       # (n,U,P)
        x = torch.zeros(n, D, device=DEVICE)
        x.scatter_(1, idx.reshape(1, -1).expand(n, -1), cand[dg].reshape(n, -1))
        r, w = rc.channel(x, snr, U, taps=taps)                       # physical order
        for u in range(U):
            ii = idx[u].reshape(-1)
            ru = r[:, u, ii].reshape(n, P_MAX, 4)[:, :, None, :]
            wu = w[:, u, ii].reshape(n, P_MAX, 4)[:, :, None, :]
            s = -(wu * (ru - cand[None, None]) ** 2).sum(-1)          # (n,P,16)
            err[u] += float((s.argmax(-1) != dg[:, u]).any(-1).sum())
        tot += n
    return err / tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=10_000_000)
    ap.add_argument("--frames_30", type=int, default=100_000_000,
                    help="rotated rows at 30 dB, where the SER is near 1e-6")
    a = ap.parse_args()
    rots, lays = rotations(), layouts()
    for name, phys in lays.items():
        assert sorted(phys.reshape(-1).tolist()) == list(range(D)), name
    perm = rc.interleaver(D, P_MAX)
    logical = torch.tensor([[[p * 16 + u + 4 * j for j in range(4)] for p in range(4)]
                            for u in range(4)])
    assert torch.equal(perm[logical], lays["comb8"]), "comb8 is not the interleaved comb"
    div = {k: diversity(R) for k, R in rots.items()}
    for k, R in rots.items():
        e = float((R @ R.T - torch.eye(4)).abs().max())
        print(f"[matrix] {k} orth err {e:.1e} dp_min {div[k][0]:.4f} L_min {div[k][1]}", flush=True)
    sse = load_or_train("sse", U, "sse_P4.pt")
    hdr = ["matrix", "layout", "taps", "snr_db", "frames", "dp_min", "L_min",
           "ser_mean", "ser_worst"]
    rows = []

    def add(mat, lay, T, snr, fr, v):
        dp, lm = div.get(mat, ("", ""))
        rows.append([mat, lay, T, snr, fr, dp, lm, float(v.mean()), float(v.max())])
        print("[rot]", mat, lay, T, snr, fr, f"{float(v.mean()):.6f}/{float(v.max()):.6f}", flush=True)
        write_csv(DATA / "rev_rot_oma.csv", hdr, rows)

    for T in TAPS:
        for snr in SNRS:
            fr = a.frames_30 if snr >= 30 else a.frames
            for mat, R in rots.items():
                add(mat, "comb8", T, snr, fr, run_rot(R, lays["comb8"], snr, fr, T))
            if T > 1:
                for lay in ("block2", "nointer", "contig"):
                    for mat in ("C4", "K4"):
                        add(mat, lay, T, snr, fr, run_rot(rots[mat], lays[lay], snr, fr, T))
            add("plainOMA", "contig", T, snr, a.frames,
                run("oma", None, U, snr, a.frames, 100_000, 777, taps=T))
            kw = dict(taps=T, inter=T > 1, rule="wmd" if T > 1 else "corr")
            add("SSE_P4", "interleaved" if T > 1 else "contig", T, snr, a.frames,
                run("sse", sse, U, snr, a.frames, 100_000, 777, **kw))


if __name__ == "__main__":
    main()
