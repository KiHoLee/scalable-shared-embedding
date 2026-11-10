"""High-SNR points of the channel figure (Fig. 2) with ten times the frames.

At 12-20 dB the static entry and the adapted masks make few errors in the
40,000 frames of exp_pm_ext.py --part figs (graph g1), so their curves are
noisy or end at zero errors. This script repeats those points with 400,000
frames. The seeds and the chunking are those of part figs, so the first
39,936 frames of each point are drawn as there (the 40,000 frames of part
figs end with a shorter chunk). The constant c of the adapted masks is
drawn with seed 0, as in part sig, whereas part figs leaves that draw
unseeded, so the adapted points can differ from graph g1 by a few errors on
the shared frames.

Run from the code folder (GPU recommended), after exp_pm.py --part channel:
    python exp_pm_chanhi.py --part hi      # data/pm/pm_figs_hi.csv
    python exp_pm_chanhi.py --part class   # data/pm/pm_figs_class.csv
    python exp_pm_chanhi.py --part oma_adapt   # data/pm/pm_oma_adapt.csv
Part class evaluates the static entries trained for each frequency-selective
class, which the figure shows as the static entry. The figure takes only the
adapted masks from part hi; its flat-trained static entry is stored but not
drawn.
"""
from __future__ import annotations

import torch

import exp_pm as pm
import exp_pm_ext as px
from sse_lib import DEVICE, D, write_csv

FRAMES = 400_000


def setup():
    src = pm.Source("bert")
    pdc = src.dist()
    E = pm.codebook(src.W, D, "pwhite", pd=pdc).to(DEVICE)
    pd = pdc.to(DEVICE)
    return E, pd


def load(name, L=16):
    sd = torch.load(pm.MODELS / f"{name}.pt", map_location="cpu")
    m = pm.Masks(sd["E"], L, px.U)
    m.load_state_dict(sd)
    return m.to(DEVICE).eval()


def part_hi():
    """Static entry trained under flat fading and adapted masks at 12-20 dB
    with 400,000 frames (data/pm/pm_figs_hi.csv)."""
    E, pd = setup()
    comb = torch.zeros(px.U, D, device=DEVICE)
    for u in range(px.U):
        comb[u, u::4] = 2.0
    torch.manual_seed(0)                       # fixed draw for the constant c
    idx0 = px.sample(pd, 65536, px.U, 0.0)
    c_comb = float(((E[idx0] * comb[None]).sum(1)).pow(2).sum(1).mean().sqrt())
    static = load("white_U4_L16_flat_s1")
    rows = []
    for taps, cn in ((4, "fs4"), (8, "fs8")):
        for snr in (12, 14, 16, 18, 20):
            v_st = px.ser(static, pd, 0.0, snr, pm.COND[cn], frames=FRAMES)
            v_ad = px.run_adapt(E, pd, c_comb, snr, taps, 2, False, frames=FRAMES)
            rows.append(["g1", cn, "static_L16", snr, FRAMES, v_st])
            rows.append(["g1", cn, "adapted_B2", snr, FRAMES, v_ad])
            print("[chanhi]", cn, snr, f"static {v_st:.3e} adapted {v_ad:.3e}", flush=True)
    write_csv(pm.DATA / "pm_figs_hi.csv", ["graph", "channel", "scheme", "snr_db", "frames", "ser"], rows)


def part_class():
    """Static entries trained for the channel class itself (exp_pm.py --part
    channel: white_U4_L16_fs4_s1, white_U4_L16_fs8_s1), as the mask set
    stores them, at 0-20 dB; 40,000 frames below 12 dB and 400,000 frames
    from 12 dB on (data/pm/pm_figs_class.csv)."""
    E, pd = setup()
    rows = []
    for cn in ("fs4", "fs8"):
        m = load(f"white_U4_L16_{cn}_s1")
        for snr in range(0, 21, 2):
            fr = FRAMES if snr >= 12 else 40_000
            v = px.ser(m, pd, 0.0, snr, pm.COND[cn], frames=fr)
            rows.append(["g1", cn, "static_class_L16", snr, fr, v])
            print("[class]", cn, snr, fr, f"{v:.3e}", flush=True)
    write_csv(pm.DATA / "pm_figs_class.csv", ["graph", "channel", "scheme", "snr_db", "frames", "ser"], rows)


@torch.no_grad()
def run_oma(snr, taps, adapt, frames=200_000, seed=61):
    """SER of OMA with each user's 16 uncoded bits on 16 real dimensions,
    one bit per dimension, detected by sign after equalization with exact
    gains. Without adapt every user owns eight adjacent subcarriers, as in
    Fig. 2. With adapt every user owns the blocks of two subcarriers that the
    adapted masks of Fig. 2 give it (px.adaptive_masks, B = 2)."""
    import math
    torch.manual_seed(seed + int(10 * snr) + 100 * taps + (1000 if adapt else 0))
    K = D // 2
    sig = math.sqrt(1.0 / (D * 10 ** (snr / 10)))
    ar = torch.arange(D, device=DEVICE)
    err = 0
    for k0 in range(0, frames, 4096):
        n = min(4096, frames - k0)
        g = px.gains(n, taps, DEVICE)                                   # (n,U,K)
        if adapt:
            own = px.adaptive_masks(g, 2, sig * sig, 1.0, False) > 0     # (n,U,D)
        else:
            own = (ar[None, None, :] // (D // px.U)
                   == torch.arange(px.U, device=DEVICE)[None, :, None]).expand(n, px.U, D)
        bits = torch.randint(0, 2, (n, px.U, D // px.U), device=DEVICE)
        sym = (2.0 * bits - 1.0) / math.sqrt(D)                          # unit frame energy
        x = torch.zeros(n, D, device=DEVICE)
        pos = own.float().argsort(-1, descending=True)[..., :D // px.U]  # (n,U,16) owned dims
        x.scatter_(1, pos.reshape(n, -1), sym.reshape(n, -1))
        xc = torch.complex(x[:, 0::2], x[:, 1::2])
        w = torch.complex(torch.randn(n, px.U, K, device=DEVICE),
                          torch.randn(n, px.U, K, device=DEVICE)) * sig
        q = (g * xc[:, None, :] + w) / g
        r = torch.stack([q.real, q.imag], -1).reshape(n, px.U, D)
        rb = r.gather(2, pos)
        wrong = ((rb > 0).long() != bits).any(-1)
        err += int(wrong.sum())
    return err / (frames * px.U)


def part_oma_adapt():
    """OMA on fixed adjacent subcarriers and on the adapted blocks of two
    subcarriers, 4 and 8 taps, 0-20 dB, 200,000 frames
    (data/pm/pm_oma_adapt.csv). It separates the gain of choosing strong
    subcarriers from the gain of the codeword."""
    rows = []
    for taps, cn in ((4, "fs4"), (8, "fs8")):
        for snr in range(0, 21, 2):
            v_f = run_oma(snr, taps, False)
            v_a = run_oma(snr, taps, True)
            rows.append(["g1", cn, "oma", snr, 200_000, v_f])
            rows.append(["g1", cn, "oma_adapted_B2", snr, 200_000, v_a])
            print("[oma_adapt]", cn, snr, f"fixed {v_f:.4e} adapted {v_a:.4e}", flush=True)
    write_csv(pm.DATA / "pm_oma_adapt.csv", ["graph", "channel", "scheme", "snr_db", "frames", "ser"], rows)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="hi", choices=["hi", "class", "oma_adapt"])
    a = ap.parse_args()
    {"hi": part_hi, "class": part_class, "oma_adapt": part_oma_adapt}[a.part]()


if __name__ == "__main__":
    main()
