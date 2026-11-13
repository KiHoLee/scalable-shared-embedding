"""Earlier design: near-far in an uplink with a separate gain per user, and the
mechanisms behind the robustness rows, on the released models (no training).

Uplink model: z_k = sum_u g_u x_{u,k} + w_k, synchronous users, flat fading,
g_u = 10^(off_u/20) CN(0,1), each gain known, one zero-forcing and decoding
branch per user. Offsets {0}, {+-6}, {+-12} dB as in Table II.

Layouts of the proposal:
  trained : masks as trained
  aligned : a fixed reordering of the L coordinates that places each user on
            whole subcarriers, off-support mask entries zeroed. Unchanged in
            the downlink up to the zeroed entries.

Also, on the downlink simulator rev_channel.py:
  runs      : flat, offsets, NMSE, T=4 and T=8 over the six 4,000-iteration runs
  sign      : sign patterns on the same masks (tests the codeword shape)
  oma_inter : OMA with the same interleaver as the proposal
  floor     : NMSE 0.01, 0.05 at 30 and 40 dB
  combo     : T=4 with NMSE 0.01, 0.05
  even      : even mask spacing under the interleaver
  u8p2      : (Vu, P) = (256, 2) at U = 8

Outputs: data/rev_uplink.csv, data/rev_mechanism.csv
"""
import argparse
import copy
import math

import torch

import rev_channel as rc
from sse_lib import D, DATA, DEVICE, P_MAX, SSE, VU, FullCodebook, write_csv

V = VU ** P_MAX
U = 4
OFFS = {"equal": [0, 0, 0, 0], "d6": [-6, -2, 2, 6], "d12": [-12, -4, 4, 12]}
RUNS = ["sse_P4.pt"] + [f"rev_sse_seed{i}.pt" for i in range(1, 6)]
CH = 100_000


def load(name):
    m = SSE(P=4, users=4)
    m.load_state_dict(torch.load(DATA / name, map_location="cpu"))
    return m.to(DEVICE).eval()


def digits(ids):
    d, x = [], ids.clone()
    for _ in range(P_MAX):
        d.append(x % VU)
        x = x // VU
    return torch.stack(d[::-1], -1)


def cn(*shape):
    return torch.complex(torch.randn(*shape, device=DEVICE),
                         torch.randn(*shape, device=DEVICE)) / math.sqrt(2)


def support(m):
    M = m.masks().detach() ** 2
    return [(M[u] > 0.5).nonzero().flatten().tolist() for u in range(U)]


def aligned(m):
    """Whole-subcarrier layout. Returns None when the supports cannot be ordered."""
    sup = support(m)
    empty = [l for l in range(16) if not any(l in s for s in sup)]
    order = []
    for s in sup:
        order += s
        if len(s) % 2 and empty:
            order.append(empty.pop())
    order += [l for l in empty if l not in order]
    if sorted(order) != list(range(16)):
        return None
    W = torch.zeros_like(m.W.data)
    for u, s in enumerate(sup):
        W[u, s] = m.W.data[u, s]
    n = copy.deepcopy(m)
    n.W.data = W[:, order]
    n.B.data = m.B.data[:, order]
    n.calibrate_power()
    return n


def sign_patterns(m):
    """Sign patterns on the four strongest dimensions of each user."""
    M = m.masks().detach() ** 2
    act = [M[u].topk(4).indices.sort().values for u in range(U)]
    B, W = torch.zeros_like(m.B.data), torch.zeros_like(m.W.data)
    for u, a in enumerate(act):
        W[u, a] = 1.0
        for v in range(16):
            for i, l in enumerate(a.tolist()):
                B[v, l] = 1.0 if (v >> i) & 1 else -1.0
    n = copy.deepcopy(m)
    n.B.data, n.W.data = B, W
    n.calibrate_power()
    return n


def even_spacing(m):
    M = m.masks().detach() ** 2
    act = [M[u].topk(4).indices.sort().values for u in range(U)]
    pi = torch.zeros(16, dtype=torch.long, device=DEVICE)
    for u, a in enumerate(act):
        for i, l in enumerate(a.tolist()):
            pi[i * 4 + u] = l
    n = copy.deepcopy(m)
    n.B.data, n.W.data = m.B.data[:, pi], m.W.data[:, pi]
    n.calibrate_power()
    return n


def user_frames(scheme, m, ids):
    n = ids.shape[0]
    if scheme == "sse":
        dg = digits(ids)
        e = m.unit_codebook().detach()[dg] / math.sqrt(P_MAX)
        x = (e * m.masks().detach()[None, :, None, :]).reshape(n, U, D) / m.c
        return x, dg
    sym = torch.stack([(ids >> k) & 1 for k in range(16)], -1)
    s = rc.pam_levels(2, ids.device)[sym] / math.sqrt(D)
    x = torch.zeros(n, U, D, device=ids.device)
    for u in range(U):
        x[:, u, u * 16:(u + 1) * 16] = s[:, u]
    return x, sym


@torch.no_grad()
def uplink(scheme, m, snr, off, frames, seed=7):
    torch.manual_seed(seed + int(10 * snr))
    err, tot = torch.zeros(U), 0
    sig = math.sqrt(1 / (D * 10 ** (snr / 10)))
    amp = 10 ** (torch.tensor(off, dtype=torch.float32, device=DEVICE) / 20)
    for _ in range(frames // CH):
        ids = torch.randint(V, (CH, U), device=DEVICE)
        x, lab = user_frames(scheme, m, ids)
        xc = torch.complex(x[..., 0::2], x[..., 1::2])
        g = cn(CH, U, 1) * amp[None, :, None]
        z = (g * xc).sum(1) + cn(CH, D // 2) * math.sqrt(2) * sig
        q = z[:, None, :] / g
        r = torch.stack([q.real, q.imag], -1).reshape(CH, U, D)
        w = torch.ones_like(r)
        if scheme == "sse":
            wrong = (rc.sse_scores(m, r, w, "corr").argmax(-1) != lab).any(-1)
        else:
            wrong = (rc.oma_detect(r, w, U, D, 2, 1) != lab).any(-1)
        err += wrong.float().sum(0).cpu()
        tot += CH
    return err / tot


@torch.no_grad()
def uplink_full(m, snr, off, frames, seed=7, chunk=128):
    torch.manual_seed(seed + int(10 * snr))
    err, tot = torch.zeros(U), 0
    sig = math.sqrt(1 / (D * 10 ** (snr / 10)))
    amp = 10 ** (torch.tensor(off, dtype=torch.float32, device=DEVICE) / 20)
    E, Mk, c = m.codebook().detach(), m.masks().detach(), float(m.c)
    for _ in range(frames // chunk):
        ids = torch.randint(V, (chunk, U), device=DEVICE)
        x = torch.stack([E[ids[:, u]] * Mk[u] for u in range(U)], 1) / c
        xc = torch.complex(x[..., 0::2], x[..., 1::2])
        g = cn(chunk, U, 1) * amp[None, :, None]
        z = (g * xc).sum(1) + cn(chunk, D // 2) * math.sqrt(2) * sig
        q = z[:, None, :] / g
        r = torch.stack([q.real, q.imag], -1).reshape(chunk, U, D)
        w = torch.ones_like(r)
        err += (rc.full_scores(m, r, w, "wmd").argmax(-1) != ids).float().sum(0).cpu()
        tot += chunk
    return err / tot


@torch.no_grad()
def downlink(scheme, m, snr, frames, taps=1, nmse=0.0, inter=False, rule="corr",
             offsets=None, seed=777):
    torch.manual_seed(seed + int(10 * snr))
    err, tot = torch.zeros(U), 0
    perm = rc.interleaver(D, P_MAX) if inter else None
    for _ in range(frames // CH):
        ids = torch.randint(V, (CH, U), device=DEVICE)
        xu, lab = user_frames(scheme, m, ids)
        r, w = rc.channel(xu.sum(1), snr, U, taps=taps, nmse=nmse, perm=perm,
                          offsets_db=offsets)
        if scheme == "sse":
            wrong = (rc.sse_scores(m, r, w, rule).argmax(-1) != lab).any(-1)
        else:
            wrong = (rc.oma_detect(r, w, U, D, 2, 1) != lab).any(-1)
        err += wrong.float().sum(0).cpu()
        tot += CH
    return err / tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=1_000_000)
    ap.add_argument("--frames-full", type=int, default=100_000)
    a = ap.parse_args()
    F = a.frames
    # ---- uplink
    rows = []
    hdr = ["layout", "model", "offsets", "snr_db", "ser", "worst"]

    def add(layout, model, off, s, t):
        rows.append((layout, model, off, s, float(t.mean()), float(t.max())))
        print(rows[-1], flush=True)
        write_csv(DATA / "rev_uplink.csv", hdr, rows)
    for name in RUNS:
        m = load(name)
        al = aligned(m)
        for ok, off in OFFS.items():
            for s in (10.0, 20.0):
                add("trained", name, ok, s, uplink("sse", m, s, off, F))
                if al is not None:
                    add("aligned", name, ok, s, uplink("sse", al, s, off, F))
    for ok, off in OFFS.items():
        for s in (10.0, 20.0):
            add("oma", "-", ok, s, uplink("oma", None, s, off, F))
    fc = FullCodebook(V, users=4)
    fc.load_state_dict(torch.load(DATA / f"full_V{V}.pt", map_location="cpu"))
    fc.to(DEVICE).eval()
    for ok, off in OFFS.items():
        for s in (10.0, 20.0):
            add("full", f"full_V{V}.pt", ok, s, uplink_full(fc, s, off, a.frames_full))
    # ---- downlink mechanisms
    rows2 = []
    hdr2 = ["case", "model", "taps", "nmse", "offsets", "snr_db", "ser", "worst"]

    def add2(case, model, taps, nmse, off, s, t):
        rows2.append((case, model, taps, nmse, off, s, float(t.mean()), float(t.max())))
        print(rows2[-1], flush=True)
        write_csv(DATA / "rev_mechanism.csv", hdr2, rows2)
    for s in (10.0, 20.0):
        add2("oma_flat", "-", 1, 0, "equal", s, downlink("oma", None, s, F))
        add2("oma_d12", "-", 1, 0, "d12", s, downlink("oma", None, s, F, offsets=OFFS["d12"]))
        for nm in (0.01, 0.05, 0.1):
            add2("oma_nmse", "-", 1, nm, "equal", s, downlink("oma", None, s, F, nmse=nm))
        for T in (4, 8):
            add2("oma_block", "-", T, 0, "equal", s, downlink("oma", None, s, F, taps=T))
            add2("oma_inter", "-", T, 0, "equal", s, downlink("oma", None, s, F, taps=T, inter=True))
    for name in RUNS:
        m = load(name)
        for s in (10.0, 20.0):
            add2("runs_flat", name, 1, 0, "equal", s, downlink("sse", m, s, F))
            add2("runs_d12", name, 1, 0, "d12", s, downlink("sse", m, s, F, offsets=OFFS["d12"]))
            for nm in (0.01, 0.05):
                add2("runs_nmse", name, 1, nm, "equal", s, downlink("sse", m, s, F, nmse=nm))
            for T in (4, 8):
                add2("runs_fs", name, T, 0, "equal", s,
                     downlink("sse", m, s, F, taps=T, inter=True, rule="wmd"))
    m = load("sse_P4.pt")
    sg, ev, al = sign_patterns(m), even_spacing(m), aligned(m)
    for s in (10.0, 20.0):
        add2("sign_flat", "sse_P4.pt", 1, 0, "equal", s, downlink("sse", sg, s, F))
        for nm in (0.05, 0.1):
            add2("sign_nmse", "sse_P4.pt", 1, nm, "equal", s, downlink("sse", sg, s, F, nmse=nm))
            add2("aligned_nmse", "sse_P4.pt", 1, nm, "equal", s, downlink("sse", al, s, F, nmse=nm))
        for T in (4, 8):
            add2("sign_fs", "sse_P4.pt", T, 0, "equal", s,
                 downlink("sse", sg, s, F, taps=T, inter=True, rule="wmd"))
            add2("aligned_fs", "sse_P4.pt", T, 0, "equal", s,
                 downlink("sse", al, s, F, taps=T, inter=True, rule="wmd"))
            add2("even_fs", "sse_P4.pt", T, 0, "equal", s,
                 downlink("sse", ev, s, F, taps=T, inter=True, rule="wmd"))
        for nm in (0.01, 0.05):
            add2("combo", "sse_P4.pt", 4, nm, "equal", s,
                 downlink("sse", m, s, F, taps=4, nmse=nm, inter=True, rule="wmd"))
            add2("combo_oma", "-", 4, nm, "equal", s, downlink("oma", None, s, F, taps=4, nmse=nm))
    for nm in (0.01, 0.05):
        for s in (30.0, 40.0):
            add2("floor", "sse_P4.pt", 1, nm, "equal", s, downlink("sse", m, s, F, nmse=nm))
            add2("floor_oma", "-", 1, nm, "equal", s, downlink("oma", None, s, F, nmse=nm))
    # (256,2) at U=8 on the complex simulator
    sd = torch.load(DATA / "rev_sse_U8_P2.pt", map_location="cpu")
    vu, L = sd["B"].shape
    x = SSE(P=64 // L, vu=vu, users=sd["W"].shape[0], d=64)
    x.load_state_dict(sd)
    x.to(DEVICE).eval()
    for s in (10.0, 20.0):
        torch.manual_seed(777 + int(10 * s))
        e, n = 0.0, 0
        with torch.no_grad():
            for _ in range(F // 20000):
                dg = torch.randint(vu, (20000, x.users, x.P), device=DEVICE)
                fr = x.tx_frame(dg).reshape(20000, 64) / x.c
                r, w = rc.channel(fr, s, x.users)
                e += (rc.sse_scores(x, r, w, "corr").argmax(-1) != dg).any(-1).float().mean().item()
                n += 1
        rows2.append(("u8_vu256_p2", "rev_sse_U8_P2.pt", 1, 0, "equal", s, e / n, float("nan")))
        print(rows2[-1], flush=True)
        write_csv(DATA / "rev_mechanism.csv", hdr2, rows2)


if __name__ == "__main__":
    main()
