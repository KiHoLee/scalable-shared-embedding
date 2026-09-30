"""Revision experiment R1.6: scalability in the number of users and
robustness to unequal average SNRs, imperfect channel estimates, and
frequency-selective fading. All schemes pass through the common complex
channel simulator of rev_channel.py at V = 65,536 and d = 64.

  R-U  : U in {2, 4, 8, 16}; SSE retrained per U (Vu=16, P=4, same
         budget), full codebook retrained per U (same budget), OMA with
         M-PAM per real dimension at the same per-user rate, ToDMA with
         OMP sparsity U. Output rev_users.csv.
  R-C  : U = 4, one scenario per row at 10 and 20 dB:
           flat, perfect CSI (reference)
           near-far spreads of the average SNR, offsets
             {-D, -D/3, D/3, D} dB with D in {6, 12}
           channel-estimation NMSE in {0.01, 0.05, 0.1}
           frequency-selective fading with T in {4, 8} equal-power taps
             over 32 subcarriers, contiguous or interleaved periods, and
             correlation or weighted minimum-distance decoding.
         Output rev_robust.csv (average and worst-user SER).
Trained models are used as trained on flat fading with perfect CSI.
"""
from __future__ import annotations

import argparse
import math

import torch

from sse_lib import (DATA, DEVICE, D, VU, P_MAX, SSE, FullCodebook,
                     train_sse, train_full, set_seed, write_csv,
                     TRAIN_ITERS, TRAIN_BATCH, LR)
from exp_sota import make_dictionary, omp_support
import rev_channel as rc

V = VU ** P_MAX

SCEN = [("flat", {}),
        ("nearfar6", {"offsets": [-6, -2, 2, 6]}),
        ("nearfar12", {"offsets": [-12, -4, 4, 12]}),
        ("csi0.01", {"nmse": 0.01}),
        ("csi0.05", {"nmse": 0.05}),
        ("csi0.1", {"nmse": 0.1}),
        ("fs4", {"taps": 4}),
        ("fs4_int", {"taps": 4, "inter": True}),
        ("fs4_int_wmd", {"taps": 4, "inter": True, "rule": "wmd"}),
        ("fs8_int", {"taps": 8, "inter": True}),
        ("fs8_int_wmd", {"taps": 8, "inter": True, "rule": "wmd"})]



def train_full_accum(model, micro, iters=TRAIN_ITERS, batch=TRAIN_BATCH,
                     lr=LR, seed=1):
    """train_full of sse_lib with the batch split into `micro` equal
    micro-batches whose mean losses are averaged before the step, which
    is the same objective and gradient as one batch of `batch` samples
    (used only to fit large U into GPU memory)."""
    from sse_lib import TRAIN_SNR_DB
    set_seed(seed)
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ce = torch.nn.CrossEntropyLoss()
    mb = batch // micro
    for it in range(1, iters + 1):
        idx = torch.randint(model.Vc, (batch, model.users), device=DEVICE)
        snr = torch.empty(batch).uniform_(*TRAIN_SNR_DB)
        model.calibrate_power(8192)
        opt.zero_grad()
        for k in range(micro):
            sl = slice(k * mb, (k + 1) * mb)
            sc = model(idx[sl], snr[sl]) * model.logit_scale.exp()
            loss = ce(sc.reshape(-1, model.Vc), idx[sl].reshape(-1)) / micro
            loss.backward()
        opt.step()
    model.calibrate_power()


def load_or_train(kind, users, name):
    p = DATA / name
    m = SSE(P=P_MAX, users=users) if kind == "sse" else FullCodebook(V, users=users)
    if p.exists():
        m.load_state_dict(torch.load(p, map_location="cpu"))
        return m.to(DEVICE).eval()
    set_seed(1)
    if kind == "sse":
        train_sse(m)
    elif users <= 4:
        train_full(m)
    else:
        train_full_accum(m, micro=users // 2)
    torch.save(m.state_dict(), p)
    return m.to(DEVICE).eval()


def ids_to_digits(ids):
    d, x = [], ids.clone()
    for _ in range(P_MAX):
        d.append(x % VU); x = x // VU
    return torch.stack(d[::-1], dim=-1)


@torch.no_grad()
def run(scheme, model, users, snr, frames, chunk, seed, taps=1, nmse=0.0,
        offsets=None, inter=False, rule="corr"):
    """Returns per-user SER tensor (U,)."""
    torch.manual_seed(seed + int(10 * snr))
    perm = rc.interleaver(D, P_MAX) if inter else None
    err = torch.zeros(users)
    tot = 0
    if scheme == "todma":
        A = make_dictionary(D, V)
    for n0 in range(0, frames, chunk):
        n = min(chunk, frames - n0)
        ids = torch.randint(V, (n, users), device=DEVICE)
        if scheme == "sse":
            dg = ids_to_digits(ids)
            x = model.tx_frame(dg).reshape(n, D) / model.c
        elif scheme == "full":
            x = model.tx_frame(ids) / model.c
        elif scheme == "oma":
            n_sym, M, rep = rc.oma_layout(users, D)
            bps = int(math.log2(M))
            sym = torch.stack([(ids >> (bps * k)) & (M - 1)
                               for k in range(n_sym)], -1)          # (n,U,n_sym)
            x = rc.oma_frames(sym, users, D, M, rep)
        else:
            x = A.T[ids].sum(1) / math.sqrt(users)
        r, w = rc.channel(x, snr, users, taps=taps, nmse=nmse,
                          offsets_db=offsets, perm=perm)
        if scheme == "sse":
            wrong = (rc.sse_scores(model, r, w, rule).argmax(-1) != dg).any(-1)
        elif scheme == "full":
            # the full codebook is decoded at its best, by (weighted)
            # minimum distance, since its masked norms are unequal
            wrong = rc.full_scores(model, r, w, "wmd").argmax(-1) != ids
        elif scheme == "oma":
            wrong = (rc.oma_detect(r, w, users, D, M, rep) != sym).any(-1)
        else:
            wrong = torch.zeros(n, users, dtype=torch.bool, device=DEVICE)
            for u in range(users):
                sup = omp_support(r[:, u], A, k=users)
                wrong[:, u] = ~(sup == ids[:, u:u + 1]).any(1)
        err += wrong.float().sum(0).cpu()
        tot += n
    return err / tot


def users_sweep(frames, fr_full, fr_todma):
    rows = []
    for users in (2, 4, 8, 16):
        sse = load_or_train("sse", users,
                            "sse_P4.pt" if users == 4 else f"rev_sse_U{users}.pt")
        full = load_or_train("full", users,
                             f"full_V{V}.pt" if users == 4 else f"rev_full_U{users}.pt")
        for snr in (10.0, 20.0):
            s = float(run("sse", sse, users, snr, frames, 100_000, 777).mean())
            f = float(run("full", full, users, snr, fr_full,
                          max(16, 512 // users), 777).mean())
            o = float(run("oma", None, users, snr, frames, 100_000, 777).mean())
            t = float(run("todma", None, users, snr, fr_todma, 512, 777).mean())
            rows.append((users, snr, s, f, t, o))
            print("[R-U]", rows[-1], flush=True)
    write_csv(DATA / "rev_users.csv",
              ["U", "snr_db", "ser_SSE", "ser_Full", "ser_ToDMA", "ser_OMA"], rows)


def robust_full(fr_full):
    """Full codebook only, minimum-distance decoding, all scenarios."""
    users = 4
    full = load_or_train("full", users, f"full_V{V}.pt")
    rows = []
    for name, kw in SCEN:
        kf = {k: v for k, v in kw.items() if k not in ("inter", "rule")}
        for snr in (10.0, 20.0):
            r = run("full", full, users, snr, fr_full, 128, 777, **kf)
            rows.append((name, snr, float(r.mean()), float(r.max())))
            print("[R-F]", rows[-1], flush=True)
    write_csv(DATA / "rev_robust_fullmd.csv",
              ["scenario", "snr_db", "ser_Full", "worst_Full"], rows)


def robust(frames, fr_full, fr_todma):
    users = 4
    sse = load_or_train("sse", users, "sse_P4.pt")
    full = load_or_train("full", users, f"full_V{V}.pt")
    scen = SCEN
    rows = []
    for name, kw in scen:
        for snr in (10.0, 20.0):
            res = {}
            res["SSE"] = run("sse", sse, users, snr, frames, 100_000, 777, **kw)
            kf = {k: v for k, v in kw.items() if k != "inter"}
            res["Full"] = run("full", full, users, snr, fr_full, 128, 777, **kf)
            ko = {k: v for k, v in kw.items() if k not in ("inter", "rule")}
            res["OMA"] = run("oma", None, users, snr, frames, 100_000, 777, **ko)
            res["ToDMA"] = run("todma", None, users, snr, fr_todma, 512, 777, **ko)
            row = [name, snr]
            for k in ("SSE", "Full", "ToDMA", "OMA"):
                row += [float(res[k].mean()), float(res[k].max())]
            rows.append(row)
            print("[R-C]", [f"{v:.4f}" if isinstance(v, float) else v
                            for v in row], flush=True)
    hdr = ["scenario", "snr_db"]
    for k in ("SSE", "Full", "ToDMA", "OMA"):
        hdr += [f"ser_{k}", f"worst_{k}"]
    write_csv(DATA / "rev_robust.csv", hdr, rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="robust,users")
    ap.add_argument("--frames", type=int, default=1_000_000)
    ap.add_argument("--fr_full", type=int, default=200_000)
    ap.add_argument("--fr_todma", type=int, default=100_000)
    a = ap.parse_args()
    if "fullmd" in a.part:
        robust_full(a.fr_full)
    if a.part == "robust" or "robust," in a.part:
        robust(a.frames, a.fr_full, a.fr_todma)
    if "users" in a.part:
        users_sweep(a.frames, a.fr_full, a.fr_todma)


if __name__ == "__main__":
    main()
