"""Main experiments for the SSE letter. Writes CSVs to ../data/.

Stage 1 (controlled, uniform indices):
  snr_sweep.csv     : SER vs SNR at V=65536 for Proposed SSE, Full codebook
                      (same V, same training budget), OMA digital, and the
                      interference-free single-user reference; plus the
                      union-bound analysis curve.
  scaling.csv       : SER at 10 dB and parameter counts vs V (16^P).
  convergence.csv   : eval SER at 10 dB vs training iteration and time.
  complexity.json   : parameter counts, per-frame MACs, training seconds.

Models are saved to ../data/*.pt so evaluation is repeatable without
retraining. Seeds fixed (SEED=1 for training, 777 for evaluation).
"""
from __future__ import annotations

import json
import time

import torch

from sse_lib import (DATA, DEVICE, D, U, VU, P_MAX, SEED, SSE, FullCodebook,
                     train_sse, train_full, eval_ser_sse, eval_ser_full,
                     oma_ser, oma_ser_mc, sse_union_bound, set_seed, write_csv)

SNR_GRID = list(range(0, 21, 2))
EVAL_FRAMES = 2_000_000
LOG_EVERY = 100


def get_or_train_sse(P: int, tag: str, log_every: int = 0, vu: int = VU):
    suf = f"P{P}" if vu == VU else f"Vu{vu}_P{P}"
    path = DATA / f"sse_{suf}.pt"
    curve_path = DATA / f"curve_sse_{suf}.json"
    model = SSE(P=P, vu=vu)
    if path.exists():
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.to(DEVICE)
        curve = json.loads(curve_path.read_text()) if curve_path.exists() else {}
        return model, curve
    t0 = time.time()
    curve = train_sse(model, log_every=log_every)
    dt = time.time() - t0
    torch.save(model.state_dict(), path)
    curve_path.write_text(json.dumps({"curve": curve, "train_s": dt}))
    print(f"[train] {tag}: {dt:.1f}s, params={model.n_params()}")
    return model, json.loads(curve_path.read_text())


def get_or_train_full(V: int, log_every: int = 0):
    path = DATA / f"full_V{V}.pt"
    curve_path = DATA / f"curve_full_V{V}.json"
    model = FullCodebook(V)
    if path.exists():
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.to(DEVICE)
        return model, (json.loads(curve_path.read_text())
                       if curve_path.exists() else {})
    t0 = time.time()
    curve = train_full(model, log_every=log_every)
    dt = time.time() - t0
    torch.save(model.state_dict(), path)
    curve_path.write_text(json.dumps({"curve": curve, "train_s": dt}))
    print(f"[train] full V={V}: {dt:.1f}s, params={model.n_params()}")
    return model, json.loads(curve_path.read_text())


def main():
    set_seed(SEED)
    info = {"device": str(DEVICE)}

    # ------------------------------------------------------------------
    # train main models (with convergence logging)
    # ------------------------------------------------------------------
    sse_main, curve_sse = get_or_train_sse(P_MAX, "sse", log_every=LOG_EVERY)
    V = sse_main.V
    full_main, curve_full = get_or_train_full(V, log_every=LOG_EVERY)

    # interference-free reference: single user occupies the same per-user
    # resources (identical codebook structure, U=1, same rate per user)
    path = DATA / "sse_single.pt"
    single = SSE(P=P_MAX, users=1)
    if path.exists():
        single.load_state_dict(torch.load(path, map_location="cpu"))
        single.to(DEVICE)
    else:
        train_sse(single)
        torch.save(single.state_dict(), path)

    # ------------------------------------------------------------------
    # snr sweep
    # ------------------------------------------------------------------
    ser_sse = eval_ser_sse(sse_main, SNR_GRID, frames=EVAL_FRAMES)
    ser_full = eval_ser_full(full_main, SNR_GRID, frames=EVAL_FRAMES // 4)
    ser_single = eval_ser_sse(single, SNR_GRID, frames=EVAL_FRAMES)
    ser_oma = oma_ser(SNR_GRID)
    ser_oma_mc = oma_ser_mc(SNR_GRID, frames=1_000_000)
    ser_ub = sse_union_bound(sse_main, SNR_GRID)
    write_csv(DATA / "snr_sweep.csv",
              ["snr_db", "ser_SSE", "ser_Full", "ser_OMA", "ser_OMAmc",
               "ser_Single", "ser_UB"],
              [(s, a, b, c, d_, e, f) for s, a, b, c, d_, e, f in
               zip(SNR_GRID, ser_sse, ser_full, ser_oma, ser_oma_mc,
                   ser_single, ser_ub)])

    # ------------------------------------------------------------------
    # scaling in V: proposed (P=4 fixed, Vu in {2,4,8,16}) vs full
    # codebook at the same V = Vu^4 and identical training budget
    # ------------------------------------------------------------------
    rows = []
    for vu in (2, 4, 8, 16):
        Vp = vu ** P_MAX
        m_sse, _ = get_or_train_sse(P_MAX, f"sse Vu={vu}", vu=vu)
        m_full, _ = get_or_train_full(Vp)
        s10_sse = eval_ser_sse(m_sse, [10.0], frames=EVAL_FRAMES)[0]
        s16_sse = eval_ser_sse(m_sse, [16.0], frames=EVAL_FRAMES)[0]
        s10_full = eval_ser_full(m_full, [10.0], frames=EVAL_FRAMES // 4)[0]
        s16_full = eval_ser_full(m_full, [16.0], frames=EVAL_FRAMES // 4)[0]
        rows.append((Vp, s10_sse, s16_sse, m_sse.n_params(),
                     s10_full, s16_full, m_full.n_params()))
        print("[scaling]", rows[-1])
    write_csv(DATA / "scaling.csv",
              ["V", "ser10_SSE", "ser16_SSE", "par_SSE",
               "ser10_Full", "ser16_Full", "par_Full"], rows)

    # ------------------------------------------------------------------
    # convergence curves
    # ------------------------------------------------------------------
    rows = []
    for it, tsec, loss, ser in curve_sse.get("curve", []):
        rows.append((int(it), tsec, loss, ser, "SSE"))
    for it, tsec, loss, ser in curve_full.get("curve", []):
        rows.append((int(it), tsec, loss, ser, "Full"))
    write_csv(DATA / "convergence.csv",
              ["iter", "time_s", "loss", "ser10", "scheme"], rows)

    # ------------------------------------------------------------------
    # complexity accounting (explicit operation counting, per frame)
    # ------------------------------------------------------------------
    L = D // P_MAX
    info.update({
        "V": V,
        "params_sse": sse_main.n_params(),
        "params_full": full_main.n_params(),
        "param_ratio": full_main.n_params() / sse_main.n_params(),
        # receiver MACs per user per frame: score correlations
        "rx_macs_sse": P_MAX * VU * L,          # P digit correlations
        "rx_macs_full": V * D,                   # V correlations of length d
        "train_s_sse": curve_sse.get("train_s"),
        "train_s_full": curve_full.get("train_s"),
    })
    (DATA / "complexity.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
