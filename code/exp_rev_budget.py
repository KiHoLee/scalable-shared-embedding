"""Revision experiment R1.2 / R2.4: training-budget matched comparisons at
V = 65,536.

Trains, under the same optimizer, batch, SNR range, and loss family:
  full      : directly scaled codebook, exact V-way softmax (the Full
              codebook scheme), long budget
  sampled   : directly scaled codebook trained with sampled softmax,
              K uniformly drawn shared negatives per step plus the true
              codewords (uniform proposal, so no logQ correction)
  hier      : directly scaled codebook trained with a two-level
              hierarchical softmax (256 classes x 256 members), class
              vectors learned at the receiver, greedy class-then-member
              decoding
  sse       : Proposed SSE (Vu=16, P=4), long budget

Every run logs, at geometric checkpoints, the iteration, the wall-clock
time, the cumulative score MACs of the training forward pass, and the
SER at 10 dB over 20,000 frames. Final models get a 2e6-frame sweep.

Outputs (data/, prefix rev_):
  rev_budget_curve.csv   : iter,time_s,macs,ser10,scheme
  rev_budget_final.csv   : snr_db, ser_<scheme> for the final models
  rev_<scheme>_long.pt   : final models
"""
from __future__ import annotations

import argparse
import math
import time

import torch

from sse_lib import (DATA, DEVICE, D, U, VU, P_MAX, SSE, FullCodebook,
                     TRAIN_BATCH, TRAIN_SNR_DB, LR, rayleigh_gain,
                     snr_to_sigma2, set_seed, eval_ser_sse, eval_ser_full,
                     write_csv)

V = VU ** P_MAX
N_CLS = 256                     # hierarchical softmax: 256 x 256
K_NEG = 1024                    # sampled softmax negatives per step
SNR_GRID = list(range(0, 21, 2))


def log_points(iters: int):
    pts = set()
    k = 1
    while k <= iters:
        for m in (1, 2, 5):
            if m * k <= iters:
                pts.add(m * k)
        k *= 10
    step = max(iters // 40, 1)
    pts.update(range(step, iters + 1, step))
    pts.add(iters)
    return pts


# ----------------------------------------------------------------------
# hierarchical softmax variant of the directly scaled codebook
# ----------------------------------------------------------------------
class HierCodebook(FullCodebook):
    """Transmit codebook E (V x d) as in FullCodebook. The receiver holds
    N_CLS learned class vectors C; index s belongs to class s // M with
    M = V / N_CLS members. Level 1 scores the classes, level 2 scores the
    M members of a class by correlation with the masked codewords."""

    def __init__(self, V: int, d: int = D, users: int = U, n_cls: int = N_CLS):
        super().__init__(V, d, users)
        self.n_cls, self.M = n_cls, V // n_cls
        self.C = torch.nn.Parameter(torch.randn(n_cls, d) / math.sqrt(d))

    def equalized(self, idx, snr_db, h=None):
        N = idx.shape[0]
        y = self.tx_frame(idx) / self.c
        if h is None:
            h = rayleigh_gain((N, self.users), device=y.device)
        sigma = snr_to_sigma2(snr_db).to(y.device).sqrt()
        n = torch.randn(N, self.users, self.d, device=y.device)
        y_rx = h[:, :, None] * y[:, None] + sigma[:, None, None] * n
        return y_rx / h[:, :, None].clamp_min(1e-6)       # (N,U,d)

    def class_scores(self, r):
        Cn = self.C / self.C.norm(dim=1, keepdim=True).clamp_min(1e-8)
        m = self.masks()
        return torch.einsum("nud,ukd->nuk", r, Cn[None] * m[:, None, :])

    def member_scores(self, r, cls):
        """scores of the M members of class cls (N,U) -> (N,U,M)."""
        En = self.codebook().view(self.n_cls, self.M, self.d)
        m = self.masks()
        cand = En[cls] * m[None, :, None, :]              # (N,U,M,d)
        return torch.einsum("nud,numd->num", r, cand)

    @torch.no_grad()
    def decode(self, r):
        cls = self.class_scores(r).argmax(-1)
        mem = self.member_scores(r, cls)
        if getattr(self, "md", False):
            En = self.codebook().view(self.n_cls, self.M, self.d)
            cand = En[cls] * self.masks()[None, :, None, :]
            a = 1.0 / float(self.c)
            mem = 2 * a * mem - a * a * cand.pow(2).sum(-1)
        return cls * self.M + mem.argmax(-1)


@torch.no_grad()
def eval_ser_hier(model: HierCodebook, snr_list, frames=2_000_000,
                  chunk=1_000, seed=777):
    model.eval().to(DEVICE)
    out = []
    for s in snr_list:
        g = torch.Generator(device="cpu").manual_seed(seed + int(10 * s))
        err = tot = 0
        for n0 in range(0, frames, chunk):
            n = min(chunk, frames - n0)
            idx = torch.randint(model.Vc, (n, model.users), generator=g).to(DEVICE)
            r = model.equalized(idx, torch.full((n,), float(s)))
            err += int((model.decode(r) != idx).sum()); tot += n * model.users
        out.append(err / tot)
    model.train()
    return out


# ----------------------------------------------------------------------
# per-step score MACs of the training forward pass (per sample x batch)
# ----------------------------------------------------------------------
def step_macs(scheme: str) -> int:
    b, u = TRAIN_BATCH, U
    if scheme == "sse":
        return b * u * P_MAX * VU * (D // P_MAX)          # U P Vu L = U Vu d
    if scheme == "full":
        return b * u * V * D
    if scheme == "sampled":
        return b * u * (K_NEG + b * u) * D                # shared negatives
    if scheme == "hier":
        return b * u * (N_CLS + V // N_CLS) * D
    raise ValueError(scheme)


def evaluator(scheme, model):
    if scheme == "sse":
        return lambda s, f: eval_ser_sse(model, s, frames=f)
    if scheme == "hier":
        return lambda s, f: eval_ser_hier(model, s, frames=f)
    return lambda s, f: eval_ser_full(model, s, frames=f)


def train(scheme: str, iters: int, seed: int = 1, cosine: bool = True,
          log_frames: int = 20000):
    set_seed(seed)
    if scheme == "sse":
        model = SSE(P=P_MAX)
    elif scheme == "hier":
        model = HierCodebook(V)
    else:
        model = FullCodebook(V)
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
             if cosine else None)
    ce = torch.nn.CrossEntropyLoss()
    ev = evaluator(scheme, model)
    pts = log_points(iters)
    curve = []
    t_train = 0.0
    macs = 0
    for it in range(1, iters + 1):
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        snr = torch.empty(TRAIN_BATCH).uniform_(*TRAIN_SNR_DB)
        model.calibrate_power(8192)
        scale = model.logit_scale.exp()
        if scheme == "sse":
            dg = torch.randint(VU, (TRAIN_BATCH, U, P_MAX), device=DEVICE)
            sc = model(dg, snr) * scale
            loss = ce(sc.reshape(-1, VU), dg.reshape(-1))
        elif scheme == "full":
            idx = torch.randint(V, (TRAIN_BATCH, U), device=DEVICE)
            sc = model(idx, snr) * scale
            loss = ce(sc.reshape(-1, V), idx.reshape(-1))
        elif scheme == "sampled":
            idx = torch.randint(V, (TRAIN_BATCH, U), device=DEVICE)
            neg = torch.randint(V, (K_NEG,), device=DEVICE)
            cand_idx = torch.cat([idx.reshape(-1), neg])           # (bU+K,)
            y = model.tx_frame(idx) / model.c
            h = rayleigh_gain((TRAIN_BATCH, U))
            sigma = snr_to_sigma2(snr).to(DEVICE).sqrt()
            n = torch.randn(TRAIN_BATCH, U, D, device=DEVICE)
            r = (h[:, :, None] * y[:, None] + sigma[:, None, None] * n) \
                / h[:, :, None].clamp_min(1e-6)
            En = model.codebook()[cand_idx]                        # (C,d)
            m = model.masks()
            sc = torch.einsum("nud,ucd->nuc", r, En[None] * m[:, None, :]) * scale
            # the true codeword of (n,u) sits at column n*U+u; duplicates
            # of the true index among the candidates are masked out
            tgt = torch.arange(TRAIN_BATCH * U, device=DEVICE).view(TRAIN_BATCH, U)
            dup = (cand_idx[None, None, :] == idx[:, :, None])
            dup.scatter_(2, tgt[:, :, None], False)
            sc = sc.masked_fill(dup, -1e9)
            loss = ce(sc.reshape(-1, sc.shape[-1]), tgt.reshape(-1))
        else:  # hier
            idx = torch.randint(V, (TRAIN_BATCH, U), device=DEVICE)
            r = model.equalized(idx, snr)
            cls, mem = idx // model.M, idx % model.M
            s1 = model.class_scores(r) * scale
            s2 = model.member_scores(r, cls) * scale
            loss = ce(s1.reshape(-1, model.n_cls), cls.reshape(-1)) \
                + ce(s2.reshape(-1, model.M), mem.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if sched is not None:
            sched.step()
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        t_train += time.time() - t0
        macs += step_macs(scheme)
        if it in pts:
            model.calibrate_power()
            ser = ev([10.0], log_frames)[0]
            if scheme != "sse":
                model.md = True
                ser_md = ev([10.0], log_frames)[0]
                model.md = False
            else:
                ser_md = ser
            model.train()
            curve.append((it, t_train, macs, ser, ser_md, scheme))
            print(f"[{scheme}] it={it} t={t_train:.1f}s macs={macs:.3e} "
                  f"ser10={ser:.4f} md={ser_md:.4f}", flush=True)
    model.calibrate_power()
    if iters == 200_000:
        tag = "" if seed == 1 else f"_s{seed}"
        torch.save(model.state_dict(), DATA / f"rev_{scheme}_long_md{tag}.pt")
    return model, curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schemes", default="sse,hier,sampled,full")
    ap.add_argument("--iters", type=int, default=200_000)
    ap.add_argument("--frames", type=int, default=2_000_000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--log-frames", type=int, default=20000,
                    help="frames per logged SER point (seed 1 used 20000)")
    a = ap.parse_args()
    tag = "" if a.seed == 1 else f"_s{a.seed}"
    for sch in a.schemes.split(","):
        model, curve = train(sch, a.iters, seed=a.seed,
                             log_frames=a.log_frames)
        write_csv(DATA / f"rev_budget_curve_{sch}{tag}.csv",
                  ["iter", "time_s", "macs", "ser10", "ser10_md", "scheme"],
                  curve)
        ser = evaluator(sch, model)(SNR_GRID, a.frames)
        if sch != "sse":
            model.md = True
            ser_md = evaluator(sch, model)(SNR_GRID, a.frames)
            model.md = False
        else:
            ser_md = ser
        write_csv(DATA / f"rev_budget_final_{sch}{tag}.csv",
                  ["snr_db", f"ser_{sch}", f"ser_{sch}_md"],
                  list(zip(SNR_GRID, ser, ser_md)))
        print(f"[final] {sch}", ser, ser_md, flush=True)


if __name__ == "__main__":
    main()
