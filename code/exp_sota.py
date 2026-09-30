"""SOTA comparison schemes under the identical resource budget.

Scheme 1 -- ToDMA (token-domain multiple access): every vocabulary
entry receives a fixed random Gaussian signature of length d, the
transmitter superposes the U active signatures at unit frame energy,
and each receiver equalizes and recovers the U active tokens by
batched orthogonal matching pursuit (OMP) over the 65,536-column
dictionary. A genie-aided association counts the user correct when
its true token lies in the detected support, which upper-bounds any
practical association. No training. Also evaluated at a doubled
signature length (two frames per token) as a diagnostic.

Scheme 2 -- Full codebook with user-wise attention receiver: the
conventional shared-embedding architecture scaled directly to
V=65,536, with the trained V x d codebook and full-d masks of the
FullCodebook scheme, plus the user-wise attention receiver of the
prior architecture: per-user mask-aligned candidates, learned key and
value projections, learned user queries, and logits tied to the
codebook. Same optimizer, batch, iterations, and loss as all trained
schemes.

Outputs: data/sota_snr.csv, data/sota_real.csv, data/full_att.pt,
data/curve_full_att.json.
"""
from __future__ import annotations

import json
import math
import time

import torch

from sse_lib import (DATA, DEVICE, D, U, VU, P_MAX, SEED, FullCodebook,
                     rayleigh_gain, snr_to_sigma2, set_seed, write_csv)

SNR_GRID = list(range(0, 21, 2))
V_MAIN = VU ** P_MAX
TODMA_FRAMES = 200_000
ATT_EVAL_FRAMES = 500_000


# ----------------------------------------------------------------------
# ToDMA: random signature dictionary + batched OMP, genie association
# ----------------------------------------------------------------------
def make_dictionary(dim: int, V: int = V_MAIN, seed: int = 12345):
    g = torch.Generator(device="cpu").manual_seed(seed)
    A = torch.randn(dim, V, generator=g)
    A = A / A.norm(dim=0, keepdim=True)
    return A.to(DEVICE)


@torch.no_grad()
def omp_support(r: torch.Tensor, A: torch.Tensor, k: int = U):
    """Batched OMP. r: (n, dim) residual targets, A: (dim, V).
    Returns support indices (n, k)."""
    n, dim = r.shape
    Vc = A.shape[1]
    support = torch.zeros(n, k, dtype=torch.long, device=r.device)
    R = r.clone()
    eye = torch.eye(k, device=r.device)
    for t in range(k):
        c = (R @ A).abs_()                       # (n, V)
        if t > 0:
            c.scatter_(1, support[:, :t], -1.0)
        support[:, t] = c.argmax(dim=1)
        As = A.T[support[:, :t + 1]]             # (n, t+1, dim)
        # normal equations written with elementwise products to avoid
        # degenerate bmm shapes on this backend
        G = torch.einsum("nkd,njd->nkj", As, As) + 1e-6 * eye[:t + 1, :t + 1]
        b = (As * r[:, None, :]).sum(dim=2, keepdim=True)   # (n, t+1, 1)
        beta = torch.linalg.solve(G, b)
        R = r - (As * beta).sum(dim=1)
    return support


@torch.no_grad()
def todma_ser(snr_db_list, frames: int = TODMA_FRAMES, chunk: int = 2048,
              n_frames_per_token: int = 1, ids: torch.Tensor | None = None,
              seed: int = 777, V: int = V_MAIN):
    """SER (miss rate of the true token in the OMP support) under the
    system conventions: unit frame energy, per-user Rayleigh gain known
    at the receiver, sigma^2 = 1/(dim*snr) per dimension with
    dim = D * n_frames_per_token channel dims per token slot."""
    dim = D * n_frames_per_token
    A = make_dictionary(dim, V)
    out = []
    for s in snr_db_list:
        torch.manual_seed(seed + int(10 * s) + 100000 * n_frames_per_token)
        sigma = math.sqrt(float(snr_to_sigma2(torch.tensor(float(s)))))
        sigma *= math.sqrt(D / dim)              # keep per-dim SNR convention
        err = tot = 0
        n_done = 0
        while n_done < frames:
            n = min(chunk, frames - n_done)
            if ids is None:
                tok = torch.randint(V, (n, U), device=DEVICE)
            else:
                sel = torch.randint(ids.shape[0], (n,), device=ids.device)
                tok = ids[sel].to(DEVICE)
            x = A.T[tok].sum(dim=1) / math.sqrt(U)          # (n, dim)
            h = rayleigh_gain((n, U))
            for u in range(U):
                y = h[:, u:u + 1] * x + sigma * torch.randn(n, dim,
                                                            device=DEVICE)
                r = y / h[:, u:u + 1].clamp_min(1e-6)
                sup = omp_support(r, A)
                hit = (sup == tok[:, u:u + 1]).any(dim=1)
                err += int((~hit).sum()); tot += n
            n_done += n
        out.append(err / tot)
        print(f"[todma x{n_frames_per_token}] {s} dB: {out[-1]:.5f}")
    return out


# ----------------------------------------------------------------------
# Full codebook + user-wise attention receiver (scaled prior scheme)
# ----------------------------------------------------------------------
class FullAttention(FullCodebook):
    """FullCodebook transmitter with the user-wise attention receiver:
    candidates r_i = r .* m_i, learned W_k/W_v projections, learned
    user queries, softmax combination, logits tied to the codebook."""

    def __init__(self, V: int, d: int = D, users: int = U):
        super().__init__(V, d, users)
        # identity initialization keeps the attention informative at the
        # start of training (random projections destroy the codeword
        # geometry and the V-way loss never recovers within the budget)
        self.Wk = torch.nn.Parameter(torch.eye(d) + 0.02 * torch.randn(d, d))
        self.Wv = torch.nn.Parameter(torch.eye(d) + 0.02 * torch.randn(d, d))
        self.q = torch.nn.Parameter(torch.randn(users, d) / math.sqrt(d))

    def forward(self, idx: torch.Tensor, snr_db: torch.Tensor,
                h: torch.Tensor | None = None):
        N = idx.shape[0]
        y = self.tx_frame(idx) / self.c
        if h is None:
            h = rayleigh_gain((N, self.users), device=y.device)
        sigma = snr_to_sigma2(snr_db).to(y.device).sqrt()
        n = torch.randn(N, self.users, self.d, device=y.device)
        y_rx = h[:, :, None] * y[:, None] + sigma[:, None, None] * n
        r = y_rx / h[:, :, None].clamp_min(1e-6)          # (N,U,d)
        m = self.masks()
        cand = r[:, :, None, :] * m[None, None, :, :]     # (N,U,Ucand,d)
        k = cand @ self.Wk                                # keys
        v = cand @ self.Wv                                # values
        # broadcast forms (not einsum) keep the autograd graph on regular
        # kernels; einsum backward emits degenerate outer-product bmms
        # that this backend cannot compile
        att = (self.q[None, :, None, :] * k).sum(-1) / math.sqrt(self.d)
        att = att.softmax(dim=2)                          # (N,U,Ucand)
        z = (att[..., None] * v).sum(dim=2)               # (N,U,d)
        En = self.codebook()
        cand_cb = En[None, :, :] * m[:, None, :]          # (U,V,d)
        return torch.einsum("nud,uvd->nuv", z, cand_cb)

    def n_params(self) -> int:
        return (self.E.numel() + self.W.numel() + self.Wk.numel()
                + self.Wv.numel() + self.q.numel())


def train_full_att(iters: int = 4000, batch: int = 256, lr: float = 3e-3,
                   log_every: int = 100, seed: int = SEED):
    from sse_lib import TRAIN_SNR_DB
    path = DATA / "full_att.pt"
    curve_path = DATA / "curve_full_att.json"
    model = FullAttention(V_MAIN)
    if path.exists():
        model.load_state_dict(torch.load(path, map_location="cpu"))
        return model.to(DEVICE), json.loads(curve_path.read_text())
    set_seed(seed)
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ce = torch.nn.CrossEntropyLoss()
    curve = []
    t0 = time.time()
    for it in range(1, iters + 1):
        idx = torch.randint(V_MAIN, (batch, U), device=DEVICE)
        snr = torch.empty(batch).uniform_(*TRAIN_SNR_DB)
        model.calibrate_power(8192)
        scores = model(idx, snr) * model.logit_scale.exp()
        loss = ce(scores.reshape(-1, V_MAIN), idx.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if it % log_every == 0 or it == 1:
            ser = eval_ser_att(model, [10.0], frames=20000)[0]
            curve.append((it, time.time() - t0, float(loss.detach()), ser))
    model.calibrate_power()
    torch.save(model.state_dict(), path)
    curve_path.write_text(json.dumps({"curve": curve,
                                      "train_s": time.time() - t0}))
    print(f"[train] full-att: {time.time()-t0:.1f}s, params={model.n_params()}")
    return model, json.loads(curve_path.read_text())


@torch.no_grad()
def eval_ser_att(model: FullAttention, snr_list, frames: int = ATT_EVAL_FRAMES,
                 seed: int = 777, ids: torch.Tensor | None = None):
    model.eval().to(DEVICE)
    chunk = max(256, (1 << 22) // model.Vc)
    out = []
    for snr_db in snr_list:
        g = torch.Generator(device="cpu").manual_seed(seed + int(10 * snr_db))
        err = tot = 0
        for n0 in range(0, frames, chunk):
            n = min(chunk, frames - n0)
            if ids is None:
                idx = torch.randint(model.Vc, (n, model.users),
                                    generator=g).to(DEVICE)
            else:
                sel = torch.randint(ids.shape[0], (n,), generator=g)
                idx = ids[sel].to(DEVICE)
            snr = torch.full((n,), float(snr_db))
            scores = model(idx, snr)
            err += int((scores.argmax(-1) != idx).sum()); tot += n * model.users
        out.append(err / tot)
    return out


def main():
    set_seed(SEED)

    model, curve = train_full_att()
    print("full-att params:", model.n_params(),
          "train_s:", curve.get("train_s"))

    ser_att = eval_ser_att(model, SNR_GRID)
    print("[att]", list(zip(SNR_GRID, ser_att)))
    ser_todma = todma_ser(SNR_GRID)
    ser_todma2 = todma_ser(SNR_GRID, n_frames_per_token=2,
                           frames=TODMA_FRAMES // 2)
    write_csv(DATA / "sota_snr.csv",
              ["snr_db", "ser_FullATT", "ser_ToDMA", "ser_ToDMA2"],
              list(zip(SNR_GRID, ser_att, ser_todma, ser_todma2)))

    # real BERT token streams (reuse exp_real stream construction)
    from exp_real import load_streams
    streams, bounds, vocab = load_streams()
    ids_all = torch.tensor(list(zip(*streams)), dtype=torch.long)
    ter_att = eval_ser_att(model, SNR_GRID, frames=200_000, ids=ids_all)
    ter_todma = todma_ser(SNR_GRID, frames=100_000, ids=ids_all)
    write_csv(DATA / "sota_real.csv",
              ["snr_db", "ter_FullATT", "ter_ToDMA"],
              list(zip(SNR_GRID, ter_att, ter_todma)))

    # dictionary coherence for the analysis sentence in the paper
    A = make_dictionary(D)
    gram = (A.T[:4096] @ A).abs()
    gram.fill_diagonal_(0)
    print("coherence (4096-col sample): max",
          float(gram.max()), "typ", float(gram.mean()))


if __name__ == "__main__":
    main()
