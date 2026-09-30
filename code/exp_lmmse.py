"""Exploratory: LMMSE-whitened decoder on the cached SSE model.

The equalized frame for user u is  r = x_u + sum_{u'!=u} x_{u'} + (sigma/h_u) n,
so the interference term is h-independent with a closed-form per-period mean
and covariance over the uniform digit distribution:

  mean_I  = sum_{u'!=u} (bbar . mu_u') / (c sqrt(P))
  C_I,u   = sum_{u'!=u} [(Bn^T Bn / Vu - bbar bbar^T) . (mu_u' mu_u'^T)] / (c^2 P)

The decoder whitens with C_u(h) = C_I,u + (sigma^2/h^2) I via a per-user
eigendecomposition (C^{-1} = Q diag(1/(lam+s)) Q^T, batched over frames) and
uses the Gaussian metric  t_v^T C^{-1} (r - mean_I) - 0.5 t_v^T C^{-1} t_v.
Matched filter is re-evaluated on the identical realizations for a paired
comparison.  No retraining; reads data/sse_P4.pt only.
"""
import math
import torch

from sse_lib import DATA, DEVICE, SSE, rayleigh_gain, snr_to_sigma2

SNRS = [0.0, 5.0, 10.0, 15.0, 20.0]
FRAMES = 2_000_000
CHUNK = 100_000
SEED = 777


@torch.no_grad()
def eval_paired(model: SSE, snr_list, frames=FRAMES, chunk=CHUNK, seed=SEED):
    model.eval().to(DEVICE)
    P, L, Vu, U = model.P, model.L, model.vu, model.users
    Bn = model.unit_codebook()                       # (Vu,L)
    m = model.masks()                                # (U,L)
    c = float(model.c)
    scale = 1.0 / (c * math.sqrt(P))

    t = Bn[None, :, :] * m[:, None, :] * scale       # templates (U,Vu,L)
    bbar = Bn.mean(dim=0)                            # (L,)
    cov_b = Bn.t() @ Bn / Vu - bbar[:, None] * bbar[None, :]   # (L,L)

    mean_I = torch.zeros(U, L, device=DEVICE)
    Q = torch.zeros(U, L, L, device=DEVICE)
    lam = torch.zeros(U, L, device=DEVICE)
    for u in range(U):
        mean_I[u] = sum(bbar * m[v] * scale for v in range(U) if v != u)
        C = sum((cov_b * (m[v][:, None] * m[v][None, :])) * scale ** 2
                for v in range(U) if v != u)
        ev, evec = torch.linalg.eigh(C)
        lam[u], Q[u] = ev.clamp_min(0.0), evec

    # rotate templates and interference mean into each user's eigenbasis
    tq = (t[:, :, None, :] * Q.permute(0, 2, 1)[:, None, :, :]).sum(-1)  # (U,Vu,L)

    out = []
    for snr_db in snr_list:
        g = torch.Generator(device="cpu").manual_seed(seed + int(10 * snr_db))
        err_mf = err_lm = tot = 0
        for n0 in range(0, frames, chunk):
            n = min(chunk, frames - n0)
            digits = torch.randint(Vu, (n, U, P), generator=g).to(DEVICE)
            y = model.tx_frame(digits) / model.c                 # (n,P,L)
            h = rayleigh_gain((n, U))
            sigma = snr_to_sigma2(torch.full((n,), float(snr_db))).to(DEVICE).sqrt()
            noise = torch.randn(n, U, P, L, device=DEVICE)
            r = y[:, None] + (sigma[:, None] / h.clamp_min(1e-6))[:, :, None, None] * noise

            # matched filter (paper receiver), same realizations
            sc_mf = (r[:, :, :, None, :] * (t * (c * math.sqrt(P)))[None, :, None]).sum(-1)
            err_mf += int((sc_mf.argmax(-1) != digits).any(dim=2).sum())

            # LMMSE-whitened Gaussian metric
            s = (sigma[:, None] / h.clamp_min(1e-6)) ** 2        # (n,U)
            w = 1.0 / (lam[None] + s[:, :, None])                # (n,U,L)
            rq = ((r - mean_I[None, :, None, :])[:, :, :, None, :]
                  * Q.permute(0, 2, 1)[None, :, None, :, :]).sum(-1)   # (n,U,P,L)
            cross = (rq[:, :, :, None, :] * tq[None, :, None]
                     * w[:, :, None, None, :]).sum(-1)           # (n,U,P,Vu)
            self_e = ((tq ** 2)[None, :, None] * w[:, :, None, None, :]).sum(-1)
            sc_lm = cross - 0.5 * self_e
            err_lm += int((sc_lm.argmax(-1) != digits).any(dim=2).sum())
            tot += n * U
        out.append((snr_db, err_mf / tot, err_lm / tot))
        print(f"SNR {snr_db:4.0f} dB:  MF {out[-1][1]:.5f}   "
              f"LMMSE {out[-1][2]:.5f}   "
              f"({(1 - out[-1][2] / max(out[-1][1], 1e-12)) * 100:+.1f}%)",
              flush=True)
    return out


if __name__ == "__main__":
    model = SSE()
    model.load_state_dict(torch.load(DATA / "sse_P4.pt", map_location=DEVICE))
    print("interference covariance rank structure per user:")
    eval_paired(model, SNRS)
