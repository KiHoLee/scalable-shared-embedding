"""Extensions of exp_pm.py: shared content, channel-adapted masks, signaling.

part corr : correlated content. With probability rho every user receives
            the same token (a common message), otherwise the tokens are
            independent. Compared at each rho: the zero-cross-energy comb
            pattern (L = U), fully overlapping masks (all users weight every
            dimension equally), and masks trained under that rho (L = 16 and
            L = 64). Reports SER and the cross-energy of the trained masks.
part csit : coherence-aware mask selection. Within its coherence time the
            transmitter knows every user's subcarrier gains and, frame by
            frame, selects one of 24 entries. In each entry every mask
            weights one of four contiguous blocks of eight subcarriers
            (about four coherence bandwidths at four taps, where the gains
            stay correlated above 0.9 over about two subcarriers), and the
            24 entries are the orders of the four users on these blocks.
            The selected entry has the largest sum over users of
            log(1 + SNR) on the block that each mask weights. Compared with
            the same blocks in a fixed order and with the comb pattern
            (L = U).
part adapt: masks adapted per update to known subcarrier gains, one value
            per block of B subcarriers. The users take turns in an order
            reversed every round, and each sets its weight on the block with
            its largest sum of log(1 + SNR) among those that no mask weights
            yet. Two weight levels, or real weights water-filled over these
            blocks, B = 1, 2, 4, 8, at 4 and 8 taps (supplementary signaling
            table).
part sic  : receivers that cancel the other users' decoded tokens, for the
            comb pattern, full overlap, and masks trained for each rho.
part rs   : one common and one private token per user and frame: common
            codeword on every dimension with and without its removal before
            the private token, against copies of the common token on masks
            without overlap (no_overlap_copies) (supplementary table).
part figs : the data of Figs. 2 and 3 (graphs g1, g2) and the single-run
            signaling data g3, which part sig replaces in Fig. 4; needs the
            models white_U4_L16_flat_s1 (exp_pm.py --part channel) and
            bertw_U4_L*_flat_s1 (exp_pm.py --part period). Fig. 2 also uses
            exp_pm_chanhi.py (static entry trained for each class, adapted
            masks at 12-20 dB with 400,000 frames).
part sig  : the data of Figs. 4 and 5, J = 1..64 values per mask. Static
            entries of length L = J for independent content (three runs of
            exp_pm.py --part period) and trained for rho = 0.9 (three runs),
            each scored on validation frames (seed 21) and test frames, and
            masks adapted to known gains on J blocks of 64/J real dimensions
            at 4 and 8 taps; needs bertw_U4_L*_flat_s{1,2,3}.
part content_snr: the content step at 0, 10 and 20 dB with masks trained
            on SNRs from 0 to 20 dB (as in Fig. 3) and masks trained at the
            SNR they serve, SER split into frames with a shared token and
            frames with independent tokens.
part sig10: the static entries of Fig. 4 trained at the 10 dB they serve,
            for rho = 0 and rho = 0.9, three runs per pattern length.
part sig10rho: the same for rho = 0.25, 0.5, 0.75 and 1.
All parts: U = 4, d = 64, BERT codebook whitened under the source
statistics, weighted minimum-distance decoding over all 30,522 tokens,
SER on tokens drawn from the source statistics.
Outputs in data/pm/: pm_corr.csv, pm_corr.json, pm_csit.csv, pm_adapt.csv,
pm_sic.csv, pm_rs.csv, pm_figs.csv, pm_sig.csv, pm_content_snr.csv, pm_sig10.csv,
pm_sig10_rho.csv
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import time

import torch

import rev_channel as rc
from sse_lib import DATA, DEVICE, D, set_seed, write_csv
import exp_pm as pm

DATA = pm.DATA                     # data/pm/

U = 4


def sample(pd, n, users, rho, gen=None):
    common = torch.multinomial(pd, n, replacement=True, generator=gen)
    indep = torch.multinomial(pd, n * users, replacement=True, generator=gen).view(n, users)
    same = torch.rand(n, generator=gen, device=pd.device) < rho if gen is None else \
        (torch.rand(n, generator=gen) < rho).to(pd.device)
    return torch.where(same[:, None], common[:, None].expand(n, users), indep)


class FreeMasks(pm.Masks):
    """Masks given directly (no normalization by training)."""

    def __init__(self, E, M):
        super().__init__(E, M.shape[1], M.shape[0])
        self.Wm.data.copy_(M)

    def masks(self):
        return self.Wm


@torch.no_grad()
def calibrate_rho(m, pd, rho, n=65536):
    idx = sample(pd, n, m.users, rho)
    m.c.fill_(float(m.tx(idx).pow(2).sum(1).mean().sqrt()))


@torch.no_grad()
def ser(m, pd, rho, snr, cond, frames=40000, seed=11):
    torch.manual_seed(seed + int(10 * snr))
    err = 0
    for k in range(0, frames, 512):
        n = min(512, frames - k)
        idx = sample(pd, n, m.users, rho)
        x = m.tx(idx) / m.c
        r, w = rc.channel(x, snr, m.users, **cond)
        err += int((m.scores(r, w).argmax(-1) != idx).sum())
    return err / (frames * m.users)


def train_rho(E, L, pd, rho, iters, seed=1, snr_range=(0.0, 20.0)):
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(seed)
    m = pm.Masks(E, L, U).to(DEVICE)
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    ce = torch.nn.CrossEntropyLoss()
    for it in range(iters):
        idx = sample(pd, 256, U, rho)
        snr = torch.empty(256).uniform_(*snr_range)
        calibrate_rho(m, pd, rho, 8192)
        x = m.tx(idx) / m.c
        r, w = rc.channel(x, snr, U, **pm.COND["flat"])
        sc = m.scores(r, w) * m.logit_scale.exp()
        loss = ce(sc.reshape(-1, sc.shape[-1]), idx.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    torch.backends.cuda.matmul.allow_tf32 = False
    calibrate_rho(m, pd, rho)
    return m.eval()


def part_corr(a):
    src = pm.Source("bert")
    pdc = src.dist()
    E = pm.codebook(src.W, D, "pwhite", pd=pdc).to(DEVICE)
    pd = pdc.to(DEVICE)
    rows, info = [], {}
    comb = torch.zeros(U, 4)
    for u in range(U):
        comb[u, u] = 2.0                       # ||mu_u||^2 = L = 4
    comb = comb.repeat(1, D // 4)
    over = torch.ones(U, D)
    for rho in (0.0, 0.5, 0.9, 1.0):
        cands = {"comb": FreeMasks(E, comb.to(DEVICE)).to(DEVICE),
                 "overlap": FreeMasks(E, over.to(DEVICE)).to(DEVICE)}
        for L in (16, 64):
            cands[f"trained_L{L}"] = train_rho(E, L, pd, rho, a.iters)
        for name, m in cands.items():
            calibrate_rho(m, pd, rho)
            Mk = m.masks().detach().cpu()
            iota = pm.cross_energy(Mk)
            info[f"{name}_rho{rho}"] = dict(iota=iota, c=float(m.c))
            for snr in (0, 10, 20):
                v = ser(m, pd, rho, snr, pm.COND["flat"])
                rows.append([rho, name, snr, v, sum(iota) / U])
                print("[corr]", rho, name, snr, round(v, 4), "iota", round(sum(iota) / U, 4), flush=True)
    write_csv(DATA / "pm_corr.csv", ["rho", "masks", "snr_db", "ser", "mean_iota"], rows)
    json.dump(info, open(DATA / "pm_corr.json", "w"), indent=1)


# ---------------- coherence-aware selection -----------------------------
def gains(N, taps, device):
    K = D // 2

    def cn(*shape):
        return torch.complex(torch.randn(*shape, device=device),
                             torch.randn(*shape, device=device)) / math.sqrt(2)
    if taps == 1:
        return cn(N, U, 1).expand(N, U, K).contiguous()
    a = cn(N, U, taps) / math.sqrt(taps)
    k = torch.arange(K, device=device)
    t = torch.arange(taps, device=device)
    F = torch.exp(-2j * math.pi * torch.outer(t, k).float() / K)
    return a @ F.to(a.dtype)


def block_masks(perm):
    """perm[b] = user whose mask weights block b; four blocks of 8
    subcarriers = 16 dims."""
    M = torch.zeros(U, D)
    for b, u in enumerate(perm):
        M[u, 16 * b:16 * (b + 1)] = 2.0          # ||m_u||^2 = 64
    return M



def waterfill(inv, total):
    """inv (..., n) = noise-to-gain ratios of the dims of one user; returns
    powers p >= 0 with sum total that maximize sum log(1 + 1/inv * p)."""
    srt, order = inv.sort(-1)
    n = inv.shape[-1]
    ks = torch.arange(1, n + 1, device=inv.device, dtype=inv.dtype)
    mu = (total + srt.cumsum(-1)) / ks
    valid = mu > srt
    kstar = (valid * ks).argmax(-1, keepdim=True)          # last valid index
    mu_s = mu.gather(-1, kstar)
    p = (mu_s - inv).clamp_min(0.0)
    return p * (total / p.sum(-1, keepdim=True).clamp_min(1e-12))


def wf_masks(M, g, sig2, c):
    """Keep the support of M (n,U,D), water-fill each user's weight over its
    support with the per-dimension gains |g|^2 (n,U,K), same total d."""
    gd = (g.abs() ** 2).repeat_interleave(2, dim=-1)        # (n,U,D)
    sup = M > 0
    inv = torch.where(sup, sig2 * c * c / gd.clamp_min(1e-12), torch.full_like(gd, 1e9))
    nsup = int(sup[0, 0].sum())
    # gather the supported dims of every user (same count for every user)
    pos = sup.float().argsort(-1, descending=True)[..., :nsup]
    inv_s = inv.gather(-1, pos)
    p = waterfill(inv_s, float(D))
    out = torch.zeros_like(M)
    out.scatter_(-1, pos, p.sqrt())
    return out


@torch.no_grad()
def run_csit(E, pd, c, snr, taps, policy, frames=40000, seed=21):
    torch.manual_seed(seed + int(10 * snr) + 100 * taps)
    perms = list(itertools.permutations(range(U)))
    entries = torch.stack([block_masks(p) for p in perms]).to(DEVICE)  # (24,U,D)
    comb = torch.zeros(U, D, device=DEVICE)
    for u in range(U):
        comb[u, u::4] = 2.0
    K = D // 2
    gam = 10 ** (snr / 10)
    sig = math.sqrt(1.0 / (D * gam))
    err = 0
    for k0 in range(0, frames, 1024):
        n = min(1024, frames - k0)
        idx = sample(pd, n, U, 0.0)
        g = gains(n, taps, DEVICE)                                       # (n,U,K)
        if policy.startswith("comb"):
            M = comb[None].expand(n, U, D)
        elif policy == "block_fixed":
            M = entries[0][None].expand(n, U, D)
        else:   # block_csit: select the entry with the largest sum of log(1+SNR)
            snr_k = gam * g.abs() ** 2                                    # (n,U,K)
            blk = torch.log1p(snr_k).view(n, U, 4, 8).sum(-1)            # (n,U,4) users x blocks
            perm_t = torch.tensor(perms, device=DEVICE)                  # (24,4) block->user
            util = torch.stack([blk[:, perm_t[p], torch.arange(4, device=DEVICE)].sum(-1)
                                for p in range(len(perms))], 1)          # (n,24)
            M = entries[util.argmax(1)]                                    # (n,U,D)
        if policy.endswith("_wf"):
            M = wf_masks(M.contiguous(), g, sig * sig, c)
        x = (E[idx] * M).sum(1) / c                                       # (n,D)
        xc = torch.complex(x[:, 0::2], x[:, 1::2])
        w = torch.complex(torch.randn(n, U, K, device=DEVICE), torch.randn(n, U, K, device=DEVICE)) * sig
        z = g * xc[:, None, :] + w
        q = z / g
        r = torch.stack([q.real, q.imag], -1).reshape(n, U, D)
        wt = (g.abs() ** 2).repeat_interleave(2, dim=-1)
        a_ = 1.0 / c
        sc = 2 * a_ * torch.einsum("nud,vd->nuv", wt * r * M, E) \
            - a_ * a_ * torch.einsum("nud,vd->nuv", wt * M * M, E * E)
        err += int((sc.argmax(-1) != idx).sum())
    return err / (frames * U)


def part_csit(a):
    src = pm.Source("bert")
    pdc = src.dist()
    E = pm.codebook(src.W, D, "pwhite", pd=pdc).to(DEVICE)
    pd = pdc.to(DEVICE)
    # in every policy each mask has energy d (unit mean-square weight) and the
    # masks do not overlap (no cross-energy), so c^2 is the same for all of them
    comb = torch.zeros(U, D, device=DEVICE)
    for u in range(U):
        comb[u, u::4] = 2.0
    idx = sample(pd, 65536, U, 0.0)
    c = float(((E[idx] * comb[None]).sum(1)).pow(2).sum(1).mean().sqrt())
    rows = []
    for taps in (1, 4, 8):
        for snr in (0, 10, 20):
            for pol in ("comb", "comb_wf", "block_fixed", "block_csit", "block_csit_wf"):
                v = run_csit(E, pd, c, snr, taps, pol)
                rows.append([taps, pol, snr, v])
                print("[csit]", taps, pol, snr, round(v, 4), flush=True)
    write_csv(DATA / "pm_csit.csv", ["taps", "policy", "snr_db", "ser"], rows)



@torch.no_grad()
def adaptive_masks(g, B, sig2, c, wf):
    """Per-frame masks from the gains g (n,U,K). The band is cut into units of
    B subcarriers, one partition for all users. The users take turns
    (reversing the order every round), and each sets its weight on the unit
    with its largest sum of log(1+SNR) among those that no mask weights yet,
    so every mask has weight on K/(B U) units and the masks do not overlap.
    With wf, each mask holds real weights water-filled over these units, one
    value per unit; otherwise it holds one level on all of them (two weight
    levels, that level and zero)."""
    n, Uu, K = g.shape
    nu = K // B
    per = nu // Uu
    g2 = g.abs() ** 2                                              # (n,U,K)
    gam = 1.0 / (D * sig2)
    util = torch.log1p(gam * g2).view(n, Uu, nu, B).sum(-1)       # (n,U,nu)
    taken = torch.zeros(n, nu, dtype=torch.bool, device=g.device)
    owner = torch.full((n, nu), -1, dtype=torch.long, device=g.device)
    ar = torch.arange(n, device=g.device)
    for r in range(per):
        order = range(Uu) if r % 2 == 0 else range(Uu - 1, -1, -1)
        for u in order:
            sc = util[:, u].masked_fill(taken, -1e9)
            k = sc.argmax(1)
            taken[ar, k] = True
            owner[ar, k] = u
    M = torch.zeros(n, Uu, nu, device=g.device)
    if not wf:
        M[ar[:, None], owner, torch.arange(nu, device=g.device)[None]] = 1.0
        M = M * (D / (2 * B * per))                                 # weight^2 per dim
    else:
        mg = g2.view(n, Uu, nu, B).mean(-1)                         # (n,U,nu) unit gain
        for u in range(Uu):
            mine = owner == u                                       # (n,nu)
            pos = mine.float().argsort(-1, descending=True)[:, :per]
            inv = (sig2 * c * c / mg[:, u].gather(1, pos).clamp_min(1e-12))
            pw = waterfill(inv, D / (2 * B))                        # per-dim power per unit
            M[ar[:, None], u, pos] = pw
    W2 = M.repeat_interleave(2 * B, dim=-1)                         # (n,U,D) weight^2
    return W2.sqrt()


@torch.no_grad()
def run_adapt(E, pd, c, snr, taps, B, wf, frames=40000, seed=31):
    torch.manual_seed(seed + int(10 * snr) + 100 * taps + 7 * B)
    K = D // 2
    sig = math.sqrt(1.0 / (D * 10 ** (snr / 10)))
    err = 0
    for k0 in range(0, frames, 384):
        n = min(384, frames - k0)
        idx = sample(pd, n, U, 0.0)
        g = gains(n, taps, DEVICE)
        M = adaptive_masks(g, B, sig * sig, c, wf)
        x = (E[idx] * M).sum(1) / c
        xc = torch.complex(x[:, 0::2], x[:, 1::2])
        w = torch.complex(torch.randn(n, U, K, device=DEVICE), torch.randn(n, U, K, device=DEVICE)) * sig
        q = (g * xc[:, None, :] + w) / g
        r = torch.stack([q.real, q.imag], -1).reshape(n, U, D)
        wt = (g.abs() ** 2).repeat_interleave(2, dim=-1)
        a_ = 1.0 / c
        sc = 2 * a_ * torch.einsum("nud,vd->nuv", wt * r * M, E) \
            - a_ * a_ * torch.einsum("nud,vd->nuv", wt * M * M, E * E)
        err += int((sc.argmax(-1) != idx).sum())
    return err / (frames * U)


def part_adapt(a):
    """Channel-adaptive masks at different unit sizes B (subcarriers):
    performance against the number of mask values a user must learn per
    frame (K/(B U) unit indices, plus as many power values with wf)."""
    src = pm.Source("bert")
    pdc = src.dist()
    E = pm.codebook(src.W, D, "pwhite", pd=pdc).to(DEVICE)
    pd = pdc.to(DEVICE)
    comb = torch.zeros(U, D, device=DEVICE)
    for u in range(U):
        comb[u, u::4] = 2.0
    idx = sample(pd, 65536, U, 0.0)
    c = float(((E[idx] * comb[None]).sum(1)).pow(2).sum(1).mean().sqrt())
    rows = []
    for taps in (4, 8):
        for snr in (0, 5, 10, 15, 20):
            for B in (1, 2, 4, 8):
                for wf in (False, True):
                    v = run_adapt(E, pd, c, snr, taps, B, wf)
                    units = (D // 2) // (B * U)
                    rows.append([taps, B, int(wf), units, snr, v])
                    print("[adapt]", taps, B, int(wf), units, snr, round(v, 4), flush=True)
    write_csv(DATA / "pm_adapt.csv", ["taps", "unit_subcarriers", "waterfill", "units_per_user", "snr_db", "ser"], rows)



@torch.no_grad()
def joint_receivers(m, pd, rho, snr, frames=20000, iters=3, seed=41):
    """Receiver u knows every user's mask (the stored entry) and decodes all U
    tokens from its own observation. single: own-mask minimum distance.
    pic: parallel interference cancellation, iters rounds. pic_common: also
    tests the common-message hypothesis (one token on the summed mask) and
    keeps the hypothesis with the smaller residual energy."""
    torch.manual_seed(seed + int(10 * snr))
    E = m.E
    M = m.masks().detach()                                   # (U,D)
    a_ = 1.0 / float(m.c)
    cand = E[None] * M[:, None, :] * a_                      # (U,V,D) scaled candidates
    cn2 = cand.pow(2).sum(-1)                                # (U,V)
    msum = M.sum(0)
    ccom = E * msum[None] * a_                               # (V,D)
    ccn2 = ccom.pow(2).sum(-1)
    err = {"single": 0, "pic": 0, "pic_common": 0}
    for k0 in range(0, frames, 256):
        n = min(256, frames - k0)
        idx = sample(pd, n, U, rho)
        x = m.tx(idx) / m.c
        r_all, w_all = rc.channel(x, snr, U, **pm.COND["flat"])     # (n,U,D)
        for u in range(U):
            r = r_all[:, u]                                          # (n,D)
            # single-user decision of user u
            s_own = (2 * r @ cand[u].T - cn2[u][None]).argmax(1)
            err["single"] += int((s_own != idx[:, u]).sum())
            # PIC over all users from receiver u's observation
            est = (2 * torch.einsum("nd,kvd->nkv", r, cand) - cn2[None]).argmax(-1)   # (n,U)
            for _ in range(iters):
                recon = cand[torch.arange(U)[None], est]              # (n,U,D)
                tot = recon.sum(1)
                res = r[:, None, :] - (tot[:, None, :] - recon)       # remove the others
                est = (2 * torch.einsum("nkd,kvd->nkv", res, cand) - cn2[None]).argmax(-1)
            err["pic"] += int((est[:, u] != idx[:, u]).sum())
            # common hypothesis
            vc = (2 * r @ ccom.T - ccn2[None]).argmax(1)
            res_c = (r - ccom[vc]).pow(2).sum(1)
            recon = cand[torch.arange(U)[None], est].sum(1)
            res_i = (r - recon).pow(2).sum(1)
            dec = torch.where(res_c < res_i, vc, est[:, u])
            err["pic_common"] += int((dec != idx[:, u]).sum())
    return {k: v / (frames * U) for k, v in err.items()}


def part_sic(a):
    src = pm.Source("bert")
    pdc = src.dist()
    E = pm.codebook(src.W, D, "pwhite", pd=pdc).to(DEVICE)
    pd = pdc.to(DEVICE)
    comb = torch.zeros(U, 4)
    for u in range(U):
        comb[u, u] = 2.0
    comb = comb.repeat(1, D // 4)
    rows = []
    for rho in (0.0, 0.5, 0.9, 1.0):
        cands = {"comb": FreeMasks(E, comb.to(DEVICE)).to(DEVICE),
                 "overlap": FreeMasks(E, torch.ones(U, D, device=DEVICE)).to(DEVICE),
                 "trained_L16": train_rho(E, 16, pd, rho, a.iters)}
        for name, m in cands.items():
            calibrate_rho(m, pd, rho)
            iota = sum(pm.cross_energy(m.masks().detach().cpu())) / U
            for snr in (0, 10, 20):
                res = joint_receivers(m, pd, rho, snr)
                for rx, v in res.items():
                    rows.append([rho, name, rx, snr, v, iota])
                print("[sic]", rho, name, snr, {k: round(v, 4) for k, v in res.items()},
                      "iota", round(iota, 3), flush=True)
    write_csv(DATA / "pm_sic.csv", ["rho", "masks", "receiver", "snr_db", "ser", "mean_iota"], rows)



@torch.no_grad()
def run_rs(E, pd, snr, alpha, scheme, frames=30000, seed=51):
    """Common + private content in every frame: one common token for all
    users and one private token per user. Total frame energy one.
      shared_sic   : common codeword on all d dims (energy alpha), private
                     codewords on the dims that each comb mask weights
                     (energy (1-alpha)/U each); receiver decodes the common
                     token first, removes it, then decodes its private token
      shared_nosic : same transmission, private token decoded with the common
                     component left as interference
      no_overlap_copies : no overlap; the dims that each comb mask weights
                     are halved, one half carries the user's copy of the
                     common token (energy alpha/U), the other its private
                     token
    Flat Rayleigh fading per user, exact gains, minimum-distance decoding."""
    torch.manual_seed(seed + int(10 * snr) + int(100 * alpha))
    sig = math.sqrt(1.0 / (D * 10 ** (snr / 10)))
    comb = torch.zeros(U, D, device=DEVICE)
    for u in range(U):
        comb[u, u::4] = 2.0                      # ||m||^2 = d, so ||e.m||^2 ~ 1
    half_c = torch.zeros(U, D, device=DEVICE)
    half_p = torch.zeros(U, D, device=DEVICE)
    for u in range(U):
        dims = torch.arange(u, D, 4, device=DEVICE)
        half_c[u, dims[0::2]] = math.sqrt(8.0)   # 8 dims, ||m||^2 = d
        half_p[u, dims[1::2]] = math.sqrt(8.0)
    ec = ep = 0
    for k0 in range(0, frames, 512):
        n = min(512, frames - k0)
        cidx = torch.multinomial(pd, n, replacement=True)
        pidx = torch.multinomial(pd, n * U, replacement=True).view(n, U)
        if scheme.startswith("shared"):
            x = math.sqrt(alpha) * E[cidx] + math.sqrt((1 - alpha) / U) * (E[pidx] * comb[None]).sum(1)
        else:
            x = (math.sqrt(alpha / U) * E[cidx][:, None, :] * half_c[None]
                 + math.sqrt((1 - alpha) / U) * E[pidx] * half_p[None]).sum(1)
        h = pm.rayleigh_gain((n, U)) if hasattr(pm, "rayleigh_gain") else None
        h = torch.sqrt(-torch.log(torch.rand(n, U, device=DEVICE).clamp_min(1e-12)))
        r = x[:, None, :] + sig * torch.randn(n, U, D, device=DEVICE) / h[:, :, None]
        for u in range(U):
            ru = r[:, u]
            if scheme.startswith("shared"):
                cand_c = math.sqrt(alpha) * E                                   # (V,D)
                c_hat = (2 * ru @ cand_c.T - cand_c.pow(2).sum(1)[None]).argmax(1)
                if scheme == "shared_sic":
                    ru = ru - math.sqrt(alpha) * E[c_hat]
                cand_p = math.sqrt((1 - alpha) / U) * E * comb[u][None]
            else:
                cand_c = math.sqrt(alpha / U) * E * half_c[u][None]
                c_hat = (2 * ru @ cand_c.T - cand_c.pow(2).sum(1)[None]).argmax(1)
                cand_p = math.sqrt((1 - alpha) / U) * E * half_p[u][None]
            p_hat = (2 * ru @ cand_p.T - cand_p.pow(2).sum(1)[None]).argmax(1)
            ec += int((c_hat != cidx).sum())
            ep += int((p_hat != pidx[:, u]).sum())
    return ec / (frames * U), ep / (frames * U)


def part_rs(a):
    src = pm.Source("bert")
    pdc = src.dist()
    E = pm.codebook(src.W, D, "pwhite", pd=pdc).to(DEVICE)
    pd = pdc.to(DEVICE)
    rows = []
    for alpha in (0.2, 0.35, 0.5):
        for snr in (0, 10, 20):
            for sch in ("shared_sic", "shared_nosic", "no_overlap_copies"):
                c_ser, p_ser = run_rs(E, pd, snr, alpha, sch)
                rows.append([alpha, sch, snr, c_ser, p_ser])
                print("[rs]", alpha, sch, snr, "common", round(c_ser, 4), "private", round(p_ser, 4), flush=True)
    write_csv(DATA / "pm_rs.csv", ["alpha", "scheme", "snr_db", "ser_common", "ser_private"], rows)



def part_figs(a):
    """Consistent data for the three result graphs, all on tokens drawn with
    the source statistics, 40,000 frames per point, U = 4.
      g1: SER vs SNR under flat, 4-tap and 8-tap fading for OMA, the trained
          static entry (L = 16, flat), and masks adapted per frame to known
          gains (blocks of B = 2 subcarriers)
      g2: SER at 0, 10, 20 dB vs the probability rho that all users receive
          the same token, for the pattern without overlap, full overlap, and
          masks trained for each rho (L = 16)
      g3: SER at 10 dB vs the mask values a receiver must learn per user:
          static entries of length L (flat, independent content), entries of
          length L trained for rho = 0.9, adapted masks with blocks of B
          subcarriers (32/B values per user and frame) at 4 and 8 taps"""
    src = pm.Source("bert")
    pdc = src.dist()
    E = pm.codebook(src.W, D, "pwhite", pd=pdc).to(DEVICE)
    pd = pdc.to(DEVICE)
    comb = torch.zeros(U, D, device=DEVICE)
    for u in range(U):
        comb[u, u::4] = 2.0
    idx0 = sample(pd, 65536, U, 0.0)
    c_comb = float(((E[idx0] * comb[None]).sum(1)).pow(2).sum(1).mean().sqrt())
    rows = []

    def load(name, L):
        sd = torch.load(pm.MODELS / f"{name}.pt", map_location="cpu")
        m = pm.Masks(sd["E"], L, U)
        m.load_state_dict(sd)
        return m.to(DEVICE).eval()

    # ---- g1
    static = load("white_U4_L16_flat_s1", 16)
    for taps, cn in ((1, "flat"), (4, "fs4"), (8, "fs8")):
        cond = pm.COND[cn]
        for snr in range(0, 21, 2):
            torch.manual_seed(5000 + snr + 100 * taps)
            ids = sample(pdc, 40000, U, 0.0).cpu()
            dec = pm.dec_oma(ids, snr, cond, 777 + snr)
            v_oma = float((dec != ids).float().mean())
            v_st = ser(static, pd, 0.0, snr, cond)
            v_ad = run_adapt(E, pd, c_comb, snr, taps, 2, False)
            for name, v in (("oma", v_oma), ("static_L16", v_st), ("adapted_B2", v_ad)):
                rows.append(["g1", cn, name, snr, 0.0, 0, v])
            print("[g1]", cn, snr, round(v_oma, 4), round(v_st, 4), round(v_ad, 4), flush=True)
    # ---- g2
    for rho in (0.0, 0.25, 0.5, 0.75, 0.9, 1.0):
        ms = {"no_overlap": FreeMasks(E, comb).to(DEVICE),
              "full_overlap": FreeMasks(E, torch.ones(U, D, device=DEVICE)).to(DEVICE),
              "trained_L16": train_rho(E, 16, pd, rho, a.iters)}
        for name, m in ms.items():
            calibrate_rho(m, pd, rho)
            io = sum(pm.cross_energy(m.masks().detach().cpu())) / U
            for snr in (0, 10, 20):
                v = ser(m, pd, rho, snr, pm.COND["flat"])
                rows.append(["g2", "flat", name, snr, rho, io, v])
                print("[g2]", rho, name, snr, round(v, 4), "iota", round(io, 3), flush=True)
    # ---- g3
    for L in (64, 32, 16, 8, 4, 2, 1):
        m = load(f"bertw_U4_L{L}_flat_s1", L)
        v = ser(m, pd, 0.0, 10, pm.COND["flat"])
        rows.append(["g3", "flat", "static", 10, 0.0, L, v])
        print("[g3] static", L, round(v, 4), flush=True)
    for L in (64, 32, 16, 8, 4, 2, 1):
        m = train_rho(E, L, pd, 0.9, a.iters)
        v = ser(m, pd, 0.9, 10, pm.COND["flat"])
        rows.append(["g3", "flat", "shared_rho0.9", 10, 0.9, L, v])
        print("[g3] shared", L, round(v, 4), flush=True)
    for taps, cn in ((4, "fs4"), (8, "fs8")):
        for B in (1, 2, 4, 8):
            v = run_adapt(E, pd, c_comb, 10, taps, B, False)
            rows.append(["g3", cn, "adapted", 10, 0.0, (D // 2) // B, v])
            print("[g3] adapted", cn, B, round(v, 4), flush=True)
    write_csv(DATA / "pm_figs.csv", ["graph", "channel", "scheme", "snr_db", "rho", "aux", "ser"], rows)


@torch.no_grad()
def unit_masks(g, ud, sig2):
    """Per-update masks from the gains g (n,U,K) on units of ud real
    dimensions (ud = 2B for blocks of B subcarriers, ud = 1 for single
    dimensions). With at least U units the users take turns (reversing the
    order every round), and each sets its level on the unit with its largest
    sum of log(1+SNR) among those that no mask weights yet, as
    adaptive_masks, so every mask has weight on D/(ud U) units and the masks
    do not overlap. With fewer units than users each user sets its level on
    its best unit, so the masks overlap. One value per unit, two weight
    levels, ||m_u||^2 = D."""
    n, Uu, K = g.shape
    nu = D // ud
    gam = 1.0 / (D * sig2)
    util = torch.log1p(gam * g.abs() ** 2).repeat_interleave(2, dim=-1).view(n, Uu, nu, ud).sum(-1)
    ar = torch.arange(n, device=g.device)
    M = torch.zeros(n, Uu, nu, device=g.device)
    if nu >= Uu:
        per = nu // Uu
        taken = torch.zeros(n, nu, dtype=torch.bool, device=g.device)
        owner = torch.full((n, nu), -1, dtype=torch.long, device=g.device)
        for r in range(per):
            order = range(Uu) if r % 2 == 0 else range(Uu - 1, -1, -1)
            for u in order:
                k = util[:, u].masked_fill(taken, -1e9).argmax(1)
                taken[ar, k] = True
                owner[ar, k] = u
        M[ar[:, None], owner, torch.arange(nu, device=g.device)[None]] = D / (ud * per)
    else:
        k = util.argmax(-1)                                          # (n,U) best unit
        for u in range(Uu):
            M[ar, u, k[:, u]] = D / ud
    return M.repeat_interleave(ud, dim=-1).sqrt()                    # (n,U,D)


@torch.no_grad()
def run_units(E, pd, c, snr, taps, ud, frames=40000, seed=31):
    """SER of the unit masks; for ud = 2B >= 2 the seed and the frames equal
    those of run_adapt(..., B, wf=False)."""
    torch.manual_seed(seed + int(10 * snr) + 100 * taps + int(7 * ud / 2))
    K = D // 2
    sig = math.sqrt(1.0 / (D * 10 ** (snr / 10)))
    err = 0
    for k0 in range(0, frames, 384):
        n = min(384, frames - k0)
        idx = sample(pd, n, U, 0.0)
        g = gains(n, taps, DEVICE)
        M = unit_masks(g, ud, sig * sig)
        x = (E[idx] * M).sum(1) / c
        xc = torch.complex(x[:, 0::2], x[:, 1::2])
        w = torch.complex(torch.randn(n, U, K, device=DEVICE), torch.randn(n, U, K, device=DEVICE)) * sig
        q = (g * xc[:, None, :] + w) / g
        r = torch.stack([q.real, q.imag], -1).reshape(n, U, D)
        wt = (g.abs() ** 2).repeat_interleave(2, dim=-1)
        a_ = 1.0 / c
        sc = 2 * a_ * torch.einsum("nud,vd->nuv", wt * r * M, E) \
            - a_ * a_ * torch.einsum("nud,vd->nuv", wt * M * M, E * E)
        err += int((sc.argmax(-1) != idx).sum())
    return err / (frames * U)


def part_sig(a):
    """Data of the signaling figure on every mask length 1..64 for every
    curve: static entries for independent content (the three training runs
    of exp_pm.py --part period), static entries trained for rho = 0.9 (three
    training runs each), and masks adapted to known gains on units of
    64/J real dimensions for J = 1..64 values per mask at 4 and 8 taps
    (with J < U the masks overlap; J = 64 has units of single dimensions). SER at
    10 dB on tokens drawn with the source statistics; the static entries are
    also scored on separate validation frames (seed 21)."""
    src = pm.Source("bert")
    pdc = src.dist()
    E = pm.codebook(src.W, D, "pwhite", pd=pdc).to(DEVICE)
    pd = pdc.to(DEVICE)
    comb = torch.zeros(U, D, device=DEVICE)
    for u in range(U):
        comb[u, u::4] = 2.0
    torch.manual_seed(0)                       # fixed draw for the constant c
    idx0 = sample(pd, 65536, U, 0.0)
    c_comb = float(((E[idx0] * comb[None]).sum(1)).pow(2).sum(1).mean().sqrt())
    print("[sig] c", c_comb, flush=True)
    rows = []
    Ls = (64, 32, 16, 8, 4, 2, 1)
    for L in Ls:
        for s in (1, 2, 3):
            sd = torch.load(pm.MODELS / f"bertw_U4_L{L}_flat_s{s}.pt", map_location="cpu")
            m = pm.Masks(sd["E"], L, U)
            m.load_state_dict(sd)
            m = m.to(DEVICE).eval()
            v_val = ser(m, pd, 0.0, 10, pm.COND["flat"], seed=21)
            v = ser(m, pd, 0.0, 10, pm.COND["flat"])
            rows.append(["static", "flat", 0.0, L, s, v_val, v])
            print("[sig] static", L, s, round(v_val, 4), round(v, 4), flush=True)
    for L in Ls:
        for s in (1, 2, 3):
            m = train_rho(E, L, pd, 0.9, a.iters, seed=s)
            v_val = ser(m, pd, 0.9, 10, pm.COND["flat"], seed=21)
            v = ser(m, pd, 0.9, 10, pm.COND["flat"])
            sign = [round(float(z), 3) for z in m.masks().detach()[:, 0].cpu()] if L == 1 else ""
            rows.append(["shared_rho0.9", "flat", 0.9, L, s, v_val, v])
            print("[sig] shared", L, s, round(v_val, 4), round(v, 4), sign, flush=True)
    for taps, cn in ((4, "fs4"), (8, "fs8")):
        for v_ in (64, 32, 16, 8, 4, 2, 1):
            ud = D // v_
            v = run_units(E, pd, c_comb, 10, taps, ud)
            rows.append(["adapted", cn, 0.0, v_, 1, "", v])
            print("[sig] adapted", cn, v_, round(v, 4), flush=True)
    write_csv(DATA / "pm_sig.csv", ["scheme", "channel", "rho", "values", "seed", "ser_val", "ser"], rows)



@torch.no_grad()
def ser_split(m, pd, rho, snr, frames=40000, seed=11):
    """SER under flat fading, split into frames whose users share the token
    and frames with independent tokens (a frame counts as shared when all
    users receive the same token)."""
    torch.manual_seed(seed + int(10 * snr))
    e_s = n_s = e_i = n_i = 0
    for k in range(0, frames, 512):
        n = min(512, frames - k)
        idx = sample(pd, n, m.users, rho)
        x = m.tx(idx) / m.c
        r, w = rc.channel(x, snr, m.users, **pm.COND["flat"])
        wrong = (m.scores(r, w).argmax(-1) != idx)
        same = (idx == idx[:, :1]).all(1)
        e_s += int(wrong[same].sum()); n_s += int(same.sum()) * m.users
        e_i += int(wrong[~same].sum()); n_i += int((~same).sum()) * m.users
    tot = (e_s + e_i) / (n_s + n_i)
    return tot, (e_s / n_s if n_s else float("nan")), (e_i / n_i if n_i else float("nan"))


def part_content_snr(a):
    """Content step with masks trained at the SNR they serve. For every rho
    and every SNR of 0, 10 and 20 dB: masks without overlap, full overlap,
    masks trained on SNRs drawn from 0 to 20 dB (as in Fig. 3), and masks
    trained at that SNR alone; SER in total and split by frame kind
    (data/pm/pm_content_snr.csv)."""
    src = pm.Source("bert")
    pdc = src.dist()
    E = pm.codebook(src.W, D, "pwhite", pd=pdc).to(DEVICE)
    pd = pdc.to(DEVICE)
    comb = torch.zeros(U, D, device=DEVICE)
    for u in range(U):
        comb[u, u::4] = 2.0
    rows = []
    for rho in (0.0, 0.25, 0.5, 0.75, 0.9, 1.0):
        fixed = {"no_overlap": FreeMasks(E, comb).to(DEVICE),
                 "full_overlap": FreeMasks(E, torch.ones(U, D, device=DEVICE)).to(DEVICE),
                 "trained_0_20dB": train_rho(E, 16, pd, rho, a.iters)}
        for snr in (0, 10, 20):
            ms = dict(fixed)
            ms["trained_at_snr"] = train_rho(E, 16, pd, rho, a.iters, snr_range=(float(snr), float(snr)))
            for name, m in ms.items():
                calibrate_rho(m, pd, rho)
                io = sum(pm.cross_energy(m.masks().detach().cpu())) / U
                tot, s_sh, s_in = ser_split(m, pd, rho, snr)
                rows.append([name, snr, rho, io, tot, s_sh, s_in])
                print("[content_snr]", rho, snr, name, f"ser {tot:.4f} shared {s_sh:.4f} indep {s_in:.4f} iota {io:.3f}", flush=True)
    write_csv(DATA / "pm_content_snr.csv", ["scheme", "snr_db", "rho", "cross_energy", "ser", "ser_shared", "ser_indep"], rows)


def part_sig10(a):
    """Static entries of Fig. 4 trained at the 10 dB they serve, for
    independent content (rho = 0) and for rho = 0.9, every pattern length
    1..64, three training runs each, scored on validation frames (seed 21)
    and on test frames (data/pm/pm_sig10.csv)."""
    src = pm.Source("bert")
    pdc = src.dist()
    E = pm.codebook(src.W, D, "pwhite", pd=pdc).to(DEVICE)
    pd = pdc.to(DEVICE)
    rows = []
    for rho, name in ((0.0, "static_at10dB"), (0.9, "shared_rho0.9_at10dB")):
        for L in (64, 32, 16, 8, 4, 2, 1):
            for s in (1, 2, 3):
                m = train_rho(E, L, pd, rho, a.iters, seed=s, snr_range=(10.0, 10.0))
                v_val = ser(m, pd, rho, 10, pm.COND["flat"], seed=21)
                v = ser(m, pd, rho, 10, pm.COND["flat"])
                io = sum(pm.cross_energy(m.masks().detach().cpu())) / U
                rows.append([name, "flat", rho, L, s, v_val, v, io])
                print("[sig10]", name, L, s, round(v_val, 4), round(v, 4), "iota", round(io, 3), flush=True)
    write_csv(DATA / "pm_sig10.csv", ["scheme", "channel", "rho", "values", "seed", "ser_val", "ser", "cross_energy"], rows)


def part_sig10rho(a):
    """Static entries of Fig. 4 trained at the 10 dB they serve for the
    intermediate shared-token probabilities rho = 0.25, 0.5, 0.75 and 1, every
    pattern length 1..64, three training runs each, scored on validation
    frames (seed 21) and on test frames (seed 11), as part sig10 does for
    rho = 0 and 0.9 (data/pm/pm_sig10_rho.csv). A run whose signs cancel the
    frame of a shared token (c = 0, possible with one value per mask) is kept
    as a failed run with nan values."""
    src = pm.Source("bert")
    pdc = src.dist()
    E = pm.codebook(src.W, D, "pwhite", pd=pdc).to(DEVICE)
    pd = pdc.to(DEVICE)
    rows = []
    for rho in (0.25, 0.5, 0.75, 1.0):
        name = f"shared_rho{rho:g}_at10dB"
        for L in (64, 32, 16, 8, 4, 2, 1):
            for s in (1, 2, 3):
                try:
                    m = train_rho(E, L, pd, rho, a.iters, seed=s, snr_range=(10.0, 10.0))
                except ZeroDivisionError:
                    # with one value per mask the signs of the users can cancel on a
                    # shared token, so the frame carries no signal and c = 0; the run
                    # is kept as failed (nan) and never selected on validation frames
                    rows.append([name, "flat", rho, L, s, float("nan"), float("nan"), float("nan")])
                    print("[sig10rho]", name, L, s, "failed: signal cancelled (c = 0)", flush=True)
                    continue
                v_val = ser(m, pd, rho, 10, pm.COND["flat"], seed=21)
                v = ser(m, pd, rho, 10, pm.COND["flat"])
                io = sum(pm.cross_energy(m.masks().detach().cpu())) / U
                rows.append([name, "flat", rho, L, s, v_val, v, io])
                print("[sig10rho]", name, L, s, round(v_val, 4), round(v, 4), "iota", round(io, 3), flush=True)
    write_csv(DATA / "pm_sig10_rho.csv", ["scheme", "channel", "rho", "values", "seed", "ser_val", "ser", "cross_energy"], rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True)
    ap.add_argument("--iters", type=int, default=3000)
    a = ap.parse_args()
    t0 = time.time()
    {"corr": part_corr, "csit": part_csit, "adapt": part_adapt, "sic": part_sic,
     "rs": part_rs, "figs": part_figs, "sig": part_sig,
     "content_snr": part_content_snr, "sig10": part_sig10,
     "sig10rho": part_sig10rho}[a.part](a)
    print("[done]", a.part, round((time.time() - t0) / 60, 1), "min", flush=True)


if __name__ == "__main__":
    main()
