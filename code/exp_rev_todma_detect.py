"""Revision experiment R1.2: how far the ToDMA row depends on its detector,
and what causes its high-SNR error floor.

The transmitter, channel, and scoring are those of exp_sota.todma_ser
(Table I): the seed-12345 random Gaussian dictionary with V = 65,536 and
d = 64, unit frame energy, one Rayleigh gain per receiver known at that
receiver, sigma^2 = 1/(d*snr), and the genie-aided rule that counts a user
correct when its token is in the detected set. Only the detector changes:
  omp     exp_sota.omp_support (absolute correlation, least-squares refit)
  somp    the same with signed correlation (coefficients are known positive)
  treeB   tree search with known amplitude 1/sqrt(U): B partial token sets
          survive each pass and the final set has the least residual energy
          among the survivors. B -> inf is joint maximum likelihood.
MACs per detection count the correlations only: U*V*d for omp and somp,
(1 + (U-1)*B)*V*d for treeB.

Tests, in the "test" column of the output:
  detector  all detectors at 0, 6, 10, 16, 20 dB and without noise; treeB
            with B = 16 on the whole 0-20 dB grid of Table I, and B = 6, 8
            at 10 and 20 dB
  masked    no genie: the same dictionary with one random +/-1 mask per user,
            nothing trained, tree search over the U masked dictionaries with
            one token per user, B = 64. A user is correct only if its own
            token is decoded. MACs are U*(1 + (U-1)*B)*V*d as run; a search
            that skips users already decoded needs (U + B*U*(U-1)/2)*V*d.
  proposal  the trained model sse_P4.pt, per-user digit correlation (no genie)
            and exact joint minimum distance over the Vu^U hypotheses per
            period, on the same SNR points
  reject    noiseless OMP floor when the largest pairwise correlation of the
            dictionary is limited to tau: every column with an earlier
            partner above tau is replaced by a fresh draw until no pair
            exceeds tau. The token draws are common to all tau of a dictionary
            seed, so the differences are paired. Seeds other than 12345 give
            the spread among unrestricted dictionaries.
  floor_V   noiseless OMP floor against V at d = 64
  seed      tree B = 16 on two other dictionary seeds
  V30522    the first 30,522 columns (the BERT vocabulary size), uniform tokens
  real      token error rate on the AG News streams of exp_rev_semantic, with
            tokens drawn from the streams

Columns: ser is errors/decisions, with one decision per user per frame. se is
its standard error from the per-frame error counts, since the decisions of a
frame share its tokens. search_fail is the fraction of decisions whose
detected set differs from the true set and has more residual energy than the
true set: the detector missed a better hypothesis, so joint maximum likelihood
would not have returned that set. Without noise one detection serves all four
users, so that point runs U times the frames for the same work.

Frames per point are in the plan of main(). Fixed seeds: dictionary as in
exp_sota, evaluation seed 777 as in exp_sota, set-key table seed 99, masks
seed 4242.

Outputs: data/rev_todma_detect.csv and data/rev_todma_detect.json (the wrong
picks of noiseless OMP against the true atoms, and the pair counts of the
dictionary).
"""
from __future__ import annotations

import json
import math
import time

import torch

from sse_lib import (DATA, DEVICE, D, U, VU, P_MAX, SSE, rayleigh_gain,
                     snr_to_sigma2, set_seed, write_csv)
from exp_sota import make_dictionary, omp_support

V = VU ** P_MAX
AMP = 1.0 / math.sqrt(U)
SEED_EVAL = 777
SEED_NOISELESS = 100_000
SEED_MASKS = 4242
SEED_KEYS = 99
SNR_POINTS = [None, 0, 6, 10, 16, 20]          # None is the noiseless point
SNR_GRID = list(range(0, 21, 2))
TAUS = [None, 0.65, 0.6, 0.55, 0.5, 0.47, 0.46]
HEADER = ["test", "detector", "V", "dict_seed", "tau", "B", "snr_db", "ser",
          "se", "errors", "decisions", "frames", "search_fail", "macs",
          "coherence", "replaced"]


def sigma_of(snr_db) -> float:
    return math.sqrt(float(snr_to_sigma2(torch.tensor(float(snr_db)))))


def snr_label(snr_db) -> str:
    return "inf" if snr_db is None else str(snr_db)


# ----------------------------------------------------------------------
# detectors
# ----------------------------------------------------------------------
@torch.no_grad()
def signed_omp(r: torch.Tensor, A: torch.Tensor, k: int = U):
    """exp_sota.omp_support with signed correlation."""
    n = r.shape[0]
    support = torch.zeros(n, k, dtype=torch.long, device=r.device)
    R = r.clone()
    eye = torch.eye(k, device=r.device)
    for t in range(k):
        c = R @ A
        if t > 0:
            c.scatter_(1, support[:, :t], -1e9)
        support[:, t] = c.argmax(dim=1)
        As = A.T[support[:, :t + 1]]
        G = torch.einsum("nkd,njd->nkj", As, As) + 1e-6 * eye[:t + 1, :t + 1]
        b = (As * r[:, None, :]).sum(dim=2, keepdim=True)
        beta = torch.linalg.solve(G, b)
        R = r - (As * beta).sum(dim=1)
    return support


_KEYS: dict[int, torch.Tensor] = {}


def set_keys(size: int) -> torch.Tensor:
    """Random 62-bit value per atom. The sum over a set is the set's key, free
    of order; two different sets share a key with probability about 2^-62."""
    if size not in _KEYS:
        g = torch.Generator(device="cpu").manual_seed(SEED_KEYS)
        _KEYS[size] = torch.randint(1, 2 ** 62, (size,), generator=g).to(DEVICE)
    return _KEYS[size]


@torch.no_grad()
def tree_search(r: torch.Tensor, A: torch.Tensor, B: int,
                masks: torch.Tensor | None = None):
    """Known-amplitude tree search, r: (n, d), A: (d, V).

    masks None: U passes over one dictionary, returns the token set (n, U).
    masks (U, d): U passes over the masked dictionaries with one atom per
    user, returns the token of each user (n, U).
    Atoms are coded mask*V + token; each survivor's metric is its residual
    energy, so a child's metric follows from its parent's correlation."""
    n, d = r.shape
    Vc = A.shape[1]
    pools = 1 if masks is None else U
    keys = set_keys(pools * Vc)
    R = r[:, None, :]                                    # (n, b, d)
    metric = r.pow(2).sum(-1, keepdim=True)              # (n, b)
    atoms = torch.zeros(n, 1, 0, dtype=torch.long, device=r.device)
    key = torch.zeros(n, 1, dtype=torch.long, device=r.device)
    for t in range(U):
        b = R.shape[1]
        vals, idxs = [], []
        for v in range(pools):
            Rv = R if masks is None else R * masks[v]
            c = (Rv.reshape(-1, d) @ A).reshape(n, b, Vc)
            val, ix = c.topk(B, dim=2)                   # (n, b, B)
            del c
            if masks is not None and t > 0:              # one atom per user
                taken = (atoms // Vc == v).any(dim=2, keepdim=True)
                val = val.masked_fill(taken, -1e9)
            vals.append(val)
            idxs.append(ix + v * Vc)
        val = torch.cat(vals, dim=2)                     # (n, b, pools*B)
        idx = torch.cat(idxs, dim=2)
        width = pools * B
        m = (metric[:, :, None] - 2 * AMP * val + AMP ** 2).reshape(n, -1)
        k = (key[:, :, None] + keys[idx]).reshape(n, -1)
        # orderings of one set share a key: keep one of them
        ks, order = k.sort(dim=1)
        dup = torch.zeros_like(ks, dtype=torch.bool)
        dup[:, 1:] = ks[:, 1:] == ks[:, :-1]
        m = m + torch.zeros_like(m).scatter(1, order, dup.float() * 1e9)
        metric, pos = m.topk(min(B, m.shape[1]), dim=1, largest=False)
        parent = pos // width
        new = idx.reshape(n, -1).gather(1, pos)
        key = k.gather(1, pos)
        atoms = torch.cat([atoms.gather(1, parent[:, :, None].expand(-1, -1, t)),
                           new[:, :, None]], dim=2)
        tok, pool = new % Vc, new // Vc
        step = A.T[tok] if masks is None else A.T[tok] * masks[pool]
        R = R.gather(1, parent[:, :, None].expand(-1, -1, d)) - AMP * step
    best = atoms[:, 0, :]                                # metric is ascending
    if masks is None:
        return best
    return torch.zeros_like(best).scatter(1, best // Vc, best % Vc)


# ----------------------------------------------------------------------
# Monte Carlo
# ----------------------------------------------------------------------
@torch.no_grad()
def superpose(tok: torch.Tensor, A: torch.Tensor,
              masks: torch.Tensor | None = None):
    """Unit-energy frame for tokens (n, U)."""
    c = A.T[tok]                                         # (n, U, d)
    if masks is not None:
        c = c * masks[None]
    return c.sum(dim=1) * AMP


@torch.no_grad()
def run_point(det, A: torch.Tensor, snr_db, frames: int, chunk: int, seed: int,
              masks: torch.Tensor | None = None,
              ids: torch.Tensor | None = None) -> dict:
    """SER of det(r) -> (n, U). The genie rule applies unless masks are given."""
    torch.manual_seed(seed)
    Vc = A.shape[1]
    noiseless = snr_db is None
    if noiseless:
        frames *= U
    sigma = 0.0 if noiseless else sigma_of(snr_db)
    e1 = e2 = fail = done = 0

    def wrong(S, tok, u):
        if masks is not None:
            return S[:, u] != tok[:, u]
        return ~(S == tok[:, u:u + 1]).any(dim=1)

    def missed(r, S, tok):
        """Detected set is not the true set, and is the worse hypothesis."""
        res_S = (r - superpose(S, A, masks)).pow(2).sum(1)
        res_t = (r - superpose(tok, A, masks)).pow(2).sum(1)
        if masks is None:
            differs = (S.sort(1).values != tok.sort(1).values).any(1)
        else:
            differs = (S != tok).any(1)
        return int((differs & (res_t < res_S)).sum())

    while done < frames:
        n = min(chunk, frames - done)
        if ids is None:
            tok = torch.randint(Vc, (n, U), device=DEVICE)
        else:
            tok = ids[torch.randint(ids.shape[0], (n,), device=DEVICE)]
        x = superpose(tok, A, masks)
        e_f = torch.zeros(n, device=DEVICE)
        if noiseless:
            S = det(x)
            for u in range(U):
                e_f += wrong(S, tok, u).float()
            fail += U * missed(x, S, tok)
        else:
            h = rayleigh_gain((n, U))
            for u in range(U):
                gain = h[:, u:u + 1]
                y = gain * x + sigma * torch.randn(n, D, device=DEVICE)
                r = y / gain.clamp_min(1e-6)
                S = det(r)
                e_f += wrong(S, tok, u).float()
                fail += missed(r, S, tok)
        e1 += float(e_f.sum())
        e2 += float(e_f.pow(2).sum())
        done += n
    var = max(e2 / done - (e1 / done) ** 2, 0.0)
    return {"ser": e1 / (U * done), "se": math.sqrt(var / done) / U,
            "errors": int(e1), "decisions": U * done, "frames": done,
            "search_fail": fail / (U * done)}


@torch.no_grad()
def proposal_point(model: SSE, snr_db, frames: int, chunk: int, seed: int):
    """Per-user digit correlation and exact joint minimum distance on the same
    frames. The joint metric decouples over the P periods."""
    torch.manual_seed(seed)
    P, L, vu = model.P, model.L, model.vu
    Bn, mk = model.unit_codebook(), model.masks()
    cand = Bn[None] * mk[:, None]                        # (U, Vu, L)
    grid = torch.cartesian_prod(*[torch.arange(vu, device=DEVICE)] * U)
    hyp = sum(cand[u][grid[:, u]] for u in range(U)) / math.sqrt(P) / model.c
    hyp_e = hyp.pow(2).sum(1)
    sigma = 0.0 if snr_db is None else sigma_of(snr_db)
    e_cor = e_cor2 = e_jnt = e_jnt2 = done = 0
    while done < frames:
        n = min(chunk, frames - done)
        dig = torch.randint(vu, (n, U, P), device=DEVICE)
        y = model.tx_frame(dig) / model.c
        h = rayleigh_gain((n, U))
        f_cor = torch.zeros(n, device=DEVICE)
        f_jnt = torch.zeros(n, device=DEVICE)
        for u in range(U):
            gain = h[:, u, None, None]
            r = y + (sigma / gain.clamp_min(1e-6)) * torch.randn(n, P, L, device=DEVICE)
            d0 = torch.einsum("npl,vl->npv", r, cand[u]).argmax(2)
            f_cor += (d0 != dig[:, u]).any(1).float()
            met = hyp_e - 2 * r.reshape(-1, L) @ hyp.T
            d1 = grid[met.argmin(1), u].reshape(n, P)
            f_jnt += (d1 != dig[:, u]).any(1).float()
        e_cor += float(f_cor.sum()); e_cor2 += float(f_cor.pow(2).sum())
        e_jnt += float(f_jnt.sum()); e_jnt2 += float(f_jnt.pow(2).sum())
        done += n

    def res(e, e_sq):
        var = max(e_sq / done - (e / done) ** 2, 0.0)
        return {"ser": e / (U * done), "se": math.sqrt(var / done) / U,
                "errors": int(e), "decisions": U * done, "frames": done,
                "search_fail": ""}
    return res(e_cor, e_cor2), res(e_jnt, e_jnt2)


# ----------------------------------------------------------------------
# what causes the floor
# ----------------------------------------------------------------------
@torch.no_grad()
def pair_stats(A: torch.Tensor, thresholds=(0.5, 0.6, 0.7), chunk: int = 4096):
    """Largest |rho| over all pairs and the number of pairs above each level."""
    Vc = A.shape[1]
    mx, cnt = 0.0, [0] * len(thresholds)
    for i in range(0, Vc, chunk):
        G = (A.T[i:i + chunk] @ A).abs()
        G[torch.arange(G.shape[0]), torch.arange(i, i + G.shape[0])] = 0
        mx = max(mx, float(G.max()))
        for k, t in enumerate(thresholds):
            cnt[k] += int((G > t).sum())
    return mx, {str(t): c // 2 for t, c in zip(thresholds, cnt)}


@torch.no_grad()
def limit_pairs(A: torch.Tensor, tau: float, seed: int, batch: int = 8192,
                chunk: int = 4096, max_draws: int = 10 ** 9):
    """Replace every column with an earlier partner above tau by a fresh unit
    draw that respects tau against all kept columns. Columns keep their index,
    so a token keeps its codeword unless that codeword was replaced.
    Returns (dictionary, replaced columns, draws)."""
    Vc = A.shape[1]
    A = A.clone()
    keep = torch.ones(Vc, dtype=torch.bool, device=DEVICE)
    for i in range(0, Vc, chunk):
        G = (A.T[i:i + chunk] @ A[:, :i + chunk]).abs()
        G[torch.arange(G.shape[0]), torch.arange(i, i + G.shape[0])] = 0
        G = G * (torch.arange(i + G.shape[0], device=DEVICE)[None]
                 < torch.arange(i, i + G.shape[0], device=DEVICE)[:, None])
        keep[i:i + chunk] = ~(G > tau).any(dim=1)
    todo = (~keep).nonzero()[:, 0]
    replaced = int(todo.shape[0])
    g = torch.Generator(device="cpu").manual_seed(seed)
    draws = 0
    while todo.shape[0] > 0:
        if draws >= max_draws:
            raise RuntimeError(f"tau={tau}: {todo.shape[0]} columns left "
                               f"after {draws} draws")
        C = torch.randn(D, batch, generator=g).to(DEVICE)
        C = C / C.norm(dim=0, keepdim=True)
        draws += batch
        kept = A[:, keep]
        ok = torch.ones(batch, dtype=torch.bool, device=DEVICE)
        for i in range(0, kept.shape[1], 16384):
            ok &= ~((C.T @ kept[:, i:i + 16384]).abs() > tau).any(dim=1)
        C = C[:, ok]
        if C.shape[1] > 1:
            G = (C.T @ C).abs().triu(1)
            C = C[:, ~(G > tau).any(dim=0)]
        m = min(C.shape[1], todo.shape[0])
        A[:, todo[:m]] = C[:, :m]
        keep[todo[:m]] = True
        todo = todo[m:]
    return A, replaced, draws


@torch.no_grad()
def wrong_pick_stats(A: torch.Tensor, frames: int = 200_000, chunk: int = 2048,
                     seed: int = 3) -> dict:
    """Noiseless OMP: in each failed frame, the first wrong pick against the
    four true atoms, and the correlation among the true atoms themselves."""
    torch.manual_seed(seed)
    mx_all, sum_all, sum_p1, pass_hist = [], [], [], torch.zeros(U).long()
    mp_fail, mp_ok = [], []
    n_fail = done = 0
    iu = torch.triu_indices(U, U, 1)
    while done < frames:
        n = min(chunk, frames - done)
        tok = torch.randint(A.shape[1], (n, U), device=DEVICE)
        sup = omp_support(superpose(tok, A), A)
        pick_wrong = (sup[:, :, None] != tok[:, None, :]).all(dim=2)  # (n, U)
        fail = pick_wrong.any(dim=1)
        first = pick_wrong.float().argmax(dim=1)
        Gt = torch.einsum("nkd,njd->nkj", A.T[tok], A.T[tok])
        mp = Gt[:, iu[0], iu[1]].abs().max(dim=1).values
        mp_fail.append(mp[fail].cpu()); mp_ok.append(mp[~fail].cpu())
        ix = fail.nonzero()[:, 0]
        if ix.shape[0]:
            w = sup[ix, first[ix]]
            rho = torch.einsum("nd,nkd->nk", A.T[w], A.T[tok[ix]])
            mx_all.append(rho.abs().max(dim=1).values.cpu())
            sum_all.append(rho.sum(dim=1).abs().cpu())
            sum_p1.append(rho.sum(dim=1).abs()[first[ix] == 0].cpu())
            pass_hist += torch.bincount(first[ix].cpu(), minlength=U)
            n_fail += int(ix.shape[0])
        done += n
    mx, sm, s1 = torch.cat(mx_all), torch.cat(sum_all), torch.cat(sum_p1)
    mpf, mpo = torch.cat(mp_fail), torch.cat(mp_ok)
    q = lambda t, p: float(t.quantile(p))
    return {"frames": done, "failed_frames": n_fail,
            "first_wrong_pass_hist": pass_hist.tolist(),
            "maxabsrho_wrong_vs_true_q10_q50_q90": [q(mx, .1), q(mx, .5), q(mx, .9)],
            "frac_maxabsrho_above_0.5": float((mx > .5).float().mean()),
            "frac_maxabsrho_above_0.6": float((mx > .6).float().mean()),
            "frac_maxabsrho_above_0.7": float((mx > .7).float().mean()),
            "abs_sumrho_wrong_vs_true_q10_q50_q90": [q(sm, .1), q(sm, .5), q(sm, .9)],
            "abs_sumrho_pass1_q10_q50_q90": [q(s1, .1), q(s1, .5), q(s1, .9)],
            "maxabsrho_among_true_atoms_q50_failed": q(mpf, .5),
            "maxabsrho_among_true_atoms_q50_ok": q(mpo, .5)}


# ----------------------------------------------------------------------
def main():
    set_seed(SEED_EVAL)
    t0 = time.time()
    A = make_dictionary(D, V)                            # Table I, seed 12345
    Vd = V * D
    rows = []

    def add(test, det, res, snr_db, Vc=V, dict_seed=12345, tau="", B="",
            macs="", coherence="", replaced=""):
        rows.append([test, det, Vc, dict_seed, tau, B, snr_label(snr_db),
                     res["ser"], res["se"], res["errors"], res["decisions"],
                     res["frames"], res["search_fail"], macs, coherence,
                     replaced])
        write_csv(DATA / "rev_todma_detect.csv", HEADER, rows)
        print(f"[{time.time() - t0:7.0f}s] {test} {det} V={Vc} seed={dict_seed} "
              f"tau={tau} B={B} snr={snr_label(snr_db)}: ser={res['ser']:.5f} "
              f"se={res['se']:.5f} err={res['errors']} dec={res['decisions']} "
              f"fail={res['search_fail']}", flush=True)

    def seed_of(snr_db):
        return SEED_EVAL + (SEED_NOISELESS if snr_db is None else int(10 * snr_db))

    # ---------------- the detector on the Table I dictionary --------------
    tree_macs = lambda B: (1 + (U - 1) * B) * Vd
    tree = lambda B: (lambda r, Ax: tree_search(r, Ax, B))
    plan = [("omp", omp_support, "", U * Vd, 1_000_000, 2048, SNR_POINTS),
            ("somp", signed_omp, "", U * Vd, 1_000_000, 2048, SNR_POINTS),
            ("tree1", tree(1), 1, tree_macs(1), 250_000, 1024, SNR_POINTS),
            ("tree4", tree(4), 4, tree_macs(4), 250_000, 512, SNR_POINTS),
            ("tree6", tree(6), 6, tree_macs(6), 100_000, 256, [10, 20]),
            ("tree8", tree(8), 8, tree_macs(8), 100_000, 256, [10, 20]),
            ("tree16", tree(16), 16, tree_macs(16), 250_000, 128, SNR_POINTS),
            ("tree16", tree(16), 16, tree_macs(16), 100_000, 128,
             [s for s in SNR_GRID if s not in SNR_POINTS]),
            ("tree64", tree(64), 64, tree_macs(64), 50_000, 32, SNR_POINTS)]
    for name, det, B, macs, frames, chunk, points in plan:
        for s in points:
            res = run_point(lambda r: det(r, A), A, s, frames, chunk, seed_of(s))
            add("detector", name, res, s, B=B, macs=macs)

    # ---------------- no genie: untrained masked dictionary ---------------
    g = torch.Generator(device="cpu").manual_seed(SEED_MASKS)
    masks = (torch.randint(0, 2, (U, D), generator=g).float() * 2 - 1).to(DEVICE)
    for s in SNR_POINTS:
        res = run_point(lambda r: tree_search(r, A, 64, masks), A, s, 25_000, 32,
                        seed_of(s), masks=masks)
        add("masked", "mask_tree64", res, s, B=64, macs=U * tree_macs(64))

    # ---------------- the proposal on the same SNR points -----------------
    sse = SSE(P=P_MAX)
    sse.load_state_dict(torch.load(DATA / "sse_P4.pt", map_location="cpu"))
    sse.to(DEVICE).eval()
    for s in [None] + SNR_GRID:
        cor, jnt = proposal_point(sse, s, 1_000_000, 512, seed_of(s))
        add("proposal", "sse_peruser", cor, s, macs=P_MAX * VU * sse.L)
        add("proposal", "sse_jointmd", jnt, s, macs=P_MAX * (VU ** U) * sse.L)

    # ---------------- cause of the floor ----------------------------------
    coh, pairs = pair_stats(A)
    stats = {"dictionary_seed": 12345, "coherence": coh, "pairs_above": pairs,
             "total_pairs": V * (V - 1) // 2,
             "noiseless_omp_wrong_picks": wrong_pick_stats(A)}
    (DATA / "rev_todma_detect.json").write_text(json.dumps(stats, indent=1))
    print("[cause]", json.dumps(stats), flush=True)

    for Vc in (256, 1024, 4096, 16384, 65536):
        Ax = make_dictionary(D, Vc)
        res = run_point(lambda r: omp_support(r, Ax), Ax, None, 250_000, 2048,
                        seed_of(None))
        add("floor_V", "omp", res, None, Vc=Vc, macs=U * Vc * D,
            coherence=pair_stats(Ax)[0])

    for ds in (12345, 1, 2, 3):
        base = make_dictionary(D, V, seed=ds)
        for k, tau in enumerate(TAUS):
            if tau is None:
                Ak, rep = base, 0
            else:
                Ak, rep, _ = limit_pairs(base, tau, seed=1000 * ds + k)
            res = run_point(lambda r: omp_support(r, Ak), Ak, None, 250_000, 2048,
                            seed_of(None))
            add("reject", "omp", res, None, dict_seed=ds,
                tau="none" if tau is None else tau, macs=U * Vd,
                coherence=pair_stats(Ak)[0], replaced=rep)

    # ---------------- other dictionaries and the real streams --------------
    for ds in (1, 2):
        Ax = make_dictionary(D, V, seed=ds)
        for s in (10, 20):
            res = run_point(lambda r: tree_search(r, Ax, 16), Ax, s, 100_000, 128,
                            seed_of(s))
            add("seed", "tree16", res, s, dict_seed=ds, B=16, macs=tree_macs(16))

    Ab = A[:, :30522]
    for s in (10, 20):
        res = run_point(lambda r: omp_support(r, Ab), Ab, s, 1_000_000, 2048,
                        seed_of(s))
        add("V30522", "omp", res, s, Vc=30522, macs=U * 30522 * D)
        res = run_point(lambda r: tree_search(r, Ab, 16), Ab, s, 250_000, 128,
                        seed_of(s))
        add("V30522", "tree16", res, s, Vc=30522, B=16,
            macs=(1 + (U - 1) * 16) * 30522 * D)

    from exp_rev_semantic import load_streams
    _, ids_all, _ = load_streams()
    ids_all = ids_all.to(DEVICE)
    for s in (10, 20):
        res = run_point(lambda r: omp_support(r, A), A, s, 1_000_000, 2048,
                        seed_of(s), ids=ids_all)
        add("real", "omp", res, s, macs=U * Vd)
        res = run_point(lambda r: tree_search(r, A, 16), A, s, 250_000, 128,
                        seed_of(s), ids=ids_all)
        add("real", "tree16", res, s, B=16, macs=tree_macs(16))


if __name__ == "__main__":
    main()
