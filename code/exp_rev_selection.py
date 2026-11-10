"""Revision: the reported model on fresh evaluation seeds, and the best-of-six
procedure with selection and reporting kept apart.

Question. The reported model (sse_P4.pt) is the training-seed-1 run of the
original submission. During the revision it ranked first of six 4,000-iteration
runs (with rev_sse_seed1..5.pt, trained afterwards on a CPU by exp_rev_seeds.py).
Its Fig. 2 and Table I SERs and that ranking come from one evaluation protocol,
eval_ser_sse of sse_lib.py: digits from a CPU generator seeded 777 + int(10*snr),
gains and noise from the global generator of the device (CUDA for exp_main.py
after set_seed(1), 2e6 frames; CPU for exp_rev_seeds.py after training, 5e5
frames). The objection is that the ranking shares the test evaluation. This script re-evaluates all
runs on seeds that no script of the project has used.

Setup. U = 4, d = 64, Vu = 16, P = 4, V = 65,536, flat Rayleigh fading with
known gains, real-vector convention of sse_lib.py. Sixteen 4,000-iteration
runs of (Vu,P) = (16,4):
  reported     sse_P4.pt
  six_other    rev_sse_seed1..5.pt          (the other five of the six)
  ten_further  rev_factor_16_4_s1..s10.pt   (exp_rev_factor_seeds.py)
Decoder. The correlation rule of SSE.forward (Fig. 2), reproduced here with
the gains and noise passed in. A check against SSE.forward runs first.
OMA. One bit per real dimension for the 16 index bits on d/U = 16 real dimensions per
user at unit frame energy (amplitude 1/sqrt(d) per dimension), the scheme of
oma_ser() behind Table I and Fig. 2, simulated at the dimension level. The
closed form oma_ser() is reported next to it.

Common random numbers. In every frame all schemes share the indices (OMA sends
the 16 bits of the same four digits), the gain h_u of each user, and the noise
vector of each user's receiver.
  OMA_contig   user u on the 16 dimensions of period u.
  OMA_aligned  the four bits of digit p of user u on the four positions of
               period p where the reported model's mask of user u is largest.
               For sse_P4.pt these sets cover each position of the period once (checked). Same
               SER law, tighter coupling with the reported model only.

Seed sets. Two fresh sets, A and B, each with --frames frames per SNR in chunks
of CHUNK. Chunk k at SNR index i of set S uses a CUDA generator seeded
BASE[S] + 100_000*i + k, BASE = {A: 31_000_000, B: 47_000_000}. Set A serves
the selection and set B the report in the best-of-six study.

Intervals. The four users of a frame share its indices, so every standard error
takes the frame as the unit (per-frame error counts 0..4). The paired ratio of
two schemes uses the delta method with the covariance of these counts.

Best of six. For every six-run subset of the sixteen runs (all 8,008, without
replacement) and for 2e5 draws of six with replacement, the run with the lowest
10 dB SER on set A is selected, and its set-B SER at 10 and 20 dB is compared
with OMA_contig on set B. margin = SER_selected / SER_OMA - 1 (negative means
below OMA). An outer bootstrap over the sixteen runs (iid draws of six) gives an
interval for the probability that the selected run lies below OMA. The reverse
assignment (select on B, report on A) is a check.

Old-seed rows (seed_set old777). eval_ser_sse of sse_lib after set_seed(1),
2e6 frames, the Fig. 2 protocol, for all sixteen runs.

Outputs (new files only):
  data/rev_selection.csv          one row per model, seed set, and SNR
  data/rev_selection_summary.csv  long format: section, seed_set, snr_db,
                                  quantity, value, lo, hi
Usage: CUDA_VISIBLE_DEVICES=1 python exp_rev_selection.py [--frames N]
"""
from __future__ import annotations

import argparse
import itertools
import math
import time

import numpy as np
import torch

from sse_lib import (D, DATA, DEVICE, P_MAX, SSE, U, VU, eval_ser_sse, oma_ser,
                     set_seed, snr_to_sigma2, write_csv)

SNRS = (0.0, 4.0, 10.0, 16.0, 20.0)
L = D // P_MAX
BASE = {"A": 31_000_000, "B": 47_000_000}
SEED_CHECK = 71_000_000
SEED_BOOT = 59_000_000
CHUNK = 200_000
Z = 1.959963984540054

RUNS = ([("sse_P4", "reported", "sse_P4.pt")]
        + [(f"seed{s}", "six_other", f"rev_sse_seed{s}.pt") for s in range(1, 6)]
        + [(f"factor_s{s}", "ten_further", f"rev_factor_16_4_s{s}.pt")
           for s in range(1, 11)])
OMAS = ("OMA_contig", "OMA_aligned")
NAMES = [r[0] for r in RUNS] + list(OMAS)
GROUPS = [r[1] for r in RUNS] + ["oma", "oma"]
FILES = [r[2] for r in RUNS] + ["-", "-"]
NR = len(RUNS)
IC, IA = NR, NR + 1                      # indices of the two OMA columns
OUT = DATA / "rev_selection.csv"
OUT_SUM = DATA / "rev_selection_summary.csv"


def load(name: str) -> SSE:
    m = SSE(P=P_MAX, vu=VU, users=U)
    m.load_state_dict(torch.load(DATA / name, map_location="cpu"))
    return m.to(DEVICE).eval()


def sigma_of(snr_db: float) -> torch.Tensor:
    return snr_to_sigma2(float(snr_db)).to(DEVICE).sqrt()


@torch.no_grad()
def sse_wrong(m: SSE, digits, h, noise, sigma):
    """Correlation rule of SSE.forward with given gains and noise -> (n,U) bool."""
    y = m.tx_frame(digits) / m.c                                 # (n,P,L)
    y_rx = h[:, :, None, None] * y[:, None] + sigma * noise      # (n,U,P,L)
    r = y_rx / h[:, :, None, None].clamp_min(1e-6)
    cand = m.unit_codebook()[None, :, :] * m.masks()[:, None, :]  # (U,Vu,L)
    s = torch.einsum("nupl,uvl->nupv", r, cand)
    return (s.argmax(-1) != digits).any(dim=2)


def antipodal_bits(digits):
    """(n,U,P) digits in [0,16) -> (n,U,4P) bits in {-1,+1}, digit-major."""
    k = torch.arange(3, -1, -1, device=digits.device)
    b = ((digits[..., None] >> k) & 1).float() * 2.0 - 1.0       # (n,U,P,4)
    return b.reshape(digits.shape[0], digits.shape[1], -1)


@torch.no_grad()
def oma_wrong(b, h, noise_sel, sigma):
    """OMA at unit frame energy. b, noise_sel (n,U,16) -> (n,U) bool."""
    y = h[:, :, None] * b / math.sqrt(D) + sigma * noise_sel
    return ((y * b) < 0).any(dim=2)


def draw(gen_seed: int, n: int):
    g = torch.Generator(device=DEVICE)
    g.manual_seed(gen_seed)
    digits = torch.randint(VU, (n, U, P_MAX), generator=g, device=DEVICE)
    u = torch.rand((n, U), generator=g, device=DEVICE).clamp_min(1e-12)
    h = torch.sqrt(-torch.log(u))                                # h^2 ~ Exp(1)
    noise = torch.randn((n, U, P_MAX, L), generator=g, device=DEVICE)
    return digits, h, noise


@torch.no_grad()
def check_forward(m: SSE, n: int = 100_000, snr_db: float = 10.0) -> int:
    """Decisions of sse_wrong equal those of SSE.forward on the same draws."""
    digits, h, _ = draw(SEED_CHECK, n)
    torch.manual_seed(SEED_CHECK + 1)
    noise = torch.randn(n, U, P_MAX, L, device=DEVICE)
    torch.manual_seed(SEED_CHECK + 1)
    s = m(digits, torch.full((n,), snr_db), h=h)                 # same randn inside
    ref = (s.argmax(-1) != digits).any(dim=2)
    mine = sse_wrong(m, digits, h, noise, sigma_of(snr_db))
    return int((ref != mine).sum())


@torch.no_grad()
def aligned_support(m: SSE):
    """Per user, the L/U positions of the largest |mask| entries (sorted)."""
    k = L // U
    mu = m.masks().abs()
    supp = torch.sort(torch.topk(mu, k, dim=1).indices, dim=1).values   # (U,k)
    part = bool(torch.equal(torch.sort(supp.reshape(-1)).values,
                            torch.arange(L, device=DEVICE)))
    frac = float(((mu ** 2).gather(1, supp).sum(1) / (mu ** 2).sum(1)).min())
    return supp, part, frac


@torch.no_grad()
def accumulate(models, supp, seed_base: int, isnr: int, snr_db: float, frames: int):
    """Sums over frames of the per-frame error counts and their products."""
    M = NR + 2
    S1 = torch.zeros(M, dtype=torch.float64, device=DEVICE)
    S2 = torch.zeros(M, M, dtype=torch.float64, device=DEVICE)
    PU = torch.zeros(M, U, dtype=torch.float64, device=DEVICE)
    sigma = sigma_of(snr_db)
    ar = torch.arange(U, device=DEVICE)
    k = 0
    for n0 in range(0, frames, CHUNK):
        n = min(CHUNK, frames - n0)
        digits, h, noise = draw(seed_base + 100_000 * isnr + k, n)
        k += 1
        W = [sse_wrong(m, digits, h, noise, sigma) for m in models]
        b = antipodal_bits(digits)                               # (n,U,16)
        W.append(oma_wrong(b, h, noise[:, ar, ar, :], sigma))    # period u of user u
        na = torch.stack([noise[:, u][:, :, supp[u]].reshape(n, -1)
                          for u in range(U)], dim=1)             # (n,U,16)
        W.append(oma_wrong(b, h, na, sigma))
        W = torch.stack(W, 0)                                    # (M,n,U)
        a = W.sum(2).double()                                    # (M,n)
        S1 += a.sum(1)
        S2 += a @ a.T
        PU += W.sum(1).double()
    return S1.cpu().numpy(), S2.cpu().numpy(), PU.cpu().numpy()


class Point:
    """Frame-level moments of one seed set (or a pooled pair) at one SNR."""

    def __init__(self, S1, S2, PU, N):
        self.S1, self.S2, self.PU, self.N = S1, S2, PU, N
        self.mean = S1 / N
        self.cov = S2 / N - np.outer(self.mean, self.mean)

    def __add__(self, o):
        return Point(self.S1 + o.S1, self.S2 + o.S2, self.PU + o.PU, self.N + o.N)

    def ser(self, i):
        return self.mean[i] / U

    def se(self, i):
        return math.sqrt(max(self.cov[i, i], 0.0) / self.N) / U

    def worst(self, i):
        return float((self.PU[i] / self.N).max())

    def ratio(self, i, j):
        """Paired ratio SER_i / SER_j with a delta-method 95 percent interval."""
        R = self.mean[i] / self.mean[j]
        v = (self.cov[i, i] - 2 * R * self.cov[i, j] + R * R * self.cov[j, j]) \
            / (self.N * self.mean[j] ** 2)
        s = math.sqrt(max(v, 0.0))
        return R, R - Z * s, R + Z * s


def p_iid_best(sel_vals, below):
    """Six iid draws from the runs, best by sel_vals: P(selected run is below)."""
    n = len(sel_vals)
    order = np.argsort(sel_vals, kind="stable")
    j = np.arange(n)
    pj = ((n - j) / n) ** 6 - ((n - j - 1) / n) ** 6            # P(min rank = j)
    return float((pj * below[order]).sum())


def best_of_six(sel10, rep, oma_rep, ci_hi, rng, n_iid=200_000, n_boot=10_000):
    """sel10 (16,) selection SER, rep {snr: (16,)} report SER, oma_rep {snr: float},
    ci_hi {snr: (16,)} upper end of each run's paired ratio interval."""
    out = []
    n = len(sel10)
    subsets = np.array(list(itertools.combinations(range(n), 6)))         # (8008,6)
    sel_exact = subsets[np.arange(len(subsets)), np.argmin(sel10[subsets], axis=1)]
    draws = rng.integers(0, n, size=(n_iid, 6))
    sel_iid = draws[np.arange(n_iid), np.argmin(sel10[draws], axis=1)]
    for snr in sorted(rep):
        margin = rep[snr] / oma_rep[snr] - 1.0
        below = (margin < 0).astype(float)
        sig = (ci_hi[snr] < 1.0).astype(float)
        for tag, sel in (("exact", sel_exact), ("iid", sel_iid)):
            mg = margin[sel]
            out += [(tag, snr, "p_below_oma", float(below[sel].mean())),
                    (tag, snr, "p_below_oma_ci_excludes_1", float(sig[sel].mean())),
                    (tag, snr, "margin_median", float(np.median(mg))),
                    (tag, snr, "margin_p10", float(np.percentile(mg, 10))),
                    (tag, snr, "margin_p90", float(np.percentile(mg, 90))),
                    (tag, snr, "margin_mean", float(mg.mean())),
                    (tag, snr, "margin_worst", float(mg.max()))]
        boot = []
        for _ in range(n_boot):
            idx = rng.integers(0, n, size=n)
            boot.append(p_iid_best(sel10[idx], below[idx]))
        out.append(("iid_boot", snr, "p_below_oma_point", p_iid_best(sel10, below),
                    float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))))
        out += [("single_run", snr, "p_below_oma", float(below.mean())),
                ("single_run", snr, "margin_median", float(np.median(margin))),
                ("single_run", snr, "margin_p10", float(np.percentile(margin, 10))),
                ("single_run", snr, "margin_p90", float(np.percentile(margin, 90)))]
    return out, sel_exact


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=10_000_000)
    ap.add_argument("--old-frames", type=int, default=2_000_000)
    ap.add_argument("--outdir", default=str(DATA), help="smoke tests write elsewhere")
    a = ap.parse_args()
    from pathlib import Path
    out = Path(a.outdir) / OUT.name
    out_sum = Path(a.outdir) / OUT_SUM.name
    assert not out.exists() and not out_sum.exists(), "outputs exist, not overwriting"
    t0 = time.time()
    print("[device]", DEVICE, torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "")
    models = [load(f) for _, _, f in RUNS]
    for (name, _, _), m in zip(RUNS, models):
        bad = check_forward(m)
        print(f"[check] {name}: decisions differing from SSE.forward = {bad}", flush=True)
        assert bad == 0
    supp, part, frac = aligned_support(models[0])
    print(f"[aligned] support {supp.tolist()} covers_period={part} min energy frac={frac:.4f}")
    assert part, "aligned OMA needs supports that cover the period"

    pts = {}
    for S in ("A", "B"):
        for i, snr in enumerate(SNRS):
            S1, S2, PU = accumulate(models, supp, BASE[S], i, snr, a.frames)
            pts[(S, snr)] = Point(S1, S2, PU, a.frames)
            p = pts[(S, snr)]
            print(f"[{S} {snr:4.0f} dB] rep {p.ser(0):.5f} omaC {p.ser(IC):.5f} "
                  f"omaA {p.ser(IA):.5f} ratio {p.ratio(0, IC)} t={time.time()-t0:.0f}s",
                  flush=True)
    for snr in SNRS:
        pts[("AB", snr)] = pts[("A", snr)] + pts[("B", snr)]
    oma_cf = dict(zip(SNRS, oma_ser(list(SNRS))))

    # old protocol of Fig. 2: eval_ser_sse at seed 777 after set_seed(1)
    old = {}
    for j, m in enumerate(models):
        set_seed(1)
        old[j] = eval_ser_sse(m, list(SNRS), frames=a.old_frames)
        print("[old777]", NAMES[j], [f"{v:.5f}" for v in old[j]], flush=True)

    # ------------------------------------------------------------------
    # per-model rows
    # ------------------------------------------------------------------
    rows = []
    for S in ("A", "B", "AB"):
        for snr in SNRS:
            p = pts[(S, snr)]
            for i in range(NR + 2):
                R, lo, hi = p.ratio(i, IC)
                rows.append((NAMES[i], GROUPS[i], FILES[i], S, snr, p.N, p.ser(i),
                             p.se(i), p.se(i) / p.ser(i), p.worst(i), R, lo, hi))
    for j in range(NR):
        for k, snr in enumerate(SNRS):
            v = old[j][k]
            se = math.sqrt(v * (1 - v) / (U * a.old_frames))
            rows.append((NAMES[j], GROUPS[j], FILES[j], "old777", snr, a.old_frames, v,
                         se, se / v, float("nan"), v / oma_cf[snr], float("nan"),
                         float("nan")))
    for snr in SNRS:
        rows.append(("OMA_closed_form", "oma", "-", "exact", snr, 0, oma_cf[snr],
                     0.0, 0.0, oma_cf[snr], 1.0, 1.0, 1.0))
    write_csv(out, ["model", "group", "file", "seed_set", "snr_db", "frames", "ser",
                    "ser_se", "rel_se", "ser_worst_user", "ratio_oma_contig",
                    "ratio_lo95", "ratio_hi95"], rows)

    # ------------------------------------------------------------------
    # summary, long format
    # ------------------------------------------------------------------
    nan = float("nan")
    srows = []

    def add(section, S, snr, q, v, lo=nan, hi=nan):
        srows.append((section, S, snr, q, v, lo, hi))

    six = list(range(6))
    for S in ("A", "B", "AB"):
        for snr in SNRS:
            p = pts[(S, snr)]
            sec = "reported_vs_oma"
            add(sec, S, snr, "ser_reported", p.ser(0), p.ser(0) - Z * p.se(0),
                p.ser(0) + Z * p.se(0))
            add(sec, S, snr, "rel_se_reported", p.se(0) / p.ser(0))
            add(sec, S, snr, "ser_oma_contig", p.ser(IC), p.ser(IC) - Z * p.se(IC),
                p.ser(IC) + Z * p.se(IC))
            add(sec, S, snr, "ser_oma_aligned", p.ser(IA), p.ser(IA) - Z * p.se(IA),
                p.ser(IA) + Z * p.se(IA))
            add(sec, S, snr, "ser_oma_closed_form", oma_cf[snr])
            add(sec, S, snr, "ratio_vs_oma_contig", *p.ratio(0, IC))
            add(sec, S, snr, "ratio_vs_oma_aligned", *p.ratio(0, IA))
            r = p.ser(0) / oma_cf[snr]
            add(sec, S, snr, "ratio_vs_oma_closed_form", r,
                r - Z * p.se(0) / oma_cf[snr], r + Z * p.se(0) / oma_cf[snr])
            add(sec, S, snr, "corr_frame_counts_rep_omacontig",
                p.cov[0, IC] / math.sqrt(p.cov[0, 0] * p.cov[IC, IC]))
            add(sec, S, snr, "corr_frame_counts_rep_omaaligned",
                p.cov[0, IA] / math.sqrt(p.cov[0, 0] * p.cov[IA, IA]))
            sec = "reported_vs_runs"
            sers = np.array([p.ser(i) for i in range(NR)])
            add(sec, S, snr, "rank_in_six", int(np.argsort(np.argsort(sers[six]))[0]) + 1)
            add(sec, S, snr, "rank_in_sixteen", int(np.argsort(np.argsort(sers))[0]) + 1)
            for j in range(1, NR):
                add(sec, S, snr, f"ratio_vs_{NAMES[j]}", *p.ratio(0, j))
            sec = "runs_vs_oma"
            rat = [p.ratio(i, IC) for i in range(NR)]
            add(sec, S, snr, "n_runs", NR)
            add(sec, S, snr, "n_below_oma_point", sum(r[0] < 1 for r in rat))
            add(sec, S, snr, "n_below_oma_ci", sum(r[2] < 1 for r in rat))
            add(sec, S, snr, "n_above_oma_ci", sum(r[1] > 1 for r in rat))
            add(sec, S, snr, "mean_ser_sixteen", float(sers.mean()))
            add(sec, S, snr, "mean_ser_ten_further", float(sers[6:].mean()))
            add(sec, S, snr, "mean_ser_six", float(sers[six].mean()))
            add(sec, S, snr, "min_other_five", float(sers[1:6].min()))
            add(sec, S, snr, "max_other_five", float(sers[1:6].max()))
    for snr in SNRS:
        add("old777", "old777", snr, "ser_reported", old[0][SNRS.index(snr)])
        add("old777", "old777", snr, "ser_oma_closed_form", oma_cf[snr])
        add("old777", "old777", snr, "rank_in_six",
            int(np.argsort(np.argsort([old[j][SNRS.index(snr)] for j in six]))[0]) + 1)

    rng = np.random.default_rng(SEED_BOOT)
    for sel_set, rep_set in (("A", "B"), ("B", "A")):
        sel10 = np.array([pts[(sel_set, 10.0)].ser(i) for i in range(NR)])
        rep = {s: np.array([pts[(rep_set, s)].ser(i) for i in range(NR)]) for s in (10.0, 20.0)}
        oma_rep = {s: pts[(rep_set, s)].ser(IC) for s in (10.0, 20.0)}
        ci_hi = {s: np.array([pts[(rep_set, s)].ratio(i, IC)[2] for i in range(NR)])
                 for s in (10.0, 20.0)}
        res, sel_exact = best_of_six(sel10, rep, oma_rep, ci_hi, rng)
        tag = f"sel{sel_set}_rep{rep_set}"
        for r in res:
            add(f"bestof6_{r[0]}", tag, r[1], r[2], *r[3:])
        # how often the selected run is the reported model and how often selection on
        # the other set would pick the same run
        sel10_other = np.array([pts[(rep_set, 10.0)].ser(i) for i in range(NR)])
        subsets = np.array(list(itertools.combinations(range(NR), 6)))
        sel_other = subsets[np.arange(len(subsets)), np.argmin(sel10_other[subsets], axis=1)]
        add("bestof6_exact", tag, 10.0, "same_run_selected_on_both_sets",
            float((sel_other == sel_exact).mean()))
        has_rep = (subsets == 0).any(1)
        add("bestof6_exact", tag, 10.0, "reported_selected_when_in_subset",
            float((sel_exact[has_rep] == 0).mean()))
    write_csv(out_sum, ["section", "seed_set", "snr_db", "quantity", "value", "lo", "hi"],
              srows)
    print(f"[done] {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
