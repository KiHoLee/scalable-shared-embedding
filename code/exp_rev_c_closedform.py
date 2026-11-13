"""Earlier design: closed form of the power normalizer c of the proposal against
its Monte Carlo value, the multiplication counts of both, and training with it.

Model (sse_lib.SSE): y = (1/c) sum_u x_u with period p of x_u equal to
b_{i_p} o m_u / sqrt(P) and independent uniform digits. The digits of different
users and periods are independent, so with bbar = mean_i b_i

    c^2 = sum_u mean_i ||b_i o m_u||^2
          + sum_{u != v} (bbar o m_u)^T (bbar o m_v).

P cancels, and the cost does not depend on V.

Part 1, six runs of the main configuration (V = 65,536, Vu = 16, P = 4): the
reported model sse_P4.pt and rev_sse_seed1-5.pt. Compares the closed form with a
Monte Carlo over 4e6 frames (c_mc, with standard error), with the stored buffer
c (65,536 frames, end of training), and gives the relative standard deviation of
the 8,192-frame estimate that training re-draws at every step.
Output: data/rev_c_closedform.csv, with the multiplication counts per evaluation
of c in the columns mult_*.

Part 2, multiplication counts of a training step at b = 256, against the score
MACs of the letter, and their ratios per step and summed over
training. The MC term is counted as mask products plus power squares, and as coded
for the proposal also with the division by sqrt(P) per element.
Output: data/rev_c_closedform_ops.csv

Part 3, training at 4,000 iterations with c from the closed form at every step
against the 8,192-frame estimate at every step, seeds 1-10, SER at 10 and 20 dB
over 1e6 frames. The Monte Carlo arm is rev_factor_16_4_s*.pt of
exp_rev_factor_seeds.py, the same protocol. Random streams diverge between the
arms, so seeds are not paired.
Output: data/rev_c_closedform_train.csv, models data/rev_c_cf_s*.pt
"""
from __future__ import annotations

import csv
import json
import math

import torch

from sse_lib import (D, DATA, DEVICE, LR, P_MAX, SSE, TRAIN_BATCH, TRAIN_ITERS,
                     TRAIN_SNR_DB, U, VU, eval_ser_sse, set_seed, write_csv)

RUNS = [("reported", "sse_P4.pt")] + [(f"seed{s}", f"rev_sse_seed{s}.pt") for s in range(1, 6)]
L = D // P_MAX
V = VU ** P_MAX
N_CAL = 8192                       # frames per step of calibrate_power(8192)
K_NEG = 1024                       # sampled softmax negatives, exp_rev_budget.py


@torch.no_grad()
def closed_form_c(m: SSE) -> float:
    B = m.unit_codebook().double()
    w = m.masks().double()
    a = B.mean(0)[None] * w                         # (U,L) bbar o m_u
    c2 = (B.pow(2).mean(0) * w.pow(2)).sum() + a.sum(0).pow(2).sum() - a.pow(2).sum()
    return float(c2.sqrt())


@torch.no_grad()
def mc_c(m: SSE, frames: int = 4_000_000, chunk: int = 200_000, seed: int = 777):
    """Returns the Monte Carlo c and its standard error (delta method)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    s1 = s2 = 0.0
    for n0 in range(0, frames, chunk):
        n = min(chunk, frames - n0)
        dg = torch.randint(m.vu, (n, m.users, m.P), generator=g).to(m.B.device)
        e = m.tx_frame(dg).pow(2).sum(dim=(1, 2)).double()
        s1 += float(e.sum())
        s2 += float(e.pow(2).sum())
    mean = s1 / frames
    se = math.sqrt(max(s2 / frames - mean * mean, 0.0) / frames)
    c = math.sqrt(mean)
    return c, se / (2.0 * c)


@torch.no_grad()
def step_spread(m: SSE, n: int = N_CAL, draws: int = 200) -> float:
    """Relative standard deviation of the per-step estimate calibrate_power(n)."""
    torch.manual_seed(778)
    v = []
    for _ in range(draws):
        dg = torch.randint(m.vu, (n, m.users, m.P), device=m.B.device)
        v.append(m.tx_frame(dg).pow(2).sum(dim=(1, 2)).mean().sqrt())
    v = torch.stack(v).double()
    return float(v.std() / v.mean())


def counts() -> dict:
    """Multiplication counts, one line per array operation in closed_form_c and
    in calibrate_power. Additions and the final square root are not counted."""
    cf = (VU * L          # B.pow(2)
          + U * L         # w.pow(2)
          + U * L         # mean(B^2) * w^2
          + U * L         # bbar * w
          + L             # a.sum(0).pow(2)
          + U * L)        # a.pow(2)
    mask = N_CAL * U * D  # e * m in tx_frame
    power = N_CAL * D     # y.pow(2)
    root_p = N_CAL * U * D  # division by sqrt(P) in tx_frame, proposal only
    return {"cf": cf, "mask": mask, "power": power, "root_p": root_p}


def crossings(target: float = 0.26) -> dict:
    """Logged iterations at the target SER at 10 dB. Baselines: the last logged
    iteration still above the target, a lower bound on their crossing. Proposal:
    the first logged iteration at or below it, an upper bound on its crossing."""
    def rows(name):
        with open(DATA / name) as f:
            return [(int(r["iter"]), float(r["ser10_md"])) for r in csv.DictReader(f)]

    def last_above(cv):
        first = next(i for i, (_, s) in enumerate(cv) if s <= target)
        return cv[first - 1][0]

    def first_below(cv):
        return next(it for it, s in cv if s <= target)

    rep = json.load(open(DATA / "curve_sse_P4.json"))["curve"]
    return {"full": last_above(rows("rev_budget_curve_full.csv")),
            "sampled": last_above(rows("rev_budget_curve_sampled.csv")),
            "reported": next(int(r[0]) for r in rep if r[3] <= target),
            "slower": first_below(rows("rev_budget_curve_sse_s3.csv"))}


def part1():
    k = counts()
    rows = []
    for name, f in RUNS:
        m = SSE(P=P_MAX)
        m.load_state_dict(torch.load(DATA / f, map_location="cpu"))
        m.to(DEVICE).eval()
        cf = closed_form_c(m)
        c, se = mc_c(m)
        st = float(m.c)
        rows.append((name, cf, c, se, abs(cf - c) / c, (cf - c) / se, st, abs(cf - st) / cf,
                     step_spread(m), k["cf"], k["mask"] + k["power"],
                     k["mask"] + k["power"] + k["root_p"]))
        print(rows[-1], flush=True)
    write_csv(DATA / "rev_c_closedform.csv",
              ["run", "c_closed", "c_mc", "c_mc_se", "rel_diff_mc", "z_mc", "c_stored",
               "rel_diff_stored", "rel_std_8192", "mult_closed", "mult_mc8192",
               "mult_mc8192_ascoded"], rows)


def part2():
    k = counts()
    b = TRAIN_BATCH
    s_sse = b * U * VU * D
    s_full = b * U * V * D
    s_samp = b * U * (b * U + K_NEG) * D
    mc = k["mask"] + k["power"]
    coded = mc + k["root_p"]
    x = crossings()
    print("crossings", x, flush=True)
    rows = [("mult_closed_form", k["cf"]),
            ("mult_mc8192_mask_only", k["mask"]),
            ("mult_mc8192_mask_power", mc),
            ("mult_mc8192_as_coded_proposal", coded),
            ("mac_score_sse", s_sse), ("mac_score_full", s_full), ("mac_score_sampled", s_samp),
            ("ratio_full_score", s_full / s_sse), ("ratio_sampled_score", s_samp / s_sse),
            # the baselines always keep the 8,192-frame estimate here
            ("ratio_full_mc_mask_power", (s_full + mc) / (s_sse + mc)),
            ("ratio_sampled_mc_mask_power", (s_samp + mc) / (s_sse + mc)),
            ("ratio_full_mc_as_coded", (s_full + mc) / (s_sse + coded)),
            ("ratio_sampled_mc_as_coded", (s_samp + mc) / (s_sse + coded)),
            ("ratio_full_closed_form", (s_full + mc) / (s_sse + k["cf"])),
            ("ratio_sampled_closed_form", (s_samp + mc) / (s_sse + k["cf"])),
            ("iter_full_above_026", x["full"]), ("iter_sampled_above_026", x["sampled"]),
            ("iter_reported_at_026", x["reported"]), ("iter_slower_at_026", x["slower"])]
    for name, sb, it in (("full", s_full, x["full"]), ("sampled", s_samp, x["sampled"])):
        for run, r in (("reported", x["reported"]), ("slower", x["slower"])):
            rows.append((f"cum_{name}_vs_{run}_score", it * sb / (r * s_sse)))
            rows.append((f"cum_{name}_vs_{run}_mc_mask_power", it * (sb + mc) / (r * (s_sse + mc))))
            rows.append((f"cum_{name}_vs_{run}_mc_as_coded", it * (sb + mc) / (r * (s_sse + coded))))
            rows.append((f"cum_{name}_vs_{run}_closed_form", it * (sb + mc) / (r * (s_sse + k["cf"]))))
    for r in rows:
        print(r, flush=True)
    write_csv(DATA / "rev_c_closedform_ops.csv", ["quantity", "value"], rows)


def train_cf(seed: int, iters: int = TRAIN_ITERS) -> SSE:
    """train_sse of sse_lib with closed_form_c in place of calibrate_power."""
    set_seed(seed)
    m = SSE(P=P_MAX)
    set_seed(seed)
    m.to(DEVICE)
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    ce = torch.nn.CrossEntropyLoss()
    for _ in range(iters):
        dg = torch.randint(VU, (TRAIN_BATCH, U, P_MAX), device=DEVICE)
        snr = torch.empty(TRAIN_BATCH).uniform_(*TRAIN_SNR_DB)
        m.c.fill_(closed_form_c(m))
        sc = m(dg, snr) * m.logit_scale.exp()
        loss = ce(sc.reshape(-1, VU), dg.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    m.c.fill_(closed_form_c(m))
    return m.eval()


def part3():
    with open(DATA / "rev_factor_seeds.csv") as f:
        mc = {int(r["seed"]): r for r in csv.DictReader(f)
              if (int(r["vu"]), int(r["P"])) == (16, 4)}
    rows = []
    for seed in range(1, 11):
        p = DATA / f"rev_c_cf_s{seed}.pt"
        if p.exists():
            m = SSE(P=P_MAX)
            m.load_state_dict(torch.load(p, map_location="cpu"))
            m.to(DEVICE).eval()
        else:
            m = train_cf(seed)
            torch.save({k: v.cpu() for k, v in m.state_dict().items()}, p)
        set_seed(seed)                  # the channel draws of the evaluation
        s = eval_ser_sse(m, [10.0, 20.0], frames=1_000_000)
        c_mc, _ = mc_c(m, frames=1_000_000)
        rows.append((seed, float(mc[seed]["ser10"]), float(mc[seed]["ser20"]), s[0], s[1],
                     float(m.c), c_mc))
        print(rows[-1], flush=True)
        write_csv(DATA / "rev_c_closedform_train.csv",
                  ["seed", "ser10_mc", "ser20_mc", "ser10_cf", "ser20_cf", "c_closed", "c_mc"], rows)


if __name__ == "__main__":
    part1()
    part2()
    part3()
