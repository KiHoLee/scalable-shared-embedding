"""Shared content with U = 8 users superposed on the same 64 real dimensions,
with U = 4 as a control.

Setting of every part: flat Rayleigh fading with exact gains, d = 64 real
dimensions (K = 32 subcarriers), the BERT codebook whitened under the source
statistics (pwhite, as exp_pm_ext.part_content_snr), weighted minimum-distance
decoding over all V = 30,522 tokens. In every frame all users receive one
common token with probability rho, otherwise independent tokens, each drawn
with the source statistics p_v (sample, copied from exp_pm_ext.sample).
rho in {0, 0.5, 0.9, 1}, evaluation at 10 dB. Schemes:
  no_overlap    the untrained comb with L = U: user u weights the real
                dimensions u, u + U, u + 2U, ... with sqrt(U) (sqrt(8) at
                U = 8, 2 at U = 4), so ||m_u||^2 = d and the cross-energy is 0
  full_overlap  every user weights every dimension with 1 (||m_u||^2 = d,
                cross-energy U - 1)
  trained_10dB  masks with L = 16 trained at 10 dB under the rho served
                (train_rho, exp_pm_ext.train_rho with the load U as a
                parameter, snr_range = (10, 10)), training seeds 1 and 2,
                both reported
The SER is given in total, on frames whose users all receive the same token
(ser_shared) and on the other frames (ser_indep), as exp_pm_ext.ser_split,
together with the smallest and largest per-user SER and the mean
cross-energy (1/U) sum_u iota_u (exp_pm.cross_energy).

part u4    : U = 4, the control: trained_10dB with seed 1 draws the masks
             and the test frames of exp_pm_ext.part_content_snr and must
             reproduce its trained_at_snr rows at 10 dB (pm_content_snr.csv,
             0.184 / 0.163 / 0.098 / 0.032 at rho = 0 / 0.5 / 0.9 / 1); the
             part prints both. Outputs data/pm/pm_content8_u4.{csv,json}.
part u8    : U = 8. Outputs data/pm/pm_content8_u8.{csv,json}.
part merge : joins the two parts into data/pm/pm_content8.csv and
             data/pm/pm_content8.json and adds the coherent overlap
             kappa_u = (1/d) sum_i m_{u,i} sum_{k!=u} m_{k,i}, the signed
             weight the other users place where user u has weight, which
             adds to its codeword when the token is shared (0 without
             overlap, U - 1 for full overlap, computed from the stored masks).
part all   : u4, u8 and merge in one process.
pm_content8.csv columns: users, scheme, seed (0 for the untrained masks), L,
rho, snr_db, cross_energy (mean of iota_u), coherent_overlap (mean of
kappa_u), ser, ser_shared, ser_indep, ser_user_min, ser_user_max,
frames_shared, frames_indep (the per-part files lack coherent_overlap).
pm_content8.json holds, per entry, the per-user cross-energies iota_u and
coherent overlaps kappa_u, the per-user SER (total, shared, independent),
the calibration constant c, the stored pattern of the trained masks (first
L dimensions, 5 decimals) and the training time. No model file is written.
On a shared GPU a CUDA out-of-memory error makes a training run restart
from its seed, or a scoring slice repeat, after a pause (oom_retry); the
results do not change.

Seeds: training uses sse_lib.set_seed(s), s = 1, 2, with 3,000 steps of
256 frames, Adam 1e-2, TF32 during training only (exp_pm_ext.train_rho);
before every evaluation the constant c is recalibrated on 65,536 frames of
the statistics served after torch.manual_seed(0); test frames and channels
use torch.manual_seed(11 + int(10 snr)) (seed 111 at 10 dB) and batches of
512 frames, 40,000 frames per point, as exp_pm_ext.ser_split (the scores
of a batch are computed in slices of 128 frames, the same frames). A frame
counts as shared when all its users receive the same token (also an
independent frame whose tokens coincide by chance), as exp_pm_ext.ser_split.

Run from the code folder (GPU via sse_lib.DEVICE):
    python exp_pm_content8.py --part u4       # data/pm/pm_content8_u4.{csv,json}
    python exp_pm_content8.py --part u8       # data/pm/pm_content8_u8.{csv,json}
    python exp_pm_content8.py --part merge    # data/pm/pm_content8.{csv,json}
    python exp_pm_content8.py --part all      # the three above in one process
    python exp_pm_content8.py --part u8 --smoke   # short check, *_smoke files
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time

import torch

import exp_pm as pm
import rev_channel as rc
from exp_pm_ext import FreeMasks
from sse_lib import DEVICE, D, set_seed, write_csv

DATA = pm.DATA                     # data/pm/
RHOS = (0.0, 0.5, 0.9, 1.0)
SEEDS = (1, 2)
L_TRAIN = 16
SNR_TRAIN = 10.0
SNRS = (10,)
TEST_SEED = 11
CAL_SEED = 0
BATCH = 512
SLICE = 128                        # frames scored at once (memory on a shared GPU)
HDR = ["users", "scheme", "seed", "L", "rho", "snr_db", "cross_energy", "ser", "ser_shared",
       "ser_indep", "ser_user_min", "ser_user_max", "frames_shared", "frames_indep"]


def oom_retry(fn, *args, tries=60, wait=30, **kw):
    """Calls fn again after a CUDA out-of-memory error (shared GPU). Every
    function passed here draws no random numbers or reseeds itself, so a
    repeat gives the same result."""
    for t in range(tries):
        try:
            return fn(*args, **kw)
        except torch.OutOfMemoryError:
            pass                       # leave the handler so the failed call's tensors are freed
        torch.cuda.empty_cache()
        print(f"[oom] {fn.__name__} retry {t + 1} in {wait} s", flush=True)
        time.sleep(wait)
    raise RuntimeError(f"{fn.__name__}: out of memory after {tries} tries")


@torch.no_grad()
def decide(m, r, w):
    """argmax of exp_pm.Masks.scores, slice by slice."""
    return torch.cat([oom_retry(lambda a, b: m.scores(a, b).argmax(-1), r[j:j + SLICE], w[j:j + SLICE])
                      for j in range(0, r.shape[0], SLICE)])


# ----------------------------------------------------------------------
# copies of exp_pm_ext with the load U as a parameter
# ----------------------------------------------------------------------
def sample(pd, n, users, rho, gen=None):
    """exp_pm_ext.sample: one common token with probability rho, else
    independent tokens, all drawn with pd."""
    common = torch.multinomial(pd, n, replacement=True, generator=gen)
    indep = torch.multinomial(pd, n * users, replacement=True, generator=gen).view(n, users)
    same = torch.rand(n, generator=gen, device=pd.device) < rho if gen is None else \
        (torch.rand(n, generator=gen) < rho).to(pd.device)
    return torch.where(same[:, None], common[:, None].expand(n, users), indep)


@torch.no_grad()
def calibrate_rho(m, pd, rho, n=65536):
    """exp_pm_ext.calibrate_rho: c^2 = mean frame energy under rho."""
    idx = sample(pd, n, m.users, rho)
    m.c.fill_(float(m.tx(idx).pow(2).sum(1).mean().sqrt()))


def train_rho(E, L, pd, rho, iters, seed=1, snr_range=(0.0, 20.0), users=4):
    """exp_pm_ext.train_rho with the number of users as a parameter (the
    original fixes users = 4); identical draws for users = 4."""
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(seed)
    m = pm.Masks(E, L, users).to(DEVICE)
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    ce = torch.nn.CrossEntropyLoss()
    for it in range(iters):
        idx = sample(pd, 256, users, rho)
        snr = torch.empty(256).uniform_(*snr_range)
        calibrate_rho(m, pd, rho, 8192)
        x = m.tx(idx) / m.c
        r, w = rc.channel(x, snr, users, **pm.COND["flat"])
        sc = m.scores(r, w) * m.logit_scale.exp()
        loss = ce(sc.reshape(-1, sc.shape[-1]), idx.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    torch.backends.cuda.matmul.allow_tf32 = False
    calibrate_rho(m, pd, rho)
    return m.eval()


@torch.no_grad()
def ser_split(m, pd, rho, snr, frames=40000, seed=TEST_SEED):
    """exp_pm_ext.ser_split (same frames, same split by frame kind) that also
    keeps the errors of every user. Returns the totals (ser, ser_shared,
    ser_indep), per-user lists and the frame counts of each kind."""
    torch.backends.cuda.matmul.allow_tf32 = False
    users = m.users
    torch.manual_seed(seed + int(10 * snr))
    e_s = torch.zeros(users, dtype=torch.long)
    e_i = torch.zeros(users, dtype=torch.long)
    n_s = n_i = 0
    for k in range(0, frames, BATCH):
        n = min(BATCH, frames - k)
        idx = sample(pd, n, users, rho)
        x = m.tx(idx) / m.c
        r, w = rc.channel(x, snr, users, **pm.COND["flat"])
        wrong = (decide(m, r, w) != idx)
        same = (idx == idx[:, :1]).all(1)
        e_s += wrong[same].sum(0).cpu(); n_s += int(same.sum())
        e_i += wrong[~same].sum(0).cpu(); n_i += int((~same).sum())
    nan = float("nan")
    tot = float(e_s.sum() + e_i.sum()) / ((n_s + n_i) * users)
    sh = float(e_s.sum()) / (n_s * users) if n_s else nan
    ind = float(e_i.sum()) / (n_i * users) if n_i else nan
    pu_tot = ((e_s + e_i).double() / (n_s + n_i)).tolist()
    pu_sh = (e_s.double() / n_s).tolist() if n_s else [nan] * users
    pu_in = (e_i.double() / n_i).tolist() if n_i else [nan] * users
    return dict(ser=tot, ser_shared=sh, ser_indep=ind, user=pu_tot, user_shared=pu_sh,
                user_indep=pu_in, frames_shared=n_s, frames_indep=n_i)


# ----------------------------------------------------------------------
# masks and one load
# ----------------------------------------------------------------------
def comb_masks(users: int, d: int = D) -> torch.Tensor:
    """Untrained comb with L = U: m_u = sqrt(U) on dims u, u+U, ...
    (||m_u||^2 = d, masks without overlap)."""
    M = torch.zeros(users, d, device=DEVICE)
    for u in range(users):
        M[u, u::users] = math.sqrt(users)
    return M


def load_book():
    src = pm.Source("bert")
    pdc = src.dist()
    E = pm.codebook(src.W, D, "pwhite", pd=pdc).to(DEVICE)
    return E, pdc.to(DEVICE)


def r4(v):
    return None if isinstance(v, float) and math.isnan(v) else round(float(v), 6)


def run_load(users, a, E, pd):
    """All schemes at one load; returns csv rows and the json entries."""
    rows, info = [], {}
    rhos = (0.5,) if a.smoke else RHOS
    seeds = (1,) if a.smoke else SEEDS
    for rho in rhos:
        ms = {("no_overlap", 0): FreeMasks(E, comb_masks(users)).to(DEVICE),
              ("full_overlap", 0): FreeMasks(E, torch.ones(users, D, device=DEVICE)).to(DEVICE)}
        t_tr = {}
        for s in seeds:
            ts = time.time()
            ms[("trained_10dB", s)] = oom_retry(train_rho, E, L_TRAIN, pd, rho, a.iters, seed=s,
                                                 snr_range=(SNR_TRAIN, SNR_TRAIN), users=users)
            t_tr[s] = round(time.time() - ts, 1)
            print(f"[train] U={users} rho={rho} seed={s} {t_tr[s]} s", flush=True)
        for (name, s), m in ms.items():
            torch.manual_seed(CAL_SEED)
            calibrate_rho(m, pd, rho)
            M = m.masks().detach().cpu()
            iota = pm.cross_energy(M)
            io = sum(iota) / users
            ent = dict(users=users, scheme=name, seed=s, L=m.L, rho=rho, c=float(m.c),
                       iota=[r4(x) for x in iota])
            if name == "trained_10dB":
                ent["pattern"] = [[round(float(x), 5) for x in row] for row in M[:, :m.L]]
                ent["train_s"] = t_tr[s]
            for snr in SNRS:
                o = ser_split(m, pd, rho, snr, a.frames)
                pu = o["user"]
                rows.append([users, name, s, m.L, rho, snr, io, o["ser"], o["ser_shared"], o["ser_indep"],
                             min(pu), max(pu), o["frames_shared"], o["frames_indep"]])
                ent[f"snr{snr}"] = {k: ([r4(x) for x in v] if isinstance(v, list) else
                                        (r4(v) if isinstance(v, float) else v)) for k, v in o.items()}
                print(f"[content8] U={users} rho={rho} {name} seed={s} snr={snr} ser {o['ser']:.4f} "
                      f"shared {o['ser_shared']:.4f} indep {o['ser_indep']:.4f} "
                      f"users {min(pu):.4f}-{max(pu):.4f} iota {io:.3f}", flush=True)
            info[f"U{users}_{name}_s{s}_rho{rho}"] = ent
    return rows, info


def check_control(rows):
    """Prints the U = 4 trained_10dB rows next to the trained_at_snr rows at
    10 dB of pm_content_snr.csv (exp_pm_ext.part_content_snr)."""
    ref_path = DATA / "pm_content_snr.csv"
    if not ref_path.exists():
        print("[control] pm_content_snr.csv not found", flush=True)
        return
    ref = {}
    with open(ref_path) as f:
        for r in csv.DictReader(f):
            if r["scheme"] == "trained_at_snr" and float(r["snr_db"]) == 10.0:
                ref[float(r["rho"])] = (float(r["ser"]), float(r["cross_energy"]))
    for row in rows:
        if row[0] == 4 and row[1] == "trained_10dB" and row[5] == 10 and row[4] in ref:
            rs, ri = ref[row[4]]
            print(f"[control] rho={row[4]} seed={row[2]} ser {row[7]:.4f} (pm_content_snr {rs:.4f}, "
                  f"diff {row[7] - rs:+.4f}) iota {row[6]:.3f} (pm_content_snr {ri:.3f})", flush=True)


def save(tag, rows, info, a):
    write_csv(DATA / f"pm_content8{tag}{a.suffix}.csv", HDR, rows)
    json.dump(info, open(DATA / f"pm_content8{tag}{a.suffix}.json", "w"), indent=1)
    print("[json]", DATA / f"pm_content8{tag}{a.suffix}.json", flush=True)


# ----------------------------------------------------------------------
# parts
# ----------------------------------------------------------------------
def part_load(users, a, book=None):
    E, pd = book or load_book()
    rows, info = run_load(users, a, E, pd)
    if users == 4:
        check_control(rows)
    save(f"_u{users}", rows, info, a)
    return rows, info


def part_u4(a, book=None):
    return part_load(4, a, book)


def part_u8(a, book=None):
    return part_load(8, a, book)


def coherent_overlap(M: torch.Tensor) -> list[float]:
    """kappa_u = (1/d) sum_i m_{u,i} sum_{k!=u} m_{k,i}: the signed weight the
    other users place on the dimensions of user u, which adds to its wanted
    codeword when all users share the token (0 without overlap, U - 1 for
    full overlap; the first-order counterpart of exp_pm.cross_energy)."""
    tot = M.sum(0, keepdim=True) - M
    return [float(x) for x in (M * tot).mean(1)]


def stored_masks(ent) -> torch.Tensor:
    """The masks of a json entry on the d dimensions (CPU)."""
    users = ent["users"]
    if ent["scheme"] == "no_overlap":
        return comb_masks(users).cpu()
    if ent["scheme"] == "full_overlap":
        return torch.ones(users, D)
    W = torch.tensor(ent["pattern"])
    return W.repeat(1, D // W.shape[1])


def part_merge(a):
    """Joins the parts and adds the coherent overlap (mean over users of
    coherent_overlap, from the stored patterns) after cross_energy."""
    rows, info = [], {}
    for users in (4, 8):
        with open(DATA / f"pm_content8_u{users}{a.suffix}.csv") as f:
            rows.extend(csv.DictReader(f))
        info.update(json.load(open(DATA / f"pm_content8_u{users}{a.suffix}.json")))
    hdr = HDR[:HDR.index("cross_energy") + 1] + ["coherent_overlap"] + HDR[HDR.index("cross_energy") + 1:]
    out = []
    for r in rows:
        key = f"U{int(r['users'])}_{r['scheme']}_s{int(r['seed'])}_rho{float(r['rho'])}"
        ent = info[key]
        kap = coherent_overlap(stored_masks(ent))
        ent["kappa"] = [r4(x) for x in kap]
        r = dict(r)
        r["coherent_overlap"] = f"{sum(kap) / len(kap):.10g}"
        out.append([r[h] for h in hdr])
    with open(DATA / f"pm_content8{a.suffix}.csv", "w") as f:
        f.write(",".join(hdr) + "\n")
        for r in out:
            f.write(",".join(r) + "\n")
    print("[csv]", DATA / f"pm_content8{a.suffix}.csv", flush=True)
    json.dump(info, open(DATA / f"pm_content8{a.suffix}.json", "w"), indent=1)
    print("[json]", DATA / f"pm_content8{a.suffix}.json", flush=True)


def part_all(a):
    book = load_book()
    part_u4(a, book)
    part_u8(a, book)
    part_merge(a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["u4", "u8", "merge", "all"])
    ap.add_argument("--frames", type=int, default=40000)
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--smoke", action="store_true",
                    help="2,048 frames, 50 training steps, rho = 0.5, seed 1, *_smoke outputs")
    a = ap.parse_args()
    a.suffix = ""
    if a.smoke:
        a.frames, a.iters, a.suffix = 2048, 50, "_smoke"
    t0 = time.time()
    {"u4": part_u4, "u8": part_u8, "merge": part_merge, "all": part_all}[a.part](a)
    print("[done]", a.part, round((time.time() - t0) / 60, 1), "min", flush=True)


if __name__ == "__main__":
    main()
