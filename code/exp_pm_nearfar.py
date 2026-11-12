"""Near-far study: per-user errors under SNR offsets, class entries trained
with one logit scale per receiver, and an energy split across the users.

The offset classes off6 and off12 of exp_pm.COND give user u the nominal SNR
plus -6, -2, 2, 6 dB or -12, -4, 4, 12 dB (user 0 is the weakest). The class
entries of exp_pm.py --part channel are trained with exp_pm.Masks, whose one
learned logit scale exp(s) multiplies the decoder scores of all four
receivers in the cross-entropy. Under off12 that training leaves the mask of
the weakest user with weight on one pattern position and spreads the masks of
the others over the rest. This script retrains the class entries with one
logit scale per receiver, scores every user separately, and tests an energy
split across the users for the comb and OMA.

Parts (--part):
  retrain  off6 and off12 entries trained exactly as exp_pm.train_masks
           (3,000 Adam steps of 256 frames, step size 1e-2, SNR uniform in
           [0, 20] dB per frame, the class channel, BERT pwhite codebook,
           TF32 in training only, c from 8,192 frames per step and 65,536 at
           the end) except that receiver u has its own learned logit scale
           exp(s_u), s_u initialized to 2.0 (MasksPR). L = 4, 16, 64, seeds
           1, 2, 3. With the same seed the training draws the same tokens,
           SNRs, gains and noise as exp_pm.train_masks.
           Masks: data/pm/pm_models/white_U4_L{L}_{cond}_pr_s{seed}.pt
           Log: data/pm/pm_nearfar_train.json. Per entry: the pattern (U x L),
           the number of pattern positions with weight per user (|m| > 0.5,
           where the mean square over the L positions is one), the effective
           number of positions (sum m^2)^2 / sum m^4, the real dimensions
           with such weight, the share of the unit frame energy per user,
           the logit scales, the
           cross-energy (exp_pm.cross_energy), the SER at 10 dB on tokens
           drawn with p_v (exp_pm.quick_ter, as in pm_channel_train.json) and
           the training time. Key "reference" holds the same statistics of the
           released entries white_U4_L16_flat_s1 and white_U4_L{16,64}_{cond}_s1.
  peruser  per-user SER and semantic distortion, and per-user and mean topic
           accuracy and GTR similarity (texts and scorer of exp_pm.Evaluator),
           on the test streams (24,998 frames at U = 4), at 0, 5, 10, 15,
           20 dB under off6 and off12, for OMA, the flat entry
           white_U4_L16_flat_s1, the comb (L = U, exp_pm.fixed_pattern), the
           released class entries white_U4_L{16,64}_{cond}_s1 and the entries
           of part retrain (L = 4, 16, 64, seeds 1-3); each class entry is
           scored under its own class. Decoding seeds 777 + 10 snr as in
           exp_pm.py, so the means of OMA, the flat entry and the released
           class entries reproduce pm_channel_eval.csv; the log prints the
           largest differences ([check]). The constant c of the comb is drawn
           with seed 0 (exp_pm.py --part comb left it unseeded, so its rows in
           pm_comb_eval.csv can differ by a few errors).
           Output: data/pm/pm_nearfar_eval.csv
  split    the comb (L = U) and OMA with user energies
           p_u = 10^(-a off_u / 10) / mean_k 10^(-a off_k / 10),
           a = 0, 0.25, 0.5, 0.75, 1 (a = 0 is the equal split of part
           peruser, a = 1 equalizes the average SNRs of the users), off6 and
           off12, 0-20 dB, same seeds and metrics as part peruser. The comb
           scales the mask of user u by sqrt(p_u) and draws c again (seed 0),
           so the frame energy stays one; receiver u knows its own scaled mask.
           OMA scales the 16 BPSK symbols of user u by sqrt(p_u); detection is
           by sign, so its receivers need no knowledge of p_u.
           Output: data/pm/pm_nearfar_split.csv
  all      retrain, peruser and split in one process

CSV columns. pm_nearfar_eval.csv: scheme (oma, flat_entry, comb, class_entry
= released entry with one logit scale, class_entry_pr = retrained with one
scale per receiver), entry (mask file or "-"), L, seed, train_cond, cond,
snr_db, user (0-3 or "mean"), ser, sem_dist, topic_acc, gtr_sim.
pm_nearfar_split.csv: scheme (comb, oma), a, cond, snr_db, user, energy (p_u;
1 on the mean rows), ser, sem_dist, topic_acc, gtr_sim. The rows with user =
"mean" are the values over all users, computed as exp_pm.Evaluator.score
(ser is its ter); the per-user topic accuracy and GTR similarity are taken
over the articles of that user.

Seeds: training set_seed(s), s = 1, 2, 3; codebook rotation generator seed 0;
decoding on the test streams 777 + 10 snr; constant c of the comb seed 0;
quick SER exp_pm.quick_ter seed 4242.

Run from the code folder (GPU), after exp_pm.py --part channel:
    python exp_pm_nearfar.py --part retrain
    python exp_pm_nearfar.py --part peruser     # needs part retrain
    python exp_pm_nearfar.py --part split
--smoke: 50 training steps, off12 only, L = 16, seed 1, 10 dB, a = 0 and 1,
the first 2,000 frames of the test streams; masks go to
data/pm/pm_models_smoke/ and outputs get the suffix _smoke.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time

import torch

import exp_pm as pm
import exp_pm_ext as px
import rev_channel as rc
from sse_lib import DEVICE, D, set_seed, write_csv

U = 4
CONDS = ["off6", "off12"]
SNRS = [0, 5, 10, 15, 20]
LS = [4, 16, 64]
SEEDS = [1, 2, 3]
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
REF_MODELS = pm.DATA / "pm_models"          # released entries (read only)
NEW_MODELS = pm.DATA / "pm_models"          # entries of part retrain
SUFFIX = ""
FRAMES = 0                                  # 0: the whole test streams

EVAL_HDR = ["scheme", "entry", "L", "seed", "train_cond", "cond", "snr_db", "user",
            "ser", "sem_dist", "topic_acc", "gtr_sim"]
SPLIT_HDR = ["scheme", "a", "cond", "snr_db", "user", "energy",
             "ser", "sem_dist", "topic_acc", "gtr_sim"]


# ----------------------------------------------------------------------
# masks with one logit scale per receiver, and their training
# ----------------------------------------------------------------------
class MasksPR(pm.Masks):
    """exp_pm.Masks with one learned logit scale exp(s_u) per receiver."""

    def __init__(self, E: torch.Tensor, L: int, users: int):
        super().__init__(E, L, users)
        self.logit_scale = torch.nn.Parameter(torch.full((users,), 2.0))


def train_masks_pr(E, L, users, pd, cond, iters=3000, seed=1, lr=1e-2, batch=256):
    """exp_pm.train_masks with the scores of receiver u multiplied by its own
    exp(s_u) in the cross-entropy; everything else is unchanged."""
    torch.backends.cuda.matmul.allow_tf32 = True      # training only; evaluation runs in full fp32
    set_seed(seed)
    m = MasksPR(E, L, users).to(DEVICE)
    pdd = pd.to(DEVICE)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    ce = torch.nn.CrossEntropyLoss()
    t0 = time.time()
    for it in range(1, iters + 1):
        idx = torch.multinomial(pdd, batch * users, replacement=True).view(batch, users)
        snr = torch.empty(batch).uniform_(0.0, 20.0)
        m.calibrate(pdd, 8192)
        x = m.tx(idx) / m.c
        r, w = rc.channel(x, snr, users, **cond)
        sc = m.scores(r, w) * m.logit_scale.exp()[None, :, None]
        loss = ce(sc.reshape(-1, sc.shape[-1]), idx.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    torch.backends.cuda.matmul.allow_tf32 = False
    m.calibrate(pdd)
    m.eval()
    return m, time.time() - t0


def load_entry(path):
    """Load a saved mask set; one logit scale -> exp_pm.Masks, one per
    receiver -> MasksPR."""
    sd = torch.load(path, map_location="cpu")
    users, L = sd["Wm"].shape
    cls = MasksPR if sd["logit_scale"].dim() == 1 else pm.Masks
    m = cls(sd["E"], L, users)
    m.load_state_dict(sd)
    return m.to(DEVICE).eval()


@torch.no_grad()
def describe(m, pdd):
    """Pattern statistics of a mask set (see the module docstring)."""
    M = m.masks().detach()                                   # (U,d)
    pat = M[:, :m.L]
    w2 = pat.pow(2)
    held = (pat.abs() > 0.5).sum(1)
    s2 = pdd @ m.E.pow(2)                                    # (d,) second moment per dimension
    share = (M.pow(2) @ s2) / float(m.c) ** 2
    return dict(L=m.L,
                positions=[int(x) for x in held],
                eff_positions=[round(float(x), 2) for x in w2.sum(1).pow(2) / w2.pow(2).sum(1)],
                real_dims=[int(x) * (D // m.L) for x in held],
                energy_share=[round(float(x), 4) for x in share],
                logit_scales=[round(float(x), 2) for x in m.logit_scale.detach().exp().reshape(-1)],
                iota=[round(x, 4) for x in pm.cross_energy(M.cpu())],
                c=round(float(m.c), 5),
                pattern=[[round(float(x), 4) for x in row] for row in pat.cpu()])


def part_retrain(a):
    src = pm.Source("bert")
    pd = src.dist()
    E = pm.codebook(src.W, D, "pwhite", pd=pd)
    pdd = pd.to(DEVICE)
    conds, Ls, seeds = (["off12"], [16], [1]) if a.smoke else (CONDS, LS, SEEDS)
    log = dict(settings=dict(users=U, iters=a.iters, batch=256, lr=1e-2, train_snr_db=[0.0, 20.0],
                             codebook="bert pwhite", logit_scale="one per receiver, s_u(0) = 2.0"),
               entries={}, reference={})
    for cn in conds:
        cond = pm.cond_for(cn, U)
        for L in Ls:
            for s in seeds:
                name = f"white_U4_L{L}_{cn}_pr_s{s}"
                m, tt = train_masks_pr(E, L, U, pd, cond, a.iters, seed=s)
                torch.save(m.state_dict(), NEW_MODELS / f"{name}.pt")
                info = dict(cond=cn, seed=s, train_s=round(tt, 1), **describe(m, pdd),
                            ter10=pm.quick_ter(m, pdd, cond, U))
                log["entries"][name] = info
                print("[train]", name, f"{tt:.1f}s ter10 {info['ter10']:.4f} positions",
                      info["positions"], "scales", info["logit_scales"], flush=True)
    refs = ["white_U4_L16_flat_s1"] + [f"white_U4_L{L}_{cn}_s1" for cn in conds for L in (16, 64)]
    for name in refs:
        if (REF_MODELS / f"{name}.pt").exists():
            m = load_entry(REF_MODELS / f"{name}.pt")
            cn = name.split("_")[3]
            log["reference"][name] = dict(cond=cn, **describe(m, pdd),
                                          ter10=pm.quick_ter(m, pdd, pm.cond_for(cn, U), U))
            print("[reference]", name, log["reference"][name]["positions"], flush=True)
    json.dump(log, open(pm.DATA / f"pm_nearfar_train{SUFFIX}.json", "w"), indent=1)


# ----------------------------------------------------------------------
# per-user scoring on the test streams
# ----------------------------------------------------------------------
class UserEvaluator(pm.Evaluator):
    """exp_pm.Evaluator with per-user scores. score_users returns the mean
    row [ser, sem_dist, topic_acc, gtr_sim] computed as Evaluator.score
    (same texts, same classifier batches, same GTR reference) and one such
    row per user."""

    def __init__(self, src, users, scorer, frames=0):
        super().__init__(src, users, scorer)
        if frames:                                 # smoke runs: the first frames only
            self.ids = self.ids[:frames]
            self.items = [[t for t in iu if t[1] <= frames] for iu in self.items]
            self.labels = torch.tensor([t[2] for u in range(users) for t in self.items[u]])
            self.ref = scorer.emb(src.texts(self.ids, self.items))
        self.n_items = [len(iu) for iu in self.items]

    @torch.no_grad()
    def _predict(self, texts, bs=256):
        sc, pr = self.sc, []
        for i in range(0, len(texts), bs):
            enc = sc.ctok(texts[i:i + bs], padding=True, truncation=True, max_length=128,
                          return_tensors="pt").to(DEVICE)
            pr.append(sc.cls(**enc).logits.argmax(-1).cpu())
        return torch.cat(pr)

    def score_users(self, dec):
        s = self.src
        d2 = dec.clone()
        d2[(d2 < 0) | (d2 >= s.V)] = s.unk
        err = (d2 != self.ids).float()                         # (N,U)
        cos = (s.Wn[self.ids] * s.Wn[d2]).sum(-1)               # (N,U)
        txt = s.texts(d2, self.items)
        hit = (self._predict(txt) == self.labels).float()
        sim = (self.sc.emb(txt) * self.ref).sum(-1)
        mean = [float(err.mean()), float(1 - cos.mean()), float(hit.mean()), float(sim.mean())]
        per, o = [], 0
        for u in range(self.users):
            n = self.n_items[u]
            per.append([float(err[:, u].mean()), float(1 - cos[:, u].mean()),
                        float(hit[o:o + n].mean()), float(sim[o:o + n].mean())])
            o += n
        return mean, per


@torch.no_grad()
def dec_oma_split(ids, snr, cond, seed, p, chunk=100_000):
    """exp_pm.dec_oma with the symbols of user u scaled by sqrt(p_u); with
    p_u = 1 it is exp_pm.dec_oma. BPSK per real dimension (U = 4), detected
    by sign, so the receivers need no knowledge of p_u."""
    users = ids.shape[1]
    torch.manual_seed(seed)
    n_sym, M, rep = rc.oma_layout(users, D, pm.BITS)
    assert M == 2, "sign detection needs BPSK (U = 4)"
    bps = int(math.log2(M))
    amp = p.to(DEVICE).sqrt()
    out = torch.empty_like(ids)
    for n0 in range(0, ids.shape[0], chunk):
        x_ids = ids[n0:n0 + chunk].to(DEVICE)
        sym = torch.stack([(x_ids >> (bps * k)) & (M - 1) for k in range(n_sym)], -1)
        x = rc.oma_frames(sym, users, D, M, rep)
        x = (x.view(-1, users, D // users) * amp[None, :, None]).reshape(-1, D)
        r, w = rc.channel(x, snr, users, **cond)
        sh = rc.oma_detect(r, w, users, D, M, rep)
        val = torch.zeros_like(x_ids)
        for k in range(n_sym):
            val = val | (sh[..., k] << (bps * k))
        out[n0:n0 + x_ids.shape[0]] = val.cpu()
    return out


def energy_split(cn, alpha):
    off = torch.tensor(pm.cond_for(cn, U)["offsets_db"], dtype=torch.float64)
    p = 10 ** (-alpha * off / 10)
    return (p / p.mean()).float()


def comb_masks(E, pd, p):
    """Comb (L = U) with the mask of user u scaled by sqrt(p_u); c drawn with
    seed 0. With p_u = 1 the masks equal exp_pm.fixed_pattern(E, U, 'comb')."""
    M = torch.zeros(U, D)
    for u in range(U):
        M[u, u::U] = math.sqrt(U * float(p[u]))
    m = px.FreeMasks(E, M).to(DEVICE)
    torch.manual_seed(0)
    m.calibrate(pd.to(DEVICE))
    return m.eval()


def setup(frames):
    src = pm.Source("bert")
    pd = src.dist()
    E = pm.codebook(src.W, D, "pwhite", pd=pd)
    sc = pm.Scorer()
    ev = UserEvaluator(src, U, sc, frames)
    print(f"[clean] frames {ev.ids.shape[0]} acc {ev.clean_acc:.4f}", flush=True)
    return src, pd, E, sc, ev


def fmt(per):
    return "[" + " ".join(f"{v[0]:.4f}" for v in per) + "]"


def part_peruser(a):
    src, pd, E, sc, ev = setup(FRAMES)
    conds, snrs = (["off12"], [10]) if a.smoke else (CONDS, SNRS)
    flat = load_entry(REF_MODELS / "white_U4_L16_flat_s1.pt")
    print("[codebook] max |E - E(flat entry)|", float((flat.E.cpu() - E).abs().max()), flush=True)
    comb = pm.fixed_pattern(E, U, "comb").to(DEVICE)
    torch.manual_seed(0)
    comb.calibrate(pd.to(DEVICE))
    comb.eval()
    # (scheme, entry, L, seed, train_cond, model); class entries only under their own class
    common = [("flat_entry", "white_U4_L16_flat_s1", 16, 1, "flat", flat),
              ("comb", "-", U, 0, "-", comb)]
    per_cond = {}
    for cn in conds:
        lst = []
        for L in (16, 64):
            name = f"white_U4_L{L}_{cn}_s1"
            if (REF_MODELS / f"{name}.pt").exists():
                lst.append(("class_entry", name, L, 1, cn, load_entry(REF_MODELS / f"{name}.pt")))
        for L in ([16] if a.smoke else LS):
            for s in ([1] if a.smoke else SEEDS):
                name = f"white_U4_L{L}_{cn}_pr_s{s}"
                lst.append(("class_entry_pr", name, L, s, cn, load_entry(NEW_MODELS / f"{name}.pt")))
        per_cond[cn] = lst
    rows = []
    checked = False
    for cn in conds:
        cond = pm.cond_for(cn, U)
        for snr in snrs:
            sd = 777 + int(10 * snr)
            dec = pm.dec_oma(ev.ids, snr, cond, sd)
            res = ev.score_users(dec)
            if not checked:                       # the mean row equals Evaluator.score
                ref = pm.Evaluator.score(ev, dec)
                print("[check-evaluator] ours", [round(x, 6) for x in res[0]], "Evaluator.score",
                      [round(ref[k], 6) for k in (0, 1, 4, 5)], flush=True)
                checked = True
            add_rows(rows, ["oma", "-", 0, 0, "-", cn, snr], res)
            for scheme, name, L, s, tc, m in common + per_cond[cn]:
                res = ev.score_users(pm.dec_mask(m, ev.ids, snr, cond, sd))
                add_rows(rows, [scheme, name, L, s, tc, cn, snr], res)
    write_csv(pm.DATA / f"pm_nearfar_eval{SUFFIX}.csv", EVAL_HDR, rows)
    check_released(rows)


def add_rows(rows, head, res, energy=None):
    mean, per = res
    for u, v in enumerate(per):
        rows.append(head + [u] + ([float(energy[u])] if energy is not None else []) + v)
    rows.append(head + ["mean"] + ([1.0] if energy is not None else []) + mean)
    print("[eval]", " ".join(str(h) for h in head), "ser per user", fmt(per),
          f"mean {mean[0]:.4f} topic {mean[2]:.4f} gtr {mean[3]:.4f}", flush=True)


def check_released(rows):
    """Largest difference of the mean rows from pm_channel_eval.csv and
    pm_comb_eval.csv (same decoding seeds)."""
    ref = {}
    for fn, part in (("pm_channel_eval.csv", "channel"), ("pm_comb_eval.csv", "comb")):
        path = pm.DATA / fn
        if not path.exists():
            continue
        for r in csv.DictReader(open(path)):
            if r["part"] != part:
                continue
            key = (r["codebook"], int(r["L"]), r["train_cond"], r["eval_cond"], int(float(r["snr_db"])))
            ref[key] = (float(r["ter"]), float(r["topic_acc"]), float(r["gtr_sim"]))
    worst = {}
    for scheme, entry, L, seed, tc, cn, snr, user, ser, _, acc, gsim in rows:
        if user != "mean":
            continue
        key = {"oma": ("oma", 0, "-"), "flat_entry": ("white", 16, "flat"),
               "class_entry": ("white", L, cn), "comb": ("pwhite", U, "comb")}.get(scheme)
        if key is None or key + (cn, snr) not in ref:
            continue
        r = ref[key + (cn, snr)]
        d = [abs(ser - r[0]), abs(acc - r[1]), abs(gsim - r[2])]
        w = worst.setdefault(f"{scheme}_L{L}", [0.0, 0.0, 0.0, 0])
        for k in range(3):
            w[k] = max(w[k], d[k])
        w[3] += 1
    for k, w in worst.items():
        print(f"[check] {k}: {w[3]} points, max |d ser| {w[0]:.2e} |d topic| {w[1]:.2e} "
              f"|d gtr| {w[2]:.2e}", flush=True)


def part_split(a):
    src, pd, E, sc, ev = setup(FRAMES)
    conds, snrs, alphas = (["off12"], [10], [0.0, 1.0]) if a.smoke else (CONDS, SNRS, ALPHAS)
    rows = []
    for cn in conds:
        cond = pm.cond_for(cn, U)
        for al in alphas:
            p = energy_split(cn, al)
            m = comb_masks(E, pd, p)
            print("[split]", cn, al, "p", [round(float(x), 4) for x in p], "c", round(float(m.c), 5), flush=True)
            for snr in snrs:
                sd = 777 + int(10 * snr)
                add_rows(rows, ["comb", al, cn, snr], ev.score_users(pm.dec_mask(m, ev.ids, snr, cond, sd)), p)
                add_rows(rows, ["oma", al, cn, snr], ev.score_users(dec_oma_split(ev.ids, snr, cond, sd, p)), p)
    write_csv(pm.DATA / f"pm_nearfar_split{SUFFIX}.csv", SPLIT_HDR, rows)


def part_all(a):
    for f in (part_retrain, part_peruser, part_split):
        t = time.time()
        f(a)
        print("[part-done]", f.__name__, round((time.time() - t) / 60, 1), "min", flush=True)


def main():
    global NEW_MODELS, SUFFIX, FRAMES
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["retrain", "peruser", "split", "all"])
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:                  # keep smoke-test masks and outputs apart from the real ones
        NEW_MODELS = pm.DATA / "pm_models_smoke"
        NEW_MODELS.mkdir(exist_ok=True)
        SUFFIX, FRAMES = "_smoke", 2000
        if a.iters == 3000:
            a.iters = 50
    t0 = time.time()
    {"retrain": part_retrain, "peruser": part_peruser, "split": part_split, "all": part_all}[a.part](a)
    print("[done]", a.part, round((time.time() - t0) / 60, 1), "min", flush=True)


if __name__ == "__main__":
    main()
