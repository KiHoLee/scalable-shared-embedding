"""Trained-codebook reference and long mask training for the periodic-mask design.

The letter keeps the codebook fixed (the BERT token embeddings whitened
under the source statistics, pwhite, exp_pm.codebook) and trains only the
user masks. This script trains the codebook as well, as a strong
reference, and trains the masks ten times
longer than the letter does.

Parts (--part):
  train     a V x d codebook (V = 30,522 BERT word pieces, d = 64) trained
            jointly with L = 16 masks for U = 4 users under flat Rayleigh
            fading, tokens drawn with p_v, SNR uniform in [0, 20] dB per
            frame, 256 frames per step, c from 8,192 frames per step, TF32 in
            training only, as exp_pm.train_masks (CodebookMasks: exp_pm.Masks
            with the codebook as a parameter). After every step the codebook
            rows are renormalized to unit norm, the energy convention of the
            letter. Losses: the exact softmax over all V tokens (full), and
            sampled softmax with 1,024 negatives per step drawn with p_v and
            shared by the batch, logits corrected by -log(1,024 p_w), hits of
            the true token among the negatives removed (sampled). Codebook
            initializations: random Gaussian (rand, exp_pm.codebook "rand")
            and pwhite (fine-tuning). The masks start from the random
            initialization of exp_pm.Masks in every run. Adam with step size
            1e-3 for the codebook and 1e-2 for the masks and the logit scale;
            both are held for the first 1,000 steps and then decay along a
            cosine to zero at the planned last step. The codebook step size
            was chosen among 1e-4, 3e-4, 1e-3, 3e-3 and 1e-2 by 2,000-step
            pilot runs with the full softmax (SER at 10 dB on the validation
            frames, both initializations). The planned last step is
            the largest of 1e4, 3e4 and 1e5 whose training time, projected
            from steps 200 to 1,000, fits within --budget_min minutes (25).
            SER at 10 dB on 20,000 validation frames (exp_pm_ext.ser, seed
            21) at steps 1e3, 3e3, 1e4, 3e4, 1e5, with the training time and
            the wall-clock time. Final models: SER, semantic distortion
            (1 - cosine of the 768-dim BERT embeddings), similarity of the
            wrong tokens, and the share of errors on tokens with p_v = 0, at
            0, 10, 20 dB on 40,000 test frames drawn with p_v (the frames of
            exp_pm_ext.ser, seed 11); the metrics of exp_pm.Evaluator on the
            test streams at 0 and 10 dB (decoding seeds 777 + 10 snr as in
            exp_pm.py --part models). The frozen codebooks of exp_pm.py
            --part models (bert_pwhite_U4_L16_flat_s{1,2,3},
            bert_rand_U4_L16_flat_s1) are scored by the same code. Codebook
            statistics per model: drift from the initialization, Pearson
            correlation between codeword cosines and BERT cosines over the
            1,000 most frequent tokens, nearest-codeword cosine of those
            tokens. Training seed 1 for every run.
            Outputs: data/pm/pm_trainedcb.csv, data/pm/pm_trainedcb_train.json,
            models data/pm/pm_models/tcb_{full,sampled}_{rand,pwhite}_U4_L16_flat_s1.pt
            (state dicts of exp_pm.Masks: load with exp_pm.Masks(sd["E"], 16, 4)).
  converge  the mask training of exp_pm.train_masks (pwhite codebook, Adam
            step size 1e-2, 256 frames, SNR uniform in [0, 20] dB) run for
            30,000 steps instead of 3,000, L = 16 and 64, flat fading and four
            taps (fs4), seeds 1 and 2. With the same seed the training draws
            the same tokens, SNRs, gains and noise as exp_pm.train_masks; the
            evaluations restore the random state. SER at 10 dB under the
            training class on 20,000 validation frames (exp_pm_ext.ser, seed
            21) at steps 200, 500 and every 1,000 steps, the training loss
            averaged since the previous log, the mean cross-energy
            (exp_pm.cross_energy), and at 3,000 and 30,000 steps the SER on
            40,000 test frames (seed 11). At step 3,000 of seed 1 the masks
            are compared with the released entries white_U4_L{L}_{cond}_s1
            (largest absolute difference). Output: data/pm/pm_converge.csv,
            masks data/pm/pm_models/conv30k_U4_L{L}_{cond}_s{seed}.pt.

Run (from code/, one GPU):
  python exp_pm_trainedcb.py --part train
  python exp_pm_trainedcb.py --part converge
or in parallel processes that merge their results into the same outputs:
  python exp_pm_trainedcb.py --part train --train_only --runs full_rand   (one per run)
  python exp_pm_trainedcb.py --part train --eval_only
  python exp_pm_trainedcb.py --part converge --conds flat --Ls 16          (and so on)
--smoke runs a few steps and frames and writes *_smoke outputs and models
to data/pm/pm_models_smoke/. --eval_only (part train) scores saved models.
--reserve_gb G waits for G GiB of free GPU memory and keeps it in the cache
of the process (for a shared GPU; no effect on the results).
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import time

import torch

import rev_channel as rc
from sse_lib import DEVICE, D, set_seed, write_csv
import exp_pm as pm
import exp_pm_ext as ext

DATA = pm.DATA                     # data/pm/
REF_MODELS = pm.MODELS             # released entries of exp_pm.py
OUT_MODELS = pm.MODELS

U = 4
L_CB = 16
BATCH = 256
K_NEG = 1024
LR_CB = 1e-3
LR_MASK = 1e-2
WARM = 1000
TRAIN_LOG = (1_000, 3_000, 10_000, 30_000, 100_000)
PLANS = (10_000, 30_000, 100_000)
CONV_ITERS = 30_000
RUNS = ("full_rand", "full_pwhite", "sampled_rand", "sampled_pwhite")
REFS = ("bert_pwhite_U4_L16_flat_s1", "bert_pwhite_U4_L16_flat_s2",
        "bert_pwhite_U4_L16_flat_s3", "bert_rand_U4_L16_flat_s1")


class FastMasks(pm.Masks):
    """exp_pm.Masks without host synchronization in calibrate() and
    scores(): c stays on the GPU, and the coefficients 2a and a^2, a = 1/c,
    are formed in float64 and rounded to float32 as Python does for
    exp_pm.Masks, so the outputs are the same bit for bit. On a GPU shared
    by many processes every synchronization waits for a time slice."""

    @torch.no_grad()
    def calibrate(self, pd, n=65536, users=None):
        users = users or self.users
        idx = torch.multinomial(pd, n * users, replacement=True).view(n, users).to(self.E.device)
        self.c.copy_(self.tx(idx).pow(2).sum(1).mean().sqrt())

    def scores(self, r, w):
        m = self.masks()[: r.shape[1]]
        cand = self.E[None] * m[:, None, :]
        a = 1.0 / self.c.double()
        c1, c2 = (2 * a).float(), (a * a).float()
        wc = torch.einsum("nud,uvd->nuv", w * r, cand)
        wn = torch.einsum("nud,uvd->nuv", w, cand.pow(2))
        return c1 * wc - c2 * wn


def channel(x, snr_db, users, *, taps=1, nmse=0.0, offsets_db=None):
    """rev_channel.channel (perm = None) with the noise levels moved to the
    GPU from pinned memory without a host synchronization; same draws and
    the same outputs bit for bit."""
    N, d = x.shape
    K = d // 2
    xc = torch.complex(x[:, 0::2], x[:, 1::2])
    off = torch.zeros(users) if offsets_db is None else torch.tensor(offsets_db, dtype=torch.float32)
    snr_t = torch.as_tensor(snr_db, dtype=torch.float32).reshape(-1, 1)
    gam = 10.0 ** ((snr_t + off[None, :]) / 10.0)
    sig = torch.sqrt(1.0 / (d * gam))
    if x.device.type == "cuda":
        sig = sig.pin_memory().to(x.device, non_blocking=True)

    def cn(*shape):
        return torch.complex(torch.randn(*shape, device=x.device),
                             torch.randn(*shape, device=x.device)) / math.sqrt(2)

    if taps == 1:
        g = cn(N, users, 1).expand(N, users, K)
        e = cn(N, users, 1).expand(N, users, K) * math.sqrt(nmse)
    else:
        a = cn(N, users, taps) / math.sqrt(taps)
        k = torch.arange(K, device=x.device)
        t = torch.arange(taps, device=x.device)
        F = torch.exp(-2j * math.pi * torch.outer(t, k).float() / K)
        g = a @ F.to(a.dtype)
        e = cn(N, users, K) * math.sqrt(nmse)
    gh = g + e
    w = cn(N, users, K) * math.sqrt(2.0) * sig[:, :, None]
    z = g * xc[:, None, :] + w
    q = z / gh
    r = torch.stack([q.real, q.imag], dim=-1).reshape(N, users, d)
    wt = (gh.abs() ** 2).repeat_interleave(2, dim=-1)
    return r, wt


class CodebookMasks(FastMasks):
    """exp_pm.Masks with the codebook E as a trained parameter whose rows
    are kept at unit norm by renorm() after every step."""

    def __init__(self, E: torch.Tensor, L: int, users: int):
        super().__init__(E, L, users)
        E0 = self._buffers.pop("E")
        self.E = torch.nn.Parameter(E0)

    @torch.no_grad()
    def renorm(self):
        self.E.div_(self.E.norm(dim=1, keepdim=True).clamp_min(1e-8))


@contextlib.contextmanager
def frozen_rng():
    """Evaluate in full fp32 and leave the training random state untouched."""
    cpu = torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    tf = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        with torch.no_grad():
            yield
    finally:
        torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)
        torch.backends.cuda.matmul.allow_tf32 = tf


def sync():
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


def reserve_gpu(gb: float, wait_s: float = 7200.0):
    """On a shared GPU: wait until gb GiB are free and keep them in the
    PyTorch cache of this process (no effect on the results)."""
    if DEVICE.type != "cuda" or gb <= 0:
        return
    t0 = time.time()
    while True:
        free, _ = torch.cuda.mem_get_info()
        if free > (gb + 0.25) * 2 ** 30:
            try:
                x = torch.empty(int(gb * 2 ** 30), dtype=torch.uint8, device=DEVICE)
                del x
                print(f"[gpu] reserved {gb} GiB after {time.time() - t0:.0f} s", flush=True)
                return
            except torch.OutOfMemoryError:
                pass
        if time.time() - t0 > wait_s:
            raise RuntimeError("GPU memory not available")
        time.sleep(10)


@torch.no_grad()
def ser_metrics(m, pd, snr, cond, Wn, unseen, frames=40000, seed=11):
    """exp_pm_ext.ser on the same frames (same seed and draws), with the
    semantic distortion (1 - cosine of the BERT embeddings), the mean
    similarity of the wrong tokens, the share of wrong tokens with
    similarity above 0.4, and the share of errors decoded to tokens with
    p_v = 0. Sums stay on the GPU until the end."""
    torch.manual_seed(seed + int(10 * snr))
    z = lambda dt: torch.zeros((), dtype=dt, device=pd.device)
    err, eun, ecl = z(torch.long), z(torch.long), z(torch.long)
    sem, esim = z(torch.float64), z(torch.float64)
    for k in range(0, frames, 512):
        n = min(512, frames - k)
        idx = ext.sample(pd, n, m.users, 0.0)
        x = m.tx(idx) / m.c
        r, w = rc.channel(x, snr, m.users, **cond)
        dec = m.scores(r, w).argmax(-1)
        wrong = dec != idx
        cos = (Wn[idx] * Wn[dec]).sum(-1).double()
        err += wrong.sum()
        sem += (1 - cos).sum()
        esim += (cos * wrong).sum()
        ecl += ((cos > 0.4) & wrong).sum()
        eun += (unseen[dec] & wrong).sum()
    N = frames * m.users
    e = max(int(err), 1)
    return dict(ser=int(err) / N, sem_dist=float(sem) / N, err_tok_sim=float(esim) / e,
                err_close=int(ecl) / e, err_unseen=int(eun) / e)


def sampled_loss(m, r, w, idx, pdd, logq, scale, ce, tgt0):
    """Sampled softmax: the true token and K_NEG negatives drawn with p_v
    (shared by the batch), logits corrected by -log(K_NEG p_w), hits of the
    true token among the negatives removed."""
    neg = torch.multinomial(pdd, K_NEG, replacement=True)
    M = m.masks()
    a = 1.0 / m.c.double()
    c1, c2 = (2 * a).float(), (a * a).float()
    cn = m.E[neg][None] * M[:, None, :]                                    # (U,K,d)
    s_neg = c1 * torch.einsum("nud,ukd->nuk", w * r, cn) \
        - c2 * torch.einsum("nud,ukd->nuk", w, cn.pow(2))
    ct = m.E[idx] * M[None]                                                # (n,U,d)
    s_t = c1 * (w * r * ct).sum(-1) - c2 * (w * ct.pow(2)).sum(-1)         # (n,U)
    lt = s_t * scale - logq[idx]
    ln = (s_neg * scale - logq[neg]).masked_fill(neg[None, None, :] == idx[:, :, None],
                                                 float("-inf"))
    logits = torch.cat([lt[..., None], ln], -1).reshape(-1, K_NEG + 1)
    return ce(logits, tgt0)


def train_codebook(E0, pd, loss_kind, seed=1, lr_cb=LR_CB, log_steps=TRAIN_LOG,
                   plans=PLANS, val_frames=20000, budget_s=1500.0, cond=None):
    cond = cond or pm.COND["flat"]
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(seed)
    m = CodebookMasks(E0, L_CB, U).to(DEVICE)
    pdd = pd.to(DEVICE)
    opt = torch.optim.Adam([{"params": [m.E], "lr": lr_cb},
                            {"params": [m.Wm, m.logit_scale], "lr": LR_MASK}])
    base = [g["lr"] for g in opt.param_groups]
    ce = torch.nn.CrossEntropyLoss()
    logq = torch.log((K_NEG * pdd).clamp_min(1e-30))
    tgt0 = torch.zeros(BATCH * U, dtype=torch.long, device=DEVICE)
    warm = min(WARM, log_steps[0])
    total = max(log_steps)                     # replaced by the plan after the warm phase
    curve, t_train, t200 = [], 0.0, None
    loss_acc = torch.zeros((), device=DEVICE)
    n_acc = 0
    t0 = t_seg = time.time()
    it = 0
    while it < total:
        it += 1
        if it > warm:                          # cosine decay from the end of the warm phase
            f = 0.5 * (1 + math.cos(math.pi * (it - warm) / max(total - warm, 1)))
            for g, b in zip(opt.param_groups, base):
                g["lr"] = b * f
        idx = torch.multinomial(pdd, BATCH * U, replacement=True).view(BATCH, U)
        snr = torch.empty(BATCH).uniform_(0.0, 20.0)
        m.calibrate(pdd, 8192)
        x = m.tx(idx) / m.c
        r, w = channel(x, snr, U, **cond)
        scale = m.logit_scale.exp()
        if loss_kind == "full":
            sc = m.scores(r, w) * scale
            loss = ce(sc.reshape(-1, sc.shape[-1]), idx.reshape(-1))
        else:
            loss = sampled_loss(m, r, w, idx, pdd, logq, scale, ce, tgt0)
        opt.zero_grad()
        loss.backward()
        opt.step()
        m.renorm()
        loss_acc += loss.detach()
        n_acc += 1
        if it == min(200, warm // 5) and t200 is None:
            sync()
            t200 = (it, t_train + time.time() - t_seg)
        if it == warm and it not in log_steps:
            log_steps = tuple(sorted(set(log_steps) | {it}))
        if it in log_steps:
            sync()
            t_train += time.time() - t_seg
            with frozen_rng():
                m.calibrate(pdd)
                v = ext.ser(m, pdd, 0.0, 10, cond, frames=val_frames, seed=21)
            wall = time.time() - t0
            curve.append([it, round(t_train, 1), round(wall, 1), float(loss_acc / n_acc),
                          v, opt.param_groups[0]["lr"]])
            print(f"[train] {loss_kind} it={it} train_s={t_train:.1f} wall_s={wall:.1f} "
                  f"loss={curve[-1][3]:.4f} ser10_val={v:.4f}", flush=True)
            loss_acc.zero_()
            n_acc = 0
            if it == warm:                     # fix the plan from the rate of steps 200..warm
                rate = (t_train - t200[1]) / max(it - t200[0], 1)
                val_s = wall - t_train
                fit = [p for p in plans if t_train + rate * (p - it) + 3 * val_s <= budget_s]
                total = max(fit) if fit else min(plans)
                log_steps = tuple(s for s in log_steps if s <= total)
                print(f"[plan] {loss_kind} {1000 * rate:.2f} ms/step -> {total} steps", flush=True)
            t_seg = time.time()
    torch.backends.cuda.matmul.allow_tf32 = False
    m.calibrate(pdd)
    m.eval()
    info = dict(loss=loss_kind, seed=seed, lr_codebook=lr_cb, lr_masks=LR_MASK, batch=BATCH,
                negatives=K_NEG if loss_kind == "sampled" else None, warm_steps=warm,
                schedule="constant for warm_steps, then cosine to zero at steps",
                steps=total, steps_requested=max(plans), budget_s=budget_s,
                stopped_before_requested=total < max(plans),
                ms_per_step=round(1000 * t_train / total, 3), train_s=round(t_train, 1),
                curve_cols=["step", "train_s", "wall_s", "loss", "ser10_val", "lr_codebook"],
                curve=curve, logit_scale=float(m.logit_scale.detach().exp()), c=float(m.c))
    return m, info


@torch.no_grad()
def codebook_stats(E, E0, pd, Wn, top=1000):
    """Drift from the initialization and the semantic geometry of a codebook."""
    E, E0 = E.to(DEVICE), E0.to(DEVICE)
    cos0 = (E * E0).sum(1)
    out = dict(drift_cos_pv=float((pd * cos0).sum()), drift_cos_uniform=float(cos0.mean()))
    if (pd == 0).any():
        out["drift_cos_unseen"] = float(cos0[pd == 0].mean())
    t = pd.topk(top).indices
    Cb = Wn[t] @ Wn[t].T
    Ce = E[t] @ E[t].T
    off = ~torch.eye(top, dtype=torch.bool, device=DEVICE)
    x, y = Cb[off], Ce[off]
    x, y = x - x.mean(), y - y.mean()
    out["sim_corr_top1000"] = float((x * y).sum() / (x.norm() * y.norm()))
    nn = E[t] @ E.T
    nn[torch.arange(top, device=DEVICE), t] = -2.0
    out["nn_cos_top1000"] = float(nn.max(1).values.mean())
    return out


def load_masks(path, L=L_CB):
    sd = torch.load(path, map_location="cpu")
    m = pm.Masks(sd["E"], L, U)
    m.load_state_dict(sd)
    return m.to(DEVICE).eval()


@contextlib.contextmanager
def data_lock():
    """Exclusive lock on data/pm/ while one process merges its results into
    an output shared with other processes."""
    import fcntl
    import os
    fd = os.open(DATA, os.O_RDONLY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def merge_json(path, upd: dict) -> dict:
    with data_lock():
        cur = json.load(open(path)) if path.exists() else {}
        cur.update(upd)
        json.dump(cur, open(path, "w"), indent=1)
    return cur


def merge_csv(path, hdr, rows, nkey):
    """Rows of this process replace the rows with the same first nkey
    columns; the file is sorted by those columns."""
    with data_lock():
        old = []
        if path.exists():
            lines = open(path).read().splitlines()[1:]
            old = [ln.split(",") for ln in lines if ln]
        keys = {tuple(str(v) for v in r[:nkey]) for r in rows}
        allr = [r for r in old if tuple(r[:nkey]) not in keys] + list(rows)
        allr.sort(key=lambda r: tuple(int(v) if str(v).isdigit() else str(v) for v in r[:nkey + 1]))
        write_csv(path, hdr, allr)


HDR = ["codebook", "init", "loss", "steps", "frames", "snr_db", "ser", "sem_dist",
       "err_tok_sim", "err_close", "topic_acc", "gtr_sim", "err_unseen"]


def part_train(a):
    src = pm.Source("bert")
    pd = src.dist()
    pdd = pd.to(DEVICE)
    Wn = src.Wn.to(DEVICE)
    unseen = pdd == 0
    books = {"rand": pm.codebook(src.W, D, "rand"), "pwhite": pm.codebook(src.W, D, "pwhite", pd=pd)}
    jpath = DATA / f"pm_trainedcb_train{a.tag}.json"
    info = merge_json(jpath, {"setup": dict(
        V=src.V, d=D, users=U, L=L_CB, unseen_tokens=int(unseen.sum()),
        trained_values=src.V * D + U * L_CB + 1, budget_min=a.budget_min,
        lr_codebook=a.lr_cb, lr_masks=LR_MASK, negatives=K_NEG, warm_steps=WARM,
        log_steps=list(a.log_steps), plans=list(a.plans), val_frames=a.val_frames,
        test_frames=a.test_frames)})
    models = {}
    for run in a.runs.split(","):
        loss_kind, init = run.split("_")
        name = f"tcb_{loss_kind}_{init}_U4_L16_flat_s1"
        if a.eval_only:
            models[name] = (load_masks(OUT_MODELS / f"{name}.pt"), init, loss_kind,
                            info[name]["steps"])
            continue
        m, rinfo = train_codebook(books[init], pd, loss_kind, lr_cb=a.lr_cb,
                                  log_steps=a.log_steps, plans=a.plans,
                                  val_frames=a.val_frames, budget_s=60 * a.budget_min)
        rinfo["init"] = init
        rinfo["stats"] = codebook_stats(m.E.detach(), books[init], pdd, Wn)
        torch.save(m.state_dict(), OUT_MODELS / f"{name}.pt")
        info = merge_json(jpath, {name: rinfo})
        print("[saved]", name, rinfo["stats"], flush=True)
        models[name] = (m, init, loss_kind, rinfo["steps"])
    if a.train_only:
        return
    refs = {}
    for name in REFS:
        m = load_masks(REF_MODELS / f"{name}.pt")
        init = name.split("_")[1]
        dE = float((m.E.cpu() - books[init]).abs().max())
        refs[name] = dict(codebook_vs_exp_pm_codebook_maxabs=dE,
                          stats=codebook_stats(m.E, books[init], pdd, Wn))
        models[name] = (m, init, "frozen", 3000)
    merge_json(jpath, {"reference": refs})
    sc = pm.Scorer()
    ev = pm.Evaluator(src, U, sc)
    print(f"[clean] acc {ev.clean_acc:.4f}", flush=True)
    rows = []
    for name, (m, init, loss_kind, steps) in models.items():
        for snr in a.snrs:
            r = ser_metrics(m, pdd, snr, pm.COND["flat"], Wn, unseen, frames=a.test_frames)
            rows.append([name, init, loss_kind, steps, "pv_tokens", snr, r["ser"], r["sem_dist"],
                         r["err_tok_sim"], r["err_close"], "", "", r["err_unseen"]])
            print("[eval]", name, snr, {k: round(v, 4) for k, v in r.items()}, flush=True)
        for snr in a.text_snrs:
            dec = pm.dec_mask(m, ev.ids, snr, pm.COND["flat"], 777 + int(10 * snr))
            s = ev.score(dec)
            wrong = dec != ev.ids
            dd = dec.clone()
            dd[(dd < 0) | (dd >= src.V)] = src.unk
            eu = float((unseen.cpu()[dd] & wrong).sum()) / max(int(wrong.sum()), 1)
            rows.append([name, init, loss_kind, steps, "test_streams", snr] + s + [eu])
            print("[text]", name, snr, [round(v, 4) for v in s], round(eu, 4), flush=True)
        write_csv(DATA / f"pm_trainedcb{a.tag}.csv", HDR, rows)


def train_masks_long(E, L, pd, cond, cn, iters, seed, val_frames, test_frames, steps_log,
                     test_steps, lr=1e-2, batch=256):
    """exp_pm.train_masks with logged evaluations that restore the random
    state, so the training draws equal those of exp_pm.train_masks."""
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(seed)
    m = FastMasks(E, L, U).to(DEVICE)
    pdd = pd.to(DEVICE)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    ce = torch.nn.CrossEntropyLoss()
    rows = []
    loss_acc = torch.zeros((), device=DEVICE)
    n_acc = 0
    t_train = 0.0
    t_seg = time.time()
    for it in range(1, iters + 1):
        idx = torch.multinomial(pdd, batch * U, replacement=True).view(batch, U)
        snr = torch.empty(batch).uniform_(0.0, 20.0)
        m.calibrate(pdd, 8192)
        x = m.tx(idx) / m.c
        r, w = channel(x, snr, U, **cond)
        sc = m.scores(r, w) * m.logit_scale.exp()
        loss = ce(sc.reshape(-1, sc.shape[-1]), idx.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        loss_acc += loss.detach()
        n_acc += 1
        if it in steps_log:
            sync()
            t_train += time.time() - t_seg
            with frozen_rng():
                m.calibrate(pdd)
                v = ext.ser(m, pdd, 0.0, 10, cond, frames=val_frames, seed=21)
                vt = ext.ser(m, pdd, 0.0, 10, cond, frames=test_frames) if it in test_steps else ""
                iota = sum(pm.cross_energy(m.masks().detach().cpu())) / U
                diff = ""
                ref = REF_MODELS / f"white_U4_L{L}_{cn}_s1.pt"
                if it == 3000 and seed == 1 and ref.exists():
                    sd = torch.load(ref, map_location="cpu")
                    r0 = pm.Masks(sd["E"], L, U)
                    r0.load_state_dict(sd)
                    diff = float((r0.masks() - m.masks().detach().cpu()).abs().max())
            rows.append([cn, L, seed, it, round(t_train, 1), float(loss_acc / n_acc), v, vt,
                         iota, diff])
            print("[conv]", cn, L, seed, it, f"train_s={t_train:.1f} loss={rows[-1][5]:.4f} "
                  f"ser10_val={v:.4f} test={vt} iota={iota:.4f} diff={diff}", flush=True)
            loss_acc.zero_()
            n_acc = 0
            t_seg = time.time()
    torch.backends.cuda.matmul.allow_tf32 = False
    m.calibrate(pdd)
    return m.eval(), rows


def part_converge(a):
    src = pm.Source("bert")
    pd = src.dist()
    E = pm.codebook(src.W, D, "pwhite", pd=pd)
    iters = a.conv_iters
    steps_log = set(s for s in (200, 500) if s < iters) | set(range(1000, iters + 1, 1000))
    if a.smoke:
        steps_log = {50, 100, iters}
    test_steps = {3000, iters} if not a.smoke else {iters}
    hdr = ["channel", "L", "seed", "step", "train_s", "loss", "ser10_val", "ser10_test",
           "mean_iota", "mask_diff_released"]
    for cn in a.conds.split(","):
        for L in a.Ls:
            for s in a.seeds:
                m, rr = train_masks_long(E, L, pd, pm.COND[cn], cn, iters, s, a.val_frames,
                                         a.test_frames, steps_log, test_steps)
                torch.save(m.state_dict(), OUT_MODELS / f"conv30k_U4_L{L}_{cn}_s{s}.pt")
                merge_csv(DATA / f"pm_converge{a.tag}.csv", hdr, rr, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["train", "converge"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--eval_only", action="store_true")
    ap.add_argument("--runs", default=",".join(RUNS))
    ap.add_argument("--lr_cb", type=float, default=LR_CB)
    ap.add_argument("--budget_min", type=float, default=25.0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--train_only", action="store_true")
    ap.add_argument("--conds", default="flat,fs4")
    ap.add_argument("--Ls", default="16,64")
    ap.add_argument("--seeds", default="1,2")
    ap.add_argument("--reserve_gb", type=float, default=0.0)
    a = ap.parse_args()
    reserve_gpu(a.reserve_gb)
    a.log_steps, a.plans = TRAIN_LOG, PLANS
    a.val_frames, a.test_frames = 20000, 40000
    a.snrs, a.text_snrs = (0, 10, 20), (0, 10)
    a.conv_iters = CONV_ITERS
    a.Ls = tuple(int(v) for v in a.Ls.split(","))
    a.seeds = tuple(int(v) for v in a.seeds.split(","))
    if a.smoke:                  # keep smoke outputs apart from the real ones
        global OUT_MODELS
        OUT_MODELS = DATA / "pm_models_smoke"
        OUT_MODELS.mkdir(exist_ok=True)
        a.tag = a.tag + "_smoke"
        a.log_steps, a.plans = (20, 40, 80), (40, 80)
        a.val_frames, a.test_frames = 1024, 2048
        a.snrs, a.text_snrs = (10,), (10,)
        a.conv_iters, a.Ls, a.seeds = 150, (16,), (1,)
    t0 = time.time()
    {"train": part_train, "converge": part_converge}[a.part](a)
    if DEVICE.type == "cuda":
        print(f"[gpu] peak allocated {torch.cuda.max_memory_allocated() / 2 ** 30:.2f} GiB, "
              f"reserved {torch.cuda.max_memory_reserved() / 2 ** 30:.2f} GiB", flush=True)
    print("[done]", a.part, round((time.time() - t0) / 60, 1), "min", flush=True)


if __name__ == "__main__":
    main()
