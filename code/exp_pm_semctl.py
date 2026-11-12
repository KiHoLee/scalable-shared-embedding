"""Semantic controls for the pretrained codebook (U = 4, flat fading, L = 16).

The proposal's codebook is the BERT token-embedding table whitened under the
source statistics p_v (exp_pm.codebook kind "pwhite", eq. (1)). The controls
below keep what the channel sees and remove what the tokens mean, so that the
text metrics separate the gain of decoding a token as one codeword from the
placement of the errors near related tokens. Every mask set is trained as in
exp_pm.py --part models (exp_pm.train_masks: U = 4, L = 16, flat fading,
3000 iterations, SNR drawn uniformly on 0-20 dB, mask training seed s) and
graded with exp_pm.Evaluator on the AG News test streams (first 2,000 test
articles, 1,948 graded articles, 24,998 frames of four tokens) at SNR 0, 2,
..., 20 dB with decoding seed 777 + 10 SNR, as in pm_models_eval.csv. The
columns ter (= SER), sem_dist (D_u), err_tok_sim, err_close, topic_acc and
gtr_sim are those of pm_models_eval.csv.

Parts (--part):
  ctrl    codebooks
            shuffle  the pwhite codewords permuted among the tokens inside
                     groups of 50 adjacent ranks of p_v (ranks by a stable
                     sort): the codeword set is kept, and each token keeps a
                     codeword of a token of nearly the same frequency, so
                     the SER of each frequency class stays close, but a
                     codeword no longer lies near the codewords of related
                     tokens. The tokens of a group differ in p_v, most in
                     the first groups, so the mean and covariance under p_v
                     change (see the isotropy entries of the JSON).
                     Permutation seeds 1, 2, 3; mask training seed =
                     permutation seed.
            randpw   a Gaussian table of the BERT size (30,522 x 768, entries
                     N(0,1), table seed = draw 1, 2) put through the same
                     projection under p_v (exp_pm.codebook kind "pwhite",
                     rotation seed 0 as for the proposal); mask seed = draw.
            GTR      the shuffle of the GTR pwhite codebook, draws 1, 2, 3 as
                     above, graded on the GTR streams by the BERT topic
                     classifier (a pair that shares no model) and by the GTR
                     similarity.
          They are graded together with the saved references of exp_pm.py
          --part models: pwhite (white_U4_L16_flat_s1..s3 if all three are
          present, else <source>_pwhite_U4_L16_flat_s1..s3), the i.i.d.
          random codebook (<source>_rand_U4_L16_flat_s1) and OMA (16 uncoded
          bits, BPSK per real dimension, out-of-range ids deleted as in the
          paper). The reference rows reproduce pm_models_eval.csv; the JSON
          stores the largest difference. Every row also carries the SER on
          the stream tokens split by the p_v rank of the sent token: the 100
          most frequent token types (ser_top100), ranks 101-5000
          (ser_r101_5000) and ranks above 5000 (ser_gt5000). The JSON stores
          the isotropy of every codebook as exp_pm.py --part plen measures it
          (norm of the mean, largest and smallest eigenvalue of the
          covariance under p_v over the mean eigenvalue), the training time
          and the 10 dB SER (exp_pm.quick_ter) of every trained mask set, the
          share of stream tokens in each frequency class, the split at 10 dB
          per scheme, and for each shuffle the p_v-weighted share of tokens
          whose codeword moved and the mean cosine, in the source table,
          between a token and the token whose codeword it now carries.
  omasub  OMA on the BERT and GTR streams, 0-20 dB in steps of 2 dB, decoding
          seed 777 + 10 SNR, with the received ids outside the vocabulary
          (id >= V) scored three ways: deleted as in the paper (codebook
          "oma", identical to pm_models_eval.csv), replaced by an id drawn
          uniformly over the V ids of the vocabulary ("oma_sub", generator
          seed 5000 + 10 SNR), and replaced by an id drawn uniformly over the
          text tokens, i.e. without the special and [unused] entries
          ("oma_subtext", seed 6000 + 10 SNR). Extra columns: oor_share (out-
          of-range ids among the wrong ids), n_err, n_oor.

Outputs in data/pm/: pm_semctl_eval.csv and pm_semctl_train.json (part
ctrl), pm_semctl_omasub.csv (part omasub). The trained masks go to
data/pm/pm_models/semctl_*.pt (not released). With --smoke (50 iterations,
10 dB, one draw per control) every output goes to data/pm/pm_models_smoke/.

Run from the code folder on a GPU, after exp_pm.py --part models:
    python exp_pm_semctl.py --part ctrl      # 72 min on a GPU shared with other jobs
    python exp_pm_semctl.py --part omasub    # 13 min on a shared GPU
    python exp_pm_semctl.py --part ctrl --smoke
"""
from __future__ import annotations

import argparse
import csv
import json
import time

import torch

import exp_pm as pm
from sse_lib import DEVICE, D, write_csv

USERS = 4
L = 16
GRID = list(range(0, 21, 2))
BIN = 50                                   # tokens per shuffle group (adjacent p_v ranks)
SHUF_DRAWS = (1, 2, 3)
RANDPW_DRAWS = (1, 2)
REF_SEEDS = (1, 2, 3)
CLASSES = (("top100", 0, 100), ("r101_5000", 100, 5000), ("gt5000", 5000, None))
SPLIT_HDR = [f"ser_{n}" for n, _, _ in CLASSES]
OMA_HDR = ["oor_share", "n_err", "n_oor"]


# ----------------------------------------------------------------------
# codebooks and statistics
# ----------------------------------------------------------------------
def isotropy(E: torch.Tensor, pd: torch.Tensor) -> dict:
    """Mean and covariance of the codewords under p_v, as exp_pm.py --part plen."""
    E, pd = E.float().cpu(), pd.float().cpu()
    mu = (pd[:, None] * E).sum(0)
    C = (E * pd[:, None]).T @ E - torch.outer(mu, mu)
    ev = torch.linalg.eigvalsh(C)
    return dict(mean_norm=float(mu.norm()), eig_max_over_mean=float(ev.max() / ev.mean()),
                eig_min_over_mean=float(ev.min() / ev.mean()), trace=float(ev.sum()))


def ranks(pd: torch.Tensor):
    """Tokens sorted by p_v (stable, ties by id) and the rank of every token (0 = most frequent)."""
    order = torch.argsort(pd, descending=True, stable=True)
    rank = torch.empty_like(order)
    rank[order] = torch.arange(len(pd))
    return order, rank


def shuffle_perm(order: torch.Tensor, draw: int, bin_size: int = BIN) -> torch.Tensor:
    """Permutation inside groups of bin_size adjacent p_v ranks: token v
    carries the codeword of token perm[v]."""
    g = torch.Generator().manual_seed(draw)
    perm = torch.arange(len(order))
    for k in range(0, len(order), bin_size):
        blk = order[k:k + bin_size]
        perm[blk] = blk[torch.randperm(len(blk), generator=g)]
    return perm


def gaussian_pwhite(V: int, width: int, pd: torch.Tensor, draw: int) -> torch.Tensor:
    """Gaussian table of the source-table size through the projection of eq. (1) under p_v."""
    g = torch.Generator().manual_seed(draw)
    return pm.codebook(torch.randn(V, width, generator=g), D, "pwhite", pd=pd)


def class_masks(ids: torch.Tensor, rank: torch.Tensor):
    r = rank[ids]
    return [(r >= lo) if hi is None else ((r >= lo) & (r < hi)) for _, lo, hi in CLASSES]


def split_ser(dec, ids, cls, V, unk):
    d2 = dec.clone()
    d2[(d2 < 0) | (d2 >= V)] = unk
    err = d2 != ids
    return [float(err[c].float().mean()) for c in cls]


# ----------------------------------------------------------------------
# mask sets
# ----------------------------------------------------------------------
def retry(fn, *args, tries=30, wait=30):
    """Repeat a call that ran out of GPU memory on a shared GPU. Every call
    wrapped here seeds its own draws (train_masks, dec_mask, dec_oma) or has
    none (Evaluator, scoring), so a repeat returns the same result."""
    for k in range(tries):
        try:
            return fn(*args)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"[oom] waiting {wait} s, retry {k + 1}", flush=True)
            time.sleep(wait)
    return fn(*args)


def load_saved(name: str):
    path = pm.MODELS / f"{name}.pt"
    if not path.exists():
        raise SystemExit(f"missing {path}: run exp_pm.py --part models first")
    sd = torch.load(path, map_location="cpu")
    m = pm.Masks(sd["E"], sd["Wm"].shape[1], USERS)
    m.load_state_dict(sd)
    return m.to(DEVICE).eval()


def train(E, pd, seed, iters, name, tlog, out_dir):
    m, _, tt = retry(pm.train_masks, E, L, USERS, pd, pm.COND["flat"], iters, seed)
    torch.save(m.state_dict(), out_dir / f"{name}.pt")
    tlog[name] = dict(seed=seed, iters=iters, train_s=round(tt, 1),
                      ter10=retry(pm.quick_ter, m, pd.to(DEVICE), pm.COND["flat"], USERS))
    print("[train]", name, round(tt, 1), "s ter10", round(tlog[name]["ter10"], 4), flush=True)
    return m


def pwhite_names(kind: str, seeds) -> list[str]:
    alt = [f"white_U4_L16_flat_s{s}" for s in seeds]
    if kind == "bert" and all((pm.MODELS / f"{n}.pt").exists() for n in alt):
        return alt
    return [f"{kind}_pwhite_U4_L16_flat_s{s}" for s in seeds]


def repro_check(rows) -> dict:
    """Largest difference of the reference rows from the released pm_models_eval.csv."""
    path = pm.DATA / "pm_models_eval.csv"
    if not path.exists():
        return {"available": False}
    ref = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            ref[(r["source"], r["codebook"], int(r["seed"]), int(float(r["snr_db"])))] = r
    cols = ["ter", "sem_dist", "err_tok_sim", "err_close", "topic_acc", "gtr_sim"]
    out = {c: 0.0 for c in cols}
    n = 0
    for row in rows:
        key = (row[1], row[2], int(row[7]), int(row[8]))
        if row[2] in ("pwhite", "rand", "oma") and key in ref:
            n += 1
            for j, c in enumerate(cols):
                out[c] = max(out[c], abs(row[9 + j] - float(ref[key][c])))
    out["rows_compared"] = n
    return out


# ----------------------------------------------------------------------
# parts
# ----------------------------------------------------------------------
def part_ctrl(a):
    out_dir = a.out
    grid = [10] if a.smoke else GRID
    shuf_draws = SHUF_DRAWS[:1] if a.smoke else SHUF_DRAWS
    rp_draws = RANDPW_DRAWS[:1] if a.smoke else RANDPW_DRAWS
    ref_seeds = REF_SEEDS[:1] if a.smoke else REF_SEEDS
    sc = retry(pm.Scorer)
    rows = []
    info = dict(setup=dict(users=USERS, L=L, iters=a.iters, grid_db=grid, bin=BIN,
                           shuffle_draws=list(shuf_draws), randpw_draws=list(rp_draws),
                           ref_seeds=list(ref_seeds), decode_seed="777 + 10 SNR",
                           classes={n: [lo + 1, hi] for n, lo, hi in CLASSES}))
    hdr = pm.HDR + SPLIT_HDR
    for kind in ("bert", "gtr"):
        src = pm.Source(kind)
        pd = src.dist()
        V = src.V
        ev = retry(pm.Evaluator, src, USERS, sc)
        order, rank = ranks(pd)
        cls = class_masks(ev.ids, rank)
        print(f"[clean] {kind} V={V} acc {ev.clean_acc:.4f} frames {ev.ids.shape[0]} "
              f"articles {len(ev.labels)}", flush=True)
        rec = dict(V=V, clean_acc=ev.clean_acc, frames=int(ev.ids.shape[0]), articles=len(ev.labels),
                   class_share={n: float(c.float().mean()) for (n, _, _), c in zip(CLASSES, cls)},
                   train={}, isotropy={}, codebook_check={}, shuffle={})
        Ep = pm.codebook(src.W, D, "pwhite", pd=pd)
        schemes = []                                      # (codebook label, seed, mask set)
        rec["pwhite_models"] = pwhite_names(kind, ref_seeds)
        for s, nm in zip(ref_seeds, rec["pwhite_models"]):
            m = load_saved(nm)
            rec["codebook_check"][nm] = float((m.E.cpu() - Ep).abs().max())
            schemes.append(("pwhite", s, m))
        mr = load_saved(f"{kind}_rand_U4_L16_flat_s1")
        rec["codebook_check"][f"{kind}_rand_U4_L16_flat_s1"] = float(
            (mr.E.cpu() - pm.codebook(src.W, D, "rand")).abs().max())
        schemes.append(("rand", 1, mr))
        rec["isotropy"]["pwhite"] = isotropy(Ep, pd)
        rec["isotropy"]["rand"] = isotropy(mr.E, pd)
        for draw in shuf_draws:
            perm = shuffle_perm(order, draw)
            Es = Ep[perm]
            moved = (perm != torch.arange(V)).float()
            donor = (src.Wn * src.Wn[perm]).sum(1)
            rec["shuffle"][f"d{draw}"] = dict(moved_pv=float((pd * moved).sum()),
                                              donor_cos_pv=float((pd * donor).sum()),
                                              donor_cos_moved_pv=float((pd * moved * donor).sum()
                                                                       / (pd * moved).sum()))
            rec["isotropy"][f"shuffle_d{draw}"] = isotropy(Es, pd)
            m = train(Es, pd, draw, a.iters, f"semctl_{kind}_shuffle_U4_L16_flat_s{draw}",
                      rec["train"], a.models)
            schemes.append(("shuffle", draw, m))
        if kind == "bert":
            for draw in rp_draws:
                Er = gaussian_pwhite(V, src.W.shape[1], pd, draw)
                rec["isotropy"][f"randpw_d{draw}"] = isotropy(Er, pd)
                m = train(Er, pd, draw, a.iters, f"semctl_{kind}_randpw_U4_L16_flat_s{draw}",
                          rec["train"], a.models)
                schemes.append(("randpw", draw, m))
        for k, v in rec["isotropy"].items():
            print("[isotropy]", kind, k, {a_: round(b_, 4) for a_, b_ in v.items()}, flush=True)
        info[kind] = rec
        for snr in grid:
            sd = 777 + int(10 * snr)
            dec = retry(pm.dec_oma, ev.ids, snr, pm.COND["flat"], sd)
            pm.log_row(rows, ["ctrl", kind, "oma", USERS, 0, "-", "flat", 0, snr]
                       + retry(ev.score, dec) + split_ser(dec, ev.ids, cls, V, src.unk))
            for book, s, m in schemes:
                dec = retry(pm.dec_mask, m, ev.ids, snr, pm.COND["flat"], sd)
                pm.log_row(rows, ["ctrl", kind, book, USERS, L, "flat", "flat", s, snr]
                           + retry(ev.score, dec) + split_ser(dec, ev.ids, cls, V, src.unk))
            write_csv(out_dir / "pm_semctl_eval.csv", hdr, rows)
        # SER split at 10 dB per scheme (per seed and mean over seeds)
        rec["split10"] = {}
        for book in ["oma"] + sorted({b for b, _, _ in schemes}):
            sel = [r for r in rows if r[1] == kind and r[2] == book and r[8] == 10]
            if sel:
                cols = ["ter"] + SPLIT_HDR
                idx = [hdr.index(c) for c in cols]
                rec["split10"][book] = dict(seeds=[r[7] for r in sel],
                                            **{c: [r[i] for r in sel] for c, i in zip(cols, idx)},
                                            **{f"{c}_mean": sum(r[i] for r in sel) / len(sel)
                                               for c, i in zip(cols, idx)})
        info["repro_vs_pm_models_eval"] = repro_check(rows)
        json.dump(info, open(out_dir / "pm_semctl_train.json", "w"), indent=1)
        del schemes
        torch.cuda.empty_cache()
    print("[repro]", info["repro_vs_pm_models_eval"], flush=True)


def part_omasub(a):
    out_dir = a.out
    grid = [0, 10] if a.smoke else GRID
    sc = retry(pm.Scorer)
    rows = []
    hdr = pm.HDR + OMA_HDR
    for kind in ("bert", "gtr"):
        src = pm.Source(kind)
        V = src.V
        ev = retry(pm.Evaluator, src, USERS, sc)
        special = set(src.tok.all_special_ids)
        toks = src.tok.convert_ids_to_tokens(list(range(V)))
        text = torch.tensor([i for i, t in enumerate(toks)
                             if i not in special and not t.startswith("[unused")])
        print(f"[clean] {kind} V={V} acc {ev.clean_acc:.4f} text tokens {len(text)}", flush=True)
        for snr in grid:
            sd = 777 + int(10 * snr)
            dec = retry(pm.dec_oma, ev.ids, snr, pm.COND["flat"], sd)
            oor = (dec < 0) | (dec >= V)
            n_err, n_oor = int((dec != ev.ids).sum()), int(oor.sum())
            sub = dec.clone()
            sub[oor] = torch.randint(0, V, (n_oor,), generator=torch.Generator().manual_seed(5000 + 10 * snr))
            subt = dec.clone()
            subt[oor] = text[torch.randint(0, len(text), (n_oor,),
                                           generator=torch.Generator().manual_seed(6000 + 10 * snr))]
            extra = [n_oor / max(1, n_err), n_err, n_oor]
            for book, d in (("oma", dec), ("oma_sub", sub), ("oma_subtext", subt)):
                pm.log_row(rows, ["omasub", kind, book, USERS, 0, "-", "flat", 0, snr]
                           + retry(ev.score, d) + extra)
            write_csv(out_dir / "pm_semctl_omasub.csv", hdr, rows)


def main():
    ap = argparse.ArgumentParser(description="Semantic controls for the pretrained codebook.")
    ap.add_argument("--part", required=True, choices=["ctrl", "omasub"])
    ap.add_argument("--iters", type=int, default=None, help="mask training iterations (3000; 50 with --smoke)")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.iters is None:
        a.iters = 50 if a.smoke else 3000
    # CSV/JSON outputs and trained masks; smoke runs stay apart from the real ones
    a.out = pm.DATA / "pm_models_smoke" if a.smoke else pm.DATA
    a.models = a.out if a.smoke else pm.MODELS
    a.out.mkdir(exist_ok=True)
    t0 = time.time()
    {"ctrl": part_ctrl, "omasub": part_omasub}[a.part](a)
    print("[done]", a.part, round((time.time() - t0) / 60, 1), "min", flush=True)


if __name__ == "__main__":
    main()
