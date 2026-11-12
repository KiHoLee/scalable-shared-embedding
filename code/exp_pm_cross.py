"""Cross points of load, channel and channel knowledge (forward only, no
training).

Every point draws one token per user and frame i.i.d. with the source
statistics p_v (exp_pm_ext.sample), uses d = 64 real dimensions = K = 32
subcarriers and decodes the masked schemes by weighted minimum distance
over all V = 30,522 BERT tokens (codebook pwhite of exp_pm.codebook, fixed at
both ends). Gains follow the T-tap model of Section II (exp_pm_ext.gains,
here with the number of users U as a parameter); T = 1 is flat Rayleigh
fading. Schemes:
  comb         static entry without overlap, L = U: user u weights the real
               dimensions u, u+U, u+2U, ... with sqrt(U), so ||m_u||^2 = d
               (exp_pm_ext.FreeMasks); c is calibrated on 65,536 frames drawn
               with seed 0 (exp_pm_ext.calibrate_rho)
  adapted      masks formed at every update from the gains known at the
               transmitter (adaptive_masks, copied from exp_pm_ext with U
               taken from the gains): J = 16 blocks of B = 2 subcarriers, the
               users take turns in an order reversed every round, and each
               sets the level sqrt(U) on the block with its largest sum of
               log(1 + SNR) among those that no mask weights yet (eight
               blocks per mask at U = 2, four at U = 4, two at U = 8); the c
               of the comb
  oma          uncoded OMA on adjacent dimensions: user u owns the real
               dimensions [u d/U, (u+1) d/U) and sends the 16 bits of its token
               id with the layout of rev_channel.oma_layout (U = 2: BPSK, each
               bit twice on the two real dimensions of one subcarrier; U = 4:
               BPSK per real dimension, i.e. QPSK on eight adjacent
               subcarriers; U = 8: natural-mapped 4-PAM), weighted combining
               and nearest-level detection as exp_pm.dec_oma; frame energy one
               at U = 2 and 4, about 1.22 at U = 8 (top bit of the id is zero)
  oma_adapted  the same OMA on the blocks that each adapted mask weights
               (exp_pm_chanhi.run_oma generalized to U)

Channel knowledge (columns tx_nmse, rx_nmse, age_rho):
  rx_nmse  receiver u holds g_hat = g + e, e ~ CN(0, rx_nmse) relative to the
           unit gain power, common to all subcarriers under flat fading and
           independent across subcarriers otherwise (Section II,
           rev_channel.channel); it zero-forces by g_hat and weights by
           |g_hat|^2
  tx_nmse  the transmitter forms the adapted masks (and the OMA blocks) from
           g + e', e' ~ CN(0, tx_nmse) drawn like e (independent across
           subcarriers at T = 4 and 8) and independent of the receiver's
           error; tx_nmse = 0 for comb and oma, which use no gains at the
           transmitter
  age_rho  the masks are formed from the gains g_old of the last update and
           the frame sees g = age_rho g_old + sqrt(1 - age_rho^2) g_new, g_new
           an independent draw of the same tap model; the receiver knows g
  pilot    NMSE = 1/SNR at both ends: the least-squares estimate from one
           pilot symbol per subcarrier with the energy of a data symbol
           (2/d per subcarrier against noise CN(0, 2/(d SNR)))

Parts (--part):
  load   U in {2, 4, 8}, T in {4, 8}, SNR in {0, 5, 10, 15, 20} dB, exact gains
         at both ends: comb, adapted, oma, oma_adapted; flat fading (T = 1)
         for comb and oma as a control against pm_period_eval.csv
  csi    U in {4, 8}, T in {4, 8}, SNR in {10, 20} dB: comb and oma with
         rx_nmse in {0.01, 0.05, 0.1}; adapted and oma_adapted with tx_nmse in
         {0.01, 0.05, 0.1} (exact receiver), with the same NMSE at both ends
         (independent errors), and with age_rho in {0.95, 0.9, 0.7}. The pilot
         estimate NMSE = 1/SNR at both ends is added at 0, 5 and 15 dB for all
         four schemes; at 10 and 20 dB it equals the rows with NMSE 0.1 and
         0.01 at both ends. The exact-gain reference of every csi point is the
         load row with the same U, T and SNR (same seeds).
  merge  concatenates the outputs of load and csi
Frames per point: 40,000 up to 10 dB and 200,000 from 15 dB; 400,000 at
every SNR for U = 2, whose errors are rare.

Seeds: comb and adapted share torch.manual_seed(31 + 10 snr + 100 T + 7 B),
B = 2, the seed of exp_pm_ext.run_adapt, and draw tokens, gains and noise
from it in the order of run_adapt (384 frames per chunk), so at U = 4 the
adapted rows with exact gains repeat pm_adapt.csv up to the draw of c; oma
and oma_adapted use 61 + 10 snr + 100 T (+1000 on the adapted blocks), the
seed of exp_pm_chanhi.run_oma (16,384 frames per chunk: tokens, gains,
noise). The estimation errors and the new gains of the aging come from a
second generator seeded with the same value + 100,000, so all CSI settings
of one point see the same tokens, gains (g_old under aging) and noise. The
constant c: seed 0. Run time on one RTX PRO 5000 shared with other jobs:
about 8 min (load) and 13 min (csi).

Outputs in data/pm/: pm_cross_load.{csv,json}, pm_cross_csi.{csv,json}, and
after --part merge pm_cross.csv (columns part, scheme, users, taps, snr_db,
tx_nmse, rx_nmse, age_rho, frames, ser) and pm_cross.json (c per load and,
for every row, the SER of each user, the error count and the measured mean
frame energy). The part files are rewritten after every point; --resume
skips the points already stored there. --smoke runs 768 frames per point and
writes *_smoke files.

Run from the code folder (GPU chosen by sse_lib.DEVICE):
    python exp_pm_cross.py --part load
    python exp_pm_cross.py --part csi
    python exp_pm_cross.py --part merge
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
from sse_lib import DEVICE, D, write_csv

K = D // 2
B = 2                                   # subcarriers per block of the adapted masks
SNRS = (0, 5, 10, 15, 20)
NMSES = (0.01, 0.05, 0.1)
AGES = (0.95, 0.9, 0.7)
EXTRA = 100_000                         # offset of the second generator's seed
SUPER = 40                              # chunks of 384 frames per call of adaptive_masks
HDR = ["part", "scheme", "users", "taps", "snr_db", "tx_nmse", "rx_nmse", "age_rho", "frames", "ser"]


# ----------------------------------------------------------------------
# channel (exp_pm_ext.gains with U as a parameter and an optional generator)
# ----------------------------------------------------------------------
def cn(shape, gen=None):
    return torch.complex(torch.randn(*shape, device=DEVICE, generator=gen),
                         torch.randn(*shape, device=DEVICE, generator=gen)) / math.sqrt(2)


def gains(n, users, taps, gen=None):
    """Subcarrier gains (n,U,K): T equal-power taps a_t ~ CN(0, 1/T) and their
    DFT; one CN(0,1) gain on all subcarriers for T = 1."""
    if taps == 1:
        return cn((n, users, 1), gen).expand(n, users, K).contiguous()
    a = cn((n, users, taps), gen) / math.sqrt(taps)
    k = torch.arange(K, device=DEVICE)
    t = torch.arange(taps, device=DEVICE)
    F = torch.exp(-2j * math.pi * torch.outer(t, k).float() / K)
    return a @ F.to(a.dtype)


def est_error(n, users, taps, nmse, gen):
    """CN(0, nmse) estimation error, common to all subcarriers for T = 1 and
    independent across them otherwise."""
    if taps == 1:
        return cn((n, users, 1), gen).expand(n, users, K) * math.sqrt(nmse)
    return cn((n, users, K), gen) * math.sqrt(nmse)


def noise(n, users, sig):
    """CN(0, 2 sig^2) noise (n,U,K) from the default generator, as run_adapt."""
    return torch.complex(torch.randn(n, users, K, device=DEVICE),
                         torch.randn(n, users, K, device=DEVICE)) * sig


def receive(x, g, w, rx_nmse, taps, gen):
    """x (n,d) real frames, g (n,U,K) true gains, w (n,U,K) noise. Returns r
    (n,U,d) after zero forcing by the receiver estimate and the weights
    |g_hat|^2 (n,U,d)."""
    n, users, _ = g.shape
    xc = torch.complex(x[:, 0::2], x[:, 1::2])
    gh = g if rx_nmse == 0 else g + est_error(n, users, taps, rx_nmse, gen)
    q = (g * xc[:, None, :] + w) / gh
    r = torch.stack([q.real, q.imag], -1).reshape(n, users, D)
    wt = (gh.abs() ** 2).repeat_interleave(2, dim=-1)
    return r, wt


def tx_gains(g_old, taps, tx_nmse, gen):
    """Gains known at the transmitter: g_old plus an estimation error."""
    if tx_nmse == 0:
        return g_old
    n, users, _ = g_old.shape
    return g_old + est_error(n, users, taps, tx_nmse, gen)


def aged(g_old, taps, age_rho, gen):
    if age_rho >= 1.0:
        return g_old
    n, users, _ = g_old.shape
    return age_rho * g_old + math.sqrt(1.0 - age_rho ** 2) * gains(n, users, taps, gen)


# ----------------------------------------------------------------------
# masks
# ----------------------------------------------------------------------
def comb_masks(users):
    M = torch.zeros(users, D, device=DEVICE)
    for u in range(users):
        M[u, u::users] = math.sqrt(users)            # ||m_u||^2 = d
    return M


def calib_c(E, pd, users):
    m = px.FreeMasks(E, comb_masks(users)).to(DEVICE)
    torch.manual_seed(0)                              # fixed draw for the constant c
    px.calibrate_rho(m, pd, 0.0)
    return float(m.c)


@torch.no_grad()
def adaptive_masks(g, B, sig2, c, wf=False):
    """exp_pm_ext.adaptive_masks (U taken from g): per-frame masks from the
    gains g (n,U,K). The band is cut into units of B subcarriers. The users
    take turns (reversing the order every round), and each sets its weight
    on the unit with its largest sum of log(1+SNR) among those that no mask
    weights yet, so every mask has weight on K/(B U) units. With wf, each
    mask holds real weights water-filled over these units; otherwise it
    holds one level on all of them, ||m_u||^2 = d."""
    n, Uu, Kk = g.shape
    nu = Kk // B
    per = nu // Uu
    g2 = g.abs() ** 2
    gam = 1.0 / (D * sig2)
    util = torch.log1p(gam * g2).view(n, Uu, nu, B).sum(-1)        # (n,U,nu)
    taken = torch.zeros(n, nu, dtype=torch.bool, device=g.device)
    owner = torch.full((n, nu), -1, dtype=torch.long, device=g.device)
    ar = torch.arange(n, device=g.device)
    for r in range(per):
        order = range(Uu) if r % 2 == 0 else range(Uu - 1, -1, -1)
        for u in order:
            k = util[:, u].masked_fill(taken, -1e9).argmax(1)
            taken[ar, k] = True
            owner[ar, k] = u
    M = torch.zeros(n, Uu, nu, device=g.device)
    if not wf:
        M[ar[:, None], owner, torch.arange(nu, device=g.device)[None]] = 1.0
        M = M * (D / (2 * B * per))                                  # weight^2 per dim
    else:
        mg = g2.view(n, Uu, nu, B).mean(-1)
        for u in range(Uu):
            pos = (owner == u).float().argsort(-1, descending=True)[:, :per]
            inv = sig2 * c * c / mg[:, u].gather(1, pos).clamp_min(1e-12)
            M[ar[:, None], u, pos] = px.waterfill(inv, D / (2 * B))
    return M.repeat_interleave(2 * B, dim=-1).sqrt()                 # (n,U,d)


# ----------------------------------------------------------------------
# evaluators; both return (per-user SER list, error count, mean frame energy)
# ----------------------------------------------------------------------
@torch.no_grad()
def run_adapt(E, pd, c, users, snr, taps, scheme, frames, tx_nmse=0.0, rx_nmse=0.0,
              age_rho=1.0, seed=31, chunk=384):
    """exp_pm_ext.run_adapt with U as a parameter, the comb as the static
    alternative, transmitter and receiver estimates, aging, per-user SER. The
    two terms of the weighted minimum-distance score (pm.Masks.scores) are
    computed as one fp32 matrix product to halve the GPU memory. Tokens,
    gains and noise are drawn chunk by chunk (384 frames) in the order of
    run_adapt; the masks are formed for SUPER chunks at once, which leaves
    the draws unchanged and saves the launch latency of adaptive_masks."""
    assert scheme in ("comb", "adapted")
    s = seed + int(10 * snr) + 100 * taps + 7 * B
    torch.manual_seed(s)
    gen = torch.Generator(device=DEVICE).manual_seed(EXTRA + s)
    sig = math.sqrt(1.0 / (D * 10 ** (snr / 10)))
    comb = comb_masks(users)
    EE = torch.cat([E, E * E], -1)                                  # (V,2d)
    a_ = 1.0 / c
    err = torch.zeros(users, dtype=torch.long, device=DEVICE)
    energy = 0.0
    for k0 in range(0, frames, chunk * SUPER):
        draws = []
        for k1 in range(k0, min(k0 + chunk * SUPER, frames), chunk):
            n = min(chunk, frames - k1)
            draws.append((px.sample(pd, n, users, 0.0), gains(n, users, taps), noise(n, users, sig)))
        idx, g_old, w = (torch.cat(z) for z in zip(*draws))
        nb = idx.shape[0]
        if scheme == "comb":
            M = comb[None].expand(nb, users, D)
        else:
            M = adaptive_masks(tx_gains(g_old, taps, tx_nmse, gen), B, sig * sig, c)
        g = aged(g_old, taps, age_rho, gen)
        x = (E[idx] * M).sum(1) / c
        energy += float(x.pow(2).sum())
        r, wt = receive(x, g, w, rx_nmse, taps, gen)
        A = torch.cat([2 * a_ * wt * r * M, -a_ * a_ * wt * M * M], -1)  # (nb,U,2d)
        for j in range(0, nb, chunk):
            sc = (A[j:j + chunk].reshape(-1, 2 * D) @ EE.T).view(-1, users, EE.shape[0])
            err += (sc.argmax(-1) != idx[j:j + chunk]).sum(0)
    err = err.cpu()
    return (err.double() / frames).tolist(), int(err.sum()), energy / frames


@torch.no_grad()
def run_oma(pd, users, snr, taps, adapt, frames, tx_nmse=0.0, rx_nmse=0.0, age_rho=1.0,
            seed=61, chunk=16384):
    """exp_pm_chanhi.run_oma with U as a parameter: the 16 bits of each token
    id (tokens drawn with p_v) uncoded on the d/U real dimensions of the user,
    with the layout of rev_channel.oma_layout, on adjacent dimensions or on
    the blocks of the adapted masks (formed from the transmitter's gains);
    weighted combining over repetitions and nearest-level detection."""
    s = seed + int(10 * snr) + 100 * taps + (1000 if adapt else 0)
    torch.manual_seed(s)
    gen = torch.Generator(device=DEVICE).manual_seed(EXTRA + s)
    sig = math.sqrt(1.0 / (D * 10 ** (snr / 10)))
    n_sym, M, rep = rc.oma_layout(users, D, pm.BITS)
    bps = int(math.log2(M))
    lv = rc.pam_levels(M, DEVICE)
    nd = D // users
    adj = (torch.arange(users, device=DEVICE)[:, None] * nd
           + torch.arange(nd, device=DEVICE)[None])                      # (U,nd)
    err = torch.zeros(users, dtype=torch.long, device=DEVICE)
    energy = 0.0
    for k0 in range(0, frames, chunk):
        b = min(chunk, frames - k0)
        ids = px.sample(pd, b, users, 0.0)
        sym = torch.stack([(ids >> (bps * k)) & (M - 1) for k in range(n_sym)], -1)
        sv = lv[sym].repeat_interleave(rep, dim=-1) / math.sqrt(D)       # (b,U,nd)
        g_old = gains(b, users, taps)
        w = noise(b, users, sig)
        if adapt:
            own = adaptive_masks(tx_gains(g_old, taps, tx_nmse, gen), B, sig * sig, 1.0) > 0
            pos = torch.sort(own.float(), dim=-1, descending=True, stable=True).indices[..., :nd].sort(-1).values
        else:
            pos = adj[None].expand(b, users, nd)
        g = aged(g_old, taps, age_rho, gen)
        x = torch.zeros(b, D, device=DEVICE)
        x.scatter_(1, pos.reshape(b, -1), sv.reshape(b, -1))
        energy += float(x.pow(2).sum())
        r, wt = receive(x, g, w, rx_nmse, taps, gen)
        rb = r.gather(2, pos).view(b, users, n_sym, rep)
        wb = wt.gather(2, pos).view(b, users, n_sym, rep)
        est = (rb * wb).sum(-1) / wb.sum(-1).clamp_min(1e-12) * math.sqrt(D)
        sh = (est[..., None] - lv).abs().argmin(-1)
        err += (sh != sym).any(-1).sum(0)
    err = err.cpu()
    return (err.double() / frames).tolist(), int(err.sum()), energy / frames


# ----------------------------------------------------------------------
# parts
# ----------------------------------------------------------------------
def setup():
    torch.backends.cuda.matmul.allow_tf32 = False      # evaluation in full fp32
    torch.backends.cudnn.allow_tf32 = False
    src = pm.Source("bert")
    pdc = src.dist()
    E = pm.codebook(src.W, D, "pwhite", pd=pdc).to(DEVICE)
    return E, pdc.to(DEVICE)


def frames_for(users, snr, smoke):
    if smoke:
        return 768
    if users == 2:
        return 400_000
    return 40_000 if snr <= 10 else 200_000


def key(scheme, users, taps, snr, tx, rx, age, frames):
    return (scheme, int(users), int(taps), int(snr), float(tx), float(rx), float(age), int(frames))


class Log:
    """Collects the rows of one part and rewrites its outputs after every
    point; with --resume, points already in the part's JSON are taken from
    it instead of being run again (same seeds, same result)."""

    def __init__(self, part, cs, sfx, resume):
        self.part, self.rows, self.info, self.t0 = part, [], [], time.time()
        self.cs = {str(k): v for k, v in cs.items()}
        self.csv = pm.DATA / f"pm_cross_{part}{sfx}.csv"
        self.json = pm.DATA / f"pm_cross_{part}{sfx}.json"
        self.done = {}
        if resume and self.json.exists():
            for r in json.load(open(self.json))["rows"]:
                self.done[key(r["scheme"], r["users"], r["taps"], r["snr_db"], r["tx_nmse"],
                              r["rx_nmse"], r["age_rho"], r["frames"])] = r
            print(f"[resume] {len(self.done)} points from {self.json.name}", flush=True)

    def save(self):
        write_csv(self.csv, HDR, self.rows)
        json.dump({"c": self.cs, "rows": self.info}, open(self.json, "w"), indent=1)

    def point(self, E, pd, c, scheme, users, taps, snr, frames, tx=0.0, rx=0.0, age=1.0):
        t = time.time()
        old = self.done.get(key(scheme, users, taps, snr, tx, rx, age, frames))
        if old is not None:
            self.rows.append([self.part, scheme, users, taps, snr, float(tx), float(rx), float(age),
                              frames, old["ser"]])
            self.info.append(old)
            return
        if scheme in ("comb", "adapted"):
            pu, ne, en = run_adapt(E, pd, c, users, snr, taps, scheme, frames, tx, rx, age)
        else:
            pu, ne, en = run_oma(pd, users, snr, taps, scheme == "oma_adapted", frames, tx, rx, age)
        ser = ne / (frames * users)
        self.rows.append([self.part, scheme, users, taps, snr, float(tx), float(rx), float(age), frames, ser])
        self.info.append(dict(part=self.part, scheme=scheme, users=users, taps=taps, snr_db=snr,
                              tx_nmse=tx, rx_nmse=rx, age_rho=age, frames=frames, ser=ser,
                              errors=ne, ser_user=pu, energy=en))
        print(f"[cross] {self.part} {scheme:11s} U={users} T={taps} {snr:2d} dB tx={tx:.4g} rx={rx:.4g} "
              f"age={age:g} frames={frames} ser={ser:.4e} err={ne} energy={en:.4f} "
              f"({time.time() - t:.0f} s, total {(time.time() - self.t0) / 60:.1f} min)", flush=True)
        self.save()


def part_load(a, E, pd, lg, cs):
    for users in (2, 4, 8):
        c = cs[users]
        for taps in (1, 4, 8):
            schemes = ("comb", "oma") if taps == 1 else ("comb", "adapted", "oma", "oma_adapted")
            for snr in SNRS:
                fr = frames_for(users, snr, a.smoke)
                for sch in schemes:
                    lg.point(E, pd, c, sch, users, taps, snr, fr)


def part_csi(a, E, pd, lg, cs):
    for users in (4, 8):
        c = cs[users]
        for taps in (4, 8):
            for snr in (10, 20):
                fr = frames_for(users, snr, a.smoke)
                for nm in NMSES:                                  # receiver estimate only
                    lg.point(E, pd, c, "comb", users, taps, snr, fr, rx=nm)
                    lg.point(E, pd, c, "oma", users, taps, snr, fr, rx=nm)
                for sch in ("adapted", "oma_adapted"):
                    for nm in NMSES:                              # transmitter estimate only
                        lg.point(E, pd, c, sch, users, taps, snr, fr, tx=nm)
                    for nm in NMSES:                              # the same NMSE at both ends
                        lg.point(E, pd, c, sch, users, taps, snr, fr, tx=nm, rx=nm)
                    for rho in AGES:                              # aged gains at the transmitter
                        lg.point(E, pd, c, sch, users, taps, snr, fr, age=rho)
            for snr in (0, 5, 15):                                # pilot estimate, NMSE = 1/SNR
                fr = frames_for(users, snr, a.smoke)
                nm = 10 ** (-snr / 10)
                lg.point(E, pd, c, "comb", users, taps, snr, fr, rx=nm)
                lg.point(E, pd, c, "oma", users, taps, snr, fr, rx=nm)
                lg.point(E, pd, c, "adapted", users, taps, snr, fr, tx=nm, rx=nm)
                lg.point(E, pd, c, "oma_adapted", users, taps, snr, fr, tx=nm, rx=nm)


def part_merge(a):
    sfx = "_smoke" if a.smoke else ""
    rows, info = [], {"c": {}, "rows": []}
    for part in ("load", "csi"):
        with open(pm.DATA / f"pm_cross_{part}{sfx}.csv") as f:
            rd = csv.reader(f)
            assert next(rd) == HDR
            rows += list(rd)
        js = json.load(open(pm.DATA / f"pm_cross_{part}{sfx}.json"))
        info["c"].update(js["c"])
        info["rows"] += js["rows"]
    write_csv(pm.DATA / f"pm_cross{sfx}.csv", HDR, rows)
    json.dump(info, open(pm.DATA / f"pm_cross{sfx}.json", "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["load", "csi", "merge"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    if a.part == "merge":
        part_merge(a)
    else:
        E, pd = setup()
        cs = {users: calib_c(E, pd, users) for users in ((2, 4, 8) if a.part == "load" else (4, 8))}
        print("[c]", cs, flush=True)
        lg = Log(a.part, cs, "_smoke" if a.smoke else "", a.resume)
        {"load": part_load, "csi": part_csi}[a.part](a, E, pd, lg, cs)
        lg.save()
    print("[done]", a.part, round((time.time() - t0) / 60, 1), "min", flush=True)


if __name__ == "__main__":
    main()
