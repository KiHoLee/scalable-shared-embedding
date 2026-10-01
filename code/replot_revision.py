"""Replot script for the revised letter. Reuses the geometry, style
dictionary, and save() of replot.py and writes the revised figures with
the _revision suffix (the original figures stay untouched):

  fig_conv_revision.pdf : SER at 10 dB versus cumulative training MACs,
                          Proposed SSE, Full codebook, Sampled softmax,
                          Hierarchical softmax, 200,000 iterations each
                          (rev_budget_curve_*.csv), ToDMA/OMA references
  fig_real_revision.pdf : topic-classification accuracy versus SNR on the
                          received AG News items (rev_semantic.csv)
  fig_users_revision.pdf: SER at 10 dB versus number of users U
                          (rev_users.csv)
"""
from __future__ import annotations

import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullFormatter

import replot as rp

# Line style carries the role: solid for the proposal, dash-dot variants
# for comparison schemes, dotted for reference levels (union bound,
# error-free text). Colors and markers are those of replot.py.
rp.STY.update({
    "Full":    ("#2c5fa8", "s", (0, (5, 1.5, 1, 1.5)), "Full codebook"),
    "Sampled": ("#d68910", "D", (0, (3, 1, 1, 1, 1, 1)), "Sampled softmax"),
    "Hier":    ("#1e8449", "h", (0, (6, 1.5, 1, 1.5, 1, 1.5)),
                "Hierarchical softmax"),
    "OMA":     ("#7f8c8d", "^", (0, (1.5, 1, 1.5, 1, 4, 1)), "OMA"),
    "UB":      ("#c0392b", "",  (0, (1, 1)), "Union bound"),
    "Clean":   ("#000000", "",  (0, (1, 1)), "Error-free text"),
})
# 8:6 axes box on the shared 8:6 canvas (width 0.79, height 0.79)
rp.AXES_RECT = dict(left=0.175, right=0.965, top=0.962, bottom=0.172)


def save(fig, name):
    """rp.save with two enforced checks (skill 12.7): every axis label
    lies inside the canvas, and no curve, marker, or segment between
    markers enters a legend box."""
    fig.subplots_adjust(**rp.AXES_RECT)
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    F = fig.bbox
    for ax in fig.axes:
        for t in (ax.xaxis.label, ax.yaxis.label):
            b = t.get_window_extent(r)
            if b.x0 < F.x0 or b.y0 < F.y0 or b.x1 > F.x1 or b.y1 > F.y1:
                raise RuntimeError(f"{name}: clipped label {t.get_text()}")
        lg = ax.get_legend()
        if lg is None:
            continue
        L = lg.get_window_extent(r).expanded(1.02, 1.04)
        for ln in ax.lines:
            P = ax.transData.transform(ln.get_xydata())
            P = P[np.isfinite(P).all(1)]
            if len(P) < 2:
                continue
            s = np.linspace(0, 1, 60)[:, None, None]
            pts = (P[:-1][None] + (P[1:] - P[:-1])[None] * s).reshape(-1, 2)
            inside = ((pts[:, 0] > L.x0) & (pts[:, 0] < L.x1)
                      & (pts[:, 1] > L.y0) & (pts[:, 1] < L.y1))
            if inside.any():
                raise RuntimeError(f"{name}: legend overlaps {ln.get_label()}")
    fig.savefig(rp.FIG / f"{name}.pdf")
    plt.close(fig)
    print("[OK]", name)


def logmarks(x, n=8):
    """indices of about n markers evenly spaced on a log axis"""
    tgt = np.logspace(np.log10(x[0]), np.log10(x[-1]), n)
    return sorted(set(np.clip(np.searchsorted(x, tgt), 0, len(x) - 1).tolist()))
ORDER = ["SSE", "Full", "Sampled", "Hier", "ToDMA", "OMA", "Single", "UB",
         "Clean"]


def read(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def legend(ax, **kw):
    h, l = ax.get_legend_handles_labels()
    lab2key = {v[3]: k for k, v in rp.STY.items()}
    idx = sorted(range(len(l)), key=lambda i: ORDER.index(lab2key[l[i]]))
    ax.legend([h[i] for i in idx], [l[i] for i in idx],
              labelspacing=0.25, borderpad=0.3, **kw)


def fig_conv():
    fig, ax = plt.subplots()
    for key, fn in (("SSE", "sse"), ("Full", "full"), ("Sampled", "sampled"),
                    ("Hier", "hier")):
        rows = read(rp.DATA / f"rev_budget_curve_{fn}.csv")
        x = np.array([float(r["macs"]) for r in rows])
        # full-codebook variants are decoded by minimum distance
        col = "ser10_md" if "ser10_md" in rows[0] else "ser10"
        y = np.array([float(r[col]) for r in rows])
        c, mk, ls, lb = rp.STY[key]
        ax.loglog(x, y, linestyle=ls, marker=mk, color=c, label=lb,
                  markevery=logmarks(x))
    snr = np.loadtxt(rp.DATA / "snr_sweep.csv", delimiter=",", skiprows=1)
    h_snr = rp.cols(rp.DATA / "snr_sweep.csv")
    oma10 = snr[snr[:, 0] == 10, h_snr.index("ser_OMA")][0]
    sota = np.loadtxt(rp.DATA / "sota_snr.csv", delimiter=",", skiprows=1)
    h_sota = rp.cols(rp.DATA / "sota_snr.csv")
    td10 = sota[sota[:, 0] == 10, h_sota.index("ser_ToDMA")][0]
    for key, v in (("ToDMA", td10), ("OMA", oma10)):
        c, mk, ls, lb = rp.STY[key]
        xs = np.logspace(6.2, 14.8, 7)
        ax.plot(xs, np.full_like(xs, v), linestyle=ls, marker=mk, color=c,
                label=lb)
    ax.set_xlabel("Training MACs")
    ax.set_ylabel("SER")
    ax.set_xlim(8e5, 2e15)
    ax.set_ylim(0.18, 3.2)
    from matplotlib.ticker import FixedLocator
    ax.yaxis.set_minor_locator(FixedLocator([0.4, 0.6, 0.7, 0.8, 0.9]))
    ax.xaxis.labelpad = 1
    ax.set_yticks([0.2, 0.3, 0.5, 1])
    ax.set_yticklabels(["0.2", "0.3", "0.5", "1"])
    ax.yaxis.set_minor_formatter(NullFormatter())
    legend(ax, loc="upper right", ncol=2, columnspacing=0.8)
    save(fig, "fig_conv_revision")


def fig_real():
    rows = read(rp.DATA / "rev_semantic.csv")
    import json
    en = json.load(open(rp.DATA / "rev_energy.json"))
    snr = [float(r["snr_db"]) for r in rows]
    fig, ax = plt.subplots()
    for key in ("SSE", "Full", "ToDMA", "OMA"):
        c, mk, ls, lb = rp.STY[key]
        ax.plot(snr, [float(r[f"acc_{key}"]) for r in rows], linestyle=ls,
                marker=mk, color=c, label=lb)
    c, _, ls, lb = rp.STY["Clean"]
    ax.axhline(en["clean_acc"], linestyle=ls, color=c, linewidth=1.0, label=lb)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Accuracy")
    ax.set_xlim(-0.6, 20.6)
    ax.set_xticks(range(0, 21, 4))
    ax.set_ylim(0.2, 1.0)
    legend(ax, loc="lower right")
    save(fig, "fig_real_revision")


def fig_users():
    rows = [r for r in read(rp.DATA / "rev_users.csv")
            if float(r["snr_db"]) == 10.0]
    U = [int(r["U"]) for r in rows]
    fig, ax = plt.subplots()
    for key in ("SSE", "Full", "ToDMA", "OMA"):
        c, mk, ls, lb = rp.STY[key]
        ax.semilogy(U, [float(r[f"ser_{key}"]) for r in rows], linestyle=ls,
                    marker=mk, color=c, label=lb)
    ax.set_xscale("log", base=2)
    ax.set_xticks(U)
    ax.set_xticklabels([str(u) for u in U])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("Number of users $U$")
    ax.set_ylabel("SER")
    legend(ax, loc="lower right")
    save(fig, "fig_users_revision")


def fig_snr():
    """Fig. 2 revised: the full codebook is decoded by minimum distance
    (rev_snr_fullmd.csv); the single-user reference is omitted; all other
    curves as in the original figure."""
    p = rp.DATA / "snr_sweep.csv"
    hdr = rp.cols(p)
    d = np.loadtxt(p, delimiter=",", skiprows=1)
    ds = np.loadtxt(rp.DATA / "sota_snr.csv", delimiter=",", skiprows=1)
    hs = rp.cols(rp.DATA / "sota_snr.csv")
    fm = np.loadtxt(rp.DATA / "rev_snr_fullmd.csv", delimiter=",", skiprows=1)
    snr = d[:, 0]
    series = {m: d[:, hdr.index(f"ser_{m}")] for m in ["SSE", "OMA"]}
    series["ToDMA"] = ds[:, hs.index("ser_ToDMA")]
    series["Full"] = fm[:, 2]
    fig, ax = plt.subplots()
    for m in ["SSE", "Full", "ToDMA", "OMA"]:
        c, mk, ls, lb = rp.STY[m]
        ax.semilogy(snr, series[m], linestyle=ls, marker=mk, color=c, label=lb)
    c, mk, ls, lb = rp.STY["UB"]
    ax.semilogy(snr, d[:, hdr.index("ser_UB")], linestyle=ls, color=c,
                alpha=0.55, linewidth=1.0, label=lb)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("SER")
    ax.set_xlim(-0.6, 20.6)
    ax.set_ylim(2e-2, 1.15)
    ax.set_xticks(range(0, 21, 4))
    legend(ax, loc="lower left", ncol=1, handlelength=2.2)
    save(fig, "fig_snr_revision")


def fig_scaling():
    """Fig. 3 revised: Proposed SSE with the best structured factorization
    P in {2, 4} (scaling_sse_P.csv), full codebook decoded by minimum
    distance (rev_scaling_fullmd.csv), ToDMA and OMA from scaling.csv."""
    p = rp.DATA / "scaling.csv"
    hdr = rp.cols(p)
    d = np.loadtxt(p, delimiter=",", skiprows=1)
    sp = read(rp.DATA / "scaling_sse_P.csv")
    fm = read(rp.DATA / "rev_scaling_fullmd.csv")
    V = d[:, 0]
    x = np.log2(V)
    sse = [min(float(r["ser10_P2"]), float(r["ser10_P4"])) for r in sp]
    full = [float(r["ser10_md"]) for r in fm]
    assert [int(float(r["V"])) for r in sp] == [int(v) for v in V]
    assert [int(float(r["V"])) for r in fm] == [int(v) for v in V]
    fig, ax = plt.subplots()
    series = {"SSE": sse, "Full": full,
              "ToDMA": d[:, hdr.index("ser10_ToDMA")],
              "OMA": d[:, hdr.index("ser10_OMA")]}
    for m in ["SSE", "Full", "ToDMA", "OMA"]:
        c, mk, ls, lb = rp.STY[m]
        ax.semilogy(x, series[m], linestyle=ls, marker=mk, color=c, label=lb)
    ax.set_xlabel(r"$\log_2 V$")
    ax.set_ylabel("SER")
    ax.set_xticks(range(4, 17, 2))
    ax.set_xlim(3.6, 16.4)
    ax.xaxis.labelpad = 0
    ax.set_yticks([0.03, 0.1, 0.3, 0.6])
    ax.set_yticklabels(["0.03", "0.1", "0.3", "0.6"])
    ax.yaxis.set_minor_formatter(NullFormatter())
    legend(ax, loc="lower right")
    save(fig, "fig_scaling_revision")


def fig_norms():
    """Letter Fig. R3: distribution of the masked candidate norms of the
    proposal relative to the per-user mean (rev_norms.csv, 64 values),
    with the full-codebook range (rev_norms_full.json) marked by its two
    extremes on the same axis."""
    import json
    rows = read(rp.DATA / "rev_norms.csv")
    rel = np.array([float(r["norm_rel"]) for r in rows])
    fj = json.load(open(rp.DATA / "rev_norms_full.json"))
    fig, ax = plt.subplots()
    c, _, _, lb = rp.STY["SSE"]
    ax.hist(rel, bins=np.linspace(0.95, 1.03, 17), color=c, alpha=0.85,
            edgecolor="white", linewidth=0.4, label=lb)
    cf, _, ls, lbf = rp.STY["Full"]
    for k, v in enumerate((fj["rel_min"], fj["rel_max"])):
        ax.axvline(v, color=cf, linestyle=ls, linewidth=1.3,
                   label=f"{lbf} extremes" if k == 0 else None)
    ax.set_xlabel("Masked norm / per-user mean")
    ax.set_ylabel("Count")
    ax.set_xlim(0.3, 1.55)
    ax.set_xticks([0.4, 0.6, 0.8, 1.0, 1.2, 1.4])
    ax.legend(loc="upper left", labelspacing=0.25, borderpad=0.3)
    save(fig, "fig_norms_revision")


if __name__ == "__main__":
    import sys
    for name in (sys.argv[1:] or ["snr", "scaling", "conv", "real"]):
        globals()[f"fig_{name}"]()
