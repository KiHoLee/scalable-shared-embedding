"""Figures of the letter on periodic masks for shared-embedding multiple
access, from the CSV files in data/pm/ that code/exp_pm.py,
code/exp_pm_ext.py and code/exp_pm_chanhi.py write. Plotting only, no
simulation. Figs. 2-5 share one 8:6 canvas with the same axes box and keep
their legends inside the axes; save() refuses a figure whose legend covers
a curve, a marker, a reference line or an annotation, or whose text leaves
the canvas. One style per scheme (STYLE): the static entries of the mask
set (trained masks) are blue circles, adapted masks red squares, OMA black
diamonds, masks without overlap orange down-triangles and full overlap
purple triangles; a dashed line with open markers marks the second variant
of the same scheme.

  fig/fig2_pm.pdf : channel step. SER versus SNR at T = 4 and 8 taps for
                    OMA, the static entry trained for the class
                    (pm_figs_class.csv; exp_pm_chanhi.py --part class), and
                    masks adapted to the known gains (pm_figs.csv graph g1,
                    exp_pm_ext.py --part figs, with the 400,000-frame
                    points of pm_figs_hi.csv at 12-20 dB, exp_pm_chanhi.py
                    --part hi)
  fig/fig3_pm.pdf : content step. SER at 10 dB versus the shared-token
                    probability rho, with the cross-energy of the masks
                    trained at 10 dB (pm_content_snr.csv; exp_pm_ext.py
                    --part content_snr)
  fig/fig4_pm.pdf : signaling step. SER at 10 dB versus the values per
                    mask J, 1 to 64 for every curve: static entries trained
                    at 10 dB (pm_sig10.csv; exp_pm_ext.py --part sig10) and
                    adapted masks (pm_sig.csv; exp_pm_ext.py --part sig)
  fig/fig5_pm.pdf : trade-off. Net efficiency of the adapted masks at
                    10 dB and four taps versus J for an update every 1, 10
                    or 100 frames (pm_sig.csv, pm_figs_class.csv)
  fig/fig_codebook_pm.pdf : supplementary. (a) SER and (b) topic accuracy
                    versus SNR, U = 4, flat fading, L = 16, for the
                    proposed BERT codebook, a random codebook, and OMA
                    (pm_models_eval.csv; exp_pm.py --part models)

fig_plen() draws the SER at 10 dB versus the pattern length for U = 2, 4, 8
(pm_period_eval.csv, pm_plen_eval.csv) to fig/fig_plen_pm.pdf; it is not a
figure of the letter.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA, FIG = ROOT / "data" / "pm", ROOT / "fig"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8,
    "legend.fontsize": 6.6, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "lines.linewidth": 1.1, "lines.markersize": 3.6, "axes.linewidth": 0.6,
    "grid.linewidth": 0.4, "grid.alpha": 0.5,
    "pdf.fonttype": 42,                # TrueType, not Type 3, in the PDFs
})

# one style per scheme across all figures: (colour, marker)
STYLE = {"static": ("C0", "o"), "adapted": ("C3", "s"), "oma": ("k", "d"),
         "no_overlap": ("C1", "v"), "full_overlap": ("C4", "^"),
         "random": ("C2", "^")}


def rows(name):
    with open(DATA / name) as f:
        return list(csv.DictReader(f))


def curve(rs, key, **sel):
    acc = defaultdict(list)
    for r in rs:
        if all(str(r[k]) == str(v) for k, v in sel.items()):
            acc[float(r["snr_db"])].append(float(r[key]))
    x = sorted(acc)
    return np.array(x), np.array([np.mean(acc[s]) for s in x])


W43, H43 = 2.6, 1.95                  # 8:6 canvas of Figs. 2-5 (inches)
BOX43 = dict(left=0.18, right=0.97, bottom=0.205, top=0.955)


def check_layout(fig, name):
    """Raise if a text artist leaves the canvas or a legend covers a curve,
    a marker, a reference line or an annotation (legend box inflated by the
    marker radius plus the line width)."""
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    fb = fig.bbox
    for ax in fig.axes:
        arts = [ax.xaxis.label, ax.yaxis.label, ax.title] + list(ax.texts)
        for axis in (ax.xaxis, ax.yaxis):          # tick labels inside the view only
            lo, hi = sorted(axis.get_view_interval())
            arts += [tk.label1 for tk in axis.get_major_ticks()
                     if lo - 1e-12 <= tk.get_loc() <= hi + 1e-12]
        for a in arts:
            if not a.get_visible() or not a.get_text():
                continue
            b = a.get_window_extent(rend)
            if b.x0 < fb.x0 - 0.5 or b.y0 < fb.y0 - 0.5 or b.x1 > fb.x1 + 0.5 or b.y1 > fb.y1 + 0.5:
                raise RuntimeError(f"{name}: text outside the canvas: {a.get_text()!r}")
        leg = ax.get_legend()
        if leg is None:
            continue
        lb = leg.get_window_extent(rend)
        ab = ax.bbox
        if lb.x0 < ab.x0 - 0.5 or lb.y0 < ab.y0 - 0.5 or lb.x1 > ab.x1 + 0.5 or lb.y1 > ab.y1 + 0.5:
            raise RuntimeError(f"{name}: legend leaves the axes box")
        for line in ax.get_lines():
            xy = np.asarray(line.get_xydata(), float)
            if len(xy) == 0:
                continue
            d = line.get_transform().transform(xy)    # data or blended (axvline/axhline)
            seg = [d[:1]] + [a_ + (b_ - a_) * np.linspace(0, 1, 30)[1:, None]
                             for a_, b_ in zip(d[:-1], d[1:])]
            pts = np.vstack(seg)
            r = (line.get_markersize() / 2 + line.get_linewidth()) * fig.dpi / 72
            vis = (pts[:, 0] >= ab.x0) & (pts[:, 0] <= ab.x1) & (pts[:, 1] >= ab.y0) & (pts[:, 1] <= ab.y1)
            inside = (pts[:, 0] > lb.x0 - r) & (pts[:, 0] < lb.x1 + r) & \
                     (pts[:, 1] > lb.y0 - r) & (pts[:, 1] < lb.y1 + r)
            if np.any(vis & inside):
                raise RuntimeError(f"{name}: legend covers the curve {line.get_label()!r}")
        for a in ax.texts:
            if a.get_window_extent(rend).overlaps(lb):
                raise RuntimeError(f"{name}: legend covers the text {a.get_text()!r}")


def save(fig, name, tight=True):
    check_layout(fig, name)
    if tight:
        fig.savefig(FIG / name, bbox_inches="tight", pad_inches=0.02)
    else:                                  # fixed canvas: same axes box in Figs. 2-5
        fig.savefig(FIG / name)
    plt.close(fig)
    print("wrote", FIG / name)


def legend_in(ax, **kw):
    """Legend inside the axes box with a light frame."""
    opts = dict(frameon=True, framealpha=0.92, edgecolor="0.75", fontsize=6.2,
                handlelength=2.0, borderpad=0.35, labelspacing=0.25, borderaxespad=0.35)
    opts.update(kw)
    leg = ax.legend(**opts)
    leg.get_frame().set_linewidth(0.4)
    return leg


def fig_codebook():
    """(a) SER and (b) topic accuracy of the received articles versus SNR,
    U = 4, flat fading, L = 16, for the proposed BERT codebook, a random
    codebook with the same masks and decoder, and OMA (pm_models_eval.csv,
    means over seeds; the error-free accuracy comes from
    pm_models_train.json). The GTR-T5 codebook and the projections without
    source-statistics whitening are in the same CSV and quoted in the text."""
    import json
    rs = rows("pm_models_eval.csv")
    clean = json.load(open(DATA / "pm_models_train.json"))["bert_clean_acc"]
    spec = [
        (dict(source="bert", codebook="pwhite"), "Proposed", *STYLE["static"][:1], "-", STYLE["static"][1]),
        (dict(source="bert", codebook="rand"), "Random codebook", *STYLE["random"][:1], "--", STYLE["random"][1]),
        (dict(source="bert", codebook="oma"), "OMA", *STYLE["oma"][:1], ":", STYLE["oma"][1]),
    ]
    fig, ax = plt.subplots(1, 2, figsize=(3.5, 1.75))
    for sel, lab, col, ls, mk in spec:
        x, y = curve(rs, "ter", **sel)
        ax[0].plot(x, y, color=col, ls=ls, marker=mk, markevery=2, label=lab)
        x, y = curve(rs, "topic_acc", **sel)
        ax[1].plot(x, 100 * y, color=col, ls=ls, marker=mk, markevery=2, label=lab)
    ax[1].axhline(100 * clean, color="0.5", lw=0.7, ls="--", label="_nolegend_")
    ax[1].text(20, 100 * clean + 1.2, "error-free text", fontsize=5.8, color="0.35", ha="right", va="bottom")
    ax[0].set_yscale("log")
    ax[0].set_ylabel("SER")
    ax[1].set_ylabel("Topic accuracy (%)")
    ax[1].set_ylim(10, 106)
    ax[1].set_yticks([20, 40, 60, 80, 100])
    for a, t_ in zip(ax, ("(a)", "(b)")):
        a.set_xlabel("SNR (dB)")
        a.set_xticks([0, 5, 10, 15, 20])
        a.grid(True, which="both")
        a.set_title(t_, fontsize=8, pad=2)
    legend_in(ax[1], loc="lower right")
    fig.tight_layout(w_pad=0.6)
    save(fig, "fig_codebook_pm.pdf")


def fig_plen():
    rp = rows("pm_period_eval.csv")
    rl = rows("pm_plen_eval.csv") if (DATA / "pm_plen_eval.csv").exists() else []
    fig, ax = plt.subplots(figsize=(3.4, 1.85))
    spec = [(2, "C2", "^"), (4, "C0", "o"), (8, "C3", "s")]
    for U, col, mk in spec:
        acc = defaultdict(list)
        for r in rp:
            if (int(r["users"]) == U and r["codebook"] != "oma" and r["train_cond"] == "flat"
                    and float(r["snr_db"]) == 10):
                acc[int(r["L"])].append(float(r["ter"]))
        L = sorted(acc)
        ax.plot(L, [np.mean(acc[k]) for k in L], color=col, marker=mk, label=f"$U={U}$, flat")
        oma = [float(r["ter"]) for r in rp if int(r["users"]) == U and r["codebook"] == "oma"
               and float(r["snr_db"]) == 10]
        if oma:
            ax.axhline(oma[0], color=col, ls=":", lw=0.8)
    acc = {}
    for r in rl:
        if r["eval_cond"] == "nmse05" and float(r["snr_db"]) == 10:
            acc[int(r["L"])] = float(r["ter"])
    if acc:
        L = sorted(acc)
        ax.plot(L, [acc[k] for k in L], color="C0", ls="--", marker="o", mfc="white",
                label="$U=4$, NMSE $0.05$")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks([1, 2, 4, 8, 16, 32, 64])
    ax.set_xticklabels(["1", "2", "4", "8", "16", "32", "64"])
    ax.set_xlabel("Pattern length $L$ (dotted: OMA)")
    ax.set_ylabel("SER at 10 dB")
    ax.grid(True, which="both")
    ax.legend(loc="upper right", ncol=2, frameon=True, framealpha=0.9)
    fig.tight_layout()
    save(fig, "fig_plen_pm.pdf")


def fig_channel():
    """Graph 1: SER versus SNR under 4-tap and 8-tap fading for OMA, the
    static entry trained for the class (L = 16, data/pm/pm_figs_class.csv,
    exp_pm_chanhi.py --part class), and masks adapted per update to the known
    gains (blocks of two subcarriers) (data/pm/pm_figs.csv, graph g1, with
    the 400,000-frame points of data/pm/pm_figs_hi.csv at 12-20 dB). Points
    with no error in the decoded frames are not drawn."""
    from matplotlib.lines import Line2D
    rs = [r for r in rows("pm_figs.csv") if r["graph"] == "g1"]
    hi = {(r["channel"], r["scheme"], r["snr_db"]): r["ser"] for r in rows("pm_figs_hi.csv")}
    for r in rs:                                   # 400,000-frame points at 12-20 dB
        key = (r["channel"], r["scheme"], str(int(float(r["snr_db"]))))
        if key in hi:
            r["ser"] = hi[key]
    # the static entry is the one the mask set stores for the class itself
    rs = [r for r in rs if r["scheme"] != "static_L16"]
    for r in rows("pm_figs_class.csv"):
        rs.append(dict(r, scheme="static_L16"))
    fig = plt.figure(figsize=(W43, H43))
    fig.subplots_adjust(**BOX43)
    ax = fig.add_subplot(111)
    sty = {"oma": (*STYLE["oma"], "OMA"), "static_L16": (*STYLE["static"], "Static entry"),
           "adapted_B2": (*STYLE["adapted"], "Adapted masks")}
    hs = []
    for sc, (col, mk, lab) in sty.items():
        for cn, ls, tag in (("fs4", "-", "4 taps"), ("fs8", "--", "8 taps")):
            pts = sorted((float(r["snr_db"]), float(r["ser"])) for r in rs
                         if r["channel"] == cn and r["scheme"] == sc and float(r["ser"]) > 0)
            mfc = "white" if ls == "--" else col
            ax.plot([q[0] for q in pts], [q[1] for q in pts], color=col, ls=ls, marker=mk,
                    markevery=2, mfc=mfc, label=f"{lab}, {tag}")
            hs.append(Line2D([], [], color=col, ls=ls, marker=mk, mfc=mfc, label=f"{lab}, {tag}"))
    ax.set_yscale("log")
    ax.set_ylim(1e-6, 1.0)
    ax.set_xlim(-0.5, 20.5)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("SER")
    ax.set_xticks([0, 4, 8, 12, 16, 20])
    ax.grid(True, which="both")
    legend_in(ax, handles=hs, loc="lower left", handlelength=1.6, fontsize=6.0, labelspacing=0.18)
    save(fig, "fig2_pm.pdf", tight=False)


def fig_content():
    """Graph 2: SER at 10 dB versus the probability rho that all users
    receive the same token, with masks trained at the 10 dB they serve
    (data/pm/pm_content_snr.csv; exp_pm_ext.py --part content_snr)."""
    rs = []
    for r in rows("pm_content_snr.csv"):
        if float(r["snr_db"]) != 10 or r["scheme"] == "trained_0_20dB":
            continue
        sc = "trained_L16" if r["scheme"] == "trained_at_snr" else r["scheme"]
        rs.append({"scheme": sc, "rho": r["rho"], "ser": r["ser"], "aux": r["cross_energy"]})
    fig = plt.figure(figsize=(W43, H43))
    fig.subplots_adjust(**BOX43)
    ax = fig.add_subplot(111)
    sty = {"trained_L16": (*STYLE["static"], "-", "Trained masks, $L=16$"),
           "no_overlap": (*STYLE["no_overlap"], "-.", "Masks without overlap"),
           "full_overlap": (*STYLE["full_overlap"], "-.", "Full overlap")}
    for sc, (col, mk, ls, lab) in sty.items():
        pts = sorted((float(r["rho"]), float(r["ser"]), float(r["aux"])) for r in rs if r["scheme"] == sc)
        fixed = sc != "trained_L16"                     # fixed masks: open markers drawn on top
        ax.plot([q[0] for q in pts], [q[1] for q in pts], color=col, marker=mk, ls=ls,
                mfc="white" if fixed else col, zorder=3 if fixed else 2, label=lab)
        if sc == "trained_L16":
            for q in pts:
                if q[0] > 0.95:                         # last point: label to the left
                    off, ha, va = (-6, -2), "right", "center"
                elif q[0] > 0.85:                       # steep segment: label below left
                    off, ha, va = (-4, -10), "right", "baseline"
                else:
                    off, ha, va = (0, -10), "center", "baseline"
                ax.annotate(f"{q[2]:.2f}", (q[0], q[1]), textcoords="offset points",
                            xytext=off, ha=ha, va=va, fontsize=5.6, color=col)
    ax.set_yscale("log")
    ax.set_ylim(0.02, 1.2)
    ax.set_xlim(-0.04, 1.04)
    ax.set_xlabel(r"Shared-token probability $\rho$")
    ax.set_ylabel("SER at 10 dB")
    ax.grid(True, which="both")
    legend_in(ax, loc="lower left")
    save(fig, "fig3_pm.pdf", tight=False)


def fig_signal():
    """Graph 3: SER at 10 dB versus the mask values a receiver learns per
    user. Static entries trained at the 10 dB they serve come from
    data/pm/pm_sig10.csv (exp_pm_ext.py --part sig10), adapted masks from
    data/pm/pm_sig.csv (exp_pm_ext.py --part sig). Each static entry
    is the best of three training runs on separate validation frames
    (column ser_val); the plotted SER comes from the test frames (column
    ser). Adapted masks use J blocks of 64/J real dimensions, shared among
    users when J < U."""
    rs = [r for r in rows("pm_sig.csv") if r["scheme"] == "adapted"]
    for r in rows("pm_sig10.csv"):                 # static entries trained at 10 dB
        rs.append(dict(r, scheme={"static_at10dB": "static",
                                  "shared_rho0.9_at10dB": "shared_rho0.9"}[r["scheme"]]))
    fig = plt.figure(figsize=(W43, H43))
    fig.subplots_adjust(**BOX43)
    ax = fig.add_subplot(111)
    spec = [(("flat", "static"), *STYLE["static"], "-", r"Static entry, $\rho=0$"),
            (("flat", "shared_rho0.9"), *STYLE["static"], "--", r"Static entry, $\rho=0.9$"),
            (("fs4", "adapted"), *STYLE["adapted"], "-", "Adapted masks, 4 taps"),
            (("fs8", "adapted"), *STYLE["adapted"], "--", "Adapted masks, 8 taps")]
    for (cn, sc), col, mk, ls, lab in spec:
        best = {}                       # static entries: best of three runs on validation frames
        for r in rs:
            if r["channel"] == cn and r["scheme"] == sc:
                v_, key = int(r["values"]), float(r["ser_val"]) if r["ser_val"] else 0.0
                if v_ not in best or key < best[v_][0]:
                    best[v_] = (key, float(r["ser"]))
        xs = sorted(best)
        ax.plot(xs, [best[k][1] for k in xs], color=col, marker=mk, ls=ls,
                mfc="white" if ls == "--" else col, label=lab)
    ax.axvline(64, color="0.5", lw=0.6, ls=":", label="_nolegend_")
    ax.text(58, 3.2e-2, "full-length mask", ha="right", va="bottom",
            fontsize=5.8, color="0.35")                       # between the static and adapted curves
    ax.axvline(4, ymax=0.6, color="0.5", lw=0.6, ls="--", label="_nolegend_")  # J = U, below the legend
    ax.text(3.7, 1.12e-3, "$J=U$", ha="right", va="bottom", fontsize=5.8, color="0.35")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks([1, 2, 4, 8, 16, 32, 64])
    ax.set_xticklabels(["1", "2", "4", "8", "16", "32", "64"])
    ax.set_xlim(0.8, 80)
    ax.set_ylim(1e-3, 30.0)                                   # headroom for the legend
    ax.set_yticks([1e-3, 1e-2, 1e-1, 1.0])
    ax.set_xlabel("Values per mask $J$")
    ax.set_ylabel("SER at 10 dB")
    ax.grid(True, which="both")
    legend_in(ax, loc="upper right", bbox_to_anchor=(0.935, 0.995))   # left of the x = 64 line
    save(fig, "fig4_pm.pdf", tight=False)


def fig_tradeoff():
    """Trade-off of the adapted masks between adaptation and signaling at
    10 dB and four taps: net efficiency versus the number J of values per
    mask for updates every F frames (data/pm/pm_sig.csv, adapted fs4; the
    static entry trained for the class from pm_figs_class.csv). Net
    efficiency = (U (1 - SER) log2 V - b_J / F) / (U log2 V), with
    b_J = J log2 U bits per update for the owners of J >= U blocks and
    U ceil(log2 J) bits when J < U users name the block they share. Stars
    mark the best J of each curve."""
    import math
    U_, V_ = 4, 30522
    pay = U_ * math.log2(V_)
    sig = defaultdict(dict)
    for r in rows("pm_sig.csv"):
        if r["scheme"] == "adapted":
            sig[r["channel"]][int(r["values"])] = float(r["ser"])
    st = {r["channel"]: float(r["ser"]) for r in rows("pm_figs_class.csv")
          if float(r["snr_db"]) == 10}                # entry stored for the class

    def bits(v):
        return v * math.log2(U_) if v >= U_ else U_ * math.ceil(math.log2(v)) if v > 1 else 0.0

    fig = plt.figure(figsize=(W43, H43))
    fig.subplots_adjust(**BOX43)
    ax = fig.add_subplot(111)
    xs = sorted(sig["fs4"])
    col, mk = STYLE["adapted"]                     # all three curves are adapted masks
    for F, c_, ls, mfc in ((1, col, "-", col), (10, col, "--", "white"), (100, "#7a1416", "-.", "#7a1416")):
        eta = [(U_ * (1 - sig["fs4"][v]) * math.log2(V_) - bits(v) / F) / pay for v in xs]
        lab = "Update every frame" if F == 1 else f"Update every {F} frames"
        ax.plot(xs, eta, color=c_, marker=mk, ls=ls, mfc=mfc, label=lab)
        k = int(np.argmax(eta))
        ax.plot([xs[k]], [eta[k]], ls="none", marker="*", ms=9, mfc="none", mec="k", mew=0.8,
                label="Best $J$" if F == 100 else "_nolegend_")
        if F == 1 and eta[-1] < -0.55:              # the last point lies below the axis
            ax.annotate(f"${eta[-1]:.2f}$", xy=(xs[-1], -0.55), xytext=(xs[-1], -0.36),
                        fontsize=6.0, color="0.25", ha="center", va="center",
                        arrowprops=dict(arrowstyle="-|>", color="0.35", lw=0.6, shrinkA=1, shrinkB=0))
    ax.axhline(1 - st["fs4"], color="0.45", lw=0.8, ls=":", label="Static entry, no update")
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16, 32, 64])
    ax.set_xticklabels(["1", "2", "4", "8", "16", "32", "64"])
    ax.set_xlim(0.8, 80)
    ax.axvspan(0.8, U_, color="0.88", lw=0, zorder=0)           # J < U: users share blocks
    ax.text(1.0, 0.80, "users share\nblocks, $J<U$", fontsize=6.0, color="0.25", ha="left", va="center")
    ax.annotate("signaling\nexceeds gain", xy=(27.5, 0.037), xytext=(60, 0.42), fontsize=6.0,
                color="0.25", ha="right", va="center",
                arrowprops=dict(arrowstyle="-|>", color="0.35", lw=0.6, shrinkA=1, shrinkB=1))
    ax.set_ylim(-0.55, 1.05)
    ax.set_yticks([-0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("Values per mask $J$")
    ax.set_ylabel("Net efficiency at 10 dB")
    ax.grid(True, which="both")
    legend_in(ax, loc="lower left")
    save(fig, "fig5_pm.pdf", tight=False)


if __name__ == "__main__":
    fig_codebook()
    fig_channel()
    fig_content()
    fig_signal()
    fig_tradeoff()
