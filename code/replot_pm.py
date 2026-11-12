"""Figures of the letter on low-dimensional masks for shared-embedding
multiple access, from the CSV files in data/pm/. Plotting only, no
simulation. Figs. 2-4 of the manuscript (fig2_pm, fig4_pm, fig5_pm) share
one 8:6 canvas of 3.0 x 2.25 in (W43, H43), drawn at the printed width
(\\resw = 3.0in in main_revision.tex, 0.86 of the column), so that the
8 pt axis labels, the 7 pt tick labels and the 6.5 pt legends and
annotations (LEG43, ANN43) print at their designed size; they share the
same axes box (BOX43) and keep their legends inside the axes. The
supplementary content-step figure fig3_pm keeps its former 2.6 in canvas
and axes box (BOX_R2), and fig_codebook_pm and fig_plen_pm keep their
sizes; save() refuses a figure whose legend covers
a curve, a marker, a reference line or an annotation, or whose text leaves
the canvas. One style per scheme (STYLE): the static entries of the mask
set (trained masks) are blue circles, adapted masks red squares, OMA black
diamonds, the untrained pattern with L = U (masks without overlap)
orange down-triangles and full overlap
purple triangles; a dashed line with open markers marks the second variant
of the same scheme.

Metric. "SER" on every axis is the semantic error rate of the manuscript,
  SER_u = E[1 - zeta(s_u, s_hat_u)]
        = sum_v p_v sum_{w != v} Pr[s_hat_u = w | s_u = v] (1 - zeta(v, w)),
with zeta(v, w) the cosine similarity of the BERT input embeddings w_v and
w_w, averaged over the users (column ser_sem, or sem_dist in
pm_models_eval.csv). A correct token costs nothing. The token error rate
TER = Pr[s_hat_u != s_u] (column ter, or ser in the older files) is used
only where bits are counted, in the net efficiency of fig5_pm, whose
axis is the loss 1 - eta.

  fig/fig2_pm.pdf : Fig. 2, channel step. SER versus SNR at T = 4 (solid,
                    filled markers) and 8 taps (dashed, hollow markers) for
                    OMA, coded OMA with the bit interleaver of
                    pm_oma_coded_il.json (on every U-th subcarrier and on the
                    adapted blocks), ToDMA, the static entry trained for the
                    class, and masks adapted to the known gains, all from
                    pm_semfig2.csv (exp_pm_semfig2.py); points with fewer
                    than 20 token errors are not drawn
  fig/fig3_pm.pdf : supplementary figure, content step (not in the
                    manuscript), static entry L = 16, untrained pattern
                    L = U and full overlap. SER at 10 dB versus the shared-token
                    probability rho, with the cross-energy of the masks
                    trained at 10 dB (pm_semfig3.csv; exp_pm_semfig34.py
                    --part content)
  fig/fig4_pm.pdf : Fig. 3 of the manuscript (content and signaling steps).
                    SER at 10 dB versus the mask size J, 1 to 64 for every
                    curve: static entries trained at 10 dB for rho = 0, 0.5,
                    0.75 and 0.9, best of three runs by the SER on validation
                    frames (pm_semfig4_static.csv; exp_pm_semfig34.py --part
                    static) and adapted masks (pm_semfig4_adapted.csv;
                    exp_pm_semfig4a.py)
  fig/fig5_pm.pdf : Fig. 4 of the manuscript, trade-off. Loss 1 - eta of
                    the net efficiency (9) on a log axis, adapted masks at
                    10 dB and four taps versus J for an update every 1, 3,
                    10, 30 or 100 frames, and the static entry, from the TER,
                    since the payload counts bits (pm_semfig4_adapted.csv,
                    pm_semfig2.csv)
  fig/fig_codebook_pm.pdf : supplementary figure. (a) SER and (b) topic accuracy
                    versus SNR, U = 4, flat fading, L = 16, for the
                    proposed BERT codebook, a random codebook, and OMA
                    (pm_models_eval.csv; exp_pm.py --part models)

fig_plen() draws the TER at 10 dB versus the pattern length for U = 2, 4, 8
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
         "random": ("C2", "^"), "coded_comb": ("C7", "h"),
         "coded_adapted": ("C5", "p"), "todma": ("C8", "v")}


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


MIN_ERR = 20                          # Fig. 2 draws only points with at least this many token errors
W43, H43 = 3.0, 2.25                  # 8:6 canvas of Figs. 2-4 (inches), drawn at the printed width
# (the manuscript includes Figs. 2-4 at \resw = 3.0in). Axes box of Figs. 2-4: margins of 0.456 in
# (left), 0.072 in (right), 0.396 in (bottom) and 0.065 in (top) hold the 8 pt axis labels and the
# 7 pt tick labels; the axes box takes the rest of the canvas
BOX43 = dict(left=0.152, right=0.976, bottom=0.176, top=0.971)
LEG43 = 6.5                           # legend font of Figs. 2-4 (pt), one size for the three figures
ANN43 = 6.5                           # annotation font of Figs. 3 and 4 (pt)
BOX_R2 = dict(left=0.19, right=0.97, bottom=0.215, top=0.955)   # supplementary fig3_pm, 2.6 in canvas


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
    else:                                  # fixed canvas: same axes box in Figs. 2-4
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
    """(a) SER (semantic error rate, column sem_dist) and (b) topic accuracy
    of the received articles versus SNR,
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
        x, y = curve(rs, "sem_dist", **sel)          # semantic error rate
        ax[0].plot(x, y, color=col, ls=ls, marker=mk, markevery=2, label=lab)
        x, y = curve(rs, "topic_acc", **sel)
        ax[1].plot(x, 100 * y, color=col, ls=ls, marker=mk, markevery=2, label=lab)
    ax[1].axhline(100 * clean, color="0.5", lw=0.7, ls="--", label="_nolegend_")
    ax[1].text(20, 100 * clean + 1.2, "error-free text", fontsize=5.8, color="0.35", ha="right", va="bottom")
    ax[0].set_yscale("log")
    ax[0].set_ylim(0.01, 1.0)                     # labels at 10^-2, 10^-1 and 10^0
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
    ax.set_ylabel("TER at 10 dB")
    ax.grid(True, which="both")
    ax.legend(loc="upper right", ncol=2, frameon=True, framealpha=0.9)
    fig.tight_layout()
    save(fig, "fig_plen_pm.pdf")


def fig_channel():
    """Fig. 2: SER (semantic error rate, column ser_sem) versus SNR under
    4-tap (solid, filled markers) and 8-tap (dashed, hollow markers) fading,
    U = 4, tokens drawn with p_v, all from data/pm/pm_semfig2.csv
    (exp_pm_semfig2.py), for
      oma               OMA, uncoded 16 bits on eight adjacent subcarriers
      coded_il_comb     coded OMA, RM(2,5) in Gray 16-QAM with the bit
                        interleaver of pm_oma_coded_il.json, ML over the V
                        valid ids, on every U-th subcarrier
      todma             ToDMA, fixed random codewords, interference-plus-
                        noise weighted OMP, genie association
      static_class_L16  static entry trained for the class, L = 16
      coded_il_adapted  the same interleaved code on the blocks that each
                        adapted mask weights
      adapted_B2        adapted masks on blocks of two subcarriers
    Points that rest on fewer than MIN_ERR = 20 token errors (column
    errors) are not drawn, nor points without errors. Colour and marker
    mark the scheme, line style and marker face the tap class (solid line
    and filled marker 4 taps, dashed line and hollow marker 8 taps); the
    legend shows the tap classes by the marker face only, without a line,
    so that they do not read as two more schemes (author's choice)."""
    from matplotlib.lines import Line2D
    rs = rows("pm_semfig2.csv")
    fig = plt.figure(figsize=(W43, H43))
    fig.subplots_adjust(**BOX43)
    ax = fig.add_subplot(111)
    sty = {"oma": (*STYLE["oma"], "OMA"),
           "coded_il_comb": (*STYLE["coded_comb"], "Coded OMA, interleaved"),
           "todma": (*STYLE["todma"], "ToDMA"),
           "static_class_L16": (*STYLE["static"], "Static entry"),
           "coded_il_adapted": (*STYLE["coded_adapted"], "Coded OMA, adapted blocks"),
           "adapted_B2": (*STYLE["adapted"], "Adapted masks")}
    hs = []
    for sc, (col, mk, lab) in sty.items():
        for cn, ls in (("fs4", "-"), ("fs8", "--")):
            pts = sorted((float(r["snr_db"]), float(r["ser_sem"])) for r in rs
                         if r["channel"] == cn and r["scheme"] == sc and float(r["ser_sem"]) > 0
                         and int(r["errors"]) >= MIN_ERR)
            mfc = "white" if ls == "--" else col
            ax.plot([q[0] for q in pts], [q[1] for q in pts], color=col, ls=ls, marker=mk,
                    markevery=2, mfc=mfc, label=f"{lab}, {cn}")
        hs.append(Line2D([], [], color=col, ls="-", marker=mk, label=lab))
    # tap classes by the marker face only, without a line (author's choice)
    hs.append(Line2D([], [], color="0.3", ls="none", marker="o", mfc="0.3", label="4 taps"))
    hs.append(Line2D([], [], color="0.3", ls="none", marker="o", mfc="white", label="8 taps"))
    ax.set_yscale("log")
    ax.set_ylim(8e-9, 2.0)                     # room below the curves for the legend; the top
                                               # spine clears the markers at 0 dB (SER up to 0.73)
    ax.set_xlim(-0.5, 20.5)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("SER")
    ax.set_xticks([0, 4, 8, 12, 16, 20])
    ax.set_yticks([1e-7, 1e-5, 1e-3, 1e-1])
    ax.grid(True, which="both")
    legend_in(ax, handles=hs, loc="lower left", handlelength=1.6, fontsize=LEG43,
              labelspacing=0.12, borderpad=0.25)
    save(fig, "fig2_pm.pdf", tight=False)


def fig_content():
    """Supplementary content-step figure: SER (semantic error rate) at 10 dB versus the
    probability rho that all users receive the same token, with masks
    trained at the 10 dB they serve (data/pm/pm_semfig3.csv;
    exp_pm_semfig34.py --part content). Labels give the mean cross-energy."""
    rs = []
    for r in rows("pm_semfig3.csv"):
        if float(r["snr_db"]) != 10:
            continue
        sc = "trained_L16" if r["scheme"] == "trained_at_snr" else r["scheme"]
        rs.append({"scheme": sc, "rho": r["rho"], "ser": r["ser_sem"], "aux": r["cross_energy"]})
    fig = plt.figure(figsize=(2.6, 1.95))          # the supplementary figure keeps the former 2.6 in canvas
    fig.subplots_adjust(**BOX_R2)
    ax = fig.add_subplot(111)
    sty = {"trained_L16": (*STYLE["static"], "-", "Static entry, $L=16$"),
           "no_overlap": (*STYLE["no_overlap"], "-.", "Untrained pattern, $L=U$"),
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
                elif q[0] < 0.05:                       # first point: label to the right of the y axis
                    off, ha, va = (-3, -10), "left", "baseline"
                else:
                    off, ha, va = (0, -10), "center", "baseline"
                ax.annotate(f"{q[2]:.2f}", (q[0], q[1]), textcoords="offset points",
                            xytext=off, ha=ha, va=va, fontsize=5.6, color=col)
    ax.set_yscale("log")
    ax.set_ylim(0.009, 1.2)
    ax.set_xlim(-0.04, 1.04)
    ax.set_xlabel(r"Shared-token probability $\rho$")
    ax.set_ylabel("SER at 10 dB")
    ax.grid(True, which="both")
    legend_in(ax, loc="lower left")
    save(fig, "fig3_pm.pdf", tight=False)


def fig_signal(rhos=(0.0, 0.5, 0.75, 0.9)):
    """Fig. 3: SER (semantic error rate) at 10 dB versus the mask size J
    (values per mask). Static entries trained at the 10 dB they serve for
    the shared-token probabilities rho come from
    data/pm/pm_semfig4_static.csv (exp_pm_semfig34.py --part static); each
    is the best of three training runs by the SER on separate validation
    frames (column ser_sem_val, failed runs with nan skipped), and the
    plotted SER comes from the test frames (column ser_sem). Adapted masks
    come from data/pm/pm_semfig4_adapted.csv (exp_pm_semfig4a.py, 10 dB),
    with J blocks of 64/J real dimensions, where the masks overlap when J < U.
    The static entries share the blue of the scheme and differ in line
    style and shade by rho."""
    rs = []
    for r in rows("pm_semfig4_adapted.csv"):
        if float(r["snr_db"]) == 10:
            rs.append(dict(r, scheme="adapted", ser=r["ser_sem"], ser_val=""))
    for r in rows("pm_semfig4_static.csv"):          # static entries trained at 10 dB
        rs.append(dict(r, scheme=f"static_rho{float(r['rho']):g}", ser=r["ser_sem"],
                       ser_val=r["ser_sem_val"]))
    fig = plt.figure(figsize=(W43, H43))
    fig.subplots_adjust(**BOX43)
    ax = fig.add_subplot(111)
    col, mk = STYLE["static"]
    shade = {0.0: col, 0.25: "#2c7fb8", 0.5: "#3d8fd0", 0.75: "#5aa5dd", 0.9: "#0b4f8a", 1.0: "#08306b"}
    lstyle = {0.0: "-", 0.25: (0, (5, 1, 1, 1)), 0.5: "-.", 0.75: ":", 0.9: "--", 1.0: (0, (1, 1))}
    spec = [(("flat", f"static_rho{r:g}"), shade[r], mk, lstyle[r],
             "white" if r >= 0.75 else None, rf"Static entry, $\rho={r:g}$") for r in rhos]
    spec += [(("fs4", "adapted"), *STYLE["adapted"], "-", None, "Adapted masks, 4 taps"),
             (("fs8", "adapted"), *STYLE["adapted"], "--", "white", "Adapted masks, 8 taps")]
    for k, ((cn, sc), c_, m_, ls, mfc, lab) in enumerate(spec):
        best = {}                       # static entries: best of three runs on validation frames
        for r in rs:
            if r["channel"] == cn and r["scheme"] == sc:
                if r["ser"] in ("", "nan") or (r["ser_val"] not in ("",) and r["ser_val"] == "nan"):
                    continue
                v_, key = int(r["values"]), float(r["ser_val"]) if r["ser_val"] else 0.0
                if v_ not in best or key < best[v_][0]:
                    best[v_] = (key, float(r["ser"]))
        if not best:
            continue
        xs = sorted(best)
        ax.plot(xs, [best[q][1] for q in xs], color=c_, marker=m_, ls=ls,
                mfc=mfc if mfc else c_, label=lab)
    ax.axvline(64, color="0.5", lw=0.6, ls=":", label="_nolegend_")
    ax.text(58, 2.6e-2, "full-length mask", ha="right", va="bottom",
            fontsize=ANN43, color="0.35")                       # between the static and adapted curves
    ax.axvline(4, ymax=0.5, color="0.5", lw=0.6, ls="--", label="_nolegend_")  # J = U, below the legend
    ax.text(3.7, 5.6e-4, "$J=U$", ha="right", va="bottom", fontsize=ANN43, color="0.35")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks([1, 2, 4, 8, 16, 32, 64])
    ax.set_xticklabels(["1", "2", "4", "8", "16", "32", "64"])
    ax.set_xlim(0.8, 80)
    ax.set_ylim(5e-4, 60.0)                                   # headroom for the legend
    ax.set_yticks([1e-3, 1e-2, 1e-1, 1.0])
    ax.set_xlabel("Mask size $J$ (values per mask)")
    ax.set_ylabel("SER at 10 dB")
    ax.grid(True, which="both")
    legend_in(ax, loc="upper right", bbox_to_anchor=(0.935, 0.995), fontsize=LEG43,
              labelspacing=0.12, borderpad=0.25)              # left of the x = 64 line
    save(fig, "fig4_pm.pdf", tight=False)


# Update intervals F of Fig. 4 and their styles: one red ramp for the adapted masks (darker =
# rarer updates), solid lines with filled markers and dashed lines with hollow markers in turn
F_STYLE = ((1, "#fb8a6a", "-", True), (3, "#f0553a", "--", False), (10, "#d62728", "-", True),
           (30, "#a81419", "--", False), (100, "#6b0a0f", "-", True))


def fig_tradeoff():
    """Fig. 4: trade-off of the adapted masks between adaptation and
    signaling at 10 dB and four taps, versus the mask size J for updates
    every F = 1, 3, 10, 30 or 100 frames (F_STYLE). The axis is the loss
    1 - eta on a log scale, with eta the net efficiency (9),
        1 - eta = TER + b_J / (F U log2 V),
    which stays positive, so that points with eta < 0 (loss above 1)
    remain on the axis and the gap to the static entry near eta = 1 stays
    visible. The payload counts bits, so eta uses the token error rate
    (column ter): adapted masks from pm_semfig4_adapted.csv (fs4, 10 dB),
    the static entry stored for the class from pm_semfig2.csv
    (static_class_L16, fs4, 10 dB, the entry of Fig. 2), which needs no
    update and so loses its TER. The static entry is the thick line in the
    blue of the static entries, named by a bold label on the line (author's
    choice). b_J = J log2 U bits per update for the user index of each of
    the J >= U blocks and U ceil(log2 J) bits when J < U and the update
    names the block that each user weights. Stars mark the best J of each
    curve."""
    import math
    U_, V_ = 4, 30522
    pay = U_ * math.log2(V_)
    ter = {int(r["values"]): float(r["ter"]) for r in rows("pm_semfig4_adapted.csv")
           if r["channel"] == "fs4" and float(r["snr_db"]) == 10}
    st = [float(r["ter"]) for r in rows("pm_semfig2.csv")
          if r["scheme"] == "static_class_L16" and r["channel"] == "fs4"
          and float(r["snr_db"]) == 10]
    assert len(st) == 1
    st = st[0]

    def bits(v):
        return v * math.log2(U_) if v >= U_ else U_ * math.ceil(math.log2(v)) if v > 1 else 0.0

    fig = plt.figure(figsize=(W43, H43))
    fig.subplots_adjust(**BOX43)
    ax = fig.add_subplot(111)
    xs = sorted(ter)
    mk = STYLE["adapted"][1]                       # every curve is an adapted mask
    c_st = STYLE["static"][0]
    ax.axhline(st, color=c_st, lw=1.9, ls="-", zorder=1.8, label="_nolegend_")   # static entry, bold
    ax.text(0.86, st * 1.13, "Static entry", color=c_st, fontweight="bold", fontsize=ANN43 + 0.5,
            ha="left", va="bottom")
    best = {}
    for F, c_, ls, filled in F_STYLE:
        loss = [ter[v] + bits(v) / (F * pay) for v in xs]       # 1 - eta of eq. (9)
        ax.plot(xs, loss, color=c_, marker=mk, ls=ls, mfc=c_ if filled else "white", label=f"$F={F}$")
        k = int(np.argmin(loss))
        best[F] = (xs[k], loss[k])
        ax.plot([xs[k]], [loss[k]], ls="none", marker="*", ms=9, mfc="none", mec="k", mew=0.8,
                label="Best $J$" if F == F_STYLE[-1][0] else "_nolegend_", zorder=4)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks([1, 2, 4, 8, 16, 32, 64])
    ax.set_xticklabels(["1", "2", "4", "8", "16", "32", "64"])
    ax.set_xlim(0.8, 80)
    ax.set_ylim(2e-3, 4)
    ax.axvspan(0.8, U_, color="0.88", lw=0, zorder=0)           # J < U: the masks overlap
    ax.text(1.0, 2.2, "masks overlap,\n$J<U$", fontsize=ANN43, color="0.25", ha="left", va="center")
    xa = 22.6                                                   # on the F = 1 curve between J = 16 and 32
    ya = math.exp(np.interp(math.log(xa), [math.log(16), math.log(32)],
                            [math.log(ter[16] + bits(16) / pay), math.log(ter[32] + bits(32) / pay)]))
    ax.annotate("signaling\nexceeds gain", xy=(xa, ya), xytext=(13, 2.2), fontsize=ANN43,
                color="0.25", ha="right", va="center",
                arrowprops=dict(arrowstyle="-|>", color="0.35", lw=0.6, shrinkA=1, shrinkB=1))
    ax.set_xlabel("Mask size $J$ (values per mask)")
    ax.set_ylabel(r"Loss $1-\eta$ at 10 dB", labelpad=1.0)
    ax.grid(True, which="both")
    legend_in(ax, loc="lower left", fontsize=LEG43, labelspacing=0.12,
              borderpad=0.25)                       # one legend size with Figs. 2 and 3
    save(fig, "fig5_pm.pdf", tight=False)
    print("Fig. 4: best J and loss per F:",
          {F: (j, round(l, 4)) for F, (j, l) in best.items()}, "static entry", round(st, 4))


if __name__ == "__main__":
    fig_codebook()
    fig_channel()
    fig_content()
    fig_signal()
    fig_tradeoff()
