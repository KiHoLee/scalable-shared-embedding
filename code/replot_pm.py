"""Figures of the letter on low-dimensional masks for shared-embedding
multiple access, from the CSV files in data/pm/. Plotting only, no
simulation. Figs. 2-4 of the manuscript (fig2_pm, fig4_pm, fig5_pm) share
one 8:6 canvas of 3.0 x 2.25 in (W43, H43), drawn at the printed width
(\\resw = 3.0in in the LaTeX source, 0.86 of the column), so that the
8 pt axis labels, the 7 pt tick labels and the 6.5 pt legends and
annotations (LEG43, ANN43) print at their designed size; they share the
same axes box (BOX43) and keep their legends inside the axes. The
supplementary content-step figure fig3_pm keeps its former 2.6 in canvas
and axes box (BOX_R2), and fig_codebook_pm and fig_plen_pm keep their
sizes; save() refuses a figure whose legend covers
a curve, a marker, a reference line or an annotation, or whose text leaves
the canvas. One style per scheme (STYLE): the static entries of the mask
set (trained masks) are blue circles, adapted masks red squares (pink
squares in Figs. 2 and 4 for the overlap chosen for rho = 0.9), OMA black
diamonds, the untrained pattern with L = U (masks without overlap)
orange down-triangles and full overlap
purple triangles. Line style and marker face carry the second attribute
of each figure: the tap class in Fig. 2 (dashed line and hollow marker
for 8 taps), rho in Fig. 3 (solid 0, dash-dot 0.7, dashed 0.9, static
entries filled and adapted masks hollow), and the update interval F in
Fig. 4 (solid and dashed lines in turn, filled markers, dash-dotted for
rho = 0.9).

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
                    class, and masks adapted to the known gains, from
                    pm_semfig2.csv (exp_pm_semfig2.py), and the adapted
                    masks with the overlap chosen for rho = 0.9 (pink
                    squares, pm_content_ov_snr.csv; exp_pm_content_ovfig.py
                    --part snr); points with fewer than 20 token errors are
                    not drawn, and the SER axis is cut at 1e-6
  fig/fig3_pm.pdf : supplementary figure, content step (not in the
                    manuscript), static entry L = 16, untrained pattern
                    L = U and full overlap. SER at 10 dB versus the shared-token
                    probability rho, with the cross-energy of the masks
                    trained at 10 dB (pm_semfig3.csv; exp_pm_semfig34.py
                    --part content)
  fig/fig4_pm.pdf : Fig. 3 of the manuscript (content and signaling steps).
                    SER at 10 dB versus the mask size J, 1 to 64 for every
                    curve, rho = 0, 0.7 and 0.9: static entries trained at
                    10 dB under flat fading, best of three runs by the SER on
                    validation frames (pm_semfig4_static.csv and
                    pm_semfig4_static_rho0.7.csv; exp_pm_semfig34.py --part
                    static), and adapted masks at 8 taps, two-level at rho = 0
                    (pm_semfig4_adapted.csv; exp_pm_semfig4a.py) and with the
                    overlap chosen for rho (pm_content_ov_j_rho0.7.csv,
                    pm_content_ov_j.csv; exp_pm_content_ovfig.py --part jsweep)
  fig/fig5_pm.pdf : Fig. 4 of the manuscript, trade-off. Loss 1 - eta of
                    the net efficiency (9) on a log axis, adapted masks at
                    10 dB and four taps versus J for an update every 1, 3,
                    10, 30 or 100 frames, and the static entry, from the TER,
                    since the payload counts bits (pm_semfig4_adapted.csv,
                    pm_semfig2.csv), and at rho = 0.9 (dash-dotted) the
                    adapted masks with the overlap chosen for rho, updated
                    every 100 frames (pink squares, pm_content_ov_j.csv),
                    and the static entry trained for rho = 0.9 at 10 dB
                    (thin line, pm_content_fs.csv, static_trained)
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
    "mathtext.fontset": "cm", "font.size": 8, "axes.labelsize": 8,
    "legend.fontsize": 6.6, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "lines.linewidth": 1.1, "lines.markersize": 3.6, "axes.linewidth": 0.6,
    "grid.linewidth": 0.4, "grid.alpha": 0.5,
    "pdf.fonttype": 42,                # TrueType, not Type 3, in the PDFs
})

# one style per scheme across all figures: (colour, marker)
STYLE = {"static": ("C0", "o"), "adapted": ("C3", "s"), "oma": ("k", "d"),
         "no_overlap": ("C1", "v"), "full_overlap": ("C4", "^"),
         "random": ("C2", "^"), "coded_comb": ("C7", "h"),
         "coded_adapted": ("C5", "p"), "todma": ("C8", "v"),
         "adapted_ov": ("C6", "s")}            # adapted masks with the overlap chosen for rho (pink)


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
LEGKW43 = dict(handlelength=1.6, labelspacing=0.12, borderpad=0.25, columnspacing=0.8)  # Figs. 2-4
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
        from matplotlib.legend import Legend
        for leg in [a for a in ax.get_children() if isinstance(a, Legend)]:   # every legend
            _check_legend(fig, ax, leg, rend, name)
    for leg in fig.legends:                         # figure legends: inside the canvas, off the axes
        lb = leg.get_window_extent(rend)
        if lb.x0 < fb.x0 - 0.5 or lb.y0 < fb.y0 - 0.5 or lb.x1 > fb.x1 + 0.5 or lb.y1 > fb.y1 + 0.5:
            raise RuntimeError(f"{name}: figure legend leaves the canvas")
        for ax in fig.axes:
            if lb.overlaps(ax.bbox):
                raise RuntimeError(f"{name}: figure legend overlaps the axes")


def _check_legend(fig, ax, leg, rend, name):
    # the legend tests of check_layout for one legend
    if True:
        lb = leg.get_window_extent(rend)
        ab = ax.bbox
        if lb.x0 < ab.x0 - 0.5 or lb.y0 < ab.y0 - 0.5 or lb.x1 > ab.x1 + 0.5 or lb.y1 > ab.y1 + 0.5:
            raise RuntimeError(f"{name}: legend leaves the axes box")
        for line in ax.get_lines():
            if line.get_gid() == "ref":           # horizontal reference lines may pass under a legend
                continue
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
    ax[1].text(0, 100 * clean + 1.2, "error-free text", fontsize=5.8, color="0.35", ha="left", va="bottom")
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
    and, from data/pm/pm_content_ov_snr.csv (exp_pm_content_ovfig.py --part
    snr), the adapted masks on the same blocks with the overlap chosen for
    rho = 0.9 (rows role best, split test), the only curves at rho = 0.9;
    OMA, coded OMA and the two-level adapted masks do not change with rho.
    Points that rest on fewer than MIN_ERR = 20 token errors (column
    errors) are not drawn, nor points without errors. Colour and marker
    mark the scheme, line style and marker face the tap class (solid line
    and filled marker 4 taps, dashed line and hollow marker 8 taps); the
    legend shows the tap classes by the marker face only, without a line,
    so that they do not read as two more schemes (author's choice), in a
    second framed box beside the scheme legend. The SER axis is cut at 1e-6
    (author's choice); the six 8-tap points below it, at 18 and 20 dB, leave
    through the lower edge and stay in the CSV files."""
    from matplotlib.lines import Line2D
    rs = rows("pm_semfig2.csv")
    fig = plt.figure(figsize=(W43, H43))
    fig.subplots_adjust(**BOX43)
    ax = fig.add_subplot(111)
    # one legend vocabulary for Figs. 2 to 4: Static (static entry), Adapted (adapted
    # masks), ", rho = 0.9" for the masks with the overlap chosen for rho = 0.9 (pink)
    sty = {"oma": (*STYLE["oma"], "OMA"),
           "coded_il_comb": (*STYLE["coded_comb"], "Coded OMA"),
           "todma": (*STYLE["todma"], "ToDMA"),
           "static_class_L16": (*STYLE["static"], "Static"),
           "coded_il_adapted": (*STYLE["coded_adapted"], "Coded OMA, adapted"),
           "adapted_B2": (*STYLE["adapted"], "Adapted")}
    hs = {}
    for sc, (col, mk, lab) in sty.items():
        for cn, ls in (("fs4", "-"), ("fs8", "--")):
            pts = sorted((float(r["snr_db"]), float(r["ser_sem"])) for r in rs
                         if r["channel"] == cn and r["scheme"] == sc and float(r["ser_sem"]) > 0
                         and int(r["errors"]) >= MIN_ERR)
            mfc = "white" if ls == "--" else col
            # coded OMA and ToDMA mark 2, 6, ..., 18 dB, between the markers of the
            # curves they nearly coincide with (markevery offset, so no marker hides another)
            me = (1, 2) if sc in ("coded_il_comb", "coded_il_adapted", "todma") else 2
            ax.plot([q[0] for q in pts], [q[1] for q in pts], color=col, ls=ls, marker=mk,
                    markevery=me, mfc=mfc, label=f"{lab}, {cn}")
        hs[sc] = Line2D([], [], color=col, ls="-", marker=mk, label=lab)
    col_o, mk_o = STYLE["adapted_ov"]                # overlap chosen for rho = 0.9
    ro = rows("pm_content_ov_snr.csv")
    for cn, ls in (("fs4", "-"), ("fs8", "--")):
        pts = sorted((float(r["snr_db"]), float(r["ser_sem"])) for r in ro
                     if r["channel"] == cn and r["role"] == "best" and r["split"] == "test"
                     and float(r["ser_sem"]) > 0 and int(r["errors"]) >= MIN_ERR)
        # markers at 2, 6, ..., 18 dB, between those of the two-level adapted masks
        ax.plot([q[0] for q in pts], [q[1] for q in pts], color=col_o, ls=ls, marker=mk_o,
                markevery=(1, 2), mfc="white" if ls == "--" else col_o,
                label=f"Adapted masks, overlap, rho 0.9, {cn}")
    hs["adapted_ov"] = Line2D([], [], color=col_o, ls="-", marker=mk_o,
                              label=r"Adapted, $\rho=0.9$")
    # legend order: the proposal first, then the compared schemes in the
    # order of the paragraph on the compared schemes
    hs = [hs[k] for k in ("static_class_L16", "adapted_B2", "adapted_ov", "oma", "coded_il_comb",
                          "coded_il_adapted", "todma")]
    # tap classes by the marker face only, without a line (author's choice)
    hs_taps = [Line2D([], [], color="0.3", ls="none", marker="o", mfc="0.3", label="4 taps"),
               Line2D([], [], color="0.3", ls="none", marker="o", mfc="white", label="8 taps")]
    ax.set_yscale("log")
    ax.set_ylim(1e-6, 2.0)                     # cut at 10^-6 (author): curves below leave through
                                               # the lower edge; the top spine clears the markers at 0 dB
    ax.set_xlim(-0.5, 20.5)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("SER")
    ax.set_xticks([0, 4, 8, 12, 16, 20])
    ax.set_yticks([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0])
    ax.grid(True, which="both")
    leg = legend_in(ax, handles=hs, loc="lower left", fontsize=LEG43, **LEGKW43)   # schemes
    ax.add_artist(leg)
    fig.canvas.draw()
    lb = leg.get_window_extent(fig.canvas.get_renderer()).transformed(ax.transAxes.inverted())
    legend_in(ax, handles=hs_taps, loc="lower left", bbox_to_anchor=(lb.x1 + 0.01, 0.0),
              fontsize=LEG43, **LEGKW43)       # tap classes, a second box beside the first
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


def fig_signal(rhos=(0.0, 0.7, 0.9), taps="fs8"):
    """Fig. 3: SER (semantic error rate) at 10 dB versus the mask size J
    (values per mask) for the shared-token probabilities rho = 0, 0.7 and 0.9,
    line style by rho for both schemes. Static entries (filled blue circles)
    trained at the 10 dB they serve under flat fading come from
    data/pm/pm_semfig4_static.csv (exp_pm_semfig34.py --part static) and, for
    rho = 0.7, pm_semfig4_static_rho0.7.csv (--part static --rho 0.7); each is
    the best of three training runs by the SER on separate validation frames
    (column ser_sem_val, failed runs with nan skipped), and the plotted SER
    comes from the test frames (column ser_sem). Adapted masks (hollow
    markers) at 8 taps, where they gain most over the static entries
    (author's choice), with J blocks of 64/J real dimensions (the masks
    overlap when J < U): at rho = 0 the two-level masks (red squares,
    pm_semfig4_adapted.csv, exp_pm_semfig4a.py), whose best overlap is zero
    (pm_content_ov.csv), and at rho = 0.7 and 0.9 the masks with the overlap
    chosen for rho (red hollow squares, pm_content_ov_j_rho0.7.csv and
    pm_content_ov_j.csv, exp_pm_content_ovfig.py --part jsweep --rho 0.7 and
    --part jsweep, rows role best, split test). Coded OMA, which depends on
    neither J nor rho, gives two constant reference lines with the markers of
    Fig. 2: under flat fading (pm_content_ov_codedref.csv,
    exp_pm_content_ovfig.py --part codedref) for the static entries, and at
    8 taps on the blocks of the adapted masks (pm_semfig2.csv,
    coded_il_adapted) for the adapted masks. One framed legend column at the
    lower left names the four schemes with one marker and one colour each
    (every static entry a blue circle, every adapted mask a red square, as in
    the legend); rho is written on the graph with arrows that end in small
    upright ellipses on the curves and is also marked by the line style,
    solid 0, dash-dot 0.7, dashed 0.9 (author's choice, so that the curves
    keep most of the height of the axes)."""
    from matplotlib.legend_handler import HandlerTuple
    from matplotlib.lines import Line2D
    stat = defaultdict(dict)                        # rho -> J -> (validation SER, test SER)
    for r in rows("pm_semfig4_static.csv") + rows("pm_semfig4_static_rho0.7.csv"):   # trained at 10 dB
        if r["ser_sem"] in ("", "nan") or r["ser_sem_val"] in ("", "nan"):
            continue
        rho, J, key = float(r["rho"]), int(r["values"]), float(r["ser_sem_val"])
        if J not in stat[rho] or key < stat[rho][J][0]:
            stat[rho][J] = (key, float(r["ser_sem"]))
    adap = {0.0: {int(r["values"]): float(r["ser_sem"]) for r in rows("pm_semfig4_adapted.csv")
                  if r["channel"] == taps and float(r["snr_db"]) == 10}}
    for rho, fname in ((0.7, "pm_content_ov_j_rho0.7.csv"), (0.9, "pm_content_ov_j.csv")):
        adap[rho] = {int(r["J"]): float(r["ser_sem"]) for r in rows(fname)
                     if r["channel"] == taps and r["role"] == "best" and r["split"] == "test"}
    fig = plt.figure(figsize=(W43, H43))
    fig.subplots_adjust(**BOX43)
    ax = fig.add_subplot(111)
    lstyle = {0.0: "-", 0.7: "-.", 0.9: "--"}
    sshade = {r: STYLE["static"][0] for r in rhos}          # one colour per scheme, as in the legend
    ashade = {r: STYLE["adapted"][0] for r in rhos}
    amark = {r: STYLE["adapted"][1] for r in rhos}
    for rho in rhos:                                # legend column 1: Static
        xs = sorted(stat[rho])
        ax.plot(xs, [stat[rho][q][1] for q in xs], color=sshade[rho], marker=STYLE["static"][1],
                ls=lstyle[rho], mfc=sshade[rho], label=rf"Static, $\rho={rho:g}$")
    js_ = [1, 2, 4, 8, 16, 32, 64]                 # coded OMA: constant over J and rho
    cr = [float(r["ser_sem"]) for r in rows("pm_content_ov_codedref.csv")
          if r["channel"] == "flat" and float(r["snr_db"]) == 10]
    ca = [float(r["ser_sem"]) for r in rows("pm_semfig2.csv")
          if r["channel"] == taps and r["scheme"] == "coded_il_adapted" and float(r["snr_db"]) == 10]
    assert len(cr) == 1 and len(ca) == 1
    ax.plot(js_, [cr[0]] * len(js_), color=STYLE["coded_comb"][0], marker=STYLE["coded_comb"][1],
            ls=":", lw=1.0, mfc=STYLE["coded_comb"][0], label="Coded OMA", gid="ref")
    for rho in rhos:                                # legend column 2: Adapted
        xs = sorted(adap[rho])
        ax.plot(xs, [adap[rho][q] for q in xs], color=ashade[rho], marker=amark[rho], ls=lstyle[rho],
                mfc="white", label=rf"Adapted, $\rho={rho:g}$", **({"ms": 3.6} if amark[rho] == "D" else {}))
    ax.plot(js_, [ca[0]] * len(js_), color=STYLE["coded_adapted"][0], marker=STYLE["coded_adapted"][1],
            ls=":", lw=1.0, mfc="white", label="Coded OMA, adapted", gid="ref")
    C_JU, C_FULL = "#2ca02c", "#9467bd"             # boundary lines and their labels (author)
    ax.axvline(64, color=C_FULL, lw=1.4, ls="--", zorder=1.8, label="_nolegend_")   # dashed as J = U
    ax.text(58, 1.3, "full-length mask", ha="right", va="top",
            fontsize=ANN43, color=C_FULL)                       # at the top, left of the x = 64 line

    def at(curve, x):                               # SER of a curve at mask size x, log-log
        xs_ = sorted(curve)
        return float(np.exp(np.interp(np.log2(x), np.log2(xs_), np.log([curve[q] for q in xs_]))))

    from matplotlib.patches import Ellipse
    from matplotlib.transforms import ScaledTranslation

    def arrow(txt, x, y, dx, dy, col):              # rho written on the graph with an arrow (author)
        ax.add_patch(Ellipse((0, 0), width=0.035, height=0.06, fill=False, ec=col, lw=0.6, zorder=5,
                             transform=fig.dpi_scale_trans + ScaledTranslation(x, y, ax.transData)))
        ax.annotate(txt, xy=(x, y), xytext=(dx, dy), textcoords="offset points", fontsize=ANN43,
                    color=col, ha="center", va="center",
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=0.6, mutation_scale=6,
                                    shrinkA=1.0, shrinkB=2.2))
    sv = {r: {q: v[1] for q, v in stat[r].items()} for r in rhos}
    # offsets in points from the curve: static labels in one row above, adapted labels beside
    arrow(r"$\rho=0$", 12, at(sv[0.0], 12), -6, 22, sshade[0.0])
    arrow(r"$\rho=0.7$", 22.6, at(sv[0.7], 22.6), -6, 20, sshade[0.7])   # below "full-length mask"
    arrow(r"$\rho=0.9$", 45, at(sv[0.9], 45), -6, -22, sshade[0.9])
    arrow(r"$\rho=0$", 11.3, at(adap[0.0], 11.3), 14, 18, ashade[0.0])
    arrow(r"$\rho=0.9$", 22.6, at(adap[0.9], 22.6), -16, -14, ashade[0.9])
    arrow(r"$\rho=0.7$", 45, at(adap[0.7], 45), -14, -17, ashade[0.7])
    ax.axvline(4, ymin=0.36, color=C_JU, lw=1.4, ls="--", zorder=1.8, label="_nolegend_")  # above the legend
    ax.text(3.7, 1.25, "$J=U$", ha="right", va="top", fontsize=ANN43, color=C_JU)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks([1, 2, 4, 8, 16, 32, 64])
    ax.set_xticklabels(["1", "2", "4", "8", "16", "32", "64"])
    ax.set_xlim(0.8, 80)
    ax.set_ylim(1.2e-4, 2.0)                                    # room for the legend column
    ax.set_yticks([1e-3, 1e-2, 1e-1, 1.0])
    ax.set_xlabel("Mask size $J$ (values per mask)")
    ax.set_ylabel("SER at 10 dB")
    ax.grid(True, which="both")
    # one framed legend column at the lower left, as in Fig. 2 (author): the schemes, one marker
    # each; rho is written on the graph; the coded OMA lines may pass under the legend
    h_st = Line2D([], [], color=STYLE["static"][0], marker=STYLE["static"][1], ls="-")
    h_a0 = Line2D([], [], color=STYLE["adapted"][0], marker=STYLE["adapted"][1], mfc="white", ls="-")
    h_co = Line2D([], [], color=STYLE["coded_comb"][0], marker=STYLE["coded_comb"][1], ls=":")
    h_ca = Line2D([], [], color=STYLE["coded_adapted"][0], marker=STYLE["coded_adapted"][1],
                  mfc="white", ls=":")
    legend_in(ax, handles=[h_st, h_a0, h_co, h_ca],
              labels=["Static", "Adapted", "Coded OMA", "Coded OMA, adapted"], loc="lower left",
              fontsize=LEG43, **LEGKW43)
    save(fig, "fig4_pm.pdf", tight=False)


# Update intervals F of Fig. 4 and their styles: one red ramp for the adapted masks (darker =
# rarer updates), solid lines with filled markers and dashed lines with hollow markers in turn
F_STYLE = ((1, "#fb8a6a", "-", True), (3, "#f0553a", "--", True), (10, "#d62728", "-", True),
           (30, "#a81419", "--", True), (100, "#6b0a0f", "-", True))   # filled: four taps (Figs. 2-4)


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
    curve. At rho = 0.9 (dash-dotted, data/pm/pm_content_ov_j.csv, fs4, rows
    role best, split test) the adapted masks with the overlap chosen for
    rho = 0.9, updated every F = 100 frames (the overlap is a value of the
    entry and adds no bits), and the static entry trained for the class and
    rho = 0.9 at 10 dB (pm_content_fs.csv, static_trained, fs4) as a thin
    line."""
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
    ter_o = {int(r["J"]): float(r["ter"]) for r in rows("pm_content_ov_j.csv")
             if r["channel"] == "fs4" and r["role"] == "best" and r["split"] == "test"}
    st_o = [float(r["ter"]) for r in rows("pm_content_fs.csv")
            if r["scheme"] == "static_trained" and r["channel"] == "fs4"
            and float(r["snr_db"]) == 10 and abs(float(r["rho"]) - 0.9) < 1e-9]
    assert len(st_o) == 1
    st_o = st_o[0]

    def bits(v):
        return v * math.log2(U_) if v >= U_ else U_ * math.ceil(math.log2(v)) if v > 1 else 0.0

    fig = plt.figure(figsize=(W43, H43))
    fig.subplots_adjust(**BOX43)
    ax = fig.add_subplot(111)
    xs = sorted(ter)
    mk = STYLE["adapted"][1]                       # every curve is an adapted mask
    c_st = STYLE["static"][0]
    ax.axhline(st, color=c_st, lw=1.9, ls="-", zorder=1.8, label="_nolegend_")   # static entry, bold
    ax.text(0.86, st * 0.89, "Static", color=c_st, fontweight="bold", fontsize=ANN43 + 0.5,
            ha="left", va="top")                    # below the line, clear of the rho = 0.9 curve
    best = {}
    for F, c_, ls, filled in F_STYLE:
        loss = [ter[v] + bits(v) / (F * pay) for v in xs]       # 1 - eta of eq. (9)
        ax.plot(xs, loss, color=c_, marker=mk, ls=ls, mfc=c_ if filled else "white", label=f"$F={F}$")
        k = int(np.argmin(loss))
        best[F] = (xs[k], loss[k])
        ax.plot([xs[k]], [loss[k]], ls="none", marker="*", ms=9, mfc="none", mec="k", mew=0.8,
                label="Best $J$" if F == F_STYLE[-1][0] else "_nolegend_", zorder=4)
    c_o, mk_o = STYLE["adapted_ov"]                 # rho = 0.9: overlap, F = 100
    xo = sorted(ter_o)
    loss_o = [ter_o[v] + bits(v) / (100 * pay) for v in xo]
    ax.plot(xo, loss_o, color=c_o, marker=mk_o, ls="-.", mfc=c_o,
            label=r"$F=100$, $\rho=0.9$")
    ko = int(np.argmin(loss_o))
    best["100, rho 0.9"] = (xo[ko], loss_o[ko])
    ax.plot([xo[ko]], [loss_o[ko]], ls="none", marker="*", ms=9, mfc="none", mec="k", mew=0.8,
            label="_nolegend_", zorder=4)
    ax.axhline(st_o, color=c_st, lw=0.9, ls="-.", zorder=1.8, label="_nolegend_")
    ax.text(0.86, st_o * 0.87, r"Static, $\rho=0.9$", color=c_st, fontsize=ANN43,
            ha="left", va="top")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks([1, 2, 4, 8, 16, 32, 64])
    ax.set_xticklabels(["1", "2", "4", "8", "16", "32", "64"])
    ax.set_xlim(0.8, 80)
    ax.set_ylim(5e-4, 4)                            # room below the curves for the legend
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
    legend_in(ax, loc="lower right", fontsize=LEG43, ncol=2, **LEGKW43)   # as Figs. 2 and 3
    save(fig, "fig5_pm.pdf", tight=False)
    print("Fig. 4: best J and loss per F:",
          {F: (j, round(l, 4)) for F, (j, l) in best.items()}, "static entry", round(st, 4),
          "static entry rho 0.9", round(st_o, 4))


if __name__ == "__main__":
    fig_codebook()
    fig_channel()
    fig_content()
    fig_signal()
    fig_tradeoff()
