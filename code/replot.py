"""Canonical replot script: regenerates every result figure from the
CSVs in ../data/ and writes paper-ready PDFs to ../fig/. No experiment
is rerun here. All result plots share one canvas and axes rectangle
(8:6 box), matching the geometry used in the earlier letters.

  fig_snr.pdf     : SER vs SNR at V = 65,536 (Fig. 2)
  fig_scaling.pdf : SER at 10 dB vs V, best factorization (Fig. 3)
  fig_real.pdf    : token error rate vs SNR on real BERT streams (Fig. 4)
  fig_conv.pdf    : SER at 10 dB vs training iteration (Fig. 5)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIG = ROOT / "fig"
FIG.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman"],
    "font.size": 9,
    "axes.labelsize": 9,
    "legend.fontsize": 6.6,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.linewidth": 0.4,
    "grid.alpha": 0.6,
    "lines.linewidth": 1.3,
    "lines.markersize": 3.4,
    "figure.figsize": (3.15, 2.36),
    "pdf.fonttype": 42,
})
AXES_RECT = dict(left=0.175, right=0.965, top=0.965, bottom=0.165)

STY = {
    "SSE":     ("#c0392b", "o", "-",  "Proposed SSE"),
    "Full":    ("#2c5fa8", "s", "--", "Full codebook"),
    "ToDMA":   ("#8e44ad", "P", (0, (3, 1, 1, 1)), "ToDMA"),
    "OMA":     ("#7f8c8d", "^", ":",  "OMA"),
    "Single":  ("#16a085", "v", "-.", "Single-user bound"),
    "UB":      ("#c0392b", "",  "--", "Union bound"),
}


def save(fig, name):
    fig.subplots_adjust(**AXES_RECT)
    fig.savefig(FIG / f"{name}.pdf")
    plt.close(fig)
    print("[OK]", name)


def cols(path):
    with open(path) as f:
        return f.readline().strip().split(",")


def fig_snr():
    p = DATA / "snr_sweep.csv"
    hdr = cols(p)
    d = np.loadtxt(p, delimiter=",", skiprows=1)
    ps = DATA / "sota_snr.csv"
    hs = cols(ps)
    ds = np.loadtxt(ps, delimiter=",", skiprows=1)
    snr = d[:, 0]
    series = {m: d[:, hdr.index(f"ser_{m}")]
              for m in ["SSE", "Full", "OMA", "Single"]}
    series.update({m: ds[:, hs.index(f"ser_{m}")]
                   for m in ["ToDMA"]})
    fig, ax = plt.subplots()
    for m in ["SSE", "Full", "ToDMA", "OMA", "Single"]:
        c, mk, ls, lb = STY[m]
        ax.semilogy(snr, series[m], linestyle=ls, marker=mk, color=c,
                    label=lb)
    c, mk, ls, lb = STY["UB"]
    ax.semilogy(snr, d[:, hdr.index("ser_UB")], linestyle=ls, color=c,
                alpha=0.55, linewidth=1.0, label=lb)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("SER")
    ax.set_xlim(0, 20)
    ax.set_xticks(range(0, 21, 4))
    ax.legend(loc="lower left", ncol=1, handlelength=2.2,
              labelspacing=0.25, borderpad=0.3)
    save(fig, "fig_snr")


def fig_scaling():
    p = DATA / "scaling.csv"
    hdr = cols(p)
    d = np.loadtxt(p, delimiter=",", skiprows=1)
    V = d[:, 0]
    x = np.log2(V)
    fig, ax = plt.subplots()
    for m in ["SSE", "Full", "ToDMA", "OMA"]:
        key = f"ser10_{m}"
        if key not in hdr:
            continue
        c, mk, ls, lb = STY[m]
        ax.semilogy(x, d[:, hdr.index(key)], linestyle=ls, marker=mk,
                    color=c, label=lb)
    ax.set_xlabel(r"$\log_2 V$")
    ax.set_ylabel("SER")
    ax.set_xticks(range(4, 17, 2))
    # compact decimal ticks: sub-decade ranges otherwise render wide
    # mantissa labels that push the y-label off the fixed canvas
    from matplotlib.ticker import NullFormatter
    ax.set_yticks([0.03, 0.1, 0.3, 0.6])
    ax.set_yticklabels(["0.03", "0.1", "0.3", "0.6"])
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.legend(loc="lower right", labelspacing=0.25, borderpad=0.3)
    save(fig, "fig_scaling")


def fig_real():
    p = DATA / "real_ter.csv"
    hdr = cols(p)
    d = np.loadtxt(p, delimiter=",", skiprows=1)
    ps = DATA / "sota_real.csv"
    hs = cols(ps)
    ds = np.loadtxt(ps, delimiter=",", skiprows=1)
    snr = d[:, 0]
    series = {m: d[:, hdr.index(f"ter_{m}")] for m in ["SSE", "Full", "OMA"]}
    series.update({m: ds[:, hs.index(f"ter_{m}")]
                   for m in ["ToDMA"]})
    fig, ax = plt.subplots()
    for m in ["SSE", "Full", "ToDMA", "OMA"]:
        c, mk, ls, lb = STY[m]
        ax.semilogy(snr, series[m], linestyle=ls, marker=mk, color=c,
                    label=lb)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("TER")
    ax.set_xlim(0, 20)
    ax.set_xticks(range(0, 21, 4))
    ax.legend(loc="lower left", labelspacing=0.25, borderpad=0.3)
    save(fig, "fig_real")


def fig_conv():
    import json
    p = DATA / "convergence.csv"
    rows = [l.strip().split(",") for l in open(p)][1:]
    fig, ax = plt.subplots()
    for m in ["SSE", "Full"]:
        c, mk, ls, lb = STY[m]
        it = [int(r[0]) for r in rows if r[4] == m]
        ser = [float(r[3]) for r in rows if r[4] == m]
        ax.semilogy(it, ser, linestyle=ls, color=c, label=lb,
                    marker=mk, markevery=5)
    # training-free references at 10 dB
    snr = np.loadtxt(DATA / "snr_sweep.csv", delimiter=",", skiprows=1)
    h_snr = cols(DATA / "snr_sweep.csv")
    oma10 = snr[snr[:, 0] == 10, h_snr.index("ser_OMA")][0]
    sota = np.loadtxt(DATA / "sota_snr.csv", delimiter=",", skiprows=1)
    h_sota = cols(DATA / "sota_snr.csv")
    td10 = sota[sota[:, 0] == 10, h_sota.index("ser_ToDMA")][0]
    c, _, ls, lb = STY["ToDMA"]
    ax.axhline(td10, linestyle=ls, color=c, linewidth=1.0, label=lb)
    c, _, ls, lb = STY["OMA"]
    ax.axhline(oma10, linestyle=ls, color=c, linewidth=1.0, label=lb)
    ax.set_xlabel("Training iteration")
    ax.set_ylabel("SER")
    ax.set_xlim(0, 4000)
    ax.set_xticks(range(0, 4001, 1000))
    ax.set_ylim(top=2.4)
    from matplotlib.ticker import NullFormatter
    ax.set_yticks([0.3, 0.5, 1, 2])
    ax.set_yticklabels(["0.3", "0.5", "1", "2"])
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.legend(loc="upper center", ncol=2, labelspacing=0.25,
              borderpad=0.3, columnspacing=0.9)
    save(fig, "fig_conv")


if __name__ == "__main__":
    fig_snr()
    fig_scaling()
    fig_real()
    fig_conv()
