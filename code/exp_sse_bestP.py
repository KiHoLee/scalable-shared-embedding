"""Per-vocabulary factorization selection for the Proposed SSE curve.

The construction contains the unstructured codebook as the P=1 special
case, so the proposal is a family over factorizations V = Vu^P. The
original scaling sweep fixed P=4 for every V, which handicaps the
proposal at small V where an unconstrained codebook is trainable. This
script evaluates P in {1, 2, 4} per grid vocabulary and selects the
best at 10 dB:

  P=1: architecturally identical to the Full codebook scheme (B = E,
       full-length masks), so its measured ser10_Full/ser16_Full
       columns are reused verbatim.
  P=2: SSE(P=2, Vu = sqrt(V)) trained fresh (cached as
       sse_Vu{vu}_P2.pt), same budget.
  P=4: the existing ser10_SSE/ser16_SSE columns.

Outputs: data/scaling_sse_P.csv (per-P raw values and the selected P),
and scaling.csv's ser10_SSE/ser16_SSE columns rewritten with the
selected-factorization values.
"""
from __future__ import annotations

import math

from sse_lib import DATA, P_MAX, SEED, set_seed, eval_ser_sse, write_csv
from exp_main import get_or_train_sse, EVAL_FRAMES

VS = (16, 81, 256, 625, 1296, 4096, 10000, 20736, 65536)


def main():
    set_seed(SEED)
    lines = (DATA / "scaling.csv").read_text().strip().split("\n")
    hdr = lines[0].split(",")
    col = {name: hdr.index(name) for name in hdr}
    table = {int(float(l.split(",")[0])): l.split(",") for l in lines[1:]}

    rows = []
    best = {}
    for V in VS:
        f = table[V]
        p1_10 = float(f[col["ser10_Full"]])
        p1_16 = float(f[col["ser16_Full"]])
        p4_10 = float(f[col["ser10_SSE"]])
        p4_16 = float(f[col["ser16_SSE"]])
        vu2 = math.isqrt(V)
        assert vu2 * vu2 == V
        m2, _ = get_or_train_sse(2, f"sse P=2 Vu={vu2}", vu=vu2)
        p2_10, p2_16 = eval_ser_sse(m2, [10.0, 16.0], frames=EVAL_FRAMES)
        cands = {1: (p1_10, p1_16), 2: (p2_10, p2_16), 4: (p4_10, p4_16)}
        sel = min(cands, key=lambda P: cands[P][0])
        best[V] = (*cands[sel], sel)
        rows.append((V, p1_10, p2_10, p4_10, p1_16, p2_16, p4_16, sel))
        print(f"[bestP] V={V}: P1={p1_10:.5f} P2={p2_10:.5f} "
              f"P4={p4_10:.5f} -> P={sel}", flush=True)

    write_csv(DATA / "scaling_sse_P.csv",
              ["V", "ser10_P1", "ser10_P2", "ser10_P4",
               "ser16_P1", "ser16_P2", "ser16_P4", "selectedP"], rows)

    out = [lines[0]]
    for ln in lines[1:]:
        f = ln.split(",")
        V = int(float(f[0]))
        f[col["ser10_SSE"]] = f"{best[V][0]:.10g}"
        f[col["ser16_SSE"]] = f"{best[V][1]:.10g}"
        out.append(",".join(f))
    (DATA / "scaling.csv").write_text("\n".join(out) + "\n")
    print("[csv] scaling.csv SSE columns updated with selected-P values")


if __name__ == "__main__":
    main()
