# Scalable Shared Embedding (SSE): Reproducibility Package

Code and raw results for

> K.-H. Lee, H.-H. Choi, and J.-R. Lee, "Scaling Shared-Embedding Multiple
> Access to Large Vocabularies via Periodic Codebook Reuse,"
> submitted to *IEEE Wireless Communications Letters*.

The construction divides a d-dimensional embedding frame into P periods,
reuses one unit codebook of V_u codewords in every period, and carries the
vocabulary index in its base-V_u digits, so V = V_u^P indices are served
with V_u·d/P stored values and a training cost per step independent of V.

## Contents

| Folder | Content |
|---|---|
| `code/` | Shared library `sse_lib.py`, one script per experiment, and the replot scripts |
| `data/` | Raw result of every figure and table (CSV/JSON) and the small trained models (`*.pt` below 200 kB) |
| `fig/` | Figure PDFs regenerated from `data/` (`*_revision.pdf` belong to the current manuscript) and the TikZ source of Fig. 1 |

The large full-codebook checkpoints (`full_V*.pt` for V ≥ 4,096, the
long-budget models, about 300 MB) are not stored. The scripts retrain them
with the fixed seeds when they are missing.

## Figure and table map

| Item | Script | Data |
|---|---|---|
| Fig. 1 (system) | `fig/fig1_integrated_revision.tex` (standalone TikZ) | none |
| Fig. 2 (SER vs SNR) | `replot_revision.py snr` | `snr_sweep.csv` (`exp_main.py`), `sota_snr.csv` (`exp_sota.py`), `rev_snr_fullmd.csv` (`exp_rev_fullmd.py`) |
| Fig. 3 (SER vs training MACs) | `replot_revision.py conv` | `rev_budget_curve_{sse,full,sampled,hier}_mean.csv` (3-seed means by `exp_rev_budget_eval.py` of `rev_budget_curve_{scheme}[_s2|_s3].csv` from `exp_rev_budget.py --seed {1,2,3} --log-frames 200000`) |
| Table I | none | `snr_sweep.csv`, `sota_snr.csv`, `rev_snr_fullmd.csv`, `rev_budget_final_mean.csv` (3-seed means; per seed `rev_budget_final_{scheme}[_s2|_s3].csv`), `complexity.json` |
| Table II | none | `rev_robust.csv` and `rev_robust_fullmd.csv` (`exp_rev_robust.py`, `rev_channel.py`), `rev_users.csv`, `rev_users_fixedload.csv` (`exp_rev_users_fixedload.py`) |
| Analysis checks (Sec. III-B) | `exp_rev_analysis.py`, `exp_rev_scalingbound.py`, `exp_rev_fullmd.py` | `rev_norms.csv`, `rev_norms_full.json`, `rev_decoders.csv`, `rev_rho.json`, `rev_indep.csv`, `rev_isotropy.json`, `rev_scaleP.csv`, `rev_factor.csv`, `rev_scalingbound.csv` |
| CSI-aware retraining (response letter, R1.6) | `exp_rev_csitrain.py` | `rev_csitrain.csv` |
| Real AG News streams (Sec. IV, letter Fig. R1 via `replot_revision.py real`) | `exp_rev_semantic.py` | `rev_semantic.csv`, `rev_energy.json` |
| Original-submission experiments (not in the revised manuscript) | `exp_main.py`, `exp_sota.py`, `exp_real.py`, `exp_scaling_dense.py`, `exp_sse_bestP.py`, `exp_att_seeds.py`, `exp_d128.py`, `exp_lmmse.py`, `replot.py` | `convergence.csv`, `real_ter.csv`, `real_stats.json`, `sota_real.csv`, `scaling_att_seeds.csv`, `d128_snr.json`, `curve_*.json` |
| Masked-norm histogram (response letter, Fig. R3, `replot_revision.py norms`) | `exp_rev_analysis.py`, `exp_rev_fullmd.py` | `rev_norms.csv`, `rev_norms_full.json` |
| Seed spread | `exp_rev_seeds.py`, `exp_rev_seeds_long.py` | `rev_seeds_sse.csv`, `rev_seeds_sse_20k.csv` |
| Fixed-load user sweep | `exp_rev_users_fixedload.py` | `rev_users_fixedload.csv` |
| Vocabulary sweep | `replot_revision.py scaling` | `scaling.csv`, `scaling_sse_P.csv`, `rev_scaling_fullmd.csv` |

`replot_revision.py` reads only `data/` and writes `fig/`. It fails if any
axis label is clipped or any legend covers a curve.

## Conventions

- Real-vector model. A complex subcarrier carries two real dimensions.
  Unit average frame energy. SNR γ = 1/(dσ²). Flat Rayleigh gain with
  h² ~ Exp(1), known at the receivers unless a script states otherwise.
- Main configuration: U = 4, d = 64, V_u = 16, P = 4, V = 65,536.
- Training: Adam, learning rate 3e-3, batch 256, 4,000 iterations, training
  SNR uniform in 0-20 dB. The long budget uses 2e5 iterations with
  half-cosine decay.
- The full codebook and its sampled and hierarchical softmax variants are
  decoded by exact minimum distance (`model.md = True`). SSE uses the
  correlation rule.
- Seeds: training seed 1 and evaluation seed 777. The reported SSE main
  model (`sse_P4.pt`) is the best of six training runs; the other five
  seeds are in `rev_seeds_sse.csv` (`exp_rev_seeds.py`). Long-budget
  (2e5-iteration) results are means of training seeds 1, 2, 3
  (`exp_rev_budget_eval.py`).
- GPU training is not bit-reproducible: re-running `exp_main.py` with
  seed 1 gives an SER near 0.30 at 10 dB rather than the 0.257 of the
  released `sse_P4.pt`. The released checkpoints are the reported models;
  expect the stated seed spread when retraining.

## Requirements

Python 3.10 or later, PyTorch 2.x (CUDA optional, `SSE_DEVICE` overrides
the device), NumPy, Matplotlib, and PyMuPDF for figure checks. The
real-data stage also needs `transformers` and `datasets`. It downloads
`bert-base-uncased`, `textattack/bert-base-uncased-ag-news`, and the AG
News test split.

Run from `code/`, for example:

```bash
cd code
python exp_main.py
python replot_revision.py
```
