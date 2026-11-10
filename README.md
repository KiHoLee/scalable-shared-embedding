# Scalable Shared Embedding: Reproducibility Package

Code and raw results for

> K.-H. Lee, H.-H. Choi, and J.-R. Lee, "Scaling Shared-Embedding Multiple
> Access to Foundation-Model Vocabularies via Periodic Masks,"
> submitted to *IEEE Wireless Communications Letters*.

Corresponding authors: Hyun-Ho Choi and Jung-Ryun Lee.

Each user's token is sent as its row of the token-embedding table of a
pretrained language model (BERT base uncased, with the GTR base model as
an alternative), reduced to 64 real dimensions by a projection whitened under
the source token statistics. The transmitter and every receiver hold this
codebook and never train it. Only one mask per user is trained, the frame
is the sum of the masked codewords of all users, and each receiver decodes
its token by weighted minimum distance over the whole vocabulary. The best
mask, often partially overlapping across users, changes with the channel
state, the shared content and the number of users, so a transmitter keeps
many masks, and masks defined on periodic units (a short pattern repeated
along the frame, or one value per block of subcarriers about one coherence
bandwidth wide) cut the stored and signaled mask values several times with
little loss. Decoding errors land on tokens of similar meaning, so the
received text keeps its topic and meaning more often than under orthogonal
multiple access (OMA), most of all at low SNR.

The last section keeps the earlier design of this repository, periodic
codebook reuse, with its code and data.

Every simulation parameter (system, channel classes, codebook, masks and
training, decoder, compared schemes, shared content, metrics, parameters
per figure, seeds, software) is listed in [ENVIRONMENT.md](ENVIRONMENT.md).

## Contents

| Path | Content |
|---|---|
| `code/exp_pm.py` | Codebooks, mask training, pattern length and load, channel classes, bounds, frame energy of the masks |
| `code/exp_pm_ext.py` | Shared content, masks adapted to known gains, signaling, and the data of Figs. 2 to 5 |
| `code/exp_pm_chanhi.py` | Fig. 2 points of the static entry trained for each frequency-selective class, the adapted-mask points at 12 to 20 dB with 400,000 frames, and OMA placed on the blocks of the adapted masks |
| `code/exp_pm_oma_energy.py` | Frame energy of the OMA reference on the test streams |
| `code/replot_pm.py` | Every figure of the study from `data/pm/` (plotting only) |
| `code/sse_lib.py`, `code/rev_channel.py` | Shared helpers and the channel simulator, used by both designs |
| `data/pm/` | Every result of the periodic-mask study (28 CSV and JSON files) |
| `fig/fig1_pm.tex`, `fig/fig1_pm.pdf` | Fig. 1, standalone TikZ source and PDF |
| `fig/fig2_pm.pdf`, `fig/fig3_pm.pdf`, `fig/fig4_pm.pdf`, `fig/fig5_pm.pdf` | Figs. 2 to 5 |
| `fig/fig_codebook_pm.pdf` | Supplementary figure, SER and topic accuracy of the codebooks |

The trained masks are not stored. Each mask file also holds the fixed
codebook, so it takes about 7.8 MB (8.2 MB with the GTR codebook), and
the 70 files of a full run take about 0.55 GB. The scripts retrain them into
`data/pm/pm_models/` (git-ignored). One 3,000-step training takes about
20 s on one GPU.

## Figure and table map (periodic masks)

All data files are in `data/pm/`, and all scripts are in `code/`.
`replot_pm.py` draws Figs. 2 to 5 and the supplementary figure from these
files only. It refuses a figure whose legend covers a curve, a marker, a
reference line or an annotation, or whose text leaves the canvas.

A static entry holds the masks of all users trained offline for one
channel class, load and level of sharing, and the mask set holds one entry
for each of them. The values per mask J are the pattern length L of a
static entry or the number of blocks of an adapted mask, and J = 64 is the
full-length mask.

| Item | Script (part) | Data |
|---|---|---|
| Fig. 1, system | `fig/fig1_pm.tex` (standalone TikZ) | none |
| Fig. 2, SER versus SNR under 4-tap and 8-tap fading for OMA, the static entry trained for each class (L = 16) and masks adapted to known gains (blocks of two subcarriers) | `exp_pm_ext.py --part figs` (OMA, adapted masks below 12 dB), `exp_pm_chanhi.py --part class` (static entry), `exp_pm_chanhi.py --part hi` (adapted masks from 12 dB) | `pm_figs.csv`, graph `g1`, rows `fs4` and `fs8` of schemes `oma` and `adapted_B2`. `pm_figs_class.csv`, scheme `static_class_L16` (40,000 frames below 12 dB and 400,000 from 12 dB). `pm_figs_hi.csv`, scheme `adapted_B2` at 12 to 20 dB with 400,000 frames, which replace the `g1` values. The rows `static_L16` of `pm_figs.csv` and `pm_figs_hi.csv` (entry trained under flat fading) and the `flat` rows of `g1` are stored but not drawn. Points without errors are not drawn |
| Fig. 3, SER at 10 dB versus the shared-token probability rho (all users receive the same token), for masks without overlap, full overlap and masks trained at 10 dB (L = 16) | `exp_pm_ext.py --part content_snr` | `pm_content_snr.csv`, rows at 10 dB of schemes `no_overlap`, `full_overlap` and `trained_at_snr`; column `cross_energy` holds the mean cross-energy printed next to the trained curve. The file also stores 0 and 20 dB, masks trained on SNRs drawn from 0 to 20 dB (`trained_0_20dB`, the masks of `pm_figs.csv` graph `g2`), and the SER split into frames with a shared token (`ser_shared`) and frames with independent tokens (`ser_indep`) |
| Fig. 4, SER at 10 dB versus the values per mask J = 1 to 64, for static entries with L = J trained at 10 dB under flat fading and masks adapted with J blocks at 4 and 8 taps | `exp_pm_ext.py --part sig10` (static entries), `exp_pm_ext.py --part sig` (adapted masks) | `pm_sig10.csv` (schemes `static_at10dB` for independent content and `shared_rho0.9_at10dB` for rho = 0.9, three runs each, column `values` holds J) plotted as the best of three runs on separate validation frames (column `ser_val`), with the SER of the test frames (column `ser`). `pm_sig.csv`, scheme `adapted`: masks at 4 and 8 taps on J blocks of 64/J real dimensions, shared among users when J < 4. The static rows of `pm_sig.csv` (entries trained on SNRs from 0 to 20 dB) are stored but not drawn |
| Fig. 5, net efficiency of the adapted masks at 10 dB and four taps versus J, for an update every 1, 10 and 100 frames, with the static entry trained for the class as a reference line | `replot_pm.py` (`fig_tradeoff`, computed from the data, no simulation) | `pm_sig.csv` (scheme `adapted`, channel `fs4`) and `pm_figs_class.csv` (`fs4` at 10 dB). Net efficiency (U (1 − SER) log2 V − b_J/F)/(U log2 V) with U = 4 and V = 30,522, where b_J = J log2 U bits per update for J ≥ U and b_J = U⌈log2 J⌉ bits when J < U users share blocks. Stars mark the best J of each curve (4, 8 and 16) |
| Supplementary table, signaling per update of the adapted masks and SER at 10 dB | `exp_pm_ext.py --part adapt` | `pm_adapt.csv`. The SER of the static entry at 10 dB is in `pm_figs_class.csv` (entry trained for each class) and in graph `g1` of `pm_figs.csv` (entry trained under flat fading) |
| Supplementary check of Fig. 2, OMA on fixed adjacent subcarriers and on the blocks of two subcarriers that the adapted masks pick, 4 and 8 taps, 0 to 20 dB | `exp_pm_chanhi.py --part oma_adapt` | `pm_oma_adapt.csv`, schemes `oma` (reproduces the OMA curve of Fig. 2) and `oma_adapted_B2`, 200,000 frames per point. On the same blocks OMA errs at least as often as the static entry at every point |
| Supplementary figure `fig/fig_codebook_pm.pdf`, (a) SER and (b) topic accuracy versus SNR for the proposed BERT codebook, a random codebook and OMA | `exp_pm.py --part models` | `pm_models_eval.csv` (mean over seeds 1 to 3 for the proposed codebook), `pm_models_train.json` (accuracy on error-free text). The GTR codebook and the projections without source-statistics whitening are in the same files |
| Supplementary table, one common and one private token per user and frame with fixed masks | `exp_pm_ext.py --part rs` | `pm_rs.csv`. Schemes `shared_sic` and `shared_nosic` put the common codeword on every dimension and decode the private token with and without removing the common token first. Scheme `no_overlap_copies` sends copies of the common token on masks without overlap |

### Supporting results

| Result | Script (part) | Data |
|---|---|---|
| Pattern length for two, four and eight users, masks of one load used at another, training curves | `exp_pm.py --part period` | `pm_period_eval.csv`, `pm_period_train.json` |
| Pattern length per channel class, isotropy statistics of the codebook | `exp_pm.py --part plen` | `pm_plen_eval.csv`, `pm_plen_train.json` |
| Eight channel classes with masks trained for each class, the flat-trained masks used unchanged, a random codebook, and every class-trained mask applied in every class (rows `cross_class`) | `exp_pm.py --part channel` | `pm_channel_eval.csv`, `pm_channel_train.json` |
| Fixed comb pattern without overlap (L = U) and with pairs (L = 2U), for four and eight users | `exp_pm.py --part comb` | `pm_comb_eval.csv`, `pm_comb.json` |
| Union bounds on the SER and on the semantic distortion against Monte Carlo, unclipped | `exp_pm.py --part bound` | `pm_bound.csv` |
| The same bounds conditioned on the gain and clipped | `exp_pm.py --part bound2 --bound_nv 600` | `pm_bound2.csv` |
| The same bounds at a fixed unit gain | `exp_pm.py --part bound3 --bound_nv 600` | `pm_bound3.csv` |
| Frame energy of every saved mask set on the test streams | `exp_pm.py --part energy` | `pm_energy.json` |
| Frame energy of OMA on the test streams | `exp_pm_oma_energy.py` | `pm_oma_energy.csv` |
| Correlated content, rho = 0, 0.5, 0.9 and 1, trained masks with L = 16 and L = 64 against the comb pattern and full overlap | `exp_pm_ext.py --part corr` | `pm_corr.csv`, `pm_corr.json` (cross-energy per user and constant c) |
| Coherence-aware selection among the 24 assignments of four blocks of eight subcarriers, with and without water-filling | `exp_pm_ext.py --part csit` | `pm_csit.csv` |
| Receivers that cancel the other users' decoded tokens, with and without a test of the common-token hypothesis | `exp_pm_ext.py --part sic` | `pm_sic.csv` |
| Single-run signaling data (graph `g3`), which `pm_sig.csv` replaces in Fig. 4 | `exp_pm_ext.py --part figs` | `pm_figs.csv`, graph `g3` |

## Running the periodic-mask experiments

Run from `code/` in a fresh clone. Every script writes its results to
`data/pm/` and overwrites the released file of the same name (git restores
it), and `exp_pm.py` writes the trained masks to `data/pm/pm_models/`. The
minutes are wall-clock times of the released runs, each on one NVIDIA RTX
PRO 5000 Blackwell shared with other jobs, so they are approximate.

| Step | Command | Minutes | Needs |
|---|---|---|---|
| 1 | `python exp_pm.py --part models` | 12.4 | |
| 2 | `python exp_pm.py --part period` | 27.0 | |
| 3 | `python exp_pm.py --part channel` | 30.9 | |
| 4 | `python exp_pm.py --part plen` | 14.0 | |
| 5 | `python exp_pm.py --part bound` | 2.4 | |
| 6 | `python exp_pm.py --part comb` | 3.4 | |
| 7 | `python exp_pm.py --part energy` | 0.3 | steps 1 and 3 |
| 8 | `python exp_pm.py --part bound2 --bound_nv 600` | 2.0 | step 3 |
| 9 | `python exp_pm.py --part bound3 --bound_nv 600` | 1.2 | step 3 |
| 10 | `python exp_pm_ext.py --part corr` | 7.1 | |
| 11 | `python exp_pm_ext.py --part csit` | 1.0 | |
| 12 | `python exp_pm_ext.py --part adapt` | 10.2 | |
| 13 | `python exp_pm_ext.py --part sic` | 6.5 | |
| 14 | `python exp_pm_ext.py --part rs` | 1.0 | |
| 15 | `python exp_pm_ext.py --part figs` | 13.6 | steps 2 and 3 |
| 16 | `python exp_pm_ext.py --part sig` | 7.3 | step 2 |
| 17 | `python exp_pm_chanhi.py --part hi` | 1.4 | step 3 |
| 18 | `python exp_pm_chanhi.py --part class` | 0.8 | step 3 |
| 18a | `python exp_pm_chanhi.py --part oma_adapt` | below 1 | |
| 18b | `python exp_pm_ext.py --part content_snr` | 8.0 | |
| 18c | `python exp_pm_ext.py --part sig10` | 13.8 | |
| 19 | `python exp_pm_oma_energy.py` | below 1 (CPU) | |
| 20 | `python replot_pm.py` | below 1 | steps 1, 15 to 18c |

Steps 1 to 5 also run in one process as `python exp_pm.py --part all`
(about 87 min). Part `figs` loads `white_U4_L16_flat_s1.pt` (step 3) and
`bertw_U4_L{64,32,16,8,4,2,1}_flat_s1.pt` (step 2), and part `sig` loads
`bertw_U4_L{64,...,1}_flat_s{1,2,3}.pt` (step 2). Part `hi` of
`exp_pm_chanhi.py` loads `white_U4_L16_flat_s1.pt`, and part `class`
loads `white_U4_L16_fs4_s1.pt` and `white_U4_L16_fs8_s1.pt` (all from
step 3). The other parts of `exp_pm_ext.py` train or build their own
masks. Fig. 1 compiles inside `fig/` with `latexmk -pdf fig1_pm.tex`.

A quick test such as `python exp_pm.py --part period --smoke --iters 40
--tag _smoke` writes its masks to `data/pm/pm_models_smoke/` and its
results with the suffix `_smoke` (both git-ignored). Without `--tag` it
overwrites the released results. Keep the default `--book pwhite`, since
the mask file names do not record the codebook and a run with another
codebook replaces the masks that later parts load. The smoke option exists
only in `exp_pm.py` and trains only some of the masks, and `exp_pm_ext.py`
and `exp_pm_chanhi.py` always load the full masks from
`data/pm/pm_models/`.

### Known differences between a rerun and the released files

- `pm_bound.csv` uses the default `--bound_nv 400`, whereas `pm_bound2.csv`
  and `pm_bound3.csv` need `--bound_nv 600`. This value sets the sample of
  tokens in the bound columns, not the Monte Carlo columns.
- `pm_energy.json` lists 30 mask sets. Two of them,
  `bert_white_U4_L16_flat_s2` and `bert_white_U4_L16_flat_s3`, come from
  masks of an earlier, stopped run, so a clean rerun lists 28.
- In graph `g3` of `pm_figs.csv`, the rows for rho = 0.9 at L = 32 and
  L = 1 come from a separate run of the same training and evaluation calls
  with the same seeds, whose control at L = 64 reproduces the stored value
  exactly. The current part `figs` computes them in its loop.
- Two labels differ from the labels that the runs printed, the scheme
  `no_overlap_copies` in `pm_rs.csv` and the rows `cross_class` in
  `pm_channel_eval.csv`. The current code prints the released labels.
- Reruns match the released values to within GPU and library
  nondeterminism. Two trainings with the same code and seed on different
  GPUs and thread counts differ by up to 2e-5 in the mask weights and
  give the same SER at 10 dB. Parts `csit`, `adapt` and `figs` also
  estimate the normalization constant of the fixed masks from 65,536 draws
  without a seed of their own, whereas part `sig` and
  `exp_pm_chanhi.py --part hi` seed this draw (seed 0).
- `exp_pm_chanhi.py` uses the seeds and the chunking of part `figs`, so
  the first 39,936 frames of each point are drawn as in graph `g1`, whose
  40,000 frames end with a shorter chunk. Its adapted masks use the seeded
  normalization constant, so on these frames they can differ from `g1` by
  a few errors.
- The Hugging Face model and dataset revisions are not pinned.

## Conventions (periodic masks)

- Real-vector model with d = 64 real dimensions per frame. Real dimensions
  2k and 2k+1 form subcarrier k of K = 32, counted from 0 as in the code
  (the paper counts from 1, dimensions 2k−1 and 2k). The average frame
  energy is one under the source token statistics, with the normalization
  constant c estimated by Monte Carlo, and the SNR is γ = 1/(dσ²).
- Users: U = 4 unless a part states otherwise (part `period` also uses two
  and eight users, part `comb` also eight).
- Codebook (never trained): the input-embedding table of `bert-base-uncased`
  (V = 30,522) or `sentence-transformers/gtr-t5-base` (V = 32,100),
  centered and projected on the top 64 principal directions of the
  covariance weighted by the token frequencies p_v of the first 40,000
  articles of the AG News train split, whitened, turned by a fixed random
  rotation (seed 0), and normalized to unit norm (label `pwhite`).
  `pm_models_eval.csv` also holds uniform whitening (`white`), the raw
  projection (`raw`) and a Gaussian codebook (`rand`).
- Labels: in `pm_period_*`, `pm_channel_*`, `pm_plen_*` and the mask files
  `white_*` and `bertw_*`, the label `white` denotes the codebook `pwhite`.
  In `pm_models_eval.csv`, `white` denotes uniform whitening and `pwhite`
  the proposed codebook.
- Masks: L trained values per user, scaled to norm √L and repeated d/L
  times, so every mask has energy d. The frame is
  x = (1/c) Σ_u e_{s_u} ⊙ m_u, where e_{s_u} is the codeword of user u.
  The values per mask J are L for a static pattern and the number of
  blocks for an adapted mask.
- Training: Adam with learning rate 0.01, 256 frames per step, 3,000 steps,
  tokens drawn from p_v, training SNR uniform in 0 to 20 dB per frame
  (fixed at 10 dB for the masks of Figs. 3 and 4), and cross-entropy over the whole vocabulary with a learned scale. TF32 is
  on during training only. Each mask set is trained for one channel class,
  load and content statistics, and parts `period` and `channel` also test
  masks away from their training conditions.
- Decoder: weighted minimum distance over all V tokens after zero
  forcing, with weight |ĝ|² on each real dimension.
- Channel classes: `flat` (one Rayleigh gain per user and frame), `off6`
  and `off12` (per-user SNR offsets of −6, −2, 2, 6 dB and −12, −4, 4,
  12 dB), `nmse01`, `nmse05` and `nmse10` (estimation error
  with NMSE 0.01, 0.05 and 0.1, one error per user and frame), `fs4` and
  `fs8` (T = 4 or 8 taps). A channel with T taps has T paths one sample
  apart with independent amplitudes a_{u,t} ~ CN(0, 1/T), drawn for every
  user and frame, and the 32 subcarrier gains are their discrete Fourier
  transform, g_{u,k} = Σ_t a_{u,t} exp(−j2πtk/32). The gains of
  neighbouring subcarriers stay correlated above 0.9 over about 32/(4T)
  subcarriers, the coherence bandwidth, that is two subcarriers at four
  taps and one at eight taps.
- Shared content: in each frame all users receive the same token with
  probability rho and independent tokens otherwise. A user keeps its mask
  whether or not the tokens are shared, since switching masks frame by
  frame would need signaling in every frame, and the receivers are not
  told which kind of frame they decode.
- Fixed masks: the comb pattern gives user u the dimensions u, u+4,
  u+8, ... (L = U, no overlap), and full overlap weights every dimension
  equally for all users.
- Adapted masks: the transmitter knows every user's subcarrier gains. At
  each update the users take blocks of B subcarriers in turn, each its
  best remaining block by the sum of log(1 + SNR), with one value per
  block, optionally water-filled (part `adapt`). The owner of a block takes
  log2(U) bits. Part `sig` uses J blocks of 64/J real dimensions, and with
  J < U users share blocks, so an update takes b_J = J log2 U bits for
  J ≥ U and U⌈log2 J⌉ bits for J < U.
- OMA: the 16 bits of the token index, uncoded, on d/U adjacent real
  dimensions per user. At U = 4 every real dimension carries one bit
  (BPSK per dimension, that is QPSK on eight adjacent subcarriers), at
  U = 2 every bit is sent twice, and at U = 8 natural-mapped 4-PAM
  carries two bits per dimension. Its frame energy on the test streams is
  1 at U = 2 and 4 and 1.22 (BERT) or 1.29 (GTR) at U = 8
  (`pm_oma_energy.csv`), because the bits of the token indices are not
  equiprobable (the top bit is always zero).
- Test streams (`exp_pm.py`): the first 2,000 articles of the AG News test
  split, article i to user i mod U, every stream cut to the shortest
  (24,998 frames at U = 4 with BERT tokens, one token per user and frame).
  Columns `ter` (the SER), `sem_dist` (one minus the mean cosine between
  the 768-dimensional embeddings of sent and decoded tokens), `err_tok_sim`
  and `err_close` (mean cosine between sent and decoded token over the
  wrong tokens, and the share of wrong tokens with cosine above 0.4),
  `topic_acc` (accuracy of `textattack/bert-base-uncased-ag-news` on the
  decoded articles) and `gtr_sim` (cosine between `gtr-t5-base` sentence
  embeddings of decoded and clean articles).
- `exp_pm_ext.py` scores the SER on tokens drawn from p_v, 40,000 frames
  per point (20,000 in part `sic`, 30,000 in part `rs`), and
  `exp_pm_chanhi.py` uses 400,000 frames from 12 dB on with the seeds of
  part `figs`.
- Seeds: codebook 0, training 1 (1 to 3 where a part states it), decoders
  on the test streams 777 + 10·SNR, and in `exp_pm_ext.py` test frames
  11 + 10·SNR and validation frames of part `sig` 21 + 10·SNR.

## Requirements (periodic masks)

The released runs use Python 3.12.3, PyTorch 2.11.0 (CUDA 12.8),
transformers 5.22.1, sentence-transformers 6.2.0, datasets 5.1.0, NumPy
2.5.4 and Matplotlib 3.11.2. The scripts download the models
`bert-base-uncased`, `sentence-transformers/gtr-t5-base` and
`textattack/bert-base-uncased-ag-news` and the dataset `fancyzhx/ag_news`
(train and test splits), about 1.1 GB in all. A GPU is recommended, and
`SSE_DEVICE=cpu` forces the CPU. `replot_pm.py` needs NumPy, Matplotlib
3.10 and the Times New Roman font (the released figures come from
Matplotlib 3.10.8). Its layout check refuses a figure whose text leaves
the canvas or whose legend covers a curve. Without Times New Roman,
Matplotlib uses the wider DejaVu Serif instead and the check refuses
every figure. Matplotlib 3.11 draws taller legend rows, so even with a
font of Times metrics the check refuses `fig5_pm.pdf` and
`fig_codebook_pm.pdf`. Fig. 1 needs LaTeX with TikZ and the `standalone`
class.

## Earlier design: periodic codebook reuse (original submission and first revision)

The original submission, "Scaling Shared-Embedding Multiple Access to
Foundation-Model Vocabularies", and the first revision, "Scaling
Shared-Embedding Multiple Access to Large Vocabularies via Periodic
Codebook Reuse", used this design. The figure, table and section numbers
in this section refer to the first revision.

The construction divides a d-dimensional embedding frame into P periods,
reuses one unit codebook of V_u codewords in every period, and carries the
vocabulary index in its base-V_u digits, so V = V_u^P indices are served
with V_u·d/P stored values and a training cost per step independent of V.

### Contents

| Folder | Content |
|---|---|
| `code/` | Shared library `sse_lib.py`, one script per experiment, and the replot scripts |
| `data/` | Raw result of every figure and table (CSV/JSON) and the small trained models (`*.pt` below 200 kB) |
| `fig/` | Figure PDFs regenerated from `data/` (`*_revision.pdf` belong to the first-revision manuscript) and the TikZ source of Fig. 1 |

The larger models of the full codebook and of the other conventional
schemes, among them `full_V*.pt` for V ≥ 4,096 and the long-budget models
of the full codebook and of sampled and hierarchical softmax for all ten
seeds, about 0.8 GB in all, are not stored. The scripts retrain them with the
fixed seeds when they are missing. The long-budget models of the proposal
(`rev_sse_long_md*.pt`, ten seeds) are stored.

### Figure and table map

| Item | Script | Data |
|---|---|---|
| Fig. 1 (system) | `fig/fig1_integrated_revision.tex` (standalone TikZ) | none |
| Fig. 2 (SER vs SNR) | `replot_revision.py snr` | `snr_sweep.csv` (`exp_main.py`), `sota_snr.csv` (`exp_sota.py`), `rev_snr_fullmd.csv` (`exp_rev_fullmd.py`) |
| Fig. 3 (SER vs training MACs) | `replot_revision.py conv` | `rev_budget_curve_{sse,full,sampled,hier}_mean.csv` (10-seed means by `exp_rev_budget_eval.py --aggregate-only` of `rev_budget_curve_{scheme}[_s2|...|_s10].csv`. Seed 1 comes from `exp_rev_budget.py --seed 1`, which logs 2e4 frames per point, and seeds 2-10 from `exp_rev_budget.py --seed {2,...,10} --log-frames 200000`, seeds 4-10 through `fig3_seed_queue.sh`) |
| Table I | none | `snr_sweep.csv`, `sota_snr.csv`, `rev_snr_fullmd.csv`, `rev_budget_final_mean.csv` (10-seed means; per seed `rev_budget_final_{scheme}[_s2|...|_s10].csv`), `complexity.json` |
| Table II | none | `rev_robust.csv` and `rev_robust_fullmd.csv` (`exp_rev_robust.py`, `rev_channel.py`), `rev_users.csv`, `rev_users_fixedload.csv` (`exp_rev_users_fixedload.py`) |
| Analysis checks (Sec. III-B) | `exp_rev_analysis.py`, `exp_rev_scalingbound.py`, `exp_rev_fullmd.py` | `rev_norms.csv`, `rev_norms_full.json`, `rev_decoders.csv`, `rev_rho.json`, `rev_indep.csv`, `rev_isotropy.json`, `rev_scaleP.csv`, `rev_factor.csv`, `rev_scalingbound.csv` |
| CSI-aware retraining | `exp_rev_csitrain.py` | `rev_csitrain.csv` |
| Real AG News streams (Sec. IV, supplementary figure via `replot_revision.py real`) | `exp_rev_semantic.py` | `rev_semantic.csv`, `rev_energy.json` |
| Original-submission experiments (not in the first-revision manuscript) | `exp_main.py`, `exp_sota.py`, `exp_real.py`, `exp_scaling_dense.py`, `exp_sse_bestP.py`, `exp_att_seeds.py`, `exp_d128.py`, `exp_lmmse.py`, `replot.py` | `convergence.csv`, `real_ter.csv`, `real_stats.json`, `sota_real.csv`, `scaling_att_seeds.csv`, `d128_snr.json`, `curve_*.json` |
| Masked-norm histogram (supplementary figure, `replot_revision.py norms`) | `exp_rev_analysis.py`, `exp_rev_fullmd.py` | `rev_norms.csv`, `rev_norms_full.json` |
| Full-codebook norm spread after the long budget | `exp_rev_full_long_norms.py` (needs the large `full_V65536.pt` and `rev_full_long_md.pt`, retrained by `exp_main.py` and `exp_rev_budget.py`) | `rev_norms_full_long.json` |
| Seed spread | `exp_rev_seeds.py`, `exp_rev_seeds_long.py` | `rev_seeds_sse.csv`, `rev_seeds_sse_20k.csv` |
| Fixed-load user sweep | `exp_rev_users_fixedload.py` | `rev_users_fixedload.csv` |
| User-limit mechanism: interference variance and per-user candidate distance vs U | `exp_rev_users_rho.py` (uses `rho_closed_form` of `exp_rev_analysis.py`) | `rev_users_rho.csv` (from `sse_P4.pt`, `rev_sse_U{2,8,16}.pt`, `rev_sse_fixedload_U*.pt`) |
| Vocabulary sweep | `replot_revision.py scaling` | `scaling.csv` (ToDMA), `rev_oma_sweep.csv` (OMA at unit frame energy, `exp_rev_oma_sweep.py`), `scaling_sse_P.csv`, `rev_scaling_fullmd.csv`. The `ser*_OMA` columns of `scaling.csv` spend only b/16 of the frame energy and are kept for the record only |
| Factorizations over ten runs | `exp_rev_factor_seeds.py`, `exp_rev_factor_sweep.py`, `exp_rev_factor_bound.py`, `exp_rev_factor_mech.py` | `rev_factor_seeds.csv`, `rev_factor_sweep.csv`, `rev_factor_bound.csv`, `rev_factor_mech.csv` |
| Masked norms over runs | `exp_rev_norms_runs.py` | `rev_norms_runs.csv` |
| Scaling bound with interference, hypercube, fixed packing, full-codebook packing | `exp_rev_scalingbound_rho.py`, `exp_rev_hypercube.py`, `exp_rev_fixedkappa.py`, `exp_rev_full_kappa.py` | `rev_scalingbound_rho.csv`, `rev_hypercube.csv`, `rev_fixedkappa.csv`, `rev_full_kappa.csv` |
| User limit at other factorizations | `exp_rev_users_factor.py` | `rev_users_factor.csv` |
| Role of the residual interference as users are added | `exp_rev_users_interf.py` | `rev_users_interf.csv` |
| Uplink check on the trained masks | `exp_rev_uplink.py` | `rev_uplink.csv`, `rev_mechanism.csv` |
| Frequency selectivity: floors, rotated OMA | `exp_rev_fs_floor.py`, `exp_rev_rot_oma.py` | `rev_fs_floor.csv`, `rev_fs_floor_crossover.csv`, `rev_rot_oma.csv` |
| Reported model on fresh seeds and best-of-six statistics, selection and reporting on separate seeds | `exp_rev_selection.py` | `rev_selection.csv`, `rev_selection_summary.csv` |
| Source of the lead over OMA: energy, distance spectrum and near neighbours of each user's digit code | `exp_rev_spectrum.py` | `rev_spectrum.csv`, `rev_spectrum_pairs.csv`, `rev_spectrum_awgn.csv` |
| Estimation-error floors | `exp_rev_csifloor.py` | `rev_csifloor.csv` |
| SNR gap to the full codebook | `exp_rev_gap_curves.py`, `exp_rev_gap_vs_V.py` | `rev_gap_curves.csv`, `rev_gap_db.csv`, `rev_gap_vs_V.csv` |
| Per-step accounting and timing | `exp_rev_c_closedform.py`, `exp_rev_timing.py` | `rev_c_closedform.csv`, `rev_c_closedform_ops.csv`, `rev_c_closedform_train.csv`, `rev_timing.csv` |
| ToDMA detectors (Sec. IV) | `exp_rev_todma_detect.py` | `rev_todma_detect.csv`, `rev_todma_detect.json` |
| Topic accuracy over runs and under block fading (Sec. IV) | `exp_rev_sem_runs.py`, `exp_rev_sem_block.py` | `rev_sem_runs.csv`, `rev_sem_runs_pairs.csv`, `rev_sem_block.csv` |
| Energy decomposition on real streams | `exp_rev_energy_decomp.py` | `rev_energy_decomp.json` |

`replot_revision.py` reads only `data/` and writes `fig/`. It fails if any
axis label is clipped or any legend covers a curve.

### Conventions

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
  model (`sse_P4.pt`) is the training-seed-1 run of the original
  submission (`exp_main.py`) and has the lowest SER of six 4,000-iteration
  runs. The other five, trained during the revision, are in
  `rev_seeds_sse.csv` (`exp_rev_seeds.py`, labels 1-5). That script builds
  each model before it sets the seed, so its labels set the training
  samples but not the starting weights, and its run labeled 1 is not a
  re-run of `sse_P4.pt`. The two-user models of `exp_rev_robust.py` (run
  with `--part users`) and `exp_rev_users_fixedload.py` and the NMSE 0.05
  model of `exp_rev_csitrain.py` were likewise built before any seed was
  set, so their starting weights were not recorded. Long-budget
  (2e5-iteration) results are means of
  training seeds 1 to 10 (`exp_rev_budget_eval.py`).
- Training results depend on the GPU, the software version, and the
  evaluations logged during training, which draw on the generator of the
  training samples. The run `rev_factor_16_4_s1.pt` of
  `exp_rev_factor_seeds.py`, seeded like `exp_main.py` (`set_seed(1)`
  before `SSE(P=4)`) but trained without the logged evaluations, gives
  0.276 at 10 dB rather than the 0.257 of the released `sse_P4.pt`. The released checkpoints are the reported models,
  and a retrained model is expected to lie within the stated seed spread.
- Timing hardware: the 13.5 s and 108.4 s of `complexity.json` and the
  `time_s` of `convergence.csv` come from the CUDA GPU of the workstation
  used for the original submission (WSL, GPU model not logged) and include
  the evaluation logged every 100 iterations. The `time_s` columns of
  `rev_budget_curve_*.csv` count training steps only and come from an
  NVIDIA RTX A4500 for seeds 1-3. Seed 1 shared that GPU with other runs,
  while seeds 2 and 3 ran one at a time. Seeds 4-10 ran on an NVIDIA RTX
  PRO 5000 Blackwell, four runs sharing one GPU. The `time_s` of the
  `_mean` files therefore mixes several loads and is not used for any
  wall-clock statement. `rev_timing.csv` comes from an NVIDIA RTX PRO
  5000 Blackwell (logged as `cuda:0`) and one CPU thread of the same
  server.

### Requirements

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
