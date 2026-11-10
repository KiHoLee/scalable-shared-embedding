# Simulation Environment (periodic masks)

This file lists every parameter behind the results of "Scaling
Shared-Embedding Multiple Access to Foundation-Model Vocabularies via
Periodic Masks". The letter states the main parameters, and the tables
below give the rest. Code
references point to `code/`.

## 1. System and source

| Item | Value |
|---|---|
| Users per frame `U` | 4 (main); 2 and 8 for the pattern-length rule |
| Frame | `d = 64` real dimensions = `K = 32` complex subcarriers; real dimensions `2k-1, 2k` form subcarrier `k` (counted from 1 as in the letter; the code counts from 0, dimensions `2k, 2k+1`) |
| Tokens per frame | one token per user and frame |
| Language model (codebook source) | `bert-base-uncased`, `V = 30,522` word pieces, `D0 = 768`; alternative the GTR base model `sentence-transformers/gtr-t5-base` (its T5 `shared` table, `V = 32,100`) |
| Source text | AG News (`fancyzhx/ag_news`), four topics |
| Token frequencies `p_v` | first 40,000 articles of the train split, tokenized without special tokens (`exp_pm.py`, `Source.dist`) |
| Test streams | first 2,000 articles of the test split; article `i` goes to user `i mod U`; each user's token ids are concatenated and all streams are cut to the shortest one (`Source.streams`), 24,998 frames at `U = 4` with BERT tokens |
| Tokens for Figs. 2-5 | drawn i.i.d. with `p_v` (`exp_pm_ext.py`, `sample`) |
| Tokens for the text metrics | the test streams (`exp_pm.py`, `Evaluator`) |
| Frame energy | unit average energy; `c` is estimated from frames drawn with `p_v` (8,192 frames per training step, 65,536 at the end) |

## 2. Channel model and channel classes

User `u` sees subcarrier `k` as `z_k = g_{u,k} x_k + w_k`, `w_k ~ CN(0, 2 sigma_u^2)`,
with SNR `gamma_u = 1 / (d sigma_u^2)` (`rev_channel.channel`). Gains are drawn
independently for every user and every frame.

| Class (`COND` in `exp_pm.py`) | Gains | SNR offsets per user | Estimation error |
|---|---|---|---|
| `flat` | one `g_u ~ CN(0,1)` on all subcarriers | none | none |
| `off6` | flat | -6, -2, 2, 6 dB | none |
| `off12` | flat | -12, -4, 4, 12 dB | none |
| `nmse01`, `nmse05`, `nmse10` | flat | none | `g_hat = g + e`, `e ~ CN(0, NMSE)`, NMSE = 0.01, 0.05, 0.1, one `e` per user and frame |
| `fs4`, `fs8` | `T = 4` or `8` taps `a_{u,t} ~ CN(0, 1/T)`, `g_{u,k} = sum_t a_{u,t} exp(-j 2 pi t k / K)` | none | none (`e` would be independent per subcarrier) |

| Derived quantity | Value |
|---|---|
| Subcarrier correlation at distance `Delta` | `|sin(pi T Delta / K) / (T sin(pi Delta / K))|` |
| Coherence bandwidth (correlation >= 0.9) | about `K / (4T)` subcarriers: 2 at four taps, 1 at eight taps |
| Receiver front end | zero forcing `z_k / g_hat_{u,k}`, reliability weight `|g_hat_{u,k}|^2` on both real dimensions of subcarrier `k` |

## 3. Codebook (never trained)

| Kind (`--book`) | Construction (`exp_pm.py`, `codebook`) |
|---|---|
| `pwhite` (proposed) | weighted mean and covariance of the 768-dim table under `p_v` (float64), top 64 eigenpairs, whitening by `Lambda^{-1/2}` |
| `white` | uniform mean, top 64 right singular vectors, division by the singular values |
| `raw` | the same projection without whitening |
| `rand` | i.i.d. N(0,1), `V x d` |
| All kinds | fixed random rotation `Q` (QR of a 64x64 Gaussian matrix, generator seed 0), rows normalized to unit norm |

## 4. Masks and their training

| Item | Value |
|---|---|
| Parameter | pattern `W_m` of size `U x L`, initialized `randn / sqrt(L)` |
| Mask | each row normalized to norm `sqrt(L)` and repeated `P = d / L` times, so `||m_u||^2 = d` |
| Values per mask `J` | `J = L` for a static pattern; `J` blocks of `d / J` real dimensions for an adapted mask; `J = d` is the full-length mask |
| Optimizer | Adam, step size 1e-2 |
| Steps and batch | 3,000 steps of 256 frames (`--iters 3000`) |
| Training SNR | uniform in [0, 20] dB, drawn per frame; fixed at 10 dB for the masks of Figs. 3 and 4 (`train_rho`, `snr_range`) |
| Training channel | the channel class of the entry (`cond`) |
| Loss | cross-entropy over all `V` tokens of the decoder scores times a learned scale `exp(s)`, with `s` initialized to 2.0 (`logit_scale`) |
| Shared content | `train_rho` trains under the same `rho` as used at test (`exp_pm_ext.py`) |
| Precision | TF32 during training only; evaluation in full fp32 |
| Time per entry | about 20 s on one NVIDIA RTX PRO 5000 Blackwell |

## 5. Decoder

| Item | Value |
|---|---|
| Rule | weighted minimum distance over all `V` tokens: score `2a sum_i w_i r_i (e_v m_u)_i - a^2 sum_i w_i (e_v m_u)_i^2`, `a = 1/c` (`Masks.scores`) |
| Knowledge at receiver `u` | the codebook, its own mask, `c`, and its own channel estimate |
| Cost | about `V d` multiply-accumulate operations per user and token |

## 6. Compared schemes

| Scheme | Definition | What it isolates |
|---|---|---|
| OMA | each user owns `d/U` real dimensions and sends the 16 bits of its token index uncoded: one bit per real dimension at `U = 4` (BPSK per dimension, that is QPSK on eight adjacent subcarriers), BPSK with each bit sent twice at `U = 2`, natural-mapped 4-PAM at `U = 8`; weighted combining and nearest-level detection (`rev_channel.oma_*`) | conventional digital multiple access |
| OMA on adapted blocks | OMA as above, but every user sends its 16 bits on the blocks of two subcarriers that the adapted masks give it (`exp_pm_chanhi.py`, `run_oma`), detected by sign with exact gains | the gain of choosing strong subcarriers without a spread codeword |
| Random codebook | `rand` codebook with the same masks, normalization and decoder | the pretrained geometry |
| Static entry | mask trained offline for a channel class and stored in the mask set; no gains at the transmitter | the stored mask set |
| Adapted masks | formed at every update from the gains known at the transmitter: the band is cut into `J` blocks; with `J >= U` the users pick blocks in turn (alternating order), each its remaining block with the largest sum of `log(1 + SNR)`; with `J < U` every user takes its best block and users share blocks; equal weight on the own blocks (`exp_pm_ext.py`, `adaptive_masks`, `unit_masks`) | adaptation to the channel |
| Water-filled adapted masks | as above, plus water-filling of each user's weight over its blocks (`wf=True`) | weight resolution |
| Masks without overlap | user `u` weights only dimensions `u, u+U, u+2U, ...` with weight 2 (`FreeMasks`) | separation |
| Full overlap | all users weight every dimension with weight 1 | combination |
| Trained masks for shared content | `L = 16` patterns trained at the given `rho` | learned balance |

Energy check: the measured frame energy of every saved mask set on the test
streams lies within 0.3 percent of one (`pm_energy.json`). The OMA frame
energy is exactly one at `U = 2, 4`, and 1.22 (BERT) or 1.29 (GTR) at
`U = 8` because the top bit of every index is zero (`pm_oma_energy.csv`).

## 7. Shared content

| Item | Value |
|---|---|
| Model | in every frame all users receive the same token with probability `rho`, independent tokens otherwise (`sample`) |
| Mask use | a user keeps its mask whether or not the tokens are shared, since switching masks frame by frame would need signaling in every frame; the receivers are not told which kind of frame they decode |
| Levels | `rho` = 0, 0.25, 0.5, 0.75, 0.9, 1 (Fig. 3); `rho` = 0.9 for the shared static entries of Fig. 4 |
| Cross-energy label | mean over users of `(1/d) sum_i m_{u,i}^2 sum_{k != u} m_{k,i}^2` of the trained masks |

## 8. Metrics

| Metric | Definition |
|---|---|
| SER | fraction of wrongly decoded tokens |
| Semantic distortion `D` | `1 - cos` between the 768-dim input embeddings of the sent and the decoded token, averaged over tokens |
| Topic accuracy | `textattack/bert-base-uncased-ag-news` on the decoded articles (max length 128); error-free text gives 0.950 (BERT tokens) |
| GTR similarity | cosine between `sentence-transformers/gtr-t5-base` embeddings of the decoded and the sent article |
| Net efficiency (Fig. 5) | `eta = (U (1 - SER) log2 V - b_J / F) / (U log2 V)` with `b_J = J log2 U` bits per update for owners of `J >= U` blocks, `b_J = U ceil(log2 J)` when `J < U` users name the block they share, and `F` frames served by one update while the gains stay fixed |

## 9. Parameters per figure and claim

| Item | Script and part | SNR | Frames per point | Data |
|---|---|---|---|---|
| Fig. 2, OMA | `exp_pm_ext.py --part figs` (g1); tokens seeded `5000 + snr + 100 T`, decoder `777 + snr` | 0-20 dB, step 2 | 40,000 | `pm_figs.csv` |
| Fig. 2, static entry trained for the class (`white_U4_L16_fs4_s1`, `_fs8_s1`) | `exp_pm_chanhi.py --part class` | 0-20 dB, step 2 | 40,000 below 12 dB, 400,000 from 12 dB | `pm_figs_class.csv` |
| Fig. 2, adapted masks, blocks of two subcarriers | `exp_pm_ext.py --part figs` (g1), `exp_pm_chanhi.py --part hi` | 0-20 dB, step 2 | 40,000 below 12 dB, 400,000 from 12 dB | `pm_figs.csv`, `pm_figs_hi.csv` |
| Fig. 2 check, OMA on fixed and on adapted blocks | `exp_pm_chanhi.py --part oma_adapt`; seed `61 + 10 snr + 100 T` (+1000 for adapted blocks) | 0-20 dB, step 2 | 200,000 | `pm_oma_adapt.csv` |
| Fig. 3 | `exp_pm_ext.py --part content_snr` (masks trained at 10 dB, seed 1); also 0 and 20 dB, and masks trained on 0-20 dB | 10 dB (0 and 20 dB also stored) | 40,000 (test seed 11) | `pm_content_snr.csv` |
| Fig. 4, static entries `rho = 0` | `exp_pm_ext.py --part sig10` (`train_rho` at 10 dB, seeds 1-3) | 10 dB | 40,000 (test seed 11, validation seed 21) | `pm_sig10.csv` |
| Fig. 4, static entries `rho = 0.9` | `exp_pm_ext.py --part sig10` (`train_rho` at 10 dB, seeds 1-3) | 10 dB | 40,000 (test seed 11, validation seed 21) | `pm_sig10.csv` |
| Fig. 4, adapted masks, 4 and 8 taps, `J = 1..64` | `exp_pm_ext.py --part sig` (`run_units`) | 10 dB | 40,000 | `pm_sig.csv` |
| Fig. 4 selection | each static entry is the best of three training runs on the validation frames; the plotted SER is on the test frames | | | |
| Fig. 5 | computed from Fig. 4 (adapted, four taps) and the class-trained static entry at 10 dB (`replot_pm.py`, `fig_tradeoff`) | 10 dB | | `pm_sig.csv`, `pm_figs_class.csv` |
| Codebook comparison (text) | `exp_pm.py --part models` | 0-20 dB, step 2 | test streams, 24,998 frames with BERT tokens (one token per user and frame) | `pm_models_eval.csv` |
| Pattern length for `U = 2, 4, 8`, load mismatch | `exp_pm.py --part period` | 0, 5, 10, 15, 20 dB | test streams | `pm_period_eval.csv` |
| Pattern length under channel classes | `exp_pm.py --part plen` | 10, 20 dB | test streams | `pm_plen_eval.csv` |
| Static entries across channel classes | `exp_pm.py --part channel` | 0, 5, 10, 15, 20 dB | test streams | `pm_channel_eval.csv` |
| Bounds of Proposition 1 | `exp_pm.py --part bound2`, `--part bound3` (`--bound_nv 600`) | 0-20 dB (bound2), 0-12 dB (bound3) | 200,000 Monte Carlo frames; 600 tokens drawn with `p_v` for the sum over `v` | `pm_bound2.csv`, `pm_bound3.csv` |

## 10. Random seeds

| Use | Seed |
|---|---|
| Training | `set_seed(s)` with `s = 1` (main), `s = 1, 2, 3` where three runs are reported |
| Codebook rotation and random codebook | generator seed 0 |
| Decoding on the test streams | `777 + 10 snr` |
| `ser` (tokens drawn with `p_v`) | `11 + 10 snr`; validation `21 + 10 snr` |
| Adapted masks | `31 + 10 snr + 100 T + int(7 u_d / 2)`, `u_d = d / J` real dimensions per block |
| Constant `c` of the fixed comb in `--part sig` and `exp_pm_chanhi.py` | 0 |
| Bounds | token sample 5; Monte Carlo `900 + snr` (bound2), `700 + snr` (bound3) |

Reruns match the released files to within GPU and library
nondeterminism, not bit for bit. Hugging Face model and dataset revisions
are not pinned.

## 11. Software and hardware

| Item | Version |
|---|---|
| Python | 3.12.3 |
| PyTorch | 2.11.0 with CUDA 12.8 |
| transformers / sentence-transformers / datasets | 5.22.1 / 6.2.0 / 5.1.0 |
| NumPy / Matplotlib | 2.5.4 / 3.11.2 |
| GPU | NVIDIA RTX PRO 5000 Blackwell (device chosen by `sse_lib.DEVICE`, override with `SSE_DEVICE`) |
