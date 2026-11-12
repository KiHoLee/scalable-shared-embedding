# Simulation Environment (low-dimensional masks)

This file lists every parameter behind the results of "Scaling
Shared-Embedding Multiple Access to Large Vocabularies via Low-Dimensional
Masks". The paper states the main parameters, and the tables below give
the rest. Code references point to `code/`. Figure numbers are those of
the paper. The figure files keep the numbering of an earlier draft
(`fig2_pm.pdf`, `fig4_pm.pdf` and `fig5_pm.pdf` are Figs. 2, 3 and 4, and
`fig3_pm.pdf` is the supplementary content figure).

## 1. System and source

| Item | Value |
|---|---|
| Users per frame `U` | 4 (main); 2 and 8 for the pattern-length rule, the load tests and shared content with eight users |
| Frame | `d = 64` real dimensions = `K = 32` complex subcarriers; real dimensions `2k-1, 2k` form subcarrier `k` (counted from 1 as in the paper; the code counts from 0, dimensions `2k, 2k+1`) |
| Tokens per frame | one token per user and frame |
| Language model (codebook source) | `bert-base-uncased`, `V = 30,522` word pieces, `D0 = 768`; second model the GTR base model `sentence-transformers/gtr-t5-base` (its T5 `shared` table, `V = 32,100`) |
| Source text | AG News (`fancyzhx/ag_news`), four topics |
| Token frequencies `p_v` | first 40,000 articles of the train split, tokenized without special tokens (`exp_pm.py`, `Source.dist`) |
| Test streams | first 2,000 articles of the test split; article `i` goes to user `i mod U`; each user's token ids are concatenated and all streams are cut to the shortest one (`Source.streams`), 24,998 frames at `U = 4` with BERT tokens, 1,948 graded articles |
| Tokens for Figs. 2 to 4 and the supplementary content figure | drawn i.i.d. with `p_v` (`exp_pm_ext.py`, `sample`) |
| Tokens for the text metrics | the test streams (`exp_pm.py`, `Evaluator`) |
| Frame energy | unit average energy; `c` is estimated from frames drawn with `p_v` (8,192 frames per training step, 65,536 at the end) |

## 2. Channel model and channel classes

User `u` sees subcarrier `k` as `z_k = g_{u,k} x_k + w_k`, `w_k ~ CN(0, 2 sigma_u^2)`,
with SNR `gamma_u = 1 / (d sigma_u^2)` (`rev_channel.channel`). Gains are drawn
independently for every user and every frame unless a row below holds them.

| Class (`COND` in `exp_pm.py`) | Gains | SNR offsets per user | Estimation error |
|---|---|---|---|
| `flat` | one `g_u ~ CN(0,1)` on all subcarriers | none | none |
| `off6` | flat | -6, -2, 2, 6 dB | none |
| `off12` | flat | -12, -4, 4, 12 dB | none |
| `nmse01`, `nmse05`, `nmse10` | flat | none | `g_hat = g + e`, `e ~ CN(0, NMSE)`, NMSE = 0.01, 0.05, 0.1, one `e` per user and frame |
| `fs4`, `fs8` | `T = 4` or `8` taps `a_{u,t} ~ CN(0, 1/T)`, `g_{u,k} = sum_t a_{u,t} exp(-j 2 pi t k / K)` | none | none (`e` would be independent per subcarrier) |

| Derived quantity or variant | Value |
|---|---|
| Subcarrier correlation at distance `Delta` | `\|sin(pi T Delta / K) / (T sin(pi Delta / K))\|` |
| Coherence bandwidth (correlation >= 0.9) | about `K / (4T)` subcarriers: 2 at four taps, 1 at eight taps |
| Receiver front end | zero forcing `z_k / g_hat_{u,k}`, reliability weight `\|g_hat_{u,k}\|^2` on both real dimensions of subcarrier `k` |
| Estimation errors at either end (`exp_pm_cross.py`) | `rx_nmse`: receiver estimate `g + e`; `tx_nmse`: the transmitter forms the adapted masks from `g + e'`; `e`, `e'` ~ CN(0, NMSE), common to all subcarriers under flat fading and independent across subcarriers otherwise; pilot estimates use NMSE = 1/SNR at both ends |
| Aged gains (`exp_pm_cross.py`, `age_rho`) | masks formed from `g_old`, frame sent through `rho g_old + sqrt(1 - rho^2) g_new`, `rho` = 0.95, 0.9, 0.7 |
| Gain held over time (`exp_pm_block.py`) | one gain per user held over 1, 16 or 64 frames or over each article of the test streams |

## 3. Codebook (never trained, except in `exp_pm_trainedcb.py`)

| Kind (`--book`) | Construction (`exp_pm.py`, `codebook`) |
|---|---|
| `pwhite` (proposed) | weighted mean and covariance of the 768-dim table under `p_v` (float64), top 64 eigenpairs, whitening by `Lambda^{-1/2}` |
| `white` | uniform mean, top 64 right singular vectors, division by the singular values |
| `raw` | the same projection without whitening |
| `rand` | i.i.d. N(0,1), `V x d` |
| All kinds | fixed random rotation `Q` (QR of a 64x64 Gaussian matrix, generator seed 0), rows normalized to unit norm |
| Shuffled (`exp_pm_semctl.py`) | the `pwhite` codewords permuted among the tokens inside groups of 50 adjacent ranks of `p_v`, permutation seeds 1, 2, 3 |
| Random table under the same projection (`exp_pm_semctl.py`, `randpw`) | an N(0,1) table of 30,522 x 768 entries (table seeds 1, 2) put through `pwhite` |
| Restricted vocabularies and GPT-2 (`exp_pm_vocab.py`) | `pwhite` of the BERT table restricted to the 1,024, 4,096, 16,384 or 30,522 most frequent tokens (`p_v` renormalized), and of the GPT-2 `wte` table (50,257 tokens, `p_v` counted under the GPT-2 tokenizer) |
| Trained codebook (`exp_pm_trainedcb.py`) | a `V x d` table trained jointly with `L = 16` masks from the `rand` or the `pwhite` start, rows renormalized to unit norm after every step, full softmax over `V` or sampled softmax with 1,024 negatives drawn with `p_v`; Adam with step 1e-3 for the codebook and 1e-2 for the masks, cosine decay after 1,000 steps, up to 30,000 steps |

## 4. Masks and their training

| Item | Value |
|---|---|
| Mask | a real-valued vector `m_u` of `d` weights; `m_{u,i}` sets the sign and `m_{u,i}^2` the energy of user `u` on dimension `i`; `sum_u \|\|m_u\|\|^2 = U d`, and `\|\|m_u\|\|^2 = d` unless a row states otherwise (`exp_pm_nearfar.py --part split` scales the user energies); masks without overlap, with separate dimensions per user as in OMA, and full overlap are two special cases, and real weights also allow any partial overlap |
| Low-dimensional mask | a real-valued mask described by `J` values, far fewer than `d`: a static entry or an adapted mask (rows below) |
| Mask size `J` | the number of values that define one mask, not a number of masks: `J = L` for a static entry, `J = K/B` blocks of `B` subcarriers for an adapted mask, `J = d` for the full-length mask |
| Static entry | a trained pattern `W_m` of size `U x L`, initialized `randn / sqrt(L)`, real weights of either sign; each row is normalized to norm `sqrt(L)` and repeated `P = d / L` times |
| Mask set | one static entry per load, channel class and level of shared content |
| Optimizer | Adam, step size 1e-2 |
| Steps and batch | 3,000 steps of 256 frames (`--iters 3000`); 30,000 steps in `exp_pm_trainedcb.py --part converge` |
| Training SNR | uniform in [0, 20] dB, drawn per frame; fixed at 10 dB for the static entries of Fig. 3 and the trained masks of the supplementary content figure (`train_rho`, `snr_range`) |
| Training channel | the channel class of the entry (`cond`) |
| Loss | cross-entropy over all `V` tokens of the decoder scores times a learned score factor `exp(s)`, with `s` initialized to 2.0 (`logit_scale`); `exp_pm_nearfar.py` learns one factor `exp(s_u)` per receiver |
| Shared content | `train_rho` trains under the same `rho` as used at test (`exp_pm_ext.py`) |
| Precision | TF32 during training only; evaluation in full fp32 |
| Time per entry | about 19 s on one idle NVIDIA RTX PRO 5000 Blackwell |
| Adapted masks | formed at every update from the gains known at the transmitter; an adapted mask holds one real value on each of `J` blocks, in one partition shared by all users; the tested masks are the coarsest such masks, with two weight levels; with `J >= U` the users take turns (order reversed every round), each sets the level `sqrt(U)` on the block with its largest sum of `log(1 + \|g\|^2 SNR)` among those with no weight yet, and the mask of user `u` holds `sqrt(U)` on `J/U` blocks and zero on the others, so the masks do not overlap and `c` stays fixed; with `J < U` each user sets the level `sqrt(J)` on its best block, so the masks overlap (`exp_pm_ext.py`, `adaptive_masks`, `unit_masks`) |
| Finer real weights per block | weights water-filled over the blocks that each adapted mask weights (`wf=True`, part `adapt`); they change the TER by at most two percent (points with at least 20 token errors), and with `q` bits per value an update costs `J (q + log2 U)` bits when each block carries the weight of one user and `U J q` bits in general, against `U d q` for full-length masks |
| Signaling per update | two-level masks: `b_J = J log2 U` bits for all users for `J >= U` (the user index of each block, broadcast once), `b_J = U ceil(log2 J)` bits for `J < U` (the block that each user weights); the block size, the weight rule and `c` are fixed by the entry in use, so every receiver derives its adapted mask from the signaled user indices |

## 5. Decoder

| Item | Value |
|---|---|
| Rule | weighted minimum distance over all `V` tokens: score `2a sum_i w_i r_i (e_v m_u)_i - a^2 sum_i w_i (e_v m_u)_i^2`, `a = 1/c` (`Masks.scores`) |
| Knowledge at receiver `u` | the codebook, its own mask, `c`, and its own channel estimate |
| Cost | about `V d` multiply-accumulate operations per user and token, `2 V d` under frequency-selective fading |
| Variants (`exp_pm_decoder.py`) | unweighted minimum distance, the correlator of the earlier design, normalized correlation, and a rule aware of the estimation error (Gaussian likelihood given `g_hat` with the other users as interference) |
| Receivers that decide the frame kind (`exp_pm_content.py`) | a test of "one token shared by all users" against "independent tokens" with the prior `rho` or 0.5, then decoding under the chosen kind; a genie told the kind |
| The same receivers with adapted masks (`exp_pm_content_fs.py`) | every receiver knows the user index of each block, tests the two kinds with the prior `rho` clipped to [0.001, 0.999], and decodes a shared token on all blocks with the summed masks or its own token on the blocks that its mask weights; a genie told the kind |

## 6. Compared schemes

| Scheme | Definition | What it isolates |
|---|---|---|
| OMA | each user owns `d/U` real dimensions and sends the 16 bits of its token index uncoded: one bit per real dimension at `U = 4` (BPSK per dimension, that is QPSK on eight adjacent subcarriers), BPSK with each bit sent twice at `U = 2`, natural-mapped 4-PAM at `U = 8`; weighted combining and nearest-level detection (`rev_channel.oma_*`) | conventional digital multiple access |
| OMA on adapted blocks | OMA as above, but every user sends its 16 bits on the blocks of two subcarriers that its adapted mask weights (`exp_pm_chanhi.py`, `run_oma`), detected by sign with exact gains | the gain of choosing strong subcarriers without a spread codeword |
| Coded OMA | the 16 index bits encoded by RM(2,5) into 32 code bits, Gray-mapped 16-QAM on the eight subcarriers of the user, table scaled to unit energy under `p_v`, weighted maximum likelihood over the `V` valid tokens; the code bits pass the bit interleaver of `pm_oma_coded_il.json` (searched over 20,000 permutations, seed 0), so two tokens differ on at least 4 of the 8 subcarriers (2 without it); on every `U`-th subcarrier (`coded_il_comb`) or on the blocks of the adapted masks (`coded_il_adapted`, gains known at the transmitter); not trained (`exp_pm_oma.py`) | the gain of a channel code at the same rate, with and without known gains |
| Coded OMA, other variants | single parity check with Wagner decoding, RM(2,5) without the interleaver, adjacent placement, 2 and 8 users (`exp_pm_oma.py --part coded`, `--part sem`) | sensitivity of the coded reference |
| ToDMA | every token has a fixed unit-norm random Gaussian codeword (the `rand` codebook, seed 0), the frame is the plain sum of the `U` codewords scaled by `c` (`c^2 = 4.066` under `p_v`), no masks; each receiver zero-forces and detects `U` tokens by orthogonal matching pursuit with signed correlation, a weighted least-squares refit and the weight `1 / (sigma^2/\|g_hat_k\|^2 + (U-1)/(c^2 d))` per dimension, the best of five detectors tested; a genie counts user `u` correct when its own token is among the detected ones, which favors ToDMA; not trained (`exp_pm_todma.py`) | a fixed codeword per token unrelated to its meaning, without masks |
| Random codebook | `rand` codebook with the same masks, mask training, normalization and decoder | the pretrained geometry |
| Static entry | the pattern with `L = 16` trained offline for a channel class and stored in the mask set; no gains at the transmitter | the stored mask set |
| Adapted masks | two weight levels on blocks of `B = 2` subcarriers in Fig. 2 and on `J` blocks in Figs. 3 and 4, formed from the gains known at the transmitter (section 4) | adaptation to the channel |
| Untrained pattern with `L = U` (masks without overlap) | user `u` has the weight `sqrt(U)` on dimensions `u, u+U, u+2U, ...` (`FreeMasks`); a fixed reference pattern, the special case without overlap | separation |
| Full overlap | one common mask of equal weights, so every user has the same weight on every dimension and energy `d`; a fixed reference pattern, the other special case | combination |
| Trained masks for shared content | `L = 16` patterns trained at the given `rho` and at 10 dB, one mask per user in every frame, decoded by the rule of section 5 | learned partial overlap |

Energy check: the measured frame energy of every trained entry on the test
streams lies within 0.3 percent of one (`pm_energy.json`). The frame energy
of ToDMA is 1.0001 (`pm_todma_energy.json`) and that of coded OMA 0.9992 to
0.9999 (`pm_oma_coded_sem.csv`). The OMA frame energy is exactly one at
`U = 2, 4`, and 1.22 (BERT) or 1.29 (GTR) at `U = 8` (`pm_oma_energy.csv`),
because the bits of the token indices are not equiprobable (the top bit is
always zero).

## 7. Shared content

| Item | Value |
|---|---|
| Model | in every frame all users receive the same token with probability `rho`, independent tokens otherwise (`sample`); `exp_pm_content.py --part pairs` lets users (1, 2) and (3, 4) share tokens with probability `rho`, independently per pair |
| Mask use | a user keeps one mask whether or not the tokens are shared, and its receiver decodes its own token by the rule of section 5 without being told the frame kind |
| Levels | `rho` = 0, 0.25, 0.5, 0.75, 0.9, 1 (supplementary content figure, `pm_semfig4_static.csv` and `pm_content_fs.csv`); Fig. 3 draws `rho` = 0, 0.5, 0.75, 0.9 |
| Cross-energy label | mean over users of `(1/d) sum_i m_{u,i}^2 sum_{k != u} m_{k,i}^2` of the trained masks |

## 8. Metrics

| Metric | Definition |
|---|---|
| TER | `TER_u = Pr[s_hat_u != s_u]`, the fraction of wrongly decoded tokens, averaged over the users (columns `ter`, `ser`, and the other columns whose name contains `ser` but not `sem`) |
| SER (semantic error rate) | `SER_u = E[1 - zeta(s_u, s_hat_u)] = sum_v p_v sum_{w != v} Pr[s_hat_u = w \| s_u = v] (1 - zeta(v, w))`, with `zeta(v, w)` the cosine of the 768-dim BERT input embeddings `w_v` and `w_w`; a correct token costs 0, and decoded ids outside `[0, V)` count as `[UNK]` (columns `ser_sem*`, `sem_dist`, `ub_sem`, `mc_sem`) |
| Topic accuracy | `textattack/bert-base-uncased-ag-news` on the decoded articles (max length 128); error-free text gives 0.950 (BERT tokens) |
| GTR similarity | cosine between `sentence-transformers/gtr-t5-base` embeddings of the decoded and the sent article |
| Net efficiency (Fig. 4) | `eta = (U (1 - TER) log2 V - b_J / F) / (U log2 V)` with `b_J` of section 4 and `F` frames served by one update while the gains stay fixed; the static entry needs no update, `eta = 1 - TER`; payload 59.6 bits per frame at `U = 4`; Fig. 4 plots the loss `1 - eta = TER + b_J / (F U log2 V)` on a logarithmic axis for `F` = 1, 3, 10, 30 and 100 |
| Feedback variant | the same `eta` with `2 U J` bits added to `b_J`, two bits per user and block for the gains at every update |

## 9. Parameters per figure and claim

| Item | Script and part | SNR | Frames per point | Data |
|---|---|---|---|---|
| Fig. 2, all schemes, 4 and 8 taps | `exp_pm_semfig2.py --part all` | 0-20 dB, step 2 | 200,000 for OMA and ToDMA; 200,000 below 12 dB and 2,000,000 from 12 dB for coded OMA, the static entry and the adapted masks | `pm_semfig2.csv`, `pm_semfig2_check.csv` |
| Fig. 2, static entry trained for the class | the saved `white_U4_L16_fs4_s1`, `_fs8_s1` (`exp_pm.py --part channel`) | | | |
| Fig. 3, static entries, `rho` = 0, 0.25, 0.5, 0.75, 0.9, 1, `L = J` = 1 to 64 | `exp_pm_semfig34.py --part static` (`train_rho` at 10 dB, training seeds 1-3) | 10 dB | 40,000 test frames (seed 111) and 40,000 validation frames (seed 121) | `pm_semfig4_static.csv` |
| Fig. 3 selection | each static entry is the run with the lowest `ser_sem_val` on the validation frames; the plotted SER is on the test frames | | | |
| Fig. 3, adapted masks, 4 and 8 taps, `J = 1..64` | `exp_pm_semfig4a.py --part adapted` | 10 dB (0 and 20 dB also for `J` = 16, 32) | 40,000; 400,000 when a point has fewer than 100 token errors | `pm_semfig4_adapted.csv` |
| Fig. 4 | computed from the TER of the adapted masks of Fig. 3 at four taps and of the class-trained static entry of Fig. 2 (`replot_pm.py`, `fig_tradeoff`), update intervals `F` = 1, 3, 10, 30, 100 | 10 dB | 40,000 (adapted masks), 200,000 (static entry) | `pm_semfig4_adapted.csv`, `pm_semfig2.csv` |
| Supplementary content figure | `exp_pm_semfig34.py --part content` (masks trained at 10 dB, seed 1) | 10 dB | 40,000 (seed 111) | `pm_semfig3.csv` |
| Supplementary codebook figure and codebook comparison | `exp_pm.py --part models` | 0-20 dB, step 2 | test streams, 24,998 frames with BERT tokens | `pm_models_eval.csv` |
| Semantic controls | `exp_pm_semctl.py --part ctrl`, `--part omasub` | 0-20 dB, step 2 | test streams | `pm_semctl_eval.csv`, `pm_semctl_omasub.csv` |
| Pattern length for `U = 2, 4, 8`, load mismatch | `exp_pm.py --part period` | 0, 5, 10, 15, 20 dB | test streams | `pm_period_eval.csv` |
| Pattern length under channel classes | `exp_pm.py --part plen` | 10, 20 dB | test streams | `pm_plen_eval.csv` |
| Static entries across channel classes | `exp_pm.py --part channel` | 0, 5, 10, 15, 20 dB | test streams | `pm_channel_eval.csv` |
| Bounds of Proposition 1 | `exp_pm.py --part bound2`, `--part bound3` (`--bound_nv 600`) | 0-20 dB (bound2), 0-12 dB (bound3) | 200,000 Monte Carlo frames; 600 tokens drawn with `p_v` for the sum over `v` | `pm_bound2.csv`, `pm_bound3.csv` |
| Fixed-gain bounds on all tokens | `exp_pm_boundov.py --part exact` | 0-14 dB, step 2 | 4,000,000 from 10 dB, 1,000,000 at 6-8 dB, 200,000 below; all 22,615 tokens with `p_v > 0` | `pm_bound_exact.csv` |
| Gaussian step with and without overlap | `exp_pm_boundov.py --part overlap` | 0, 5, 10, 15, 20 dB | 65,536 interference frames per pair test; 200,000 for the bounds | `pm_boundov.csv` |
| Coded OMA with the interleaver, text metrics | `exp_pm_oma.py --part coded_il`, `--part sem_il` | 0-20 dB, step 2 (`coded_il`); 0, 5, 10, 15, 20 dB (`sem_il`) | 200,000 below 16 dB, 2,000,000 from 16 dB; test streams | `pm_oma_coded_il.csv`, `pm_oma_coded_il_sem.csv` |
| Spread of the difference between nearest codewords over the subcarriers | `exp_pm_spread.py` | no noise | all 22,615 tokens with `p_v > 0`; 100,000 Gaussian draws (seed 1) for the reference value | `pm_spread.json` |
| ToDMA | `exp_pm_todma.py --part all` | 0-20 dB, step 2; detectors at 6, 10, 20 dB and without noise | 200,000; 100,000 for the detectors | `pm_todma.csv`, `pm_todma_detector.csv` |
| Loads, estimation errors, aged gains | `exp_pm_cross.py --part load`, `--part csi` | 0, 5, 10, 15, 20 dB (`load`); 10, 20 dB, pilot estimates also at 0, 5, 15 dB (`csi`) | 40,000 to 400,000 (column `frames`) | `pm_cross.csv` |
| SNR offsets, and user energies `10^(-a offset_u / 10)` normalized to unit mean with `a` = 0, 0.25, 0.5, 0.75, 1 (`split`; `a` = 0.5 is the rule `SNR_u^(-1/2)`) | `exp_pm_nearfar.py --part retrain`, `--part peruser`, `--part split` | 0, 5, 10, 15, 20 dB | test streams | `pm_nearfar_eval.csv`, `pm_nearfar_split.csv` |
| Overlap against the training SNR | `exp_pm_ext.py --part content_snr` | 0, 10, 20 dB | 40,000 (test seed 11) | `pm_content_snr.csv` |
| Receivers that decide the frame kind, pairs | `exp_pm_content.py --part test`, `--part pairs` | 0, 10, 20 dB (`test`); 10 dB (`pairs`) | 40,000 | `pm_content_test.csv`, `pm_content_pairs.csv` |
| Shared content with eight users | `exp_pm_content8.py` (training seeds 1, 2 at 10 dB) | 10 dB | 40,000 | `pm_content8.csv` |
| Shared content under 4 and 8 taps, `rho` = 0, 0.25, 0.5, 0.75, 0.9, 1: adapted masks (`B = 2`) and the untrained pattern with `L = U`, each with rule (4) and with receivers that decide the frame kind, full overlap, the static entry of Fig. 2, and static entries trained at 10 dB for the class and `rho` (`L = 16`, training seed 1) | `exp_pm_content_fs.py --part main`, `--part check`, `--part hi` | 10 dB; 20 dB also for the adapted masks and full overlap | 40,000; 400,000 for the adapted masks at 20 dB (`hi`) | `pm_content_fs.csv`, `pm_content_fs_hi.csv`, `pm_content_fs.json` |
| Water-filled real weights | `exp_pm_ext.py --part adapt` | 0, 5, 10, 15, 20 dB | 40,000 | `pm_adapt.csv` |
| Gain held over time | `exp_pm_block.py --part block` | 0, 5, 10, 15, 20 dB | test streams, two draws | `pm_block_eval.csv` |
| Vocabulary size | `exp_pm_vocab.py --part vocab --offline` | 0, 5, 10, 15, 20 dB | 40,000 | `pm_vocab.csv` |
| Decoders and norms | `exp_pm_decoder.py --part norms`, `--part decoders` | 10, 20 dB, NMSE 0, 0.01, 0.05, 0.1 | 40,000 | `pm_norms.csv`, `pm_decoder.csv` |
| Trained codebook, long training | `exp_pm_trainedcb.py --part train`, `--part converge` | 0, 10, 20 dB; validation at 10 dB | 40,000 test, 20,000 validation frames | `pm_trainedcb.csv`, `pm_converge.csv` |

## 10. Random seeds

| Use | Seed |
|---|---|
| Training | `set_seed(s)` with `s = 1` (main), `s = 1, 2, 3` where three runs are reported (`s = 1, 2` in `exp_pm_content8.py`) |
| Codebook rotation and random codebook | generator seed 0 |
| Decoding on the test streams | `777 + 10 snr` |
| Tokens drawn with `p_v` (`exp_pm_ext.ser` and the scripts that reuse it) | `11 + 10 snr`; validation `21 + 10 snr` |
| Adapted masks | `31 + 10 snr + 100 T + int(7 u_d / 2)`, `u_d = d / J` real dimensions per block |
| Constant `c` of fixed and adapted masks in `--part sig`, `exp_pm_chanhi.py` and the scripts added later | 0 |
| Fig. 2 (`exp_pm_semfig2.py`) | OMA `5000 + snr + 100 T + 1000 b` for block `b` of 40,000 frames, decoding `777 + snr + 1000 b`; coded OMA `9500 + 100 T + snr`; ToDMA `71 + int(10 snr)`; static entry `11 + int(10 snr)`, `13 + int(10 snr)` at 16-20 dB; adapted masks `31 + int(10 snr) + 100 T + 14` |
| Coded OMA without the interleaver (`exp_pm_oma.py --part coded`) | `9000 + 100 T + snr` |
| `exp_pm_cross.py` | masks `31 + 10 snr + 100 T + 14`, OMA `61 + 10 snr + 100 T` (+1000 on the adapted blocks), estimation errors and aging from a second generator seeded with the point seed + 100000 |
| `exp_pm_content.py --part pairs`, `exp_pm_semfig34.py` | test frames 111, validation frames 121 at 10 dB |
| `exp_pm_content_fs.py` | frames and gains `11 + 10 snr` in batches of 512 frames, training 1, constant `c` 0; part `check` uses the seeds of the released points that it reproduces |
| `exp_pm_semctl.py` | permutation seeds 1-3, table seeds 1-2, mask seed = draw; `--part omasub` replacement tokens `5000 + 10 snr` and `6000 + 10 snr` |
| `exp_pm_block.py` | noise `777 + 10 snr + 1000 i_hold + 10000 draw`; draw 0 at hold 1 reproduces `pm_models_eval.csv` |
| Bounds | token sample 5; Monte Carlo `900 + snr` (bound2), `700 + snr` (bound3); `exp_pm_boundov.py` tokens `100 + u`, interference `200 + u`, 5,000 tokens seed 6, redraws 1000-2999, simulation `7000 + snr` |

Reruns match the released files to within GPU and library
nondeterminism, not bit for bit. Hugging Face model and dataset revisions
are not pinned.

## 11. Software and hardware

| Item | Version |
|---|---|
| Python | 3.12.3 |
| PyTorch | 2.11.0 with CUDA 12.8 |
| transformers / sentence-transformers / datasets | 5.22.1 / 6.2.0 / 5.1.0 |
| NumPy / Matplotlib | 2.5.4 / 3.11.2 (figures drawn with Matplotlib 3.10.8) |
| Extra model | `gpt2` from the local Hugging Face cache, for `exp_pm_vocab.py` only |
| GPU | NVIDIA RTX PRO 5000 Blackwell (device chosen by `sse_lib.DEVICE`, override with `SSE_DEVICE`) |
