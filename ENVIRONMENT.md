# Simulation Environment (low-dimensional masks)

This file lists every parameter behind the results of "Scaling
Shared-Embedding Multiple Access to Large Vocabularies via Low-Dimensional
Masks". The paper states the main parameters, and the tables below give
the rest. Code references point to `code/`. Figure numbers are those of
the paper. The figure files keep the numbering of an earlier draft
(`fig2_pm.pdf`, `fig4_pm.pdf` and `fig5_pm.pdf` are Figs. 2, 3 and 4, and
`fig3_pm.pdf` is the supplementary content figure). Section 9 ends with
the axes, curves and legends of Figs. 2 to 4 as drawn, and README.md
describes them in full.

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
| Tokens for Figs. 2 to 4 and the supplementary content figure | drawn with `p_v` (`exp_pm_ext.py`, `sample`; `exp_pm_content.py`, `draw_all`), independent across users at `rho = 0` and, with the shared-token probability `rho`, the same token for all users (section 7) |
| Tokens for the text metrics | the test streams (`exp_pm.py`, `Evaluator`) |
| Frame energy | unit average energy; `c` is estimated from frames drawn with `p_v` (8,192 frames per training step, 65,536 at the end) |

## 2. Channel model and channel classes

User `u` sees subcarrier `k` as `z_k = g_{u,k} x_k + w_k`, `w_k ~ CN(0, 2 sigma_u^2)`,
with SNR `gamma_u = 1 / (d sigma_u^2)` (`rev_channel.channel`). Gains are drawn
independently for every user and every frame unless a row below holds them.
A channel class fixes the gain and SNR statistics of the users, and each
static entry is trained for one class.

| Class (`COND` in `exp_pm.py`) | Gains | SNR offsets per user | Estimation error |
|---|---|---|---|
| `flat` | one `g_u ~ CN(0,1)` on all subcarriers | none | none |
| `off6` | flat | -6, -2, 2, 6 dB | none |
| `off12` | flat | -12, -4, 4, 12 dB | none |
| `nmse01`, `nmse05`, `nmse10` | flat | none | `g_hat = g + e`, `e ~ CN(0, NMSE)`, NMSE = 0.01, 0.05, 0.1, one `e` per user and frame |
| `fs4`, `fs8` | `T = 4` or `8` taps `a_{u,t} ~ CN(0, 1/T)`, `g_{u,k} = sum_t a_{u,t} exp(-j 2 pi t k / K)` | none | none (`e` would be independent per subcarrier) |

| Derived quantity or variant | Value |
|---|---|
| Subcarrier correlation at distance `delta` | `\|sin(pi T delta / K) / (T sin(pi delta / K))\|` |
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
| Mask set | one static entry per load, channel class (`cond`, with its training SNR) and shared-token probability |
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
| Overlap level of adapted masks, `t` in the code and `tau` in the paper (`exp_pm_content_ov.py`, `exp_pm_content_ovfig.py`) | the two-level mask of user `u` has weight on the fraction `f = 1/min(J, U)` of the dimensions; with the overlap level `t` it keeps the level `a` there and places `b = t a`, of the same sign, on every other dimension, with `f a^2 + (1 - f) b^2 = 1`, so `\|\|m_u\|\|^2 = d` for every `t`; `t = 0` is the two-level mask, `t = 1` full overlap (one common mask of equal weights), and with `J = 1` every `t` gives full overlap; for `J >= U` the mean cross-energy is 0 at `t = 0`, 0.015 at `t = 0.025`, 0.059 at `t = 0.05` and `U - 1 = 3` at `t = 1`; `t` is one real number of the entry, chosen on validation frames (lowest SER, ties to the smaller `t`) from `T_GRID` = 0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.3, 0.5, 1 or, from 12 dB in `exp_pm_content_ovfig.py --part snr`, from `T_FINE` = 0, 0.0125, 0.025, 0.0375, 0.05, 0.075; not trained; `c` from 65,536 frames drawn at `rho` (seed 0), on the masks of the comb pattern in `exp_pm_content_ov.py`, where it depends on `rho` and `t` only, and with the gains of the class at 10 dB for every channel class, `J`, `t` and `rho` in `exp_pm_content_ovfig.py` (the `c` of the two-level masks for `t = 0` with `J >= U`) |
| Signaling per update | two-level masks: `b_J = J log2 U` bits for all users for `J >= U` (the user index of each block, broadcast once), `b_J = U ceil(log2 J)` bits for `J < U` (the block that each user weights); the block size, the weight rule and `c` are fixed by the entry in use, so every receiver derives its adapted mask from the signaled indices; the overlap level `t` is a value of the entry as well and adds no bits |

## 5. Decoder

| Item | Value |
|---|---|
| Rule | weighted minimum distance over all `V` tokens: score `2a sum_i w_i r_i (e_v m_u)_i - a^2 sum_i w_i (e_v m_u)_i^2`, `a = 1/c` (`Masks.scores`) |
| Knowledge at receiver `u` | the codebook, its own mask, `c`, and its own channel estimate |
| Cost | about `V d` multiply-accumulate operations per user and token, `2 V d` under frequency-selective fading |
| Variants (`exp_pm_decoder.py`) | unweighted minimum distance, the correlator of the earlier design, normalized correlation, and a rule aware of the estimation error (Gaussian likelihood given `g_hat` with the other users as interference) |
| Receivers that decide the frame kind (`exp_pm_content.py`) | a test of "one token shared by all users" against "independent tokens" with the prior `rho` or 0.5, then decoding under the chosen kind; a genie told the kind |
| The same receivers with adapted masks (`exp_pm_content_fs.py`) | every receiver knows the user index of each block, tests the two kinds with the prior `rho` clipped to [0.001, 0.999], and decodes a shared token on all blocks with the summed masks or its own token on the blocks that its mask weights; a genie told the kind |
| Adapted masks with an overlap level (`exp_pm_content_ov.py`, `exp_pm_content_ovfig.py`) | the rule of the first row (rule (4) of the paper), unchanged; receiver `u` forms its mask from the signaled indices and the overlap level `t` of the entry and uses the weights `\|g_{u,k}\|^2` |

## 6. Compared schemes

| Scheme | Definition | What it isolates |
|---|---|---|
| OMA | each user sends on `d/U` adjacent real dimensions the 16 bits of its token index uncoded: one bit per real dimension at `U = 4` (BPSK per dimension, that is QPSK on eight adjacent subcarriers), BPSK with each bit sent twice at `U = 2`, natural-mapped 4-PAM at `U = 8`; weighted combining and nearest-level detection (`rev_channel.oma_*`) | conventional digital multiple access |
| OMA on adapted blocks | OMA as above, but every user sends its 16 bits on the blocks of two subcarriers that its adapted mask weights (`exp_pm_chanhi.py`, `run_oma`), detected by sign with exact gains | the gain of choosing strong subcarriers without a spread codeword |
| Coded OMA | the 16 index bits encoded by RM(2,5) into 32 code bits, Gray-mapped 16-QAM on the eight subcarriers of the user, table scaled to unit energy under `p_v`, weighted maximum likelihood over the `V` valid tokens; the code bits pass the bit interleaver of `pm_oma_coded_il.json` (searched over 20,000 permutations, seed 0), so two tokens differ on at least 4 of the 8 subcarriers (2 without it); on every `U`-th subcarrier (`coded_il_comb`) or on the blocks of the adapted masks (`coded_il_adapted`, gains known at the transmitter); not trained (`exp_pm_oma.py`); its SER depends on neither `J` nor `rho`, so Fig. 3 draws it at 10 dB as two constant lines, under flat fading on every `U`-th subcarrier (`pm_content_ov_codedref.csv`, SER 0.155, `exp_pm_content_ovfig.py --part codedref`) and at 8 taps on the adapted blocks (`pm_semfig2.csv`, SER 0.00166) | the gain of a channel code at the same rate, with and without known gains |
| Coded OMA, other variants | single parity check with Wagner decoding, RM(2,5) without the interleaver, adjacent placement, 2 and 8 users (`exp_pm_oma.py --part coded`, `--part sem`); the text metrics of `--part sem` use RM(2,5) without the interleaver, whose TER under flat fading at 10 dB is 0.220 on tokens drawn with `p_v` (`pm_oma_coded.csv`) against 0.205 with the interleaver, and whose SER on the test streams is 0.166 (`pm_oma_coded_sem.csv`) against 0.155 | sensitivity of the coded reference |
| ToDMA | every token has a fixed unit-norm random Gaussian codeword (the `rand` codebook, seed 0), the frame is the plain sum of the `U` codewords scaled by `c` (`c^2 = 4.066` under `p_v`), no masks; each receiver zero-forces and detects `U` tokens by orthogonal matching pursuit with signed correlation, a weighted least-squares refit and the weight `1 / (sigma^2/\|g_hat_k\|^2 + (U-1)/(c^2 d))` per dimension, the best of five detectors tested; a genie counts user `u` correct when its own token is among the detected ones, which favors ToDMA; not trained (`exp_pm_todma.py`) | a fixed codeword per token unrelated to its meaning, without masks |
| Random codebook | `rand` codebook with the same masks, mask training, normalization and decoder | the pretrained geometry |
| Static entry | the pattern with `L = 16` trained offline for a channel class and stored in the mask set; no gains at the transmitter | the stored mask set |
| Adapted masks | two weight levels on blocks of `B = 2` subcarriers in Fig. 2 and on `J` blocks in Figs. 3 and 4, formed from the gains known at the transmitter (section 4) | adaptation to the channel |
| Adapted masks with an overlap level | the adapted masks above with the level `b = t a` on every other block (section 4), `t` chosen on validation frames for the channel class, mask size, SNR and `rho`; decoded by rule (4); Fig. 2 and Fig. 4 at `rho = 0.9`, Fig. 3 at `rho = 0.7` and 0.9 | adaptation to the shared content |
| Untrained pattern with `L = U` (masks without overlap) | user `u` has the weight `sqrt(U)` on dimensions `u, u+U, u+2U, ...` (`FreeMasks`); a fixed reference pattern, the special case without overlap | separation |
| Full overlap | one common mask of equal weights, so every user has the same weight on every dimension and energy `d`; a fixed reference pattern, the other special case | combination |
| Trained masks for shared content | `L = 16` patterns trained at the given `rho` and at 10 dB, one mask per user in every frame, decoded by the rule of section 5 (under flat fading in the supplementary content figure, and for 4 and 8 taps in `exp_pm_content_fs.py`, whose entry for four taps and `rho = 0.9` gives the thin line of Fig. 4) | learned partial overlap |

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
| Model | in every frame all users receive the same token with the shared-token probability `rho`, independent tokens otherwise (`sample`); it stands for many users that need the same tokens, for example several cameras that observe the same scene or robots that converse at one site; the tokens are identical, and similar rather than identical content is left as future work; `exp_pm_content.py --part pairs` lets users (1, 2) and (3, 4) share tokens with probability `rho`, independently per pair |
| Mask use | the transmitter knows `rho` and keeps the entry for that `rho` in every frame, so a user keeps one mask whether or not the tokens are shared, and its receiver decodes its own token by the rule of section 5 without being told the frame kind |
| Values of `rho` | `rho` = 0, 0.25, 0.5, 0.75, 0.9, 1 (supplementary content figure, `pm_semfig4_static.csv`, `pm_content_fs.csv` and `pm_content_ov.csv`), 0.7 (`pm_semfig4_static_rho0.7.csv`, `pm_content_ov_j_rho0.7.csv`), 0.5 (`pm_content_ov_j_rho0.5.csv`) and 0.9 (`pm_content_ov_j.csv`, `pm_content_ov_snr.csv`); Fig. 2 draws `rho = 0` and, for the adapted masks with an overlap level, `rho = 0.9`; Fig. 3 draws `rho` = 0, 0.7, 0.9 and writes these values on the plot beside the curves; Fig. 4 draws `rho` = 0 and 0.9 |
| Overlap level of the adapted masks | one real level `t` per entry, chosen on validation frames for `rho` (section 4); the chosen `t` is 0 at `rho = 0` in every tested case, grows with `rho` and shrinks as the SNR grows (`pm_content_ov.json`, `pm_content_ov_fig.json`) |
| Cross-energy label | mean over users of `(1/d) sum_i m_{u,i}^2 sum_{k != u} m_{k,i}^2` of the trained masks |
| Cross-energy `iota_u` and signed overlap `gamma_u` | `iota_u = (1/d) sum_i m_{u,i}^2 sum_{k != u} m_{k,i}^2` and `gamma_u = m_u^T mbar_u / d` with `mbar_u = sum_{k != u} m_k`; both are 0 without overlap and `U - 1` for full overlap; fields `iota` and `kappa` of `pm_content8.json` hold `iota_u` and `gamma_u` per user |
| Shared and independent frames under an isotropic codebook | after equalization a shared token `v` reaches receiver `u` as `c^-1 e_v * (m_u + mbar_u)` plus noise (`*` elementwise), so the weights of `mbar_u` with the sign of `m_u` add to the wanted codeword and its amplitude is multiplied by `1 + gamma_u`; independent tokens see the interference `iota_u / d`; the frame constant is `c^2 = U + rho sum_u gamma_u`, and the stored `c` of `pm_content8.json` lies within 0.3 percent of this value in all 32 entries |

## 8. Metrics

| Metric | Definition |
|---|---|
| TER | `TER_u = Pr[s_hat_u != s_u]`, the fraction of wrongly decoded tokens, averaged over the users (columns `ter`, `ser`, and the other columns whose name contains `ser` but not `sem`) |
| SER (semantic error rate) | `SER_u = E[1 - zeta(s_u, s_hat_u)] = sum_v p_v sum_{w != v} Pr[s_hat_u = w \| s_u = v] (1 - zeta(v, w))`, with `zeta(v, w)` the cosine of the 768-dim BERT input embeddings `w_v` and `w_w`; a correct token costs 0, and decoded ids outside `[0, V)` count as `[UNK]` (columns `ser_sem*`, `sem_dist`, `ub_sem`, `mc_sem`) |
| Topic accuracy | `textattack/bert-base-uncased-ag-news` on the decoded articles (max length 128); error-free text gives 0.950 (BERT tokens) |
| GTR similarity | cosine between `sentence-transformers/gtr-t5-base` embeddings of the decoded and the sent article |
| Net efficiency (Fig. 4) | `eta = (U (1 - TER) log2 V - b_J / F) / (U log2 V)` with `b_J` of section 4 and `F` frames served by one update while the gains stay fixed; the static entry needs no update, `eta = 1 - TER`; payload 59.6 bits per frame at `U = 4`; Fig. 4 plots the loss `1 - eta = TER + b_J / (F U log2 V)` on a logarithmic axis for `F` = 1, 3, 10, 30 and 100 at `rho = 0`, and at `rho = 0.9` for the adapted masks with the overlap level chosen for 0.9 at `F = 100` (the overlap level adds nothing to `b_J`) and for the static entry trained for the class and `rho = 0.9` (`eta = 1 - TER`) |
| Feedback variant | the same `eta` with `2 U J` bits added to `b_J`, two bits per user and block for the gains at every update |

## 9. Parameters per figure and claim

| Item | Script and part | SNR | Frames per point | Data |
|---|---|---|---|---|
| Fig. 2, all schemes, 4 and 8 taps | `exp_pm_semfig2.py --part all`, then `--part extra` | 0-20 dB, step 2 | 200,000 for OMA and ToDMA; 200,000 below 12 dB and 2,000,000 from 12 dB for coded OMA, the static entry and the adapted masks; part `extra` at 8 taps: 40,000,000 for the adapted masks at 20 dB, and 20,000,000, 100,000,000 and 1,000,000,000 for coded OMA on the adapted blocks at 16, 18 and 20 dB. The SER axis of Fig. 2 runs from 10^-6 to 2, so of these four points only coded OMA at 16 dB (20,000,000 frames, the longest point shown) appears; the other three and the adapted masks at 8 taps and 18 dB (2,000,000) lie below 10^-6 and stay in the file. Every point shown rests on at least 29 token errors | `pm_semfig2.csv`, `pm_semfig2_check.csv` |
| Fig. 2, static entry trained for the class | the saved `white_U4_L16_fs4_s1`, `_fs8_s1` (`exp_pm.py --part channel`) | | | |
| Fig. 2, adapted masks with the overlap level chosen for `rho = 0.9`, `B = 2` (`J = 16`), 4 and 8 taps | `exp_pm_content_ovfig.py --part snr` | 0-20 dB, step 2 | validation 40,000 below 12 dB, 400,000 at 12 and 14 dB, 2,000,000 from 16 dB; test 200,000 below 12 dB and 2,000,000 from 12 dB, extended to about 80 token errors at the validation TER of the chosen `t` (at most 120,000,000), which gives 3,555,556 at 20 dB (4 taps, shown) and 6,956,522 and 53,333,334 at 18 and 20 dB (8 taps, below the axis of Fig. 2 and kept in the file); rows for `t = 0` on the same test frames where these number at most 2,000,000. Every point shown rests on at least 83 token errors | `pm_content_ov_snr.csv`, `pm_content_ov_fig.json` |
| Fig. 3, static entries, `rho` = 0, 0.25, 0.5, 0.75, 0.9, 1, `L = J` = 1 to 64 | `exp_pm_semfig34.py --part static` (`train_rho` at 10 dB, training seeds 1-3) | 10 dB | 40,000 test frames (seed 111) and 40,000 validation frames (seed 121) | `pm_semfig4_static.csv` |
| Fig. 3, static entries, `rho = 0.7`, `L = J` = 1 to 64 | `exp_pm_semfig34.py --part static --rho 0.7` (`train_rho` at 10 dB, training seeds 1-3) | 10 dB | 40,000 test frames (seed 111) and 40,000 validation frames (seed 121) | `pm_semfig4_static_rho0.7.csv` |
| Fig. 3 selection | each static entry is the run with the lowest `ser_sem_val` on the validation frames; the plotted SER is on the test frames | | | |
| Fig. 3 (8 taps) and Fig. 4 (4 taps), two-level adapted masks, `J = 1..64` | `exp_pm_semfig4a.py --part adapted` | 10 dB (0 and 20 dB also for `J` = 16, 32) | 40,000; 400,000 when a point has fewer than 100 token errors, which only the 20 dB points needed, so every point that Figs. 3 and 4 take from this file has 40,000 | `pm_semfig4_adapted.csv` |
| Fig. 3 (8 taps) and Fig. 4 (4 taps), adapted masks with the overlap level chosen for `rho = 0.9`, and for `rho` = 0.7 and 0.5, `J = 1..64` | `exp_pm_content_ovfig.py --part jsweep`, `--part jsweep --rho 0.7`, `--part jsweep --rho 0.5` | 10 dB | 40,000 validation and 40,000 test frames; 400,000 test frames when the chosen `t` has fewer than 100 token errors, which no point needed | `pm_content_ov_j.csv`, `pm_content_ov_j_rho0.7.csv`, `pm_content_ov_j_rho0.5.csv`, `pm_content_ov_fig.json` |
| Fig. 3, coded OMA under flat fading (every `U`-th subcarrier, the code of Fig. 2) | `exp_pm_content_ovfig.py --part codedref` | 10 dB | 200,000 | `pm_content_ov_codedref.csv` |
| Fig. 4 | computed from the TER of the adapted masks of Fig. 3 at four taps and of the class-trained static entry of Fig. 2 (`replot_pm.py`, `fig_tradeoff`), update intervals `F` = 1, 3, 10, 30, 100; at `rho = 0.9`, the adapted masks with the overlap level chosen for 0.9 at four taps (`F = 100`) and the static entry trained for the class and `rho = 0.9` | 10 dB | 40,000 (adapted masks, and the static entry at `rho = 0.9`), 200,000 (static entry at `rho = 0`) | `pm_semfig4_adapted.csv`, `pm_semfig2.csv`, `pm_content_ov_j.csv`, `pm_content_fs.csv` |
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
| Overlap level of the adapted masks, `B = 2` (`J = 16`), 4 and 8 taps, `rho` = 0, 0.25, 0.5, 0.75, 0.9, 1, every `t` of `T_GRID` | `exp_pm_content_ov.py` | 5, 10, 15 dB | 40,000 validation and 40,000 test frames (the test frames of `exp_pm_content_fs.py --part main`) | `pm_content_ov.csv`, `pm_content_ov.json` |
| Water-filled real weights | `exp_pm_ext.py --part adapt` | 0, 5, 10, 15, 20 dB | 40,000 | `pm_adapt.csv` |
| Gain held over time | `exp_pm_block.py --part block` | 0, 5, 10, 15, 20 dB | test streams, two draws | `pm_block_eval.csv` |
| Vocabulary size | `exp_pm_vocab.py --part vocab --offline` | 0, 5, 10, 15, 20 dB | 40,000 | `pm_vocab.csv` |
| Decoders and norms | `exp_pm_decoder.py --part norms`, `--part decoders` | 10, 20 dB, NMSE 0, 0.01, 0.05, 0.1 | 40,000 | `pm_norms.csv`, `pm_decoder.csv` |
| Trained codebook, long training | `exp_pm_trainedcb.py --part train`, `--part converge` | 0, 10, 20 dB; validation at 10 dB | 40,000 test, 20,000 validation frames | `pm_trainedcb.csv`, `pm_converge.csv` |

Figures as drawn by `replot_pm.py` (data only), on one 3.0 x 2.25 in canvas:

| Figure | Axes, curves and legends |
|---|---|
| Fig. 2 | SER from 10^-6 to 2, a tick at every decade, against 0-20 dB; 4 taps solid with filled markers, 8 taps dashed with hollow markers; Static (blue circles), Adapted (red squares), "Adapted, rho = 0.9" (pink squares, the adapted masks with the overlap level chosen for `rho = 0.9`, the only curves at `rho = 0.9`), OMA (black diamonds), Coded OMA (grey hexagons, every `U`-th subcarrier), "Coded OMA, adapted" (brown pentagons, the blocks of the adapted masks), ToDMA (olive triangles); a framed legend with these seven at the lower left and a second framed box beside it with "4 taps" (filled marker) and "8 taps" (hollow marker), shown by the marker face only; the six 8-tap points below 10^-6 (two-level adapted masks, adapted masks with the overlap level, coded OMA on the adapted blocks, each at 18 and 20 dB) leave through the lower edge and stay in `pm_semfig2.csv` and `pm_content_ov_snr.csv`; only points with at least 20 token errors are drawn |
| Fig. 3 | SER at 10 dB from 1.2 x 10^-4 to 2 against `J` = 1 to 64, `U = 4`; static entries trained at 10 dB under flat fading (blue, filled circles, best of three runs by the validation SER) and adapted masks at 8 taps (red, hollow squares, two-level at `rho = 0`, with the overlap level chosen for `rho` = 0.7 and 0.9), solid at `rho = 0`, dash-dotted at 0.7, dashed at 0.9; the values of `rho` written on the plot beside the curves, with arrows that end in small ellipses; dotted constant lines of coded OMA under flat fading (grey hexagons) and at 8 taps on the adapted blocks (brown pentagons); one framed legend at the lower left with Static, Adapted, Coded OMA and "Coded OMA, adapted"; a green dashed line at `J = U` labeled "J = U" and a purple dashed line at `J = 64` labeled "full-length mask" |
| Fig. 4 | loss `1 - eta` at 10 dB from 5 x 10^-4 to 4 against `J`, `U = 4`, four taps; adapted masks updated every `F` = 1, 3, 10, 30, 100 frames (red ramp, filled squares, solid and dashed lines in turn), stars at the best `J` of each curve ("Best J"), the pink dash-dotted curve "F = 100, rho = 0.9", the thick blue line labeled "Static" (the static entry) and the thin dash-dotted blue line labeled "Static, rho = 0.9" (the static entry trained for `rho = 0.9`); a gray region labeled "masks overlap, J < U" and the note "signaling exceeds gain" with an arrow; one framed legend in two columns at the lower right |

## 10. Random seeds

| Use | Seed |
|---|---|
| Training | `set_seed(s)` with `s = 1` (main), `s = 1, 2, 3` where three runs are reported (`s = 1, 2` in `exp_pm_content8.py`) |
| Codebook rotation and random codebook | generator seed 0 |
| Decoding on the test streams | `777 + 10 snr` |
| Tokens drawn with `p_v` (`exp_pm_ext.ser` and the scripts that reuse it) | `11 + 10 snr`; validation `21 + 10 snr` |
| Adapted masks | `31 + 10 snr + 100 T + int(7 u_d / 2)`, `u_d = d / J` real dimensions per block |
| Constant `c` of fixed and adapted masks in `--part sig`, `exp_pm_chanhi.py` and the scripts added later | 0 |
| Fig. 2 (`exp_pm_semfig2.py`) | OMA `5000 + snr + 100 T + 1000 b` for block `b` of 40,000 frames, decoding `777 + snr + 1000 b`; coded OMA `9500 + 100 T + snr`; ToDMA `71 + int(10 snr)`; static entry `11 + int(10 snr)`, `13 + int(10 snr)` at 16-20 dB; adapted masks `31 + int(10 snr) + 100 T + 14`; part `extra` keeps these seeds, so the first frames of its points equal the released points |
| `exp_pm_content_ov.py` | validation frames `21 + int(10 snr) + 100 T`, test frames `11 + int(10 snr)` (those of `exp_pm_content_fs.py --part main`), batches of 512; constant `c` 0 |
| `exp_pm_content_ovfig.py` | part `jsweep`: validation `41 + int(10 snr) + 100 T + int(7 u_d / 2)`, test `31 + int(10 snr) + 100 T + int(7 u_d / 2)` (the seeds of `exp_pm_semfig4a.py`); part `snr`: validation `41 + int(10 snr) + 100 T + 14`, test `31 + int(10 snr) + 100 T + 14` (the seed of the adapted masks of Fig. 2); chunks of 384 in both; part `codedref`: `9500 + 100 T + snr` with `T = 1`, that is 9610; constant `c` 0 |
| Coded OMA without the interleaver (`exp_pm_oma.py --part coded`) | `9000 + 100 T + snr` |
| `exp_pm_cross.py` | masks `31 + 10 snr + 100 T + 14`, OMA `61 + 10 snr + 100 T` (+1000 on the adapted blocks), estimation errors and aging from a second generator seeded with the point seed + 100000 |
| `exp_pm_content.py --part pairs`, `exp_pm_semfig34.py` (also with `--rho 0.7`) | test frames 111, validation frames 121 at 10 dB |
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
