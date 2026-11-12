# Scalable Shared Embedding: Simulation Code and Data

Code and raw results for

> K.-H. Lee, H.-H. Choi, and J.-R. Lee, "Scaling Shared-Embedding Multiple
> Access to Large Vocabularies via Low-Dimensional Masks,"
> submitted to *IEEE Wireless Communications Letters*.

Corresponding authors: Hyun-Ho Choi and Jung-Ryun Lee.

Each user's token is sent as its row of the token-embedding table of a
pretrained language model (BERT base uncased, with the GTR base model as a
second model), reduced to 64 real dimensions by a projection whitened under
the source token statistics. The transmitter and every receiver hold this
codebook and never train it. Only the real-valued user masks are trained.
The frame is the sum of the masked codewords of all users, and each
receiver decodes its token by weighted minimum distance over the whole
vocabulary.

A real-valued mask sets the sign and the energy of its user on every frame
dimension. Separate dimensions per user, as in orthogonal multiple access
(OMA), and full overlap, one common mask of equal weights, are two special
cases, and real weights also allow any partial overlap and unequal
energies. A mask can therefore be trained or formed for each load, channel
state and shared content, so a transmitter needs many masks, which it
keeps in a mask set and announces to the receivers. A low-dimensional mask
is a real-valued mask described by J values, far fewer than the d = 64
dimensions of the frame. A static entry of the mask set repeats a trained
periodic pattern of length L (J = L), and a mask adapted to the subcarrier
gains known at the transmitter holds one real value per block of B
subcarriers (J = 32/B, up to J = 64 with blocks of one real dimension).
The adapted masks tested here are the coarsest such masks, with two weight
levels, so an update carries no weight bits. Low-dimensional masks cut the
stored and signaled mask values several times with little loss. For
independent content under flat fading the best mask is the special case
without overlap, which an untrained pattern as long as the number of users
already reaches, and under shared content training finds a partial
overlap. Attention-based transceivers that use such masks to handle
interference and to extract or combine meaning are left as future work,
and no result is claimed for them.

At low and moderate SNR the scheme has a lower semantic error rate than
OMA, coded or not, with the same channel knowledge, and the received text
keeps its topic more often than with uncoded OMA. Coded OMA leads at high
SNR, where both error rates are small, and in token errors it passes our
curves between 8 and 16 dB. Against a random codebook, the pretrained
codebook loses less meaning per wrong token at every SNR.

The last section keeps the earlier design of this repository, periodic
codebook reuse, with its code and data.

Every simulation parameter (system, channel classes, codebook, masks and
training, decoder, compared schemes, shared content, metrics, parameters
per figure, seeds, software) is listed in [ENVIRONMENT.md](ENVIRONMENT.md).

## Metrics and column names

Every rate is the mean over the users.

- Token error rate, TER_u = Pr[ŝ_u ≠ s_u].
- Semantic error rate, SER_u = E[1 − ζ(s_u, ŝ_u)]
  = Σ_v p_v Σ_{w≠v} Pr[ŝ_u = w | s_u = v] (1 − ζ(v, w)), where ζ(v, w) is
  the cosine similarity of the 768-dimensional BERT input embeddings w_v
  and w_w. A correct token costs nothing, a wrong token costs the meaning it
  loses, and the SER equals the TER when wrong tokens carry no meaning
  (ζ = 0). A decoded identifier outside the vocabulary counts as `[UNK]`.
- Topic accuracy, the share of decoded AG News articles that
  `textattack/bert-base-uncased-ag-news` assigns to their topic (error-free
  text gives 0.950 with BERT tokens).
- GTR similarity, the cosine of the `sentence-transformers/gtr-t5-base`
  embeddings of the decoded and the sent article.

Figs. 2 and 3 of the paper plot the SER. Fig. 4 plots the loss 1 − η of the
net efficiency η, which uses the TER, since it counts payload bits. Files
written before the SER was defined call the TER `ser`, so the columns of
`data/pm/` read as follows.

| Columns | Quantity |
|---|---|
| `ter`, `ser`, and every other column whose name contains `ser` but not `sem` (for example `ser_val`, `ser_shared`, `ub_ser`, `mc_ser`, `ser10_val`) | TER |
| `ser_sem`, `ser_sem_val`, `ser_sem_shared`, `ser_sem_indep` (files `pm_semfig*.csv` and `pm_content_fs*.csv`), `sem_dist` (files graded on the AG News test streams), `ub_sem` and `mc_sem` (bound files) | SER |

## Contents

| Path | Content |
|---|---|
| `code/exp_pm.py` | Codebooks, mask training, pattern length and load, channel classes, bounds, frame energy of the masks |
| `code/exp_pm_ext.py` | Shared content, masks adapted to known gains, signaling, static entries trained at 10 dB, and the TER data of the figures |
| `code/exp_pm_chanhi.py` | High-SNR points of the static entry trained for each frequency-selective class and of the adapted masks, and OMA placed on the blocks of the adapted masks |
| `code/exp_pm_semfig2.py`, `code/exp_pm_semfig34.py`, `code/exp_pm_semfig4a.py` | SER and TER of Figs. 2 and 3 and of the supplementary content figure |
| `code/exp_pm_oma.py` | Coded OMA (single parity check and RM(2,5), with and without a bit interleaver) on fixed and on adapted subcarriers, and its text metrics |
| `code/exp_pm_todma.py` | Token-domain multiple access (ToDMA) reference, its detectors and its frame energy |
| `code/exp_pm_cross.py` | Loads of two, four and eight users crossed with frequency-selective fading, estimation errors at either end, pilot estimates and aged gains (no training) |
| `code/exp_pm_nearfar.py` | Unequal average SNRs, entries trained with one score factor per receiver, per-user metrics, energy split across users |
| `code/exp_pm_content.py`, `code/exp_pm_content8.py` | Shared content with receivers that test the frame kind, pairs of users that share content, and eight users |
| `code/exp_pm_content_fs.py` | Shared content under 4 and 8 taps for the adapted masks, both fixed patterns and static entries, with the decoder of the paper and with receivers that decide the frame kind first |
| `code/exp_pm_semctl.py` | Semantic controls, codebooks shuffled within frequency classes or drawn at random under the same projection, and OMA with its out-of-range identifiers scored three ways |
| `code/exp_pm_block.py` | Text metrics with a gain held over 16 or 64 frames or over an article |
| `code/exp_pm_vocab.py` | Error rate against the vocabulary size and with the GPT-2 token table |
| `code/exp_pm_boundov.py` | Gaussian step and bounds of Section III-B with and without overlap, and fixed-gain bounds over all tokens |
| `code/exp_pm_decoder.py` | Masked codeword norms and decoder variants under estimation errors |
| `code/exp_pm_trainedcb.py` | A codebook trained together with the masks (full and sampled softmax), and mask training ten times longer |
| `code/exp_pm_oma_energy.py` | Frame energy of the OMA reference on the test streams |
| `code/exp_pm_spread.py` | Spread of the difference between nearest codewords over the subcarriers, for the comparison with the bit interleaver of coded OMA |
| `code/replot_pm.py` | Every figure of the study from `data/pm/` (plotting only) |
| `code/sse_lib.py`, `code/rev_channel.py` | Shared helpers and the channel simulator, used by both designs |
| `data/pm/` | Every result of the low-dimensional-mask study (75 CSV and JSON files) |
| `fig/fig1_pm.tex`, `fig/fig1_pm.pdf` | Fig. 1, standalone TikZ source and PDF |
| `fig/fig2_pm.pdf`, `fig/fig4_pm.pdf`, `fig/fig5_pm.pdf` | Figs. 2, 3 and 4 of the paper |
| `fig/fig3_pm.pdf`, `fig/fig_codebook_pm.pdf` | Supplementary figures, the content step and the codebooks |

The figure files keep the numbering of an earlier draft, in which the
content step was Fig. 3. The docstrings of the scripts other than
`replot_pm.py` use that numbering, so their Fig. 3 is the supplementary
content figure (`fig3_pm.pdf`), their Fig. 4 is Fig. 3 of the paper
(`fig4_pm.pdf`), and their Fig. 5 is Fig. 4 of the paper (`fig5_pm.pdf`).

The trained masks are not stored. Each mask file also holds the fixed
codebook, so it takes about 7.8 MB (8.2 MB with the GTR codebook), and the
70 files of `exp_pm.py` take about 0.55 GB. The scripts retrain them into
`data/pm/pm_models/` (git-ignored), where the other training scripts also
write their masks. One 3,000-step training takes about 19 s on an idle GPU.

## Figure map (low-dimensional masks)

All data files are in `data/pm/`, and all scripts are in `code/`.
`replot_pm.py` draws Figs. 2 to 4 and the two supplementary figures from
these files only. Figs. 2 to 4 share one 8:6 canvas of 3.0 x 2.25 in, drawn
at the printed width, with 8 pt axis labels, 7 pt tick labels and 6.5 pt
legends. The script refuses a figure whose legend covers a curve, a marker,
a reference line or an annotation, or whose text leaves the canvas.

A static entry holds the masks of all users trained offline for one channel
class, load and level of sharing, and the mask set holds one entry for each
of them. The mask size J counts the values that define one mask, not a
number of masks. It is the pattern length L of a static entry or the number
of blocks of an adapted mask, and J = 64 is the full-length mask.

| Item | Script (part) | Data |
|---|---|---|
| Fig. 1, system with a fixed pretrained codebook and a set of real-valued low-dimensional masks, here a static entry. Each mask cell carries a plus or a minus sign on a shade of one color, so the cells mark the sign and the size of each weight | `fig/fig1_pm.tex` (standalone TikZ) | none |
| Fig. 2, SER against SNR at 4 taps (solid lines, filled markers) and 8 taps (dashed lines, hollow markers), U = 4 and independent tokens (rho = 0), for uncoded OMA, coded OMA on every U-th subcarrier and on the adapted blocks, ToDMA, the static entry trained for each class (L = 16) and masks adapted to known gains on blocks of two subcarriers. Color and marker name the scheme, and the legend shows the two tap classes by the marker face only, without a line | `exp_pm_semfig2.py --part all` | `pm_semfig2.csv`, column `ser_sem` (and `ter`), schemes `oma`, `coded_il_comb`, `coded_il_adapted`, `todma`, `static_class_L16` and `adapted_B2`. The scheme `todma_unmatched`, a variant of the semantic cost of a missed ToDMA token, is stored but not drawn. Points with fewer than 20 token errors (column `errors`) are not drawn. `pm_semfig2_check.csv` compares the token errors with those of the released TER files `pm_figs.csv`, `pm_figs_hi.csv`, `pm_figs_class.csv`, `pm_figs_class_hi.csv`, `pm_oma_coded_il.csv` and `pm_todma.csv` |
| Fig. 3, SER at 10 dB against the mask size J = 1 to 64, for static entries with L = J trained at 10 dB under flat fading for rho = 0, 0.5, 0.75 and 0.9 and for masks adapted with J blocks at 4 and 8 taps and rho = 0 | `exp_pm_semfig34.py --part static`, `exp_pm_semfig4a.py --part adapted` | `pm_semfig4_static.csv` (rho = 0, 0.25, 0.5, 0.75, 0.9 and 1, three training runs each, column `values` holds J), plotted as the run with the lowest `ser_sem_val` on validation frames, with its `ser_sem` on the test frames. Column `cross_energy` holds the mean cross-energy of the masks. The rows for rho = 0.25 and 1 are stored but not drawn. `pm_semfig4_adapted.csv`, column `ser_sem` |
| Fig. 4, loss 1 − η of the net efficiency η on a logarithmic axis against the mask size J, at 10 dB, four taps and rho = 0, for adapted masks updated every F = 1, 3, 10, 30 or 100 frames (one red ramp, darker for rarer updates) and for the static entry trained for the class, which needs no update (thick line in the blue of the static entries, labeled on the line) | `replot_pm.py` (`fig_tradeoff`, computed from the data, no simulation) | `pm_semfig4_adapted.csv` (scheme `adapted`, channel `fs4`, 10 dB) and `pm_semfig2.csv` (scheme `static_class_L16`, channel `fs4`, 10 dB), column `ter` of both. Net efficiency η = (U (1 − TER) log2 V − b_J/F)/(U log2 V) with U = 4 and V = 30,522, so the loss is 1 − η = TER + b_J/(F U log2 V), and the static entry loses only its TER (η = 0.936). An update of the two-level adapted masks carries b_J = J log2 U bits for J ≥ U, the user index of each block, and b_J = U⌈log2 J⌉ bits for J < U, the block that each user weights. The gray span marks J < U, where the masks overlap. Stars mark the best J of each curve (4, 4, 8, 8 and 16) |
| Supplementary figure `fig/fig3_pm.pdf`, SER at 10 dB against the shared-token probability rho (all users receive the same token), for the untrained pattern with L = U (masks without overlap), full overlap and static entries trained at 10 dB (L = 16), with the mean cross-energy of the static entries | `exp_pm_semfig34.py --part content` | `pm_semfig3.csv`, schemes `no_overlap`, `full_overlap` and `trained_at_snr`. Columns `ser_sem_shared` and `ser_sem_indep` split the SER into frames with a shared token and frames with independent tokens |
| Supplementary figure `fig/fig_codebook_pm.pdf`, (a) SER and (b) topic accuracy against SNR for the BERT codebook, a random codebook and OMA | `exp_pm.py --part models` | `pm_models_eval.csv` (column `sem_dist`, mean over runs 1 to 3 for the BERT codebook), `pm_models_train.json` (accuracy on error-free text). The GTR codebook and the projections without whitening under the source statistics are in the same files |

### Numbers in the text

| Result | Script (part) | Data |
|---|---|---|
| Bounds of Proposition 1 against simulation under Rayleigh fading on 600 tokens drawn with p_v (Section III-B) | `exp_pm.py --part bound2 --bound_nv 600` | `pm_bound2.csv` |
| The same bounds at a fixed gain on all tokens with p_v > 0, 0 to 14 dB | `exp_pm_boundov.py --part exact` | `pm_bound_exact.csv` (rows `v_set = all_support`), `pm_bound_exact.json` |
| SER of the BERT codebook, the random codebook and OMA under flat fading at 10 dB, topic accuracy and GTR similarity at 0 dB, projections without whitening, GTR codebook (Section IV) | `exp_pm.py --part models` | `pm_models_eval.csv` |
| Measured frame energy of every compared scheme at four users, within 0.3 percent of one (Section IV), and of OMA at eight users | `exp_pm.py --part energy`, `exp_pm_todma.py`, `exp_pm_oma.py --part sem`, `exp_pm_oma_energy.py` | `pm_energy.json`, `pm_todma_energy.json`, `pm_oma_coded_sem.csv`, `pm_oma_energy.csv` |
| Gaussian value (6) and the TER bound for masks built to overlap (Section III-B) | `exp_pm_boundov.py --part overlap` | `pm_boundov.csv` (rows `pep_nn` and `ser`) |
| Topic accuracy of OMA at an equal TER (interpolated on the OMA rows), and the meaning gain of the pretrained geometry against shuffled and random tables | `exp_pm_semctl.py --part ctrl` | `pm_semctl_eval.csv`, `pm_semctl_train.json` |
| Adaptation factors at 10 dB, SNRs where coded OMA leads, ToDMA error floor | `exp_pm_semfig2.py --part all` | `pm_semfig2.csv` |
| Topic accuracy and GTR similarity of coded OMA with the interleaver, the static entry, the adapted masks and OMA under 4 and 8 taps | `exp_pm_oma.py --part sem_il` | `pm_oma_coded_il_sem.csv` |
| Pilot estimates with an NMSE of 1/SNR at both ends, fixed estimation errors, aged gains, two, four and eight users under 4 and 8 taps | `exp_pm_cross.py --part load`, `--part csi`, `--part merge` | `pm_cross.csv`, with the per-user rates and the constants c in `pm_cross.json` (`pm_cross_load.*` and `pm_cross_csi.*` hold the two parts) |
| Untrained pattern with L = U against the static entry for independent content, under flat fading and under 4 and 8 taps | `exp_pm.py --part comb`, `--part channel`, `exp_pm_cross.py --part load`, `exp_pm_semfig2.py --part all`, `exp_pm_content_fs.py --part main` | `pm_comb_eval.csv`, `pm_channel_eval.csv` (flat fading). Under 4 and 8 taps the TER at 10 dB is in `pm_cross.csv` (part `load`, scheme `comb`, four users) and `pm_semfig2.csv` (scheme `static_class_L16`), and the SER of both is in `pm_content_fs.csv` (schemes `comb_std` and `static_class_L16`, rho = 0) |
| SNR offsets of ±6 and ±12 dB, the untrained pattern with L = U and entries trained with one score factor per receiver against OMA, per-user rates (the weakest user is user 0) | `exp_pm_nearfar.py --part retrain`, `--part peruser` | `pm_nearfar_eval.csv`, `pm_nearfar_train.json` |
| User energies ‖m_u‖² proportional to SNR_u^(−1/2) under the same SNR offsets, for the untrained pattern with L = U and for OMA, per-user and mean rates (Section IV) | `exp_pm_nearfar.py --part split` | `pm_nearfar_split.csv`, rows with `a` = 0.5 against the equal energies of `a` = 0 (the energy of user u is 10^(−a·offset_u/10) divided by its mean over the users, column `energy`) |
| Cross-energy of the masks trained for shared content, SER at rho = 0.9 against both fixed patterns, SER per frame kind | `exp_pm_semfig34.py --part content` | `pm_semfig3.csv` |
| Sizes and signs of the weights of entries trained for shared content (stored patterns with L = 16, four and eight users, two training runs) | `exp_pm_content8.py --part u4`, `--part u8`, `--part merge` | `pm_content8.json` (field `pattern`) |
| Overlap of masks trained at 0, 10 and 20 dB | `exp_pm_ext.py --part content_snr` | `pm_content_snr.csv` |
| SER of the two-level adapted masks against the shared-token probability rho under 4 and 8 taps, with rule (4) and with a receiver that first decides whether the frame carries one shared token (Section IV) | `exp_pm_content_fs.py --part main`, `--part check`, `--part hi` | `pm_content_fs.csv` (schemes `adapted_std`, `adapted_test` and `adapted_genie` at 10 and 20 dB, with the untrained pattern, full overlap and static entries for the same rho), `pm_content_fs_hi.csv` (the adapted masks at 20 dB on 400,000 frames), `pm_content_fs.json` (training logs, measured frame energies and the checks against the released files) |
| Masks trained for eight users used by four, eight users with L = U against OMA | `exp_pm.py --part period` | `pm_period_eval.csv` |
| Pattern length under estimation errors and for two or eight users | `exp_pm.py --part plen`, `--part period` | `pm_plen_eval.csv`, `pm_period_eval.csv` |
| Real weights water-filled over the blocks that each adapted mask weights, against two weight levels | `exp_pm_ext.py --part adapt` | `pm_adapt.csv` (`waterfill` 1 against 0) |
| Loss of the best adapted masks against that of the static entry for an update every 100 frames, and the best J for an update in every frame | `replot_pm.py` (`fig_tradeoff` prints the best J and its loss for every F) | `pm_semfig4_adapted.csv`, `pm_semfig2.csv` |
| Best J with two feedback bits per user and block in every update (2UJ bits added to b_J) | arithmetic on the data of Fig. 4 | `pm_semfig4_adapted.csv`, `pm_semfig2.csv` |
| OMA ahead under flat fading with an NMSE of 0.05 or more | `exp_pm.py --part channel` | `pm_channel_eval.csv` |

### Supporting results

| Result | Script (part) | Data |
|---|---|---|
| Pattern length for two, four and eight users, masks of one load used at another, training curves | `exp_pm.py --part period` | `pm_period_eval.csv`, `pm_period_train.json` |
| Pattern length per channel class, isotropy statistics of the codebook | `exp_pm.py --part plen` | `pm_plen_eval.csv`, `pm_plen_train.json` |
| Eight channel classes with masks trained for each class, the flat-trained masks used unchanged, a random codebook, and every class-trained mask applied in every class (rows `cross_class`) | `exp_pm.py --part channel` | `pm_channel_eval.csv`, `pm_channel_train.json` |
| Untrained comb pattern without overlap (L = U) and with pairs (L = 2U), for four and eight users | `exp_pm.py --part comb` | `pm_comb_eval.csv`, `pm_comb.json` |
| Bounds on the TER and the SER against Monte Carlo, unclipped | `exp_pm.py --part bound` | `pm_bound.csv` |
| The same bounds at a fixed unit gain | `exp_pm.py --part bound3 --bound_nv 600` | `pm_bound3.csv` |
| Gaussian step of eq. (6) and the bounds for ten sets of masks from no overlap to full overlap | `exp_pm_boundov.py --part overlap` | `pm_boundov.csv` |
| Frame energy of every trained entry on the test streams | `exp_pm.py --part energy` | `pm_energy.json` |
| Frame energy of OMA on the test streams | `exp_pm_oma_energy.py` | `pm_oma_energy.csv` |
| TER of the Fig. 2 schemes from the first runs, with 40,000 frames (graph `g1`) or 400,000 frames from 12 dB, and the flat-trained static entry (rows `static_L16`) | `exp_pm_ext.py --part figs`, `exp_pm_chanhi.py --part hi`, `--part class`, `--part class_hi` | `pm_figs.csv`, `pm_figs_hi.csv`, `pm_figs_class.csv`, `pm_figs_class_hi.csv` |
| Uncoded OMA on fixed adjacent subcarriers and on the blocks of two subcarriers that each adapted mask weights, 4 and 8 taps, TER | `exp_pm_chanhi.py --part oma_adapt` | `pm_oma_adapt.csv` |
| Coded OMA without the interleaver (single parity check and RM(2,5)) on adjacent, comb and adapted subcarriers, flat fading and 4 and 8 taps, TER | `exp_pm_oma.py --part coded` | `pm_oma_coded.csv` |
| Text metrics and frame energy of coded OMA at two, four and eight users and under 4 and 8 taps | `exp_pm_oma.py --part sem` | `pm_oma_coded_sem.csv` |
| RM(2,5) with the bit interleaver, comb and adapted placements, TER, and the interleaver | `exp_pm_oma.py --part coded_il` | `pm_oma_coded_il.csv`, `pm_oma_coded_il.json` (two tokens differ on at least 4 of a user's 8 subcarriers with the interleaver and on 2 without it) |
| Spread of the difference between a codeword of codebook (1) and its nearest neighbor over the subcarriers (participation ratio over all 32 and over a user's 8 interleaved subcarriers) | `exp_pm_spread.py` | `pm_spread.json` |
| ToDMA, TER under flat fading and 4 and 8 taps, five detectors, frame energy | `exp_pm_todma.py --part all` | `pm_todma.csv`, `pm_todma_detector.csv`, `pm_todma_energy.json` |
| TER of the static entries trained at 10 dB for rho = 0 and 0.9, and for rho = 0.25, 0.5, 0.75 and 1 | `exp_pm_ext.py --part sig10`, `--part sig10rho` | `pm_sig10.csv`, `pm_sig10_rho.csv` |
| TER of the adapted masks against J at 4 and 8 taps (graph `g3` of `pm_figs.csv` holds an earlier single-run version). The values at 10 dB equal column `ter` of `pm_semfig4_adapted.csv`, which Fig. 4 reads | `exp_pm_ext.py --part sig` | `pm_sig.csv` |
| Signaling per update of the adapted masks and TER at 10 dB, with and without water-filling | `exp_pm_ext.py --part adapt` | `pm_adapt.csv` |
| Correlated content, rho = 0, 0.5, 0.9 and 1, trained masks with L = 16 and L = 64 against the comb pattern and full overlap | `exp_pm_ext.py --part corr` | `pm_corr.csv`, `pm_corr.json` (cross-energy per user and constant c) |
| Receivers that test whether a frame carries a shared token, and pairs of users that share tokens | `exp_pm_content.py --part test`, `--part pairs` | `pm_content_test.csv`, `pm_content_pairs.csv`, `pm_content_pairs.json` |
| Shared content with eight users, trained masks against both fixed patterns | `exp_pm_content8.py --part u4`, `--part u8`, `--part merge` | `pm_content8.csv`, `pm_content8.json` |
| Receivers that cancel the other users' decoded tokens, with and without a test of the common-token hypothesis | `exp_pm_ext.py --part sic` | `pm_sic.csv` |
| One common and one private token per user and frame with fixed masks | `exp_pm_ext.py --part rs` | `pm_rs.csv` (schemes `shared_sic`, `shared_nosic`, `no_overlap_copies`) |
| Coherence-aware selection among 24 stored entries, in each of which every mask weights one of four blocks of eight subcarriers, with and without water-filling | `exp_pm_ext.py --part csit` | `pm_csit.csv` |
| Other user energies under SNR offsets for the comb pattern and OMA (exponents `a` = 0.25, 0.75 and 1, where `a` = 1 equalizes the average SNRs) | `exp_pm_nearfar.py --part split` | `pm_nearfar_split.csv` |
| OMA identifiers outside the vocabulary deleted, as in the released scoring, or replaced by a token drawn at random | `exp_pm_semctl.py --part omasub` | `pm_semctl_omasub.csv` |
| Text metrics with a gain held over 16 or 64 frames or over an article | `exp_pm_block.py --part block` | `pm_block_eval.csv`, `pm_block_articles.csv` (per article) |
| Vocabularies of the 1,024 to 30,522 most frequent BERT tokens and the GPT-2 table | `exp_pm_vocab.py --part vocab --offline` | `pm_vocab.csv` |
| Masked codeword norms weighted by p_v and uniformly | `exp_pm_decoder.py --part norms` | `pm_norms.csv` |
| Decoders under estimation errors (rule (4), minimum distance, correlation, normalized correlation, a rule aware of the estimation error) | `exp_pm_decoder.py --part decoders` | `pm_decoder.csv` |
| Codebook trained with the masks, random or pretrained start, full or sampled softmax | `exp_pm_trainedcb.py --part train` | `pm_trainedcb.csv`, `pm_trainedcb_train.json` |
| Mask training for 30,000 steps, L = 16 and 64, flat fading and 4 taps | `exp_pm_trainedcb.py --part converge` | `pm_converge.csv` |

## Running the low-dimensional-mask experiments

Run from `code/` in a fresh clone. Every script writes its results to
`data/pm/` and overwrites the released file of the same name (git restores
it), and the training scripts write their masks to `data/pm/pm_models/`.
The minutes are wall-clock times of the released runs on one NVIDIA RTX PRO
5000 Blackwell. The runs from step 23 on shared that GPU with about ten
other jobs, so their minutes are upper values (an entry that trains in
20 s to 45 s on an idle GPU took 128 s to 240 s there). "Needs" names the
steps whose saved masks a step loads.

| Step | Command | Minutes | Needs |
|---|---|---|---|
| 1 | `python exp_pm.py --part models` | 12.4 | |
| 2 | `python exp_pm.py --part period` | 27.0 | |
| 3 | `python exp_pm.py --part channel` | 30.9 | |
| 4 | `python exp_pm.py --part plen` | 14.0 | |
| 5 | `python exp_pm.py --part bound` | 2.4 | |
| 6 | `python exp_pm.py --part comb` | 3.4 | |
| 7 | `python exp_pm.py --part energy` | 0.3 | 1, 3 |
| 8 | `python exp_pm.py --part bound2 --bound_nv 600` | 2.0 | 3 |
| 9 | `python exp_pm.py --part bound3 --bound_nv 600` | 1.2 | 3 |
| 10 | `python exp_pm_ext.py --part corr` | 7.1 | |
| 11 | `python exp_pm_ext.py --part csit` | 1.0 | |
| 12 | `python exp_pm_ext.py --part adapt` | 10.2 | |
| 13 | `python exp_pm_ext.py --part sic` | 6.5 | |
| 14 | `python exp_pm_ext.py --part rs` | 1.0 | |
| 15 | `python exp_pm_ext.py --part figs` | 13.6 | 2, 3 |
| 16 | `python exp_pm_ext.py --part sig` | 7.3 | 2 |
| 17 | `python exp_pm_chanhi.py --part hi` | 1.4 | 3 |
| 18 | `python exp_pm_chanhi.py --part class` | 0.8 | 3 |
| 19 | `python exp_pm_chanhi.py --part oma_adapt` | below 1 | |
| 20 | `python exp_pm_ext.py --part content_snr` | 8.0 | |
| 21 | `python exp_pm_ext.py --part sig10` | 13.8 | |
| 22 | `python exp_pm_oma_energy.py` | below 1 (CPU) | |
| 23 | `python exp_pm_vocab.py --part vocab --offline` | 1.0 | |
| 24 | `python exp_pm_oma.py --part coded` | 16.7 | |
| 25 | `python exp_pm_oma.py --part sem` | 26.2 | |
| 26 | `python exp_pm_content.py --part test` | 10.6 | |
| 27 | `python exp_pm_content.py --part pairs` | 34.1 | |
| 28 | `python exp_pm_cross.py --part load`, then `--part csi` and `--part merge` | 8.0, 12.6, below 1 | |
| 29 | `python exp_pm_nearfar.py --part retrain` | 53.0 | |
| 30 | `python exp_pm_nearfar.py --part peruser` | 34.6 | 3, 29 |
| 31 | `python exp_pm_nearfar.py --part split` | 25.3 | |
| 32 | `python exp_pm_semctl.py --part ctrl` | 71.8 | 1 |
| 33 | `python exp_pm_semctl.py --part omasub` | 12.6 | |
| 34 | `python exp_pm_block.py --part block` | 59.0 | 1 |
| 35 | `python exp_pm_boundov.py --part overlap` | 52.8 | 3 |
| 36 | `python exp_pm_boundov.py --part exact` | 17.5 | 3 |
| 37 | `python exp_pm_decoder.py --part norms`, then `--part decoders` | 0.2, 23.1 | 2, 3 |
| 38 | `python exp_pm_trainedcb.py --part train` | 22.6 per full-softmax run, 8.3 per sampled-softmax run, then 6.9 of evaluation | 1 |
| 39 | `python exp_pm_trainedcb.py --part converge` | 21.9 to 34.3 per process (four processes) | 3 |
| 40 | `python exp_pm_content8.py --part u4`, then `--part u8` and `--part merge` | 30.1, 40.6, below 1 | |
| 41 | `python exp_pm_todma.py --part all` | about 3 | |
| 42 | `python exp_pm_chanhi.py --part class_hi` | below 3 | 3 |
| 43 | `python exp_pm_oma.py --part coded_il` | 2.9 | |
| 44 | `python exp_pm_oma.py --part sem_il` | 3.1 | 3, 43 |
| 45 | `python exp_pm_ext.py --part sig10rho` | 27.0 | |
| 46 | `python exp_pm_semfig4a.py --part adapted` | 0.7 | |
| 47 | `python exp_pm_semfig34.py --part content` | 4.0 | |
| 48 | `python exp_pm_semfig34.py --part static` | 51.7 | |
| 49 | `python exp_pm_semfig2.py --part all` | about 25 as four processes (adapted 24.7 per channel, static 10.6, coded 13.0, ToDMA 5.1, OMA 0.3) | 3, 43 |
| 50 | `python exp_pm_content_fs.py --part main`, then `--part check` and `--part hi` | 5.4, 0.8, 4.1 | 3, 47 |
| 51 | `python exp_pm_spread.py` | below 1 | |
| 52 | `python replot_pm.py` | below 1 | data only |

Steps 1 to 5 also run in one process as `python exp_pm.py --part all`
(about 87 min). Part `figs` loads `white_U4_L16_flat_s1.pt` (step 3) and
`bertw_U4_L{64,32,16,8,4,2,1}_flat_s1.pt` (step 2), and part `sig` loads
`bertw_U4_L{64,...,1}_flat_s{1,2,3}.pt` (step 2). Part `hi` of
`exp_pm_chanhi.py` loads `white_U4_L16_flat_s1.pt`, and parts `class` and
`class_hi`, part `sem_il` of `exp_pm_oma.py`, `exp_pm_semfig2.py` and
`exp_pm_content_fs.py` load `white_U4_L16_fs4_s1.pt` and
`white_U4_L16_fs8_s1.pt` (all from step 3). Part `check` of
`exp_pm_content_fs.py` also loads `sem_fig3_rho0.9_L16_s1.pt` (step 47)
and compares its points with the released files of steps 26, 47 and 49.
`exp_pm_block.py` loads `bert_pwhite_U4_L16_flat_s{1,2,3}.pt` and
`bert_rand_U4_L16_flat_s1.pt` (step 1), which `exp_pm_semctl.py` and the
evaluation of `exp_pm_trainedcb.py` also grade. `exp_pm_boundov.py` loads
`white_U4_L16_flat_s1.pt`, and `exp_pm_decoder.py` loads
`white_U4_L{16,64}_flat_s1.pt` (step 3) and `bertw_U4_L64_flat_s1.pt`
(step 2). `exp_pm_oma.py --part sem_il` and `exp_pm_semfig2.py` read the
interleaver from `pm_oma_coded_il.json` (step 43), and `exp_pm_spread.py`
reads the subcarrier counts of the interleaver from the same file. The
other scripts train or build their own masks.
`exp_pm_semfig2.py` also runs as five parts (`--part oma`, `coded`,
`todma`, `static`, `adapted`, the long parts split with `--channel fs4`
or `fs8`) followed by `--part merge`. Fig. 1 compiles inside `fig/` with
`latexmk -pdf fig1_pm.tex`.

A quick test such as `python exp_pm.py --part period --smoke --iters 40
--tag _smoke` writes its masks to `data/pm/pm_models_smoke/` and its
results with the suffix `_smoke` (both git-ignored). Without `--tag` it
overwrites the released results. Keep the default `--book pwhite`, since
the mask file names do not record the codebook and a run with another
codebook replaces the masks that later parts load. The scripts added for
the semantic error rate and the new references also take `--smoke`, which
runs a few points and writes either nothing or files with the suffix
`_smoke`.

### Known differences between a rerun and the released files

- `pm_bound.csv` uses the default `--bound_nv 400`, whereas `pm_bound2.csv`
  and `pm_bound3.csv` need `--bound_nv 600`. This value sets the sample of
  tokens in the bound columns, not the Monte Carlo columns.
- `pm_energy.json` lists 30 trained entries. Two of them,
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
  give the same TER at 10 dB. Parts `csit`, `adapt` and `figs` also
  estimate the normalization constant of the fixed masks from 65,536 draws
  without a seed of their own, whereas part `sig`,
  `exp_pm_chanhi.py --part hi` and the scripts added later seed this draw
  (seed 0).
- `exp_pm_chanhi.py` uses the seeds and the chunking of part `figs`, so
  the first 39,936 frames of each point are drawn as in graph `g1`, whose
  40,000 frames end with a shorter chunk. Its adapted masks use the seeded
  normalization constant, so on these frames they can differ from `g1` by
  a few errors.
- `exp_pm_semfig2.py` reuses the seeds and chunks of the released TER runs
  and records the token errors at the released frame counts in
  `pm_semfig2_check.csv`. On the GPU type and torch version of the released
  runs they equal the released errors (column `exact`), except the adapted
  points of graph `g1`, whose constant c was drawn without a seed and which
  differ by a few errors.
- A training run whose signs cancel the frame of a shared token (c = 0) is
  kept as a failed run with `nan` values in `pm_semfig4_static.csv` and
  `pm_sig10_rho.csv` (rho = 1, L = 1, run 3).
- `exp_pm_vocab.py --offline` loads GPT-2 only from the local Hugging Face
  cache and skips it, with a message, when the model is missing.
- `exp_pm_content8.py` writes the per-part files `pm_content8_u4.*` and
  `pm_content8_u8.*`, and `exp_pm_semfig2.py` writes per-part files to
  `data/pm/semfig2_parts/`. Both are git-ignored, and only the merged files
  are released.
- The Hugging Face model and dataset revisions are not pinned.

## Conventions (low-dimensional masks)

- Real-vector model with d = 64 real dimensions per frame. Real dimensions
  2k and 2k+1 form subcarrier k of K = 32, counted from 0 as in the code
  (the paper counts from 1, dimensions 2k−1 and 2k). The average frame
  energy is one under the source token statistics, with the normalization
  constant c estimated by Monte Carlo, and the SNR is γ = 1/(dσ²).
- Users: U = 4 unless a part states otherwise (parts `period`, `comb`,
  `exp_pm_cross.py`, `exp_pm_content8.py` and `exp_pm_oma.py --part sem`
  also use two or eight users).
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
- Masks: the mask m_u of user u is a real-valued vector of d weights. The
  weight m_{u,i} sets the sign and m_{u,i}² the energy of user u on
  dimension i. Masks without overlap, with separate dimensions per user as
  in OMA, and full overlap are two special cases, and real weights also
  allow any partial overlap and unequal energies. The energies sum to
  Σ_u ‖m_u‖² = U d, and every mask has energy ‖m_u‖² = d unless a part
  states otherwise (`exp_pm_nearfar.py --part split` scales the energies
  of the users). The frame is x = (1/c) Σ_u e_{s_u} ⊙ m_u, where e_{s_u}
  is the codeword of user u.
- Static entries: L trained real values per user, of either sign, scaled to
  norm √L and repeated d/L times, so every mask has energy d.
- Training: Adam with learning rate 0.01, 256 frames per step, 3,000 steps,
  tokens drawn from p_v, training SNR uniform in 0 to 20 dB per frame
  (fixed at 10 dB for the static entries of Fig. 3 and of the supplementary
  content figure), and cross-entropy over the whole vocabulary with a
  learned score factor (one per receiver in `exp_pm_nearfar.py`). TF32 is
  on during training only. Each static entry is trained for one channel class,
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
  probability rho and independent tokens otherwise. A user keeps one mask
  whether or not the tokens are shared, and its receiver decodes its own
  token by weighted minimum distance without being told the frame kind.
  `exp_pm_content.py` tests receivers that decide the frame kind first,
  and `exp_pm_content_fs.py` tests them with the adapted masks under 4 and
  8 taps, where every receiver knows the user index of each block.
- Fixed reference patterns: the two special cases, used without training.
  The untrained comb pattern (L = U) gives user u the weight √U on the
  dimensions u, u+U, u+2U, ..., so the masks do not overlap, and full
  overlap is one common mask of equal weights with energy d for every
  user.
- Adapted masks: the transmitter knows every user's subcarrier gains. An
  adapted mask holds one real value on each of J = 32/B blocks of B
  subcarriers, in one partition shared by all users. The tested masks are
  the coarsest such masks, with two weight levels, which need no weight
  bits. With J ≥ U the users take turns in an order reversed every round,
  and each sets the level √U on the block with its largest sum of
  log(1 + SNR) among those with no weight yet. The mask of user u
  then holds √U on J/U blocks and zero on the others, so its energy is d,
  the masks do not overlap and c stays fixed, and an update broadcasts
  only the user index of each block, b_J = J log2 U bits for all users.
  With J < U each user sets the level √J on its best block, so the masks
  overlap, and the update names the block that each user weights,
  b_J = U⌈log2 J⌉ bits. Finer real levels are admissible with more
  signaling, and part `adapt` tests real weights water-filled over the
  blocks that each mask weights.
- OMA: the 16 bits of the token index, uncoded, on d/U adjacent real
  dimensions per user. At U = 4 every real dimension carries one bit
  (BPSK per dimension, that is QPSK on eight adjacent subcarriers), at
  U = 2 every bit is sent twice, and at U = 8 natural-mapped 4-PAM
  carries two bits per dimension. Its frame energy on the test streams is
  1 at U = 2 and 4 and 1.22 (BERT) or 1.29 (GTR) at U = 8
  (`pm_oma_energy.csv`), because the bits of the token indices are not
  equiprobable (the top bit is always zero).
- Coded OMA: the 16 index bits encoded by RM(2,5) into 32 code bits,
  Gray-mapped 16-QAM on the eight subcarriers of a user, scaled to unit
  energy under p_v and decoded by weighted maximum likelihood over the V
  valid tokens. The bit interleaver of `pm_oma_coded_il.json` makes two
  tokens differ on at least four of the eight subcarriers (two without
  it). Fig. 2 places it on every U-th subcarrier (`coded_il_comb`) or on
  the blocks of the adapted masks (`coded_il_adapted`).
- ToDMA: every token has a fixed unit-norm random Gaussian codeword (the
  random codebook, seed 0), the frame is the plain sum of the U codewords
  scaled to unit energy under p_v, each receiver zero-forces and detects U
  tokens by orthogonal matching pursuit weighted by noise and
  interference, and a genie counts a user correct when its own token is
  among the detected ones, which favors ToDMA.
- Test streams (`exp_pm.py`): the first 2,000 articles of the AG News test
  split, article i to user i mod U, every stream cut to the shortest
  (24,998 frames at U = 4 with BERT tokens, one token per user and frame,
  1,948 graded articles). Columns `ter` (the TER), `sem_dist` (the SER),
  `err_tok_sim` and `err_close` (mean cosine between sent and decoded
  token over the wrong tokens, and the share of wrong tokens with cosine
  above 0.4), `topic_acc` and `gtr_sim`.
- `exp_pm_ext.py` scores the TER on tokens drawn from p_v, 40,000 frames
  per point (20,000 in part `sic`, 30,000 in part `rs`), and
  `exp_pm_chanhi.py` uses 400,000 frames from 12 dB on with the seeds of
  part `figs`. Frames per point of the other scripts are in ENVIRONMENT.md.
- Seeds: codebook 0, training 1 (1 to 3 where a part states it), decoders
  on the test streams 777 + 10·SNR, and tokens drawn from p_v 11 + 10·SNR
  for test frames and 21 + 10·SNR for validation frames. ENVIRONMENT.md
  lists the seeds of every script.

## Requirements (low-dimensional masks)

The released runs use Python 3.12.3, PyTorch 2.11.0 (CUDA 12.8),
transformers 5.22.1, sentence-transformers 6.2.0, datasets 5.1.0, NumPy
2.5.4 and Matplotlib 3.11.2. The scripts download the models
`bert-base-uncased`, `sentence-transformers/gtr-t5-base` and
`textattack/bert-base-uncased-ag-news` and the dataset `fancyzhx/ag_news`
(train and test splits), about 1.1 GB in all. `exp_pm_vocab.py` also uses
`gpt2` from the local Hugging Face cache. A GPU is recommended, and
`SSE_DEVICE=cpu` forces the CPU. `replot_pm.py` needs NumPy, Matplotlib
3.10 and the Times New Roman font (the released figures come from
Matplotlib 3.10.8). Figs. 2 to 4 share an 8:6 canvas of 3.0 x 2.25 in
drawn at the printed width, and the supplementary content figure keeps a
2.6 x 1.95 in canvas. The layout check refuses a figure whose text leaves
the canvas or whose legend covers a curve. Without Times New Roman,
Matplotlib uses the wider DejaVu Serif instead and the check refuses every
figure. Matplotlib 3.11 draws taller legend rows, so its check can refuse
figures that pass under 3.10. Fig. 1 needs LaTeX with TikZ and the
`standalone` class.

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
| Fig. 3 (SER vs training MACs) | `replot_revision.py conv` | `rev_budget_curve_{sse,full,sampled,hier}_mean.csv` (10-seed means by `exp_rev_budget_eval.py --aggregate-only` of `rev_budget_curve_{scheme}[_s2\|...\|_s10].csv`. Seed 1 comes from `exp_rev_budget.py --seed 1`, which logs 2e4 frames per point, and seeds 2-10 from `exp_rev_budget.py --seed {2,...,10} --log-frames 200000`, seeds 4-10 through `fig3_seed_queue.sh`) |
| Table I | none | `snr_sweep.csv`, `sota_snr.csv`, `rev_snr_fullmd.csv`, `rev_budget_final_mean.csv` (10-seed means; per seed `rev_budget_final_{scheme}[_s2\|...\|_s10].csv`), `complexity.json` |
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
