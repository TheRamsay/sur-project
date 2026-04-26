---
title: "SUR 2025/2026: Multimodal target-speaker detector"
author: "Dominik Huml (`xhumld00`)"
date: "\\today"
papersize: a4
geometry: "top=1.6cm,bottom=1.6cm,left=1.8cm,right=1.8cm"
fontsize: 10pt
header-includes:
  - \usepackage{booktabs}
  - \usepackage{etoolbox}
  - \usepackage{graphicx}
  - \graphicspath{{/Users/ramsay/school/sur/project/docs/}{./}}
  - \AtBeginEnvironment{figure}{\centering}
---

## 1. Task

The goal is a binary detector for the target speaker `m431`. Each evaluation sample is a face image and a paired voice recording sharing the same filename stem, so multimodal fusion at the sample level is straightforward. Three systems are required (image-only, audio-only, and a multimodal fusion of the two), each producing a real-valued score and a hard decision at prior 0.5. No pretrained models or outside data are allowed. Everything was trained from scratch on the provided 222-sample corpus with augmentation only.

## 2. Validation strategy

A plain K-Fold split leaks both session and speaker identity, since the target appears in only three sessions and most non-target speakers in one. I use a 3-fold leave-one-out scheme that groups target rows by session (`01`, `02`, `03`) and non-target rows by speaker, both fed to a single `GroupKFold` (`src/data/splits.py`). All three systems share the fold assignment, which lets the fusion stack on OOF scores. I pool the `_train` and `_dev` partitions into one 222-sample manifest before folding so that every target session takes a turn as held-out, rather than relying on the single train/dev split.

I report EER and min-DCF at prior 0.5, mean ± std across folds. For fusion I pool all 222 samples into a single OOF ranking and report the overall EER. Augmentation is train-only, val is always raw, and every fitted statistic (scaler, PCA basis, UBM, MAP target) is refit per fold. As a sanity check I ran a permutation test (E029). Shuffling train labels brought val EER back to 49 % (image) and 55 % (audio), confirming the models learn from labels and not auxiliary leakage.

## 3. Audio system

### 3.1 Front-end

Speaker verification on ~148 training utterances per fold is the small-data regime, where front-end inductive bias matters more than expressive power. LPCC parameterises the vocal-tract transfer function directly via an all-pole LPC model and the cepstral recursion. That is the physical quantity that separates speakers. MFCC's Mel filterbank, by contrast, was designed for content recognition where speaker identity is the nuisance to suppress. I ablated MFCC, FBank, SDC, PLP and LPCC under identical UBM-MAP conditions. LPCC 13 + $\Delta$ + $\Delta\Delta$ at LPC order 12 won on both EER and min-DCF (4.21 % MFCC vs 3.33 % LPCC). Higher-dimensional alternatives over-parameterised GMM-32 for this dataset size, and order 12 had a 2.5 pp moat over neighbours, matching the `fs/1000 + 2` rule for 16 kHz speech.

Cepstral mean normalisation (CMN) is applied per utterance to remove the convolutional channel response (microphone, room, level), which is constant within an utterance but varies across sessions. CMVN regressed by 2 pp (E012) because the UBM covariance already absorbs per-utterance scale, so CMN is retained and CMVN is not.

### 3.2 Classifier: UBM-MAP with tied covariance

The backbone is the standard UBM-MAP setup. A 32-component GMM is trained on all non-target training frames to get the UBM, then the target model is produced by MAP-adapting only the UBM means with relevance factor `r = 16`. For an utterance with frames `x_1, …, x_T`, the score is the per-frame log-likelihood ratio averaged over the utterance:

$$\mathrm{LLR}(x_{1:T}) = \frac{1}{T} \sum_{t=1}^{T} \bigl[ \log p(x_t \mid \lambda_\text{target}) - \log p(x_t \mid \lambda_\text{UBM}) \bigr]$$

The relevance factor `r` trades off per-speaker observations against the UBM prior. I confirmed `r` $\in$ \{4, 8, 16\} is a flat plateau on both diagonal and tied covariance (E013, E044) and rejected UBM-64 (E010, over-fit). The biggest single jump in the audio track came from changing the GMM covariance type (E037).

\begin{table}[h]
\centering
\renewcommand{\arraystretch}{1.1}
\setlength{\tabcolsep}{10pt}
\begin{tabular}{@{}lrrr@{}}
\toprule
\textbf{Covariance} & \textbf{EER (\%)} & \textbf{min-DCF} & \textbf{Parameters} \\
\midrule
spherical          & 3.89 $\pm$ 3.75    & 0.0778          & 32     \\
diagonal           & 4.35 $\pm$ 4.40    & 0.0870          & 1\,248 \\
\textbf{tied}      & \textbf{0.69 $\pm$ 0.98} & \textbf{0.0139} & \textbf{1\,521} \\
full               & 1.48 $\pm$ 0.92    & 0.0296          & 48\,672 \\
\bottomrule
\end{tabular}
\caption{GMM covariance ablation under fixed UBM-32, MAP $r = 16$, LPCC features. Tied is the empirical sweet spot.}
\label{tab:cov}
\end{table}

The reason this matters: LPCC coefficients are strongly correlated since adjacent cepstra share formant structure. Diagonal ignores this. Full overfits at 48\,672 parameters. Tied is the right compromise. It captures the off-diagonal structure once and shares it across all 32 components. Numbers in Table \ref{tab:cov}.

### 3.3 Augmentation and TTA

Train-time augmentation applies two transforms to the LPCC pipeline. Pitch shift \(\pm\)1–2 semitones (E025) trains the UBM to ignore F0 while preserving formant ratios. The same shift hurt MFCC in E014 because the Mel filterbank already smooths pitch. Codec simulation (E052) downsamples each utterance to 8 kHz and back, destroying frequencies above 4 kHz where many discriminative formants live. The unaugmented system reached 0.46 % clean but collapsed to 13.33 % under bandwidth-limited val. Codec aug closes that gap at zero clean cost (Figure \ref{fig:audio-stress}).

\begin{figure}[ht]
\centering
\includegraphics[width=0.92\linewidth]{figures/alt_d_audio_robustness.pdf}
\caption{Audio stress test (E051 + E052 paired). Codec aug closes the bandwidth failure mode (13.33 $\to$ 3.33 \%) and incidentally improves moderate-noise robustness with no clean-side regression.}
\label{fig:audio-stress}
\end{figure}

At inference I apply speed-only test-time augmentation (E031). The utterance is scored at 1.0$\times$, 0.9$\times$ and 1.1$\times$, and the three LLRs are averaged. Speed retiming preserves the spectral envelope so the LPC all-pole filter (and LPCC) is invariant. Pitch shift at inference, on the other hand, drifts the cepstrum onto a manifold the UBM was never trained on, and fold 0 collapses to 9.86 %. Pitch is the right train-time augmentation but the wrong test-time one.

The final audio system is LPCC + tied-covariance UBM + MAP r = 16 + pitch and codec augmentation + speed TTA, reaching **0.46 ± 0.65 % EER** at **min-DCF 0.0092**.

## 4. Image system

### 4.1 Features and classifier

The image classifier is intentionally simple: 80\(\times\)80 grayscale crops, per-pixel standardisation, 50-D PCA, logistic regression with C = 1. LBP (E005) collapsed to 45 % EER on fold 2 under lighting shifts, since histograms are global-brightness-sensitive. Fisherfaces (E006) are capped at a 1-D projection with two classes and lose to logreg in the 50-D PCA space. Sweeps confirmed `n_pca` = 50 (E011) and `C` = 1 with L1 catastrophic (E040). Histogram equalisation and CLAHE tripled EER (E041), and pyramid multi-scale PCA lost to flat PCA + augmentation (E036). Plain PCA on standardised pixels was the empirical ceiling.

### 4.2 Augmentation: two-pass adversarial training (E033)

Augmentation is what made the image system competitive, and it runs in two passes.

**Pass 1.** Each training image is added in four versions: original, horizontal flip, brightness-scaled by U[0.7, 1.3], and Gaussian pixel noise at $\sigma$ = 15. A fresh PCA + LogReg is fit on the combined set. Brightness jitter is the key contributor: session 03 (held out in fold 2) has systematically different lighting and brightness scaling is the only standard augmentation that crosses the gap.

**Pass 2.** For each training image I query the Pass-1 model at five angles in [\(-\)10°, +10°] and pick the rotation that lands closest to the decision boundary. A rotated copy at that angle is added and PCA + LogReg is refit. Random rotation (E015) samples uniformly, so PCA fits easy angles and ignores hard ones. Adversarial selection inverts this. Principal components are reallocated towards the directions of model uncertainty. It is hard-negative mining for PCA. Robustness numbers are summarised in Figure \ref{fig:image-stress}.

\begin{figure}[ht]
\centering
\includegraphics[width=0.90\linewidth]{figures/alt_a_image_robustness.pdf}
\caption{Image stress test (E051). AdvRot is photometrically bulletproof (JPEG, blur, downsample stay at clean 0.51 \%) and halves the geometric weaknesses. JPEG, blur, contrast, HE/CLAHE and Cutout 20\(\times\)20 (E052) all regressed when added on top.}
\label{fig:image-stress}
\end{figure}

At inference I average the original and its horizontal flip. Rotation TTA was rejected (E030, eigenface corruption). E043 (flip + small-rotation TTA) was rejected after E049 failed to replicate the +0.23 pp gain.

The final image system reaches **0.51 ± 0.36 % EER** at **min-DCF 0.0102**.

## 5. Fusion

Each modality produces a raw LLR (audio) or logit (image) on its own scale. I fit a Platt calibration per modality on its OOF scores (one-feature logreg, C = 1e6, balanced weights), then form a weighted sum on the 2-simplex (`w_mfcc + w_lpcc + w_image = 1`, non-negative) with weights chosen on a 51\(\times\)51 grid that directly minimises OOF EER. Logistic-regression fusion was also tried. It is MLE-optimal but not EER-optimal, and the near-rank-2 design matrix (`r(MFCC, LPCC)` = 0.843) limits it.

\begin{table}[h]
\centering
\renewcommand{\arraystretch}{1.1}
\setlength{\tabcolsep}{10pt}
\begin{tabular}{@{}lrrl@{}}
\toprule
\textbf{Fusion configuration} & \textbf{OOF EER (\%)} & \textbf{min-DCF} & \textbf{Notes} \\
\midrule
MFCC + image                 \hfill \textsc{e009}      & 3.75              & 0.0750          & bimodal baseline \\
LPCC + image                 \hfill \textsc{e023}      & 0.52              & 0.0104          & LPCC replaces MFCC \\
MFCC + LPCC + image          \hfill \textsc{e026/27}   & 0.26              & 0.0052          & trimodal, MFCC as tiebreaker \\
\textbf{E052 + E033 + MFCC}  \hfill \textsc{e039}      & \textbf{0.26}     & \textbf{0.0052} & \textbf{new backbones, 0 / 222} \\
\bottomrule
\end{tabular}
\caption{Score-level fusion progression. EER computed on the pooled OOF ranking of all 222 samples.}
\label{tab:fusion}
\end{table}

DET curves for the three streams plus the fused decision are plotted in Figure \ref{fig:det}, and the disjoint-failure structure that makes the fusion possible is shown in Figure \ref{fig:complementarity}.

\begin{figure}[ht]
\centering
\includegraphics[width=0.64\linewidth]{figures/fig6_det_curve.pdf}
\caption{DET curves for the three modalities and the trimodal fusion. The fusion dot sits at the lower-left corner of the visible region. On this CV split it makes 0 errors out of 222 samples.}
\label{fig:det}
\end{figure}

The grid converges to `w_image` = 0.66, `w_lpcc` = 0.34, `w_mfcc` $\approx$ 0 (Table \ref{tab:fusion}). Image dominates as the lower-EER modality. LPCC adds complementary signal. 0 of 222 samples are misranked by both audio and image at their thresholds, so even though each modality misses a few targets the misses are disjoint. MFCC's weight collapses to zero because MFCC and LPCC OOF scores are correlated at `r` = 0.843 (both cepstral views of the same physics). I keep MFCC in the grid so that a future LPCC weakness can re-recruit it automatically. Quality-aware gating (E032), product-rule fusion (E046, E048) and score-level ensembles (E045) all regressed against simple weighted averaging.

\begin{figure}[ht]
\centering
\includegraphics[width=0.85\linewidth]{figures/alt_c_complementarity.pdf}
\caption{Audio $\times$ image complementarity at per-stream EER thresholds. 15 samples are rescued by audio alone, 5 by image alone, none are missed by both. This disjoint failure pattern is what makes weighted-sum fusion reach 0 errors.}
\label{fig:complementarity}
\end{figure}

## 6. Generalization and overfitting defences

I organise the safeguards into five risks, each paired with its defence.

**Risk 1: capacity exceeds data size.** Defences are scale-matched. UBM-32 leaves ~5 400 frames per component (UBM-64 regressed, E010). PCA-50 keeps logreg in a subspace the 2-class boundary fits reliably (E011). Tied covariance has 1 521 parameters vs full at 48 672.

**Risk 2: validation leaks session or speaker identity.** Defence: §2 LOSO. Per-fold scaler, PCA, UBM and MAP target are refit, augmentation is train-only, and val is always raw.

**Risk 3: hidden label leakage via auxiliary channels.** Defence: permutation test (E029). Shuffling train labels brought val EER to 49 % and 55 %, both at chance. The 48 to 55 pp gap to the unshuffled flagships is what makes the result trustworthy.

**Risk 4: post-hoc rationalisation across multi-axis changes.** Defence: one knob per experiment, hypothesis written before the run, failed runs kept in the log. Negative results (LBP, FBank, CMVN, HE/CLAHE, product-rule fusion) form part of the rationale.

**Risk 5: distribution shift at eval time.** Defence: stress testing (E028, E051, E052). The image flagship is photometrically bulletproof. JPEG q = 15, blur $\sigma$ = 3, and downsample 40\(\to\)80 all stay at clean 0.51 %. Remaining weaknesses are geometric (rot $>$ 15°: 7.6 %) and occlusion (11 %). Audio absorbs speed via TTA. Codec stress (13.33 % $\to$ 3.33 %) is what motivated E052. Moderate noise ($\le$ 10 dB SNR) is manageable.

## 7. Results

\begin{table}[h]
\centering
\renewcommand{\arraystretch}{1.15}
\setlength{\tabcolsep}{10pt}
\begin{tabular}{@{}lcc@{}}
\toprule
\textbf{System} & \textbf{EER (\%)} & \textbf{min-DCF} \\
\midrule
\multicolumn{3}{@{}l}{\textit{Audio}} \\
\quad MFCC + GMM \hfill \textsc{e001}                                  & 17.92 $\pm$ 7.81 & 0.2250 \\
\quad MFCC + UBM/MAP + aug \hfill \textsc{e008}                        &  4.21 $\pm$ 3.11 & 0.0509 \\
\quad LPCC + tied covariance \hfill \textsc{e037}                      &  0.69 $\pm$ 0.98 & 0.0139 \\
\quad \textbf{LPCC + tied + pitch\,\&\,codec + speed TTA} \hfill \textsc{e052} & \textbf{0.46 $\pm$ 0.65} & \textbf{0.0092} \\
\addlinespace[2pt]
\multicolumn{3}{@{}l}{\textit{Image}} \\
\quad PCA + LogReg \hfill \textsc{e004}                                &  4.49 $\pm$ 4.26 & 0.0565 \\
\quad PCA + LogReg + aug \hfill \textsc{e007}                          &  0.97 $\pm$ 0.86 & 0.0194 \\
\quad \textbf{PCA + LogReg + aug + AdvRot} \hfill \textsc{e033}        & \textbf{0.51 $\pm$ 0.36} & \textbf{0.0102} \\
\addlinespace[2pt]
\multicolumn{3}{@{}l}{\textit{Fusion}\,\,$^{\dagger}$} \\
\quad MFCC + image (bimodal baseline) \hfill \textsc{e009}             &  3.75 & 0.0750 \\
\quad LPCC + image \hfill \textsc{e023}                                &  0.52 & 0.0104 \\
\quad MFCC + LPCC + image (E008 + E007 backbones) \hfill \textsc{e027} &  0.26 & 0.0052 \\
\quad \textbf{Trimodal (E052 + E033 + MFCC, new backbones)} \hfill \textsc{e039} & \textbf{0.26} & \textbf{0.0052} \\
\bottomrule
\end{tabular}
\caption{Final cross-validation results across the three modalities and the trimodal fusion. $^{\dagger}$\,Fusion EER is OOF on the full 222-sample pool, where the flagship E039 makes 0 / 222 errors. Audio and image rows are mean $\pm$ std across the 3 LOSO folds.}
\label{tab:results}
\end{table}

The arc behind Table \ref{tab:results}: MFCC + GMM baseline $\to$ UBM/MAP $\to$ LPCC $\to$ tied covariance $\to$ codec aug $\to$ adversarial rotation on the image side $\to$ trimodal score-level fusion. Figure \ref{fig:progression} plots the same trajectory across all 53 experiments.

\begin{figure}[ht]
\centering
\includegraphics[width=0.95\linewidth]{figures/alt_b_progression.pdf}
\caption{Project arc across 53 experiments. Each modality flagship dropped roughly two orders of magnitude from its anchor. Audio: 17.92 $\to$ 0.46 \% ($-$97 \%). Image: 4.49 $\to$ 0.51 \% ($-$89 \%). Fusion: 3.75 $\to$ 0.26 \% ($-$93 \%).}
\label{fig:progression}
\end{figure}

### Reproduction

All systems run from a clean clone with

```
uv sync
uv run predict_audio.py   --eval-dir <dir> --output results/audio_lpcc_tied_codecaug.txt
uv run predict_image.py   --eval-dir <dir> --output results/image_pca_adv_rot.txt
uv run predict_fusion.py  --eval-dir <dir> --output results/fusion_trimodal.txt
```

Each output file contains one line per evaluation sample with three whitespace-separated fields: stem, score (higher = more confident target), and a hard decision (`1` = target, `0` = non-target) thresholded at the Bayes optimum for prior 0.5, calibrated on OOF min-DCF.

I submit six result files (the maximum allowed). Three are the flagships above. The other three are intermediate checkpoints kept for ablation evidence: `audio_mfcc_gmm_baseline.txt` (E001), `audio_mfcc_ubm_map_aug.txt` (E008), and `image_pca_baseline.txt` (E004). The story they tell, when read top-to-bottom, is the arc of §7.
