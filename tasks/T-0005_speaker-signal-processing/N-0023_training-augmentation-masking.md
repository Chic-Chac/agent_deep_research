---
id: N-0023
type: note
task: T-0005
title: "Do training-side data processing methods (augmentation, spectrogram masking, simulated mixtures, masked pretraining) raise speaker model accuracy?"
status: final
created: 2026-11-11
tags: [speaker-diarization, data-augmentation, speaker-verification]
---

# Do training-side data processing methods (augmentation, spectrogram masking, simulated mixtures, masked pretraining) raise speaker model accuracy?

Budget-limited pass (about 15 calls). Pages fetched via arXiv HTML/ar5iv through a summarizing fetcher; numbers read from that output, marked [V]. Diarization models themselves: N-0015.

## Waveform augmentation and SpecAugment (speaker verification)

**Takeaway:** In one controlled ablation, pitch-shifting speed perturbation gives about 0.05-0.2 abs EER over noise+reverb; speed-only and SpecAugment give no gain. No ablation for MUSAN-only, RIR-only, codec or bandwidth augmentation was found.

Ablation, ResNet-family SV system, VoxCeleb2 training, EER % (Vox1-O / E / H); N = MUSAN noise, R = RIR — [Build a SRE Challenge System, 2022](https://ar5iv.labs.arxiv.org/html/2211.00815) [V]:

| Augmentation | O | E | H |
|---|---|---|---|
| N+R | 0.947 | 1.098 | 2.002 |
| N+R+SpecAugment | 1.064 | 1.091 | 2.004 |
| N+R+speed only | 0.973 | 1.113 | 2.033 |
| N+R+pitch only | 0.867 | 0.984 | 1.781 |
| N+R+speed+pitch | 0.861 | 0.996 | 1.767 |
| speed+pitch, ratios 0.8-1.2 | 0.803 | 1.001 | 1.777 |

- Authors conclude the speed-perturbation gain comes from pitch change, not tempo — same source [V]. (Speed-perturbed copies as new speakers: not isolated in this paper.)
- SpecAugment on VoxCeleb1-O: +0.117 EER (worse); Vox1-E/H: no change — same source [V]. Single-source for the "SpecAugment does not help on clean-ish large data" claim.
- SpecAugment applied on-the-fly gave 1.47% EER on Vox1-O (x-vector, 2020) and 3.72% / 11.49% on SRE16 Cantonese / Tagalog; no with/without numbers seen — [Wang et al. ICASSP 2020](https://wsstriving.github.io/publication/2020-icassp-specaug) [A].
- Large-margin fine-tuning (LM-FT) after training, EER %: Vox1-O 0.782 to 0.649, Vox1-H 1.658 to 1.394, VoxSRC21-val 3.153 to 2.429; CNSRC22 ResNet34 EER 6.590 to 6.221 (DCF0.01 0.3707 to 0.3543) — same source [V].
- RepSPKNet used 3-fold speed augmentation plus Kaldi MUSAN/RIR with no augmentation ablation; the Kaldi VoxCeleb recipe is the standard baseline — [Rep Works in SV, 2021](https://ar5iv.labs.arxiv.org/html/2110.09720) [V].

**Inference:** Noise+reverb is the baseline; add pitch-changing speed perturbation if adapting with new data; skip SpecAugment unless data are small or noisy (the evidence here is on large VoxCeleb only).

**Gaps:** No numbers for mixup/CutMix/random erasing, codec/bandwidth augmentation, MUSAN-vs-RIR split, or CAM++/ERes2Net ablations (CAM++ arXiv 2303.00332 and ERes2NetV2 arXiv 2406.02167 appeared in search but were not opened). SpecAugment for diarization: nothing found.

## Masked-prediction pretraining (WavLM)

**Takeaway:** WavLM's utterance-mixing denoising task measurably helps speaker tasks, most on diarization.

- Mechanism: 20% of utterances in a batch are mixed with a noise clip or a second utterance at a random region; the model predicts the masked content of the primary speaker. Noise-mix probability 0 for Base, 10% for Base+/Large — [WavLM, 2021](https://ar5iv.labs.arxiv.org/html/2110.13900) [V].
- Ablation, WavLM Base, SUPERB: with vs without denoising task, ASV EER 4.69 vs 4.91; diarization DER 4.55 vs 6.03 (-1.48 abs) — same source [V].
- WavLM Large: SUPERB ASV EER 3.77, DER 3.24; VoxCeleb1 EER 0.617 / 0.662 / 1.318 (O/E/H), 0.383 / 0.480 / 0.986 with LM-FT and calibration; CALLHOME DER 10.35 with EEND-vector clustering — same source [V].

**Inference:** Overlap-aware pretraining targets the speaker-overlap case, which explains the larger DER gain. Pretraining is not feasible for a small team; reuse released WavLM weights.

**Gaps:** No wav2vec 2.0 / HuBERT matched ablation fetched.

## Simulated mixtures and conversations (EEND), small in-domain fine-tuning

**Takeaway:** Simulated conversations beat simulated mixtures; in-domain fine-tuning of a segmentation model roughly halves DER in one 54 h case, and 2 h of synthetic speech gave large gains in another.

2-speaker EEND DER %, [Landini et al. 2022](https://ar5iv.labs.arxiv.org/html/2204.00890) [V]:

| Training data | CALLHOME P2 | DIHARD3 CTS dev | DIHARD3 CTS eval |
|---|---|---|---|
| Simulated mixtures | 15.09 | 25.46 | 23.58 |
| + real fine-tune | 8.44 | 17.98 | 17.44 |
| Simulated conversations | 8.64 | 23.47 | 22.06 |
| + real fine-tune | 7.28 | 16.99 | 17.00 |

- Pyannote segmentation-3.0, embeddings frozen, fine-tuned on 54.18 h of AfriSpeech accented English (no augmentation): DER 21.30 to 10.65 overall; medical 31.46 to 15.73; general 19.03 to 9.52; evaluated on 7 h separate dialogs — [Domain-Aware Diarization, 2025](https://arxiv.org/html/2509.21554v1) [V]. Caution: every fine-tuned figure in the source's Table 8 equals half its baseline to two decimals (also gap 12.43 to 6.21); the paper does not explain this. Treat the size of the gain as unconfirmed; single source.
- Pyannote segmentation-3.0 on Indonesian conversation, synthetic TTS-mixed training only: zero-shot DER 53.47; about 2 h (171 samples) 34.81 (2 epochs); about 25 h 29.24 — [Indonesian adaptation, 2026](https://arxiv.org/html/2601.03684) [V]. Single source; abstract (13.68 abs gain) contradicts its own table (24.23); test-set size unreported.
- Bangla and Bengali shared-task papers fine-tune only segmentation, no numbers seen — [2605.08214](https://arxiv.org/pdf/2605.08214), [2602.21183](https://arxiv.org/pdf/2602.21183) [A].

**Inference:** Segmentation-only fine-tuning fits 16 GB. Gains at 2-25 h are large when the baseline is badly mismatched; expect smaller gains on English interviews where community-1 already scores about 11-19% DER (N-0015).

**Gaps:** No hours-vs-DER curve; no DiariZen or Sortformer fine-tuning numbers; no Korean diarization fine-tuning.

## Test-time processing

**Takeaway:** Cosine scoring plus AS-norm plus calibration gives about 0.2-0.3 abs EER over plain cosine, at no training cost.

EER %, same ResNet system (O / E / H) — [Build a SRE Challenge System](https://ar5iv.labs.arxiv.org/html/2211.00815) [V]:

| Scoring | O | E | H |
|---|---|---|---|
| PLDA | 1.633 | 1.723 | 2.857 |
| Cosine | 1.058 | 1.147 | 2.087 |
| + AS-Norm | 0.920 | 1.048 | 1.874 |
| + calibration (QMF) | 0.861 | 0.996 | 1.767 |

- Calibration improved EER but worsened minDCF on CNSRC 2022, so it was dropped there — same source [V].

**Gaps:** No test-time augmentation, centering/whitening, or duration-aware calibration numbers.

## Language mismatch (Korean)

**Takeaway:** Korean data added to VoxCeleb improves ECAPA-TDNN on all test sets, but exact figures were not obtained.

- ECAPA-TDNN trained on VoxCeleb + Korean beat VoxCeleb-only on every test set (Pusan National University, 2024) — [KoreaScience](https://koreascience.kr/article/JAKO202414143303118.pub) [A].
- Cross-lingual non-target trials show higher scores among Mandarin, Japanese, Thai and Korean in one corpus — [arXiv 2506.02777](https://arxiv.org/html/2506.02777v1) [A].
- Fine-tuning ResNets on a multilingual partition: 0.35% EER on its own trials, better conversational generalization; no VoxCeleb-only baseline seen — [arXiv 2601.16358](https://www.arxiv.org/pdf/2601.16358) [A].

**Gaps:** No Korean EER values; adaptation-method comparison (fine-tuning vs PLDA/centering adaptation) not found.
