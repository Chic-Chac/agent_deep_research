---
id: N-0024
type: note
task: T-0005
title: "Do target-speaker extraction and TS-VAD (speaker-conditioned masking) raise speaker attribution accuracy, and what enrollment do they need?"
status: final
created: 2026-11-11
tags: [speaker-extraction, speech-separation, speaker-diarization]
uses: [N-0022]
---

# Do target-speaker extraction and TS-VAD (speaker-conditioned masking) raise speaker attribution accuracy, and what enrollment do they need?

Budget 15 calls used; fetched pages were summarized by a small model, so tables are `[A]`-grade unless noted. Separation/PixIT/enhancement: N-0022 (not repeated).

## 1. TSE: SI-SDR(i) numbers

**Takeaway:** TSE reaches 16-23 dB SI-SDRi on clean synthetic mixes but 9-17 dB on Libri2Mix/noisy sets; no source found measures its effect on DER or WER of real recordings.

| Model (dataset) | Result | Source |
|---|---|---|
| USEF-TFGridNet, WSJ0-2mix | 23.3 SI-SDRi (15.2M params) | [USEF-TSE 2024](https://arxiv.org/html/2409.02615) [A] |
| X-TF-GridNet / SpEx+, WSJ0-2mix | 20.7 / 17.4 | same [A] |
| USEF-TFGridNet / USEF-SepFormer, WHAM! | 17.6 / 15.5 | same [A] |
| same, WHAMR! | 16.1 / 10.3 | same [A] |
| USEF-TFGridNet / SepFormer / SpEx+, Libri2Mix (max, 16 kHz) | 16.8 / 11.4 / 9.0 | same [A] |
| WeSep Libri2Mix train-360, SI-SDR: BSRNN 16.57, TF-GridNet 15.79, SpEx+ 14.57, DPCCN 13.80 (train-100: 13.32 / 12.09 / 12.64 / 11.45) | SI-SDR, not SDRi | [WeSep 2024](https://arxiv.org/html/2409.15799) [A] |

- USEF-TSE paper gives no speaker-confusion rate; it attributes weaker Libri2Mix/WHAMR! results qualitatively to speaker confusion — [USEF-TSE](https://arxiv.org/html/2409.02615) [A]
- Libri2Mix noisy mix (Mix_both) input -1.94 dB SI-SDR; BSRNN-based TSE 6.11 (pretrained ECAPA encoder) vs 8.20 (jointly trained encoder); 10.81 -> 11.04 with enrollment augmentation (train-460) — [Enrollment aug 2024](https://arxiv.org/html/2409.09589v1) [A]
- Not retrieved: VoiceFilter, SpeakerBeam, pBSRNN own numbers (pBSRNN = BSRNN + enrollment branch, ResNet34 embedding, Interspeech 2023) — [search summary](https://www.isca-archive.org/interspeech_2023/yu23b_interspeech.pdf) [A]

**Inference:** Libri2Mix/WHAM numbers are the relevant proxy for YouTube audio; 9-17 dB means audible residual interference and artifacts. SI-SDR gain does not imply attribution gain (cf. N-0022 Q1).

## 2. Real recordings, confusion, overlap

**Takeaway:** Synthetic-set performance is reported not to transfer to real conversations; confusion is a known failure; no study quantified degradation on single-speaker or sparse-overlap segments.

- REAL-T benchmark and SLT 2026 REAL-TSE challenge (Mandarin and English conversational recordings; metrics TER, speaker similarity, DNSMOS, target-speaker activity F1) exist because LibriMix-trained TSE degrades on natural overlap, reverberation, noise, channel mismatch — [REAL-TSE](https://arxiv.org/pdf/2607.15198) [A]
- Target confusion (extracting the interferer) is worst when target and interferer voices are similar; remedies: speaker augmentation, metric learning — [Target confusion 2022](https://arxiv.org/abs/2204.01355v1) [A]
- Claim from one preprint: intermediate overlap is hardest, low overlap leaves clean target speech for reference, high overlap resembles separation; stated in Sec. 2.2 without per-overlap test numbers (all test mixes there have overlap 0.0) — [Liu et al. 2026](https://arxiv.org/html/2603.04943) [V, single-source, unsupported by numbers]
- Same paper, BLSTM TSE (frozen ECAPA, Libri2Vox) iSDR 2/3/4 speakers: 12.38/8.56/7.16 random -> 13.22/10.08/9.21 curriculum — [Liu et al. 2026](https://arxiv.org/html/2603.04943) [V]
- Continuous TSE work treats target-speaker absence as a core problem; plain TSE outputs speech-like artifacts when the target is silent (plain TSE inferior on varying overlap and absence) — [search summary](https://arxiv.org/pdf/2405.06289.pdf) [A]
- Downstream: STCON (CHiME-8) reports its gain came mostly from diarization/speaker counting, not target-speaker extraction (via N-0022 Q4) [A]

**Inference:** For a 2-speaker interview with <5% overlap, extraction has little to remove and adds artifact and wrong-speaker risk; expected gain is near zero (unmeasured).

## 3. TS-VAD family

| System | Data | DER | Source |
|---|---|---|---|
| x-vector+AHC baseline -> TS-VAD fusion | CHiME-6 dev / eval | 63.42 -> 32.84 / 68.20 -> 36.02 | [TS-VAD 2020](https://ar5iv.labs.arxiv.org/html/2005.07272) [A] |
| AHC init -> Seq2Seq-TSVAD (ICASSP 2023) | VoxConverse test (collar 250 ms) | 5.35 -> 4.55 | [Seq2Seq-TSVAD](https://ar5iv.labs.arxiv.org/html/2210.16127) [V] |
| LSTM-SC init -> Seq2Seq-TSVAD, oracle VAD | DIHARD-III Track 1 eval | 15.40 -> 10.77 | same [A] |

- Original TS-VAD: fixed 4 speakers, multi-channel CHiME-6 setting, i-vectors from first-pass clustering re-estimated with TS-VAD posteriors; no gain until the first-pass x-vector front end improved; x-vectors in place of i-vectors worked much worse; 3rd re-estimation gave no gain — [TS-VAD](https://ar5iv.labs.arxiv.org/html/2005.07272) [A]
- Seq2Seq-TSVAD: ResNet-34 ArcFace embeddings (VoxCeleb2) taken from the first-pass output; speakers with <2 s non-overlapped speech are discarded; embedding augmentation matters (VoxConverse best 10.10 without it vs 4.55); handles 30 speakers at 10 ms on one 12 GB GPU; no DER by overlap ratio; gains larger on recordings with many speakers — [Seq2Seq-TSVAD](https://ar5iv.labs.arxiv.org/html/2210.16127) [A]
- VoxConverse gain is small (0.8 absolute) because the AHC first pass was already 5.35 — [Seq2Seq-TSVAD](https://ar5iv.labs.arxiv.org/html/2210.16127) [A]
- Flow-TSVAD (arXiv 2409.04859) claims better than Seq2Seq-TSVAD; numbers not read — [search summary](https://arxiv.org/abs/2409.04859v1) [A]
- NSD-MA-MSE, AliMeeting/AMI TS-VAD numbers, DiariZen vs TS-VAD comparison: not retrieved. DiariZen already in N-0015 and R-0006 (repo listed in sources).

**Inference:** TS-VAD is a refiner of an initial diarization; large DER drops occur when the first pass is weak (CHiME-6 63 -> 33) and shrink when it is strong (VoxConverse). Gain on 2-speaker low-overlap audio with a strong pyannote first pass is expected to be small; no direct evidence.

## 4. Enrollment

- TSE: one clean pre-enrolled utterance; WSJ0-2mix eval references average 7.3 s; USEF-TSE accepts references of unequal length; 15 s used for DNS blind test; typical 5-10 s — [USEF-TSE](https://arxiv.org/html/2409.02615) [A]; [pBSRNN summary](https://www.isca-archive.org/interspeech_2023/yu23b_interspeech.pdf) [A]
- Sensitivity: enrollment differing in speaking rate, emotion, content, or acoustics hurts; wrong-speaker output when enrollment resembles the interferer — [2505.12288 summary](https://arxiv.org/html/2505.12288v1) [A]
- Self-estimated enrollment (model's own output reused as enrollment) gives up to +2.5 dB SI-SDR on Libri2Mix; noisy enrollment (SNR -5..15 dB) augmentation helps mainly out-of-domain — [Enrollment aug](https://arxiv.org/html/2409.09589v1) [A]
- Self-enrollment from first-pass diarization: Seq2Seq-TSVAD uses non-overlapped first-pass speech (>=2 s per speaker); original TS-VAD iterates i-vectors with its own posteriors — sources above [A]
- No study of enrollment-length curves (e.g. 3 s vs 30 s) or of first-pass-error sensitivity found.

## 5. Off-the-shelf (as of 2026-11)

| Toolkit | Status | Source |
|---|---|---|
| WeSep (wenet-e2e) | Apache-2.0; recipes BSRNN, SpEx+, DPCCN, TF-GridNet (LibriMix), also multichannel/visual/text cues; v0.1 lists official pretrained models as "coming soon"; checkpoints only on `wesep_2024` branch (audio-only); speaker encoders from WeSpeaker (ECAPA, VoxCeleb2); trained on English read speech | [repo](https://github.com/wenet-e2e/wesep) [A] |
| ClearerVoice-Studio | Apache-2.0; audio-only TSE training recipe at 8 kHz; AV/EEG at 16 kHz; inference checkpoint for audio-only TSE not confirmed | [repo](https://github.com/modelscope/ClearerVoice-Studio) [A] |
| Seq2Seq-TSVAD / TS-VAD | Paper code and weights not located in this session | gap |

**Gaps:** SpeechBrain/ESPnet/NeMo TSE or TS-VAD availability (NeMo MSDD is a TS-VAD-like model, unverified); Korean evidence (none); GPU VRAM of TSE models (USEF 15-20M params, inference small, inference); DER/WER effect of TSE on real recordings; CHiME-7/8, AliMeeting, AMI TS-VAD numbers; VoiceFilter/SpeakerBeam numbers. Fetches were summarized by a model, not read verbatim; verify before citing tables.

