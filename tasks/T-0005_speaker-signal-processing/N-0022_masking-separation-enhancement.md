---
id: N-0022
type: note
task: T-0005
title: "Do masking, source separation, enhancement and target-speaker extraction as preprocessing raise diarization and speaker identification accuracy?"
status: final
created: 2026-11-11
tags: [speaker-diarization, speech-separation, speaker-extraction]
uses: [N-0015]
---

# Do masking, source separation, enhancement and target-speaker extraction as preprocessing raise diarization and speaker identification accuracy?

Budget 15 calls used. Prior diarization models: N-0015 (not repeated). Multi-channel mask beamforming (GSS) out of scope.

## Q1. Enhancement / denoising front-end for speaker verification and diarization

**Takeaway:** Off-the-shelf enhancement trained for signal quality often raises speaker-verification EER; enhancement trained with a speaker objective or blended with the noisy signal gives small gains at best. Diarization front-end gains were reported on mismatched, noisy data (DIHARD I), without a clean ablation.

- LC4SV (2023), SE = DEMUCS denoiser, EER %, noisy input vs enhanced by SE trained only for enhancement (SE-PTN): Libri2Mix/RawNet3 9.15 -> 19.63; VOiCES/RawNet3 2.81 -> 13.79; Libri2Mix/ECAPA 7.96 -> 15.89; VOiCES/ECAPA 2.70 -> 10.57; VoxCeleb1 2.18 -> 2.73 — [LC4SV](https://arxiv.org/html/2311.16604v1) [V]
- Same paper: SE fine-tuned with an SV loss (SE-SV) gives 9.27 / 2.67 / 7.68 / 2.62 on the four sets (mixed: worse on RawNet3/Libri2Mix, slightly better elsewhere) — [LC4SV](https://arxiv.org/html/2311.16604v1) [V]
- LC4SV (compensation: blends enhanced and noisy signal, SV-model-agnostic) gives 8.76 / 2.65 / 7.62 / 2.62; gain vs noisy is 0-4% relative, and 0 on unseen VoxCeleb1 model (1.018 = 1.018) — [LC4SV](https://arxiv.org/html/2311.16604v1) [V]
- On clean LibriSpeech with ECAPA, all processed variants are slightly worse than clean input (1.601 clean vs 1.604-1.621) — [LC4SV](https://arxiv.org/html/2311.16604v1) [V]
- SE metrics (STOI, PESQ, SI-SDR) of DEMUCS, DCCRN, DTLN are close on LibriMix, yet the SV effect is large: SI-SDR does not predict SV gain (inference) — [LC4SV](https://arxiv.org/html/2311.16604v1) [V]
- Cited findings: enhancement artifacts can degrade SV (VoiceID Loss, 2019); enhancing helped x-vector extractor training but hurt PLDA back-end, and combining original and enhanced signals gave ~40% relative gain (Kataria et al.); a 2026 paper reports enhanced input distorts embeddings and fuses noisy and enhanced features — [search summary](https://arxiv.org/pdf/1904.03601) [A]
- Diarization: LSTM denoiser with progressive multitask learning in the DIHARD I system (Interspeech 2018) reached DER 24.56% (Track 1 eval) and 36.05% (Track 2); the paper claims significant gains from enhancement, but the with/without-denoising DER was not on the fetched page — [Sun et al. 2018](https://www.isca-archive.org/interspeech_2018/sun18b_interspeech.html) [A]
- A related LSTM preprocessor reports consistent DER reductions on AMI, ADOS, SeedLings; the authors of the DIHARD18 recipe say the enhancement benefit is at least visible for VAD — [search summary](https://github.com/staplesinLA/denoising_DIHARD18) [A]

**Inference:** For modern embedding models (ECAPA, WavLM-based) trained on noisy augmented data, a generic denoiser is a risk not a free gain. Safe options: speaker-loss-trained SE, or blend noisy+enhanced, or use enhancement only for VAD/ASR and compute embeddings on original audio.

**Gaps:** No measured DER effect of DeepFilterNet, MetricGAN or WPE as diarization front-end found. No DIHARD III or pyannote 3.x/community-1 enhancement ablation found. All Q1 evidence is SV, not diarization.

## Q2. Music / SFX removal (Demucs, MDX, UVR) before diarization or ASR

**Takeaway:** Vocal-stem extraction is common in community pipelines and helps ASR on music-backed audio in several non-controlled reports. No controlled study isolating its effect on DER was found.

- whisper-diarization (community pipeline) extracts vocals first "to increase the speaker embedding accuracy"; `--no-stem` disables it; the README gives no measurement and does not name Demucs in the pipeline description — [repo](https://github.com/MahmoudAshraf97/whisper-diarization) [V]
- Bengali long-form competition entry, ASR (Table II, WER, private split, VAD threshold 0.3): without Demucs 0.38920, with Demucs 0.36491 — [arXiv 2602.21183](https://arxiv.org/html/2602.21183) [V]
- Same paper, diarization (Table III, fine-tuned pyannote segmentation, 18 epochs, batch 8), DER public / private: with Demucs 0.20406 / 0.28816, without 0.20292 / 0.28857. No DER effect — [arXiv 2602.21183](https://arxiv.org/html/2602.21183) [V]
- Correction 2026-11-11: an earlier version of this note reported the Table II WER figures as DER.
- ShobdoSetu (2026): HTDemucs vocal extraction WER 34.8 vs 35.9 (WhisperX with normalization) and 41.8 (manual chunking) — [arXiv 2603.19256](https://arxiv.org/pdf/2603.19256) [A]
- Lyrics transcription write-up (non-peer-reviewed): mean WER 24.3% (mix) -> 22.1% (vocals); one hallucination-prone song 92% -> 19% — [lyrics-bench](https://lyrics-bench.onrender.com/papers/lyrics_transcription_paper.pdf) [A]
- Counter-evidence: an arXiv Whisper study cites prior work that separated vocals can degrade long-form Whisper transcription (secondhand) — [arXiv 2506.15514](https://arxiv.org/pdf/2506.15514) [A]

**Inference:** Gain is largest when music or SFX overlaps speech; on clean talk with brief jingles the same artifacts risk (Q1) applies. Cheap mitigation: apply separation only to music-flagged segments, or run diarization on original audio and ASR on the stem. Untested.

**Gaps:** No Korean/English YouTube evidence. Demucs/MDX/UVR licenses and GPU cost not verified.

## Q3. Separation as diarization front-end (SSGD, CSS, PixIT, pyannote separation)

**Takeaway:** Separation helps ASR (cpWER/cpCER) markedly on overlap-heavy meetings; diarization (DER) gains are small or none. Evidence is on meetings with 20-33% overlap; interviews with low overlap would gain less (inference).

- PixIT (ToTaToNet: DPRNN + WavLM-large, 5 s windows, max 3 speakers, trained on AMI/AliMeeting train only), AMI-SDM cpWER baseline (pyannote 3.1 + ASR on mixture) -> PixIT separation: Whisper large-v2 38.3 -> 31.7 (-17.2%); medium.en 38.8 -> 32.8; NeMo conformer 48.9 -> 39.4 — [PixIT arXiv 2403.02288](https://arxiv.org/html/2403.02288v2) [V]
- PixIT AliMeeting ch.1 cpCER: Whisper large-v2 65.1 -> 48.3 (-25.8%) — [PixIT](https://arxiv.org/html/2403.02288v2) [V]
- PixIT diarization only: cpWER gain 0.8-6.7%; DER AMI-SDM 17.1 (lambda=1) vs SOTA 18.9; AliMeeting 19.5 vs pyannote 3.1 23.3; cpWER-tuned setting worsens DER to 25.3 / 28.3 — [PixIT](https://arxiv.org/html/2403.02288v2) [V]
- uWER (no speaker attribution) AMI: Whisper large-v2 35.4 -> 33.8 (-4.5%); NeMo 49.3 -> 41.6; for NeMo separated audio raises substitutions 10.7 -> 12.6 and insertions 1.8 -> 2.6 (artifacts) — [PixIT](https://arxiv.org/html/2403.02288v2) [V]
- Authors' stated limits: needs diarization labels in training; two meeting corpora only; cpWER does not penalize redundant hypothesis speakers — [PixIT](https://arxiv.org/html/2403.02288v2) [V]
- pyannote `speech-separation-ami-1.0`: MIT, gated HF, mono 16 kHz, needs pyannote.audio 3.3.2 (`[separation]` extra), trained on AMI SDM only, runs on CPU or GPU; page recommends pyannoteAI for production; no DER given on page — [HF card](https://huggingface.co/pyannote/speech-separation-ami-1.0) [V]
- SSGD (separate, then VAD per stream) on telephone speech: 8.8% DER CALLHOME (best), 12.7% in the earlier low-latency version — [Morrone et al.](https://arxiv.org/pdf/2204.02306) [A]
- LibriCSS: TF-GridNet CSS + d-vector diarization gives claimed state-of-the-art ORC-WER and cpWER; numbers not retrieved — [von Neumann et al. 2023](https://arxiv.org/abs/2309.16482) [A]

**Inference:** The speech-separation-ami weights are AMI-domain (English meeting); transfer to Korean YouTube is unverified.

**Gaps:** LibriCSS/CHiME DER and SI-SDR tables, SepFormer/Mossformer2 diarization results, GPU cost of PixIT not retrieved (PDF text not extractable).

## Q4. Target-speaker methods (TSE, TS-VAD)

**Takeaway:** TSE (VoiceFilter, SpeakerBeam) requires an enrollment utterance; TS-VAD needs initial per-speaker embeddings and is a refinement of a first-pass diarization. No verified numbers were obtained.

- SpeakerBeam and VoiceFilter need an enrollment utterance; TS-SEP derives embeddings from a first-pass diarization instead — [search summary](https://arxiv.org/pdf/2407.16447) [A]
- CHiME-8 NeMo baseline uses TS-VAD; the STCON system reports its gain came mostly from diarization/speaker counting rather than guided target-speaker extraction — [STCON](https://arxiv.org/html/2410.13411v1) [A]

**Gaps:** TS-VAD CHiME-6 DER, Seq2Seq-TS-VAD, NSD-MA-MSE, SpEx+ numbers and weights/licenses not verified (budget). Original TS-VAD: [arXiv 2005.07272](https://arxiv.org/pdf/2005.07272), not read.

## Q5. Practicality (single channel, open weights, 16 GB)

- Verified: pyannote speech-separation-ami (MIT, gated, 16 kHz mono); whisper-diarization repo (BSD-2-Clause). DEMUCS denoiser, DCCRN, DTLN used as SE in LC4SV [V].
- Not verified: licenses/VRAM of HTDemucs, DeepFilterNet, MDX/UVR, SepFormer, Mossformer2, TF-GridNet checkpoints.
