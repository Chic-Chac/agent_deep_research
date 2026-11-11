---
id: T-0005
type: task
title: "Signal-level processing to raise speaker attribution accuracy"
status: done
requester: user
received: 2026-11-11T17:09+09:00
finished: 2026-11-11T17:31+09:00
deliverable: R-0007
tags: [speaker-diarization, audio-preprocessing, speaker-extraction]
---

# Signal-level processing to raise speaker attribution accuracy

## Request

Survey whether active processing of speech audio (transform and decomposition such as wavelet/spectral/filterbank, segmentation, masking) raises accuracy of speaker extraction (diarization, target-speaker extraction, speaker embedding). Requester asked in Korean; original wording: hwaja chuchul (speaker extraction), pajang byeonhwan mit bunhae (wave transform and decomposition), gugan bunhal (segmentation), masking.

## Scope

- Tier: standard.
- Assumption (not asked): "speaker extraction" covers diarization (who spoke when), target-speaker extraction, and speaker embedding/identification. Context assumed from T-0004: long-form YouTube interviews, WhisperX/pyannote-style stack, 16 GB GPU; findings not limited to it.
- Sub-questions: N-0020 transform/decomposition front-ends; N-0021 segmentation; N-0022 masking, separation, enhancement, target-speaker extraction as preprocessing; N-0023 training-side processing (augmentation, masking, simulated mixtures).
- Reuses: N-0015 (diarization models, settings). Model choice is not re-surveyed.
- Deliverable: default report.

## Outcome

- Segmentation is the only signal-side lever with large measured gains, and only when VAD is poor: about 12 DER points weak-to-good VAD, 0.3-1.2 from overlap detection, multi-scale embeddings, tuning.
- Learnable and raw front-ends measured at parity with log-mel; wavelet, CQT and scattering are untested on modern embedding models; all need retraining. WavLM features help only through a prebuilt model.
- Generic denoising before embeddings raised SV EER 1.3-4.9x; separation helps cpWER on overlap-heavy meetings, not DER; TSE has no measured DER/WER effect on real recordings.
- In-domain fine-tuning cut 2-speaker EEND DER 15.09 to 8.44; pyannote-specific fine-tuning gains rest on two weak sources. Pitch-changing speed perturbation helps, SpecAugment does not. Demucs stems cut WER, not DER, in the one entry found.
- Open: no measurement on Korean or low-overlap 2-speaker interviews; N-0024 evidence mostly [A]; DER-to-WDER transfer unmeasured.
