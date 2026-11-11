---
id: N-0015
type: note
task: T-0004
title: "Diarization models and joint ASR+diarization / speaker-attributed ASR for long-form 2-person interviews (state as of 2026-11-03)"
status: final
created: 2026-11-03
tags: [speaker-diarization, asr]
---

# Diarization models and joint ASR+diarization / speaker-attributed ASR for long-form 2-person interviews (state as of 2026-11-03)

Research context: WhisperX/pyannote-style pipeline, 24-min YouTube two-person interviews, speaker changes landing one or a few words early or late, and short interjections absorbed into the other speaker. Target hardware: RTX 5060 Ti 16 GB, WSL2, Python 3.12. Budget-limited pass (about 20 tool calls); see Gaps.

## Q1. Diarization models: which have the best boundary precision and overlap handling for 2-speaker conversational audio?

### Takeaway
Among locally runnable open models, the latest published DER numbers rank them roughly as: NVIDIA Nemotron-3-Diarization (Sept 2026, 8 spk, commercial-OK license) and DiariZen-Large-s80-v2 (Dec 2025, non-commercial weights) ahead of Streaming Sortformer 4spk-v2, which is ahead of pyannote community-1 (Sept 2025, CC-BY-4.0), which is slightly ahead of pyannote 3.1. Each vendor reports on different datasets and collars, so the numbers are not directly comparable. No source reports boundary-precision metrics such as word-level boundary offset; DER with no collar is the closest proxy.

### Cited Findings
**pyannote (open source + pyannoteAI)**
- `pyannote/speaker-diarization-community-1` is licensed CC-BY-4.0 and gated on HF (you must accept the conditions and agree to occasional emails). Benchmark last updated 2025-09. — [HF model card](https://huggingface.co/pyannote/speaker-diarization-community-1)
- Community-1 DER (%) vs legacy 3.1 vs commercial Precision-2, with "fully automatic processing, no forgiveness collar, nor skipping overlapping speech". CALLHOME 28.5 / 26.7 / 16.6; AMI-IHM 18.8 / 17.0 / 12.9; AMI-SDM 22.7 / 19.9 / 15.6; DIHARD3 21.4 / 20.2 / 14.7; VoxConverse 11.2 / 11.2 / 8.5; MSDWild 25.4 / 22.8 / 17.3; AliMeeting 24.5 / 20.3 / 15.2; REPERE 7.9 / 8.9 / 7.4 (community-1 is worse than 3.1 here); RAMC 22.2 / 20.8 / 10.5. — [HF model card](https://huggingface.co/pyannote/speaker-diarization-community-1)
- Community-1 exposes `output.exclusive_speaker_diarization`. This is a non-overlapping version said to simplify "the reconciliation between fine-grained speaker diarization timestamps and transcription timestamps". It also supports `num_speakers`, `min_speakers` and `max_speakers`. — [HF model card](https://huggingface.co/pyannote/speaker-diarization-community-1)
- pyannote.audio 4.0.0 replaced agglomerative hierarchical clustering with VBx clustering, added exclusive speaker diarization, switched from torchaudio to torchcodec for audio I/O, supports fully offline use, and dropped Python < 3.10. The latest listed release is 4.0.7. — [GitHub releases](https://github.com/pyannote/pyannote-audio/releases). **Date conflict:** the fetched release page gave 4.0.0 as 2024-09-29, but community-1 (which requires 4.x) is dated 2025-09 on its model card. The 4.0.0 release was most likely 2025-09-29 and the fetched year is wrong, so verify before citing exact dates.
- pyannoteAI Precision-2 (commercial API) is claimed to be 14% more accurate than Precision-1 and 28% more accurate than OSS 3.1. It can force any number of speakers and is said to be "particularly good at pinpointing speaker interruptions and short backchannels". The STT product runs precision-2 plus Parakeet-tdt-0.6b-v3 or whisper-large-v3-turbo, followed by proprietary "STT reconciliation logic". pyannoteAI joined the Baseten distribution platform on 2026-07-29. — [pyannoteAI blog (vendor)](https://www.pyannote.ai/blog/precision-2); [pyannoteAI docs](https://docs.pyannote.ai/tutorials/speech-to-text-diarization)
- A pyannoteAI "Top 7 diarization APIs and models in 2026" post (2026-10-30, vendor-authored) uses 259 recordings across 10 DIHARD domains, no collar, overlap scored. It includes AssemblyAI, Deepgram, NeMo and Speechmatics but gives no numbers for them beyond saying they "trail" the dedicated systems. — [pyannoteAI blog (vendor bias)](https://www.pyannote.ai/blog/top-7-speaker-diarization-apis-and-models-in-2026)

**NVIDIA Sortformer family (NeMo)**
- `nvidia/diar_streaming_sortformer_4spk-v2` has 117M parameters (17-layer NEST FastConformer encoder plus 18-layer Transformer). It uses 80 ms frames, supports at most 4 speakers, takes 16 kHz mono input, handles multi-hour audio, and is licensed CC-BY-4.0. DIHARD III eval (1–4 spk), no collar: 13.24% DER at 1.04 s latency, and about 13.4–13.45% from 0.32 s to 30.4 s latency. CALLHOME-part2 2-speaker: 6.57% DER with a 0.25 s collar. Install is from NeMo main. — [HF model card](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2)
- The Streaming Sortformer paper (Interspeech 2025, arXiv 2507.18446) introduces an arrival-time-ordered speaker cache (AOSC). It reports DIHARD III ≤4 spk DER of 13.32% at 1.04 s latency and RTF 0.180 on an RTX 6000 Ada at batch size 1. On DIHARD III with 5–9 speakers, DER is 42.56%. — [ISCA paper](https://www.isca-archive.org/interspeech_2025/medennikov25_interspeech.pdf); figures as summarised by [Kili blog](https://kili-technology.com/blog/speaker-diarization-models-guide-benchmarks-and-failure-modes-2026)
- **NEW: `nvidia/Nemotron-3-Diarization`**, released 2026-09-23. It is a 100M-parameter, 31-layer Transformer with RoPE, supports up to 8 speakers, and needs NeMo 3.0. The license is OpenMDW v1.1 and commercial use is permitted. Latency modes run from 30.4 s (offline) down to 0.32 s. DER at 30.4 s: DIHARD III full (1–9 spk) 12.73%; CALLHOME-Part2 9.10% (0.25 s collar); AMI MHM 9.25%; AMI SDM 11.14%. Listed GPUs are Ampere, Ada, Hopper and Blackwell, with Linux preferred. It emits word-level speaker tags for streaming ASR via NeMo-Speech.cpp, Transformers and GGUF. — [HF model card](https://huggingface.co/nvidia/nemotron-3-diarization). This is a single source, and it is not stated whether the CALLHOME number covers 2-speaker only (the Sortformer-v2 6.57% figure was 2-speaker only, so the two are probably not comparable).
- The model card points to an ASR integration guide. — [ASR_INTEGRATION_GUIDE.md](https://huggingface.co/nvidia/Nemotron-3-Diarization/blob/main/ASR_INTEGRATION_GUIDE.md) (not fetched)

**DiariZen (BUT, Brno)**
- DiariZen is built on AudioZen and pyannote 3.1, using a WavLM front end plus pyannote-style clustering. The latest model is DiariZen-Large-s80-v2 (2025-12-09). Structured pruning recipes were added 2025-06-03 and multi-channel WavLM 2026-01-31. Code is MIT; **weights are CC BY-NC 4.0** (non-commercial) because of the training data (RAMC, DIHARD-3). The documented environment is Python 3.10, torch 2.1.1 and CUDA 12.1. It estimates the number of speakers automatically. — [GitHub](https://github.com/BUTSpeechFIT/DiariZen); [HF large-s80-md-v2](https://huggingface.co/BUT-FIT/diarizen-wavlm-large-s80-md-v2)
- DER with no collar, pyannote 3.1 vs DiariZen-Large-s80-v2: AMI-SDM 22.4 → 13.9; AISHELL-4 12.2 → 10.1; AliMeeting-far 24.4 → 10.8; NOTSOFAR-1 — → 16.7; VoxConverse 11.3 → 9.1; DIHARD3 21.7 → 14.5 (search snippet). — [GitHub](https://github.com/BUTSpeechFIT/DiariZen)

**Other**
- Picovoice (vendor of Falcon, updated 2026-03-11) reports on VoxConverse: Falcon DER 10.3% vs pyannote 9.0% (version unspecified), and JER 19.9% vs 27.4%. It claims pyannote needs 221× more compute and 1.5 GiB per stream. Collar and settings are not disclosed. — [Picovoice blog (vendor bias)](https://picovoice.ai/blog/state-of-speaker-diarization/)
- A Sept 2026 NTT paper (Horiguchi, Delcroix, Tawara, Plaquet) splits DER into a "pause-attributable" component (errors compatible with filling pauses) and a "core" component. It argues that pause-annotation ambiguity inflates DER and hides real boundary errors. Code is on GitHub. — [arXiv 2609.11007](https://arxiv.org/abs/2609.11007)

### Inferences
- For 2-speaker interviews, all the leading models are within a few DER points of each other on standard benchmarks. Errors of one or a few words at speaker changes are about 0.2–1 s of audio, smaller than the 0.25 s-collar plus frame-resolution effects most benchmarks absorb. **DER rankings are therefore a weak predictor of word-boundary quality.** Pick the model on overlap/backchannel recall and frame resolution, then fix boundaries at the word level (see Q3).
- Sortformer and Nemotron-3 are end-to-end, frame-level models (80 ms) with native overlap output. pyannote and DiariZen are segmentation-plus-clustering hybrids with powerset segmentation. Both families output overlap. The Sortformer family avoids clustering errors on short turns, which may help with "absorbed interjections" (inference, not measured).
- Licensing:
  - Commercial-safe locally: community-1 (CC-BY-4.0, gated), Sortformer v2 (CC-BY-4.0), Nemotron-3-Diarization (OpenMDW).
  - DiariZen weights: research and non-commercial use only.

### Gaps
- No source measured speaker-change boundary offset in milliseconds or words for any model, and no head-to-head on a single 2-speaker podcast/interview set covering all four model families.
- No per-model VRAM figures for a 24-minute file on 16 GB were found. Sizes are small (Sortformer 117M, Nemotron-3 100M, pyannote segmentation plus WeSpeaker ResNet, DiariZen WavLM-Large about 300M+), so fitting in 16 GB is likely but unverified. **Blackwell/RTX 50-series:** Nemotron-3 lists Blackwell. The others need a torch build with CUDA ≥ 12.8 (sm_120), and DiariZen's pinned torch 2.1.1/cu121 will not support an RTX 5060 Ti as-is (inference from the toolchain, not a cited source).
- No sources were fetched for EEND-VC / EEND-TA or other 2025–2026 EEND variants, or for pyannote segmentation hyperparameter defaults in 4.x.

## Q2. Joint / speaker-attributed ASR: options and reported gains vs cascaded pipelines

### Takeaway
The strongest open, locally runnable speaker-attributed ASR options in 2025–2026 are:
- **BUT DiCoW (v3) / SE-DiCoW:** Whisper conditioned frame-wise on diarization output, usually from DiariZen.
- **NVIDIA multitalker-parakeet-streaming-0.6b-v1:** an RNNT model conditioned on Sortformer activations through speaker kernels, English.

Both still depend on a diarizer, but transcribe per speaker and handle overlap natively. That removes the "word assigned by timestamp lookup" step that causes mid-sentence splits in WhisperX. Text-level speaker-error correctors (DiarizationLM, the SEC models) directly target word-level boundary errors with large reported WDER cuts.

### Cited Findings
- **DiCoW** conditions Whisper on frame-wise diarization probabilities. It uses FDDT (frame-level diarization-dependent transforms) and query-key biasing (QKb) instead of speaker embeddings. With ground-truth diarization it beats standard Whisper and embedding-conditioned systems on cpWER, ORC-WER and tcWER. — [arXiv 2501.00114](https://arxiv.org/pdf/2501.00114); [CSL article](https://www.sciencedirect.com/science/article/abs/pii/S088523082500066X)
- In the MLC-SLM 2025 challenge (2-speaker conversational, multilingual), the BUT DiCoW + DiariZen system ranked 2nd in Task 2 with micro-average tcpWER/CER of 16.75%. Before fine-tuning it scored 28.4% vs the baseline's 76.1% with real diarization; after fine-tuning it scored 20.8%. — [arXiv 2506.13414](https://arxiv.org/pdf/2506.13414)
- A model checkpoint is published as `BUT-FIT/DiCoW_v3_MLC`. — [HF](https://huggingface.co/BUT-FIT/DiCoW_v3_MLC) (license not fetched)
- **SE-DiCoW** (Jan 2026, self-enrolled) reduces macro-averaged tcpWER by 52.4% relative to the original DiCoW on the EMMA MT-ASR benchmark. — [arXiv 2601.19194](https://arxiv.org/html/2601.19194)
- DiCoW has also been adapted for end-to-end multi-talker ASR. — [arXiv 2510.03723](https://arxiv.org/pdf/2510.03723) (not fetched)
- **NVIDIA multitalker-parakeet-streaming-0.6b-v1** is a 600M-parameter FastConformer RNNT. It injects learnable "speaker kernels" into encoder layers, driven by speaker activity from Sortformer, and needs no enrollment. It runs in streaming (1.12 s latency) or offline mode under the NVIDIA Open Model License. cpWER: AMI-IHM 21.26%, AMI-SDM 37.44%, CH109 15.81%, Mixer6 23.81%. There is no cascaded baseline on the card. — [HF model card](https://huggingface.co/nvidia/multitalker-parakeet-streaming-0.6b-v1)
- A community GGUF bundle of Sortformer and multitalker Parakeet reports AMI-IHM cpWER of 19.35% (default kernel mode) vs 23.73% (masked mode). This is a single community source. — [HF myflow-ai GGUF](https://huggingface.co/myflow-ai/multitalker-parakeet-streaming-0.6b-v1-gguf)
- **DiarizationLM** (Google): a fine-tuned PaLM 2-S post-processes the ASR plus diarization text and cuts WDER by 55.5% relative on Fisher and 44.9% on Callhome English. Paper v1 is from 2024-01; v13 is from 2026-10-11. The abstract does not state whether code or models are released. — [arXiv 2401.03506](https://arxiv.org/abs/2401.03506)
- **Speaker Error Corrector (non-autoregressive LM)** (Krisp.ai/YSU, 2024): ALBERT plus a transformer that works on ±18-word windows around detected speaker-change points. Fisher WDER 2.8 → 2.42%; TAL 4.25 → 4.11%; cpWER gains on the SLT GenSEC Track-2 challenge. — [arXiv 2409.00151](https://arxiv.org/html/2409.00151v1)
- **WhisperX status:** the fetched upstream releases page showed v3.8.6 (May 2024) as latest, with no pyannote 4.x / community-1 changes visible. That result may be stale or incomplete. — [GitHub releases](https://github.com/m-bain/whisperX/releases)
- **murmurai-core** (Dec 2025) is a maintained WhisperX fork with torch 2.6+ and pyannote 4.x support, tested on Python 3.10–3.13. — [PyPI](https://pypi.org/project/murmurai-core/)
- pyannoteAI's commercial STT pipeline (diarization plus Parakeet/Whisper plus "reconciliation logic") claims "up to 52% lower transcription error rate" than alternatives. This is a vendor claim with unspecified metric details. — [pyannoteAI](https://www.pyannote.ai/blog/precision-2)

### Inferences
- The mid-sentence split ("…fire to" | "you?") is a classic cascade artifact: word timestamps from forced alignment are matched against diarization segments that are offset by a few hundred ms. Two families of fixes exist:
  - **Model level:** DiCoW or multitalker Parakeet, which assign words per speaker inside the ASR model.
  - **Post-hoc:** DiarizationLM or SEC-style text-aware correction, or simply snapping speaker changes to punctuation/sentence boundaries.
  
  For a 2-speaker interview, the post-hoc route is the cheapest and directly targets WDER.
- multitalker-parakeet is English-only by implication and RNNT-based, so you would swap Whisper out entirely. DiCoW keeps Whisper but is tied to the BUT toolchain.

### Gaps
- No t-SOT / SOT (Microsoft) 2025–2026 open checkpoints were researched (budget).
- Not researched: SA-Whisper, WhisperSpeaker, "Whisper-Diarization" (MahmoudAshraf97), WEEND, or LLM-audio models (Gemini, GPT-4o audio) as diarizers.
- No sourced 2025–2026 numbers were found for the commercial APIs AssemblyAI, Deepgram and ElevenLabs Scribe. The pyannoteAI comparison lists some of them but gives no numbers.
- No source gives a direct cascaded-vs-joint WDER comparison on 2-person interview audio.

## Q3. Practical settings that reduce boundary error

### Takeaway
Of the requested settings, only a few are documented in the sources found: forcing 2 speakers (`num_speakers=2` or min/max), using community-1's `exclusive_speaker_diarization` for word assignment, and choosing models that are trained on and scored with overlap included. Specific onset/offset/min_duration_off values were not found in primary sources this pass.

### Cited Findings
- Community-1 supports `num_speakers`, `min_speakers` and `max_speakers`, plus `exclusive_speaker_diarization`, which is explicitly meant for aligning with transcription timestamps. — [HF model card](https://huggingface.co/pyannote/speaker-diarization-community-1)
- pyannote 4.x uses VBx clustering instead of AHC. — [GitHub releases](https://github.com/pyannote/pyannote-audio/releases)
- Sortformer and Nemotron-3 report better DER "with post-processing". The model cards list latency/chunk configs (0.32 s to 30.4 s), and offline 30.4 s gives the best accuracy. — [Sortformer v2 card](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2); [Nemotron-3 card](https://huggingface.co/nvidia/nemotron-3-diarization)
- Benchmarks differ in collar: pyannote and DiariZen report with no collar, while NVIDIA's CALLHOME number uses a 0.25 s collar. Comparisons across vendors must account for this. — sources above
- Pause filling (closing short gaps) changes DER in a measurable, threshold-dependent way, and errors "compatible with pause filling" can be separated from core errors. — [arXiv 2609.11007](https://arxiv.org/abs/2609.11007)

### Inferences
- These are reasoned recommendations, not sourced:
  - Set `num_speakers=2`.
  - Assign words with the exclusive output, giving each word to the speaker with maximum overlap over the word's span rather than its start time.
  - Then apply a punctuation-aware boundary snap: move a speaker change to the nearest sentence-final punctuation within ±N words if the diarization confidence is low.
  - Keep short opposite-speaker segments (do not over-smooth with a large min_duration_off), so interjections survive.

### Gaps
- No primary sources were found for pyannote 4.x default hyperparameters, or for tuned onset/offset/min_duration_off/clustering-threshold values for 2-speaker interviews.

## Q4. VRAM / speed for a 24-min file on 16 GB; licensing summary

### Takeaway
All candidate diarizers are small (100M–~300M+ parameters) and offline-capable. The main risk on an RTX 5060 Ti (Blackwell, sm_120) is toolchain support rather than VRAM. Licenses vary from CC-BY-4.0 (gated) to non-commercial.

### Cited Findings
- Streaming Sortformer: RTF 0.18 on an RTX 6000 Ada, so a 24-min file takes roughly 4.3 min at that streaming setting (the time is computed from the cited RTF). — [ISCA paper](https://www.isca-archive.org/interspeech_2025/medennikov25_interspeech.pdf) via [Kili](https://kili-technology.com/blog/speaker-diarization-models-guide-benchmarks-and-failure-modes-2026)
- Nemotron-3-Diarization lists Blackwell support, Linux preferred. — [HF](https://huggingface.co/nvidia/nemotron-3-diarization)
- pyannote is reported at 1.5 GiB memory per stream by a vendor source (version unspecified). — [Picovoice (vendor)](https://picovoice.ai/blog/state-of-speaker-diarization/)
- DiariZen is pinned to Python 3.10 and torch 2.1.1+cu121. — [GitHub](https://github.com/BUTSpeechFIT/DiariZen)

**License summary**

| Model | License | Commercial use? | Source |
|---|---|---|---|
| community-1 | CC-BY-4.0, gated | Yes | [HF](https://huggingface.co/pyannote/speaker-diarization-community-1) |
| Sortformer 4spk-v2 | CC-BY-4.0 | Yes | [HF](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2) |
| Nemotron-3-Diarization | OpenMDW v1.1 | Yes | [HF](https://huggingface.co/nvidia/nemotron-3-diarization) |
| multitalker-parakeet | NVIDIA Open Model License | Not checked | [HF](https://huggingface.co/nvidia/multitalker-parakeet-streaming-0.6b-v1) |
| DiariZen | Code MIT, weights CC BY-NC 4.0 | No (weights) | [GitHub](https://github.com/BUTSpeechFIT/DiariZen) |

### Inferences
- Running WhisperX large-v3 (about 5–10 GB with faster-whisper) plus any of these diarizers fits in 16 GB, especially if run sequentially. This is not measured.
- DiariZen would need repinning to torch ≥ 2.7 with cu128 for Blackwell, and to Python 3.12. Feasibility is unverified.

### Gaps
- No measured VRAM or wall-clock numbers for any of these models on a 24-min file on RTX 50-series / 16 GB GPUs.
- License texts of DiCoW checkpoints were not checked.
