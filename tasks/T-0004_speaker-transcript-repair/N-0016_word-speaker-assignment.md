---
id: N-0016
type: note
task: T-0004
title: "Word-to-speaker assignment, speaker-aware sentence segmentation, and use of metadata in cascaded ASR + diarization pipelines"
status: final
created: 2026-11-03
tags: [speaker-diarization, word-alignment, sentence-segmentation]
---

# Word-to-speaker assignment, speaker-aware sentence segmentation, and use of metadata in cascaded ASR + diarization pipelines

Requester's symptoms: 14/287 spans end without terminal punctuation, all at speaker boundaries, with halves attributed to different speakers ("How" [guest] | "would you describe" [interviewer] | "it?" [guest]); 14/33 questions attributed to the guest. Stack: WhisperX/pyannote-style, word ids + timestamps, spans with boundary type "sentence" or "speaker".

Research date: 2026-11-03. All page content was read through a summarizing fetch tool, so verbatim code is reliable only where marked as quoted code; numbers come from the cited page.

## 1. How WhisperX `assign_word_speakers` works, how it fails, and the alternatives (sentence majority vote, punctuation-aware realignment, snapping)

### Takeaway
WhisperX gives each word, and separately each segment, the speaker with the largest summed time overlap. If nothing overlaps, `fill_nearest` falls back to the nearest diarization segment by midpoint. Nothing in this step looks at punctuation or sentence structure, so jittery word timestamps or a slightly offset diarization boundary split sentences into A|B|A fragments. This matches the requester's symptom. The best-known fix is MahmoudAshraf97/whisper-diarization's `get_realigned_ws_mapping_with_punctuation`. It finds a speaker change at a word without terminal punctuation, expands to the enclosing sentence (up to 50 words), and relabels the whole sentence with the majority speaker. It is cheap, deterministic and MIT-style open code, and it would fix the "How | would you describe | it?" case.

### Cited Findings
- **Current WhisperX algorithm.** Diarization intervals that overlap a word or segment are found with an interval tree. Each overlap is `min(dia_end, seg_end) - max(dia_start, seg_start)`. Overlaps are summed per speaker and the largest total wins. Words and segments use the same logic. With `fill_nearest=True` and no overlap, the nearest diarization segment by midpoint distance supplies the speaker. Speaker embeddings can be attached as `speaker_embeddings`. The default diarization model is `pyannote/speaker-diarization-community-1`, with `num_speakers`, `min_speakers` and `max_speakers` passed to it. — [whisperX diarize.py (main)](https://raw.githubusercontent.com/m-bain/whisperX/main/whisperx/diarize.py)
- **PR #1338 (merged 2026-02-08).** Replaced the O(n×m) loop with a sorted-array interval tree (`np.searchsorted`). It was about 228x faster on a 7-min video (73.9 s to 0.32 s), and maintainers confirmed the assignments are identical to before. So no accuracy change: the same max-overlap logic is still in place. — [whisperX PR #1338](https://github.com/m-bain/whisperX/pull/1338)
- **PR #590 (opened 2023-11-20, closed unmerged 2026-02-14, marked "Fixed in #1338").** It reported that the old pandas implementation summed *negative* intersections for non-overlapping segments (e.g. `sum([-1,-3]) = -4`). That picked the wrong "closest" speaker in fill-nearest situations. — [whisperX PR #590](https://github.com/m-bain/whisperX/pull/590)
- **WhisperX limitation (README).** "Transcript words which do not contain characters in the alignment models dictionary e.g. "2014." or "£13.60" cannot be aligned". Those words get no timing, so they cannot be overlap-assigned. The README also says overlapping speech is handled poorly and diarization is "far from perfect". License: BSD-2-Clause. — [whisperX README](https://github.com/m-bain/whisperX)
- **Segment-level failure mode.** Each WhisperX transcript segment gets a single speaker (the one with the most overlap), so the minority speaker's words in a segment that spans a turn are mislabeled. — [itsranveer/speakermute issue #1 (third-party repo, non-peer-reviewed)](https://github.com/itsranveer/speakermute/issues/1)
- **Other reported problems.** Users report text attributed to the wrong speaker even with the exact speaker count given, and say `min_speakers`/`max_speakers` sometimes *worsen* results by collapsing most speech onto one speaker. — [whisperX issue #1208](https://github.com/m-bain/whisperX/issues/1208) (user reports; single-source anecdote)
- **Practitioner explanation (vendor blog, non-peer-reviewed).** Whisper emits long segments that span two speakers. After alignment, splits tend to land "at the nearest clause boundary, not the actual speaker boundary", leaving "one or two words attributed to the wrong speaker at every turn". — [AssemblyAI blog](https://assemblyai.com/blog/whisper-speaker-diarization) (vendor with a commercial interest; treat as anecdotal)
- **whisper-diarization word mapping.** `get_words_speaker_mapping` does not use max overlap. It uses an *anchor* point per word (`word_anchor_option="start"` by default; mid or end are options) and walks forward through speaker turns until the anchor is ≤ the turn end. The last turn's end is stretched to cover trailing words, so every word gets a speaker. — [whisper-diarization helpers.py](https://raw.githubusercontent.com/MahmoudAshraf97/whisper-diarization/main/helpers.py)
- **whisper-diarization punctuation realignment.** In `get_realigned_ws_mapping_with_punctuation(word_speaker_mapping, max_words_in_sentence=50)`, with `sentence_ending_punctuations = ".?!"`:
  - For each k where `speaker[k] != speaker[k+1]` and word k does not end in `.?!`, it finds the sentence start by walking left while the speaker is unchanged and no sentence end is seen. It finds the sentence end by walking right up to the next `.?!`, within a 50-word window.
  - It then takes the mode speaker over [left, right]. If the mode count is ≥ len//2, the whole sentence is relabeled to the mode speaker. If no sentence boundary is found within the window, the change is left alone.
  - Afterwards, `get_sentences_speaker_mapping` builds sentences, starting a new one on a speaker change or when NLTK Punkt `text_contains_sentbreak` fires. — [whisper-diarization helpers.py](https://raw.githubusercontent.com/MahmoudAshraf97/whisper-diarization/main/helpers.py)
- **Wider pipeline.** whisper-diarization applies punctuation restoration (deepmultilingualpunctuation) before realignment, so Whisper's missing punctuation does not stop the sentence-span search. — [whisper-diarization helpers.py](https://raw.githubusercontent.com/MahmoudAshraf97/whisper-diarization/main/helpers.py). The punctuation-model detail is from the repo's known pipeline and was not re-verified on this fetch; see Gaps.
- **Max overlap at word level is the standard rule.** When a turn boundary falls inside a word, the word goes to the speaker with the largest overlap. The Lexical Speaker Error Correction (SEC) paper describes this as the standard baseline that a second-pass LM corrects. — [Paturi et al., Interspeech 2023 (Lexical SEC)](https://www.isca-archive.org/interspeech_2023/paturi23_interspeech.pdf)
- **pyannote `exclusive_speaker_diarization`.** community-1 has an output, `output.exclusive_speaker_diarization`, in which only one speaker is active at a time. pyannote says it "simplifies the reconciliation between fine-grained speaker diarization timestamps and (sometimes not so precise) transcription timestamps". It was backported from their commercial model. License CC-BY-4.0. DER vs 3.1: AMI-IHM 18.8→17.0, AliMeeting 24.5→20.3, AISHELL-4 12.2→11.7 (benchmark dated 2025-09). Supports `num_speakers=2`. — [pyannote/speaker-diarization-community-1 model card](https://huggingface.co/pyannote/speaker-diarization-community-1)

### Inferences
- **The requester's A|B|A pattern** ("How"[G] | "would you describe"[I] | "it?"[G]) is the exact case the whisper-diarization realignment targets. Trace: speaker change after "How", which has no terminal punctuation. Left boundary = "How", because the previous word ends a sentence. Right boundary = "it?". The labels are [G,I,I,I,G], so the mode is I with 3 ≥ 5//2 = 2, and the whole question goes to the interviewer. Porting the function (about 60 lines) onto word ids is low-risk.
  - **Caveat:** with mode count ≥ len//2, a 2-vs-2 tie in a 4-word sentence still relabels, using `max(set(...))` order, which is effectively arbitrary.
  - **Caveat:** the left walk stops at the first speaker change, so an A|B|A|B sentence may be skipped (left_idx returns −1).
  - **Suggested improvements:** weight votes by word duration or overlap seconds rather than word count, and break ties using the diarization overlap margin.
- **Long legitimate turns.** A pure majority vote can wrongly merge a real mid-sentence interruption ("So you're saying— Yes, exactly."). Only relabel when the minority side is short (e.g. ≤3 words or ≤1 s) or when its overlap margin is small. Otherwise keep the split and mark the span as a true "speaker" boundary.
- **Diarization source.** Feeding WhisperX's assigner `exclusive_speaker_diarization` instead of the regular (overlapping) output should cut the cases where overlapping or padded turn edges steal the first or last word of a sentence. That is what pyannote says it was built for, but I found no published WDER number for it (see Gaps).
- **Untimed words.** Before assignment, give numerals and other unaligned words interpolated timestamps from their neighbours, or inherit the neighbours' speaker. Otherwise `fill_nearest` may give them to a different speaker and split a sentence.
- **14/33 questions given to the guest.** Short interviewer turns are probably being absorbed into the guest cluster, or misassigned at turn edges. Sentence-level realignment fixes edge splits but not whole-sentence cluster errors. Those need the metadata/role step in Section 4.

### Gaps
- No published, quantitative comparison of max-overlap vs anchor-point vs sentence-majority-vote word assignment (WDER) on a common benchmark.
- No WDER numbers for pyannote `exclusive_speaker_diarization` vs regular output.
- No upstream WhisperX PR adding punctuation- or sentence-aware assignment was found; the current main branch still uses pure max overlap.
- The punctuation model used by whisper-diarization (deepmultilingualpunctuation) was not re-verified in this session.
- No dedicated published method was found for snapping speaker-change points to word boundaries or pauses.

## 2. Forced alignment quality (wav2vec2, MMS, ctc-forced-aligner, Whisper cross-attention) and VAD

### Takeaway
Word-boundary error varies a lot between aligners:
- MFA: about 21 ms MAE on human-labelled TIMIT and Buckeye.
- CrisperWhisper: about 30–41 ms (self-reported).
- WhisperX wav2vec2: about 65 ms.
- stable-ts: about 70–100 ms.

Errors of 50–100 ms matter mainly at fast turn exchanges, where a sentence's first or last word sits within about 100 ms of the diarization boundary. That is the requester's failure pattern. Because diarization boundaries themselves are imprecise, sentence-level smoothing (Section 1) probably buys more than switching aligner.

### Cited Findings
- **WhisperX paper (Interspeech 2023).** wav2vec2 forced alignment gives word timestamps around ±50 ms, against about ±500 ms for vanilla Whisper. On telephone speech, word precision at a 200 ms tolerance is 93.2% vs 85.4%. — [Bain et al., WhisperX, Interspeech 2023](https://www.isca-archive.org/interspeech_2023/bain23_interspeech.pdf)
  - The ±50/±500 framing comes from secondary summaries in the search results. The 93.2/85.4 figure is reported in the paper.
- **FA-Bench (Olewave; arXiv 2609.32396; leaderboard dated 2026-10-28; Apache-2.0 from release 2.0.0).** It scores word and phone boundaries against human labels on TIMIT (read) and Buckeye (conversational), clean and with reverb, noise, music or babble. It has two tracks: forced alignment with a reference transcript, and ASR systems that produce their own words. — [FA-Bench GitHub](https://github.com/olewave/fa-bench/), [FA-Bench leaderboard](https://olewave.com/en/leaderboard/fa-bench), [paper](https://arxiv.org/html/2609.32396)
  - MFA 3.4: word MAE 21.8 ms on TIMIT and 21.3 ms on Buckeye; mean F1 0.658, dropping to 0.628 with recognized words.
  - Charsiu: 28.1 ms MAE. PocketSphinx: 30.5 / 22.1 ms.
  - On the ASR-words track, WhisperX mean F1 is 0.257 and stable-ts 0.236.
  - stable-ts MAE is 96.8 ms on TIMIT and 71.2 ms on Buckeye. whisper-timestamped mean F1 is 0.140.
  - The leaderboard warns that some aligners misplace final words by more than 300 ms even on clean audio. The fetch did not say which systems.
  - Benchmark run by a company (Olewave) that also had its own aligner on it; it is now "unranked".
- **CrisperWhisper 2.0 (README, self-reported).** Word-timing MAE of 29.6 ms on TIMIT vs 64.8 ms for WhisperX, and about 41 ms on conversational speech. Timestamps come from "supervised cross-attention alignment", i.e. Whisper alignment heads trained for word boundaries. Inference code is MIT, but **model weights are under the Nyra Health Non-Commercial Research License** (commercial use needs a commercial license), and the "Pro" models are commercial-only. — [nyrahealth/CrisperWhisper](https://github.com/nyrahealth/CrisperWhisper) (vendor claim; not independently verified beyond FA-Bench listing it)
- **Rogers et al. 2024, "Tradition or Innovation" (arXiv 2406.19363).** MFA beats WhisperX and MMS on alignment accuracy. MMS and WhisperX are much more accurate than older HMM-GMM aligners. On TIMIT, WhisperX correctly recognized 37,685 words vs 29,057 for MMS (a recognition-coverage comparison). — [arXiv 2406.19363](https://arxiv.org/html/2406.19363v1)
- **WhisperX unaligned words.** Words with characters outside the wav2vec2 vocabulary (digits, currency) get no timing. — [whisperX README](https://github.com/m-bain/whisperX)

### Inferences
- **Hardware.** All the open aligners above (wav2vec2 in WhisperX, MMS via torchaudio or ctc-forced-aligner, stable-ts, whisper-timestamped, CrisperWhisper) are small enough for a 16 GB RTX 5060 Ti. The main integration risk on Blackwell (sm_120) under WSL2 is getting a PyTorch/CUDA build that supports the card, not VRAM. (Inference from general requirements, not from a source.)
- **When to switch aligners.** Going from about 65 ms (wav2vec2) to about 30 ms (CrisperWhisper) halves boundary error. But diarization turn boundaries are usually much less precise than that (pyannote frames plus collar-scale errors), so the expected gain in speaker assignment is likely small next to sentence-level smoothing. CrisperWhisper's non-commercial weight license also matters.
- **Useful aligner feature.** Verbatim transcription, as in CrisperWhisper, keeps fillers and pauses, which helps pause-based turn detection.

### Gaps
- No source measured how aligner timestamp error translates into speaker-assignment error (WDER) directly.
- No ms-level FA-Bench figures were retrieved for WhisperX's wav2vec2 aligner in the reference-transcript track, or for ctc-forced-aligner/MMS.
- No sourced evaluation was found of VAD settings (Silero vs pyannote VAD, `vad_onset`/`vad_offset`/`chunk_size`) and their effect on diarization or word assignment. The WhisperX README excerpt fetched did not give default VAD parameters.

## 3. Sentence segmentation that respects punctuation and speaker turns

### Takeaway
Segment sentences on punctuation first, using restored punctuation or SaT on unpunctuated text. Treat a diarization speaker change *inside* a sentence as a probable diarization error unless the evidence is strong (long minority run, large overlap margin, pause). Only split where a sentence boundary and a speaker change coincide, or nearly coincide. SaT (wtpsplit, MIT) is the strongest open sentence segmenter and runs easily on a 16 GB GPU.

### Cited Findings
- **wtpsplit / SaT (Segment any Text).** Frohmann et al., EMNLP 2024; MIT license.
  - Model sizes: `sat-3l-sm` (fast) to `sat-12l-sm` (most accurate). The "-sm" models are meant for general segmentation, including **unpunctuated, lower-cased text** (e.g. `"this is a test this is another test"`). English F1 is about 96.5–97.4.
  - A `threshold` parameter controls how conservative splitting is; it is ignored when length constraints are used.
  - LoRA domain/style adaptation needs only 10–100 segmented example sentences, and TED (ASR-style) LoRA modules exist.
  - Runs on CPU or GPU (`.half().to("cuda")`). ONNX gives about 50% speedup.
  - [wtpsplit GitHub](https://github.com/segment-any-text/wtpsplit)
- **whisper-diarization's segmentation rule.** A new sentence starts when the speaker changes or Punkt detects a sentence break, *after* punctuation-aware speaker realignment. In other words, speaker labels are fixed first so speaker changes fall on sentence boundaries. — [whisper-diarization helpers.py](https://raw.githubusercontent.com/MahmoudAshraf97/whisper-diarization/main/helpers.py)
- **Lexical SEC (Paturi et al., Interspeech 2023).** A second-pass LM uses lexical context to correct word-level speaker labels from the acoustic system, with 15–30% relative WDER reduction on RT03-CTS, Callhome American English and Fisher. This shows that text context (e.g. sentence continuity) can repair turn-boundary errors. — [Lexical SEC, Interspeech 2023](https://www.isca-archive.org/interspeech_2023/paturi23_interspeech.pdf). AG-LSEC adds audio grounding to this. — [AG-LSEC arXiv 2406.17266](https://arxiv.org/pdf/2406.17266)

### Inferences
- **Proposed decision rule** for a speaker change at word k with no terminal punctuation:
  1. Run SaT on the unpunctuated word sequence of the span, or use Whisper's punctuation. If SaT puts a boundary at k with probability above the threshold, accept the split and add punctuation.
  2. Otherwise apply the sentence majority vote from Section 1.
  3. Keep the split only if the minority run is long (e.g. ≥4 words or ≥1.5 s) *and* its diarization overlap margin is large.
  - The thresholds are heuristic and unsourced; tune them on the requester's 287 spans.
- **Validation.** Any span ending without terminal punctuation at a speaker boundary is a cheap detector of suspect splits; in the requester's data it flags all 14 cases. After repair, re-run it, and also count A|B|A sandwiches where the middle run is under N words.

### Gaps
- No sourced quantitative work was found on pause-length or question-intonation cues for turn-taking in the ASR+diarization "glue" step. Pause lengths can be computed from word timestamps, but I found no published threshold for "a gap of X ms makes a speaker change plausible".
- No published evaluation of SaT specifically on Whisper output with speaker turns.
- Punctuation-restoration models for conversational ASR (e.g. deepmultilingualpunctuation, other 2024–26 models) were not benchmarked in sources I retrieved.

## 4. Using metadata: num_speakers=2, interviewer/guest roles, names from the description, Q/A structure

### Takeaway
There is substantial published work on role-aware and LLM-based diarization refinement:
- DiarizationLM (Google, Interspeech 2024): 45–56% relative WDER reduction with a fine-tuned LLM.
- Lexical SEC: 15–30% relative WDER reduction.
- Training-free LLM refinement with identity/role assignment (arXiv 2509.15082): about 30% relative DER reduction in clinician–patient talk, where confusion error dropped from 9.6% to 2.8%.
- Role-diarization models (doctor/patient, interviewer/respondent).

For a two-person interview, the cheap path is:
1. Fix `num_speakers=2`.
2. Repair labels at sentence level.
3. Assign roles per cluster: e.g. the cluster that asks most questions or speaks less is the interviewer, and names come from the description via an LLM.
4. Have an LLM re-check only low-confidence or contradictory sentences, such as questions attributed to the guest.

### Cited Findings
- **DiarizationLM (Wang et al., Google; arXiv 2401.03506; Interspeech 2024).** ASR words and diarization labels are serialized into a compact text prompt. An LLM, fine-tuned PaLM 2-S, rewrites the speaker labels, which are then transferred back onto the original words. Relative WDER reduction: 55.5% on Fisher and 44.9% on Callhome English. — [arXiv 2401.03506](https://arxiv.org/abs/2401.03506)
  - A later paper found DiarizationLM *degraded* results on SD+ASR clinician–patient data (41.47% vs 23.05% DER baseline), which it attributes to lost temporal alignment when segments are merged. — [arXiv 2509.15082](https://arxiv.org/html/2509.15082v2)
- **Training-free identity-aware LLM refinement (arXiv 2509.15082).**
  - **Pipeline:** (1) align ASR words to diarization by timestamps; (2) mark low-confidence segments where the original label disagrees with a re-check against similar speaker embeddings (TitaNet), and send only those to the LLM with surrounding context and time gaps; (3) the LLM assigns identities or roles and merges over-split speakers "only if very sure".
  - **Models:** GPT-4.1, Qwen-3-8B and Qwen-3-32B as LLMs; pyannote 3.1 or Sortformer for diarization; Parakeet-TDT-0.6B-v2 for ASR.
  - **Results:** on clinician–patient data (10 recordings), DER 23.05 → 16.19% with GPT, and confusion 9.60 → 2.84%. On AMI-SDM, 27.12 → 24.51%.
  - **Date:** the HTML summary gave the submission date as "August 2, 2026", which conflicts with the 2509 (Sept 2025) arXiv id. Probably v2 vs v1; treat the date as uncertain.
  - The fetched summary did not give a separate number for the open Qwen-3-8B, which is the variant that would run locally.
  - [arXiv 2509.15082v2](https://arxiv.org/html/2509.15082v2)
- **Lexical SEC.** 15–30% relative WDER reduction on telephony with an LM second pass. — [Interspeech 2023](https://www.isca-archive.org/interspeech_2023/paturi23_interspeech.pdf)
- **"Language Modelling for Speaker Diarization in Telephonic Interviews" (arXiv 2501.17893, Jan 2025).** Iteratively combines GMM acoustic scores with an LSTM over character-level word embeddings in two-party call-centre interviews (interviewer vs customer roles). Reports an "84.29% improvement in terms of a word-level DER" over an HMM/VB baseline. Single source, proprietary dataset, and the baseline is weak by 2026 standards. — [arXiv 2501.17893](https://arxiv.org/abs/2501.17893)
- **ASR-guided speaker-role diarization (arXiv 2507.17765, July 2025).** An auxiliary role-diarization transducer runs alongside a frozen ASR transducer and predicts roles (doctor/patient) instead of anonymous speakers, reaching R-WDER 7.1% on an internal 1,750 h dataset. Role and word prediction need different predictor context. No code. — [arXiv 2507.17765](https://arxiv.org/html/2507.17765v1)
- **Speaker role and language diarization for multilingual interviews (arXiv 2608.09032, 2026).** Frame-level classification into interviewer / respondent / silence / overlap. — [arXiv 2608.09032](https://arxiv.org/abs/2608.09032) (abstract-level only; not read in full)
- **Iterative LLM-based improvement for French clinical interview transcription and diarization (arXiv 2603.00086, 2026).** Another LLM-refinement example in interviews. — [arXiv 2603.00086](https://arxiv.org/pdf/2603.00086) (title-level only; not read)
- **The `num_speakers=2` caveat.** Fixing `num_speakers` is supported natively by community-1. — [model card](https://huggingface.co/pyannote/speaker-diarization-community-1). But users report that speaker-count constraints sometimes make results worse. — [whisperX issue #1208](https://github.com/m-bain/whisperX/issues/1208) (anecdotal)

### Inferences
- **Proposed metadata-aware repair order** (heuristic; not sourced as a whole):
  1. Diarize with `num_speakers=2` and use `exclusive_speaker_diarization`.
  2. Assign words, then run punctuation-aware sentence realignment.
  3. Label clusters with roles. The interviewer is the cluster with a higher share of questions, shorter turns and less talk time. If names appear in the description, an LLM matches self-introductions ("I'm X", "welcome X") to clusters.
  4. Flag sentences whose label contradicts the role prior: questions labelled guest, or a guest's long answer containing a cluster switch. Send only those, with ±2 sentences of context and the gaps, to a local LLM (e.g. Qwen-3-8B in 4-bit, which fits in 16 GB) for a keep or flip decision. This mirrors the "low-confidence segments only" design of arXiv 2509.15082, which avoids DiarizationLM's temporal-alignment degradation.
  - Do not flip a question automatically: guests do ask rhetorical or real questions. With 14/33 questions on the guest, the question-to-interviewer prior is a strong signal but not proof.
- **Cost and integration risk:**

| Step | Cost | Integration risk |
|---|---|---|
| Realignment and SaT | Negligible | Low |
| LLM pass on flagged sentences | Small (a few dozen prompts per video) | Medium (prompt design, label transfer back to word ids) |
| Retraining role-diarization models (2507.17765, 2608.09032) | High | High; not practical (no code, internal data) |

### Gaps
- No open-source, ready-made "interviewer/guest role classifier" for diarized transcripts was found.
- No published gain numbers for using names from video descriptions specifically.
- No retrieved quantitative results for local open LLMs (Qwen-3-8B) on the 2509.15082 task.
- Not researched: SEAL (Kumar et al., 2025, acoustic-conditioned LLM speaker error correction), which 2509.15082 cites. Also not researched: DiarizationLM's open-source code and license (the arXiv page did not state them; Google's repo is believed to be on GitHub but was not verified).
