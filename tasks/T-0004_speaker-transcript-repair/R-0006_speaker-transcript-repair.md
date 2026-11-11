---
id: R-0006
type: report
task: T-0004
title: "Snap speakers to sentences, then let evidence decide"
status: final
created: 2026-11-03
tags: [speaker-diarization, transcript-repair, comment-linking]
uses: [N-0015, N-0016, N-0017, N-0018, N-0019, R-0005]
---

# Snap speakers to sentences, then let evidence decide

*Part 5 (T-48): speaker-attributed transcript repair for Ruminatai and EXP-009. Research notes dated 2026-11-03; PyPI versions checked that day. Builds on the Part 4 report "Transcript rechunking for comment linking", which covers chunk size, length bias, merge-first re-segmentation and link gating. That material is not repeated here.*

All three defects you observed come from one design choice in WhisperX-style pipelines: **each word gets its speaker from a timestamp lookup that never looks at punctuation or sentence structure**. Fix that rule first and most of the damage goes away. WhisperX's `assign_word_speakers` gives each word, and separately each segment, to the speaker with the largest summed time overlap. If no diarization segment overlaps a word, it falls back to the nearest segment. A diarization boundary a few hundred milliseconds off therefore cuts "So is that a five-alarm fire to | you?" in two ([whisperX diarize.py](https://raw.githubusercontent.com/m-bain/whisperX/main/whisperx/diarize.py)). In your 24-minute interview, 46 turns give about 45 speaker changes, and 14 of them fall mid-sentence: **roughly 3 in 10 speaker changes are broken**. **For your existing spans**, the most useful repair runs on the CPU and needs no new model. Detect every speaker change at a word without terminal punctuation. Relabel the enclosing sentence by duration-weighted majority, but only when the minority run is short. This is the rule in whisper-diarization ([helpers.py](https://raw.githubusercontent.com/MahmoudAshraf97/whisper-diarization/main/helpers.py)), with a locality gate borrowed from DiarizationLM ([utils.py](https://raw.githubusercontent.com/google/speaker-id/master/DiarizationLM/diarizationlm/utils.py)). After that, an LLM checks only the flagged questions and interjections. It emits label edits keyed to word ids and never rewrites text. **At extraction time**, change these:
- run pyannote community-1 with `num_speakers=2` and assign words from its `exclusive_speaker_diarization` output;
- make the realignment rule part of the pipeline;
- emit a per-word speaker margin;
- A/B test NVIDIA's Nemotron-3-Diarization, the strongest diarizer that is both licensed for commercial use and listed for Blackwell GPUs.

**The weak point of the whole field is evidence of gain on data like yours.** No source measures WDER before and after punctuation realignment. The large LLM-corrector gains (45–56% relative WDER) come from fine-tuned models on telephone speech. Zero-shot LLMs made speaker attribution *worse* in two independent studies. No study varies speaker-label accuracy and measures retrieval. So the plan is built around a small in-domain labelled set, and it measures each stage before trusting it.

## The defects trace to a timestamp-overlap rule, not a weak diarizer

### Why sentences break at speaker changes

**WhisperX assigns speakers by time overlap alone.** Its own README says the alignment model cannot time words whose characters are outside its dictionary, such as "2014." or "£13.60" ([whisperX README](https://github.com/m-bain/whisperX)). PR #1338 (merged 2026-02-08) made speaker assignment about 228× faster. Maintainers confirmed it produces identical assignments, so **the accuracy logic has not changed** ([PR #1338](https://github.com/m-bain/whisperX/pull/1338)). It did fix a long-standing bug that summed negative intersections in the nearest-speaker fallback ([PR #590](https://github.com/m-bain/whisperX/pull/590)).

**Words can be timed precisely while diarization boundaries stay coarse.** WhisperX's wav2vec2 alignment gets word timestamps to within roughly ±50 ms, and 93.2% of words land within 200 ms on telephone speech ([Bain et al., Interspeech 2023](https://www.isca-archive.org/interspeech_2023/bain23_interspeech.pdf)). On FA-Bench, MFA reaches about 21 ms word MAE ([FA-Bench](https://github.com/olewave/fa-bench/)). CrisperWhisper self-reports 29.6 ms against 64.8 ms for WhisperX ([CrisperWhisper](https://github.com/nyrahealth/CrisperWhisper), vendor claim). Diarization boundaries, by contrast, are usually reported only through DER with a 0.25 s collar. One to three words of speech is about 0.3–1.2 s.

**The fault is therefore in the glue step, not the aligner.** An ASR vendor describes the same symptom: splits land "at the nearest clause boundary, not the actual speaker boundary", leaving "one or two words attributed to the wrong speaker at every turn" ([AssemblyAI blog](https://assemblyai.com/blog/whisper-speaker-diarization), vendor, anecdotal). A September 2026 NTT paper splits DER into a component caused by pause filling and a "core" component. It argues that standard DER hides real boundary errors ([arXiv 2609.11007](https://arxiv.org/abs/2609.11007)). No source in this research measured speaker-change offset in words or milliseconds for any diarizer. That is why **DER rankings are a weak guide to the 1–3-word errors you see.**

### Why questions go to the guest

**The 14/33 questions on the guest are a mixed population.** Some are edge splits, like the "How | would you describe | it?" sandwich. Sentence realignment repairs those: the labels [G, I, I, I, G] have a 3-of-5 majority for the interviewer, so the whole question moves. Others are whole sentences placed in the wrong cluster, where a short interviewer turn has been absorbed into the guest's run. **Sentence-level realignment cannot fix those.** They need a role prior plus a check of the content.

Some of the 14 will be genuine guest questions, rhetorical or clarifying, so the true error count is unknown until someone labels them. Absorbed interjections are the third pattern. Pipelines and corpora often treat backchannels ("yeah", "mhm", "okay") as separate utterances ([arXiv 2507.09226](https://arxiv.org/pdf/2507.09226), **single source**). That supports keeping them as their own spans rather than merging them into a neighbour.

### Why span lengths are uneven

**Uneven lengths are mostly a consequence of the first two defects.** Every false speaker change creates two fragments. That accounts for many of your 27 spans of three words or fewer, and for the p10 of 18 characters.

The Part 4 report flagged that this kind of over-splitting distorts retrieval: cosine similarity rises with text length, and very short units are hard to match ([Jina 2025](https://jina.ai/news/on-the-size-bias-of-text-embeddings-and-its-impact-in-search)). One correction to Part 4: it said "only 14 of 287 boundaries are speaker boundaries". The 46-turn count shows those 14 are the *mid-sentence* speaker boundaries, not all of them. Turns average about six spans.

## Post-hoc repair: rules first, a gated LLM second

### Rule-based relabeling with guards

**The best-tested open rule is whisper-diarization's punctuation realignment, `get_realigned_ws_mapping_with_punctuation`.** It works like this:
- It finds each word k where the speaker changes and word k does not end in `.?!`.
- It walks left while the speaker stays the same and no sentence end appears. It walks right to the next sentence end. Both walks stop at 50 words.
- If the most common speaker in that range holds at least `len//2` of the words, every word in the range is relabelled to that speaker.

The code is in [helpers.py](https://raw.githubusercontent.com/MahmoudAshraf97/whisper-diarization/main/helpers.py). The notes disagree on the license: one calls it MIT-style, the other reads it as BSD-2-Clause from the repo ([repo](https://github.com/MahmoudAshraf97/whisper-diarization)). Check before you vendor it.

**The rule has three known flaws:**
- **Ties:** with the `len//2` threshold, a 2–2 tie in a four-word sentence still relabels, and which speaker wins is effectively arbitrary.
- **Alternating labels:** in an A|B|A|B sentence the left walk stops at the first change, so the sentence is skipped.
- **Real interruptions:** every mid-sentence change is treated as an error, so a genuine interruption is overwritten by whoever has more words.

**DiarizationLM's code offers the right guard.** Its `apply_locality_gated_speaker_transfer` accepts a label change only when the contiguous run of changed words is at most `max_local_span_words=5`. Longer runs keep the acoustic speaker. The same code also absorbs "ghost" third speakers: any speaker with at most `min(12, max(2, 1% of words))` words is relabelled to a neighbour ([utils.py](https://raw.githubusercontent.com/google/speaker-id/master/DiarizationLM/diarizationlm/utils.py), Apache-2.0, last push 2026-10-06).

**The combined rule** (the gates are inferred, and none has been evaluated as a package):
- Weight the vote by word duration, not word count.
- Relabel only when the minority run is no more than about 3–5 words or about 1 s.
- Do not merge when:
  - the short side is a closed-class backchannel ("yeah", "right", "mm-hm", "okay", "sure", "exactly") and the first speaker then continues;
  - the first fragment ends in an em-dash;
  - the gap is longer than about 1 s.

**Calibrate these thresholds by hand-labelling the 14 split spans and the 27 short ones.** No source supplies them.

### Where an LLM helps, and where it hurts

**Fine-tuned LLM correctors show large gains, but only in their own domain:**
- **DiarizationLM:** a fine-tuned PaLM 2-S cut WDER by 55.5% relative on Fisher (5.32% → 2.37%) and by 44.9% on Callhome ([arXiv 2401.03506](https://arxiv.org/html/2401.03506)).
- **Efstathiadis et al.:** a Mistral-7B ensemble, including a model trained on WhisperX + pyannote 3.1.1 output, reduced deltaCP by 29% on WhisperX transcripts and by 39–55% on out-of-domain medical consultations ([arXiv 2406.04927](https://arxiv.org/html/2406.04927v2)).
- **SEAL:** reduced speaker error by 24–43% relative, but it needs the diarizer's per-word posteriors and has no released code ([arXiv 2501.08421](https://arxiv.org/html/2501.08421v1)).
- **Lexical Speaker Error Correction:** a word-preserving classifier that cut WDER by 15–30% relative on telephone corpora ([Paturi et al., Interspeech 2023](https://www.isca-archive.org/interspeech_2023/paturi23_interspeech.pdf)).

**Unconstrained zero-shot LLMs are actively harmful:**
- Zero-shot PaLM 2-S more than doubled WDER, from 5.32% to 11.96%, and often "delete[d] large chunks of hypothesis text" ([arXiv 2401.03506](https://arxiv.org/html/2401.03506)).
- Zero-shot Mistral-7B raised deltaCP error by 1677% on AWS transcripts ([arXiv 2406.04927](https://arxiv.org/html/2406.04927v2)).
- On clinician–patient audio, a later study found DiarizationLM itself degraded results, with DER of 41.47% against a 23.05% baseline. The authors blame lost temporal alignment ([arXiv 2509.15082](https://arxiv.org/html/2509.15082v2)).

**The design that does work locally sends only the low-confidence material to the LLM.** In arXiv 2509.15082, the pipeline:
- flagged segments whose labels disagreed with a speaker-embedding re-check;
- sent only those, with surrounding context and the time gaps, to an LLM;
- let the LLM merge speakers "only if very sure".

The result was a fall in clinician–patient DER from 23.05% to 16.19%, and in confusion from 9.60% to 2.84%, with GPT-4.1 ([arXiv 2509.15082](https://arxiv.org/html/2509.15082v2)). The notes did not capture a separate number for the locally runnable Qwen-3-8B. The **sample is 10 recordings**, so treat this as one study.

**Two mechanisms keep text, word ids and timestamps intact whatever the LLM outputs:**
- **TPST.** DiarizationLM's `transcript_preserving_speaker_transfer` aligns the LLM's output to the original words with Levenshtein alignment, then maps speaker labels with Hungarian matching. It returns one label per *original* word ([utils.py](https://raw.githubusercontent.com/google/speaker-id/master/DiarizationLM/diarizationlm/utils.py)).
- **Constrained decoding.** SEAL lets the model emit only the next input word or a speaker label at each step.

With Claude, GPT or a local Qwen, the simpler route is to ask only for JSON edit operations that reference existing ids, and to reject any id that does not exist. Examples: `relabel span → host`, `merge [a, b]`, `split after w391`.

### Post-hoc shortlist (Area B)

| Method / tool | Version, date | License | Local on 5060 Ti 16 GB, Py3.12 | Expected gain (source) | Cost | Integration risk |
|---|---|---|---|---|---|---|
| Punctuation-aware sentence realignment (port of whisper-diarization rule, duration-weighted, ≤5-word gate) | helpers.py on main, read 2026-11 | BSD-2-Clause (one note says MIT-style; verify) | Yes, CPU only, about 60 lines | **Unquantified**: no WDER before/after in any source. It directly targets all 14 mid-sentence splits by construction | Negligible | Low. Main risk is overwriting genuine interruptions; the gate limits it |
| Locality gate + ghost-speaker absorption (`diarizationlm` utils) | 0.1.5, 2025-03-31; repo push 2026-10-06 | Apache-2.0 | Yes, CPU | Design guard, not a gain claim ([utils.py](https://raw.githubusercontent.com/google/speaker-id/master/DiarizationLM/diarizationlm/utils.py)) | Negligible | Low |
| Label-only LLM check of flagged questions/interjections, JSON edit ops, TPST back to word ids | Pattern from arXiv 2509.15082 (2025, v2 2026) | Depends on model; your local Qwen GGUF | Yes (a few dozen prompts per video) | Confusion 9.60 → 2.84% with GPT-4.1 on clinician–patient audio; **single study, n=10** ([2509.15082](https://arxiv.org/html/2509.15082v2)) | Minutes of GPU per video | Medium: prompt design, id validation, false flips of genuine guest questions |
| DiarizationLM-Gemma-4-E4B-v1 (fine-tuned corrector) | Release date not captured | Apache-2.0 per card | Yes at Q4_K_M (~5.3 GB); bf16 (~16 GB) does not fit with KV cache | Fisher WDER 2.99%, Callhome 4.92%; small gains on AMI/ICSI; **single-source community card** ([HF](https://huggingface.co/diarizers-community/DiarizationLM-Gemma-4-E4B-v1)) | Low–medium; 4–6 windows of ≤6k chars per video | Medium-high: trained on unpunctuated telephone text; corrections are ASR-specific ([2406.04927](https://arxiv.org/html/2406.04927v2)) |
| google/DiarizationLM-8b-Fisher-v2 (Llama 3 8B LoRA) | v2 | Not stated on card; presumably Llama 3 Community License (unverified) | Only quantised (8-bit ~9 GB or GGUF) | Fisher WDER 5.32 → 3.28% (−38% rel.) ([HF](https://huggingface.co/google/DiarizationLM-8b-Fisher-v2)) | Low–medium | Medium-high: same domain mismatch, 6k-char input cap |
| Zero-shot LLM that rewrites the transcript | n/a | n/a | Yes | **Negative**: WDER 5.32 → 11.96% (PaLM 2-S); deltaCP +1677% (Mistral) | — | **Do not use** |
| SEAL / contextual beam search | arXiv 2501.08421 (Jan 2025); ICASSP 2024 | No code (SEAL) | Would need retraining | 24–43% rel. (SEAL); up to 39.8% rel. ([2309.05248](https://www.emergentmind.com/papers/2309.05248)) | High | High: need per-word acoustic posteriors your spans lack |

## Extraction fixes: change the assignment rule before changing the diarizer

### Assignment and segmentation changes

**The cheapest extraction change is to feed the existing assigner a better diarization output.** pyannote's `speaker-diarization-community-1` has an output called `exclusive_speaker_diarization`, in which only one speaker is active at a time. pyannote says it "simplifies the reconciliation between fine-grained speaker diarization timestamps and (sometimes not so precise) transcription timestamps". The model also accepts `num_speakers=2` ([model card](https://huggingface.co/pyannote/speaker-diarization-community-1), CC-BY-4.0, gated).

community-1 is already WhisperX's default diarization model ([diarize.py](https://raw.githubusercontent.com/m-bain/whisperX/main/whisperx/diarize.py)). The notes did not establish whether WhisperX consumes the exclusive output. Check the call in your installed version.

**No published WDER exists for the exclusive output against the regular one, so this is a reasoned change, not a measured one.** Forcing the speaker count also carries an anecdotal risk: users report that `min_speakers`/`max_speakers` sometimes collapse speech onto one speaker ([whisperX issue #1208](https://github.com/m-bain/whisperX/issues/1208), **single-source anecdote**).

**Inside the pipeline, the segmentation order should follow whisper-diarization:**
1. assign words;
2. realign speakers to sentences;
3. segment sentences on punctuation, starting a new sentence at a speaker change only where it coincides with a sentence boundary.

When Whisper's punctuation is missing, SaT (`wtpsplit` 2.2.2, MIT) segments unpunctuated, lower-cased text at English F1 of about 96.5–97.4 ([wtpsplit](https://github.com/segment-any-text/wtpsplit)). Two further changes need no new model:
- Give words that WhisperX cannot time (numerals, currency) interpolated timestamps before assignment, so the nearest-segment fallback cannot give them to the other speaker.
- Emit a per-word speaker margin, the overlap seconds for the winning speaker minus those for the runner-up. That margin is what makes the locality gate, the LLM flagging, and a later SEAL-style "low/medium/high" confidence token possible ([SEAL](https://arxiv.org/html/2501.08421v1)).

**Role and name assignment belongs at extraction, as a per-cluster decision with explicit confidence.** Cues include question share, turn length and talk time, and self-introductions matched against the names in the video description. Role-diarization research exists, but none of it is usable here:
- **Interviewer/customer LM** (arXiv 2501.17893): reports "84.29% improvement" in word-level DER over a weak baseline, on proprietary data, **single source**.
- **Doctor/patient role transducer** (arXiv 2507.17765): R-WDER 7.1%, no code.

No ready-made open interviewer/guest classifier was found. Treat "questions belong to the host" as a soft prior that flags a sentence for review, never as a rule that flips it automatically.

### Diarizer and joint-model alternatives

**Among open diarizers, the 2025–2026 leaders are close on DER, and their benchmarks are not comparable.** Their reported DER figures:

| Model | DER figures reported |
|---|---|
| pyannote community-1 vs legacy 3.1, no collar | AMI-IHM 18.8 vs 17.0; CALLHOME 28.5 vs 26.7; VoxConverse 11.2 vs 11.2 ([card](https://huggingface.co/pyannote/speaker-diarization-community-1)) |
| Streaming Sortformer 4spk-v2 (117M, CC-BY-4.0) | DIHARD III 13.24% (no collar); 2-speaker CALLHOME 6.57% with a 0.25 s collar ([card](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2)) |
| NVIDIA Nemotron-3-Diarization (released 2026-09-23; 100M; OpenMDW v1.1, commercial use allowed; Blackwell listed; needs NeMo 3.0) | DIHARD III 12.73%; CALLHOME-Part2 9.10% (0.25 s collar) ([card](https://huggingface.co/nvidia/nemotron-3-diarization)) |
| DiariZen-Large-s80-v2 | AMI-SDM 13.9 vs 22.4 for pyannote 3.1 ([GitHub](https://github.com/BUTSpeechFIT/DiariZen)) |

Nemotron-3's figures are **vendor-only and single-source**. Its CALLHOME figure is probably not 2-speaker-only, so it does not compare with Sortformer's 6.57%. DiariZen's model weights are **CC BY-NC 4.0**, and its toolchain is pinned to torch 2.1.1 with cu121. On an RTX 5060 Ti (sm_120, which needs a CUDA ≥ 12.8 build), that pin is effectively a non-starter (inferred from the toolchain, unverified). The frame-level end-to-end models, Sortformer and Nemotron, avoid the clustering errors that can swallow short turns. Whether that helps with absorbed interjections is an untested hypothesis.

**Joint speaker-attributed ASR removes the timestamp-lookup step entirely, at the price of replacing the stack:**
- **DiCoW** conditions Whisper on frame-wise diarization output. The BUT DiCoW + DiariZen system ranked 2nd in MLC-SLM 2025 Task 2 at 16.75% tcpWER ([arXiv 2506.13414](https://arxiv.org/pdf/2506.13414)).
- **SE-DiCoW** reports a 52.4% relative tcpWER reduction over the original DiCoW ([arXiv 2601.19194](https://arxiv.org/html/2601.19194), **single source**).
- **NVIDIA multitalker-parakeet-streaming-0.6b-v1** conditions an RNNT on Sortformer output and reaches AMI-IHM cpWER of 21.26% ([card](https://huggingface.co/nvidia/multitalker-parakeet-streaming-0.6b-v1)). The card gives no cascaded baseline.

**No source compares cascaded and joint WDER on two-person interview audio.** That makes the joint models a research arm, not a near-term change. Swapping aligners (CrisperWhisper's weights are non-commercial) halves word-timing error but leaves diarization boundary error untouched, so it ranks last.

### Extraction shortlist (Area A)

| Method / tool | Version, date | License | Local on 5060 Ti 16 GB, Py3.12 | Expected gain (source) | Cost | Integration risk |
|---|---|---|---|---|---|---|
| community-1 + `num_speakers=2` + `exclusive_speaker_diarization` for word assignment | pyannote.audio 4.0.7 (2026-06-30); benchmark 2025-09 | CC-BY-4.0 (gated HF) | Yes; Py ≥3.10; needs torch with CUDA ≥12.8 for sm_120 (unverified) | DER 3.1 → community-1: AMI-IHM 18.8 → 17.0, CALLHOME 28.5 → 26.7; **no WDER for exclusive mode** ([card](https://huggingface.co/pyannote/speaker-diarization-community-1)) | Low | Low–medium: pyannote 4.0.0 date conflict in notes (2024 vs 2025); speaker-count forcing anecdotally harmful ([#1208](https://github.com/m-bain/whisperX/issues/1208)) |
| In-pipeline punctuation realignment + SaT segmentation + speaker-margin output + timestamp interpolation for unaligned words | wtpsplit 2.2.2 (2026-09-25); WhisperX 3.8.6 (PyPI 2026-05-25; one note saw a stale 2024 date) | MIT (wtpsplit); BSD-2-Clause (WhisperX) | Yes; SaT on GPU or CPU | Unquantified; same rule as Area B, applied before spans exist | Low | Low; requires extending the span schema (per-word speaker, margin) |
| Role/name assignment per cluster with confidence; hedged labels below threshold | Design (no tool) | — | Yes, rules plus local LLM | Training-free role refinement: confusion 9.60 → 2.84% ([2509.15082](https://arxiv.org/html/2509.15082v2), single study) | Low | Medium: false role flips on guest questions |
| Nemotron-3-Diarization as A/B diarizer | 2026-09-23 | OpenMDW v1.1 (commercial OK) | Blackwell listed; NeMo 3.0; VRAM for 24 min not measured | DIHARD III 12.73% DER; **vendor single source** ([card](https://huggingface.co/nvidia/nemotron-3-diarization)) | Medium | Medium: NeMo dependency, new install path |
| Streaming Sortformer 4spk-v2 | Interspeech 2025 paper | CC-BY-4.0 | Likely; RTF 0.18 on RTX 6000 Ada ≈ 4.3 min per 24-min file at the streaming setting | DIHARD III 13.24% DER ([card](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2)) | Medium | Medium: NeMo main install |
| DiCoW v3 / SE-DiCoW, multitalker-parakeet (joint SA-ASR) | 2025 / Jan 2026 / v1 | DiCoW checkpoint license not checked; NVIDIA Open Model License | Plausible; unmeasured | tcpWER 16.75% (MLC-SLM rank 2); SE-DiCoW −52.4% rel. (single source) | High | High: replaces the ASR stack; no interview benchmark |
| DiariZen-Large-s80-v2 | 2025-12-09 | Code MIT, **weights CC BY-NC 4.0** | **No as-is** (torch 2.1.1/cu121, Py3.10) | AMI-SDM 22.4 → 13.9 DER vs 3.1 ([GitHub](https://github.com/BUTSpeechFIT/DiariZen)) | High | High: license and toolchain |
| CrisperWhisper aligner | 2.0 | Weights non-commercial (Nyra Health) | Yes | Word MAE 29.6 vs 64.8 ms (self-reported) | Medium | Medium; low payoff for speaker errors |

## Evaluation must count words and boundaries, not seconds

### Which metrics fit these errors

**A 0.25 s DER collar hides most of the errors you care about.** A two-word slip at a turn changes DER by a few tenths of a percent. cpWER mixes speaker errors into ASR errors: on Fisher, the baseline cpWER of 21.19% sits on a WDER of only 5.32% ([arXiv 2401.03506](https://arxiv.org/html/2401.03506)). CHiME-8 scored tcpWER with a 5 s collar ([arXiv 2507.18161](https://arxiv.org/pdf/2507.18161)), far too loose to see one-word shifts.

**WDER is the headline metric.** It is defined as (S_IS + C_IS)/(S + C), where S counts substituted words, C counts correctly recognised words, and the _IS subscript marks those that carry the wrong speaker. It is implemented in `diarizationlm` 0.1.5 (Apache-2.0), which reports micro averages ([README](https://github.com/google/speaker-id/tree/master/DiarizationLM)).

**Report four constructed metrics alongside it.** They are not published standards, but each is easy to build on word indices:
- a **word-indexed speaker-change F1 at tolerance k = 0, 1, 2, 3 words**, which separates missed turns from turns a word or two off;
- **question and short-turn attribution accuracy** on a labelled list, because errors on short turns vanish in global averages dominated by long answers;
- a **speaker-pure span rate**, the share of spans whose words all have one reference speaker;
- sentence-boundary scores with near-miss credit: Boundary Similarity and WindowDiff from `segeval` 2.0.11 ([PyPI](https://pypi.org/project/segeval)).

segeval's last release was in 2013, so confirm it imports on Python 3.12.

Keep DER from `pyannote.metrics` 4.2.0 (defaults: collar 0, overlap scored) at both collar 0 and 0.25 s. The difference between the two is roughly the boundary-jitter component ([pyannote.metrics](https://pyannote.github.io/pyannote-metrics/reference.html)). Use `meeteval` 0.4.3 (MIT) for cpWER and tcpWER only as sanity checks ([meeteval](https://github.com/fgnt/meeteval)).

### Labelling cost and the reference

**Targeted, model-assisted labelling is the cheap protocol.** Labelling diarization from scratch costs 15–30× real time. Verifying automatic output costs about 2× ([arXiv 2007.01216](https://arxiv.org/pdf/2007.01216); [DIHARD III plan](https://arxiv.org/pdf/2006.05815)). A small pilot found unassisted annotation "the costliest and least accurate", with uncertainty highlighting the cheapest ([arXiv 2606.26842](https://arxiv.org/abs/2606.26842), **single source, nine files**). The best diarization confidence measures place about 30% of errors in the lowest-confidence 10% of segments ([arXiv 2406.17124](https://arxiv.org/html/2406.17124v1)). That supports labelling flagged regions first, but it also means flag-only labelling misses most errors. Add a random sample.

A practical budget is 30–60 change-point-rich minutes across 3–5 videos, with the annotator only confirming or moving change points and fixing the speaker of questions and short turns. That is roughly **2–5 annotator hours**. The figure is extrapolated from the 2× verification rate and is not itself sourced. Use Label Studio 1.23.2 (Apache-2.0) or an Audacity label track.

Run a paired comparison with a block bootstrap by video, because errors within a recording are correlated ([arXiv 1912.09508](https://arxiv.org/pdf/1912.09508)). Have two annotators label 10 minutes, to learn whether a k=0 tolerance is even meaningful at interruptions.

**Creator captions are a poor speaker reference and a risky word reference.** On Rev16 podcasts, one system's WER ranged from 11.21% against verbatim references to 4.43% on spans where both transcribers agreed ([arXiv 2412.07937](https://arxiv.org/html/2412.07937v1)).

**No study validates LLM-as-judge for speaker attribution against human labels.** Before using one on unlabelled videos, measure its Cohen's κ against your labelled short turns. Use a different model from the one inside the repair step.

**Reference-free proxies are useful monitors but cannot tell right from wrong:**
- spans ending without terminal punctuation at speaker boundaries (catches all 14 of your splits);
- A-B-A sandwiches whose middle run is under N words;
- the share of questions attributed to the guest;
- the share of spans with three words or fewer, and the span-length coefficient of variation.

For regression tests with exact ground truth, NeMo's speech data simulator builds multi-speaker sessions from single-speaker aligned speech ([arXiv 2310.12371](https://arxiv.org/pdf/2310.12371v1.pdf)). Read speech has no backchannels, though. The licence-free public sanity sets are VoxConverse v0.3 (YouTube-sourced, CC BY 4.0) and the Earnings-21/22 Q&A sections (speaker labels plus analyst questions, CC BY-SA 4.0) ([Rev datasets](https://raw.githubusercontent.com/revdotcom/speech-datasets/main/README.md)).

### Evaluation shortlist (Area C)

| Method / tool | Version, date | License | Local | What it catches | Cost | Risk |
|---|---|---|---|---|---|---|
| WDER via `diarizationlm` | 0.1.5, 2025-03-31 | Apache-2.0 | CPU, Py3.12 | Wrong-speaker words, independent of timing | Seconds | Low; needs a word-aligned reference |
| Speaker-change F1@k words; question/short-turn accuracy; speaker-pure span rate | Own code (constructed; **no published standard**) | — | CPU | The 1–3-word offsets and question misattribution directly | Hours to write | Low |
| Boundary Similarity, WindowDiff (`segeval`) | 2.0.11, 2013-12-21 | BSD | Pure Python; Py3.12 import unverified | Sentence-span boundary placement with near-miss credit | Seconds | Low–medium (unmaintained) |
| DER at collar 0 and 0.25 s (`pyannote.metrics`) | 4.2.0, 2026-10-16 | MIT | CPU, Py ≥3.11 | Comparability with published diarizer numbers | Seconds | Low |
| cpWER/tcpWER (`meeteval`) | 0.4.3, 2025-09-08 | MIT | CPU | Sanity check only | Seconds | Low |
| Targeted model-assisted labelling (Label Studio) | 1.23.2, 2026-09-29 | Apache-2.0 | Yes | Ground truth for all of the above | ~2–5 h for 30–60 min (extrapolated) | Low |
| LLM-as-judge | — | — | Yes | Triage only; **unvalidated** | Low | High if used as a metric |

## Downstream, speaker errors hurt grounding more than retrieval

**For finding the right passage, speaker labels matter less than whether units are intact.** No study was found that varies diarization accuracy and measures retrieval. That covers QMSum, MeetingBank, AMI and MediaSum, and TREC Podcasts runs did not use speaker labels.

The damage is concentrated in three places:
- **Comments that name or refer to a person**, such as "the host asked…".
- **Fragments as retrieval units.** A fragment like "you?" cannot be matched by retrieval, and if a comment's gold passage is a fragment, it cannot be linked correctly.
- **The LLM coder's reading.** A question shown under the guest's name reads as the guest's assertion.

Conversational-memory research supports the fragmentation concern. Single turns are "fragmentary" and lose to topically coherent multi-turn segments. In SeCom, GPT4Score with BM25 retrieval on LOCOMO was 65.58 for turn-level units against 71.57 for segment-level units ([SeCom, ICLR 2025](https://arxiv.org/html/2502.05589v3)). That is chat data, not ASR. Bad speaker boundaries manufacture exactly these fragments, so **repair is a prerequisite for the merge-first re-segmentation recommended in Part 4.** Run the repair first, then merge.

**For grounding, misattribution is the direct risk.** Pre-LLM dialogue summarisers traced up to 56.4% of factual errors to speaker confusion or missing speakers ([Geng et al., COLING 2022](https://aclweb.org/anthology/2022.coling-1.569.pdf), **single source, chat data**). Modern LLMs given correct labels misattribute rarely: wrong references appeared in only 3–4 of 35 meeting summaries per model family ([Kirstein et al. 2024](https://arxiv.org/html/2404.11124), small sample).

Robustness to noise depends on model size. On ELITR-Bench, GPT-4o stayed above 6/10 even at a 100% target WER, while LLaMA-3.1-8B degraded substantially ([arXiv 2403.20262](https://arxiv.org/html/2403.20262v3)). Speaker-label noise was not manipulated there. Where your local 27B coder falls between those two is an extrapolation.

MeetingQA shows that 40% of answers span non-consecutive sentences and 48% span several speakers ([ACL 2023](https://aclanthology.org/2023.acl-long.837)). A question-answer unit should therefore extend forward past the first answering turn.

**The practical rules follow, mostly by inference:**
- Keep the speaker in metadata and out of the embedded text. A "Name:" prefix is a large share of a 12-token span and pulls all of one speaker's spans together, which shrinks the margins your gate relies on. No retrieval study ablates this.
- Index BM25 speaker names only as a separate field.
- Show the label to the LLM coder only when role confidence is high. Otherwise render it hedged ("Host?").
- Add an interviewer-question + answer-start unit (about 100–150 tokens) alongside the sentence units. It gives comments that react to a question something to match.

**Your earlier finding that contextual prefixes hurt short chunks fits the evidence.** In LongMemEval, keys made only of derived text did not beat the raw text, while raw text plus extracted facts raised recall by 9.4% ([LongMemEval, ICLR 2025](https://arxiv.org/html/2410.10813v2), **single source**). An independent replication of contextual retrieval on 512-character chunks found only about +4.6% NDCG@5 ([Merola & Singh 2025](https://arxiv.org/html/2504.19754), single study, biomedical). With a 55-character median span, a 50–100-token prefix becomes most of what the embedding represents.

Once units are repaired and merged, the only context worth testing is a short, local one: the governing interviewer question, held well below the unit's own length or indexed as a separate vector. The "≤20–30% of tokens" ceiling is a heuristic to validate on your ~100 labelled links.

### Downstream shortlist (Area D)

| Method | Basis | Local | Expected effect (source) | Cost | Risk |
|---|---|---|---|---|---|
| Repair before rechunking (B then Part 4 merge) | SeCom fragment finding | Yes | Turn 65.58 → segment 71.57 GPT4Score on LOCOMO with BM25 ([SeCom](https://arxiv.org/html/2502.05589v3)); direct effect on comment linking **unmeasured** | Negligible | Low |
| Speaker in metadata, not embedded text; hedged labels for the coder | MediaSum convention for generation; no retrieval ablation | Yes | Unmeasured (inference) | Negligible | Low |
| Q-A unit indexed alongside sentence units, extending to multi-span answers | MeetingQA, LongMemEval rounds | Yes | Rounds improve reading; raw + facts keys +9.4% recall ([LongMemEval](https://arxiv.org/html/2410.10813v2), single source) | Low | Medium: duplicate hits complicate margins |
| Local question context as a separate key, not a long prefix | LongMemEval; Merola & Singh | Yes | ~+4.6% NDCG@5 for contextual prefixes on 512-char chunks (single study) | Low | Low if kept short |

## Recommended path

### Part 1: post-hoc repair, runnable now on existing spans

All of Part 1 runs on CPU except the optional LLM check, which uses the local GGUF you already run. It needs nothing beyond the word ids your spans already carry. **Gate every step on the labelled set.** If a step does not raise WDER-free accuracy on questions and short turns, or speaker-change F1 at k ≤ 1, drop it.

| Step | What | Why / evidence | Effort |
|---|---|---|---|
| 0 | Label a minimal set: every speaker change in the 24-min file (~45), all 33 questions, the 27 short spans, plus a random 10% of other spans; record the true speaker per word range and whether each change is real | No source predicts gains on interview audio; verification costs ~2× real time ([arXiv 2007.01216](https://arxiv.org/pdf/2007.01216)) | ~1–2 h |
| 1 | Flatten spans to words (w-ids, times, speaker). Run proxies and metrics (WDER, change-F1@k, question accuracy, speaker-pure rate) as the baseline | Area C | 1 h code |
| 2 | Realign at sentence level: change at a word without `.?!` → find enclosing sentence (≤50 words) → duration-weighted majority → relabel only if minority run ≤3–5 words or ≤1 s; skip backchannels, em-dash endings, gaps >1 s; absorb any ghost third speaker | [whisper-diarization](https://raw.githubusercontent.com/MahmoudAshraf97/whisper-diarization/main/helpers.py) + [DiarizationLM gate](https://raw.githubusercontent.com/google/speaker-id/master/DiarizationLM/diarizationlm/utils.py); gain unquantified | Half a day |
| 3 | Split absorbed interjections: em-dash followed by capitalised question or a backchannel token inside another speaker's sentence → new span, flag `is_backchannel` | Backchannels as independent utterances ([2507.09226](https://arxiv.org/pdf/2507.09226), single source) | Hours |
| 4 | Assign roles per cluster (question share, turn length, talk time, name match from the description); flag guest-labelled questions and A-B-A sandwiches | Role prior is soft; no source evaluates it | Hours |
| 5 | Send only flagged sentences (±2 sentences, with gaps and margins if available) to the local LLM; accept JSON `relabel/merge/split` ops on existing ids only; map back with TPST; log every change | Low-confidence-only design ([2509.15082](https://arxiv.org/html/2509.15082v2)); zero-shot rewriting harms ([2401.03506](https://arxiv.org/html/2401.03506)) | 1 day |
| 6 | Rebuild spans as maximal (same speaker, same sentence) runs with deterministic ids `s_{first}_{last}` and an old→new `span_map` with op, evidence and confidence, so EXP-009 labels remap | Engineering design; no published standard | Hours |
| 7 | Run Part 4's merge-first re-segmentation on the repaired spans, then re-score comment linking (top-1 hit rate, paired McNemar) against the unrepaired arm | Part 4 protocol | Existing harness |

The fine-tuned DiarizationLM-Gemma-4-E4B corrector (Q4, about 5.3 GB) is worth one comparison arm after Step 5. Expect less than its card's Fisher numbers, because of the domain shift.

### Part 2: changes to propose for Ruminatai's extraction step

**Propose these in order of payoff per unit of risk:**
1. **Use community-1's exclusive output for word assignment.** Pin pyannote.audio 4.0.7, run community-1 with `num_speakers=2`, and assign words by maximum overlap against `exclusive_speaker_diarization`. Keep a fallback run without the speaker count, given the anecdotal collapse reports.
2. **Move the Part 1 Step 2 realignment into the pipeline.** Run it before sentence segmentation, and use SaT where punctuation is missing.
3. **Interpolate timestamps for words that cannot be aligned** (numerals, currency), so they do not trigger spurious speaker changes.
4. **Extend the span schema:**
   - per-word speaker labels and a winning-margin value;
   - a `boundary` field that distinguishes `sentence`, `speaker`, `speaker_repaired` and `interruption`;
   - an `is_backchannel` flag;
   - a role label with confidence instead of a bare name.

   These fields let downstream code gate on confidence rather than treating tags as soft cues by default.
5. **Stand up the Area C metric suite as a regression test** on the labelled videos plus an Earnings-21 Q&A slice.
6. **Only then A/B test the diarizer.** Try Nemotron-3-Diarization (OpenMDW, Blackwell listed), and Sortformer v2 as the CC-BY fallback. Judge them on WDER and change-F1@k, not DER.

Leave joint speaker-attributed ASR (DiCoW, multitalker Parakeet) and aligner swaps for later. They replace the most code for gains nobody has measured on interviews. Exclude DiariZen on both license and toolchain grounds.

### Claims that rest on a single source

These claims each come from one source:
- Nemotron-3-Diarization's DER figures (vendor model card)
- DiarizationLM-Gemma-4-E4B's results (community model card)
- SE-DiCoW's 52.4% reduction
- the training-free role-refinement result (one study, 10 recordings)
- the multitalker-Parakeet GGUF cpWER figures (community card)
- the interviewer/customer LM's 84.29% improvement
- backchannels treated as independent utterances
- the WhisperX speaker-count collapse reports
- voxmap-studio's annotation-cost finding
- Geng et al.'s 56.4% speaker-confusion share
- LongMemEval's derived-key result
- Merola & Singh's contextual-retrieval replication
- CrisperWhisper's timing figures (self-reported)

A further claim, a student project's finding that diarization errors did not affect summary quality, was **unverifiable** (the PDF returned 404) and is excluded.

## Conclusion

The literature is organised around diarizers and fine-tuned correctors scored on telephone speech. Your problem sits in neither. It sits in a dozen lines that match word timestamps to speaker segments without asking whether a sentence has ended. That is good news: the highest-leverage fix costs almost nothing and is fully inspectable. It also means published gain figures transfer poorly. The 45–56% WDER reductions belong to models trained on unpunctuated Fisher text, and the same LLMs used zero-shot made things worse. **The decisive asset is a two-hour labelled set from your own videos, not a better model.**

Speaker repair and the earlier rechunking work turn out to be the same project seen from two ends. False speaker changes create the fragments that skew span lengths and the misattributed questions that mislead the coder. So repairing speakers comes before rechunking, and the speaker label belongs in what the coder reads, with its confidence, rather than in what the retriever embeds. One gap remains in the literature: no published study ties diarization accuracy to retrieval outcomes. Your paired EXP-009 comparison of repaired and unrepaired spans would be among the first measurements of it.
