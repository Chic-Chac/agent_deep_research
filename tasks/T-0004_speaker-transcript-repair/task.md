---
id: T-0004
type: task
title: "Speaker-attributed transcript repair"
status: done
requester: user
received: 2026-11-03T22:57+09:00
finished: 2026-11-03T23:06+09:00
deliverable: R-0006
tags: [speaker-diarization, transcript-repair, comment-linking]
---

# Speaker-attributed transcript repair

## Request

T-48 Part 5 (backfilled 2026-11-11): fix mid-sentence speaker changes and misattributed questions/interjections in WhisperX/pyannote-style spans; shortlist per area A extraction, B post-hoc repair, C evaluation, D downstream; recommend repair now and extraction changes for Ruminatai.

Requester brief, migrated from `00_requester_context.md` (Requester context (T-48 Part 5, from commensense_main, 2026-11-03)):

Ruminatai: YouTube video -> ASR with word timestamps (ids like w393) -> diarization (SPEAKER_00/01, matched to names from description w/ confidence) -> sentence spans {text, start/end, speaker_id, boundary "sentence"|"speaker", first/last word id}. Assume WhisperX/pyannote-style stack.
Spans feed comment-to-transcript linking (EXP-009): embedding retrieval + BM25 + LLM selection; selected passage shown to LLM coder.
Observed (24-min 2-person interview, 287 spans, 46 turns):
1. 14/287 spans end w/o terminal punctuation, all boundary="speaker", halves have different speakers ("So is that a five-alarm fire to" | "you?"). Change points off by 1-few words.
2. 14/33 questions attributed to guest; interjections absorbed into other speaker's sentence; tags unreliable with SFX/multi-party -> treated as soft cues.
3. Lengths p10/p50/p90 = 18/55/145 chars; 27 spans <=3 words.
Earlier EXP-009 finding: prefixing chunks with LLM context hurt retrieval when chunks shorter than the context.
Deliverable: shortlist per area A (extraction), B (post-hoc repair), C (evaluation), D (downstream): method/tool, version+date, license, runs locally on RTX 5060 Ti 16GB (WSL2, Py3.12), expected gain w/ sources, cost, integration risk. Then recommended path: (1) post-hoc repair now; (2) extraction changes for Ruminatai. Flag single-source claims.

## Scope

Areas: A extraction (N-0015 diarization models, N-0016 word-speaker assignment), B post-hoc repair (N-0017), C evaluation (N-0018), D downstream retrieval (N-0019). Report: R-0006; reuses R-0005 for chunking and gating.

## Outcome

- Defects trace to WhisperX time-overlap word-speaker assignment that ignores sentence structure.
- Now: post-hoc repair on existing spans (snap speaker changes to sentence boundaries, gated by evidence), then Part 4 merge-first re-segmentation.
- Ruminatai: pyannote community-1 with num_speakers=2 and exclusive diarization, realignment in pipeline, per-word speaker margin; A/B Nemotron-3-Diarization.
- Open: no source measures WDER before/after punctuation realignment; zero-shot LLM correction hurt attribution in two studies; no study links speaker-label accuracy to retrieval.
