---
id: T-0003
type: task
title: "Transcript rechunking for comment-to-segment linking"
status: done
requester: user
received: 2026-11-03T01:40+09:00
finished: 2026-11-03T01:49+09:00
deliverable: R-0005
tags: [transcript-chunking, comment-linking, link-gating]
---

# Transcript rechunking for comment-to-segment linking

## Request

T-48 Part 4 (backfilled 2026-11-11): re-chunk ASR transcript spans so comment-to-segment linking and the context shown to the LLM improve; cover chunking, ko/en re-segmentation, matching vs shown unit, and link gating calibrated on ~100 labels.

Requester brief, migrated from `00_requester_context.md` (Requester brief (commensense_main, 2026-11-03): T-48 Part 4):

Goal: re-chunk video transcripts so comment-to-segment linking and the context shown to the LLM work better. Parts 1-3 (R-0002, R-0003, R-0004) are already folded into EXP-007's design.

WHY: In C6 each comment is linked to a transcript chunk; chunk text goes into decision and open-coding prompts. Sometimes decisive (a distinctive shared phrase made sarcasm readable), sometimes noise. Chunks are unrefined: sentences finely split, chunk sizes vary a lot.

CURRENT SHAPE (video content only)
- Source R: (separate repo, cannot modify) gives ASR spans: start/end s, sentence-boundary flag, sometimes speaker id, section membership.
- hQR (en interview): 287 spans, 3-280 chars (median 55), 0.9-10 s (p10-p90), 273 sentence + 14 speaker boundaries, 5 sections.
- FBIwZuJuMIQ (ko): 173 spans, 8-99 chars (median 40), sentence boundaries only, no speakers.
- Current chunks copy R:'s retrieval chunks: sliding 5-sentence windows, overlap 2, never crossing a section. hQR: 94 chunks, 195-535 chars (p10-p90), max 780, 12-36 s. FBIw: 56 chunks, 166-294 chars.
- We CAN re-chunk from spans (text, times, speakers, sections). Cannot change ASR or R:'s segmentation.
- Linking: embed comment + chunks in one space (BGE-M3 now; Qwen3-Embedding-4B likely next), nearest above cosine 0.5. Median cosine 0.58; 210/212 comments linked.
- Planned gating: (1) explicit timestamp in comment wins; (2) else link only if best chunk distinct: cosine threshold + margin over 2nd + IDF-weighted word overlap as support; (3) later, LLM requests context when it marks a comment context-dependent.
- Thresholds from CE-08: user labels ~100 links right/wrong.

QUESTIONS
- Chunk construction for matching SHORT comments to SPOKEN content: token/time-normalized vs sentence-count windows; semantic segmentation (TextTiling, embedding breakpoints); interview-aware (Q-A pairs, speaker turns); proposition chunking (Dense X); multi-granularity/small-to-big; late chunking; contextual chunk headers.
- ASR sentence re-segmentation and punctuation restoration, esp. Korean (kss etc.) and English.
- Should MATCHING unit differ from SHOWN unit; how big the shown unit (budget ~300-500 tokens of segment in a 4k local context).
- Distinctiveness signals beyond cosine margin (BM25/IDF overlap, hybrid), combining + calibrating with ~100 labeled links.

DELIVERABLE: part4.md (now R-0005), same format as part3.md (R-0004): shortlist <=3 with why; fit to constraints (local 16 GB GPU, Python 3.12 + uv, pinned versions, Korean + English, no comment text to external services, llama.cpp local Qwen3.8-27B GGUF); smallest experiment on their data hooking into CE-08's ~100 labeled links; sources with links + dates; unverified items marked. Concise; shortlist first.

## Scope

Sub-questions: chunk construction (N-0012), ko/en re-segmentation and punctuation (N-0013), link gating and calibration (N-0014). Report: R-0005. Builds on T-0002 parts 1-3.

## Outcome

- Separate matching unit (1-3 re-segmented sentences, ~40-120 tokens) from shown unit (centered window).
- Gate links with a 3-4-feature L2 logistic regression calibrated on ~100 labels, not a cosine threshold.
- Re-segment by merging over-split spans (SaT; Kiwi for Korean) before building either unit.
- Open: almost no Korean evidence; all FBIwZuJuMIQ recommendations are hypotheses for CE-08 to confirm.
