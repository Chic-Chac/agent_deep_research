---
id: T-0002
type: task
title: "Issue-level embedding and clustering of short YouTube comments"
status: done
requester: user
received: 2026-11-02T23:25+09:00
finished: 2026-11-02T23:34+09:00
deliverable: [R-0002, R-0003, R-0004]
tags: [issue-normalization, embeddings, clustering]
---

# Issue-level embedding and clustering of short YouTube comments

## Request

T-48 Parts 1-3 (backfilled 2026-11-11): preprocessing/representation, embedding + clustering, and evaluation for issue-level clustering of ko/en YouTube comments on a local 16 GB GPU; one shortlist report per part.

Requester brief, migrated from `00_requester_context.md` (Requester context (from commensense_main, 2026-11-02): prior experiment results):

EXP-005 (hQR video, 263 English comments, nearest-option zero-shot):
- Qwen3-Embedding-4B and a Qwen 2B VL embedder beat BGE-M3 by 8–10 points on topic assignment. The 2B equals the 4B.
- Top-1 is 0.89–0.93 on the most confident half and 0.40–0.55 on hard items.
- Claims: embeddings are weak (top-1 ≈ 0.4). Whether a comment argues a specific claim is not a nearness question. A local LLM choosing among options reaches 0.82–0.86.
- Probes on ~200 labels do not beat zero-shot.

EXP-005 labeling lesson (the reason for the faceted codebook):
- LLM-named topic sets mixed axes: target person, issue, meme, polarity.
- Embedding clusters partly grouped by polarity and target, not issue.

EXP-006 (focus facet):
- People and content co-occur in ~25% of content comments. Unrelated-topic comments are ~1.5%.
- Interview format: many comments judge a person by seizing on what they said, so separating issue vs person vs stance is central.

Korean comments are about half of the corpus, but all numbers so far are English. Flag Korean-specific evidence.

Deliverable requirements (from the brief):
- Three files, part1.md (preprocessing/representation; R-0002), part2.md (embedding + clustering; R-0003) and part3.md (evaluation; R-0004), in English.
- Each file: a ranked shortlist of at most 3 options with 1–2 sentences on why each works; fit to constraints (local 16 GB GPU, Python 3.12 + uv, pinned model versions, Korean + English, no comment text to external services); the smallest experiment that would confirm it on their data; sources with links and dates; unverified items marked.
- Concise: the reader reads the shortlist first and the rest on demand.
- Prior attempts: BGE-M3, Qwen3-Embedding-4B, a k-means sampler + LLM-named topics, local Qwen3.8-27B GGUF via llama.cpp, and a calibrated choice model.

## Scope

Three parts, one report each. Part 1 preprocessing/representation: N-0006, N-0007, N-0008 -> R-0002. Part 2 embedding + clustering: N-0009, N-0010 -> R-0003. Part 3 evaluation: N-0011 -> R-0004.

## Outcome

- Part 1 (R-0002): embed an LLM-written, stance-neutral, role-anonymized English issue gist; keep stance and target person in separate, unembedded fields.
- Part 2 (R-0003): keep Qwen3-Embedding-4B with an issue-focused instruction; over-cluster with cosine average-linkage agglomerative clustering, then merge with human review.
- Part 3 (R-0004): rank variants by pairwise "same issue?" F1 (local LLM judge + human subset, PPI++); codebook reliability via Krippendorff's alpha; stability as a gate only.
- Open: no public Korean clustering numbers for any candidate embedder; LLM-rewrite gains are English-only; 27B judge must pass Alt-Test on own data.
