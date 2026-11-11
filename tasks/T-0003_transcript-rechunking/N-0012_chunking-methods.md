---
id: N-0012
type: note
task: T-0003
title: "Transcript chunking methods for short-comment to spoken-segment linking (and choice of the unit shown to the LLM)"
status: final
created: 2026-11-03
tags: [transcript-chunking, retrieval]
---

# Transcript chunking methods for short-comment to spoken-segment linking (and choice of the unit shown to the LLM)

Scope: chunk construction for matching 5-30-word YouTube comments (ko/en) to ASR transcript segments (hQR en interview, 287 spans; FBIw ko, 173 spans), BGE-M3 now and Qwen3-Embedding-4B next; shown-unit budget of 300-500 tokens in a 4k local context. Notes compiled 2026-11-03. Where a claim has no source, it appears under Inferences or Gaps. **No Korean-specific chunking evidence was found in any source below.** Every quantitative result cited is for English (or for multilingual benchmarks with no Korean breakdown).

## Q1. Token- or time-normalized windows vs sentence-count windows; chunk size effects for short queries; length bias in cosine

### Takeaway
For short, localized queries, the evidence favors small matching units (about 64-200 tokens). The best size depends on the embedder: decoder embedders like larger chunks, encoder embedders like smaller ones. Cosine similarity depends measurably on text length, so the current sentence-count windows (195-780 chars, roughly 4x length variation) add length-driven score noise. A single absolute cosine threshold is fragile under that noise. Length-normalized windows are the safer default.

### Cited Findings
- Bhat et al., "Rethinking Chunk Size for Long-Document Retrieval" (arXiv 2505.21700, May 2025). Fixed token chunks of 64/128/256/512/1024 with no overlap, tested on 6 QA datasets. Datasets with concise, localized answers did best with small chunks. SQuAD Recall@1 was 64.1% at 64 tokens and fell to 38.5% at 1024 tokens (Stella). Datasets with dispersed answers did best with large chunks: TechQA rose from 4.8% at 64 tokens to 61.3% at 512; NarrativeQA peaked at 1024 — [arXiv 2505.21700](https://arxiv.org/html/2505.21700v2)
- Same paper: model bias matters. "Stella [decoder-based] benefits from larger chunks... Snowflake [encoder] performs better with smaller chunks, excelling at fine-grained, entity-based matching." On COVID-QA, Stella peaked at 64 tokens (52.1%) while Snowflake peaked at 1024 (54.2%) — [arXiv 2505.21700](https://arxiv.org/html/2505.21700v2)
- Chroma, "Evaluating Chunking Strategies for Retrieval" (Smith & Troynikov, 2024-07-03). 472 queries over 5 corpora, text-embedding-3-large, top-5 retrieved. Shrinking chunks from 800 to 200 tokens raised precision/IoU about 5x with comparable recall. Recursive splitter at 200 tokens with no overlap: recall 88.1%, precision 7.0%, IoU 6.9%. Reducing overlap improves IoU because overlap adds redundant tokens — [Chroma research](https://www.trychroma.com/research/evaluating-chunking)
- Jina, "On the size bias of text embeddings" (2025-04-16), jina-embeddings-v3. Average pairwise cosine was 0.254 for sentences (about 27 words), 0.343 for documents (about 119 words), and 0.658 for concatenated long documents. Asymmetric query/document encoding reduced the bias but did not remove it: the sentence-vs-document gap fell only to 0.076. Their conclusion: "you can't use minimum cosine thresholds at all to estimate how good a match is"; verify with a reranker or LLM instead — [Jina blog](https://jina.ai/news/on-the-size-bias-of-text-embeddings-and-its-impact-in-search)
- "Length-Induced Embedding Collapse": embeddings of longer texts cluster together, so cosine between long texts rises with length and long texts become less distinguishable from one another — [arXiv 2410.24200](https://arxiv.org/html/2410.24200v1)
- BEIR observes that cosine-similarity retrievers prefer shorter documents and dot-product retrievers prefer longer ones — [BEIR, arXiv 2104.08663](https://arxiv.org/pdf/2104.08663)
- TREC Podcasts 2020 (spoken-content retrieval) used fixed time windows: 2-minute segments starting every minute (1-min overlap), averaging 340 ± 70 words. This is time normalization, not sentence normalization — [TREC 2020 Podcasts overview, arXiv 2103.15953](https://ar5iv.arxiv.org/html/2103.15953)

### Inferences
- Comments are short, mostly refer to one moment ("he said X", a quoted phrase), and are close in kind to SQuAD-style localized queries. That points to matching units at the small end, roughly 40-120 tokens or 1-3 sentences, rather than 5-sentence windows. BGE-M3 is an XLM-R encoder, which by the Bhat et al. pattern should favor smaller chunks. Qwen3-Embedding-4B is decoder-based and may tolerate or prefer larger ones. **Chunk size should be re-tuned when the embedder changes.**
- The current hQR windows run 195-780 chars, so length varies about 4x. The size-bias results imply that long windows get systematically different cosine distributions from short ones. That makes "cosine > 0.5" and "margin over 2nd" partly a function of chunk length. Fixes: (a) build windows to a token or character target (e.g., about 60-100 tokens, closing at the nearest sentence boundary), or a time target (e.g., about 15-20 s) for speech; or (b) z-score each candidate's cosine against that chunk's similarity to a background set of unrelated comments, a per-chunk normalization.
- Time normalization uses the speech rate as a proxy for content amount. It differs from character normalization mainly when the speaking rate changes. With small data, character or token targets snapped to sentence boundaries are simpler and keep units linguistically clean.
- The median cosine of 0.58 with 210/212 comments linked suggests the 0.5 threshold almost never rejects. This is consistent with Jina's warning about absolute thresholds.

### Gaps
- No study found that evaluates chunk size for user-comment queries against spoken transcripts, or for Korean text.
- No direct study found that compares time-normalized and token-normalized windows on ASR transcripts.
- The size-bias figures come from jina-v3. Magnitudes for BGE-M3 and Qwen3-Embedding were not found and should be measured locally (cheap at this scale).

## Q2. Semantic segmentation (TextTiling, embedding breakpoints, LumberChunker, Meta-Chunking, YTSeg)

### Takeaway
On natural (non-stitched) documents, semantic chunking does not consistently beat fixed-size chunking. It wins mainly when topics shift abruptly. Spoken-transcript topic segmentation is still hard (F1 about 43-46 on YTSeg), so its boundaries are noisy. Semantic segmentation is better used to define larger shown or context units (topics or sections) than to build the matching unit.

### Cited Findings
- Qu, Tu & Bao (Vectara/UW-Madison, arXiv 2410.13070, Oct 2024), embedder stella_en_1.5B_v5. On artificially "stitched" documents, breakpoint semantic chunking won by a wide margin: Miracl F1@5 81.89 vs 69.45 fixed; NQ 63.93 vs 43.79. On natural documents, fixed-size matched or won: HotpotQA 90.59 fixed vs 87.37 breakpoint vs 84.79 clustering; MSMARCO 93.58 vs 92.23 vs 93.18. Evidence retrieval differences were negligible, and fixed-size was best on 3 of 5 datasets. Conclusion: "computational costs associated with semantic chunking are not justified by consistent performance gains" — [arXiv 2410.13070](https://arxiv.org/html/2410.13070v1); [Vectara blog](https://www.vectara.com/blog/is-semantic-chunking-worth-the-computational-cost)
- Chroma (2024-07): the Kamradt embedding-breakpoint chunker (the LangChain SemanticChunker approach) at its default (about 660 tokens) did worst: recall 83.6%, IoU 1.5%. ClusterSemanticChunker with a 200-token cap had the best precision and IoU (8.0%). The LLM-based chunker (GPT-4o) had the best recall (91.9%) but low IoU (3.9%) — [Chroma research](https://www.trychroma.com/research/evaluating-chunking)
- Meta-Chunking (arXiv 2410.12788) reports Margin-Sampling chunking slightly above LumberChunker: 2WikiMultihopQA F1 13.78 vs 13.28; MultiFieldQA-en 33.98 vs 33.09. Perplexity chunking took 2.15 s vs LumberChunker's 5244.91 s on 2Wiki. These are small gains on QA F1 — [Meta-Chunking arXiv 2410.12788](https://arxiv.org/html/2410.12788v3)
- YTSeg (Retkowski & Waibel, EACL 2024): 19,299 English YouTube videos, with chapters as segment labels. Average segment length is 21.5 ± 34.2 sentences, and documents average 196 sentences. MiniSeg scored F1 43.37, Pk 28.73 (45.81 F1 after Wiki-727K pre-training). The paper notes that closed captions "do not always respect sentence boundaries." Pre-training on YTSeg helps on QMSum meetings — [EACL 2024 / arXiv 2402.17633](https://ar5iv.labs.arxiv.org/html/2402.17633); [HF dataset](https://huggingface.co/datasets/retkowski/ytseg)

### Inferences
- The requester's videos already have R: section membership, five sections for hQR. YouTube chapter-level segments average about 21 sentences, far larger than a matching unit. Topic segmentation therefore duplicates what sections give and is too coarse for matching. One defensible use is as hard boundaries: never let a matching or shown window cross a topic or section boundary, which is already done.
- A cheap local variant worth testing is a breakpoint chunker with a size cap (Chroma's ClusterSemantic at 200 tokens did best). Compute BGE-M3 cosine between adjacent sentences, cut at low-similarity valleys, and enforce min/max lengths. The evidence suggests it will roughly tie fixed windows, not beat them clearly.
- LumberChunker and Meta-Chunking need an LLM pass. That is feasible locally at this scale, but published gains are small and English/Chinese only.

### Gaps
- No Korean evidence for semantic chunking or topic segmentation found. YTSeg is English only.
- TextTiling on ASR transcripts vs fixed windows for retrieval (not segmentation F1): no direct comparison found within budget.

## Q3. Interview-aware chunking (speaker turns, Q-A pairs) and podcast/meeting retrieval

### Takeaway
The spoken-retrieval benchmarks (TREC Podcasts) used plain fixed time windows with overlap. Gains there came from hybrid lexical plus neural ranking, not from smarter segmentation. No found study isolates the gain from speaker-turn or Q-A chunking for retrieval.

### Cited Findings
- TREC 2020 Podcasts segment retrieval: 2-min segments with 1-min overlap, 50 topics, 7 groups, 24 runs. BM25 baseline nDCG 0.52 / P@10 0.49. The best run, UMD_IR_run3, reached nDCG 0.67 / P@10 0.60 using BM25 plus neural reranking. Some BERT-only runs underperformed (nDCG about 0.43). Full transcripts far outperformed metadata only (nDCG 0.58 vs 0.22-0.32) — [TREC 2020 Podcasts overview](https://ar5iv.arxiv.org/html/2103.15953); [NIST results appendix](https://trec.nist.gov/pubs/trec29/appendices/podcastRetrieval.html)

### Inferences
- In hQR, only 14 of 287 boundaries are speaker boundaries, so turns are long, which makes turn-level chunking too coarse as the matching unit. Use speaker-change and Q-A structure for two things instead: (a) as preferred cut points when building length-normalized windows; (b) as metadata in the shown unit ("Interviewer asks: ... / Guest answers: ..."). Comments in an interview often react to an answer, and the question frames the answer's meaning, so the shown unit should include the preceding question when the match falls in an answer. This is reasoned, not measured.
- FBIw has no speaker IDs, so interview-aware chunking does not apply there.
- The TREC result that lexical plus neural beats neural alone supports the planned IDF-overlap and BM25 support signal. BGE-M3 sparse weights are a local way to get this (see Q6).

### Gaps
- TREC 2021 Podcasts details (reranking plus audio features) were not fetched.
- No paper found that measures the retrieval gain from Q-A pair chunking in interviews or meetings. QMSum and MeetingBank are query-focused summarization benchmarks, not chunk-granularity studies.

## Q4. Proposition chunking (Dense X Retrieval)

### Takeaway
Propositions (about 11 words) gave large gains for unsupervised and out-of-domain retrievers and for long-tail entities. Gains for strong supervised retrievers were small. The method needs an LLM rewrite step and was tested only on English Wikipedia. For conversational ASR it is risky (hallucinated or paraphrased propositions lose the verbatim phrasing that comments echo), but it can be run with the local LLM at this scale.

### Cited Findings
- Chen et al., "Dense X Retrieval" (arXiv 2312.06648, Dec 2023; EMNLP 2024). Average unit lengths: passage 58.5 words, sentence 21.0, proposition 11.2. Contriever (unsupervised): Recall@5 +9.3 points (+22.5% relative); Recall@20 +7.7 on average. GTR (supervised) Recall@20 +2.7 on average. DPR on unseen EntityQuestions gained about 25% relative. The advantage is largest for less common entities — [arXiv 2312.06648](https://arxiv.org/html/2312.06648v3)
- At a fixed 100-token reader budget (LLaMA-2-7B), propositions beat passages for all 4 retrievers (GTR +2.8 EM), because of a "higher density of question-related information" — [arXiv 2312.06648](https://arxiv.org/html/2312.06648v3)
- The propositionizer was Flan-T5-Large fine-tuned on 42,857 GPT-4-generated pairs (dev F1 0.822). Limitations: English Wikipedia only, dual encoders only — [arXiv 2312.06648](https://arxiv.org/html/2312.06648v3)

### Inferences
- The current ASR spans (median 55 chars en / 40 chars ko) are already near proposition granularity, but they are not decontextualized (pronouns, ellipsis). A local Qwen LLM could rewrite each sentence into standalone statements (resolving "he" and "that"). The rewrite would only be embedded as an extra matching key, while linking and display stay on the original span. This preserves verbatim phrasing for IDF overlap while adding a decontextualized vector. Use max-sim over {raw span, rewritten span}.
- The "fixed token budget" result supports a broader point for the shown unit: denser, more relevant context beats longer context.

### Gaps
- No multilingual or Korean proposition-chunking evidence found. Korean pro-drop (omitted subjects) makes decontextualization both more valuable and more error-prone. Unverified.
- No evidence found on propositionizing conversational or ASR text.

## Q5. Multi-granularity / small-to-big / parent-document retrieval; late chunking

### Takeaway
"Match small, show big" is well supported in principle. Small units maximize precision (Chroma, Bhat), and readers benefit from context. Mix-of-Granularity learns per-query granularity but needs training data the requester does not have. Late chunking gives about 2-4% relative nDCG gains, mostly for small chunks, but was validated only on mean-pooled bidirectional encoders. It is a poor fit for last-token-pooled Qwen3-Embedding and unverified for BGE-M3.

### Cited Findings
- Mix-of-Granularity (MoG; arXiv 2406.00456, COLING 2025): a trained router weights snippets from several granularity levels per query. Reported as "significantly enhancing" downstream RAG performance in a medical QA setting; the router needs training — [arXiv 2406.00456](https://arxiv.org/pdf/2406.00456)
- Late chunking (Günther et al., Jina, arXiv 2409.04701, Sept 2024): encode the full document, then mean-pool token embeddings per chunk. Average nDCG@10 gain on BeIR: +3.63% relative (+1.9 abs) with sentence boundaries, +3.46% fixed-size, +2.70% semantic. Example, SciFact with jina-v3: 71.8 → 73.2. Gains are larger for small chunk sizes. With large chunks on synthetic data, naive chunking sometimes tied or won. Models tested: jina-v2-small, jina-v3, nomic-embed-v1. **BGE-M3 was not tested.** It requires long-context models with mean pooling — [arXiv 2409.04701](https://arxiv.org/html/2409.04701v3)
- Qwen3-Embedding-4B uses last-token pooling, 32k context, MRL dimensions 32-2560, and instructions in the form `Instruct: {task}\nQuery:{query}` (instructions reported to give +1-5%). Released 2025-06-05 — [HF model card](https://huggingface.co/Qwen/Qwen3-Embedding-4B)
- A third-party implementation (lc-rag, Rust crate) notes that late chunking "works worst on... causal, last-token-pooled Qwen3 models" because the method assumes bidirectional mean-pooled encoders. This is a secondary, non-peer-reviewed source — [lc-rag docs](https://docs.rs/lc-rag/0.22.4/src/lc_rag/late_chunking.rs.html)
- BGE-M3 supports 8192 tokens and dense, sparse (lexical weight) and multi-vector (ColBERT) outputs; hybrid retrieval is recommended — [HF model card](https://huggingface.co/BAAI/bge-m3). The model card does not state the dense pooling method. My recollection that BGE-M3 dense uses the CLS token is **unverified**. If it is CLS, mean-pooling token states for late chunking would not match the training objective.

### Inferences
- A practical small-to-big design for this data, built in place of LlamaIndex (which is not needed at this scale):
  - Matching unit: 1-3 ASR sentences, length-normalized (about 40-120 tokens). Optionally use several granularities in the index (single sentence, 3-sentence window), take max-sim per anchor sentence, and treat agreement across granularities as a distinctiveness feature.
  - Shown unit: centered on the matched anchor, expanded with neighbor sentences to a token budget, clipped at section boundaries, with speaker labels and timestamps kept (see Q8).
  - This is the "sentence-window retrieval" pattern popularized by LlamaIndex. The LlamaIndex docs were **not fetched; unverified** citation.
- A cheap way to get contextual vectors without depending on late chunking: the transcript (about 287 spans, a few thousand tokens) fits in one BGE-M3 (8k) or Qwen3 (32k) pass. Late chunking can therefore be tried directly as an A/B: mean-pool the hidden states over each span's tokens, and compare against naive per-chunk embedding on CE-08 labels. Expect a small gain at best, with BGE-M3 more plausible than Qwen3. The more promising way to add context is the textual one (Q6).
- The 4k LLM context is not a constraint on matching, only on what is shown.

### Gaps
- No published late-chunking results for BGE-M3 or Qwen3-Embedding found.
- No Korean small-to-big evidence.

## Q6. Contextual chunk headers / Anthropic Contextual Retrieval; hybrid lexical

### Takeaway
Prepending a 50-100-token, LLM-written situating context cut retrieval failures by 35%. Adding BM25 brought the reduction to 49%, and adding a reranker brought it to 67%. At this scale it is cheap to do with a local LLM. For comment matching, however, the added context can blur distinctiveness between neighboring chunks, so it should be an extra vector, not a replacement.

### Cited Findings
- Anthropic "Contextual Retrieval" (2024-09-19). Contextual embeddings alone: top-20 retrieval failure rate 5.7% → 3.7% (−35%). Plus contextual BM25: 2.9% (−49%). Plus reranking: 1.9% (−67%). Context is 50-100 tokens from the prompt "give a short succinct context to situate this chunk within the overall document..."; cost $1.02 per million document tokens with prompt caching (Claude 3 Haiku); top-20 retrieval was best — [Anthropic engineering](https://www.anthropic.com/engineering/contextual-retrieval)
- TREC Podcasts: the best runs were BM25 plus neural reranking (nDCG 0.67 vs BM25 0.52) — [TREC 2020 overview](https://ar5iv.arxiv.org/html/2103.15953)
- Jina recommends combining embeddings with rerankers or LLMs rather than relying on cosine thresholds — [Jina blog](https://jina.ai/news/on-the-size-bias-of-text-embeddings-and-its-impact-in-search)

### Inferences
- Anthropic's benchmark is document QA, where many chunks lack their document's identity, so context headers help. In a single-video transcript every chunk shares the same document context, so a generic header ("This is from an interview about X") adds the same tokens to every chunk. That shrinks between-chunk cosine differences and hurts the margin gate. If used, headers should be local and specific: section title, speaker role, and the question being answered. A cheap, deterministic version needs no LLM: "[Section 3 | Interviewer question: '...'] span text".
- The local Qwen GGUF can generate per-chunk context for about 100-300 chunks in minutes. Running over transcripts stays local; comment text never leaves the machine.
- Hybrid lexical support: BGE-M3's sparse lexical weights come from the same forward pass and handle Korean tokenization via XLM-R subwords. This avoids building a Korean BM25 analyzer, though a Kiwi/mecab-tokenized BM25 is a local alternative. This sits alongside the planned IDF overlap.

### Gaps
- No evidence on contextual headers for single-document (single-video) retrieval, where all chunks share one context.
- No Korean evidence.

## Q7. Time-sync comment / danmaku / comment-to-moment literature

### Takeaway
Work on aligning comments to video moments exists, but none of it found reports transcript-granularity choices. The closest item, ComVi (CHI 2026), maps YouTube comments to timestamps via audio-visual correlation; details of its segmentation were not available in the abstract.

### Cited Findings
- ComVi (Kim, Lee, Noh; CHI 2026; arXiv 2603.26173, 2026-03-27) maps each YouTube comment to semantically relevant timestamps via audio-visual correlation, then schedules display. The user study (n=32) was preference-focused: 71.9% chose ComVi — [arXiv 2603.26173](https://arxiv.org/abs/2603.26173); [project page](https://w-dlee.github.io/comvi). The authors appear to be KAIST/Korea-based (inferred from LinkedIn search hit; unverified). Whether Korean data was used is unknown.
- An aligned variational autoencoder (ECNU) matches danmaku to video storylines in a shared latent space — [ECNU pure](https://pure.ecnu.edu.cn/en/publications/aligned-variational-autoencoder-for-matching-danmaku-and-video-st/)

### Inferences
- The ComVi full paper is the most relevant one to read next: same task (non-time-synced YouTube comments to moments), possibly Korean authors.

### Gaps
- ComVi's window length, embedder and alignment accuracy were not retrievable from the abstract page. The full PDF was not fetched within budget.
- No danmaku paper found that reports transcript-chunk granularity effects.

## Q8. Shown unit: centered on match ± neighbors? How large for LLM classification/stance?

### Takeaway
Show a compact, match-centered window. Topically related but non-supporting context actively hurts LLMs, and performance degrades with length even on simple tasks. A shown unit of about 150-300 tokens (match plus 1-2 sentences either side, plus the framing question in interviews) is better supported than filling the 300-500-token budget by default. Keep the matched span visually marked, and do not show a second-best chunk unless it is distinct.

### Cited Findings
- Cuconasu et al., "The Power of Noise" (SIGIR 2024): high-scoring but non-relevant retrieved documents hurt LLM accuracy. Random documents did not hurt and even raised accuracy by up to 35%. The most harmful distractors are related but not relevant — [arXiv 2401.14887](https://arxiv.org/pdf/2401.14887)
- Chroma "Context Rot" (2025-07-14), 18 LLMs including Qwen3: even a single distractor reduced accuracy, and distractors' effects are non-uniform. Performance degraded with input length even on simple tasks. Large gaps between focused (about 300 tokens) and full (about 113k tokens) LongMemEval inputs. Models did worse when the haystack had coherent logical flow than when it was shuffled — [Chroma research](https://www.trychroma.com/research/context-rot)
- Lost-in-the-middle is mainly a long-context effect. One 2024 analysis (secondary summary, original paper not identified; **unverified**) says it is not the main cause of differences for inputs of about 300 tokens or less — [search-result summary, Arize recap](https://arize.com/blog/lost-in-the-middle-how-language-models-use-long-contexts-paper-reading/)
- Dense X: at a fixed 100-token budget, denser relevant units beat longer passages for the reader — [arXiv 2312.06648](https://arxiv.org/html/2312.06648v3)

### Inferences
- The requester's experience that context is "sometimes decisive (shared phrase), sometimes noise" matches the Power of Noise / Context Rot pattern. Neighboring sentences from the same video are exactly "related but not relevant" distractors. A recommended design:
  - Center on the matched span(s), marked (e.g., `>>> ... <<<`), with timestamps.
  - Expand symmetrically by whole sentences to about 150-250 tokens, and up to about 400 only when the LLM flags the comment as context-dependent (the planned step 3).
  - In interviews, prepend the interviewer question that governs the answer even when it is outside the window.
  - Never cross section boundaries.
  - When the link is not distinct (low margin), show no chunk, or show the video summary only, rather than a likely-wrong chunk.
- At 300-500 tokens in a 4k context, lost-in-the-middle is not the main risk. Relevance and distractor content are, so a centered window is fine positionally.
- **The matching unit should differ from the shown unit.** Matching wants small, length-normalized, distinctive units (Q1, Q4, Q5). Display wants coherent context (Q3, Q8). Decoupling them also removes the current coupling where the 5-sentence window serves both roles badly.

### Gaps
- No study found on the optimal context-window size for stance or sarcasm classification of short social-media comments with an attached source passage. This needs a local ablation: shown sizes 0 / ±1 / ±2 sentences / ±400 tokens on CE-08 labels plus LLM decisions.
- Original source for the "≤300 tokens, lost-in-the-middle not primary" claim not verified.

## Cross-cutting candidate shortlist (for the report writer; derived from the above)
1. **Small length-normalized matching units, decoupled from a match-centered shown window.** Anchor sentences merged to about 40-120 tokens (or about 8-20 s), never crossing sections or speaker turns where avoidable. Index at two granularities (anchor, 3-sentence) with max-sim. Show anchor ±neighbors to about 150-250 tokens plus the governing interview question. Strongest evidence: Bhat 2025, Chroma 2024, Jina size bias, Power of Noise, Context Rot.
2. **Hybrid distinctiveness:** dense cosine z-normalized per chunk (or rank-based), plus BGE-M3 sparse/lexical or IDF overlap, plus a margin feature, calibrated with logistic regression on CE-08's ~100 labels. Evidence: TREC Podcasts (hybrid best), Anthropic (BM25 adds 14 points of failure-rate reduction), Jina (no absolute thresholds).
3. **Optional extra keys:** a local LLM decontextualized rewrite per anchor (Dense X style) and a deterministic local header (section/question), each as an additional vector rather than a replacement. Late chunking only as a cheap A/B on BGE-M3, not on Qwen3.
- Semantic or topic chunking: low priority given the Vectara and Chroma evidence; existing sections already supply topic boundaries.
