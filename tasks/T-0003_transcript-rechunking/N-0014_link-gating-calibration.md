---
id: N-0014
type: note
task: T-0003
title: "Link gating and calibration: distinctiveness and confidence signals for comment-to-chunk linking (with about 100 labels)"
status: final
created: 2026-11-03
tags: [link-gating, calibration]
---

# Link gating and calibration: distinctiveness and confidence signals for comment-to-chunk linking (with about 100 labels)

Context (from requester brief): about 200-300 comments and 50-100 chunks per video. BGE-M3 median cosine is 0.58, and 210/212 comments get linked at cosine 0.5, so the threshold is far too permissive. Planned gating: explicit timestamp wins; otherwise cosine threshold + margin + IDF overlap. CE-08 will provide about 100 right/wrong labels. Videos are in Korean and English. Everything must run locally (16 GB GPU).

Research budget note: about 18 tool calls. Items not fetched from a primary source are marked UNVERIFIED.

## Q1. Which post-retrieval signals predict a correct dense link beyond raw cosine (QPP: margin, NQC/WIG/SMV, dense-specific QPP)?

### Takeaway
Score-distribution predictors (std-based NQC, WIG, top1-top2 margin) are only moderately correlated with retrieval quality for dense retrievers (Pearson around 0.3-0.35 on TREC-DL with ANCE). A small linear model over the *sorted score vector*, trained on fewer than 40 labeled examples, clearly beats any single score feature. So: use top-k score features from a supervised model, not a single hand threshold.

### Cited Findings
- On TREC-DL with the ANCE dense retriever, WIG reached Pearson r = 0.313 and NQC r = 0.350. NQC was better when the target was AP@100, WIG when it was nDCG@10 — [Meng et al., QPP: from ad-hoc to conversational search (arXiv 2305.10923, SIGIR 2023)](https://arxiv.org/pdf/2305.10923)
- NQC, the most common score-based predictor, is the standard deviation of the top-k retrieval scores, normalized. A higher spread means the top items stand out from the rest. The authors argue dense retrievers have "higher potential" to be predicted by score-based QPP because of their score distributions — [arXiv 2305.10923](https://arxiv.org/pdf/2305.10923)
- Embedding-based coherence predictors (pairwise coherence of the top-k dense document embeddings) improve QPP accuracy on dense retrieval over sparse variants: up to +92% for TCT-ColBERT and +188% for ANCE (TREC-DL). Query type explains up to 35% of the unstable predictor performance — [Vlachou & Macdonald, On Coherence-based Predictors for Dense QPP (arXiv 2310.11405, Oct 2023)](https://arxiv.org/abs/2310.11405)
- Dense-specific QPP predictors "lack generalization" to sparse rankers, and the reverse also holds — [search summary of ceur-ws Vol-3366 / related QPP analyses](https://ceur-ws.org/Vol-3366/paper-07.pdf) (snippet only; not read in full)
- Abstention for rerankers, "Towards Trustworthy Reranking" (Gisserot-Boukhlef et al., arXiv 2402.12997, Feb 2024, rev. Sep 2024):
  - Reference-free confidence functions compared: max score (u_max), std of scores (u_std), and top1-top2 margin (u_1-2).
  - A data-driven linear regression on the *sorted relevance-score vector* (u_lin) clearly outperformed them: 37.4% normalized AUC (mAP) vs 28.5% for the best reference-free method (u_std).
  - u_lin beat reference-free methods with "less than 40 samples on average" of reference data (break-even at 9-80 instances, depending on dataset).
  - Overhead was about 1.2% of scoring time.
  - Better rerankers gave better-behaved score distributions, which in turn gave better abstention.
  - [arXiv 2402.12997v5](https://arxiv.org/html/2402.12997v5)
- A 2026 study of LLM selective prediction found that combining risk signals beats an entropy-only score on both risk-coverage and calibration — [Entropy Alone is Insufficient for Safe Selective Prediction in LLMs (arXiv 2603.21172, Mar 2026)](https://arxiv.org/abs/2603.21172) (search snippet only)

### Inferences
- The setting maps onto classic QPP. Each comment is a "query" over a small corpus of 50-100 chunks, and the question is "is the top-1 correct?" That is closer to the reranking-abstention setup in 2402.12997 (one decision per query, from the score vector) than to QPP correlation studies.
- The best candidate design, given that paper's finding that fewer than 40 labels suffice:
  - Features: s1 (top cosine), s1 − s2 (margin), s1 − mean(s_top-k), the z-score of s1 against all chunk scores for that comment ((s1 − μ)/σ over the 50-100 chunks; this is cheap since the whole video is scored anyway), and std of the top-k (NQC-like).
  - Model: logistic regression on these features.
- The z-score / NQC-style features matter here because absolute BGE-M3 cosines are compressed (median 0.58). Within-comment normalization removes the per-comment offset, e.g. short generic comments that are mildly similar to everything.
- Softmax over candidates (entropy, or p(top1) = softmax(s/T)[1]) is a smooth version of margin plus spread. With a learned temperature it is roughly equivalent to margin + z features in an LR. Use one or the other, not both, to keep the feature count small.

### Gaps
- No published QPP study specifically on short social-media comments vs spoken-transcript chunks, or on Korean, was found.
- SMV (score magnitude and variance) for dense retrievers was not verified in a primary source this session.
- Exact numbers for dense-QPP papers from 2024-2025 (e.g., QPP-GenRE, BERT-QPP, the ACM TOIS 2025 survey at hal-05379351) were not read.

## Q2. Answerability / "no relevant passage" detection and entity-linking NIL: thresholding, margin-based NIL, calibration

### Takeaway
The standard practice in entity linking: link to the top candidate only if its score is at least a threshold τ learned on labeled data. Larger train-time margins improve NIL detection at some cost to recall. The key caveat is that calibration needs real positive "NIL" examples (mentions with no correct entity), not synthetic negatives. In this setting that means comments with no correct chunk.

### Cited Findings
- Traditional NIL method: a mention is unlinkable if its top score is below τ; otherwise it maps to the top entity. τ is learned from a training set — [Proxy-based Zero-Shot Entity Linking (arXiv 2301.13318)](https://arxiv.org/pdf/2301.13318); also [USFD at KBP 2011 (arXiv 1203.5073)](https://arxiv.org/pdf/1203.5073)
- Higher margins between candidates give better NIL detection, while lower-margin models do better on overall recall. Differences were small, so the margin can be set by the NIL-vs-recall trade-off — [arXiv 2012.08844, A Lightweight Neural Model for Biomedical Entity Linking](https://arxiv.org/pdf/2012.08844) (search snippet; attribution to that paper is from the search result, not verified by full read)
- "NIL calibration requires positive examples of unresolved entities, not only negative examples created by randomly pairing mentions with incorrect candidates" — [geosurge.ai article](https://geosurge.ai/corpus/gen-2950/entity-linking/entity-linking-confidence-calibration-for-ai-ready-enterprise-corpora.html) (low-authority aggregator; the principle is standard but this source is weak)
- RAG selective-answering work in 2026 (CoRM-RAG, SURE-RAG) separates "semantic relevance" from "decision safety" with a learned critic/verifier and reports steep risk-coverage curves — [arXiv 2605.01302](https://arxiv.org/pdf/2605.01302), [arXiv 2605.03534](https://arxiv.org/pdf/2605.03534) (search snippets only; UNVERIFIED details)

### Inferences
- Two kinds of wrong link need separating in the CE-08 labels:
  - (a) NIL comments: generic praise, off-topic remarks, or comments about the whole video. The correct action is no link.
  - (b) Wrong-chunk comments: they do refer to a moment, but a neighbouring chunk was picked.
- Margin mainly detects ambiguity (b), and can also *penalize* correct links when adjacent overlapping chunks share text. With sliding 5-sentence windows and overlap 2, s2 is often a neighbour of s1.
- Recommendation: compute the margin against the best *non-overlapping / non-adjacent* chunk, or treat an adjacent-chunk hit as correct for labeling purposes. Otherwise the margin feature is corrupted by the chunking itself. (Inference from the chunk design in the brief; not from a source.)
- Since 210/212 currently pass, most of the about 100 labeled links will show whether a large NIL class exists. If more than 30% of comments are NIL-like, an absolute-score feature (s1 or its z-score) will matter more than margin.

### Gaps
- No found study measures NIL rates for YouTube comments linked to transcript segments.

## Q3. Hybrid dense + sparse for short, colloquial queries and Korean; does lexical overlap catch quote-style (sarcasm) links?

### Takeaway
BGE-M3 already outputs a learned sparse (lexical-weight) score and a ColBERT multi-vector score from the same forward pass. On MIRACL-Korean, adding them to dense helps a little (dense 69.9 → dense+sparse 71.3 → all 72.2 nDCG@10). As *gating features*, lexical scores are cheap and directly target the "comment quotes a distinctive phrase" case. For Korean BM25, morpheme tokenization (kiwipiepy) is the usual choice over whitespace, but no quantitative head-to-head with char n-grams was found.

### Cited Findings
- BGE-M3 scores, all from one model:
  - dense: s_dense = ⟨e_p, e_q⟩ (normalized CLS);
  - sparse: s_lex = Σ_{t∈q∩p} w_qt·w_pt (ReLU term weights);
  - multi-vector: s_mul = (1/N) Σ_i max_j E_q[i]·E_p[j];
  - combined: s_rank = s_dense + s_lex + s_mul (equal weights in the paper).
  - [Chen et al., BGE M3-Embedding (arXiv 2402.03216; v5 Dec 2025)](https://arxiv.org/html/2402.03216v3)
- MIRACL **Korean** nDCG@10: Dense 69.9, Sparse 61.5, Multi-vec 71.2, Dense+Sparse 71.3, All 72.2. The paper says the sparse mode "outperforms the typical BM25 methods in all languages" — [arXiv 2402.03216](https://arxiv.org/html/2402.03216v3) **(Korean evidence)**
- BGE-M3 supports 100+ languages and 8192-token inputs — [arXiv 2402.03216 abstract](https://arxiv.org/abs/2402.03216)
- Korean BM25 with Kiwi morphological analysis vs default BM25, qualitative example: for "금융저축보험", default BM25 returned a document containing "금융보씨" (a name), while Kiwi-BM25 returned the relevant documents. No quantitative metrics and no pinned versions were given — [LangChain OpenTutorial, Kiwi BM25 Retriever](https://langchain-opentutorial.gitbook.io/langchain-opentutorial/10-retriever/10-kiwi-bm25-retriever) **(Korean evidence, qualitative)**
- Korean tokenization strategies studied (jamo, syllable, morpheme, subword, morpheme-aware subword) for NLP tasks other than retrieval. Morpheme-aware subwords generally did well on downstream tasks — [Park et al., An Empirical Study of Tokenization Strategies for Various Korean NLP Tasks (AACL 2020; github kakaobrain/kortok)](https://ar5iv.arxiv.org/html/2010.02534) **(Korean; not retrieval-specific)**

### Inferences
- Gating features that need no extra model:
  - BGE-M3 `lexical_weights` overlap, s_lex(top1);
  - the s_lex margin (top1 − top2 by s_lex);
  - the ColBERT score, s_mul(top1).
  - `BGEM3FlagModel.encode(..., return_sparse=True, return_colbert_vecs=True)` in FlagEmbedding gives all three. Pin the version (UNVERIFIED current version number). s_lex is a learned, IDF-like lexical match and works for Korean subwords via the XLM-R tokenizer, so it partly replaces a hand-made IDF overlap.
- An explicit IDF-weighted overlap computed per video is still worth having as one interpretable feature: the chunk corpus is the IDF corpus, so rarity *within the video* is measured. This is exactly the "distinctive shared phrase" signal.
  - Korean: kiwipiepy morphemes; keep content POS tags NN*/VV/VA/XR/SL, drop particles/endings.
  - English: lowercase + stopwords.
  - Char 2-3-grams are a robust fallback for Korean spelling variation and slang in comments (UNVERIFIED as better; no head-to-head found).
- Packages: `bm25s` (fast, numpy/scipy) or `rank_bm25`. Both accept pre-tokenized lists, so a kiwipiepy tokenizer plugs in. Exact versions to pin were not verified this session (UNVERIFIED).
  - With only 50-100 chunks per video, BM25 vs plain IDF-overlap differences will be small. Length normalization (BM25's b) matters because chunk lengths vary (195-780 chars).
- RRF is a *ranking* fusion and throws away the score magnitudes needed for gating. For a link/no-link decision, feed dense and lexical scores (or their within-comment z-scores) as separate LR features rather than fusing them first. That is effectively a learned convex combination.
- Quote-style sarcasm links: a dense model may give a modest cosine when the comment quotes a phrase while inverting its sentiment. Lexical overlap on rare n-grams should flag these strongly. No study measuring this was found; it is a hypothesis to test on CE-08 labels by checking whether the IDF-overlap feature rescues dense near-misses.

### Gaps
- No quantitative Korean BM25 tokenizer comparison (kiwipiepy vs mecab-ko vs char n-gram) in retrieval was found this session.
- No empirical work found on lexical overlap for sarcasm/quote detection in comment linking.

## Q4. Anisotropy / hubness: hub chunks attracting many comments; CSLS, per-chunk normalization, inverted softmax

### Takeaway
Hubness (a few targets being nearest neighbours of many queries) is a documented precision problem. CSLS, Inverted Softmax and QB-Norm/Sinkhorn variants fix it by normalizing each target's score by how similar it is to a bank of queries. The bank here is simply all comments of the same video, so per-chunk normalization is essentially free. The evidence comes from cross-lingual word translation and cross-modal retrieval, not from comment linking.

### Cited Findings
- "The hubness problem is a phenomenon where a small number of targets frequently appear as nearest neighbors to numerous queries, which continues to hinder the precision of similarity measurements" — [Hubness Reduction with Dual Bank Sinkhorn Normalization (arXiv 2508.02538, Aug 2025)](https://arxiv.org/html/2508.02538v1)
- The normalization methods:
  - CSLS re-normalizes similarities by subtracting local averages of similarities from a probe query bank.
  - Inverted Softmax applies softmax per gallery item over probe queries, with inverse temperature β.
  - Sinkhorn normalization balances both query and target marginals.
  - DBSN adds a target bank.
  - [arXiv 2508.02538](https://arxiv.org/html/2508.02538v1)
- Original CSLS (Conneau et al., "Word Translation Without Parallel Data", ICLR 2018): CSLS(x,y) = 2cos(x,y) − r_T(x) − r_S(y), where r is the mean cosine to the k nearest neighbours (k=10 in that paper). (From training knowledge; not re-fetched this session.)
- QB-Norm (Bogolin et al., CVPR 2022) for cross-modal retrieval, with code in the same family — [querybank normalisation overview](https://api.emergentmind.com/topics/querybank-normalisation-qb-norm); HNN code — [baidu-research/HNN](https://github.com/baidu-research/HNN) (overview only)

### Inferences
- A concrete per-video recipe:
  - For each chunk c, compute r(c) = mean cosine of c to its k most-similar comments (k ≈ 10). For each comment q, r(q) = mean cosine to its k nearest chunks.
  - Use CSLS = 2cos − r(q) − r(c) as an alternative top-1 score, or add r(c) as a "hubness penalty" feature in the LR.
  - Also log each chunk's in-degree (how many comments it wins) as a diagnostic.
- Caution: some chunks really are popular (the "moment" everyone comments on). CSLS will penalize them. Whether it helps precision must be checked on CE-08 labels; compare LR with and without r(c).
- r(q) is the per-comment analogue of the z-score in Q1. The z-score and r(q) are redundant; keep one.

### Gaps
- No evidence was found on CSLS/inverted softmax in text-to-text dense retrieval at this small scale (about 250 queries × 75 targets) or for Korean.

## Q5. Effect of chunk length on cosine scores, and normalization

### Takeaway
Longer texts systematically get higher cosine similarity to arbitrary other texts (embeddings drift toward the corpus mean). An absolute cosine threshold is therefore biased toward long chunks. Mitigations: more uniform chunk sizes, a chunk-length feature or per-chunk score centering, or verification with a cross-encoder.

### Cited Findings
- With jina-embeddings-v3 on CISI (1,460 docs):
  - sentence-sentence pairs averaged 0.254 cosine;
  - document-document pairs averaged 0.343;
  - concatenated larger texts averaged 0.658.
  - The article says cosine thresholds "cannot reliably measure absolute relevance" and recommends rerankers/LLMs for verification.
  - [Jina AI, "On the Size Bias of Text Embeddings and Its Impact in Search" (Apr 16, 2025)](https://jina.ai/news/on-the-size-bias-of-text-embeddings-and-its-impact-in-search)
- As documents get longer, their vectors converge toward the distribution's mean vector (law-of-large-numbers argument), raising pairwise cosine — [GDELT blog, "Impact of Textual Length on Embedding Similarity Part 2"](https://blog.gdeltproject.org/embedding-models-the-impact-of-textual-length-on-embedding-similarity-part-2/) (search snippet)
- A 2024 arXiv paper studies length-induced embedding collapse and proposes mitigation (attention temperature scaling) — [arXiv 2412.15241](https://arxiv.org/abs/2412.15241) (title/abstract level only; details UNVERIFIED)

### Inferences
- The current chunks range from 195 to 780 chars (hQR), so long chunks likely win more comments. Check this by regressing chunk in-degree on chunk length.
- Fixes:
  - (1) Re-chunk to a more uniform length or time span. This is the main lever and is covered by other notes.
  - (2) Add log(chunk_len) to the LR.
  - (3) Center each chunk's scores by its mean similarity to all comments. This is the CSLS r(c) term, which absorbs length bias and hubness together.
- Short comments (often under 20 tokens) vs 50-150-token chunks is an asymmetric pairing. Using BGE-M3's ColBERT score (max-sim per query token) reduces dependence on chunk length, because each comment token finds its best chunk token.

### Gaps
- No measurement of the size bias for BGE-M3 or Qwen3-Embedding specifically, or for Korean text, was found.

## Q6. Cross-encoder rerankers and LLM yes/no verifiers as link verifiers (local, 16 GB, Korean)

### Takeaway
Qwen3-Reranker-0.6B (66.36 MMTEB-R) and -4B (72.74) clearly outrank bge-reranker-v2-m3 (58.36) and jina-reranker-v2-multilingual (63.73) on the Qwen-reported multilingual benchmark. All fit in 16 GB. Qwen3-Reranker outputs P(yes) from a yes/no softmax, a natural verifier score, and accepts a task instruction. Scores are not calibrated probabilities in absolute terms; Platt-scale them on CE-08 labels.

### Cited Findings
- Qwen3-Reranker-0.6B:
  - Prompt: `<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}`. The score is the softmax over the "yes"/"no" token logits at the last position.
  - Instruction-aware; custom instructions typically give +1-5%.
  - Needs transformers ≥ 4.51.0. 32K context (8192 in practical inference), 100+ languages, Apache-2.0.
  - [HF model card Qwen/Qwen3-Reranker-0.6B](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B)
- MMTEB-R as reported by Qwen: Qwen3-Reranker-0.6B 66.36, 4B 72.74, 8B 72.94; bge-reranker-v2-m3 58.36; jina-multilingual-reranker-v2-base 63.73 — [HF card](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B); [Qwen3 Embedding tech report (arXiv 2506.05176, Jun 2025)](https://arxiv.org/html/2506.05176v3). Note these are vendor-reported numbers.
- bge-reranker-v2-m3:
  - 0.6B parameters, XLM-RoBERTa base (from bge-m3), multilingual.
  - Outputs a raw logit; `normalize=True` applies a sigmoid to map to [0,1].
  - Examples use max_length=512.
  - No calibration notes, nothing Korean-specific.
  - [HF card BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3)
- Better rerankers produce score distributions that support better abstention — [arXiv 2402.12997](https://arxiv.org/html/2402.12997v5)
- The Jina size-bias article recommends reranker/LLM verification instead of cosine thresholds — [Jina AI, Apr 2025](https://jina.ai/news/on-the-size-bias-of-text-embeddings-and-its-impact-in-search)

### Inferences
- Verifier cost is trivial at this scale. Reranking the top-3/top-5 chunks for 300 comments is 900-1500 pairs per video, seconds to a minute on a 16 GB GPU for 0.6B; 4B in bf16 (about 8 GB weights) also fits (UNVERIFIED runtime).
- Use P(yes) of top1 and the reranker margin (top1 vs top2) as LR features, or as the sole score if it dominates.
- Write the instruction for the task, e.g. "Given a viewer comment on a video, judge whether the transcript passage is the specific moment the comment refers to or quotes." Use the same English instruction for Korean pairs (Qwen recommends English instructions; UNVERIFIED for this model card).
- LLM yes/no with logprobs via the local llama.cpp Qwen3.x-27B GGUF:
  - Feasible: llama.cpp server returns token logprobs (`n_probs` / `logprobs`), so P("yes") is readable (UNVERIFIED for the exact current API field names).
  - Much slower than a 0.6B reranker. Reserve it for the uncertain band (e.g., LR probability 0.3-0.7) as a third stage.
  - Disable "thinking" for scoring.
- Korean: no Korean-specific reranker evaluation was found in this session. MMTEB-R includes Korean tasks among many languages, but no per-language number was retrieved. Flag this as a gap and check on the Korean CE-08 subset.

### Gaps
- No per-language (Korean) reranker scores were retrieved.
- No independent (non-vendor) comparison of Qwen3-Reranker vs bge-reranker-v2-m3 on Korean.
- jina-reranker-v2-base-multilingual license (CC-BY-NC, per memory) is UNVERIFIED this session.

## Q7. Combining signals and calibrating with about 100 labels (LR, Platt vs isotonic, CV, thresholds for precision targets, CIs, risk-coverage)

### Takeaway
With n ≈ 100:
- Fit a 2-4 feature L2-regularized logistic regression, which is itself a Platt-style sigmoid calibrator. Do not use isotonic; scikit-learn says it needs about 1000+ samples.
- Evaluate with repeated stratified CV or leave-one-out predictions.
- Choose the threshold from the out-of-fold risk-coverage curve.
- Report precision at the chosen threshold with a Wilson or bootstrap interval. These intervals will be wide (about ±10-15 points), so state them.
The reranker-abstention paper shows a linear model on score features beats single-signal thresholds with fewer than 40 labels.

### Cited Findings
- scikit-learn 1.9.1 calibration guide:
  - Sigmoid (Platt) is preferable for small samples and less prone to overfitting.
  - Isotonic is "more prone to overfitting, especially on small datasets" and performs as well or better only with "greater than ~1000 samples".
  - `CalibratedClassifierCV(ensemble=True)` fits k (classifier, calibrator) pairs and averages them; `ensemble=False` uses `cross_val_predict` to fit one calibrator on unbiased predictions. All classes must appear in each split.
  - A `"temperature"` method exists for multiclass.
  - [scikit-learn calibration docs](https://scikit-learn.org/stable/modules/calibration.html)
- The linear regression confidence function over sorted scores beats max/std/margin with under 40 reference samples on average (break-even 9-80) — [arXiv 2402.12997](https://arxiv.org/html/2402.12997v5)
- Combined risk scores beat single-signal (entropy) scores for risk-coverage and calibration — [arXiv 2603.21172](https://arxiv.org/abs/2603.21172) (snippet)

### Inferences
Feature budget:
- With about 100 labels and 30-60 positives (or negatives), the events-per-variable rule of thumb (about 10 events per feature) caps the model at 3-5 features.
- Candidate set:
  - (1) z-score of the top-1 dense score within the comment, or raw s1;
  - (2) margin vs the best non-adjacent chunk;
  - (3) lexical feature (BGE-M3 s_lex, or IDF overlap);
  - (4) reranker P(yes), if used.
- Standardize the features. Use `LogisticRegression(C≈1)` with L2 and compare a few C values by CV log-loss. The LR output is already a sigmoid probability, so no separate calibrator is needed.

Validation:
- Repeated stratified 5-fold (e.g., 20 repeats) or LOO to get out-of-fold probabilities for every label.
- Metrics: AUROC, Brier/log-loss, AURC (area under risk-coverage).
- Compare against the baselines "cosine only" and "cosine + margin".
- *Group* CV by video (leave-one-video-out) if there are at least 3-4 videos. It measures cross-video shift directly, which is the real deployment risk. Also report per-language results.

Threshold for a precision target:
- On the out-of-fold probabilities, sort by score and plot precision vs coverage (the risk-coverage curve).
- Choose the lowest threshold whose *lower Wilson 95% bound* on precision meets the target (e.g., ≥ 0.85). Alternatively, bootstrap the whole fit-then-threshold pipeline (about 1000 resamples) for a CI on the precision achieved.
- Wilson interval for k/n: centre (k + z²/2)/(n + z²), half-width z·sqrt(k(n−k)/n + z²/4)/(n + z²). Standard formula; not fetched this session.
- Example: 40 accepted links with 36 correct gives precision 0.90 with a Wilson 95% CI of about 0.77-0.96 (computed by formula). Expect CIs this wide.

Simpler fallback when the LR doesn't beat a 2-D rule by more than noise:
- Use a grid search over (cosine threshold, margin threshold) maximizing coverage at precision ≥ target, on the same CV folds.
- Two thresholds fitted on 100 points overfit less than people fear, but report out-of-fold numbers only.

Pitfalls:
- (a) Label shift: precision depends on the NIL rate, which varies by video and language. The Korean video has shorter spans and no speakers.
- (b) Label definition: adjacent/overlapping chunks. Decide whether "right" means the exact chunk or any chunk containing the referenced moment.
- (c) Selection bias: the labels come from links the current system made at 0.5, which is nearly all comments, so the bias is small now. It grows if labels are later drawn only from accepted links.
- (d) Tuning features on the same 100 labels inflates estimates. Freeze the feature set first, or nest the CV.

### Gaps
- No source found that gives specific guidance on calibration for retrieval linking with about 100 labels beyond the sklearn guidance and 2402.12997.
- The conformal / risk-control alternative (e.g., Learn-then-Test, conformal risk control) was not researched. It would give finite-sample precision guarantees but needs exchangeable held-out labels, which about 100 can barely support.

## Q8. Active choice of which about 100 links to label

### Takeaway
No retrieved source addresses this directly. Standard practice suggests stratified sampling over score bins and margin bins, oversampling the decision band, and balancing language and video. Then reweight to the population for precision estimates.

### Cited Findings
- No source this session; the 2402.12997 break-even of 9-80 samples suggests even 100 labels are enough for a 3-5 feature linear model if they cover the score range — [arXiv 2402.12997](https://arxiv.org/html/2402.12997v5)

### Inferences
Suggested CE-08 sampling design:
- Pool all comments from 2+ videos per language, after timestamp-based links are removed.
- Stratify on top-1 cosine quantile (5 bins) × margin tertile (3 bins) × language (2). That is 30 cells; take about 3-4 per cell, oversampling the middle bins where the threshold is likely to fall.
- Include about 10-15 comments that are probably NIL: lowest scores, generic comments.
- Record the sampling weights. For estimating population precision/coverage at a threshold, use inverse-probability weighting. For *fitting* the LR, unweighted is fine since discrimination is the goal. Recalibrate the intercept to population prevalence if needed.
- Label three-way: exact chunk correct / adjacent chunk acceptable / wrong, plus "no chunk applies (NIL)". This allows the margin and adjacency issue and the NIL rate to be estimated separately.
- If the labelling budget allows a second round, use uncertainty sampling: label the comments whose current LR probability is closest to the chosen threshold. This most improves threshold estimation.

### Gaps
- No empirical study on active label selection for threshold estimation in retrieval gating was located. The recommendations above are inferences from standard survey-sampling and active-learning practice.
