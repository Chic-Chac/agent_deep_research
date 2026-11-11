---
id: R-0005
type: report
task: T-0003
title: "Match on small sentence units, show a centered window, gate with a learned score"
status: final
created: 2026-11-03
tags: [transcript-chunking, comment-linking, link-gating]
uses: [N-0012, N-0013, N-0014]
---

# Match on small sentence units, show a centered window, gate with a learned score

*Part 4: transcript re-chunking and link gating. Research notes dated 2026-11-03. PyPI versions checked 2026-11-03.*

**Korean evidence is largely absent.** Every chunking, shown-context and gating result below comes from English or from multilingual benchmarks that report no Korean breakdown. There are three exceptions: the 2022 kss/Kiwi splitter benchmark, SaT's Korean UD/OPUS100 scores, and BGE-M3's MIRACL-ko numbers. All three are Korean, but none is ASR or comment data. Treat every recommendation for FBIwZuJuMIQ as a hypothesis that CE-08 has to confirm.

## 1. Shortlist (ranked)

1. **Use separate units for matching and for showing.** The matching unit should be small and length-normalized: 1–3 re-segmented sentences, about 40–120 tokens, no overlap. The shown unit is built separately (§4.3). For short, localized queries, small chunks win. SQuAD Recall@1 fell from 64.1% at 64 tokens to 38.5% at 1024 ([Bhat et al. 2025](https://arxiv.org/html/2505.21700v2)), and cutting chunks from 800 to 200 tokens raised precision about 5× with comparable recall ([Chroma 2024](https://www.trychroma.com/research/evaluating-chunking)). Cosine also rises with text length. On jina-v3, sentence pairs averaged 0.254 and long concatenations 0.658 ([Jina 2025](https://jina.ai/news/on-the-size-bias-of-text-embeddings-and-its-impact-in-search)). The current windows vary about 4× in length (195–780 chars on hQR), so a single cosine threshold or margin is partly measuring chunk length.
2. **Gate each link with a 3–4-feature L2 logistic regression, not a cosine threshold.** Candidate features: the within-comment z-score of the top cosine, the margin over the best unit *outside the shown window*, a lexical-overlap score, and optionally a local reranker's P(yes). Choose the threshold on out-of-fold predictions, using the lower Wilson bound on precision. A linear model over the sorted score vector beat the best single confidence signal (normalized AUC 37.4% vs 28.5%) with **fewer than 40 labels on average** ([Gisserot-Boukhlef et al. 2024](https://arxiv.org/html/2402.12997v5)). Hybrid lexical + neural ranking was the strongest family in spoken-content retrieval (TREC Podcasts nDCG 0.67 vs BM25 0.52) ([TREC 2020](https://ar5iv.arxiv.org/html/2103.15953)).
3. **Re-segment by merging, not splitting, before building either unit.** Use a greedy merge over R:'s spans that respects sections and speaker changes. Merge cues: a pause gap, a text cue that the sentence has ended (SaT for both languages; for Korean, Kiwi's sentence-final-ending tag), and minimum/maximum lengths. Korean spans (median 40 chars) and English spans (median 55) are mostly *over*-split fragments. Both kiwipiepy and SaT detect boundaries without punctuation ([kiwipiepy](https://github.com/bab2min/kiwipiepy); [SaT, EMNLP 2024](https://arxiv.org/html/2406.16678v2)). This is ranked third because it is a cheap prerequisite whose gain over R:'s own sentence flag is untested.

Not shortlisted (§4.5): semantic/topic segmentation, LLM proposition rewriting, contextual chunk headers, and late chunking.

## 2. Fit to constraints

| Constraint | How the shortlist meets it |
|---|---|
| 16 GB GPU | Re-segmentation runs on the CPU (Kiwi) or with a 3-layer SaT model, which is small (**unverified** VRAM). Embedding about 300 units plus about 300 comments takes seconds. An optional Qwen3-Reranker-0.6B scores 900–1,500 pairs per video. Its runtime is an **unverified** estimate of seconds to a minute. Run each model pass, cache the results to `.npy`, and free VRAM before llama.cpp loads. As in Part 2, the embedder cannot share the card with the 27B GGUF. |
| llama.cpp, local Qwen3.8-27B GGUF | The shown unit is 150–250 tokens by default and at most about 400 when the LLM asks for context. That leaves most of the 4k context free for the decision prompt. Measure unit length with the GGUF's own tokenizer, for example llama.cpp's tokenize endpoint (**unverified** endpoint name for the pinned build). An optional third gating stage reads the LLM's P("yes") from logprobs with thinking disabled, and only for the uncertain band (**unverified** logprob field names). |
| Python 3.12 + uv, pinned | `kiwipiepy==0.24.1` (2026-10-20, abi3 wheels, Apache-2.0). `wtpsplit==2.2.2` (2026-09-25, pure-Python wheel, MIT). `scikit-learn==1.9.1`. `llama-cpp-python==0.3.36` if you don't call the server binary. Commit `uv.lock`, and pin HF model revisions by SHA. **Avoid as defaults:** `kss==6.0.6` has no Linux wheel and pins `emoji==1.2.0`, and whether it builds on Python 3.12 is **unverified**. `spokentxt-punctuation-restoration==0.1.1` pins `transformers<5`. `punctuators==0.0.7` is stale and **unverified** on Python 3.12. Exact versions of `FlagEmbedding`, `bm25s` and `rank_bm25` are **unverified**. Qwen3-Reranker needs `transformers>=4.51.0`. |
| Korean + English | Korean: Kiwi `split_into_sents` returns character offsets and tags sentence-final endings (EF), so span ids can be mapped back. Kiwi `glue` decides whether to insert spaces when joining spans. English: SaT `-sm` models were trained on text with punctuation and casing stripped to emulate ASR. Lexical overlap for Korean uses Kiwi content morphemes (NN*/VV/VA/XR/SL). All thresholds are tuned per language, and all results are reported per language. |
| No external services | Everything runs offline. Model weights are the only download. No comment or transcript text leaves the machine. |
| Embedder switch BGE-M3 → Qwen3-Embedding-4B | **Re-tune the matching-unit size when the embedder changes.** Bhat et al. report that encoders favor smaller chunks and decoders larger ones. The pattern is not uniform: on COVID-QA it flipped. Qwen3-4B (decoder, last-token pooling) may want 2–3 sentence units where BGE-M3 (encoder) wants 1–2. **BGE-M3's free sparse and ColBERT scores disappear with Qwen3.** Keep the lexical feature independent of the embedder (per-video IDF overlap), or keep BGE-M3 loaded only for `lexical_weights`. Give comments the Qwen query instruction (`Instruct: …\nQuery:`) and embed units without one, as the model card does for documents. |

## 3. Smallest confirming experiment

**Scope.** Use hQR and FBIwZuJuMIQ and CE-08's ~100 labels, about 50 per video. Remove timestamp-linked comments before sampling. Do the steps in this order:
1. Re-segment both transcripts (§4.1).
2. Freeze the arms.
3. Collect the labels.
4. Score every arm against the same labels.

**Label the moment, not the chunk.** This is the one design choice that makes 100 labels reusable. For each comment, the annotator marks the R: span ids (equivalently, a time range) that the comment refers to, or "NIL" (generic, off-topic, or about the whole video). A retrieved unit counts as a hit if it overlaps the marked spans. Any chunking, current or future, can then be scored without relabeling. Also record a three-way grade: exact / adjacent-acceptable / wrong. That lets you estimate the adjacency problem and the NIL rate separately.

**Sampling.** Stratify by top-1 cosine quintile × margin tertile under the *current* system, per video. Oversample the middle bins, where the threshold will fall, and include 10–15 likely-NIL comments (lowest scores, generic praise). Record the inclusion probabilities. Fit the gate unweighted, and estimate population precision and coverage with inverse-probability weights. Labels drawn now carry little selection bias because 210/212 comments are linked. Later labels drawn only from accepted links would carry more.

**Arms (all scored offline in seconds, paired on the same comments).**

| Arm | Matching unit | Shown unit |
|---|---|---|
| A0 | Current 5-sentence windows, overlap 2 | Same window |
| A1 | Re-segmented units merged to about 60–100 tokens (1–3 units), no overlap, never crossing a section or speaker turn | Centered on the match, 150–250 tokens (§4.3) |
| A2 | A1 plus a 3-unit window index, with max-sim per anchor unit | Same as A1 |

Run each arm with BGE-M3 and Qwen3-Embedding-4B. **Primary contrast: A1 vs A0 on Qwen3-4B**, the embedder you will deploy. Everything else is secondary, to limit multiple comparisons.

**Metrics with confidence intervals.**
- **Retrieval:** top-1 hit rate on non-NIL comments, per arm. Compare arms with a paired exact McNemar test. Give a 95% CI on the paired difference with a bootstrap over comments (2,000 resamples).
- **Gate:** the LR from §4.4 versus the baselines "cosine only" and "cosine + margin". Compute out-of-fold probabilities from repeated stratified 5-fold CV (20 repeats). Report AUROC, Brier score, area under the risk–coverage curve, and **coverage at the lowest threshold whose Wilson 95% lower bound on precision is ≥ 0.85**. Freeze the feature set before fitting, or nest the CV.
- **Length-bias diagnostic:** the Spearman correlation between a unit's length and its in-degree (number of comments it wins), for A0 and for A1.
- **Shown-unit ablation (secondary).** Prompt sizes are 0, ±1 sentence, 150–250 tokens and about 400 tokens. Measure how often the existing decision prompt changes its output across sizes. Compare against human decisions only where CE-08 already has them. A proper evaluation needs its own labels.

**Confirmation criteria.** All four must hold:
- (a) A1 beats A0 on pooled top-1 hit rate at p < 0.05 (McNemar), and is not worse than A0 on either video by more than 5 points.
- (b) The LR gate beats cosine + margin on out-of-fold risk–coverage area, with a bootstrap CI that excludes 0, *or* it reaches higher coverage at the precision-0.85 lower bound.
- (c) The length / in-degree correlation shrinks from A0 to A1.
- (d) Korean is not worse than English by more than 0.10 in either hit rate or gate AUROC.

The 5-point, 0.85 and 0.10 thresholds are **heuristics**.

**Expected resolution (my arithmetic).**
- **Pooled hit rate:** at n = 100 and a hit rate of 0.7, the binomial SE is about 0.046, giving a 95% CI of about ±0.09.
- **Per video:** at n = 50, the CI is about ±0.13. Per-language claims will therefore be coarse.
- **Paired comparison:** pairing is what makes the A1 vs A0 comparison feasible. If 20 of 100 comments are discordant, a 15:5 split gives exact two-sided p ≈ 0.04. A net 10-point gain is therefore detectable, and a 3-point gain is not.
- **Gate precision:** a threshold that accepts 40 links with 36 correct has precision 0.90 and a Wilson 95% CI of **0.77–0.96**.

Expect CIs this wide, and say so when reporting thresholds.

## 4. Details on demand

### 4.1 Re-segmentation recipe (merge-first)

**Measure before choosing.** For each video, measure three things:
- the distribution of gaps between consecutive spans (p50/p90);
- the rate of terminal punctuation (`.?!`, and `.` for Korean);
- how often R:'s sentence flag fires.

If the spans abut with gaps of about 0 s, which is typical of segment-level timestamps, the pause cue carries no information and text cues decide. Which ASR produced R:'s spans is **unknown**. If it is Whisper, English is probably punctuated. Korean punctuation is less consistent (**unverified**).

**Greedy merge, left to right, inside a section.** This is a recipe synthesized from tool defaults. As a whole it is a **heuristic**.
1. **Hard break** at any of these:
   - a section boundary;
   - a speaker change (hQR);
   - a gap of about 1.0 s or more;
   - a merged unit that would exceed about 350 chars (English) or about 200 chars (Korean).
2. **Soft break** (a boundary is allowed) when the gap is at least 0.5 s *and* a text cue fires.
   - English text cue: terminal `.?!`, or a SaT boundary probability above the threshold. Raise SaT's `threshold`, because the problem is over-splitting.
   - Korean text cue: the span's last morpheme is tagged EF (-다/-요/-죠/-습니다/-네), or SaT marks a boundary.
3. **Force a merge** when any of these holds:
   - the unit is shorter than about 40 chars (English) or about 25 chars (Korean), which absorbs fillers such as "Yeah.", "네" and "맞아요";
   - a Korean span ends in a connective ending (EC: -고/-는데/-서/-면);
   - there is no terminal punctuation and the gap is under 0.3 s.

   Treat -는데요/-거든요 as a weak boundary.
4. Keep R:'s sentence flag as one cue only, because it over-fires. Store each unit's first start, last end and its list of span ids, so links map back to exact timestamps.
5. Before merging, collapse hallucinated repeats: consecutive spans with character-3-gram Jaccard above 0.9. Repeats inflate cosine matches.
6. Sweep the gap threshold over {0.3, 0.5, 0.8, 1.0} s against the CE-08 hit rate, not by eye.

The 0.5 s gap is stable-ts's default, not an evaluated optimum ([stable-ts](https://github.com/jianfch/stable-ts)).

**Tool evidence.**
- **Korean splitters.** On the kss author's own (partisan) 2022 benchmark, average normalized F1 was Kss-mecab 0.851, Kiwi 0.782 and a regex baseline 0.561. On the hard sentence-ending subset it was Kss 0.482 and Kiwi 0.342 ([kss 4.5.4](https://pypi.org/project/kss/4.5.4/)). Kiwi has had many releases since the 0.14.1 version tested, and kss is hard to pin, so start with Kiwi and run kss-pecab side by side only if it installs.
- **SaT on Korean.** UD-Korean F1 was 99.6–100, and OPUS100-ko F1 was 72.4 (SaT) to 81.8 (SaT+LoRA) ([SaT](https://arxiv.org/html/2406.16678v2)). No Korean ASR-style score exists.
- **Punctuation restoration** matters only for how readable the *shown* unit is. Re-segmentation needs boundaries, not commas.
  - whooray/koen_punctuation is the only spoken Korean+English model found. It reports period F1 0.922 on an undisclosed eval set (**unverified**, 32 downloads) ([HF](https://huggingface.co/whooray/koen_punctuation)).
  - 1-800-BAD-CODE's XLM-R model lists Korean but publishes no Korean metrics, was trained on news, and its card warns about conversational text ([HF](https://huggingface.co/1-800-BAD-CODE/xlm-roberta_punctuation_fullstop_truecase)).
  - Rule-based splitters (pySBD, spaCy, punkt) need punctuation. WhisperX's punkt map has no Korean entry and falls back to English ([WhisperX](https://github.com/m-bain/whisperX/blob/main/whisperx/alignment.py)).

### 4.2 Matching unit: why small, uniform and non-overlapping

- **Size.** Comments mostly point at one moment ("he said X", a quoted phrase), which resembles SQuAD-style localized questions rather than dispersed ones. TechQA, a dispersed-answer set, instead rose from 4.8% at 64 tokens to 61.3% at 512 ([Bhat et al.](https://arxiv.org/html/2505.21700v2)). The 40–120-token target is an inference from that pattern, a **heuristic**.
- **Uniform length.** Longer texts drift toward the corpus mean and score higher against everything ([Jina](https://jina.ai/news/on-the-size-bias-of-text-embeddings-and-its-impact-in-search); [length-induced collapse](https://arxiv.org/html/2410.24200v1)). Snap windows to a token target at sentence boundaries, rather than counting sentences. A time target (about 15–20 s) is the alternative, and TREC Podcasts used fixed 2-minute windows. Token targets are simpler and keep units linguistically clean.
- **No overlap.** Overlap adds redundant tokens and lowers precision ([Chroma](https://www.trychroma.com/research/evaluating-chunking)). With sliding windows at overlap 2, the second-best chunk is often the best chunk's neighbor, which corrupts the margin.
- **Structure as cut points.** In hQR only 14 of 287 boundaries are speaker boundaries, so turns are too long to serve as matching units. Use speaker changes and section boundaries as forced cuts, and the Q–A structure as display metadata (§4.3). FBIw has no speaker ids.
- **Multi-granularity (A2).** Index each anchor unit both alone and inside its 3-unit window, and take max-sim. Agreement between the two granularities is a candidate gating feature. The learned router in Mix-of-Granularity needs training data you don't have ([MoG](https://arxiv.org/pdf/2406.00456)).

### 4.3 Shown unit: compact, centered, framed, or nothing

**Default.** Start from the matched unit, marked `>>> … <<<` with its timestamp. Add whole sentences symmetrically until the unit reaches **150–250 tokens**, and never cross a section boundary. Expand to about 400 tokens only when the LLM marks the comment as context-dependent (planned step 3). In hQR, when the match falls inside an answer, prepend the interviewer question that governs it, even if the question lies outside the window. Interview comments often judge a person by what they said in reply to a question, so the question frames the answer. This is reasoned, not measured.

**When the gate rejects the link, show no chunk** (or only the video summary). A wrong chunk is worse than none.

**Why not fill 300–500 tokens by default.**
- Related-but-irrelevant passages hurt LLM accuracy, while random ones did not ([Power of Noise, SIGIR 2024](https://arxiv.org/pdf/2401.14887)). Neighboring sentences from the same video are exactly that kind of distractor.
- Across 18 LLMs including Qwen3, a single distractor lowered accuracy, accuracy fell with input length even on simple tasks, and a coherent haystack hurt more than a shuffled one ([Context Rot, Chroma 2025](https://www.trychroma.com/research/context-rot)).
- At 150–400 tokens, position in the context ("lost in the middle") is probably not the main risk. The claim that it fades below about 300 tokens is **unverified** (secondary summary only). Centering the match is fine.

**The 150–250 figure is a heuristic.** No study measures context size for stance or sarcasm classification of short comments with an attached source passage. That is what the secondary ablation in §3 is for.

### 4.4 Gating features and calibration

**Feature pool (choose 3–4).** With about 100 labels and 30–60 events in the minority class, the rule of thumb of about 10 events per feature (a **heuristic**) allows 3–5 features.

| Feature | Targets | Note |
|---|---|---|
| z(s1) = (s1 − μ)/σ over all units for this comment | absolute fit, NIL | BGE-M3 cosines are compressed (median 0.58). Normalizing within the comment removes the offset of generic comments. It is redundant with softmax entropy and CSLS r(q), so keep one of the three. |
| s1 − s(best unit outside the shown window) | wrong-chunk ambiguity | Excluding the window stops neighboring units, which share context, from penalizing correct links (inference). |
| Lexical overlap | quoted distinctive phrases | Options: per-video IDF overlap on Kiwi content morphemes (Korean) or lowercased non-stopword tokens (English), or BGE-M3 `lexical_weights` (s_lex). On MIRACL-ko, BGE-M3 scored dense 69.9 vs dense+sparse 71.3 nDCG@10 ([BGE-M3](https://arxiv.org/html/2402.03216v3)). That the overlap rescues sarcastic quote-links is an untested hypothesis. |
| Qwen3-Reranker-0.6B P(yes) for top-1 (optional) | verification | MMTEB-R 66.36 vs bge-reranker-v2-m3 58.36. These are vendor-reported, with no per-Korean figure ([card](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B)). It takes a task instruction. Its scores are uncalibrated, so the LR absorbs them. |

**Do not fuse dense and lexical scores with RRF before gating.** RRF fuses ranks and discards the score magnitudes the gate needs, so feed the scores as separate LR features.

**Fitting the LR.**
- Standardize the features and fit `LogisticRegression` with L2, choosing C by CV log-loss. The LR is already a Platt-style sigmoid calibrator.
- Do not use isotonic calibration. scikit-learn says it matches or beats sigmoid calibration only above about 1,000 samples ([scikit-learn](https://scikit-learn.org/stable/modules/calibration.html)).
- If the LR does not beat a 2-D grid over (z(s1), margin) by more than noise, ship the grid, and report out-of-fold numbers only.
- Fit thresholds per language, or include a language indicator, because the NIL rate (and with it precision) shifts by video and language.
- If more than about 30% of comments turn out NIL, z(s1) will matter more than margin (inference).

**Optional extras.**
- **CSLS / hubness penalty.** Let r(c) be a unit's mean cosine to its 10 most similar comments. Add it as a feature, or score with 2cos − r(q) − r(c) ([Conneau et al. 2018](https://arxiv.org/abs/1710.04087), formula from memory, **unverified** this session). It absorbs both length bias and hubness, but it also penalizes genuinely popular moments, so keep it only if it improves the CE-08 metrics.
- **Third stage.** Send comments with LR probability between 0.3 and 0.7 (a **heuristic** band) to the local 27B for a yes/no judgment.

### 4.5 Options not shortlisted, and why

- **Semantic or topic segmentation.** On natural documents, fixed-size chunking matched or beat breakpoint chunking (for example HotpotQA 90.59 vs 87.37) ([Qu et al. 2024](https://arxiv.org/html/2410.13070v1)). Kamradt-style breakpoints had the worst IoU (1.5%) in Chroma's test. YouTube chapter segmentation reaches only F1 about 43–46, with segments of about 21 sentences ([YTSeg](https://ar5iv.labs.arxiv.org/html/2402.17633)). R:'s sections already supply topic boundaries.
- **Proposition rewriting (Dense X).** Recall@5 rose 9.3 points for the unsupervised Contriever but only about 2.7 Recall@20 for supervised GTR. Tests covered English Wikipedia only ([Chen et al.](https://arxiv.org/html/2312.06648v3)). Rewriting loses the verbatim phrasing that comments echo. If you try it, add the rewrite as an *extra* key with max-sim, and keep linking and display on the original span. Korean pro-drop makes rewriting both more valuable and riskier (**unverified**).
- **Contextual headers.** Prepending context cut retrieval failures by 35%, or 49% with BM25 added ([Anthropic 2024](https://www.anthropic.com/engineering/contextual-retrieval)). That was across documents. In a single video, a generic header adds the same tokens to every unit and shrinks the margin (inference). The only header worth testing is a local, deterministic one: `[Section 3 | Q: "…"]`.
- **Late chunking.** It gave +2.7–3.6% relative nDCG, but only on mean-pooled bidirectional encoders ([Günther et al.](https://arxiv.org/html/2409.04701v3)). It is a poor match for last-token-pooled Qwen3, per a secondary source ([lc-rag](https://docs.rs/lc-rag/0.22.4/src/lc_rag/late_chunking.rs.html)). For BGE-M3 it is **unverified**, and the claim that BGE-M3 uses CLS pooling is also **unverified**. If you try it, run it as a cheap A/B on BGE-M3 only.
- **Comment-to-moment work.** ComVi (CHI 2026) maps YouTube comments to timestamps ([arXiv 2603.26173](https://arxiv.org/abs/2603.26173)). Its window size, embedder and accuracy were not retrieved. It is the closest prior task and worth reading in full.

## 5. Sources

| Source | Date | Status |
|---|---|---|
| [Bhat et al., Rethinking Chunk Size](https://arxiv.org/html/2505.21700v2) | May 2025 | verified |
| [Chroma, Evaluating Chunking Strategies](https://www.trychroma.com/research/evaluating-chunking) | 2024-07-03 | verified |
| [Jina, size bias of text embeddings](https://jina.ai/news/on-the-size-bias-of-text-embeddings-and-its-impact-in-search) | 2025-04-16 | verified (jina-v3 only; BGE-M3 and Qwen3 magnitudes not measured) |
| [Length-Induced Embedding Collapse](https://arxiv.org/html/2410.24200v1) | Oct 2024 | verified (abstract level) |
| [TREC 2020 Podcasts overview](https://ar5iv.arxiv.org/html/2103.15953) | Mar 2021 | verified |
| [Qu, Tu & Bao, Is semantic chunking worth it?](https://arxiv.org/html/2410.13070v1) | Oct 2024 | verified |
| [YTSeg, Retkowski & Waibel](https://ar5iv.labs.arxiv.org/html/2402.17633) | EACL 2024 | verified |
| [Dense X Retrieval, Chen et al.](https://arxiv.org/html/2312.06648v3) | Dec 2023 / EMNLP 2024 | verified |
| [Mix-of-Granularity](https://arxiv.org/pdf/2406.00456) | COLING 2025 | verified (abstract level) |
| [Late chunking, Günther et al.](https://arxiv.org/html/2409.04701v3) | Sep 2024 | verified |
| [lc-rag late-chunking note](https://docs.rs/lc-rag/0.22.4/src/lc_rag/late_chunking.rs.html) | accessed Nov 2026 | secondary, non-peer-reviewed |
| [Anthropic, Contextual Retrieval](https://www.anthropic.com/engineering/contextual-retrieval) | 2024-09-19 | verified |
| [Power of Noise, Cuconasu et al.](https://arxiv.org/pdf/2401.14887) | SIGIR 2024 | verified |
| [Chroma, Context Rot](https://www.trychroma.com/research/context-rot) | 2025-07-14 | verified |
| ["≤300 tokens, lost-in-the-middle not primary" (Arize recap)](https://arize.com/blog/lost-in-the-middle-how-language-models-use-long-contexts-paper-reading/) | 2024 | **unverified** (original source not identified) |
| [ComVi, Kim et al.](https://arxiv.org/abs/2603.26173) | CHI 2026 (2026-03-27) | verified (abstract only) |
| [kss README](https://github.com/hyunwoongko/kss) / [kss 4.5.4 benchmark](https://pypi.org/project/kss/4.5.4/) | 2024-04 / 2022 | verified. The benchmark is partisan and old |
| [kiwipiepy README](https://github.com/bab2min/kiwipiepy) | accessed 2026-11-03 | verified |
| [SaT, Frohmann et al.](https://arxiv.org/html/2406.16678v2) / [wtpsplit](https://github.com/segment-any-text/wtpsplit) | EMNLP 2024 | verified. TED-column semantics read with moderate confidence |
| [whooray/koen_punctuation](https://huggingface.co/whooray/koen_punctuation) | 2025-03-10 | metrics **unverified** (undisclosed eval set) |
| [1-800-BAD-CODE XLM-R punctuation](https://huggingface.co/1-800-BAD-CODE/xlm-roberta_punctuation_fullstop_truecase) | accessed 2026-11-03 | verified. No Korean metrics |
| [oliverguhr fullstop multilang](https://huggingface.co/oliverguhr/fullstop-punctuation-multilang-large) | accessed 2026-11-03 | verified (no Korean) |
| [stable-ts README](https://github.com/jianfch/stable-ts) | v2.19.1, 2025-08-16 | verified |
| [WhisperX README](https://github.com/m-bain/whisperX) / [alignment.py](https://github.com/m-bain/whisperX/blob/main/whisperx/alignment.py) | v3.8.6, 2026-05-25 | verified |
| [Meng et al., QPP for conversational search](https://arxiv.org/pdf/2305.10923) | SIGIR 2023 | verified |
| [Vlachou & Macdonald, coherence-based dense QPP](https://arxiv.org/abs/2310.11405) | Oct 2023 | verified (abstract) |
| [Gisserot-Boukhlef et al., Towards Trustworthy Reranking](https://arxiv.org/html/2402.12997v5) | Feb 2024, rev. Sep 2024 | verified |
| [BGE M3-Embedding, Chen et al.](https://arxiv.org/html/2402.03216v3) / [model card](https://huggingface.co/BAAI/bge-m3) | Feb 2024 (v5 Dec 2025) | verified. Includes Korean MIRACL numbers |
| [Qwen3-Embedding-4B card](https://huggingface.co/Qwen/Qwen3-Embedding-4B) | 2025-06-05 | verified |
| [Qwen3-Reranker-0.6B card](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B) / [tech report](https://arxiv.org/html/2506.05176v3) | Jun 2025 | verified. Benchmarks are vendor-reported |
| [bge-reranker-v2-m3 card](https://huggingface.co/BAAI/bge-reranker-v2-m3) | accessed 2026-11-03 | verified |
| [scikit-learn calibration guide](https://scikit-learn.org/stable/modules/calibration.html) | v1.9.1 | verified |
| [Conneau et al., CSLS](https://arxiv.org/abs/1710.04087) | ICLR 2018 | **unverified** this session (formula from memory) |
| [Dual Bank Sinkhorn hubness reduction](https://arxiv.org/html/2508.02538v1) | Aug 2025 | verified |
| [Entropy Alone is Insufficient](https://arxiv.org/abs/2603.21172) | Mar 2026 | **unverified** (snippet only) |
| [Kiwi BM25 tutorial (qualitative Korean)](https://langchain-opentutorial.gitbook.io/langchain-opentutorial/10-retriever/10-kiwi-bm25-retriever) | accessed 2026-11-03 | verified, qualitative only |
| PyPI: [kiwipiepy 0.24.1](https://pypi.org/pypi/kiwipiepy/json), [wtpsplit 2.2.2](https://pypi.org/pypi/wtpsplit/json), [kss 6.0.6](https://pypi.org/pypi/kss/json), [python-mecab-ko 1.3.7](https://pypi.org/pypi/python-mecab-ko/json), [spokentxt-punctuation-restoration 0.1.1](https://pypi.org/pypi/spokentxt-punctuation-restoration/json), [punctuators 0.0.7](https://pypi.org/pypi/punctuators/json), [stable-ts 2.19.1](https://pypi.org/pypi/stable-ts/json), [whisperx 3.8.6](https://pypi.org/pypi/whisperx/json) | released 2026-10-20, 2026-09-25, 2025-11-13, 2024-07-14, 2025-03-10, 2024-06-08, 2025-08-16, 2026-05-25 | versions checked 2026-11-03. Python 3.12 installs of kss and punctuators untested. FlagEmbedding, bm25s and rank_bm25 versions **unverified** |
| Requester brief (span and chunk statistics, gating plan) | 2026-11-03 | internal |

**Labeled heuristics (not literature standards):**
- the matching-unit target of 40–120 tokens (60–100 in A1), and the per-embedder size re-tune
- the shown-unit default of 150–250 tokens and the 400-token escalation cap
- the merge thresholds: 0.5 s soft gap (a stable-ts default), 1.0 s hard gap, 0.3 s force-merge gap, and the gap sweep values
- the length bounds: English 40 / 350 chars, Korean 25 / 200 chars
- character-3-gram Jaccard > 0.9 for detecting repeats
- treating EF as a boundary and EC as continue, with -는데요/-거든요 as weak boundaries
- margin computed against the best unit outside the shown window
- about 10 events per feature, giving a cap of 3–5 features
- the precision target of 0.85, the 5-point non-inferiority margin, and the 0.10 Korean-vs-English gap
- the 0.3–0.7 band for LLM third-stage verification
- the stratification grid (cosine quintile × margin tertile) and the 10–15 likely-NIL items
- the 30% NIL level above which absolute-score features dominate

**Korean evidence gaps:** none of the following has been measured: splitter accuracy on Korean ASR, Korean pause statistics, Korean chunk-size effects, Korean shown-context effects, per-Korean reranker scores, or a Korean BM25 tokenizer comparison. Throughput of the reranker and the 27B verifier on this card has not been measured either.
