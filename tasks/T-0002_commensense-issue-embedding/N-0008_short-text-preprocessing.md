---
id: N-0008
type: note
task: T-0002
title: "Preprocessing short, noisy, code-switched Korean–English YouTube comments before LLM gisting, embedding and issue clustering"
status: final
created: 2026-11-02
tags: [text-preprocessing, code-switching, korean]
---

# Preprocessing short, noisy, code-switched Korean–English YouTube comments before LLM gisting, embedding and issue clustering

Scope note: Research done 2026-11-02 within a budget of about 17 tool calls. Primary sources were fetched where possible. Many items are marked "not verified" because only an abstract or search snippet was available. For several sub-questions (emoji effects on BGE-M3/Qwen3, Korean code-switched embedding quality, Korean sarcasm with open 8–30B LLMs) **I found no direct primary evidence**. The recommendations for those are inferences and are labelled as such.

## 1. Normalization: emoji, ㅋㅋ/ㅎㅎ repetitions, URLs, timestamps, Korean normalizers

### Takeaway
Normalize lightly, and keep two text fields. (a) The **raw text**, with only repetitions collapsed, goes to the LLM gister, because emoji, ㅋㅋ and ㅠㅠ carry stance and sarcasm cues. (b) The **LLM-generated issue gist** is what gets embedded. soynlp's `repeat_normalize` / `emoticon_normalize` cover the Korean-specific repetition problem. KoNLPy-style morphological analysis is not needed for neural embedders. Replace timestamps with a placeholder and, where possible, link them to the transcript segment as metadata or context, not as embedding text.

### Cited Findings
- soynlp (MIT-licensed) provides `emoticon_normalize(text, num_repeats=3)`, which turns 'ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ쿠ㅜㅜㅜㅜㅜㅜ' into 'ㅋㅋㅋㅜㅜㅜ'. It also provides `repeat_normalize(text, num_repeats=2)`, which turns '와하하하하하하하하하핫' into '와하하핫'. It also offers `only_hangle`, `only_hangle_number` and `only_text` filters. — [soynlp GitHub](https://github.com/lovit/soynlp)
- Note: `emoticon_normalize` also splits jamo-merged syllables (쿠 → ㅋ + ㅜ) as part of its normalization, as the example shows. `only_hangle` strips Latin characters, so it **would destroy English in code-switched comments** and must not be used here. — [soynlp GitHub](https://github.com/lovit/soynlp)
- BERTopic practice: transformer-based topic modelling "does not require heavy preprocessing". Minimal cleaning, such as removing extra spaces and filtering very short documents, is usually enough. — [Analytics Vidhya BERTopic guide, Apr 2026](https://www.analyticsvidhya.com/blog/2026/04/bertopic-guide/) (secondary/blog source)
- Emoji handling evidence is mixed and comes from classification tasks, not embedding or clustering. Some transformer systems convert emoji to text (for example with the `emoji`/`demoji` libraries, giving ":slightly_smiling_face:") because pretrained models may lack good emoji representations. — [BRUMS at SemEval-2020 Task 12](https://arxiv.org/pdf/2010.06278). For Arabic sarcasm detection, *removing* emoji was reported to improve accuracy. — [Impact of emoji exclusion on Arabic sarcasm detection, arXiv 2405.02195 (2024)](https://emergentmind.com/papers/2405.02195)
- A DH2026 study on BERTopic for Korean vs. US YouTube comments (meditation videos) concludes that "language-specific preprocessing, embeddings, and clustering are crucial for capturing cultural nuance". It proposes combining qualitative interpretation with parameter tuning. The Zenodo landing page does not give the preprocessing details. — [Lee, Woo & Jun, DH2026, Daejeon, 26 Aug 2026 (Zenodo)](https://zenodo.org/records/22107263)

### Inferences
- **Recommended normalization pipeline** (inference, not benchmarked):
  1. Unicode NFKC, whitespace collapse, and removal of zero-width characters.
  2. `soynlp.normalizer.repeat_normalize(t, num_repeats=2)` to collapse ㅋㅋㅋㅋ, ㅎㅎㅎ, ㅠㅠㅠㅠ, !!!! and 하하하하.
  3. Collapse repeated identical emoji to a maximum of 2. Keep emoji in the text the LLM sees, because 🤣 or 🙄 often flips the polarity (sarcasm). Do not convert emoji to English words before an LLM: modern LLM tokenizers handle emoji natively, and conversion would add English tokens to Korean text.
  4. Replace URLs with `<URL>` and keep the domain as metadata. Replace `h:mm` / `mm:ss` timestamps with `<TS>` in the embedded text, and keep the parsed seconds as a field.
  5. Do not run a morphological analyzer (KoNLPy, kiwipiepy). Subword-tokenized multilingual embedders (BGE-M3 uses an XLM-R tokenizer; Qwen3 uses a BPE LLM tokenizer) do not need it. It only matters if you later build c-TF-IDF keyword labels for clusters, where kiwipiepy noun extraction helps produce readable Korean keywords.
- **Embed the gist, not the raw comment.** Since the pipeline produces an LLM "issue gist" per comment, most surface noise (emoji, ㅋㅋ, slang, spelling) is absorbed by the LLM. Embedding the normalized gist makes the embedder-robustness question largely moot. Raw-comment embeddings are still useful as a secondary signal for near-duplicate detection (see §5).
- **Timestamps:** comments like "3:45 이 부분 ㄹㅇ" are uninterpretable without the transcript. Recommendation: resolve the timestamp to the transcript segment (for example a ±15–30 s window from Whisper or YouTube captions) and give that segment to the *gisting LLM as context* ("the commenter refers to this moment: …"). Do not concatenate it into the embedded text, or all timestamped comments will cluster around transcript content. This is a design inference. I found no published evaluation of timestamp resolution for comment clustering.

### Gaps
- I found no study measuring how emoji, ㅋㅋ or repeated jamo affect BGE-M3 or Qwen3-Embedding vectors specifically. Running an ablation on your own data is recommended (embed raw vs. normalized and compare cluster ARI and silhouette).
- kiwipiepy features (typo correction, sentence splitting, newer slang handling) were not verified in this session.
- I could not access the DH2026 Korean YouTube BERTopic slides/PDF to extract their concrete preprocessing choices.

## 2. Short texts: minimum-content filtering and a "no-issue / non-substantive" bucket

### Takeaway
Use an explicit pre-clustering filter plus an LLM "NON_SUBSTANTIVE" label. Rules handle emoji-only, ㅋㅋ-only, "first!", "1빠" and timestamp-only comments. The LLM gist prompt then emits a `no_issue` flag for pure reactions and praise or insults with no issue. Keep these items as a counted bucket (they matter for reporting reaction volume), but exclude them from embedding and clustering. Treat HDBSCAN `-1` outliers as a *separate* residual category, not as the non-substantive bucket.

### Cited Findings
- In applied BERTopic work, extremely short items (for example about 5% of reviews under 20 characters) are often not meaningful and are filtered out before modelling. — [Multilingual transformer and BERTopic for short text topic modeling, arXiv 2402.03067 (2024)](https://arxiv.org/pdf/2402.03067) (search snippet; not verified in full text)
- With HDBSCAN, BERTopic does not force every point into a cluster. Noise points become topic −1, which keeps clusters cleaner. Outlier shares on short text can be large. One reported case had a 53.4% outlier rate, reduced to 5.8% by multi-topic assignment, and `reduce_outliers` with the c-TF-IDF strategy can bring outliers close to 0. — [Analytics Vidhya BERTopic guide, Apr 2026](https://www.analyticsvidhya.com/blog/2026/04/bertopic-guide/) (secondary source; the original study for the 53.4% → 5.8% figure was not identified)
- Topic-model screening metrics used in practice include outlier rate, largest-topic share, topic diversity, silhouette and Davies–Bouldin. Screening penalises outlier rate and rewards diversity and silhouette. — [same BERTopic guide](https://www.analyticsvidhya.com/blog/2026/04/bertopic-guide/)

### Inferences
- Pure-reaction comments ("ㅋㅋㅋㅋ", "👍👍", "레전드", "first") have near-identical, low-information embeddings. In a density-based clusterer they either form one large "reaction" cluster or inflate the noise set, and they compress UMAP space. Removing them before UMAP/HDBSCAN should give tighter and more interpretable issue clusters. This follows from how density clustering works. I found **no controlled study** that quantifies the gain on comment data.
- **Suggested cascade:**
  1. Rule filter: after stripping emoji, URLs, `<TS>`, punctuation and ㅋ/ㅎ/ㅠ/ㅜ jamo, fewer than 2 Hangul syllables or fewer than 2 Latin words → `NON_SUBSTANTIVE_RULE`.
  2. A small regex list for "first/1빠/1등/2빠", "알고리즘(이 나를) 데려왔다" and similar memes.
  3. The LLM gist prompt returns a JSON field `substantive: true/false` plus a `reaction_type` (praise / laugh / insult-only / meme / question-to-creator) for the bucket.
  4. Only `substantive=true` gists are embedded and clustered.
- Keep HDBSCAN outliers distinct from non-substantive items. Outliers are often *substantive but singular* issues, which matter for a codebook. Review them, or run `reduce_outliers` only after inspection, rather than forcing them into clusters.

### Gaps
- No peer-reviewed Korean comment study found that reports the share of non-substantive comments or the clustering gain from removing them.
- The original source of the "53.4% → 5.8%" outlier figure could not be traced.

## 3. Code-switching: translate first vs. rely on multilingual embedders; Korean–English alignment; do clusters split by language?

### Takeaway
Do not machine-translate the raw comments. Instead, have the gisting LLM write **every gist in one pivot language**, either Korean (if the codebook is Korean) or English. This removes language as a clustering axis without a separate MT model. Then embed gists with a strong multilingual embedder (Qwen3-Embedding or BGE-M3). If you keep mixed-language gists, check explicitly for language-split clusters, because multilingual embedders are known to cluster partly by language.

### Cited Findings
- BGE-M3 (M3-Embedding, first posted 5 Feb 2024, latest revision 12 Dec 2025) supports more than 100 working languages and inputs up to 8,192 tokens. It combines dense, sparse and multi-vector retrieval. — [arXiv 2402.03216](https://arxiv.org/abs/2402.03216) (Korean-specific MKQA/MIRACL numbers not extracted; see Gaps)
- Qwen3-Embedding (technical report 5 Jun 2025, Apache-2.0): the 8B model scored 70.58 on MTEB Multilingual and ranked #1 on that leaderboard as of 5 Jun 2025. It supports over 100 languages and claims strong cross-lingual retrieval. — [Qwen3 Embedding tech report, arXiv 2506.05176](https://arxiv.org/html/2506.05176v3); [Qwen blog](https://qwenlm.github.io/blog/qwen3-embedding)
- Language bias: embeddings from multilingual models "tend to cluster by language rather than meaning". In one practical example, the multilingual-e5-small embedding space put English and Spanish documents into separate clusters instead of mixing them by meaning. — [Emergent Mind: Language-Agnostic Embeddings](https://www.emergentmind.com/topics/language-agnostic-embeddings); [Cloudflight: Language bias in multilingual semantic search](https://www.cloudflight.io/en/blog/language-bias-in-multilingual-semantic-search-systems/) (secondary/blog sources)
- Same-language bias (preferring candidates in the query's language) was documented in LAReQA, a language-agnostic retrieval benchmark. — [LAReQA, arXiv 2004.05484 (2020)](https://arxiv.org/pdf/2004.05484) (older than the preferred window, but foundational)
- Code-switched representations in multilingual LLMs are "anchored". Source-framed code-switching stays near the source-language representation, while target-framed code-switching shifts toward the target and shows larger QA degradation. The authors propose an "Anchor Bias" metric and an inference-time fix (CANVAS). — [Park, Yoon, Jun & Lee, "Code-Switching Reveals Language Anchoring in Multilingual LLMs", arXiv 2606.19668 (Jun 2026, rev. Aug 2026)](https://arxiv.org/abs/2606.19668) (abstract only; language pairs not stated in the abstract)
- Korean–English code-switching benchmarks exist mainly for speech. HiKE (2025) is a hierarchical Korean–English code-switching ASR benchmark. — [HiKE, arXiv 2509.24613](https://arxiv.org/pdf/2509.24613). An older line of work embeds Korean and English conversational sentences with separate monolingual models plus a hybrid matcher. — [Applied Sciences 13(9):5771 (2023)](https://www.mdpi.com/2076-3417/13/9/5771)
- GLUECoS (2020, English–Hindi/Spanish): multilingual BERT beat cross-lingual embedding techniques on code-switched tasks. This is evidence that jointly trained multilingual encoders handle intra-sentence mixing better than aligned monolingual spaces. — [GLUECoS, arXiv 2004.12376](https://arxiv.org/pdf/2004.12376)

### Inferences
- **Why gist-in-one-language beats MT-then-gist:** local MT models (NLLB, M2M, Qwen-based MT) handle slang, ㅋㅋ and code-switching poorly and would remove the cues the gisting LLM needs. An 8–30B instruction LLM (for example Qwen3, Gemma 3 or EXAONE) already reads mixed Korean–English, so asking it to "write the issue gist in Korean, regardless of comment language" performs translation and abstraction in one step. All of this runs locally.
- When gists are uniform in language, the language-bias problem above largely disappears for clustering. If some raw comments are fully English (international viewers), their gists will still be Korean (or English), so clusters will not split by language.
- **Diagnostic:** after clustering, compute per-cluster language composition, for example with a simple Hangul/Latin ratio. If clusters are >90% single-language while the corpus is mixed, you have language-driven splitting. Fixes are re-gisting in the pivot language or language-mean-centring (subtracting each language's mean embedding), the subspace-removal idea in the language-agnostic-embedding literature.
- On a 16 GB GPU, BGE-M3 (568M parameters) or Qwen3-Embedding-0.6B/4B fit easily. Qwen3-Embedding-8B in fp16 needs about 16 GB and is tight, so use a quantized version or the 4B model.

### Gaps
- **No primary benchmark found** that measures Korean–English *code-switched* (intra-sentence) embedding quality for BGE-M3, multilingual-e5 or Qwen3-Embedding. Korean-specific cross-lingual numbers (MKQA-ko, MIRACL-ko, Flores/Tatoeba ko–en bitext mining) exist in the model papers and MTEB but were not extracted in this session.
- No study found that directly compares "translate-then-embed" vs. "embed-multilingual" for Korean comment clustering.
- The multilingual-e5 model card was not fetched.

## 4. Sarcasm and irony: LLM handling of Korean sarcasm; literal vs. intended meaning in gist prompts

### Takeaway
LLMs still lag supervised models on sarcasm detection, and Korean sarcasm is context-dependent. Prompt the gist for the **intended meaning** (stance and target), with separate fields for `literal_text_polarity`, `sarcasm_suspected` (bool) and `confidence`. Supply the video title and description, the transcript segment (if timestamped) and the parent comment (for replies) as context. Validate on a hand-labelled sample of about 200 comments.

### Cited Findings
- KoCoSa (LREC-COLING 2024; arXiv first posted 22 Feb 2024) has 12.8K Korean daily dialogues labelled for sarcasm on the last response. It was built by LLM generation, then automatic and manual filtering, then human annotation. Its fine-tuned baseline beats LLMs such as GPT-3.5, and the authors stress that sarcasm detection "relies deeply on the existence of sufficient context". — [arXiv 2402.14428](https://arxiv.org/abs/2402.14428); [ACL Anthology 2024.lrec-main.864](https://aclanthology.org/2024.lrec-main.864/) (exact F1 numbers not extracted)
- KoCasm / korean-sarcasm: 9,319 Korean tweets labelled sarcasm / not_sarcasm. It is available on GitHub and as the MTEB task `KorSarcasmClassification` (so embedders' MTEB Korean scores include a sarcasm classification task). — [SpellOnYou/korean-sarcasm GitHub](https://github.com/SpellOnYou/korean-sarcasm); [mteb/KorSarcasmClassification on HF](https://huggingface.co/datasets/mteb/KorSarcasmClassification)
- SarcasmBench (IEEE Trans. Affective Computing, 2025): across 16 LLMs and 8 PLMs on six text and three multimodal sarcasm datasets, **current LLMs underperform supervised PLM baselines** on sarcasm detection. GPT-4 and Gemini 2.0 are the strongest LLMs. — [SarcasmBench, IEEE TAC 2025](https://www.computer.org/csdl/journal/ta/2025/04/11146812/29GBxQiBx9m) (English-centric)
- CLIcK (2024) is a Korean cultural and linguistic intelligence benchmark (1,995 instances, 11 categories). It is useful as a proxy for Korean pragmatic and cultural competence when choosing an open LLM. — [CLIcK, arXiv 2403.06412](https://arxiv.org/pdf/2403.06412)
- Open Ko-LLM Leaderboard2 (Oct 2024) adds nine practical Korean tasks. The search found no sarcasm-specific task there. — [arXiv 2410.12445](https://arxiv.org/pdf/2410.12445)

### Inferences
- **Literal vs. intended:** for issue clustering, the *issue* (target and topic) is usually the same under both readings. "와 진짜 논리 대단하시네요 ㅋㅋ" is about the guest's argument quality either way; only the stance flips. So: (a) the gist states the issue neutrally ("guest's reasoning on X") and (b) stance and sarcasm go in separate structured fields. This keeps clustering robust to sarcasm misreadings while preserving stance for the codebook.
- Korean sarcasm cues worth naming in the prompt: exaggerated honorific praise plus ㅋㅋ/ㅎ, "~하시네요"/"~잘하는 짓이다", "참 대단하다", 🙄/🤡/👏 emoji, and quotation of the speaker followed by ㅋㅋ. Few-shot examples taken from your own data will help an 8–30B model more than generic instructions. (Inference; not benchmarked.)
- A useful local test set: a stratified sample from KoCasm/KorSarcasmClassification to sanity-check the chosen open LLM's Korean sarcasm recall before full runs.

### Gaps
- **No primary evidence found** on how specific open-weight 8–30B models (Qwen3, Gemma 3, EXAONE, Llama) perform on Korean sarcasm. KoCoSa compares against GPT-3.5 only (per the abstract).
- KoBEST (Korean balanced evaluation of significant tasks) was not checked; to my knowledge it has no sarcasm task (unverified).

## 5. Deduplication: exact and near-duplicate, MinHash/SimHash, embedding-threshold, SemDeDup; dedup before clustering with counts as weights

### Takeaway
Dedup in three tiers before gisting: exact after normalization, then character n-gram MinHash/Jaccard, then embedding cosine above a threshold within the video. Keep one representative per group with `count` and member IDs. This saves LLM gisting calls and stops copy-paste memes from creating artificial dense clusters. Re-inflate the counts as weights for cluster sizes and prevalence in the codebook.

### Cited Findings
- SemDeDup (ICML 2023; arXiv 2303.09540) embeds items with a pretrained model, clusters them with k-means, and within each cluster removes pairs above a cosine threshold. It removed about 50% of LAION with minimal performance loss, and on C4 it beat prior deduplication with roughly 15% efficiency gains, sometimes improving performance. — [arXiv 2303.09540](https://arxiv.org/abs/2303.09540v3); [ICML 2023 page](https://icml.cc/virtual/2023/26932)

### Inferences
- At a few hundred to about 2k comments per video, all-pairs cosine similarity is trivial (2k × 2k), so the k-means partitioning step of SemDeDup is unnecessary. Use a direct threshold plus connected components or agglomerative grouping.
- **Suggested thresholds (to tune on your data):**
  - Exact: hash of normalized text (NFKC, repeat-normalized, emoji collapsed, whitespace removed).
  - Near-duplicate: Jaccard of character 3-grams ≥ 0.8. Character n-grams work for Korean without tokenization; `datasketch` MinHashLSH is optional at this scale.
  - Semantic: cosine ≥ about 0.95 on raw-text embeddings for copy-paste memes with small edits.
  - Do **not** dedup at lower thresholds (around 0.85) on raw text. Different people making the same point is the *signal* for issue prevalence and should be clustered, not merged.
- Copy-paste memes and spam: flag groups with count ≥ k (for example 5) for manual review. They may be brigading (organized mass posting) or a meme, and are worth reporting separately.
- Weights: HDBSCAN has no native sample weights. Practical options are (a) cluster the deduped representatives and report weighted sizes, or (b) for prevalence metrics, multiply by `count`. Option (a) avoids artificial density peaks from copy-paste.

### Gaps
- No study found on dedup thresholds for short Korean comments. The SimHash vs. MinHash comparison for very short texts was not researched in this session.

## 6. Korean YouTube and online-comment NLP resources; prior Korean comment-clustering studies

### Takeaway
Labelled Korean comment corpora are mostly toxicity, hate and sentiment resources from news, YouTube and movie-review sites. They are useful for few-shot examples, for a toxicity side-classifier, and for picking an embedder or LLM. Prior Korean YouTube comment topic-modelling work uses BERTopic with Korean encoders (KcBERT, KLUE-RoBERTa) and reports that language-specific choices matter.

### Cited Findings
- **KOLD** (EMNLP 2022): 40,429 Korean comments from NAVER news **and YouTube**, annotated *in context* (with title) using a hierarchical offensive-language taxonomy (type and target) plus offensive and target text spans. — [arXiv 2205.11315](https://arxiv.org/pdf/2205.11315); [ACL Anthology 2022.emnlp-main.744](https://aclanthology.org/2022.emnlp-main.744)
- **K-MHaS** (COLING 2022): 109,692 Korean online news comments (raw data collected Jan 2018–Jun 2020) with 8 fine-grained multi-label hate classes (for example politics, origin, physical, age, gender). It supports binary and 1–4-label multi-label tasks. — [ACL Anthology 2022.coling-1.311](https://aclanthology.org/2022.coling-1.311); [arXiv 2208.10684](https://arxiv.org/abs/2208.10684)
- **KoCoSa** (12.8K dialogues) and **KoCasm** (9,319 tweets) for sarcasm: see §4. — [KoCoSa](https://aclanthology.org/2024.lrec-main.864/); [korean-sarcasm](https://github.com/SpellOnYou/korean-sarcasm)
- **KcBERT** was pretrained on Korean online-news comments and performs well on informal text with slang and typos. It has been used inside BERTopic pipelines for Korean reviews. — [Marketing insights from reviews using BERTopic and DCN (2023)](https://www.researchgate.net/publication/373285761_Marketing_Insights_from_Reviews_Using_Topic_Modeling_with_BERTopic_and_Deep_Clustering_Network)
- Cross-national BERTopic study (China, Japan, South Korea social media on childbirth and parenthood) produced 302 clusters from Korean comments. Language-specific encoders such as `klue/roberta-base` were used for Korean. — [arXiv 2510.06788 (Oct 2025)](https://arxiv.org/pdf/2510.06788)
- DH2026 study on Korean vs. US YouTube meditation comments with BERTopic: it stresses the limits of topic reduction and the need for human interpretation. — [Zenodo 22107263 (Aug 2026)](https://zenodo.org/records/22107263)
- Practitioner survey of open Korean corpora (2020). — [Open Korean Corpora: A Practical Report, arXiv 2012.15621](https://arxiv.org/pdf/2012.15621)
- Korean embedding benchmarking for RAG (AutoRAG example), useful for choosing an embedder on Korean text. — [Marker-Inc-Korea/AutoRAG-example-korean-embedding-benchmark](https://github.com/Marker-Inc-Korea/AutoRAG-example-korean-embedding-benchmark)

### Inferences
- KOLD is the closest public match to the target domain (it includes YouTube and context-conditioned labels). Its titles plus comments are a good source of realistic few-shot examples and a toxicity side-label for the codebook ("insult-only" vs. "issue-bearing criticism").
- Prior Korean YouTube studies used *monolingual* Korean encoders (KcBERT, KLUE). For code-switched comments with an LLM-gist step, a multilingual embedder on uniform-language gists is the better fit (§3). The Korean-only encoders remain a reasonable baseline to compare against.

### Gaps
- **BEEP!** (Korean hate speech on entertainment news comments, Moon et al., SocialNLP 2020) and **NSMC** (Naver movie-review sentiment, 200k) were not verified in this session. Both are well known, but their details are unsourced here.
- No prior study found that does *LLM-gist-then-cluster* on Korean YouTube comments specifically. HCLT/KCC proceedings were not searched (access limited).
