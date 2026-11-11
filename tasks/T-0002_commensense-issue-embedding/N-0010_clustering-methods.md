---
id: N-0010
type: note
task: T-0002
title: "Clustering methods for building an issue codebook from embeddings of short LLM-written gists (n ≈ 300–2,000, Korean/English, local only)"
status: final
created: 2026-11-02
tags: [clustering, codebook]
---

# Clustering methods for building an issue codebook from embeddings of short LLM-written gists (n ≈ 300–2,000, Korean/English, local only)

Scope note: research done 2026-11-02. "Unverified" marks claims from background knowledge that I did not re-check against a primary source in this session. Constraint check: everything must run locally (Python 3.12 + uv, 16 GB GPU shared with a llama.cpp LLM). Methods whose published evidence uses only GPT-3.5/GPT-4-class APIs are flagged **[API-only evidence]**.

---

## 1. k-means family (k selection, spherical k-means) as the baseline

### Takeaway
k-means on L2-normalized embeddings (which is effectively spherical/cosine k-means) is the most stable and outlier-free baseline. On short texts it matches HDBSCAN-based BERTopic quality without discarding documents. Its weaknesses are that k must be chosen and that it forces every comment, including no-issue and rare-issue ones, into a cluster. Most LLM-in-the-loop methods (k-LLMmeans, LLMEdgeRefine, few-shot PCKMeans) are built on k-means and assume a known k.

### Cited Findings
- Replacing HDBSCAN with k-means in BERTopic on short multi-domain text "achieve[s] similar performance, but without outliers"; HDBSCAN "classifies a majority of the documents as outliers" (de Groot, Aliannejadi, Haas; WiNLP @ EMNLP 2022, arXiv Dec 2022) — [arXiv 2212.08459](https://arxiv.org/abs/2212.08459)
- The BERTopic FAQ lists swapping HDBSCAN for k-means as an outlier-free alternative. It also says that on small datasets (~1,000 docs) topic extraction is "very difficult… due to the little amount of data available" — [BERTopic FAQ](https://maartengr.github.io/BERTopic/faq.html)
- k-LLMmeans and other k-means-based LLM methods require "the known number of clusters" as input — [k-LLMmeans v2, arXiv 2502.09667, May 2025](https://arxiv.org/html/2502.09667v2)
- Text-clustering benchmarks (Bank77, CLINC, GoEmo, MASSIVE, MTOP) run k-means on E5/Instructor embeddings as the standard baseline. LLM methods report gains over it — [Huang & He, arXiv 2410.00927](https://arxiv.org/html/2410.00927v1); [k-LLMmeans](https://arxiv.org/html/2502.09667v2)

### Inferences
- Spherical k-means: once vectors are L2-normalized, squared Euclidean distance equals 2 − 2·cos. So sklearn `KMeans` on normalized embeddings approximates cosine k-means. The only difference is that centroids are not re-normalized each iteration, which matters little in practice. A dedicated package is not needed. (Background, unverified in this session.)
- k selection on sentence embeddings (unverified background): silhouette with `metric="cosine"` tends to be low (around 0.05–0.15) and flat for short-text embeddings, so it is a weak signal. Elbow curves on inertia are usually smooth with no clear elbow. The gap statistic (Tibshirani et al. 2001) is expensive and often favors small k in high dimensions. The practical approach is to sweep k (for example k ∈ {n/40 … n/10}), compute silhouette or Davies–Bouldin plus a cluster-size distribution, and let the human/LLM review settle granularity. Treat k as a granularity knob, not as something to "discover".
- For the codebook use case, k-means' lack of a noise class is a defect. The no-issue gists and the long tail get absorbed into the nearest large cluster and contaminate its examples. Mitigations: (a) have the LLM gist step emit an explicit "NO_ISSUE" flag and filter those out before clustering; (b) over-cluster (large k) so rare issues get their own small clusters; (c) flag points whose cosine to their centroid falls below a percentile threshold as "unassigned" for review.
- The team's previous k-means sampler plus LLM naming is essentially the "k-means → LLM label" pattern that the papers below use as a baseline. The upgrades with evidence are LLM-refined centroids (k-LLMmeans), LLM edge-point reassignment (LLMEdgeRefine), and LLM merge/split adjudication.

### Gaps
- I found no 2023–2026 study that systematically compares silhouette, gap and elbow for choosing k on sentence-transformer embeddings of short texts. The statements above are practitioner knowledge, not verified.

---

## 2. UMAP + HDBSCAN (BERTopic default) for small corpora

### Takeaway
UMAP + HDBSCAN finds the number of clusters automatically and gives a noise class, which fits "unknown number of issues + no-issue comments". At n = 300–2,000 short texts, though, it often marks a large share (sometimes a majority) of documents as outliers. Because of UMAP, it is also seed-sensitive. For small n, use small `min_cluster_size` (around 3–10), a lower `min_samples`, UMAP with `min_dist=0` and a fixed `random_state`, and post-hoc outlier reassignment. Alternatively, skip UMAP and run HDBSCAN or agglomerative clustering directly on cosine distances.

### Cited Findings
- HDBSCAN `min_cluster_size` = "the smallest size grouping that you wish to consider a cluster". `min_samples` defaults to `min_cluster_size`, and "the larger the value of `min_samples`… the more conservative the clustering – more points will be declared as noise". The docs advise fixing `min_samples` first while varying `min_cluster_size` — [HDBSCAN parameter selection](https://hdbscan.readthedocs.io/en/latest/parameter_selection.html)
- `cluster_selection_epsilon` merges micro-clusters in dense regions (for example 0.5 means "don't separate clusters that are less than 0.5 units apart"). `cluster_selection_method='eom'` (default) tends to give one or two large clusters plus small ones. `'leaf'` gives "many small homogeneous clusters" (finer granularity). `allow_single_cluster` exists for data that is mostly noise. The docs advise leaving `alpha` alone — [HDBSCAN docs](https://hdbscan.readthedocs.io/en/latest/parameter_selection.html)
- BERTopic FAQ: lowering `min_samples` "will result in less noise being generated". `.reduce_outliers` offers "four built in strategies" without retraining. Too many topics → raise `min_topic_size`/`n_neighbors` or use `nr_topics="auto"`. Too few → lower them, or accept that "there simply aren't that many topics". For reproducibility, set `random_state=42` in UMAP, "at the expense of performance" — [BERTopic FAQ](https://maartengr.github.io/BERTopic/faq.html)
- UMAP docs for clustering: raise `n_neighbors` (for example 30 instead of the default 15), because small values produce "fine grained cluster structure that may be more a result of patterns of noise". Set `min_dist=0`. Explore `n_components`. UMAP "does not completely preserve density" and "can also create false tears in clusters, resulting in a finer clustering than is necessarily present"; the docs call this "somewhat controversial, and should be attempted with care". UMAP helps because "high dimensional data requires more observed samples to produce much density" — [UMAP: Using UMAP for Clustering](https://umap-learn.readthedocs.io/en/latest/clustering.html)
- Short-text outlier rates: one search-result summary reported "up to 74%" outliers for short texts and a Twitter study with ~72.66% outliers. I could not trace these exact figures to a primary source. The primary paper says only "a majority" — [arXiv 2212.08459](https://arxiv.org/abs/2212.08459) (figures **unverified**)
- BERTopic 0.17.0 (2025-03-19) added a "light-weight installation without UMAP and HDBSCAN" — [BERTopic changelog](https://maartengr.github.io/BERTopic/changelog.html)

### Inferences
- Suggested starting grid for n ≈ 300–2,000 (inference, not from a source): UMAP `n_neighbors` 10–30 (around √n or n/50, never above ~5% of n), `n_components` 5–10, `min_dist=0.0`, `metric="cosine"`, fixed `random_state`. HDBSCAN `min_cluster_size` 3–8 for n ≈ 300 and 5–15 for n ≈ 2,000; `min_samples` 1–3 (noise rate drops sharply); `cluster_selection_method="leaf"` if you want the over-clustered granularity the human will merge.
- Whether to skip UMAP at small n: with n < ~1,000 the UMAP kNN graph is small and seed variance is high. A reasonable alternative is HDBSCAN on a precomputed cosine-distance matrix of the raw (or PCA-to-~50-dim) embeddings, or agglomerative clustering (section 3). The UMAP docs' reasoning (density estimation needs dimensionality reduction) still applies, so evaluate both. Report noise rate, number of clusters and ARI across around 5 seeds as a stability check.
- Instability mitigation: run UMAP+HDBSCAN with several seeds and keep co-association (consensus) clusters, or accept a single fixed seed and rely on human review. Treat HDBSCAN `-1` points as a "possible no-issue / rare issue" review queue, not as discarded data. This is attractive for codebooks because the long tail is exactly what -1 contains.

### Gaps
- I found no published guidance specific to n < 2,000 beyond the BERTopic FAQ's "small datasets are hard". The parameter ranges above are my inference.
- I did not verify the names of the four `reduce_outliers` strategies against the docs. From memory (unverified) they are: `"probabilities"`, `"distributions"`, `"c-tf-idf"`, `"embeddings"`.

---

## 3. Agglomerative / hierarchical clustering for codebook granularity

### Takeaway
Agglomerative clustering with cosine distance and average linkage, or Ward on L2-normalized vectors, gives a full dendrogram. Cutting it with `distance_threshold` gives a k-free, deterministic clustering at chosen granularity. This fits a human-reviewed codebook well, because parent/child merges become candidate "merge" and "split" suggestions. At n ≤ 2,000 the O(n²) memory and time cost is trivial.

### Cited Findings
- sklearn `AgglomerativeClustering`: `metric` ∈ {euclidean, l1, l2, manhattan, cosine, precomputed}. "If linkage is 'ward', only 'euclidean' and 'l2' are accepted." Linkage options are ward (default), average, complete and single. `distance_threshold` requires `n_clusters=None` and `compute_full_tree=True`. `compute_distances=True` enables dendrogram plotting. A `connectivity` kNN graph is optional — [scikit-learn docs](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html)
- ClusterLLM (EMNLP 2023) already uses hierarchical clustering for granularity. It asks the LLM pairwise "do A and B belong to the same category" questions and "selects cluster hierarchies most consistent with ChatGPT responses" — [arXiv 2305.14871](https://arxiv.org/abs/2305.14871)
- BERTopic itself offers topic merging and reduction (`nr_topics="auto"`, which merges similar topics while keeping dissimilar ones) — [BERTopic FAQ](https://maartengr.github.io/BERTopic/faq.html)

### Inferences
- Practical setting: normalize the embeddings, then either (a) `linkage="average", metric="cosine", distance_threshold≈0.3–0.5` (cosine distance; tune per embedding model, since E5/bge-type models have compressed similarity ranges), or (b) `linkage="ward"` on the normalized vectors (Euclidean on the unit sphere is monotone in cosine), which tends to give more balanced cluster sizes. Average linkage isolates singletons and long-tail items more readily. Ward tends to absorb them.
- To use the dendrogram for codebook granularity: cut at a fine threshold to get draft entries, then show reviewers the next merge level ("these two entries merge at distance d") as merge suggestions. This makes "over-cluster then merge" concrete.
- ClusterLLM's pairwise-LLM granularity selection can be reproduced cheaply with a local LLM: ask "same issue?" only for the pairs of cluster representatives at the candidate cut levels.

### Gaps
- I found no 2023–2026 paper that benchmarks agglomerative against HDBSCAN specifically for codebook construction. The recommendation rests on general properties.

---

## 4. BERTopic features relevant here (version status as of late 2026)

### Takeaway
BERTopic's latest release is 0.17.4 (2025-12-03). I found no 2026 release. It natively supports zero-shot assignment to predefined topics with a similarity threshold (the "assign to existing codebook + discover new" pattern), guided/seeded modeling, model merging across corpora (`merge_models`), topic reduction, outlier reassignment, swappable clustering (k-means/agglomerative), and local LLM labeling via llama.cpp. All of it runs locally with precomputed embeddings.

### Cited Findings
- Releases: 0.16.0 (2023-11-26) added `.merge_models`, zero-shot topic modeling, seed words in `ClassTfidfTransformer`, and a **LlamaCPP representation model**. 0.16.3 (2024-07-22) simplified zero-shot. 0.16.4 (2024-10-09) fixed guided modeling. 0.17.0 (2025-03-19) added the light install without UMAP/HDBSCAN, Model2Vec, LiteLLM representation, DataMapPlot and Python 3.13, and dropped 3.8. 0.17.1/0.17.3 (2025-07-08) added FastEmbed and LangChain backends and fixed `uv add bertopic` (explicit llvmlite dependency). 0.17.4 (2025-12-03) added `.delete_topics`, made plotly optional and dropped Python 3.9. "No 2026 releases" in the changelog — [BERTopic changelog](https://maartengr.github.io/BERTopic/changelog.html); [Zenodo record](https://zenodo.org/records/15834999)
- **Zero-shot topic modeling**: `zeroshot_topic_list` (predefined topic names; "making sure this is as descriptive as possible helps… since they are based on cosine similarities between embeddings") and `zeroshot_min_similarity` (documents below the threshold "skip zero-shot assignment and enter regular clustering"). The docs example uses 0.85, and "setting… requires a bit of experimentation" because embedding models differ. Caveat: the result merges two models, so `probs` start empty; run `.transform` afterwards — [BERTopic zero-shot docs](https://maartengr.github.io/BERTopic/getting_started/zeroshot/zeroshot.html)
- **Guided topic modeling** (`seed_topic_list`): seed topics are embedded and compared by cosine similarity to the documents. Matching documents get that seed's label and the rest get -1. "These labels are then passed through UMAP to create a semi-supervised approach that should nudge the topic creation to the seeded topics." Seed words also get IDF multipliers in c-TF-IDF — [BERTopic guided docs](https://maartengr.github.io/BERTopic/getting_started/guided/guided.html)
- Topic count control: `nr_topics="auto"`, `min_topic_size`, `n_neighbors` — [BERTopic FAQ](https://maartengr.github.io/BERTopic/faq.html)

### Inferences
- Use zero-shot rather than guided for codebook carry-over. Zero-shot hard-assigns to existing entries above a threshold and clusters the remainder, which is exactly "existing codebook + novelty". Guided only nudges UMAP, so seeded topics can still drift or split.
- Use the full codebook definition text (name + definition + 1–2 example gists) as each `zeroshot_topic_list` entry, not just the name. This follows the docs' "as descriptive as possible".
- `merge_models` (0.16.0+) is the built-in way to carry topics across videos: train per video, merge into a running model, and new topics are appended only if they are dissimilar enough. From memory (unverified), it takes a `min_similarity` parameter (default around 0.7).
- Local LLM representation: BERTopic's `LlamaCPP` representation (or LiteLLM/OpenAI-compatible backends pointed at a local llama.cpp server) can generate the topic names and draft definitions. That replaces the team's existing LLM-naming step inside one framework. BERTopic accepts precomputed embeddings, so the GPU can host the LLM while embeddings are computed beforehand.
- Korean: BERTopic's default c-TF-IDF `CountVectorizer` tokenizes on whitespace, which is poor for Korean agglutinative text. Use a Korean tokenizer (for example Kiwi/Mecab) in the vectorizer, or rely on LLM representations instead of c-TF-IDF keywords. (Background, unverified.)

### Gaps
- Default value of `zeroshot_min_similarity` not confirmed (docs say no explicit default; I recall 0.7 — unverified).
- `merge_models` signature and default threshold not verified this session.

---

## 5. LLM-in-the-loop clustering methods: evidence, cost, local-model feasibility

### Takeaway
The literature's best-supported cheap LLM uses are: (a) LLM-rewritten representations of each text before embedding (keyphrases/summaries; IDAS, Viswanathan et al.). The team's "gist" step already does this. (b) LLM-refined centroids or labels with few calls (k-LLMmeans). (c) LLM adjudication of only the ambiguous or edge points (LLMEdgeRefine). (d) Generate-labels-then-classify (TopicGPT, "clustering as classification", LLM-MemCluster). Almost all published results use GPT-3.5/4-class APIs. The direct evidence for 7–8B local models is that they are fine at **assigning** texts to given labels but poor at **generating/merging** label sets (TopicGPT). 20–30B-class local models are untested in these papers.

### Cited Findings
- **ClusterLLM** (Zhang et al., EMNLP 2023; arXiv May 2023): ChatGPT answers hard triplet questions ("does A better correspond to B than C"), and the answers are used to fine-tune a small embedder. Pairwise "same category?" questions choose the hierarchy level (granularity). Cost ≈ **$0.6 per dataset** across 14 datasets — [arXiv 2305.14871](https://arxiv.org/abs/2305.14871). **[API-only evidence; needs embedder fine-tuning]**. Code: github.com/zhang-yu-wei/ClusterLLM (URL unverified)
- **Large Language Models Enable Few-Shot Clustering** (Viswanathan, Gashteovski, Lawrence, Wu, Neubig; TACL vol. 12, 2024, pp. 321–333; arXiv July 2023) — [ACL Anthology](https://aclanthology.org/2024.tacl-1.18). LLM used before clustering (keyphrase expansion concatenated to the embedding), during clustering (pairwise constraints → PCKMeans) and after clustering (correction). Uses GPT-3.5-turbo-0301. **Keyphrase expansion was best** and cheapest: for example $1.23 (Bank77), $0.95 (CLINC), $0.99 (Tweet). Pairwise constraints cost $9.77–$42.03 per dataset with about 20,000 pairwise queries. Instruction-tuned encoders alone (Instructor) were "insufficient to support detailed, task-specific prompts". Code: [github.com/viswavi/few-shot-clustering](https://github.com/viswavi/few-shot-clustering) — [arXiv 2307.00524](https://arxiv.org/html/2307.00524). **[API-only evidence]**
- **IDAS** (De Raedt et al., NLP4ConvAI @ ACL 2023): the LLM produces abstractive-summary "labels" for each utterance, bootstrapped with in-context prototypes. Labels plus utterances are encoded by a frozen encoder and clustered. Beats SOTA by up to +7.42 on Banking/StackOverflow/Transport — [arXiv 2305.19783](https://arxiv.org/abs/2305.19783). k-LLMmeans reports IDAS needs about 4,650 prompts on CLINC — [k-LLMmeans](https://arxiv.org/html/2502.09667v2)
- **LLMEdgeRefine** (Feng, Lin, Wang, Cheng, Wong; EMNLP 2024): k-means → "super-point" re-clustering to reduce outlier impact → LLM reassigns only edge/boundary points. Efficient because only edge points are queried — [ACL Anthology 2024.emnlp-main.1025](https://aclanthology.org/2024.emnlp-main.1025). On GoEmo it reached 34.8% ACC vs k-LLMmeans 24.2%, using 895 vs 135 prompts — [k-LLMmeans](https://arxiv.org/html/2502.09667v2)
- **k-LLMmeans** (Diaz-Rodriguez, York Univ.; arXiv 2502.09667, v1 2025-02-12, v2 2025-05-09): every *l* iterations, centroids are replaced by an embedding of an LLM summary of the cluster (few-shot variant: summarize m=10 sampled members, "generally superior" to whole clusters). LLM calls scale with k·(number of summary rounds), **not with n** (~750 prompts on CLINC vs 1,618 for ClusterLLM and 4,650 for IDAS). Includes a mini-batch streaming variant. Tested GPT-3.5-turbo, GPT-4o, Claude-3.7, **Llama-3.3 (70B)**, DeepSeek-V3. Requires known k. No code URL in the paper — [arXiv 2502.09667v2](https://arxiv.org/html/2502.09667v2)
- **Text Clustering as Classification with LLMs** (Huang & He; arXiv 2410.00927, v1 2024-09-30, v3 2025-10-07; preprint, no venue found): the LLM generates candidate labels over mini-batches (batch size 15), then is prompted to "merge labels with similar expressions", then classifies each text into a label. **No k required.** GPT-3.5-turbo only, with no open-source LLM tests. Beats k-means (E5/Instructor), IDAS, PAS and keyphrase clustering, and ClusterLLM by +12.44% ACC on ArxivS2S — [arXiv html v1](https://arxiv.org/html/2410.00927v1); code [github.com/ECNU-Text-Computing/Text-Clustering-via-LLM](https://github.com/ECNU-Text-Computing/Text-Clustering-via-LLM). **[API-only evidence]**
- **TopicGPT** (Pham et al., NAACL 2024; arXiv 2311.01449): two stages, topic generation (with refinement and merging) then topic assignment. "Open-source LLMs are competent topic assigners but bad topic generators": Mistral-7B-Instruct "performs fairly well for topic assignment but cannot competently follow instructions for generating topics". The `topicgpt_python` package supports local inference via vLLM — [arXiv 2311.01449](https://arxiv.org/abs/2311.01449v2); [PyPI topicgpt-python](https://pypi.org/project/topicgpt-python/0.2.3)
- **LLM-MemCluster** (Zhu et al., UIC/W&M; arXiv 2511.15424, v2 2026-04-07): single-pass LLM-native clustering with a dynamic label memory in the prompt (reuse-or-create, merge-and-refine, retroactive update). A "relaxed" prompt switches to a "strict" one when the cluster count reaches K_max, so it needs a [K_min, K_max] range. GPT-4.1-mini averaged 45.4% ARI vs ClusterLLM 28.1%. DeepSeek-V3 got 38.7% and Gemini-2.0-flash 34.6%. Small open models were not named with sizes. Cost: about 26M input tokens for six datasets (≈$2.70 on Gemini Flash-Lite), dominated by the prompt context — [arXiv 2511.15424](https://arxiv.org/html/2511.15424). **[API/frontier evidence; long-context heavy]**
- **LLMs Enable Bag-of-Texts Representations for Short-Text Clustering** (Lin, Hasibi, Verberne; arXiv 2510.06747, Oct 2025, rev. Jan 2026): training-free, embedder-agnostic; LLM judgments build bag-of-texts representations. Claimed to work with "relatively small LLMs", k unknown — [arXiv 2510.06747](https://arxiv.org/abs/2510.06747). (Model names/sizes not verified)
- **Reasoning-Based Refinement of Unsupervised Text Clusters with LLMs** (T. Islam; arXiv 2604.07562, April 2026; ACL 2026 Findings): an LLM acting as judge refines the output of *any* clustering algorithm in three steps: coherence verification, redundancy adjudication (merge/remove overlapping clusters) and label grounding. Evaluated on social-media data from two platforms, with strong human agreement on labels — [arXiv 2604.07562](https://arxiv.org/abs/2604.07562). (LLMs used not verified)
- **Agentic Clustering** (Löwe & Silcock; arXiv 2606.01255, May 2026, rev. Oct 2026): an orchestrator LLM plus proposer/synthesizer/auditor/investigator/critic agents. User supplies a k-range and a free-text "clustering lens". Claims +47% mean ARI over the strongest prior baseline on seven benchmarks. Released as a Claude Code plugin — [arXiv 2606.01255](https://arxiv.org/abs/2606.01255). **[Likely frontier-API dependent — Claude plugin; local-model compatibility unverified]**
- **RAILS: Retrieval-Augmented Incremental LLM Clustering at Scale** (arXiv 2609.24464, 2026) appeared in search results only — [arXiv 2609.24464](https://arxiv.org/pdf/2609.24464) (content **unverified**)

### Inferences
- Local feasibility (8–30B via llama.cpp, short gists):
  - **Feasible and well-supported**: per-item LLM rewriting before embedding (already done as gists; equivalent to IDAS/keyphrase expansion). Per-item **assignment** to a given label list (TopicGPT shows 7B can do this; ~2k items × 1 short call is cheap locally). k-LLMmeans-style centroid summaries (O(k) calls per round; summarizing 10 gists is easy for 8–14B). Naming and definition drafting per cluster (the current practice).
  - **Feasible with care**: LLMEdgeRefine (only edge points, a few hundred calls); pairwise "same issue?" checks between cluster representatives for merge suggestions (k² / 2 pairs, but prune to the nearest-neighbor clusters).
  - **Weak at 7–8B, plausible at ~27–32B (untested)**: free label generation plus global merging (TCAC, TopicGPT generation, LLM-MemCluster), which need long-context consistency over many labels. Published evidence is GPT-3.5/4-class. Running this at 8B will likely produce duplicate or near-duplicate labels. Mitigate by doing merges by embedding similarity plus human review rather than by the LLM alone.
  - **Not practical**: ClusterLLM-style embedder fine-tuning on a GPU shared with the LLM (possible but adds complexity for small gain at n ≤ 2k). Viswanathan PCKMeans with ~20k pairwise queries (too many calls for little benefit; keyphrase expansion was better anyway).
- Because the input is already LLM-written gists, much of the "LLM-enriched representation" gain is already captured. The marginal benefit of more elaborate LLM clustering is likely smallest on these normalized texts. The highest-value LLM step remains cluster-level labeling, definition, include/exclude drafting, and merge/split adjudication.

### Gaps
- No paper found that evaluates these methods with 8–30B local models on Korean text. TopicGPT's Mistral-7B finding is the only direct small-model evidence I verified.
- ClusterLLM and LLM-MemCluster code URLs and small-model results were not verified.
- The Bag-of-Texts paper's actual model sizes were not verified.

---

## 6. Seeded / anchored clustering and carrying a codebook across videos

### Takeaway
The simplest robust scheme is nearest-definition assignment with a novelty threshold: embed each codebook entry, assign a gist to the closest entry if cosine ≥ τ (optionally with an LLM yes/no verification), and cluster the rest to discover new issues. BERTopic's zero-shot mode implements exactly this. Constrained k-means (COP-KMeans) and seeded k-means are alternatives, but they are less suited to an open-ended long tail.

### Cited Findings
- BERTopic zero-shot implements "assign to predefined topics if similarity ≥ `zeroshot_min_similarity`, else cluster normally to discover new topics". The threshold must be tuned per embedding model (example 0.85) — [BERTopic zero-shot](https://maartengr.github.io/BERTopic/getting_started/zeroshot/zeroshot.html)
- BERTopic guided modeling nudges clusters toward seeds via semi-supervised UMAP labels, but does not hard-assign — [BERTopic guided](https://maartengr.github.io/BERTopic/getting_started/guided/guided.html)
- COP-KMeans assigns each point to the nearest centroid that violates no must-link/cannot-link constraint. Cannot-link constraints can make feasibility NP-complete. Python implementation available in `clumppy`; R has `conclust` — [clumppy PyPI](https://pypi.org/project/clumppy/); [conclust manual](https://cran.r-universe.dev/conclust/doc/manual.html); [constrained clustering overview](https://every-algorithm.github.io/2024/11/30/constrained_clustering.html)
- Seeded k-means uses labeled seed examples to initialize centroids so clusters align with known categories from the first iteration — [constrained clustering overview](https://every-algorithm.github.io/2024/11/30/constrained_clustering.html)
- Viswanathan et al. used LLM-generated pairwise constraints with PCKMeans (soft constraint penalty), which was costlier and weaker than keyphrase expansion — [arXiv 2307.00524](https://arxiv.org/html/2307.00524)
- TopicGPT shows that small open models can reliably assign to a fixed topic list — [arXiv 2311.01449](https://arxiv.org/abs/2311.01449v2)
- IDAS's semi-supervised result (beating 2 methods on CLINC without labeled data) shows that prototype-seeded labeling works for intent discovery — [arXiv 2305.19783](https://arxiv.org/abs/2305.19783)

### Inferences
- Suggested cross-video scheme (inference):
  1. Maintain each codebook entry's **anchor vector**: the mean of the definition embedding and the embeddings of approved example gists. Prototype vectors from real examples are more reliable than definition text alone, because definitions are stylistically different from gists.
  2. For a new video, compute the cosine of each gist to every anchor. Set a per-entry threshold τ_e from the distribution of approved members (for example the 5th–10th percentile of member-to-anchor cosine), not one global constant.
  3. Gists above τ go to the nearest entry. Gists in an "ambiguous band" (top-2 margin < δ, or just below τ) go to a local LLM for a yes/no or multiple-choice check against the include/exclude rules. This is TopicGPT-style assignment, which 7–8B models handle.
  4. The remainder (novel) is clustered with HDBSCAN or agglomerative (section 2/3). New clusters are drafted into candidate entries and sent to human review.
  5. After approval, update anchors. Optionally periodically check entry–entry anchor cosine to propose merges.
- Seeded or COP k-means forces every point into a seeded cluster unless k is increased. That is poor for "new issue" discovery, so use it only when the codebook is believed complete. Using existing anchors as fixed initial centroids plus extra free centroids is a middle ground (k = |codebook| + m).
- Multilingual note: if Korean and English gists coexist, check that the embedding model aligns them cross-lingually (for example bge-m3, multilingual-e5). Otherwise anchors will split by language. Alternatively, have the gist step always output in one language. (Background, unverified.)

### Gaps
- I found no published evaluation of the "anchor + novelty threshold" scheme for qualitative codebooks across documents/videos. The scheme is inferred from BERTopic zero-shot and the TopicGPT assignment results.
- `clumppy`'s maturity and maintenance were not checked.

---

## 7. Over-cluster then merge vs under-cluster then split (human-reviewed codebook)

### Takeaway
For a human-reviewed codebook with a long tail, the evidence and tooling favor **over-clustering then merging**. Merges are cheap to judge (compare two definitions plus examples) and can be suggested by embedding similarity, a dendrogram, or LLM redundancy checks. Splits require re-reading many members and are where rare issues get lost.

### Cited Findings
- LLM-generated-label methods systematically over-generate then merge: TCAC generates labels per batch of 15 and then merges similar labels — [arXiv 2410.00927](https://arxiv.org/html/2410.00927v1). LLM-MemCluster has explicit merge-and-refine and relaxed→strict modes, with merges "remarkably rare (averaging fewer than 2 per dataset)" — [arXiv 2511.15424](https://arxiv.org/html/2511.15424). The ACL 2026 Findings refinement framework has a "redundancy adjudication" stage that merges or removes overlapping clusters — [arXiv 2604.07562](https://arxiv.org/abs/2604.07562)
- HDBSCAN `leaf` selection gives "many small homogeneous clusters", and `cluster_selection_epsilon` merges near micro-clusters — [HDBSCAN docs](https://hdbscan.readthedocs.io/en/latest/parameter_selection.html)
- BERTopic supports post-hoc reduction/merging (`nr_topics`, merge) and, since 0.17.4, `.delete_topics` — [BERTopic FAQ](https://maartengr.github.io/BERTopic/faq.html); [changelog](https://maartengr.github.io/BERTopic/changelog.html)
- ClusterLLM chooses granularity by checking a hierarchy against LLM pairwise judgments — [arXiv 2305.14871](https://arxiv.org/abs/2305.14871)

### Inferences
- Recommended local pipeline for n = 300–2,000 gists (inference):
  1. Gist step outputs `{issue_gist | NO_ISSUE}` and language-normalized text. Drop NO_ISSUE.
  2. Embed with a multilingual model and L2-normalize.
  3. If a codebook exists, run the anchor assignment from section 6.
  4. On the remainder, over-cluster: agglomerative (average/cosine, tight threshold) or HDBSCAN (`min_cluster_size` 3–5, `leaf`). Keep singletons and noise as a "long-tail" bucket.
  5. Local LLM drafts name, definition and include/exclude for each cluster from about 10 sampled gists (k-LLMmeans' finding that 10 samples beat full clusters supports small samples).
  6. Compute entry–entry similarity (definition embeddings plus centroid cosine) and dendrogram merge order. Present the top merge candidates to the human. Optionally run a local LLM "same issue?" check on candidate pairs first.
  7. Human reviews, merges and approves. Approved entries become anchors for the next video.
- Under-cluster then split is preferable only if review capacity is tiny and only the few dominant issues matter. It hides the long tail by design.
- Track over-clustering cost by the number of review decisions. With n ≈ 2,000, around 40–80 draft clusters is a manageable review load (inference).

### Gaps
- No user study found that compares merge-first vs split-first review effort for qualitative codebooks. The recommendation is inferred from method design and tooling.
