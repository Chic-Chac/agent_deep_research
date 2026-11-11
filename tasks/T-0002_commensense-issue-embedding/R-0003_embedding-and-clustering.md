---
id: R-0003
type: report
task: T-0002
title: "Keep Qwen3-4B, cluster agglomeratively, test Korean"
status: final
created: 2026-11-02
tags: [embeddings, clustering, korean]
uses: [N-0009, N-0010]
---

# Keep Qwen3-4B, cluster agglomeratively, test Korean

**Bottom line:** Keep **Qwen/Qwen3-Embedding-4B** (bf16, ~8 GB). Give it an English issue-focused instruction, then over-cluster the L2-normalized gist embeddings with **cosine average-linkage agglomerative clustering** and merge the result with human review. On the public multilingual clustering benchmark this model leads every open model that fits in 16 GB without quantization (**MMTEB clustering 57.15 vs BGE-M3 40.88**), which matches your EXP-005 gap. The one unanswered question is Korean. **No public Korean clustering numbers exist for any candidate model.** The only Korean head-to-head that includes a Qwen3 embedder (the 0.6B, on retrieval) has it *below* BGE-M3 and its Korean fine-tunes. One ~200-gist Korean run through your existing EXP-005 harness settles this before anything else changes. Embeddings remain an issue-level tool only. Claim-level assignment stays with the local LLM chooser, as EXP-005 showed (0.82–0.86 vs ~0.4).

## 1. Ranked shortlist

**Embedding model**

1. **`Qwen/Qwen3-Embedding-4B`**: it has the best MMTEB clustering score of any model that fits unquantized (57.15). Going to the 8B adds only +0.5 and does not fit in bf16. It already won EXP-005, and it is Apache-2.0 ([card](https://huggingface.co/Qwen/Qwen3-Embedding-4B)).
2. **`Qwen/Qwen3-VL-Embedding-2B`**: it uses half the VRAM (~4 GB) and tied the 4B on your 263 English comments. Publicly it is weaker (clustering 52.50, mean 63.87 vs 69.45), so the tie is probably within noise. It also needs a heavier stack (`torch==2.8.0`, `transformers>=4.57`) ([card](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B)).
3. **`dragonkue/snowflake-arctic-embed-l-v2.0-ko`** (Korean control, not a default): it posts the best Korean retrieval average found (NDCG@10 0.7404 vs Qwen3-0.6B 0.6762) ([card](https://huggingface.co/dragonkue/snowflake-arctic-embed-l-v2.0-ko)). Include it in the Korean test so you can see whether the Qwen models lose on Korean. Its family clusters poorly on MMTEB (arctic-m-v2.0: 42.2) ([EmbeddingGemma paper](https://arxiv.org/html/2509.20354)), so expect it to lose unless Korean is the deciding factor.

**Prompt (Qwen template `Instruct: {task}\nQuery:{gist}`, written in English even for Korean gists, as Qwen recommends)**

- **Default:** `Instruct: Identify the specific issue discussed in this YouTube comment summary, regardless of the commenter's stance, sentiment, or the person targeted\nQuery:`. This is adapted from the MTEB clustering prompts and is untested on your data.
- **Ablation arms:** run the MTEB-style generic prompt `Identify the topic or theme of the given comments`, and no instruction (for VL-2B, its default `Represent the user's input.`). Qwen reports **+1–5%** from instructions in general ([card](https://huggingface.co/Qwen/Qwen3-Embedding-4B)). Aspect-steering instructions ("ignore stance") are unreliable in the literature ([InBedder](https://arxiv.org/abs/2402.09642); [Myntti et al. 2026](https://arxiv.org/abs/2610.10508)), so the stance-neutral, name-anonymized gist does the real work and the prompt only adds a little.

**Clustering**

1. **Agglomerative, `metric="cosine"`, `linkage="average"`, `distance_threshold` tuned to over-cluster (≈ n/50 to n/25 clusters), no UMAP.** It is deterministic and needs no k. Its dendrogram turns "over-cluster then merge" into a ranked list of merge suggestions, and at n ≤ 2k it runs in under a second ([sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html)). Average linkage leaves rare issues as small clusters or singletons instead of absorbing them. Cosine AHC also slightly beat k-means for aspect clustering of opinions ([DOC, Findings ACL 2023](https://aclanthology.org/2023.findings-acl.115.pdf)).
2. **HDBSCAN (`min_cluster_size` 3–8 at n≈300, 5–15 at n≈2k; `min_samples` 1–3; `cluster_selection_method="leaf"`), optionally after UMAP (`n_neighbors` 10–30, `n_components` 5–10, `min_dist=0`, fixed seed).** It is the only option with a built-in noise class, which works as a "no-issue / long-tail" review queue. On short texts, though, it can mark a majority of documents as outliers ([de Groot et al. 2022](https://arxiv.org/abs/2212.08459)), and UMAP adds seed variance ([UMAP docs](https://umap-learn.readthedocs.io/en/latest/clustering.html)).
3. **k-means on L2-normalized vectors with a large k (n/40 to n/10), flagging low cosine-to-centroid points as "unassigned".** This is your current sampler baseline: stable and outlier-free, but k must be chosen, and no-issue gists get pulled into real clusters ([BERTopic FAQ](https://maartengr.github.io/BERTopic/faq.html)).

**Default combination:** option 1 for the model, the issue instruction for the prompt, and agglomerative clustering, with HDBSCAN run once alongside it as a check that you are not losing the long tail.

## 2. Fit to constraints

| Constraint | Qwen3-Emb-4B (default) | Qwen3-VL-Emb-2B | arctic-l-v2.0-ko | Clustering stack |
|---|---|---|---|---|
| 16 GB GPU | ~8 GB weights + ~1–2 GB activations (estimate). **Cannot share the GPU with the 27B GGUF LLM**: embed in a separate pass, save `.npy`, free VRAM, then load llama.cpp. | ~4 GB | ~1.1 GB | CPU only |
| Python 3.12 + uv | `transformers>=4.51.0`, `sentence-transformers>=2.7.0`; FA2 optional (use SDPA + `padding_side="left"` on WSL) | `transformers>=4.57.0`, `qwen-vl-utils>=0.0.14`, **`torch==2.8.0` hard pin**: keep it in its own uv project | standard sentence-transformers; prefix `query: ` | scikit-learn (agglomerative); `hdbscan` + `umap-learn`; optional `bertopic==0.17.4` (2025-12-03; `uv add` fixed in 0.17.3) ([changelog](https://maartengr.github.io/BERTopic/changelog.html)) |
| Pinned versions | Pin the HF commit with `revision=<sha>` and record `huggingface_hub.model_info(id).sha`. **SHAs were not retrieved in this research**, so commit `uv.lock` with the resolved library versions. | same | same | `uv.lock` |
| Korean + English | 100+ languages; **no Korean clustering or retrieval number published for 4B** | **none found** | Korean-tuned; best Korean retrieval found | Korean tokenization only matters if you use BERTopic c-TF-IDF keywords. Use LLM-written labels instead. |
| No text leaves the machine | Weights download only | same | same | Agglomerative/HDBSCAN are local. If you use BERTopic, name topics with its local `LlamaCPP` representation, **not** LiteLLM/OpenAI backends. Agentic Clustering ([arXiv 2606.01255](https://arxiv.org/abs/2606.01255)) ships as a Claude plugin and is excluded. |
| License | Apache-2.0 | Apache-2.0 | Apache-2.0 | BSD / MIT |

## 3. Smallest confirming experiment (≈1–2 days)

The language of the gists determines how much the Korean risk matters. If the gist LLM writes every gist in **English**, EXP-005 already covers the embedding side and the Korean risk moves to the gist step (Part 1). If gists are in **Korean**, or left mixed, run Step 1 first.

**Step 1: Korean embedding check (reuse the EXP-005 harness).** Take ~200 gists from one Korean-majority video, label them with a topic option set the same way as hQR, and run nearest-option zero-shot top-1 for the 4B, VL-2B, arctic-ko and BGE-M3 (BGE-M3 links the result back to EXP-005). Run all three prompt arms on the instruction-aware models. Embedding 200 short texts takes seconds per arm. Compare models with a **paired McNemar test on the same items**. At n≈200 and accuracy ≈0.7, an unpaired 95% CI is about ±6 points. That also means the English 2B = 4B tie at n=263 is unresolved, not proven equal (my arithmetic). **Decision rule:** keep the 4B unless another model beats it on Korean with p<0.05. If the VL-2B ties again on both languages, prefer whichever stack is simpler to maintain.

**Step 2: Clustering check (same Korean video plus hQR).** With the winning model and prompt, run agglomerative, HDBSCAN and k-means at **matched K ≈ n/25**, because pairwise precision rises with K. Score them on **~200 human "same issue?" pairs**, sampled across cosine strata with extra weight on pairs the methods disagree about, and compute pairwise P/R/F1 (design in Part 3). Report these confound diagnostics, which address the EXP-005 lesson about clusters forming around polarity and target:
- NMI(cluster, stance label)
- NMI(cluster, target person)
- NMI(cluster, language)
- noise or singleton rate
- ARI across 10 seeds for the stochastic methods

**Decision rule:** keep agglomerative unless HDBSCAN wins on pairwise F1 at matched K. If NMI with stance, target or language is high, fix the gist (neutral wording, name placeholders, single pivot language) before changing the clusterer. Person names dominate embedding similarity ([name-bias paper, Findings ACL 2025](https://arxiv.org/abs/2502.02903)), and multilingual embedders partly cluster by language ([Cloudflight blog, secondary](https://www.cloudflight.io/en/blog/language-bias-in-multilingual-semantic-search-systems/)).

## 4. Details on demand

**Model comparison (MMTEB v2; VRAM figures are weights-only estimates)**

| Model | Params | VRAM bf16 | MMTEB mean | Clustering | Korean evidence (dragonkue Ko retrieval NDCG@10) | License |
|---|---|---|---|---|---|---|
| Qwen3-Embedding-8B | 8B | ~16 GB (int8 ~8) | 70.58 | 57.65 | none | Apache-2.0 |
| **Qwen3-Embedding-4B** | 4B | ~8 GB | 69.45 | **57.15** | none | Apache-2.0 |
| KaLM-Embedding-Gemma3-12B-2511 | 11.8B | ~23.5 GB (int8 ~12) | 72.32 | 55.77 | none | custom Tencent |
| Qwen3-VL-Embedding-2B | 2B | ~4 GB | 63.87 | 52.50 | none | Apache-2.0 |
| Qwen3-Embedding-0.6B | 0.6B | ~1.2 GB | 64.33 | 52.33 | 0.6762 | Apache-2.0 |
| embeddinggemma-300m | 0.3B | ~0.6 GB | 61.15 | 51.17 | none | Gemma terms |
| multilingual-e5-large-instruct | 0.56B | ~1.1 GB | 63.22 | 50.75 | 0.6984 | MIT [unverified] |
| jina-embeddings-v3 | 0.57B | ~1.1 GB | 58.37 | 45.65 | 0.7013 | **CC-BY-NC** |
| BGE-M3 | 0.57B | ~1.1 GB | 59.56 | 40.88 | 0.7242 | MIT |
| harrier-oss-v1-0.6b | 0.6B | ~1.2 GB | 69.0 | **not published** | none | MIT |
| arctic-embed-l-v2.0-ko | 0.6B | ~1.1 GB | n/a | n/a | **0.7404** | Apache-2.0 |

Sources: [Qwen3-4B card](https://huggingface.co/Qwen/Qwen3-Embedding-4B), [KaLM card](https://huggingface.co/tencent/KaLM-Embedding-Gemma3-12B-2511), [VL-2B card](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B), [EmbeddingGemma paper](https://arxiv.org/html/2509.20354), [harrier card](https://huggingface.co/microsoft/harrier-oss-v1-0.6b), [arctic-ko card](https://huggingface.co/dragonkue/snowflake-arctic-embed-l-v2.0-ko). Within the Qwen3 family, clustering gains level off after 4B (0.6B → 4B +4.8, 4B → 8B +0.5). Across families, model size predicts clustering poorly. **harrier-oss-v1-0.6b** is the one untested wildcard: MIT license, overall score near the 4B at a seventh of the size, but no clustering split published. Add it as a fourth arm in Step 1 if you want a cheap, fast alternative. KURE-v1 (BGE-M3 fine-tune) reaches 0.7616 on 9 MTEB(kor) retrieval tasks ([KURE](https://github.com/nlpai-lab/KURE)), but its BGE-M3 base clusters poorly.

**Why the prompt matters less than the gist.** InBedder showed that instruction-tuned encoders barely change clustering when given a different aspect instruction (INSTRUCTOR: NYT topic vs location 49.96 vs 50.07 without instruction), while embedding an LLM's *answer* to the instruction follows the aspect far better ([arXiv 2402.09642](https://arxiv.org/abs/2402.09642)). Your gist pipeline already works this way. Qwen3 may follow instructions better than older embedders, but that claim comes from a search snippet only **[unverified]**.

**Codebook carry-over across videos (after the first video).** Assign each new gist to the nearest codebook anchor, where an anchor is the mean of the definition embedding and approved example embeddings. Use a per-entry threshold at the 5th–10th percentile of member cosine. Send gists in the ambiguous band to the local LLM for a yes/no check, and cluster the remainder with option 1. BERTopic's `zeroshot_topic_list` + `zeroshot_min_similarity` implements the same pattern; its example threshold is 0.85 and it must be tuned per model ([docs](https://maartengr.github.io/BERTopic/getting_started/zeroshot/zeroshot.html)). Small local LLMs are reliable *assigners* but weak *topic generators* ([TopicGPT, NAACL 2024](https://arxiv.org/abs/2311.01449v2)). So the LLM should name clusters from ~10 sampled gists, which is the k-LLMmeans finding ([arXiv 2502.09667](https://arxiv.org/html/2502.09667v2)), and judge merge candidates. It should not generate the label set globally.

**Not recommended here:**
- Qwen3-8B / KaLM-12B / harrier-27B: they need quantization and have no clustering advantage.
- jina v3/v4/v5: non-commercial licenses.
- ClusterLLM-style embedder fine-tuning and ~20k-query pairwise PCKMeans: too costly at n ≤ 2k ([Viswanathan et al., TACL 2024](https://aclanthology.org/2024.tacl-1.18)).
- Silhouette or Davies–Bouldin for comparing *models*: they are only valid within one embedding space (Part 3).

## 5. Sources

| Source | Date | Used for |
|---|---|---|
| [Qwen3-Embedding-4B model card](https://huggingface.co/Qwen/Qwen3-Embedding-4B) | Jun 2025 (read 2026-11-02) | specs, MMTEB table, instruction template, +1–5%, library minimums |
| [Qwen3-Embedding task_prompts.json](https://media.githubusercontent.com/media/QwenLM/Qwen3-Embedding/main/evaluation/task_prompts.json) | read 2026-11-02 | exact MTEB clustering prompts |
| [Qwen3-VL-Embedding-2B card](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B) | released 2026-01-08 | specs, clustering 52.50, torch/transformers pins |
| [KaLM-Embedding-Gemma3-12B-2511 card](https://huggingface.co/tencent/KaLM-Embedding-Gemma3-12B-2511) | Nov 2025 | cross-model MMTEB clustering table |
| [EmbeddingGemma paper, arXiv 2509.20354](https://arxiv.org/html/2509.20354) | Sep 2025 | BGE-M3 / jina / gte / arctic-m clustering |
| [harrier-oss-v1-0.6b card](https://huggingface.co/microsoft/harrier-oss-v1-0.6b) | undated (read 2026-11-02) | MTEB v2 69.0, MIT |
| [dragonkue/snowflake-arctic-embed-l-v2.0-ko card](https://huggingface.co/dragonkue/snowflake-arctic-embed-l-v2.0-ko) | undated (read 2026-11-02) | Korean retrieval table (retrieval only, not clustering) |
| [nlpai-lab/KURE](https://github.com/nlpai-lab/KURE) | read 2026-11-02 | MTEB(kor) retrieval |
| [InBedder, arXiv 2402.09642](https://arxiv.org/abs/2402.09642) | ACL 2024 | instructions vs embedding answers |
| [Myntti et al., arXiv 2610.10508](https://arxiv.org/abs/2610.10508) | 2026-10-07 | instruction-following failures; Qwen3 claim **[unverified snippet]** |
| [DOC, Findings ACL 2023](https://aclanthology.org/2023.findings-acl.115.pdf) | 2023 | stance splits; AHC vs k-means |
| [Name bias, arXiv 2502.02903](https://arxiv.org/abs/2502.02903) | Findings ACL 2025 | person names dominate similarity |
| [de Groot et al., arXiv 2212.08459](https://arxiv.org/abs/2212.08459) | Dec 2022 | HDBSCAN outliers on short text; k-means similar quality. The "up to 74% outliers" figure is **unverified**. |
| [scikit-learn AgglomerativeClustering](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html) | read 2026-11-02 | parameters |
| [HDBSCAN parameter selection](https://hdbscan.readthedocs.io/en/latest/parameter_selection.html), [UMAP for clustering](https://umap-learn.readthedocs.io/en/latest/clustering.html) | read 2026-11-02 | parameters, caveats. The numeric grids above are **inference, not sourced**. |
| [BERTopic changelog](https://maartengr.github.io/BERTopic/changelog.html), [FAQ](https://maartengr.github.io/BERTopic/faq.html), [zero-shot docs](https://maartengr.github.io/BERTopic/getting_started/zeroshot/zeroshot.html) | 0.17.4 = 2025-12-03; no 2026 release | versions, outlier/small-data notes, carry-over |
| [TopicGPT, arXiv 2311.01449](https://arxiv.org/abs/2311.01449v2) | NAACL 2024 | small LLMs assign well, generate poorly |
| [k-LLMmeans, arXiv 2502.09667](https://arxiv.org/html/2502.09667v2) | May 2025 | ~10-sample cluster summaries |
| [Viswanathan et al., TACL 2024](https://aclanthology.org/2024.tacl-1.18) | 2024 | pairwise-constraint cost |
| [Cloudflight blog](https://www.cloudflight.io/en/blog/language-bias-in-multilingual-semantic-search-systems/) | read 2026-11-02 | language-split clustering (**secondary source**) |

**Unverified or missing:**
- HF commit SHAs for all models.
- Live MTEB leaderboard snapshot: the numbers here come from model cards and papers.
- Any Korean *clustering* benchmark.
- Clustering scores for harrier, jina v4/v5 and arctic-l-v2.0.
- The parameter ranges, the n/25 target K, and the instruction wording, all of which are reasoned rather than sourced.
