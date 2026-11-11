---
id: N-0009
type: note
task: T-0002
title: "Open-weight Korean+English embedding models for short-text issue clustering on a 16 GB GPU (as of Nov 2026)"
status: final
created: 2026-11-02
tags: [embeddings, model-selection, korean]
---

# Open-weight Korean+English embedding models for short-text issue clustering on a 16 GB GPU (as of Nov 2026)

Research date: 2026-11-02. Web research only; no models were downloaded or run. All leaderboard numbers are dated as given by their source. The live MTEB leaderboard (Gradio app) was not fetched directly, so numbers come from model cards and papers that copy it. Re-check them on huggingface.co/spaces/mteb/leaderboard before quoting them as final.

## Q1. Comparison table: specs, license, MMTEB clustering, Korean scores

### Takeaway
On MMTEB (Multilingual, v2) clustering, the Qwen3-Embedding family is still the strongest set of open models that fit in 16 GB: 4B = 57.15, 0.6B = 52.33, and 8B = 57.65, though 8B only fits when quantized. BGE-M3 scores only 40.88, which matches the team's finding that Qwen beat BGE-M3 by 8–10 points. Two newer families need a look. Microsoft harrier-oss-v1-0.6b (MIT license, MTEB v2 69.0) comes close to Qwen3-4B's overall mean at a seventh of the size, but I found no per-category clustering number for it. KaLM-Embedding-Gemma3-12B is #1 on the overall mean (72.32) but scores lower on clustering (55.77), needs int8 to fit, and has a custom license. No public Korean clustering benchmark results exist for any of these models. The Korean evidence is retrieval only, and there Qwen3-Embedding-0.6B is weaker than BGE-M3 and the Korean fine-tunes of BGE-M3.

### Master table

VRAM column: this is my estimate of weights only (params × 2 bytes for bf16, × 1 byte for int8). Activations for about 2k short texts at batch size 32–128 add little, roughly 1–2 GB. These figures are arithmetic inferences, not vendor figures.

| Model (HF ID) | Params | Est. VRAM bf16 / int8 | Max len | Dim / MRL | License | Instructions / prompts | MMTEB Mean(Task) | MMTEB Clustering | Korean evidence (dragonkue Ko retrieval avg NDCG@10, 8 tasks) |
|---|---|---|---|---|---|---|---|---|---|
| Qwen/Qwen3-Embedding-0.6B | 0.6B | ~1.2 GB / ~0.6 GB | 32K | 1024 (MRL, custom dims) [unverified dim; card for 4B says 32–2560 for 4B] | Apache-2.0 | `Instruct: {task}\nQuery:{text}` | 64.33 (Jun 2025) | 52.33 | 0.6762 (below bge-m3 0.7242) |
| Qwen/Qwen3-Embedding-4B | 4B (36 layers) | ~8 GB / ~4 GB | 32K | up to 2560, MRL 32–2560 | Apache-2.0 | same | 69.45 | 57.15 | none found |
| Qwen/Qwen3-Embedding-8B | 8B | ~16 GB (does not fit 16 GB with overhead) / ~8 GB int8 | 32K | 4096 per my knowledge; one aggregator says 7,168 (conflict, see below) | Apache-2.0 | same | 70.58 (#1 on Jun 5 2025) | 57.65 | none found |
| Qwen/Qwen3-VL-Embedding-2B (released 2026-01-08) | 2B (28 layers) | ~4 GB / ~2 GB | 32K | up to 2048, MRL 64–2048 | Apache-2.0 | instruction-aware; default "Represent the user's input." | 63.87 | 52.50 | none found |
| microsoft/harrier-oss-v1-270m | 270M | ~0.5 GB | 32,768 | 640 | MIT | `Instruct: {task}\nQuery: {q}` (query side only) | 66.5 (MTEB v2, date not stated) | not published on card | none |
| microsoft/harrier-oss-v1-0.6b | 0.6B | ~1.2 GB | 32,768 | 1024 | MIT | same | 69.0 | not published on card | none |
| microsoft/harrier-oss-v1-27b | 27B | ~54 GB / ~27 GB (does not fit; int4 ≈ 14 GB, very tight) | 32,768 | 5376 | MIT | same | 74.3 | not published | none |
| tencent/KaLM-Embedding-Gemma3-12B-2511 | 11.76B | ~23.5 GB / ~12 GB int8 | 32K | 3840; MRL 3840/2048/1024/512/256/128/64 | tencent-kalm-embedding-community (custom) | `Instruct: {task} \nQuery: ` | 72.32 (#1 Nov 2025) | 55.77 | none |
| google/embeddinggemma-300m | 308M | ~0.6 GB (bf16/fp32 only; no fp16) | 2048 | 768; MRL 512/256/128 | Gemma Terms of Use | `task: clustering \| query: {content}` | 61.15 | 51.17 | none |
| intfloat/multilingual-e5-large-instruct | ~0.6B (560M) | ~1.1 GB | 512 [from memory, unverified] | 1024 [unverified] | MIT [unverified] | `Instruct: {task}\nQuery: {text}` | 63.22 | 50.75 | 0.6984 |
| BAAI/bge-m3 | 568M | ~1.1 GB | 8,192 | 1024 (no MRL) | MIT | none | 59.56 | 40.88 | 0.7242 |
| jinaai/jina-embeddings-v3 | 572M | ~1.1 GB | 8,192 | 1024, MRL to 32 | CC-BY-NC-4.0 (non-commercial) | task LoRA adapters (incl. separation/clustering) | 58.37 | 45.65 | 0.7013 |
| jinaai/jina-embeddings-v4 | ~4B (Qwen2.5-VL-3B base) | ~8 GB | 32,768 | 2048; MRL 128–2048 (+128 multi-vector) | Qwen Research License (card says the earlier CC-BY-NC tag was an error); non-commercial | adapters: retrieval / text-matching / code | not found | not found | none |
| jina-embeddings-v5-text-small / -nano (released 2026-02-19) | 677M / 239M | ~1.4 GB / ~0.5 GB | 32K / 8K | 1024 (small) | CC BY-NC 4.0 | 4 task LoRA adapters (incl. clustering) | 67.0 / 65.5 (131 tasks) | not found | none |
| Alibaba-NLP/gte-multilingual-base | 305M | ~0.6 GB | 8192 [unverified] | 768 elastic [unverified] | Apache-2.0 [unverified] | none | 58.24 | 44.33 | 0.6638 |
| Snowflake/snowflake-arctic-embed-l-v2.0 | ~568M (XLM-R large) | ~1.1 GB | 8192 | 1024 (MRL 256 [unverified]) | Apache-2.0 | prefix `query: ` | not found (m-v2.0: 53.7, clust 42.2) | not found | 0.7241 |
| nomic-ai/nomic-embed-text-v2-moe | 475M total / 305M active | ~1 GB | 512 | 768, MRL to 256 | Apache-2.0 | prefixes `search_query: ` / `search_document: `; trust_remote_code, needs einops+megablocks | not found | not found | none |
| nlpai-lab/KURE-v1 | 568M (bge-m3 fine-tune) | ~1.1 GB | 8192 (inherits bge-m3) [inferred] | 1024 [inferred] | MIT | none | n/a | n/a | 0.7277; KURE repo avg NDCG@10 0.7616 on 9 MTEB(kor,v2) retrieval tasks |
| dragonkue/BGE-m3-ko | 568M | ~1.1 GB | 8192 [inferred] | 1024 [inferred] | not checked | none | n/a | n/a | 0.7300 |
| dragonkue/snowflake-arctic-embed-l-v2.0-ko | 0.6B | ~1.1 GB | 8192 | 1024 | Apache-2.0 | `query: ` | n/a | n/a | 0.7404 (best in that table) |
| upskyy/bge-m3-korean | 568M | ~1.1 GB | – | 1024 | not checked | none | n/a | n/a | 0.6567 |
| KURE-v2 (late-interaction, 154M) / KURE-v3 (dense, 1.1B) | 154M / 1.1B | small | – | – | not checked | – | n/a | n/a | KURE repo: 0.8160 / 0.8027 avg NDCG@10 |

Sources for the table rows:
- Qwen3-Embedding specs, instruction template, versions, and MMTEB per-category table (Qwen3 0.6B/4B/8B, e5-large-instruct, gte-Qwen2-7B, gemini) — [Qwen3-Embedding-4B model card](https://huggingface.co/Qwen/Qwen3-Embedding-4B)
- 8B was "No.1 in the MTEB multilingual leaderboard (as of June 5, 2025, score 70.58)" — [Qwen3-Embedding-4B card](https://huggingface.co/Qwen/Qwen3-Embedding-4B)
- KaLM-Embedding-Gemma3-12B-2511: 11.76B params, 3840 dim, MRL list, custom license, prompt format, MMTEB table with Clustering 55.77 (KaLM), 54.35 (llama-embed-nemotron-8b), 57.65 (Qwen3-8B), 54.59 (gemini-embedding-001), 57.15 (Qwen3-4B), 52.33 (Qwen3-0.6B), 52.77 (gte-Qwen2-7B), 50.60 (Linq-Embed-Mistral), 50.75 (e5-large-instruct), 51.17 (embeddinggemma-300m) — [KaLM model card](https://huggingface.co/tencent/KaLM-Embedding-Gemma3-12B-2511)
- Qwen3-VL-Embedding-2B: released Jan 8 2026, 2B params / 28 layers, 32K, dim up to 2048 (MRL 64–2048), default instruction "Represent the user's input.", MMTEB mean(task) 63.87 / mean(type) 55.84 / clustering 52.50, transformers ≥4.57.0, qwen-vl-utils ≥0.0.14, torch==2.8.0, FA2 recommended, Apache-2.0 — [Qwen3-VL-Embedding-2B card](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B)
- harrier-oss-v1: sizes 270M/0.6B/27B, dims 640/1024/5376, 32,768 tokens, MTEB v2 66.5/69.0/74.3, MIT, `Instruct: {task_description}\nQuery: {query}` on query side only, 94 languages, bf16 — [harrier-oss-v1-0.6b card](https://huggingface.co/microsoft/harrier-oss-v1-0.6b); [harrier-oss-v1-27b card](https://huggingface.co/microsoft/harrier-oss-v1-27b); [Azure AI Labs harrier page](https://labs.ai.azure.com/projects/harrier-oss-v1/)
- An aggregator says harrier-oss-v1-27b had "the highest multilingual retrieval score of any open-source embedding model" (78.27 multilingual retrieval) as of Oct 16 2026, and that Qwen3-Embedding-8B ranked 4th on MTEB Multilingual v2 as of Oct 27 2026. This comes from a search-result summary of blog posts, not the leaderboard itself, so treat it as unverified — [supermemory blog](https://supermemory.ai/blog/best-open-source-embedding-models-benchmarked-and-ranked/); [maximem blog](https://www.maximem.ai/blog/best-embedding-models-2026)
- EmbeddingGemma-300m: 768 dim, MRL 512/256/128, 2048 tokens, Gemma license, "activations do not support float16", clustering prompt `task: clustering | query: {content}`, MMTEB 61.15 / 54.31 — [EmbeddingGemma card](https://huggingface.co/google/embeddinggemma-300m)
- MMTEB v2 numbers from EmbeddingGemma paper Tables 5/6 (Sept 2025): BGE-M3 568M mean 59.56, clustering 40.88; jina-v3 572M 58.37 / 45.65; gte-multilingual-base 305M 58.24 / 44.33; snowflake-arctic-embed-m-v2.0 305M 53.7 / 42.2; Qwen3-0.6B 595M 64.34 / 52.33; EmbeddingGemma 308M 61.15 / 51.17 — [EmbeddingGemma paper, arXiv 2509.20354](https://arxiv.org/html/2509.20354)
- jina-embeddings-v4: 4B, Qwen2.5-VL-3B-Instruct base, 32,768 tokens, 2048 dim, MRL 128–2048, adapters retrieval/text-matching/code, transformers ≥4.52.0, FA2 optional. License: "initially released under cc-by-nc-4.0 due to an error. The correct license is the Qwen Research License" — [jina-embeddings-v4 card](https://huggingface.co/jinaai/jina-embeddings-v4)
- jina-embeddings-v5-text-small (677M, Qwen3-0.6B-Base) and -nano (239M, EuroBERT-210m): released Feb 19 2026, MMTEB 67.0 / 65.5, CC BY-NC 4.0, 32K / 8K context — [Jina v5 announcement](https://jina.ai/news/jina-embeddings-v5-text-distilling-4b-quality-into-sub-1b-multilingual-embeddings)
- jina-v3 CC-BY-NC-4.0, 8192, 1024 dim; BGE-M3 568M, MIT, 8192, 1024 — [PremAI blog](https://www.premai.io/blog/best-embedding-models-for-rag-2026-ranked-by-mteb-score-cost-and-self-hosting/)
- Snowflake arctic-embed-l-v2.0: Apache-2.0, XLM-R based, 74 languages — [Snowflake card](https://huggingface.co/Snowflake/snowflake-arctic-embed-l-v2.0). Its Korean fine-tune: 0.6B, 8192, 1024, `query: ` prefix, Apache-2.0 — [dragonkue/snowflake-arctic-embed-l-v2.0-ko card](https://huggingface.co/dragonkue/snowflake-arctic-embed-l-v2.0-ko)
- nomic-embed-text-v2-moe: 475M total / 305M active, 512 tokens, 768 dim with MRL to 256, Apache-2.0, needs einops + megablocks + trust_remote_code, prefixes `search_query: ` / `search_document: `, ~100 languages, BEIR 52.86, MIRACL 65.80 — [nomic card](https://huggingface.co/nomic-ai/nomic-embed-text-v2-moe)
- Korean retrieval table (NDCG@10 averaged over MrTidy, MIRACL, XPQA, Belebele, PublicHealthQA, AutoRAG, Ko-StrategyQA): arctic-l-v2.0-ko 0.7404, BGE-m3-ko 0.7300, KURE-v1 0.7277, bge-m3 0.7242, arctic-l-v2.0 0.7241, mE5-large 0.7216, KoE5 0.7114, bge-multilingual-gemma2 0.7043, jina-v3 0.7013, mE5-large-instruct 0.6984, Qwen3-Embedding-0.6B 0.6762, gte-multilingual-base 0.6638, upskyy/bge-m3-korean 0.6567. Retrieval only, no clustering — [dragonkue/snowflake-arctic-embed-l-v2.0-ko card](https://huggingface.co/dragonkue/snowflake-arctic-embed-l-v2.0-ko)
- KURE repo: KURE-v1 is a fine-tune of BAAI/bge-m3, 568M, MIT. On 9 MTEB(kor, v2) retrieval tasks, avg NDCG@10 is: KURE-v2 (late-interaction, 154M) 0.8160, KURE-v3 (dense, 1.1B) 0.8027, KURE-v1 0.7616, BGE-m3-ko 0.7547, bge-m3 0.7509, KoE5 0.7337. None of the tasks are clustering — [nlpai-lab/KURE GitHub](https://github.com/nlpai-lab/KURE)
- Octen-Embedding-8B is described as ranking #7 on MTEB with avg 67.85 (aggregator snippet, unverified; it is an 8B, so it would need int8 here) — [search result summary / codesota MTEB registry](https://www.codesota.com/benchmarks/mteb)

### Inferences
- The team's EXP-005 result matches the public clustering numbers. On MMTEB clustering, Qwen3-4B (57.15) and Qwen3-VL-Embedding-2B (52.50) lead BGE-M3 (40.88) by 12–16 points. On a 263-comment English set, an 8–10 point gap is about what you would expect.
- The 2B VL model tying the 4B text model on 263 English comments is plausible but probably within noise. On MMTEB clustering the 4B leads 57.15 vs 52.50, and on the overall MMTEB mean 69.45 vs 63.87. A 263-item, single-language test cannot reliably separate gaps that small.
- Qwen3-Embedding-0.6B retrieves Korean worse than BGE-M3 and its Korean fine-tunes (0.676 vs 0.724–0.740). This is retrieval, not clustering, but it is the only head-to-head Korean data found for a Qwen3 embedder. I found no Korean numbers for Qwen3-4B, 8B, or VL-2B. Korean behaviour of the Qwen models is the main open risk and should be tested on the team's own Korean gists.
- The harrier 270m dims (640) and 27b dims (5376) match Gemma 3 hidden sizes, while 0.6b (1024) matches Qwen3-0.6B. The Azure page says "Qwen3 backbone." This conflict could not be resolved; the base model is probably mixed per size.
- Qwen3-Embedding-8B in bf16 needs about 15 GiB for weights alone, which does not realistically fit a 16 GB card with CUDA context and activations. Running it in int8 (bitsandbytes) or a GGUF Q8 build should fit at about 8–9 GB, at some quality and speed cost. The clustering gain over 4B is only +0.50 (57.65 vs 57.15), so 8B is not worth it here.
- KaLM-Gemma3-12B (int8 ≈ 12 GB) and harrier-27B (int4 ≈ 14 GB) can only run quantized. Neither has a published clustering advantage over Qwen3-4B: KaLM's clustering is 55.77 < 57.15, and harrier's is unknown.
- License flags: jina v3, v4, and v5 are non-commercial (CC-BY-NC / Qwen Research License). EmbeddingGemma uses the Gemma terms, and KaLM uses a custom Tencent community license. Qwen3 (Apache-2.0), harrier (MIT), BGE-M3 / KURE-v1 (MIT), arctic, gte, and nomic (Apache-2.0) are permissive.

### Gaps
- Exact HF commit revisions (SHA) were not retrieved for any model. Pin them at download time with `revision=` and check `huggingface_hub.model_info(id).sha`.
- No per-category (clustering) MMTEB score was found for harrier-oss-v1, jina v4/v5, arctic-embed-l-v2.0 (only m-v2.0), nomic-v2-moe, or the Korean fine-tunes.
- I could not fetch the live MTEB leaderboard (Gradio), so I have no late-2026 rank snapshot from the primary source. The Oct 2026 rankings above come from blog aggregators.
- I found no Korean clustering benchmark results for any model. I could not confirm which Korean clustering tasks exist in MMTEB. SIB200ClusteringS2S is a multilingual MMTEB clustering task and I believe it includes kor_Hang, but this is unverified. KURE's MTEB(kor, v2) set and NanoMTEB-Korean (5 tasks: autorag, ko_strategy_qa, lawir_ko, miracl_ko, squad_kor_v1) are retrieval only — [NanoMTEB-Korean docs](https://hakari-bench-leaderboard.hf.space/docs/benchmark-tasks/NanoMTEB-Korean).
- Conflict: one aggregator lists Qwen3-Embedding-8B dim as "7,168 (flex to 32)" ([PremAI](https://www.premai.io/blog/best-embedding-models-for-rag-2026-ranked-by-mteb-score-cost-and-self-hosting/)). My understanding is 4096, but I did not verify it on the 8B card.
- multilingual-e5-large-instruct max length / dim / license, gte-multilingual-base length / dim / license, and the KURE-v1 / BGE-m3-ko length and dim are from memory or inferred from the base model, not fetched.

## Q2. Instruction / task prompts for clustering (exact MTEB strings) and evidence that custom instructions help

### Takeaway
Qwen3-Embedding, e5-instruct, harrier, and KaLM all use the same query template, `Instruct: {task}\nQuery:{text}`. The MTEB clustering prompts are short "Identify the topic or theme of …" sentences. Qwen reports that instructions give +1–5% on most tasks and recommends writing them in English. I found no published evidence that a specific aspect instruction such as "identify the issue, ignore the opinion" works with these general models. The research that does exist (InBedder) was motivated by off-the-shelf instruction embedders not reliably capturing user-specified aspects, so the team should A/B test this themselves.

### Cited Findings
- Qwen3 template: `Instruct: {task_description}\nQuery:{query_text}`. "For most downstream tasks, using instructions typically yields an improvement of 1% to 5% compared to not using them". Instructions should be task-specific and ideally written in English — [Qwen3-Embedding-4B card](https://huggingface.co/Qwen/Qwen3-Embedding-4B)
- Qwen3 MTEB evaluation clustering prompts (evaluation/task_prompts.json):
  - SIB200ClusteringS2S: "Identify the category of documents"
  - MasakhaNEWSClusteringS2S: "Identify the topic or theme of the given news articles based on the titles"
  - WikiClusteringP2P.v2: "Identify the category of wiki passages"
  - RedditClustering: "Identify the topic or theme of Reddit posts based on the titles"
  - StackExchangeClustering: "Identify the topic or theme of StackExchange posts based on the titles"
  - TwentyNewsgroupsClustering: "Identify the topic or theme of the given news articles"
  - ArXivHierarchicalClusteringS2S: "Identify the main and secondary category of Arxiv papers based on the titles"
  - The same file has Korean classification prompts, e.g. KorHateSpeechMLClassification: "Given a Korean online news comments, classify its fine-grained hate speech classes"
  - Source: [QwenLM/Qwen3-Embedding task_prompts.json](https://media.githubusercontent.com/media/QwenLM/Qwen3-Embedding/main/evaluation/task_prompts.json)
- e5 (multilingual-e5-large-instruct) template `"Instruct: {}\nQuery: "`, with clustering instructions such as:
  - RedditClustering: "Identify the topic or theme of Reddit posts based on the titles"
  - StackExchangeClustering: "Identify the topic or theme of StackExchange posts based on the titles"
  - TwentyNewsgroupsClustering: "Identify the topic or theme of the given news articles"
  - ThuNewsClusteringS2S: "Identify the topic or theme of the given news articles based on the titles"
  - ArxivClusteringS2S: "Identify the main and secondary category of Arxiv papers based on the titles"
  - Source: [microsoft/unilm e5/utils.py](https://raw.githubusercontent.com/microsoft/unilm/master/e5/utils.py)
- EmbeddingGemma clustering prompt: `task: clustering | query: {content}` — [EmbeddingGemma card](https://huggingface.co/google/embeddinggemma-300m)
- harrier: `Instruct: {task_description}\nQuery: {query}` on the query side; built-in named prompts include `web_search_query`, `sts_query`, `bitext_query` (no named clustering prompt) — [harrier-oss-v1-27b card](https://huggingface.co/microsoft/harrier-oss-v1-27b)
- KaLM: `Instruct: {task} \nQuery: `, e.g. "Instruct: Classifying the category of french news.\nQuery:" — [KaLM card](https://huggingface.co/tencent/KaLM-Embedding-Gemma3-12B-2511)
- Qwen3-VL-Embedding default instruction: "Represent the user's input." — [Qwen3-VL-Embedding-2B card](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B)
- InBedder (ACL 2024) "aims to build a text embedder that can capture characteristics of texts specified by user instructions… none of previous approaches provides a concrete solution for it". It shows that clustering the same corpus with different instructions gives different, interpretable clusters, and it tests robustness to correct, implicit, and incorrect instructions — [arXiv 2402.09642](https://arxiv.org/abs/2402.09642)

### Inferences
- For clustering, the instruction goes on every text, because every text is treated as a "query". Nothing goes on the document side because there is none.
- A reasonable starting point that mirrors the MTEB style: `Instruct: Identify the specific issue or topic discussed in the YouTube comment, regardless of the commenter's opinion or sentiment\nQuery:{gist}`. Write it in English even for Korean text, as Qwen recommends. Compare it against (a) no instruction and (b) the generic MTEB-style "Identify the topic or theme of the given comments". The team's own V-measure / ARI against a labeled subset should decide.
- Because the inputs are already LLM-written issue gists, the gist step has already removed most of the opinion. The instruction's marginal effect is likely smaller than on raw comments. This is an inference; it has not been tested.

### Gaps
- I found no quantitative published study showing that aspect-steering instructions ("ignore the opinion") improve clustering for Qwen3, e5-instruct, or harrier specifically.
- The Qwen3 technical report's ablation of instruction vs no instruction for clustering specifically was not retrieved.

## Q3. Does a larger model beat a smaller one on short-text clustering? (MMTEB clustering by size)

### Takeaway
Within one family, size helps up to about 4B and then levels off. Qwen3: 0.6B 52.33 → 4B 57.15 (+4.8) → 8B 57.65 (+0.5). Across families, size is a poor predictor. KaLM-12B (55.77) and llama-embed-nemotron-8B (54.35) score below Qwen3-4B on clustering, and gte-Qwen2-7B (52.77) is level with Qwen3-0.6B. Training recipe matters more than size.

### Cited Findings
- MMTEB clustering by size:

  | Model | Size | Clustering |
  |---|---|---|
  | embeddinggemma | 308M | 51.17 |
  | gte-multilingual-base | 305M | 44.33 |
  | snowflake-arctic-embed-m-v2.0 | 305M | 42.2 |
  | BGE-M3 | 568M | 40.88 |
  | jina-v3 | 572M | 45.65 |
  | multilingual-e5-large-instruct | 0.6B | 50.75 |
  | Qwen3 | 0.6B | 52.33 |
  | Qwen3-VL-Emb | 2B | 52.50 |
  | Qwen3 | 4B | 57.15 |
  | gte-Qwen2 | 7B | 52.77 |
  | Linq-Embed-Mistral | 7B | 50.60 |
  | llama-embed-nemotron | 8B | 54.35 |
  | Qwen3 | 8B | 57.65 |
  | KaLM-Gemma3 | 12B | 55.77 |
  | gemini-embedding-001 | API | 54.59 |

  Sources: [KaLM card](https://huggingface.co/tencent/KaLM-Embedding-Gemma3-12B-2511); [EmbeddingGemma paper](https://arxiv.org/html/2509.20354); [Qwen3-VL-Embedding-2B card](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B); [Qwen3-Embedding-4B card](https://huggingface.co/Qwen/Qwen3-Embedding-4B)
- harrier overall MTEB v2 means rise with size (270M 66.5, 0.6B 69.0, 27B 74.3), but no clustering split is published — [harrier-oss-v1-0.6b card](https://huggingface.co/microsoft/harrier-oss-v1-0.6b)

### Inferences
- MMTEB clustering is a mix of S2S (titles, short) and P2P (longer) tasks, so it is a reasonable but imperfect proxy for clustering short issue gists.
- On a 16 GB card, Qwen3-Embedding-4B in bf16 is the clustering sweet spot. The candidates worth testing against it:
  - harrier-oss-v1-0.6b: MIT, much faster, and its overall MMTEB is near 4B, but clustering is unknown.
  - Qwen3-VL-Embedding-2B: the team says it already matches 4B on their data.
  - For Korean, a Korean-tuned BGE-M3 derivative such as KURE-v1 or dragonkue/snowflake-arctic-embed-l-v2.0-ko as a Korean-side control. It is strong on Korean retrieval but probably weak on clustering, given BGE-M3's 40.88.

### Gaps
- No size-scaling clustering data for Korean specifically.
- No short-text-only (S2S) clustering breakdown by model was retrieved.

## Q4. Practical notes: library versions, flash-attention, throughput for ~2k short texts

### Takeaway
Qwen3-Embedding needs transformers ≥4.51 and sentence-transformers ≥2.7. Qwen3-VL-Embedding needs transformers ≥4.57, torch 2.8, and qwen-vl-utils. Flash-Attention 2 is recommended but optional for all the Qwen and jina models. EmbeddingGemma must run in bf16 or fp32, not fp16. For about 2k short texts, any model in this list should take seconds to about a minute on a 16 GB GPU, so throughput is not a deciding factor (my estimate, not benchmarked).

### Cited Findings
- Qwen3-Embedding: transformers ≥4.51.0, sentence-transformers ≥2.7.0. "We recommend enabling flash_attention_2 for better acceleration and memory saving, together with setting `padding_side` to 'left'" — [Qwen3-Embedding-4B card](https://huggingface.co/Qwen/Qwen3-Embedding-4B)
- Qwen3-VL-Embedding-2B: transformers ≥4.57.0, qwen-vl-utils ≥0.0.14, torch==2.8.0, FA2 recommended — [card](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B)
- jina-v4: transformers ≥4.52.0, FA2 "recommended… but not mandatory" — [card](https://huggingface.co/jinaai/jina-embeddings-v4)
- EmbeddingGemma: "activations do not support `float16`. Please use `float32` or `bfloat16`" — [card](https://huggingface.co/google/embeddinggemma-300m)
- KaLM: BF16, FA2 optional — [card](https://huggingface.co/tencent/KaLM-Embedding-Gemma3-12B-2511)
- nomic-v2-moe: needs `trust_remote_code=True`, einops, and megablocks — [card](https://huggingface.co/nomic-ai/nomic-embed-text-v2-moe)

### Inferences
- Throughput estimate: about 2k texts × about 30–80 tokens is roughly 60k–160k tokens. A 4B model in bf16 on a modern 16 GB GPU should handle this in tens of seconds at batch 32–64. A 0.6B model should take a few seconds. This is a back-of-envelope estimate, not a measured benchmark.
- Without FA2 (common on WSL or consumer setups), use SDPA attention with left padding for the last-token-pooling Qwen and harrier models. Short texts keep padding waste small.
- Make sure each model's prompt is actually applied: pass `prompt=` in `model.encode`, or use `encode_query`. Silently dropping the instruction would cost the reported 1–5%.

### Gaps
- No measured throughput numbers for these models on 16 GB GPUs were found.
- I did not find minimum transformers / sentence-transformers versions for harrier-oss-v1.
