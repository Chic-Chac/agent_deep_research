---
id: N-0006
type: note
task: T-0002
title: "What to embed for issue-level clustering of short comments: raw text vs LLM gist vs canonical proposition vs aspect/keypoint extraction"
status: final
created: 2026-11-02
tags: [issue-normalization, key-point-analysis]
---

# What to embed for issue-level clustering of short comments: raw text vs LLM gist vs canonical proposition vs aspect/keypoint extraction

Scope note: "Verified" means I read the primary paper (HTML or extracted PDF text) in this session. Items marked [UNVERIFIED] come only from search snippets, abstracts, or my prior knowledge. Dates are arXiv or venue dates. Today is 2026-11-02.

---

## Q1. Key Point Analysis (KPA) and argument mining: how arguments are normalized before matching or clustering, and what gains are reported

### Takeaway
Classic IBM KPA does not rewrite arguments. It selects short, self-contained argument sentences as key-point candidates and learns a matcher, so its "normalization" is a filter (short, no pronouns, high quality). LLM-era KPA moves the normalization upstream. The best-supported pattern for opinionated user text is PAKPA (ACL 2024): extract structured (aspect, sentiment) pairs per comment, cluster on the aspect only, keep sentiment as a separate partition, then generate key points. This beat clustering on whole-sentence embeddings for quantification precision, and an all-open 7B configuration came close to GPT-3.5 hybrids.

### Cited Findings
- **IBM KPA / ArgKP-2021 normalization by filtering.** Top ArgKP-2021 systems: BarH "filters arguments by token length and pronoun presence, scores candidates by quality, then reduces redundancy using match thresholds". SMatchToPr uses similar filtering and ranks candidates with a PageRank variant over a match graph. Key points are matched to arguments with match scorers fine-tuned on ArgKP. — [Argument Summarization and its Evaluation in the Era of LLMs (EMNLP 2025; arXiv 2503.00847)](https://arxiv.org/html/2503.00847v4)
- **Clustering-based KPA.** Li et al. (ACL 2023, "key point analysis via iterative clustering and abstractive summarisation") clusters argument embeddings (BERTopic-style: embeddings, UMAP, HDBSCAN, iterated) and summarizes each group into a key point with instruction-tuned FLAN-T5. Khosravani et al. (NAACL 2024) cluster with a match scorer and choose an extractive key point with a supervised scorer. — [ArgSum survey, EMNLP 2025](https://arxiv.org/html/2503.00847v4); [Khosravani et al., NAACL 2024](https://aclanthology.org/2024.naacl-long.454.pdf)
- **LLM-era KPA results.** The EMNLP 2025 ArgSum study tested GPT-4o, LLaMA-3.3-70B, Qwen2.5-72B and Qwen3-32B. Qwen3-32B had the highest weighted scores (0.923 on ArgKP21, 0.899 on Debate), "even surpassing GPT-4o". LLM end-to-end summarization beat hybrid systems. The paper proposes MCArgSum, which combines match-scorer clustering with LLM summarization. LLM-based evaluation correlated with humans at 0.767 (coverage) and 0.852 (redundancy). — [arXiv 2503.00847v4](https://arxiv.org/html/2503.00847v4) (numbers come from the fetch summary; I did not check them against the table myself)
- **Fukuma et al. (2024)** showed that zero-shot KPA is feasible using LLM question generation plus network analysis. — [ArgSum survey](https://arxiv.org/html/2503.00847v4) [details UNVERIFIED]
- **PAKPA pipeline (Tang, Zhang, Dinh, Cambria; ACL 2024, pp. 10691–10708). Verified.**
  1. Prompted aspect-based sentiment analysis (ABSA) runs per review sentence and outputs a JSON list of (aspect term, sentiment) pairs. The prompt has 5 parts: context, task definition, JSON format request, 18 few-shot examples, and the input.
  2. Neutral pairs are dropped. Aspects are grouped by sentiment polarity and clustered greedily. Each aspect joins the existing cluster with the highest average cosine similarity (spaCy vectors) if that similarity is ≥ 0.55; otherwise it starts a new cluster.
  3. An LLM generates one key point per cluster. That prompt includes "invalid generation examples to avoid", and cluster size gives the prevalence count.
  - The stated motivation: "clustering comments by only their sentence-level embeddings cannot accurately identify distinct KPs on different aspects, leading to inaccurate quantification." — [ACL Anthology PDF](https://aclanthology.org/2024.acl-long.576.pdf)
- **PAKPA gains. Verified.**
  - YELP quantification (matching) precision, averaged over 5 business categories: PAKPA 0.95 vs ABKPA 0.83, SKPMBase(IC)+ (Li et al. 2023 sentence-embedding clustering) 0.76, RKPA-Base (IBM) 0.66, Enigma+ 0.64.
  - Human Bradley-Terry faithfulness: 26.56 vs at most 22.28 for the baselines. The authors describe this as "more than doubling the effectiveness in reducing hallucinations" compared with other abstractive systems.
  - SPACE Soft-F1 (BARTScore): all-open Mistral-7B+Mistral-7B = 0.71, best hybrid Vicuna-7B+GPT-3.5 = 0.70, RKPA-Base = 0.66, SKPM = 0.62. On ROUGE the GPT-3.5 hybrid is better (R-1 0.648 vs 0.531). — [ACL 2024 PDF](https://aclanthology.org/2024.acl-long.576.pdf)
- **PAKPA error modes. Verified.**
  - Key points pick up extra content when specific and general aspects merge into one cluster. Example: "coffee" clustered with "breakfast" produced "Overpriced breakfast with mediocre coffee".
  - Some clusters are over-generalized, so minority opinions in the cluster get lost. — [ACL 2024 PDF](https://aclanthology.org/2024.acl-long.576.pdf)
- A 2026 preprint, "Key Point Analysis Needs Structure Recovery: Task Definition, Dataset Diagnosis, and a Structure-Aware Benchmark" (arXiv 2608.25854), argues that KPA datasets need structural diagnosis. — [arXiv PDF](https://arxiv.org/pdf/2608.25854) [content UNVERIFIED; I saw only the title]

### Inferences
- PAKPA is the closest published analogue to the project's "issue facet". It normalizes each comment into a structured target/aspect and keeps polarity in a separate field. Clustering then runs on the aspect only, inside a sentiment partition. This matches the project's diagnosis that mixing target, issue and polarity in one representation was what broke earlier option sets.
- The KPA filters (short, no pronouns, single aspect) work as a quality spec for gists: self-contained, one issue, no anaphora.
- The coffee/breakfast failure maps directly onto "target vs issue" granularity. The issue field should name the issue at one consistent level of abstraction, and the target should be a separate field.

### Gaps
- I did not retrieve the ArgKP-2021 overview paper (Friedman et al., 2021) with the exact track-1 mAP numbers for the top systems.
- I did not read Li et al. 2023 directly, so its gains over IBM KPA are unverified.
- No KPA work found covers Korean or code-switched YouTube comments.

---

## Q2. TopicGPT, LLooM, GoalEx, ClusterLLM, IDAS/IntentGPT, and claim/proposition extraction: what each embeds or summarizes, and the evidence that distilling first helps

### Takeaway
Most "LLM-first" pipelines do one of two things. Either they produce a short per-document gist and embed and cluster it (LLooM: 2–4 bullets of 5–8 words; LiSA and IDAS: a label plus description), or they skip embedding and let the LLM propose and assign labels (TopicGPT, GoalEx, IntentGPT). Either way, the strongest variants depended on GPT-3.5/GPT-4. Evidence that distillation itself is the cause of the gains is thin: LLooM reports no ablation of its distill step. TopicGPT's own report shows a 7B open model breaking topic generation, while an open 7–8B model works when clustering does the merging (LiSA).

### Cited Findings
- **LLooM (Lam et al., CHI 2024; arXiv 2404.12259). Verified.**
  - Distill/Filter: GPT-3.5-turbo extracts exact quotes. Distill/Summarize: GPT-3.5-turbo writes "2–4 bullet points" of "5–8 words" each. Both steps return JSON.
  - Cluster: embed the bullets with text-embedding-ada-002, then HDBSCAN, chosen because it "does not require all points to be placed in a cluster".
  - Synthesize: GPT-4 writes concepts with inclusion criteria. Score: GPT-3.5 or PaLM.
  - Coverage: Bills 0.74 vs GPT-4 baseline 0.56 vs BERTopic 0.29 (p<.02). Wiki 0.81 vs GPT-4 0.83 vs BERTopic 0.63. Real-world scenarios: average 93% coverage vs 77.7% for BERTopic, and 2.0× as many high-quality topics.
  - Cost: $1.44 per run, 848,323 tokens, 13.7 min; scoring was 79.9% of the cost.
  - Limitations: no verification step for exact quotes or for bullets matching quotes; "variability in the results upon re-run"; default parameters "have not been robustly tested". — [arXiv HTML](https://arxiv.org/html/2404.12259)
- **LLooM example of the gain in abstraction.** For toxic comments, BERTopic gives "women, power, female", while LLooM gives "Criticism of traditional gender roles" and "Dismissal of women's concerns". — [arXiv abstract](https://arxiv.org/abs/2404.12259)
- **TopicGPT (Pham et al., NAACL 2024; arXiv 2311.01449). Verified.**
  - Generation: an LLM (GPT-4) reads samples and existing topics and outputs topics as "a concise label paired with a broad one-sentence description".
  - Refinement: merges topic pairs with Sentence-Transformer cosine ≥ 0.5 and drops rare topics.
  - Assignment: an LLM (GPT-3.5) outputs label, document-specific description and a supporting quote.
  - Purity P1: Wiki 0.74 vs LDA 0.64 vs BERTopic 0.58; Bills 0.57 vs 0.52 vs 0.39. Misaligned topics: 30.3% vs 62.4% for LDA.
  - Mistral-7B for assignment costs about 6 purity points. Mistral-7B for generation "failed substantially": 1,418 topics vs 151, mostly over-specific.
  - Evaluated in English only; the paper notes instruction-following degrades in non-English languages. API cost was about $88–$155 per dataset. **Depends on GPT-4-class models for generation.** — [arXiv HTML](https://arxiv.org/html/2311.01449); [repo](https://github.com/chtmp223/topicGPT)
- **LiSA (Liu, Shang, Ke et al., ACL 2025 long, pp. 18420–18435). Verified.**
  - An open LLM (Llama-3-8B-Instruct or Mistral-7B-Instruct) writes "a candidate topic word for each document, accompanied by a description".
  - Each topic word is represented by the mean embedding of its descriptions (all-MiniLM-L6-v2). Documents are embedded separately. Both are clustered with KMeans to K clusters and aligned with the Hungarian algorithm.
  - When a document's topic-word cluster and its document cluster disagree, an LLM picks among candidates using nearest neighbours.
  - The paper argues that LLM-only merging (TopicGPT/PromptTopic) is unstable and needs GPT-4-level granularity control.
  - Twitter (11,171 tweets, Antypas et al. 2022): LiSA (Llama-3) P1 0.78 / ARI 0.65 / NMI 0.46, vs BERTopic 0.53 / 0.18 / 0.23 and TopicGPT (GPT-4) P1 0.75. Bills: P1 0.59 vs TopicGPT 0.57.
  - A figure shows KMeans on Sentence-BERT document embeddings at ARI 49% vs 66% for LiSA's learned representation (dataset not stated in the excerpt).
  - Caveat: LiSA is "semi-supervised" and also trains a topic prediction network, so the gain is not only from embedding LLM text. — [ACL 2025 PDF](https://aclanthology.org/2025.acl-long.902.pdf); [code](https://github.com/ljh986/LiSA)
- **IDAS (De Raedt et al., 2023; arXiv 2305.19783).**
  - Clusters utterances on abstractive "labels" that "retain the core elements while removing non-essential information".
  - An LLM generates labels with in-context learning, bootstrapped from a seed set of prototypical utterances. Utterances and their noisy labels are encoded by a frozen encoder and then clustered.
  - Unsupervised gains of up to +7.42% on Banking, StackOverflow and Transport. — [arXiv](https://arxiv.org/abs/2305.19783v1); [DeepAI summary](https://deepai.org/publication/idas-intent-discovery-with-abstractive-summarization) [the exact LLM (a GPT-3 family model) and the venue (NLP4ConvAI 2023) are UNVERIFIED]
- **GoalEx (Wang, Shang, Zhong; EMNLP 2023; arXiv 2305.13749).**
  - Propose: given a user goal, the LLM "brainstorm[s] a list of explanations each representing a cluster".
  - Assign: each sample is classified against each explanation.
  - Select: integer linear programming maximizes coverage and minimizes overlap.
  - Reports "more accurate and goal-related explanations than prior methods". — [arXiv](https://arxiv.org/abs/2305.13749) [models and numbers UNVERIFIED; from memory, GPT-4/GPT-3.5 propose and a Flan-T5 assigns, which I treat as GPT-4-dependent]
- **ClusterLLM (Zhang et al., EMNLP 2023; arXiv 2305.14871).** Embeds raw text. ChatGPT answers hard triplet questions ("does A better correspond to B than C") used to fine-tune the embedder, and pairwise questions used to pick granularity. Cost is about $0.6 per dataset over 14 datasets, with consistent gains. — [arXiv](https://arxiv.org/abs/2305.14871). Reported accuracy: Bank77 71.2, CLINC 83.8 — [Viswanathan et al., TACL 2024, Table 2](https://aclanthology.org/2024.tacl-1.18.pdf)
- **IntentGPT (Rodriguez et al.; arXiv 2411.10670; ICLR 2024 LLM Agents workshop).** Training-free. An in-context prompt generator plus a semantic few-shot sampler lets the LLM directly predict or discover intents (CLINC, BANKING). Primarily uses **GPT-4**. — [arXiv](https://arxiv.org/abs/2411.10670)
- **Dense X Retrieval / Propositionizer (Chen et al., EMNLP 2024; arXiv 2312.06648). Verified via HTML.**
  - A proposition is atomic, distinct, and self-contained, with coreferences resolved.
  - GPT-4 generated propositions for 42k passages, which were distilled into a Flan-T5-large "Propositionizer".
  - Indexing propositions instead of passages raised Recall@5 by +12.0 (SimCSE) and +9.3 (Contriever), a 22–35% relative gain. Gains were smaller for supervised retrievers, and EntityQuestions improved +25% with DPR.
  - English Wikipedia only. — [arXiv HTML](https://arxiv.org/html/2312.06648)
- **Claimify (Metropolitansky & Larson, Microsoft; ACL 2025; arXiv 2502.10855).**
  - LLM claim extraction that "extract[s] claims only when there is high confidence in the correct interpretation of the source text".
  - Includes new metrics for coverage and decontextualization, and outperforms prior methods under that framework. — [arXiv](https://arxiv.org/abs/2502.10855) [The selection → disambiguation → decomposition stages and the model used are from memory, UNVERIFIED]
- **k-LLMmeans (Diaz-Rodriguez; arXiv 2502.09667, v5 Feb 2026).** Does not summarize individual texts. Each centroid is replaced by the embedding of an LLM summary of a sample of cluster members.
  - LLMs tested: GPT-3.5/4o, Llama-3.3, Claude-3.7, DeepSeek-V3.
  - CLINC accuracy 80.2 vs 77.3 for plain k-means (text-embedding-3-small).
  - LLM calls are independent of dataset size. — [arXiv HTML](https://arxiv.org/html/2502.09667); [code](https://github.com/jairoadiazr/summaryCentroids)

### Inferences
- The pattern that works locally with a 7–9B open model is a hybrid. The LLM writes a per-item gist or label (LiSA, IDAS, LLooM-distill), but embeddings plus clustering do the merging. The free-form "LLM invents and merges the taxonomy" step (TopicGPT generation, GoalEx propose) is where small models fail: Mistral-7B produced 1,418 topics vs 151.
- For Qwen3 via llama.cpp on 16 GB: per-comment gisting with a quantized 8B-class model is in the LiSA/PAKPA regime (7–8B, open). Cluster naming and codebook synthesis are the GPT-4-dependent steps in LLooM and TopicGPT, so those should get the strongest local model affordable plus human review.
- The proposition literature (Dense X, Claimify) adds two ideas. First, decontextualize: resolve "he", "this", "그 사람" to the actual target using video metadata. Second, abstain when the interpretation is uncertain. Both map onto handling sarcasm and memes.

### Gaps
- LLooM reports no ablation that removes the distill/summarize step, so its gains cannot be attributed to distillation versus GPT-4 synthesis. The fetch found none; treat as unverified absence.
- I did not verify GoalEx's numerical results or its models.
- None of these methods were evaluated on Korean or code-switched text.

---

## Q3. Direct comparisons (2023–2026): clustering raw short texts vs LLM-rewritten, summarized or expanded texts

### Takeaway
Direct head-to-head evidence exists, but almost all of it is on English intent and tweet benchmarks. Gains from LLM-generated per-item representations are real but modest and inconsistent: roughly 0–12 accuracy points, and occasionally negative. They are largest on noisy tweets and with agglomerative clustering. The most robust published recipe concatenates the LLM keyphrase/gist embedding with the raw-text embedding rather than replacing it. An alternative that avoids rewriting altogether, LeBoT (LLM judges similarity to anchors), gives larger gains on intent data with a 9B open model.

### Cited Findings
- **Viswanathan et al., "Large Language Models Enable Few-Shot Clustering" (arXiv July 2023; TACL 2024). Verified from the PDF text.**
  - Method: gpt-3.5-turbo-0301 generates, per document, "a comprehensive set of keyphrases that could describe its intent, as a JSON-formatted list". The prompt is an instruction plus 4 demonstrations. Keyphrases are encoded with the same encoder as the text and **concatenated** to the original document embedding.
  - Results (accuracy / NMI), raw → +keyphrases:
    - Tweet, KMeans: 59.1 / 80.6 → 62.0 / 83.8.
    - Tweet, Agglomerative: 57.5 / 80.6 → 69.8 / 86.6.
    - Bank77, KMeans: 65.2 → 65.3. Bank77, Agglomerative: 64.0 → 66.4.
    - **CLINC, KMeans: 80.9 → 79.4 (worse)**. CLINC, Agglomerative: 77.7 → 81.3.
    - Entity canonicalization, OPIEC59k avg F1: 76.7 → 80.0.
  - Ablation (CLINC acc): full 79.4; without instructions 78.4; without demonstrations 78.7. Instructor-XL embedding with the same GPT-3.5 prompt as an instruction: 70.8. That is, an instruction-tuned encoder alone did not replicate the LLM-expansion gain.
  - Qualitative: "providing instructions but omitting demonstrations leads to a larger set of keyphrases with less consistency, while providing demonstrations without any instructions leads to a more focused group of keyphrases that sometimes fail to reflect the desired aspect (e.g., topic vs. intent)". Keyphrases "may overly focus on each entity's surface form rather than their textual context".
  - LLM post-correction is risky: when the LLM reassigned uncertain points, it was right only 55.0% (OPIEC), 89.7% (Tweet) and 41.7% (Bank77) of the time.
  - — [TACL 2024 PDF](https://aclanthology.org/2024.tacl-1.18.pdf); [arXiv](https://arxiv.org/abs/2307.00524)
- **IDAS (2023):** clustering on LLM-generated abstractive labels plus utterances beat the prior unsupervised state of the art by up to +7.42%. — [arXiv 2305.19783](https://arxiv.org/abs/2305.19783v1)
- **LeBoT (arXiv 2510.06747; v2 Jan 12, 2026). Verified via HTML.**
  - It does not rewrite texts. An open LLM (Gemma-2-9b-it, also tested with Qwen and Llama) selects which representative texts are similar to each input, and the input's vector becomes the average of the selected representatives' vectors.
  - With a MiniLM backbone: Bank77 73.12 vs 61.27 (plain); CLINC150 87.58 vs 73.73.
  - "Comparable or superior" to IDAS (text enrichment), better than SPILL, and competitive with fine-tuned ClusterLLM on E5.
  - Datasets are English only (Bank77, CLINC150, MTOP, Massive, GoEmo, StackOverflow); no social media in the multilingual sense. — [arXiv HTML](https://arxiv.org/html/2510.06747)
- **LiSA (ACL 2025), on tweets:** open-7/8B per-document topic word plus description, combined with document embeddings, gave P1 0.78 / ARI 0.65 vs BERTopic on raw text 0.53 / 0.18, and beat GPT-4 TopicGPT (P1 0.75). — [ACL 2025 PDF](https://aclanthology.org/2025.acl-long.902.pdf)
- **k-LLMmeans:** summarizing centroids, not inputs, gave +2.9 accuracy on CLINC; summarization was applied only at the centroid level. — [arXiv 2502.09667](https://arxiv.org/html/2502.09667)
- **Miller & Alexander (Royal Society Open Science, 2025; arXiv 2405.07278):** clusters of short texts (Twitter bios) from LLM *embeddings* are more distinctive and human-interpretable than doc2vec or LDA. This compares embedders, not rewriting. — [RSOS](https://royalsocietypublishing.org/rsos/article/12/1/241692/92905/Human-interpretable-clustering-of-short-text-using); [arXiv](https://arxiv.org/abs/2405.07278) [details beyond the abstract UNVERIFIED]
- **Akash & Chang (2024), "LLM-driven context expansion" for short-text topic modeling,** reportedly expands short texts into longer ones with an LLM before topic modeling. — [UNVERIFIED; not fetched this session]
- **Korean / multilingual evidence:**
  - Lee, Woo, Jun (KAIST; ADHO DH2026, Zenodo, Aug 26, 2026) applied BERTopic to Korean vs US YouTube comments on meditation. They argue that "language-specific preprocessing, embeddings, and clustering are crucial for capturing cultural nuance" and that relying only on multilingual embeddings is insufficient. They also flag the limits of topic reduction. This is a conference presentation without a reported LLM-gist comparison. — [Zenodo](https://zenodo.org/records/22107263)
  - TopicGPT notes degraded non-English instruction-following. — [arXiv 2311.01449](https://arxiv.org/html/2311.01449)
  - I found no paper that compares raw vs LLM-gist clustering on Korean or Korean–English code-switched text.

### Inferences
- The evidence supports "gist + raw" (concatenated or weighted embeddings) over "gist only" as the default. It also supports running both representations and comparing them on a small labeled dev set, since CLINC+KMeans got worse with expansion.
- Gains are largest on noisy, short, heterogeneous text (Tweet +2.9 to +12.3 accuracy). That is the regime closest to YouTube comments. This is an extrapolation, though: the Tweet set is English and topic-labeled, not issue-labeled.
- The demonstrations-vs-instructions finding is direct evidence for the project's axis-mixing problem. Without a clear instruction naming the axis ("the issue, not the target or the stance"), LLM rewrites drift to another axis (topic vs intent). Without demonstrations, the outputs become inconsistent and verbose.
- LeBoT shows the LLM's value may come from injecting a similarity judgment rather than from the rewritten text. A cheap local variant: use the gist for clustering, but resolve borderline merges with LLM pairwise or anchor judgments.

### Gaps
- No 2024–2026 study isolates "embed gist only" vs "embed raw" vs "embed both" on social-media comments with issue-level gold labels.
- No study with Korean, code-switched, sarcastic or meme-heavy comments.
- I did not retrieve the full Akash & Chang 2024 or Miller & Alexander papers.

---

## Q4. Recommended gist format for the "issue" facet: length, template, language, and no-issue handling

### Takeaway
No paper prescribes a format for this exact facet, so this section is an evidence-informed design. The published working formats are:
- LLooM: 2–4 bullets of 5–8 words, JSON.
- TopicGPT and LiSA: a short label plus a one-sentence description.
- PAKPA: JSON (aspect, sentiment) with sentiment held out of the clustered field.
- KPA: short, pronoun-free, single-aspect statements.
- Claimify and Dense X: decontextualized, atomic, abstain when unsure.

Combined, these suggest a structured JSON output where only a stance-free, target-abstracted issue phrase is embedded, with an explicit abstain value.

### Cited Findings
- **Bullet length:** 5–8 words, 2–4 bullets per document (LLooM defaults). The authors note defaults "have not been robustly tested". — [LLooM arXiv HTML](https://arxiv.org/html/2404.12259)
- **Label + description:** "a concise label paired with a broad one-sentence description" (TopicGPT). LiSA embeds the descriptions, averaged per topic word, rather than the bare label. — [TopicGPT](https://arxiv.org/html/2311.01449); [LiSA](https://aclanthology.org/2025.acl-long.902.pdf)
- **Structured fields with polarity separated:** (aspect, sentiment) JSON with 18 few-shot examples. Neutral items are filtered out, and clustering runs per polarity on the aspect term. — [PAKPA](https://aclanthology.org/2024.acl-long.576.pdf)
- **Self-contained, single unit:** KPA candidate filters on length and pronouns ([ArgSum](https://arxiv.org/html/2503.00847v4)). Propositions resolve coreference and cannot be split further ([Dense X](https://arxiv.org/html/2312.06648)).
- **Abstention:** extract "only when there is high confidence in the correct interpretation" ([Claimify](https://arxiv.org/abs/2502.10855)). HDBSCAN leaves noise points unclustered ([LLooM](https://arxiv.org/html/2404.12259)).
- **Prompt construction:** state the clustering goal explicitly, as in Viswanathan's "I am trying to cluster … based on whether they express the same intent". Use both an instruction and about 4 demonstrations. — [TACL 2024](https://aclanthology.org/2024.tacl-1.18.pdf)
- **Negative examples:** PAKPA's prompt includes "invalid generation examples to avoid, along with preferred correction". — [PAKPA](https://aclanthology.org/2024.acl-long.576.pdf)

### Inferences (proposed design, not empirically validated for this data)
- **Template.** Emit JSON per comment:
  ```json
  {"has_issue": true|false,
   "no_issue_type": "meme|pure_reaction|off_topic|unclear|null",
   "target": "<who/what, canonical name>",
   "issue": "<4–10 word English noun phrase, no stance/evaluative words>",
   "issue_ko": "<optional Korean gloss>",
   "evidence": "<verbatim span from comment>"}
  ```
  - Embed **only `issue`** for the issue facet, optionally concatenated with the raw-text embedding as in Viswanathan. Keep `target` and stance for the other facets. This mirrors PAKPA's separation of fields.
  - Example: "그 사람 말 진짜 웃기네 ㅋㅋ 병역 얘기는 왜 피함?" → target: "<guest name>", issue: "avoidance of military service question". Not "guest is dishonest about military service", because "dishonest" is stance.
- **Length.** About 4–10 words, one issue per item. If a comment has two issues, emit a list of up to 2–3 issue objects (LLooM's 2–4 bullets). This avoids muddled single embeddings, as in the coffee/breakfast PAKPA error.
- **Stance-free wording.** Explicitly ban evaluative adjectives and verbs and give paired demonstrations: a pro and a con comment on the same issue should produce an identical `issue` string. This operationalizes PAKPA's clustering within polarity, here with polarity removed entirely.
- **Abstraction level.** Instruct the model to name the issue "as a debate topic someone could argue either side of", not as the comment's specific claim. Give 2–3 examples each of too-specific, right and too-general gists (PAKPA's "invalid examples" pattern). This addresses both TopicGPT-7B's over-specific explosion and PAKPA's over-generalization.
- **Language.** No direct evidence exists, so these are inferences:
  - Write `issue` in English. It canonicalizes code-switched variants and benefits from stronger instruction-following in English, which TopicGPT notes degrades outside English.
  - Keep the raw Korean text and `evidence` span for audit and for an optional raw-text embedding channel.
  - The Korean YouTube study argues for language-specific handling, which supports keeping a Korean channel rather than relying only on translation. A multilingual embedder makes English-gist vs Korean-gist mostly a canonicalization choice; test both on a dev set.
- **Decontextualization.** Pass video title, guest/host names and topic list as context so that pronouns and nicknames resolve to canonical targets (the proposition principle). Instruct "do not add information not present in the comment or context" and require the `evidence` span, which allows a cheap string-match faithfulness check (LLooM lacked one).
- **No-issue handling.** Set `has_issue=false` with a typed reason for memes, pure reactions ("ㅋㅋㅋ", "🔥", "lol") and unintelligible items. Exclude them from embedding, or route them to a separate meme/reaction facet, rather than letting them form spurious "issue" clusters. Also rely on HDBSCAN-style noise for the residue. This mirrors Claimify's abstention.

### Gaps
- No empirical study validates gist length, English vs Korean gists, or stance-stripping for clustering quality on comment data. These must be tested locally, e.g. on a 200–300-comment labeled dev set comparing raw, gist, gist+raw and issue-only.
- No source tested whether sarcasm detection by a local 8B model is reliable enough to set `has_issue` and the issue correctly.

---

## Q5. Risks: hallucination, homogenization/mode collapse, loss of nuance, taxonomy bias leaking through prompts

### Takeaway
The documented risks are:
- unverified hallucinated or inaccurate distillations (LLooM explicitly does not verify);
- run-to-run variability;
- granularity failures with small models (over-splitting) and over-generalization in summaries;
- LLM post-hoc corrections that are often wrong;
- demonstrations steering the representation toward the wrong axis;
- anchoring of human reviewers toward LLM suggestions, which then inflates apparent model accuracy.

### Cited Findings
- **Hallucination and verification:** LLooM "does not make use of verification steps … to ensure that quotes are exact matches, that bullet points are accurate to quotes". — [LLooM](https://arxiv.org/html/2404.12259)
- **KPA hallucination:** abstractive KPA from many comments (SKPM) is vulnerable to hallucination. PAKPA's aspect-grounded approach scored highest on faithfulness. — [PAKPA](https://aclanthology.org/2024.acl-long.576.pdf)
- **Variability:** LLooM shows "variability in the results upon re-run", which is undesirable "where analyses must be replicable". — [LLooM](https://arxiv.org/html/2404.12259). LLM topic merging is unstable because LLMs are "very sensitive to the positioning of prompts". — [LiSA, ACL 2025](https://aclanthology.org/2025.acl-long.902.pdf)
- **Granularity failure with small models:** Mistral-7B topic generation produced 1,418 topics vs 151. — [TopicGPT](https://arxiv.org/html/2311.01449)
- **Over-generalization and merged aspects lose minority opinions:** — [PAKPA](https://aclanthology.org/2024.acl-long.576.pdf)
- **Wrong axis and surface bias:** demonstrations without instructions yield keyphrases that "fail to reflect the desired aspect (e.g., topic vs. intent)", and keyphrases can over-focus on surface form. — [Viswanathan et al. TACL 2024](https://aclanthology.org/2024.tacl-1.18.pdf)
- **LLM corrections:** when the LLM overrode uncertain cluster assignments, it was right only 41.7–55.0% of the time on Bank77 and OPIEC. — [Viswanathan et al.](https://aclanthology.org/2024.tacl-1.18.pdf)
- **Human anchoring and homogenization ("Just Put a Human in the Loop?", arXiv 2507.15821, July 2025):** a pre-registered study with 410 annotators and more than 7,000 annotations. Annotators "strongly took the LLM suggestions, significantly changing the label distribution". Evaluating models on those labels makes "reported model performance significantly increase". — [arXiv](https://arxiv.org/abs/2507.15821)
- **Bias in LLM qualitative coding:** LLM coding of interview transcripts can introduce bias that "is not random with respect to the characteristics of the interview subjects". — [Ashwin et al., arXiv 2309.17147](https://arxiv.org/pdf/2309.17147) (from search snippet; details UNVERIFIED)
- **Semantic homogenization:** a 2026 preprint on LLMs in qualitative social network analysis warns that LLMs "may inadvertently collapse distinct relational understandings into uniform categories". — [arXiv 2607.14045](https://arxiv.org/pdf/2607.14045) (search snippet only; UNVERIFIED)
- **Cost and dependency:** TopicGPT's best results need GPT-4 for generation. LLooM uses GPT-4 for synthesis. IntentGPT uses GPT-4. GoalEx (from memory) uses GPT-4-class proposers. Claimify and Dense X used GPT-4-class models to create or extract training data. **All of these are flagged as GPT-4-dependent** for their reported numbers. The open-model evidence is LiSA (Llama-3-8B/Mistral-7B), PAKPA (Mistral-7B end to end, Soft-F1 0.71), LeBoT (Gemma-2-9B) and ArgSum (Qwen3-32B top). — sources as cited above.

### Inferences
- **Taxonomy leakage:** any category names, example clusters or "issues such as X, Y" in the gist prompt will pull gists toward those labels. The Viswanathan finding that demonstrations steer which aspect is captured supports this. Demonstrations should therefore be drawn from a different video or domain and vary in surface form, and the prompt should not list candidate issues during open coding.
- **Homogenization:** gists can collapse different issues into generic strings ("controversy about guest's remarks"). Mitigations:
  - Track gist type/token ratio and the number of distinct gists per 100 comments.
  - Compare cluster structure from gist embeddings against raw embeddings.
  - Embed gist+raw.
  - Have human review use comment text, not only gists, to avoid anchoring.
- **Faithfulness:** require a verbatim evidence span and reject gists whose span is not a substring of the comment. Spot-check about 5% of gists. Fix temperature and seed in llama.cpp and cache outputs for reproducibility, given the documented re-run variability.
- **Codebook review:** to counter anchoring, reviewers should first label a random sample blind, or see LLM-proposed definitions only after an initial independent pass.

### Gaps
- No quantitative study found on gist homogenization (distinct-gist rate) in clustering pipelines specifically.
- The "argument collapse" statistic (65.3% of human main arguments unique vs 3.4% of LLM ones) appeared only on an aggregator ([feedbagel](https://feedbagel.com/post/argument-collapse-llms-flatten-long-form-public-debate)), and I could not trace the primary paper, so it is UNVERIFIED and should not be cited.
- I did not verify hallucination rates for Qwen3-class local models on Korean slang or sarcasm.
