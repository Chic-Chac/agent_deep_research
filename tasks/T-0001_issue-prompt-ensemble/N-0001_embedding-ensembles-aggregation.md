---
id: N-0001
type: note
task: T-0001
title: "Prompt/Template Ensembles for Text Embeddings and Aggregation of Generated-Text Ensembles for Clustering"
status: final
created: 2026-11-06
tags: [prompt-ensemble, embeddings, aggregation]
---

# Prompt/Template Ensembles for Text Embeddings and Aggregation of Generated-Text Ensembles for Clustering

Scope note: These notes cover (a) prompt/instruction ensembles for embeddings and multi-view representations, and (b) aggregation rules for generated-text ensembles that feed clustering, plus known failure modes. Setting to transfer to: a local LLM writes 4-12-word stance-neutral "issue questions"; Qwen3-Embedding-4B embeds them; about 40 source anchors are grouped; about 130 comment issues attach to the nearest anchor group by cosine, with a margin over the runner-up. Averaging four paraphrase-prompt embeddings raised split-half ARI only from 0.71 to 0.75.

Source-status legend: **[fetched]** means I read or searched the primary page in this session. **[known, not re-fetched]** means the citation comes from prior knowledge. The link is the canonical arXiv/ACL page, but I did not re-verify the exact numbers this session, so the report writer should treat any detail beyond the headline idea as needing checking. I only list numbers for [fetched] items.

---

## Q1. Prompt-based sentence embeddings with multiple templates/instructions: what gains, and how were they combined?

### Takeaway
The clearest evidence (MetaEOL, ACL 2024) is that averaging embeddings from **diverse task-framed prompts** gives large gains: +2.8 to +6.7 STS points over a single prompt. Mean beat concatenation, and concatenation beat max-pooling. But **eight paraphrases of the same prompt averaged together were much worse than the single prompt** (62.72 vs 70.03). Diversity of perspective is what helps, not the count of prompts. This matches the team's small gain from four paraphrase prompts.

### Cited Findings
- **MetaEOL** (Lei et al., "Meta-Task Prompting Elicits Embeddings from Large Language Models", ACL 2024 long, pp. 10141-10157). It uses training-free LLM embeddings from 4 meta-tasks (Text Classification, Sentiment Analysis, Paraphrase Identification, Information Extraction), 2 prompts each, 8 prompts in total. The sentence embedding is the **simple average** of the per-prompt last-layer embeddings. [fetched] — [ACL Anthology](https://aclanthology.org/2024.acl-long.546); [arXiv HTML v2](https://arxiv.org/html/2402.18458v2)
- MetaEOL vs PromptEOL, STS average (Spearman x100): LLaMA2-7B 75.96 vs 70.03; Mistral-7B 76.09 vs 73.32; LLaMA3-8B 77.35 vs 71.90; LLaMA2-13B 75.56 vs 68.83. [fetched] — [arXiv HTML](https://arxiv.org/html/2402.18458v2)
- MetaEOL aggregation ablation (LLaMA2-7B, STS average): **mean 75.96; concatenation 74.99; max-pooling 72.03**. [fetched] — [arXiv HTML](https://arxiv.org/html/2402.18458v2)
- MetaEOL single meta-task vs ensemble: TC only 70.92, SA only 67.06, PI only 73.03, IE only 72.06; all four 75.96. The ensemble beats its best member by about 2.9 points even though one member (SA) is below baseline. [fetched] — [arXiv HTML](https://arxiv.org/html/2402.18458v2)
- MetaEOL incremental meta-tasks: TC 70.92 → +SA 71.48 → +PI 75.06 → +IE 75.96. Gains are uneven and depend on which perspective is added, not on how many prompts there are. [fetched] — [arXiv HTML](https://arxiv.org/html/2402.18458v2)
- **Key negative control:** 7 paraphrases of the PromptEOL prompt plus the original (8 prompts) averaged scored **62.72, below single PromptEOL (70.03)**. The authors attribute MetaEOL's gains to the complementarity of the meta-tasks. They did not run an equal-size diverse-vs-non-diverse comparison, so diversity as the cause is inferred, not directly tested. [fetched] — [arXiv HTML](https://arxiv.org/html/2402.18458v2)
- **PromptEOL** (Jiang et al., "Scaling Sentence Embeddings with Large Language Models", arXiv 2023; later Findings of EMNLP 2024). It uses a single template of the form `This sentence: "[X]" means in one word:"` and takes the last-token hidden state, plus in-context examples. It is single-prompt, and MetaEOL uses it as the baseline. [known, not re-fetched] — [arXiv 2307.16645](https://arxiv.org/abs/2307.16645)
- **PromptBERT** (Jiang et al., EMNLP 2022). It introduced prompt-based sentence embeddings and "template denoising": subtract the embedding of the template filled with an empty sentence, to remove the bias the template adds. This shows that a template contributes its own direction to the embedding. [known, not re-fetched] — [arXiv 2201.04337](https://arxiv.org/abs/2201.04337)
- **Echo embeddings** (Springer et al., "Repetition Improves Language Model Embeddings", 2024). The input is repeated and pooling is done over the second copy, so causal-LM tokens can attend to the whole sentence. It is a single-prompt technique, not an ensemble. [known, not re-fetched] — [arXiv 2402.15449](https://arxiv.org/abs/2402.15449)
- **INSTRUCTOR** (Su et al., Findings of ACL 2023). It embeds text together with a task instruction and reports a 3.4% average improvement over the previous best on 70 datasets. The authors report it is **robust to changes in instructions**, meaning paraphrased instructions give similar embeddings. [fetched via search snippet] — [arXiv 2212.09741](https://arxiv.org/abs/2212.09741); [ACL Findings 2023.findings-acl.71](https://preview.aclanthology.org/setup/2023.findings-acl.71)
- **Qwen3-Embedding** (tech report, arXiv 2025). The instruction is prepended **only to the query side** as `{Instruction} {Query}<|endoftext|>`, and documents are left unchanged. The embedding is the last-layer hidden state at the [EOS] token. Cosine similarity is used in training. MRL is supported, and the default dimension of the 4B model is 2560. MTEB clustering for 4B: 57.15 on MTEB Multilingual and 57.51 on MTEB eng v2. The report does not quantify the drop without an instruction. A "1-5% drop without instruction" figure circulates but was **not confirmed** in the report text. [fetched] — [arXiv 2506.05176 HTML v3](https://arxiv.org/html/2506.05176v3)
- **E5-mistral-7b-instruct** (Wang et al., "Improving Text Embeddings with Large Language Models", ACL 2024) uses a similar one-sentence task instruction on the query side only. [known, not re-fetched] — [arXiv 2401.00368](https://arxiv.org/abs/2401.00368)
- I found **no paper that directly tests averaging embeddings across several different instructions** for an instruction-tuned embedder (INSTRUCTOR, E5-instruct, Qwen3-Embedding). The nearest related work, MoTE (2025) and InstEmb (PMLR 2026), studies instruction conditioning, not instruction ensembles. [fetched via search] — [MoTE arXiv 2506.17781](https://arxiv.org/pdf/2506.17781); [InstEmb PMLR v306](https://proceedings.mlr.press/v306/gao26h.html)

### Inferences
- **Transfer:** The team's four prompts are paraphrases. MetaEOL's negative control predicts that paraphrase ensembles give mostly correlated views and little gain, which matches 0.71 → 0.75. If you ensemble, vary the *perspective*: for example, "what policy is at stake", "who is affected", "what decision is being asked", "what value conflict". Don't vary the wording.
- **Transfer:** MetaEOL averages *hidden-state embeddings of one input under different prompts*. The team averages *embeddings of different generated strings*. The second adds a generation-noise channel, since each 4-12-word string may drift. So MetaEOL's gains are suggestive, not directly transferable.
- **Transfer:** Two ensemble axes are open and cheap. (1) The *rewrite* prompt, which is LLM-side. (2) The *embedding instruction*, which is Qwen3 query-side, e.g. "Represent the policy issue for clustering by topic" vs "...by decision asked". Because Qwen3 is instruction-aware, instruction-side ensembling is plausible. It has no direct evidence and should be tested on split-half ARI.
- **Transfer:** In the asymmetric setup, Qwen3 puts instructions on queries only. For symmetric clustering (anchors and comments in the same space), use the **same instruction on both sides**. Otherwise the instruction itself adds a constant offset between anchors and comments. PromptBERT's template-bias finding is the analogue.
- Mean beat concat in MetaEOL, and for cosine on unit-normalized members, concatenation is equivalent to averaging the per-view cosines. So mean (or normalized mean) is a sensible default. Max-pooling across views was clearly worst.

### Gaps
- PromptSTH/PromptSUM and "Contrastive Prompting" (inference-time steering for sentence embeddings): I did not verify the papers, venues or numbers this session. Report them as unverified.
- MetaEOL's per-prompt-count curve (Figure 3) has no numeric values in the text. There is no explicit diminishing-returns analysis.
- MetaEOL's MTEB *clustering* subscores were not extracted, so the STS gains may not carry over to clustering.
- No study found on instruction ensembles for Qwen3-Embedding or other instruction-tuned embedders.

---

## Q2. Query expansion / multi-view / generative augmentation: which average embeddings and which keep multiple vectors?

### Takeaway
Methods split into three families. **Average into one vector:** HyDE, MetaEOL. **Concatenate the original with a generated-text vector:** Viswanathan et al. keyphrase expansion; query2doc concatenates at the text level. **Keep several vectors or rankings and fuse at score level:** RAG-Fusion/RRF, MVR, ColBERT-style late interaction. For clustering, the published LLM-augmentation methods report modest gains on intent and topic clustering: about +1 to +4.5 accuracy points for keyphrase expansion, and up to +7.42% for IDAS summaries.

### Cited Findings
- **HyDE** (Gao et al., "Precise Zero-Shot Dense Retrieval without Relevance Labels", arXiv Dec 2022; ACL 2023). Query vector = **mean of N encoded hypothetical documents**, v = (1/N) Σ f(d_k). A variant counts the query as one more hypothesis: v = (1/(N+1)) [Σ f(d_k) + f(q)]. Samples come from InstructGPT at temperature 0.7. N is not stated in the main text. The authors argue the dense encoder acts as a **lossy "bottleneck" that filters hallucinated details**, which is their stated reason averaging works. [fetched] — [ar5iv 2212.10496](https://ar5iv.labs.arxiv.org/html/2212.10496); [arXiv abs](https://arxiv.org/abs/2212.10496)
- Framework implementations of HyDE (Haystack, LangChain) default to about 5 generated documents, embedded and **mean-averaged**. These are secondary sources. — [Haystack docs](https://docs.haystack.deepset.ai/docs/hypothetical-document-embeddings-hyde); [LangChain docs](https://langchain-doc.readthedocs.io/en/latest/modules/indexes/examples/hyde.html)
- **Viswanathan et al., "Large Language Models Enable Few-Shot Clustering"** (TACL 2024). For keyphrase expansion, GPT-3.5 generates keyphrases per document. They are encoded into one vector with the same encoder, and that vector is **concatenated** to the original embedding before k-means. Normalization and weighting are not described, and averaging is not compared. [fetched] — [arXiv HTML](https://arxiv.org/html/2307.00524v1); [TACL PDF](https://aclanthology.org/2024.tacl-1.18.pdf)
- Keyphrase expansion results (accuracy/NMI) vs k-means on the base embedding: Bank77 64.0/81.7 → 65.3/82.4; CLINC 77.7/91.5 → 79.4/92.6; Tweet 57.5/80.6 → 62.0/83.8. Entity canonicalization: OPIEC59k average F1 76.7 → 80.0; ReVerb45k 82.7 → 84.2. [fetched] — [arXiv HTML](https://arxiv.org/html/2307.00524v1)
- In the same paper, instructions alone gave larger, less consistent keyphrase sets. Demonstrations alone gave focused sets that "sometimes miss the intended aspect (topic vs intent)". Prompt framing decides *which facet* the expansion captures. [fetched] — [arXiv HTML](https://arxiv.org/html/2307.00524v1)
- **IDAS** (De Raedt et al., 2023). An LLM with in-context prototypes writes short abstractive "labels" (summaries) per utterance. Utterances and labels are encoded by a frozen encoder and clustered. IDAS reports up to +7.42% on standard cluster metrics (Banking, StackOverflow, Transport) over the prior unsupervised state of the art. The exact way utterance and label vectors are combined was not verified this session. [fetched via search] — [arXiv 2305.19783](https://arxiv.org/abs/2305.19783v1)
- **query2doc** (Wang, Yang, Wei, EMNLP 2023). An LLM pseudo-document is **concatenated as text** with the query: the query is repeated for BM25, or joined with [SEP] for dense retrievers. One string, one vector. [known, not re-fetched] — [arXiv 2303.07678](https://arxiv.org/abs/2303.07678)
- **RAG-Fusion** (Rackauckas, 2024). It generates multiple query rewrites, retrieves for each, and fuses the rankings with **Reciprocal Rank Fusion**, keeping the views separate until rank fusion. [known, not re-fetched] — [arXiv 2402.03367](https://arxiv.org/abs/2402.03367)
- **RRF** (Cormack, Clarke, Büttcher, SIGIR 2009): score(d) = Σ_views 1/(k + rank_view(d)), with k = 60. It is rank-based and does not depend on score scale. [known, not re-fetched] — [ACM DL](https://dl.acm.org/doi/10.1145/1571941.1572114)
- **MVR** (Zhang et al., "Multi-View Document Representation Learning for Open-Domain Dense Retrieval", ACL 2022). Several "viewer" tokens produce several document vectors, and the score is the max over views. A global-local loss with annealed temperature keeps the views from collapsing into the same vector. [known, not re-fetched] — [arXiv 2203.08372](https://arxiv.org/abs/2203.08372)
- **Doc2Query--** (Gospodinov, MacAvaney, Macdonald, ECIR 2023). It **filters out hallucinated or irrelevant generated expansions** with a relevance scorer before indexing. The authors report that dropping bad expansions improves effectiveness and shrinks the index, i.e. more generated views is not always better. [known, not re-fetched] — [arXiv 2301.03266](https://arxiv.org/abs/2301.03266)
- **ClusterLLM** (Zhang, Wang, Shang, EMNLP 2023) does not average generated text. It asks the LLM triplet questions ("is A closer to B or C?") to **fine-tune the embedder**, and pairwise questions to choose granularity. [known, not re-fetched] — [arXiv 2305.14871](https://arxiv.org/abs/2305.14871)
- **Goal-Ex** (Wang, Shang, Zhong, EMNLP 2023) uses goal-driven explainable clustering: the LLM proposes natural-language cluster descriptions, assigns texts to them, and selects a subset. It works through language-level cluster definitions, not embedding averaging. [known, not re-fetched] — [arXiv 2305.13749](https://arxiv.org/abs/2305.13749)

### Inferences
- **Transfer:** The team's pipeline is closest to IDAS: an LLM rewrite to a short label, then embedding and clustering. IDAS embeds the label *together with* the original utterance. The pipeline could test three variants: (i) issue-question only (current), (ii) issue-question + source-text embedding, concatenated or weighted mean, as in Viswanathan, (iii) multi-perspective issue-questions only.
- **Transfer:** HyDE's "dense bottleneck" argument suggests that averaging several generated strings cancels idiosyncratic details. For 4-12-word strings, there is little detail to cancel, and the risk is the reverse: averaging blurs the *specific issue* that separates neighbouring anchors.
- **Transfer:** Viswanathan's note that framing decides the facet (topic vs intent) is a warning for perspective ensembles. Each perspective prompt may cluster by a *different* facet. Averaging facets yields a compromise geometry, which may or may not match the gold grouping. Validate each perspective alone first.
- **Transfer:** For nearest-anchor assignment, RRF-style fusion is a useful alternative to vector averaging. Rank anchor groups per view, fuse the ranks, and apply the margin on fused scores. It is robust to differences in cosine scale between views.

### Gaps
- IDAS's exact combination rule (mean of utterance and label embeddings vs label only) was not verified.
- No paper found that directly compares mean, concat, and multi-vector for *LLM-generated paraphrase sets feeding clustering*. The comparisons are in retrieval (HyDE, MVR) or STS (MetaEOL).
- RAG-Fusion's evidence is mostly qualitative or industrial. Treat it as thin.

---

## Q3. Aggregation options: mean, normalized mean, medoid, multi-vector max-sim/Chamfer, LLM consolidation, label models, MBR

### Takeaway
Treat "pick the medoid" (MBR with an embedding utility) and "mean embedding" (centroid) as the two main single-vector options. Treat "keep all vectors + set similarity (mean-max/Chamfer)" as the multi-vector option. MBR/medoid gives a *real, readable string* and resists outliers. Centroid-based MBR is an efficient middle ground. For the discrete `has_issue` field, use majority vote or a Dawid-Skene/Snorkel-style label model. The MBR literature shows multi-prompt candidate pools beat single-prompt pools.

### Cited Findings
- **MBR = medoid when the utility is a similarity**: MBR picks the candidate with the highest expected similarity to the other candidates, which is the medoid under that similarity. [fetched via search snippet] — [Regularized Best-of-N with MBR, arXiv 2404.01054](https://arxiv.org/pdf/2404.01054)
- **Multi-prompt MBR** (Heineman, Dou, Xu, EMNLP 2024). Candidates are decoded from a **prompt bank** and the final output is chosen by MBR. This improves over single-prompt MBR across tasks (simplification, translation, code), models and metrics. The authors attribute the gain to a **more diverse and higher-quality candidate space** than one prompt gives. [fetched via search] — [ACL Anthology 2024.emnlp-main.1255](https://preview.aclanthology.org/setup/2024.emnlp-main.1255); [arXiv 2407.15343](https://arxiv.org/abs/2407.15343)
- **Centroid-based MBR** (Deguchi et al., Findings of ACL 2024) clusters candidates in embedding space and scores against centroids, which is cheaper than all-pairs MBR. [known, not re-fetched; surfaced in search] — [arXiv 2402.11197](https://arxiv.org/abs/2402.11197)
- MBR with an LLM-judge utility beat MBR with lexical or embedding metrics for open-ended instruction following (AlpacaEval, MT-Bench). [fetched via search] — [Wu et al., arXiv 2410.02902](https://arxiv.org/html/2410.02902v1)
- A 2026 preprint embeds responses in a task-dependent latent space, selects the centroid, and uses the Bayes risk (dispersion) as an **uncertainty estimate**. It is a preprint and unreviewed. [fetched via search] — [arXiv 2601.21500](https://arxiv.org/html/2601.21500v1)
- **Chamfer / MaxSim** set-to-set similarity: Chamfer(Q,P) = Σ_{q∈Q} max_{p∈P} ⟨q,p⟩. This is ColBERT's late-interaction score, also called MaxSim or relaxed earth mover distance. MUVERA (NeurIPS 2024) approximates it with fixed-dimensional encodings, so single-vector MIPS can be used. [fetched] — [MUVERA arXiv 2405.19504](https://arxiv.org/pdf/2405.19504)
- **Self-consistency** (Wang et al., ICLR 2023): majority vote over sampled reasoning paths, for discrete answers. **Universal Self-Consistency** (Chen et al., 2023) asks the LLM to pick the most consistent of several free-form candidates. It is the LLM-consolidation analogue of MBR for free text. [known, not re-fetched] — [arXiv 2203.11171](https://arxiv.org/abs/2203.11171); [arXiv 2311.17311](https://arxiv.org/abs/2311.17311)
- **Label models for noisy discrete votes:** Dawid & Skene (1979) EM over annotator confusion matrices; Snorkel (Ratner et al., VLDB 2018) learns labeling-function accuracies and correlations without ground truth. These suit a binary `has_issue` from several prompts, especially when prompts differ in reliability or are correlated. [known, not re-fetched] — [Snorkel arXiv 1711.10160](https://arxiv.org/abs/1711.10160); [Dawid & Skene, JRSS-C 1979](https://doi.org/10.2307/2346806)

### Inferences (mathematical facts and transfer)
- **Normalized mean and cosine (derivation, not from a paper):** for unit vectors v_1..v_n with mean m, ||m||² = 1/n + ((n−1)/n)·c̄, where c̄ is the average pairwise cosine. With n = 4 and c̄ = 0.8, ||m|| ≈ 0.92. With c̄ = 0.5, ||m|| ≈ 0.79. Then cos(m, a) = mean_i cos(v_i, a) / ||m||. So (i) **unnormalized** mean cosines shrink toward 0 when members disagree, and (ii) **renormalized** mean cosines are *inflated* by 1/||m|| and can exceed every member's cosine to the anchor. In both cases the score distribution moves, so **a margin threshold tuned on single-string embeddings is mis-calibrated for averaged embeddings**. Re-tune the margin, or compute the margin on mean-of-cosines (which equals the unnormalized dot product) instead.
- **Transfer, medoid/MBR over the 4-8 strings:** Pick the string with the highest mean cosine to the other strings of the same item, then embed only that string. Advantages: (a) a real issue question you can display; (b) resistance to one off-topic rewrite; (c) cosine scale identical to single-string embeddings, so the margin rule keeps its calibration. The dispersion c̄ comes for free as a per-item confidence: low c̄ means the item is ambiguous, so route it to "unassigned" or human review.
- **Transfer, multi-vector assignment:** Keep each item's K strings and each anchor group's member strings as sets. Score item→group by mean-max (Chamfer normalized by |Q|) or by the mean of the per-view top-1 cosines. This keeps facet-specific matches (e.g. a comment matching an anchor only under the "decision asked" view) that averaging would wash out. Cost is negligible at about 170 items.
- **Transfer, LLM consolidation:** Have the LLM merge K candidates into one issue question (USC-style). This can restore specificity, but it adds a generation step that can drift or hallucinate, and the output's embedding scale matches single strings. Use MBR as a non-generative baseline before trying it.
- **Transfer, `has_issue`:** With 4 prompts, majority vote with a tie rule is fine. If the prompts are correlated (paraphrases), a label model adds little, because Snorkel-style models assume or estimate dependency structure, and with only 4 sources that estimate is weak.
- Multi-prompt MBR's result (more diverse prompt bank → better pool) matches MetaEOL's finding: **diverse prompts + selection** is the strongest-evidenced generated-text recipe.

### Gaps
- No direct published comparison of mean vs medoid vs Chamfer for clustering short LLM-generated labels. The recommendations above are inferred from retrieval, STS and MBR literature.
- The claim that MBR "consensus collapse" can be avoided by structure-aware clustering (attributed to Eikema et al., 2025) appeared only in a secondary summary and was not verified.

---

## Q4. Known failure modes: generic centroid/hubness, anisotropy, loss of specificity, outliers, norm shrinkage, margin effects

### Takeaway
Averaging pulls items toward the dense center of the space. In anisotropic, hub-prone embedding spaces, that center holds "generic" vectors that are near-neighbours of many items. That is exactly the case where a nearest-anchor-plus-margin rule fails: margins compress, generic anchors become attractors, and specific distinctions blur. Outliers enter the mean with weight 1/n, so with n = 4 one bad rewrite moves the vector a lot. The medoid does not have this problem.

### Cited Findings
- **Hubness in sentence-embedding spaces:** Sentence-BERT spaces show hubness, where a few texts are nearest neighbours of many others. Hubness reduction cut hubness by about 75% and error rate by about 9% on one pretrained model (Nielsen & Hansen, NLDL 2023). [fetched via search] — [Hubness Reduction Improves Sentence-BERT Semantic Spaces, arXiv 2311.18364](https://arxiv.org/pdf/2311.18364.pdf)
- **Hubness as a general high-dimensional effect** (Radovanović, Nanopoulos, Ivanović, JMLR 2010): points close to the data mean become hubs. [known, not re-fetched] — [JMLR 11](https://www.jmlr.org/papers/v11/radovanovic10a.html)
- **Anisotropy** of contextual embeddings (Ethayarajh, EMNLP 2019): representations occupy a narrow cone, so random pairs have high cosine. [known, not re-fetched] — [arXiv 1909.00512](https://arxiv.org/abs/1909.00512)
- A 2026 preprint on vector-database attacks defines a hub as the normalized centroid of normalized vectors. It notes that the global centroid **may lie in no semantic region** when embeddings are multi-modal. [fetched via search; preprint] — [arXiv 2604.05480](https://arxiv.org/pdf/2604.05480)
- A 2026 multilingual preprint argues that hubness and anisotropy are statistically dissociable, that hubness better predicts retrieval asymmetry, and recommends **CSLS** over raw cosine. It is a single study and should be treated as a claim to verify. [fetched via search summary; URL not captured] — see the scikit-hubness tooling noted in the search results
- MVR needed an explicit global-local loss to stop multiple views from **collapsing** into the same vector, so multi-view representations tend to converge. [known, not re-fetched] — [arXiv 2203.08372](https://arxiv.org/abs/2203.08372)
- Doc2Query--: unfiltered generated expansions include hallucinated content that hurts retrieval, and filtering helps. [known, not re-fetched] — [arXiv 2301.03266](https://arxiv.org/abs/2301.03266)
- MetaEOL: a weak single view (SA, 67.06 alone) still did not hurt the 4-way mean, but 8 near-duplicate views hurt badly (62.72). The failure mode is *redundant views*, not one weak view. [fetched] — [arXiv HTML](https://arxiv.org/html/2402.18458v2)

### Inferences (transfer)
- **Generic-centroid risk is higher for short strings.** Each 4-12-word issue question is already a compressed, generic phrasing ("Should the city change X?"). Averaging four such phrasings moves the vector toward a shared template direction, such as the question form or policy vocabulary. PromptBERT's template bias is the analogue. **Mitigation:** subtract the corpus mean (mean-centering) or the embedding of the bare template/instruction before cosine. Alternatively, use a hubness-reduced similarity: CSLS, or mutual-proximity via scikit-hubness.
- **Margin rule under averaging:** averaging reduces within-item variance (good) but also pulls an item's similarities to *all* anchors toward each other (bad), so top-1 minus top-2 margins shrink. Check this directly. Compare the distribution of top-1/top-2 margins for single-string vs averaged embeddings, and the share of comments assigned at a fixed margin.
- **Hub anchors:** after anchor groups are merged, a generic anchor-group centroid may attract many comments. Check the in-degree (k-occurrence) of anchor groups. If one group gets a disproportionate share of comment assignments, apply CSLS or per-anchor score normalization (z-score each anchor's similarities over all comments).
- **Outliers:** with n = 4, one off-target rewrite contributes 25% of the mean. Use the medoid, a trimmed mean (drop the member with the lowest mean similarity to the rest), or a Doc2Query-- style filter (drop rewrites whose cosine to the source text falls below a threshold).
- **Norm shrinkage:** see the formula in Q3. Always L2-normalize after averaging, and re-tune the margin when you do.

### Gaps
- No study found that measures the effect of view-averaging specifically on nearest-centroid-plus-margin assignment accuracy. This should be tested empirically.
- I found no Qwen3-Embedding-specific measurement of anisotropy or hubness.

---

## Q5. How many prompts/views are enough? Diminishing returns

### Takeaway
The evidence is thin. MetaEOL used 8 prompts across 4 meta-tasks. The gains came mostly from adding a *new perspective*: +3.58 when Paraphrase Identification was added, versus +0.56 and +0.90 for other additions. Adding paraphrase prompts was harmful. HyDE-style practice uses about 5 samples, with no published ablation found. Multi-prompt MBR reports that larger, more diverse prompt banks help, but I did not extract its scaling curve.

### Cited Findings
- MetaEOL incremental gains by meta-task: +0.56 (SA), +3.58 (PI), +0.90 (IE). Within-task prompt-count gains appear only in a figure. [fetched] — [arXiv HTML](https://arxiv.org/html/2402.18458v2)
- MetaEOL: 8 paraphrase prompts scored 62.72 vs 70.03 for 1 prompt, so more redundant views gave a worse result. [fetched] — [arXiv HTML](https://arxiv.org/html/2402.18458v2)
- HyDE: the number of samples N is not specified in the paper text. Frameworks default to about 5 (Haystack). [fetched] — [ar5iv](https://ar5iv.labs.arxiv.org/html/2212.10496); [Haystack](https://docs.haystack.deepset.ai/docs/hypothetical-document-embeddings-hyde)
- Multi-prompt MBR: gains are attributed to candidate diversity and quality from the prompt bank. [fetched via search] — [arXiv 2407.15343](https://arxiv.org/abs/2407.15343)
- Self-consistency (majority vote) accuracy rises with the number of samples and then levels off. This is reported for reasoning tasks, not embeddings. [known, not re-fetched] — [arXiv 2203.11171](https://arxiv.org/abs/2203.11171)

### Inferences (transfer)
- For about 170 short items, 3-5 **perspective-distinct** prompts is a reasonable range to test, with leave-one-prompt-out ablation on split-half ARI. A useful diagnostic is the average pairwise cosine c̄ between views of the same item. If c̄ > ~0.9, the views are effectively redundant (paraphrase regime) and adding more will not help. Lower c̄ with stable or rising ARI means real complementary information.
- Expect diminishing returns to be governed by **view correlation**, not by count. The 0.71 → 0.75 result with four paraphrase prompts is consistent with highly correlated views.

### Gaps
- No embedding-specific study found that plots quality against the number of *diverse* prompts beyond MetaEOL's figure. There is no evidence specific to clustering with nearest-anchor assignment.
- Late-2026 work: searches found no newer instruction-ensemble or prompt-ensemble embedding papers beyond those listed (MoTE 2025, InstEmb 2026, and the centroid-MBR latent-space preprint 2601.21500). Coverage after mid-2026 is thin.
