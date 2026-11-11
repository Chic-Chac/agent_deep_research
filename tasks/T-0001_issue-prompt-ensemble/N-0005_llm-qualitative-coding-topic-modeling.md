---
id: N-0005
type: note
task: T-0001
title: "LLM-assisted qualitative coding, codebook induction, and LLM topic modeling: label generation, consolidation, and prompt-sensitivity controls"
status: final
created: 2026-11-06
tags: [qualitative-coding, topic-modeling]
---

# LLM-assisted qualitative coding, codebook induction, and LLM topic modeling: label generation, consolidation, and prompt-sensitivity controls

Scope note: Items marked **[verified]** were checked against the primary paper (full text or HTML) in this session. Items marked **[abstract-level]** come from abstracts or search snippets only. Items marked **[unverified-detail]** cite a primary URL whose specific mechanics were not re-read this session; the report writer should treat their mechanics as lower confidence. All examples are synthetic.

Target setting (for the "transfers" notes): a local LLM at temperature 0 with JSON output rewrites video-side note blocks into one stance-neutral "issue question" (4-12 words) and codes each comment into 1-2 issues. Issues are embedded, merged or clustered into a codebook, and comment issues attach to the nearest video-side anchor group. Paraphrased prompts give ARI 0.58-0.71 between prompt variants.

## Q1. LLM topic modeling and concept-induction systems: what the LLM generates, whether it is embedded, how near-duplicates merge, and whether multiple prompts, seeds, or stability are reported

### Takeaway
Almost every applied system follows the same shape: (1) the LLM generates a short label or summary per document or per batch, (2) embeddings plus clustering, or an LLM merge prompt, consolidate near-duplicates, and (3) a separate assignment or scoring pass maps documents to a fixed label set. None of the major systems ensembles *paraphrased prompts* to stabilize labels. Robustness comes instead from (a) **decoupling generation from assignment** (labels are frozen before assignment), (b) **embedding-based merging with an LLM judge**, (c) **multiple runs or trials with selection** (TnT-LLM, LLooM's 10 trials), and (d) in TopicGPT, an explicit **robustness ablation** over generation sample, prompt domain, example topics, and document order. TopicGPT reports ARI of about 0.39-0.42 across those perturbations against ground truth, and 0.92 run-to-run ARI with an identical setup.

### Cited Findings

**TopicGPT (Pham, Hoyle, Sun, Resnik, Iyyer; NAACL 2024) [verified]**
- Generation: documents are sampled uniformly at random (1,000 for Bills, 1,100 for Wiki). For each sampled document, GPT-4 either assigns it to an existing topic or creates a new one. A topic is a short label plus a one-sentence description. The prompt is seeded with 2 human-written example topics, which need not be dataset-specific. The authors suggest sampling until no new topics appear for about 200 documents; refined topic counts plateaued around 600 documents. — [arXiv HTML 2311.01449v2](https://arxiv.org/html/2311.01449v2)
- Refinement (optional): Sentence-Transformer embeddings find topic pairs with **cosine similarity ≥ 0.5**. The LLM sees **5 candidate pairs at a time** and merges near-duplicates. Topics generated less often than a threshold are dropped (10 for Bills, 5 for Wiki), and the authors recommend tuning this threshold. — [arXiv HTML 2311.01449v2](https://arxiv.org/html/2311.01449v2)
- Assignment: the LLM (GPT-3.5-turbo by default) receives the topic list, 2-3 examples, and the document. It outputs a label, a document-specific description, and a **supporting quote**. A self-correction loop re-prompts on invalid outputs (up to 10 retries). Decoding uses temperature 0 and top_p 0, and the authors note the API was still not fully deterministic. — [arXiv HTML 2311.01449v2](https://arxiv.org/html/2311.01449v2)
- Results (P1 / ARI / NMI): Wiki default 0.73 / 0.58 / 0.71 and refined 0.74 / 0.60 / 0.70. Bills default 0.57 / 0.42 / 0.52 and refined 0.57 / 0.40 / 0.49. BERTopic scored 0.54 / 0.24 / 0.50 on Wiki. In the human evaluation, the share of misaligned topics on Bills fell from 70.1% unrefined to 31.9% refined. — [arXiv HTML 2311.01449v2](https://arxiv.org/html/2311.01449v2)
- **Stability ablation (Bills, P1 / ARI / NMI vs ground truth):**
  - default 0.57 / 0.42 / 0.52
  - different generation sample 0.57 / 0.40 / 0.51
  - out-of-domain prompt (Wiki prompts on Bills) 0.55 / 0.39 / 0.51
  - 5 example topics instead of 2: 0.50 / 0.33 / 0.49, the worst setting, so the authors recommend 2-3 high-quality examples
  - shuffled document order 0.55 / 0.40 / 0.52
  - Mistral-7B for assignment 0.51 / 0.37 / 0.46
  - **repeated identical run vs the earlier output: 0.95 / 0.92 / 0.92**

  — [arXiv HTML 2311.01449v2](https://arxiv.org/html/2311.01449v2)
- With open models for generation, Mistral-7B produced 1,418 topics and GPT-3.5-turbo 151, most of them overly specific. The authors recommend GPT-4-class models for the generation step. — [arXiv HTML 2311.01449v2](https://arxiv.org/html/2311.01449v2)
- *Transfers:* (1) The cosine ≥ 0.5 candidate retrieval followed by batched LLM merge judgments is a direct template for merging the team's issue questions. (2) Freezing the label list and then running a separate quote-grounded assignment pass is the main lever against label drift. (3) TopicGPT's measure is *partition agreement against ground truth* across perturbations, not label-string agreement, which matches the team's ARI framing. Prompt perturbation changed ARI by only about 0.03 when assignment used a frozen list. (4) Adding more example topics *hurt*, so prompt-ensemble members should not overload few-shot examples. (5) Small local models over-generate fine-grained labels, so an explicit granularity constraint and a strong merge step are needed.

**LLooM concept induction (Lam, Teoh, Landay, Heer, Bernstein; CHI 2024) [verified]**
- Operators:
  - Optional *Seed* term conditions the prompts (e.g., "social issues").
  - *Distill-filter* extracts exact quotes with gpt-3.5-turbo.
  - *Distill-summarize* writes 2-4 bullets of 5-8 words each. Both Distill steps run at temperature 0.
  - *Cluster* pools all bullets, embeds them with text-embedding-ada-002, and clusters them with **HDBSCAN**, chosen because it does not force every point into a cluster.
  - *Synthesize* uses GPT-4 with chain-of-thought on each cluster. It outputs a 2-4 word concept name, a one-sentence inclusion-criteria prompt, and 1-2 representative example IDs.
  - *Score* applies each criteria prompt to every document on a 5-point scale, thresholded to binary.
  - *Loop* re-runs on outliers: documents matching no concept, and documents matching only "generic" concepts (those matching at least 50% of data).

  — [arXiv HTML 2404.12259v1](https://arxiv.org/html/2404.12259v1)
- Merging is analyst-driven in the Workbench (merge, split, edit, rescore). The paper describes no automatic dedup operator. — [arXiv HTML 2404.12259v1](https://arxiv.org/html/2404.12259v1)
- Stability: 10 independent trials per method and dataset, with shuffled documents, new API sessions, and new BERTopic models. Automated coverage had MAE 0.07 vs manual matching. The authors acknowledge re-run variability but **do not report concept overlap across runs**. Mean ground-truth concept coverage: LLooM 0.74 (Bills) and 0.81 (Wiki); GPT-4 Turbo 0.56 and 0.82; BERTopic 0.29 and 0.63. — [arXiv HTML 2404.12259v1](https://arxiv.org/html/2404.12259v1)
- *Transfers:* LLooM's Distill-summarize step is the closest analogue to the team's "rewrite to a 4-12 word issue question." LLooM then embeds and density-clusters the short summaries, and only afterwards writes *one* canonical concept name per cluster plus explicit inclusion criteria. Using criteria-based scoring (rather than nearest-label matching) for comment-to-anchor attachment is a transferable design. HDBSCAN's noise class offers a principled "no anchor" option for comments.

**TnT-LLM (Wan et al., Microsoft; arXiv 2403.12173, 2024; widely cited as KDD 2025, but the venue was not confirmed on the fetched page) [verified mechanics]**
- Stage 1: an LLM summarizes each conversation (target about 20 words). Stage 2: summaries go in minibatches of 200. An initial prompt drafts a taxonomy, an *update* prompt revises it batch by batch (evaluate, note issues, modify), and a final *review* prompt checks format and quality. The authors frame this as SGD (Θ(m+1) = Θ(m) − η∇L), with the learning rate set implicitly by the LLM. — [arXiv HTML 2403.12173](https://arxiv.org/html/2403.12173)
- Stability controls:
  - **10 trials, with the best taxonomy selected on a validation set by an LLM selection prompt.** K-means baselines also ran 10 trials, selected by Silhouette.
  - Option positions randomized, with repeated evaluations to reduce position bias.
  - Temperature 0.5 for generation and 0.2 for update. Up to 5 retries on failure, each raising temperature by 0.1.
  - GPT-3.5-Turbo completed only 4/10 (intent) and 1/10 (domain) runs, against 10/10 for GPT-4.

  — [arXiv HTML 2403.12173](https://arxiv.org/html/2403.12173)
- Evaluation: coverage above 99.5%. Human Fleiss κ for label accuracy and relevance was about 0.38-0.48. GPT-4 vs human majority Cohen κ was 0.29-0.58. In Phase 2, GPT-4 pseudo-labels train lightweight classifiers (logistic regression, MLP) on ada2 embeddings, which matched GPT-4's own primary-label accuracy (about 0.64-0.66 vs human gold). — [arXiv HTML 2403.12173](https://arxiv.org/html/2403.12173)
- *Transfers:* (1) "Summarize, then induce a taxonomy over the summaries" separates per-item rewriting from codebook construction, as the team's pipeline already does. (2) Explicit **multi-run plus selection** (generate N codebooks, pick one by a held-out criterion) is a documented alternative to averaging prompt variants. (3) Distilling to an embedding classifier freezes assignment and removes prompt sensitivity at inference time. (4) Weaker models fail often during iterative taxonomy updates, a risk for a small local model.

**Clio (Tamkin et al., Anthropic; arXiv 2412.13678, Dec 2024) [verified, partial]**
- Pipeline: an LLM extracts per-conversation *facets* (e.g., topic, language). Natural-language facet summaries are embedded (the paper cites Sentence-BERT / all-mpnet-base-v2) and clustered with **k-means**. An LLM names and summarizes each cluster from sample conversations. Clusters are recursively organized into a multi-level hierarchy with k-means plus prompting, from a few dozen top-level patterns to thousands of fine-grained ones. — [arXiv HTML 2412.13678](https://arxiv.org/html/2412.13678)
- Privacy: the summary prompts omit private information, minimum unique-account and conversation thresholds apply per cluster, and a model audits clusters. On 19,476 synthetic multilingual transcripts with known topic distributions, Clio reconstructed the ground-truth categories with **94% accuracy** (about 5% for random). Cluster concern scores correlated with safety-classifier flags at r = 0.71. — [arXiv HTML 2412.13678](https://arxiv.org/html/2412.13678)
- No run-to-run or prompt-variant stability analysis appeared in the portion read (first ~100k characters). — [arXiv HTML 2412.13678](https://arxiv.org/html/2412.13678)
- *Transfers:* Clio clusters the embeddings of *LLM-written summaries* rather than raw text, and **names clusters after clustering**. That makes the final label a function of the cluster members, not of any single item's generated label. Validating with a synthetic corpus of known structure (planted issues) is a cheap way to test whether prompt variants recover the same partition.

**k-LLMmeans (Diaz-Rodriguez; arXiv 2502.09667, v1 Feb 2025, revised to 2026) [abstract-level]**
- This is k-means where, at scheduled iterations, an LLM summary of each cluster's texts is embedded and **replaces the numeric centroid**. Ordinary k-means updates run in between. LLM calls do not scale with dataset size, and a mini-batch variant handles streams. v2 retitled the paper to emphasize "Stable" clustering, and later versions add k-NLPmeans (extractive summaries). — [arXiv 2502.09667v2](https://arxiv.org/html/2502.09667v2); [arXiv abs](https://arxiv.org/abs/2502.09667v2)
- *Transfers:* The video-side anchor groups could be re-summarized into a canonical issue question whose embedding serves as the anchor centroid. Comments would then attach to an LLM-synthesized centroid rather than to one generated label, which dampens per-label wording noise. Stability claims were not checked in full text (see Gaps).

**GoalEx, "Goal-Driven Explainable Clustering via Language Descriptions" (Wang, Shang, Zhong; EMNLP 2023) [unverified-detail]**
- A propose-assign-select loop: an LLM proposes natural-language cluster *descriptions* conditioned on a user goal, an LLM judges which descriptions each document satisfies, and an integer linear program selects a subset of descriptions that covers documents with minimal overlap. — [arXiv 2305.13749](https://arxiv.org/abs/2305.13749)
- *Transfers:* Over-generate candidate issue labels, possibly from several prompts, then use a *coverage/overlap optimizer* to pick a non-redundant subset. This is an explicit consolidation alternative to embedding thresholds.

**PromptTopic (Wang et al., "Prompting Large Language Models for Topic Modeling", IEEE BigData 2023) [unverified-detail]**
- The LLM extracts topics per document (sentence level), then topics are condensed by prompt-based merging or word-similarity-based merging. — [arXiv 2312.09693](https://arxiv.org/abs/2312.09693)
- *Transfers:* Another per-document generate-then-merge design. It reported no prompt-ensemble controls (unverified).

**QualIT (Kapoor et al., Amazon, 2024) [unverified-detail]**
- The LLM extracts key phrases per document, checks them for hallucination (coherence with the source), then embeds and clusters them in two levels (main topics, subtopics). — [arXiv 2409.15626](https://arxiv.org/abs/2409.15626)
- *Transfers:* A per-item faithfulness check on generated phrases could filter issue questions that drift from the note block before embedding.

**BERTopic with LLM representation [unverified-detail]**
- BERTopic clusters raw-document embeddings (UMAP + HDBSCAN), then optionally asks an LLM to *name* each topic from representative documents and keywords. The LLM only labels clusters and does not define them, so partition stability is independent of prompt wording. — [BERTopic docs: LLM representation](https://maartengr.github.io/BERTopic/getting_started/representation/llm.html)
- *Transfers:* The "cluster first, label later" ordering is the strongest structural guard against the team's problem. In the team's pipeline, prompt variation enters *before* clustering, because generated texts are what get embedded.

### Inferences
- Across all systems, only TopicGPT quantifies the effect of prompt variation on the final partition, and only at the level of one domain-swapped prompt. None uses an ensemble of paraphrased prompts as a stabilizer. The team's ARI of 0.58-0.71 between prompts is therefore not directly benchmarked in the literature. TopicGPT's 0.92 repeated-run ARI (same prompt) and its ~0.39-0.42 ARI vs ground truth under perturbations are the closest reference points, and they are different quantities.
- The dominant robustness patterns are structural: (a) cluster raw or summarized items, then name clusters (Clio, BERTopic, LLooM); (b) freeze the codebook, then assign with grounding (TopicGPT quotes, LLooM criteria); (c) multi-run plus selection (TnT-LLM); (d) over-generate, then optimize selection (GoalEx). For the team, a plausible design is to generate issue questions with K prompts, pool all K×N texts into one embedding space, cluster the pool, then synthesize one canonical label per cluster. This combines LLooM/Clio-style naming with pooling, but it is an inference, not a documented system.
- TopicGPT's frequency-threshold pruning offers a natural hook for prompt ensembles: a code generated under only one prompt variant can be treated as low-support (inference).

### Gaps
- HiCode / hierarchical inductive code generation: no reliable primary source was located this session, so its mechanics are not reported.
- "LLM-in-the-loop topic models" as a distinct named system was not separately verified beyond Dai et al. 2023 (Q2).
- k-LLMmeans' quantitative stability results were not read in full text.
- Clio's appendix (second half of the paper) was not read, so any stability or manual-validation numbers there are missing.
- No source found reports **label-string agreement** (e.g., the share of identical or near-identical labels) across paraphrased prompts for any of these systems.

## Q2. LLM-assisted qualitative coding and thematic analysis: who uses multiple prompts, personas, or coders, and how they reconcile

### Takeaway
Early work (2023) is mostly single-prompt, human-in-the-loop: the LLM proposes codes and humans fix the codebook. From 2025-2026, multi-agent "LLM coders" setups appear: independent coding, then debate on disagreements, then consensus or new rules. The evidence says persona diversity and debate give **at most marginal accuracy gains**. Multi-agent discussion produces **convergence** that can mask ambiguity. Disagreement is most useful as a **diagnostic for fuzzy code boundaries** rather than as something to average away.

### Cited Findings
- **Xiao et al. 2023 (IUI '23 Companion), "Supporting Qualitative Analysis with LLMs: Combining Codebook with GPT-3 for Deductive Coding" [unverified-detail]:** the codebook is given in the prompt, zero-shot vs example-based. Agreement with experts was fair to substantial depending on prompt design, and the paper's own framing is that prompt design matters. — [arXiv 2304.10548](https://arxiv.org/abs/2304.10548)
- **Dai, Xiong, Ku 2023 (Findings of EMNLP 2023), "LLM-in-the-loop: Leveraging LLMs for Thematic Analysis" [unverified-detail]:** the LLM and a human coder iteratively develop the codebook in a human-human-style discussion loop, aiming to match human thematic analysis quality at lower cost. — [arXiv 2310.15100](https://arxiv.org/abs/2310.15100)
- **Chew et al. 2023, "LLM-Assisted Content Analysis (LACA)" [unverified-detail]:** for deductive coding, the LLM co-develops the codebook with humans, then intercoder reliability between the LLM and humans is tested per code. The paper also proposes checking whether LLM agreement exceeds a random-coder baseline. — [arXiv 2306.14924](https://arxiv.org/abs/2306.14924)
- **De Paoli 2023/2024 (Social Science Computer Review), inductive thematic analysis of semi-structured interviews with GPT-3.5 [unverified-detail]:** the LLM generates initial codes from interview chunks, then themes from the codes, following Braun & Clarke phases. Results are compared qualitatively with original human analyses, and some themes were reproduced. No multi-prompt control was used. — [arXiv 2305.13014](https://arxiv.org/abs/2305.13014)
- **CollabCoder (Gao et al., CHI 2024) [unverified-detail]:** a collaborative coding tool where the LLM suggests codes and helps human coders compare and merge independent codes into a shared codebook. The LLM is a mediator for *human* inter-coder disagreement, not one of several LLM coders. — [arXiv 2304.07366](https://arxiv.org/abs/2304.07366)
- **Borchers et al., "Temperature and Persona Shape LLM Agent Consensus With Minimal Accuracy Gains in Qualitative Coding" (arXiv 2507.11198; EDM 2026 full paper) [abstract-level]:**
  - Setup: a deductive 8-code codebook coded by multiple LLM agents with personas (neutral, assertive, empathetic).
  - Persona effects: mixed personas *delayed* consensus in 4 of 6 LLMs compared with uniform personas.
  - Accuracy: multi-agent deliberation only **marginally** beat single-agent predictions, and only under narrow conditions.
  - Temperature: higher temperature weakened persona effects in 3 LLMs.

  — [arXiv 2507.11198](https://arxiv.org/abs/2507.11198); [EDM 2026 PDF](https://educationaldatamining.org/wp-content/uploads/2026/proceedings/2026.EDM.full-papers/2026.EDM.full-papers.21.pdf)
- **Kim & Mitchell (Stanford), "How AI Coders Discuss, Disagree, and Reach Consensus" (arXiv 2609.11109, Sep 2026; no venue stated) [verified, first ~40% of text]:**
  - Setup: two gpt-4o-mini coders with **no personas**. They code independently (present / absent / "Undecidable", with justification), then discuss only conflicting cases for up to 3 rounds. Where consensus forms, they write a **new labeling rule**, and both re-code the full dataset with the new rules. Temperature 0 for coding, 0.7 for discussion. Four domains, five labels each.
  - AI-AI Cohen κ was above 0.85, yet F1 was only 0.31-0.89 (mean 0.68), and accuracy on conflict cases averaged 0.60.
  - Shorter codebooks and shorter excerpts predicted higher accuracy. *More* discussion turns predicted higher accuracy, while a *higher share of resolved conflicts* predicted *lower* accuracy.
  - The authors warn that quick AI-AI agreement can hide ambiguity and yield confidently wrong labels. No prompt-sensitivity analysis was reported.

  — [arXiv HTML 2609.11109v1](https://arxiv.org/html/2609.11109v1)
- **"Emergent Convergence in Multi-Agent LLM Annotation" (BlackboxNLP 2025) [abstract-level]:** 7,500 simulated multi-agent, multi-round discussions on an *inductive* coding task. Agents' outputs converged lexically and semantically through interaction. — [ACL Anthology PDF](https://aclanthology.org/2025.blackboxnlp-1.12.pdf)
- **"Disagreement as Data: Reasoning Trace Analytics in Multi-Agent Systems" (arXiv 2601.12618, 2026) [abstract-level]:** LLM-LLM disagreement is treated as a signal of fuzzy or overlapping codebook boundaries and used to refine the codebook. — [arXiv 2601.12618](https://www.arxiv.org/pdf/2601.12618)
- **Halterman & Keith, "Codebook LLMs" (Political Analysis 2025; arXiv 2407.10747) [verified]:**
  - Framework: a five-stage procedure. Codebooks are restructured into label, definition, clarifications, negative clarifications, and examples, then put through label-free behavioral tests, zero-shot evaluation, error analysis, and finally QLoRA fine-tuning.
  - Order invariance (Test IV): **all models were sensitive to category order** when categories were reversed or shuffled.
  - Generic labels (Test VI): replacing label names with LABEL_1-style names **hurt** performance.
  - Swapped labels (Test VII): swapping label names with definitions also **hurt** performance. The models relied heavily on label-name semantics over definitions.
  - Exclusion criteria (Test V): models often failed to follow them.
  - Mistral-7B was drawn to labels that shared words with the text.
  - Instruction tuning raised BFRS weighted F1 from about 0.53-0.57 zero-shot to 0.81-0.82.

  — [arXiv HTML 2407.10747v2](https://arxiv.org/html/2407.10747v2); [Political Analysis PDF](https://www.cambridge.org/core/services/aop-cambridge-core/content/view/7B323A0E47F782F2698A0AE849EA00DE/S104719872510017Xa.pdf/div-class-title-codebook-llms-evaluating-llms-as-measurement-tools-for-political-science-concepts-div.pdf)

### Inferences
- *Transfers (multi-agent):* Persona diversity is not a proven route to better or more stable codes (Borchers). Debate tends toward convergence, which inflates agreement without improving validity (Kim & Mitchell; BlackboxNLP 2025). For the team, multi-prompt runs are better used as **parallel independent coders whose disagreement flags ambiguous items or codes** (Disagreement-as-Data) than as a debate-to-consensus step.
- *Transfers (Halterman & Keith):* Their order-invariance test is directly reusable. Shuffling the order of anchor groups or codebook entries in the comment-coding prompt is a cheap prompt-sensitivity probe. Their finding that models lean on label-name semantics over definitions suggests the wording of the generated 4-12 word issue question will dominate downstream comment attachment. This argues for attaching comments by criteria or definitions plus embeddings rather than by label text alone.
- *Transfers (Kim & Mitchell):* The "Undecidable" option plus rule-writing after disagreement maps onto adding an explicit "no matching anchor / ambiguous" output to the comment-coding JSON, and logging rule refinements as codebook clarifications.
- Most 2023 qualitative-coding papers (Xiao, Dai, Chew, De Paoli, CollabCoder) rely on a **human-fixed codebook** as the stability mechanism, not on prompt ensembles.

### Gaps
- "MindCoder" and similar named multi-persona *inductive* coding systems: no reliable primary source was located this session.
- No study found compares *inductive* code sets generated from paraphrased prompts using a partition metric (ARI/NMI) together with a label-overlap metric. Evidence is thin for inductive (open) coding specifically. Most multi-agent studies are deductive.
- Exact numbers for Xiao, Dai, Chew, De Paoli, and CollabCoder were not re-verified this session.

## Q3. Prompt sensitivity reports and recommended mitigations in annotation and coding

### Takeaway
The annotation literature consistently finds that prompt wording changes outputs substantially, more so on interpretive tasks than on knowledge-anchored ones. The recommended mitigations are:
- majority vote over repeated runs and/or over prompt variants (Reiss; Inter-Prompt Reliability),
- low or zero temperature (Gilardi; Reiss),
- fixed, well-specified codebooks,
- explicit order and label-name robustness tests (Halterman & Keith),
- reporting stability as a distribution across prompts rather than a single-prompt score.

Temperature 0 alone does not address prompt sensitivity.

### Cited Findings
- **Reiss 2023, "Testing the Reliability of ChatGPT for Text Annotation and Classification: A Cautionary Remark" (arXiv 2304.11085):** on a news/not-news website classification task, small prompt variations caused large inconsistencies, and lower temperature increased within-setting consistency. Consistency can fall short of scientific reliability thresholds. The paper recommends a **majority vote over ≥3 (ideally more) repetitions** per input and advises against unsupervised use. — [arXiv 2304.11085](https://arxiv.org/pdf/2304.11085) [abstract-level/snippet]
- **Gilardi, Alizadeh, Kubli 2023 (PNAS), "ChatGPT Outperforms Crowd-Workers for Text-Annotation Tasks":** intercoder agreement was about 56% for MTurk, 79% for trained annotators, 91% for ChatGPT at temperature 1, and 97% at temperature 0.2. The authors suggest low temperature increases consistency without reducing accuracy. — [arXiv 2303.15056](https://arxiv.org/pdf/2303.15056) [snippet]
- **Liu 2026, "What Is Actually Being Annotated? Inter-Prompt Reliability" (arXiv 2604.16413):** proposes Inter-Prompt Reliability (IPR), modeled on inter-rater reliability. It is measured with the Pairwise Agreement Rate (PAR) and its distribution across semantically equivalent prompts. Stochastic variation was substantial on the interpretive task (TREC) and lower on the knowledge-anchored one (Politifact). **Majority voting across prompt variants significantly improves reproducibility and reduces variance.** The prompt is framed as a measurement instrument whose wording carries methodological uncertainty. — [arXiv 2604.16413](https://arxiv.org/abs/2604.16413) [abstract-level; number of variants and models not visible]
- **"What Did I Do Wrong? Quantifying LLMs' Sensitivity and Consistency to Prompt Engineering" (NAACL 2025):** defines *sensitivity* as how often predictions change under rephrasings, and *consistency* as how stable predictions are within a class. Underspecified prompts showed higher variance than prompts with explicit instructions. — [ACL Anthology](https://aclanthology.org/2025.naacl-long.73.pdf) [snippet]
- **"Observational Equivalence of LLM and Human Annotation" (arXiv 2609.22133, 2026):** prompt disagreement tracked expert-defined ambiguity, with model-dependent strength. — [arXiv 2609.22133](https://arxiv.org/pdf/2609.22133) [snippet]
- **Relevance-judgment prompt sensitivity (arXiv 2504.12408):** binary and pairwise formats were least sensitive to prompt choice, and graded formats more variable. — [arXiv 2504.12408](https://arxiv.org/html/2504.12408v1) [snippet]
- **Halterman & Keith 2025:** all tested open models were sensitive to category order, and outputs depended on label-name semantics. — [arXiv HTML 2407.10747v2](https://arxiv.org/html/2407.10747v2)
- **TopicGPT:** even at temperature 0 / top_p 0, the API was not fully deterministic. A repeated identical run gave ARI 0.92 vs the previous run, while prompt and sample perturbations shifted ground-truth ARI by about 0.02-0.09. — [arXiv HTML 2311.01449v2](https://arxiv.org/html/2311.01449v2)
- **TnT-LLM:** randomized option positions and repeated evaluations against position bias; 10 trials plus validation-based selection. — [arXiv HTML 2403.12173](https://arxiv.org/html/2403.12173)

### Inferences
- The team's setup (local model, temperature 0) removes sampling noise, so the residual ARI 0.58-0.71 is pure *prompt-instrument* variance. The IPR framing fits it exactly: report the distribution of pairwise ARI across K paraphrases, and aggregate across prompts (for discrete comment-to-anchor assignments, majority vote across prompt variants is the documented fix).
- For *generated text* (issue questions), majority vote is not directly applicable. The analogue is to pool generated texts across prompts and cluster, or to choose a medoid label per item across prompt variants in embedding space. This is an inference; no source above demonstrates it for open-ended labels.
- Ambiguity-linked disagreement (2609.22133; Disagreement-as-Data) suggests items with low cross-prompt agreement should be flagged as ambiguous rather than forced. Per-item cross-prompt agreement can serve as a confidence score.
- Interpretive tasks are more prompt-sensitive than knowledge-anchored ones (IPR). Stance-neutral issue reformulation is interpretive, so high sensitivity is expected, and ARI of about 0.6-0.7 is plausible rather than anomalous (inference).

### Gaps
- Exact numbers of prompt variants, models, and effect sizes in IPR (2604.16413) and Reiss 2023 were not visible in the abstract-level fetches.
- No study found quantifies prompt sensitivity for *short-label generation* (4-12 word summaries) followed by clustering. All quantified sensitivity work is classification or annotation.

## Q4. Evaluation measures used for generated themes and codebooks

### Takeaway
Systems evaluate generated codebooks by:
- **partition agreement with ground truth** (P1 harmonic purity, ARI, NMI: TopicGPT),
- **coverage of ground-truth concepts or documents** (LLooM, TnT-LLM, Clio),
- **human ratings** of label accuracy and relevance with Fleiss or Cohen κ (TnT-LLM),
- **misaligned-topic share in human evaluation** (TopicGPT),
- **LLM-human or LLM-LLM κ / F1** for deductive coding (Kim & Mitchell; Halterman & Keith).

Run-to-run or prompt-to-prompt stability is rarely reported, and redundancy is mostly handled procedurally (merge steps) rather than measured.

### Cited Findings
- TopicGPT: P1 (harmonic mean of purity and inverse purity), ARI, NMI vs human labels; human judgment of misaligned topics; repeated-run agreement (0.95 / 0.92 / 0.92). — [arXiv HTML 2311.01449v2](https://arxiv.org/html/2311.01449v2)
- LLooM: ground-truth concept coverage over 10 trials (LLooM 0.74/0.81 vs BERTopic 0.29/0.63); share of uncategorized examples (e.g., 9.5% LLooM vs 44.4% BERTopic on toxic content); cross-method coverage (LLooM covered 87.5% of BERTopic's concepts, and BERTopic covered 50% of LLooM's). — [arXiv HTML 2404.12259v1](https://arxiv.org/html/2404.12259v1)
- TnT-LLM: coverage (above 99.5%); human-rated label accuracy and relevance (Fleiss κ about 0.38-0.48); GPT-4-as-rater vs human κ; downstream classifier accuracy vs human gold. — [arXiv HTML 2403.12173](https://arxiv.org/html/2403.12173)
- Clio: reconstruction accuracy on synthetic data with known topic distributions (94%); correlation with independent classifiers (r = 0.71). — [arXiv HTML 2412.13678](https://arxiv.org/html/2412.13678)
- Kim & Mitchell: AI-AI Cohen κ (above 0.85) vs F1 against human labels (mean 0.68). This shows high inter-LLM agreement does not imply validity. — [arXiv HTML 2609.11109v1](https://arxiv.org/html/2609.11109v1)
- Halterman & Keith: weighted F1; percent agreement and Fleiss κ across category orderings (order-invariance test). — [arXiv HTML 2407.10747v2](https://arxiv.org/html/2407.10747v2)
- IPR: Pairwise Agreement Rate and its distribution across semantically equivalent prompts. — [arXiv 2604.16413](https://arxiv.org/abs/2604.16413)

### Inferences
- A complete evaluation for the team's pipeline, assembled from these sources:
  - (a) pairwise ARI/NMI across prompt variants, as an IPR-style distribution;
  - (b) P1/ARI against a small human-coded gold subset (TopicGPT);
  - (c) coverage and "no-anchor" rate for comments (LLooM);
  - (d) human-rated issue-question accuracy and neutrality (TnT-LLM style);
  - (e) order-invariance and label-swap probes for comment attachment (Halterman & Keith);
  - (f) a planted-structure synthetic set (Clio-style).
- Stability metrics (a) and validity metrics (b)/(d) should be reported separately. Kim & Mitchell show agreement can be high while accuracy is mediocre, so raising cross-prompt ARI alone (e.g., by making prompts more templated) could increase agreement without improving validity.

### Gaps
- No standard redundancy metric for generated codebooks was found. Systems merge duplicates procedurally (TopicGPT cosine ≥ 0.5 plus LLM merge; LLooM analyst merge) but do not report post-merge redundancy rates.
- No source found evaluates label-string stability (e.g., embedding similarity between matched labels across runs). This would complement partition ARI for the team.
