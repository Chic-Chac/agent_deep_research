---
id: N-0011
type: note
task: T-0002
title: "Evaluating issue-level clustering of short texts and an emerging codebook with a small human label budget (100–300 judgments/video) plus a local LLM judge"
status: final
created: 2026-11-02
tags: [evaluation, llm-judge, clustering]
---

# Evaluating issue-level clustering of short texts and an emerging codebook with a small human label budget (100–300 judgments/video) plus a local LLM judge

Scope note: verification status. Sources marked **[verified]** were opened or confirmed via search snippets/abstracts in this session (Nov 2026). Sources marked **[from memory, unverified this session]** are well-known papers whose details I recall but could not re-open within the tool budget. Treat their specific numbers with caution. Any arithmetic (CI widths, sample sizes) is my own and is marked as an inference.

---

## Q1. Pairwise "same issue?" judgments vs full cluster labels: sample efficiency, pair sampling, and pair-based metrics

### Takeaway
Use pairwise "same issue?" judgments as the main human currency. Each judgment takes seconds and does not depend on any one pipeline's partition, so the same labeled pairs can score every variant (raw vs gist, embedder, clusterer). Compute pairwise precision/recall/F1 with design-based (inverse-inclusion-probability) estimators over a cosine-stratified pair sample. Use the entity-resolution (ER) evaluation literature (Binette et al.; OASIS) as the statistical template. Full cluster labels (or "fully resolved" sampled clusters) are needed only for B-cubed and cluster-level estimates, and they cost much more per unit.

### Cited Findings
- Binette et al., "Estimating the Performance of Entity Resolution Algorithms: Lessons Learned Through PatentsView.org" (arXiv Oct 2022): treats pairwise precision and pairwise recall (defined over links between records) as the core metrics. It estimates them from a sample of ground-truth clusters, not from the full ground truth. **[verified]** — [arXiv 2210.01230](https://arxiv.org/pdf/2210.01230)
- Binette et al., "How to Evaluate Entity Resolution Systems: An Entity-Centric Framework…" (arXiv Apr 2024): proposes an *entity-centric* labeling method in which you sample entities (true clusters) and fully resolve them. It then estimates cluster and pairwise precision/recall from that sample, with uncertainty. **[verified]** — [arXiv 2404.05622](https://arxiv.org/pdf/2404.05622)
- The companion `er_evaluation` Python package has estimators for pairwise P/R/F, cluster P/R/F, and **B-cubed precision/recall**, plus summary statistics such as average cluster size. All of them assume "a weighted sample of ground truth clusters" with user-supplied sampling weights (e.g., inverse probability weights), and they return a point estimate plus a standard-deviation estimate. The predicted clustering should cover the whole population. **[verified]** — [er-evaluation docs](https://er-evaluation.readthedocs.io/en/latest/er_evaluation.estimators.html); [PyPI](https://pypi.org/project/ER-Evaluation/2.3.0/)
- B³ (B-cubed) precision/recall gives equal weight per item/ground-truth cluster, which is how it differs from pairwise metrics that over-weight large clusters. **[verified, via search summary of Binette 2024]** — [arXiv 2404.05622](https://arxiv.org/abs/2404.05622v1)
- Marchant & Rubinstein, "In Search of an Entity Resolution OASIS: Optimal Asymptotic Sequential Importance Sampling" (PVLDB 2017): an adaptive importance-sampling method for estimating F-measure/precision/recall under heavy class imbalance (few true matches among many pairs). It updates its instrumental distribution through a Bayesian model of the oracle as labels arrive, its estimates are statistically consistent, and it cut labels by up to 83% with no loss of accuracy. Open-source Python package. **[verified]** — [arXiv 1703.00617](https://arxiv.org/abs/1703.00617v2)
- ClusterLLM (Zhang et al., EMNLP 2023) uses *pairwise* questions ("do A and B belong to the same category?") to choose clustering **granularity**, i.e., which level of a hierarchy best matches the oracle's pairwise judgments. It reached that on 14 datasets at about $0.6/dataset of API cost. **[verified]** — [arXiv 2305.14871](https://arxiv.org/abs/2305.14871); [ACL Anthology 2023.emnlp-main.858](https://preview.aclanthology.org/setup/2023.emnlp-main.858)
- Viswanathan et al., "Large Language Models Enable Few-Shot Clustering" (TACL 2024): uses an LLM as a pseudo-oracle for pairwise must-link/cannot-link constraints and for keyphrase expansion before clustering. **[from memory, unverified this session]** — [arXiv 2307.00524](https://arxiv.org/abs/2307.00524)
- Amigó et al. (2009), "A comparison of extrinsic clustering evaluation metrics based on formal constraints" (Information Retrieval 12): argues that B-cubed is the only common family that satisfies all four formal constraints (homogeneity, completeness, rag bag, size-vs-quantity). This is the usual justification for preferring B³ over purity or pairwise metrics. **[from memory, unverified this session]** — [Springer DOI 10.1007/s10791-008-9066-8](https://doi.org/10.1007/s10791-008-9066-8)

### Inferences
- **Why pairs, not full labels, at 100–300 judgments/video.** Full labels need an agreed label set first, and the codebook is still emerging, so the label set itself is what's being evaluated. Pair judgments avoid that circularity: "same issue?" can be answered without a codebook. One shared pair sample can score all pipeline variants, so the variant comparison becomes a paired comparison on the same items, which has lower variance than labeling separate samples per variant.
- **Imbalance problem.** With K≈10–30 issues per video, a uniformly random pair is "same issue" with probability around 1/K to 1/(K/2), so roughly 3–10%. Out of 200 uniform pairs you would get only about 6–20 positives, which is far too few to estimate recall. Stratify instead.
- **Recommended pair-sampling design (per video; my synthesis):**
  1. Enumerate all comment pairs (2k comments gives about 2M pairs, which is cheap to enumerate). Compute cosine similarity under a *reference* embedding, preferably a different embedder from the ones being compared so the design doesn't favor one of them. Bin into about 5–8 cosine strata (e.g., quantile bins, with the top 1% split further).
  2. Cross the strata with **co-clustering status under each candidate pipeline**: "same cluster in all variants", "same in some" (the disagreement stratum), and "same in none".
  3. Allocate labels roughly as: 40% to disagreement/boundary strata (most informative for ranking variants), 40% to high-cosine strata (for precision), and 20% to low/mid-cosine strata (to catch recall misses, i.e., same issue phrased differently, or Korean vs English). Record each pair's inclusion probability π.
  4. Estimate the counts TP (same-issue and co-clustered), FP, and FN with Horvitz–Thompson weights 1/π. Pairwise P = TP/(TP+FP), R = TP/(TP+FN), F1 = harmonic mean. Report bootstrap CIs over pairs, resampling within strata. This is the same estimator logic as `er_evaluation` and OASIS, applied to pairs.
  5. **Rand index estimate:** RI = (TP+TN)/all pairs, so it falls out of the same weighted counts. It is dominated by TN when K is large, so report ARI-style chance-corrected pairwise F1 or pairwise P/R instead of raw RI.
  6. **B-cubed from samples:** needs, for each sampled *item*, the set of items truly sharing its issue. A cheap approximation is to sample m≈10–20 anchor comments per video and have a human mark, among the anchor's top-N nearest neighbors (N≈30–50 under a high-recall union of embedders) plus its assigned cluster members, which ones share the anchor's issue. That gives per-anchor B³ precision directly and a lower-bounded B³ recall. This mirrors Binette's entity-centric sampling.
- **CI arithmetic (my own).** With n=100 labeled pairs in a stratum and a true proportion of 0.8, the binomial SE is about 0.04 (95% CI ±0.08). With n=300 it is about 0.023 (±0.045). So 100–300 judgments can separate pipelines whose pairwise F1 differs by about 0.08–0.10 or more, but not by 0.02. Paired designs (the same pairs scored under every variant) and PPI with an LLM judge (Q6) are the main ways to tighten this.
- **Granularity confound.** Pairwise precision rises and recall falls as K grows. Compare variants either at matched K or on the whole P/R curve (e.g., over HDBSCAN `min_cluster_size` values or agglomerative cut levels). ClusterLLM's approach of picking the cut that best agrees with pairwise oracle labels is directly applicable.

### Gaps
- I did not find a paper that evaluates *pairwise human labels vs full cluster labels* for short-text topic clustering in terms of sample efficiency. The ER literature (Binette, OASIS) is the closest rigorous analog. A head-to-head efficiency comparison for social-media comments would need a simulation on the team's own data.
- I did not verify whether `er_evaluation` accepts pair samples directly. The docs I saw describe samples of *clusters*, not pairs.

---

## Q2. Triplet judgments ("is B or C more about the same issue as A?")

### Takeaway
Triplet (relative) judgments suit the *embedding/preprocessing* choice, i.e., which representation puts same-issue comments closer. ClusterLLM shows an LLM can answer them, and entropy-based selection of hard triplets keeps query counts low. C-STS shows that conditioning the similarity on a stated aspect (here: "the issue/claim being argued") is essential, because unconditioned similarity is ill-posed. Triplets don't yield clustering P/R. Use them to rank embedders and preprocessing variants, and pairs to score partitions.

### Cited Findings
- ClusterLLM (EMNLP 2023): Stage 1 prompts an LLM with hard triplets ("does A better correspond to B than C?") to learn the user's preferred *clustering perspective* (topic vs intent vs emotion) and fine-tunes a small embedder on the answers. An **entropy-based triplet sampling** strategy picks the most informative triplets, which reduces query cost. **[verified]** — [arXiv 2305.14871](https://arxiv.org/abs/2305.14871); [EMNLP 2023 PDF](https://preview.aclanthology.org/fix-sig-urls/2023.emnlp-main.858.pdf)
- C-STS (Deshpande et al., EMNLP 2023): argues general STS "may be inherently ill-defined" because similarity depends on which aspect you focus on. It defines similarity conditioned on a natural-language condition, building a dataset of about 19K (sentence pair, condition, similarity) instances. Models that do well on standard STS (SimCSE, Flan-T5, GPT-4) all score below 50 Spearman on C-STS. **[verified]** — [ACL Anthology 2023.emnlp-main.345](https://aclanthology.org/2023.emnlp-main.345); [arXiv 2305.15093](https://arxiv.org/pdf/2305.15093v2.pdf)

### Inferences
- **Why condition matters for this project.** A comment from a debate video can be similar to another by *speaker targeted*, *stance*, *emotion*, or *issue*. Raw-text embeddings mix these together, and an LLM gist step is effectively a way to project onto the "issue" condition. The C-STS result (general embedders do poorly when similarity is conditioned) is indirect evidence that gist+embed should beat raw+embed for issue clustering. Test this directly with issue-conditioned triplets.
- **Triplet protocol (my synthesis).** For each anchor A, draw B and C so that the variants under comparison *disagree*: embedder/variant X ranks B closer to A, and variant Y ranks C closer. The human (or LLM judge) answers "Which is about the same issue/claim as A: B, C, or neither/both?". Keep the "neither/both" option, because forced choice inflates noise on short comments. Score each variant by triplet accuracy on decisive triplets. Disagreement-targeted triplets are maximally informative for *ranking* variants but biased for *absolute* accuracy, so report them as "win rate on contested triplets", not as an accuracy estimate.
- **Budget.** Triplets take about 1.5–2× the time of a pair. 100–150 contested triplets per comparison usually suffice to detect a 60/40 preference (my arithmetic: at n=100, SE≈0.049, so 0.60 is about 2 SE from 0.5).
- Include cross-lingual triplets (Korean A, English B/C and vice versa). These are where embedder choice matters most.

### Gaps
- I did not find a study that measures human inter-annotator agreement on issue-level triplets for social-media comments. C-STS reports agreement for its own annotation, but I did not extract the number.

---

## Q3. Intrinsic metrics (silhouette, Davies–Bouldin, coherence, diversity): limits and evidence of poor correlation with humans

### Takeaway
There is strong 2021–2025 evidence that automated coherence metrics such as NPMI fail to track human judgments. In ProxAnn (ACL 2025), NPMI had near-zero or negative correlation with human topic-fit judgments. LLM-based evaluation correlates better with humans. Silhouette and Davies–Bouldin measure geometric separation *in the embedding being evaluated*, so they cannot fairly compare different embedders or preprocessing variants. Use them only as sanity diagnostics within a fixed representation.

### Cited Findings
- Hoyle et al., "Is Automated Topic Model Evaluation Broken? The Incoherence of Coherence" (arXiv Jul 2021; NeurIPS 2021 spotlight): finds a validation gap, since automatic coherence had never been validated against humans for neural topic models. Using topic rating and word intrusion, "automated evaluation will declare one model significantly different from another when corresponding human evaluations do not." **[verified]** — [arXiv 2107.02173](https://arxiv.org/abs/2107.02173); [NeurIPS 2021](https://neurips.cc/virtual/2021/poster/27300)
- Stammbach, Zouhar, Hoyle, Sachan, Ash, "Revisiting Automated Topic Model Evaluation with Large Language Models" (EMNLP 2023): LLM ratings of topics correlate more strongly with human judgments than existing automated metrics. LLMs do better on coherence *ratings* than on *intrusion detection*. Putting the research question in the prompt helps estimate the optimal number of topics. **[verified]** — [arXiv 2305.12152](https://arxiv.org/abs/2305.12152v2); [ACL Anthology 2023.emnlp-main.581](https://preview.aclanthology.org/setup/2023.emnlp-main.581)
- Hoyle et al., ProxAnn: "Use-Oriented Evaluations of Topic Models and Document Clustering" (arXiv Jul 2025; ACL 2025), all **[verified]**:
  - Covers topic models *and document clustering*.
  - Protocol: annotators (or an LLM proxy) see 7 exemplar documents plus 15 keywords and write a category label (Label step). They then rate 7 held-out documents 1–5 for fit (Fit step) and rank them (Rank step).
  - Scale: 48 topics (Mallet, CTM, BERTopic × Wiki, Bills), at least 4 annotators per topic.
  - **NPMI vs human: Kendall τ = −0.15 / −0.18 (Wiki Fit/Rank), 0.01 / −0.02 (Bills).** That is essentially zero.
  - Classic Mallet LDA matched or beat neural methods.
  — [arXiv 2507.00828](https://arxiv.org/html/2507.00828v1); [ACL 2025](https://preview.aclanthology.org/setup/2025.acl-long.772); [code](https://github.com/ahoho/proxann)
- Hoyle et al., "Are Neural Topic Models Broken?" (Findings EMNLP 2022): examines the reliability and stability of neural vs classical topic models. **[existence verified via search result; findings from memory]** — [arXiv 2210.16162](https://arxiv.org/pdf/2210.16162)
- Doogan & Buntine, "Topic Model or Topic Twaddle? Re-evaluating Semantic Interpretability Measures" (NAACL 2021): also reports that standard coherence measures don't reliably reflect human interpretability. **[from memory, unverified this session]** — [ACL Anthology 2021.naacl-main.300](https://aclanthology.org/2021.naacl-main.300)
- TopicGPT (Pham et al., NAACL 2024) evaluates *against human-annotated categories* with extrinsic metrics rather than coherence: harmonic mean of purity and inverse purity ("P1"), ARI, and NMI. **[verified]** — [arXiv 2311.01449](https://arxiv.org/html/2311.01449v2)

### Inferences
- **Silhouette and Davies–Bouldin (my reasoning; no direct human-correlation study found).** Both are computed in the representation space, so:
  - (a) Embedder A with a more anisotropic or compressed space can score "better" without grouping issues any better.
  - (b) Gist-then-embed compresses paraphrase variation, which inflates silhouette almost by construction.
  - (c) HDBSCAN noise points are usually dropped before computing them, which rewards variants that throw more comments into noise. Always report the noise/"other" rate next to any intrinsic score.
  - (d) Both prefer convex, similar-sized clusters, while issue distributions in comments are heavy-tailed: one or two dominant issues and a long tail.
  Use them only to compare clusterer hyperparameters *within one fixed embedding*, never across embedders or preprocessing variants.
- **Coherence/diversity on keyword lists** (c-TF-IDF in BERTopic) are weak proxies for short code-switched Korean/English comments. Korean tokenization also affects NPMI directly. The ProxAnn-style "infer label → apply to held-out docs" protocol is closer to what the codebook reviewer actually does.
- **Recommendation.** Report intrinsic metrics in an appendix. Choose between variants with human pairwise P/R (Q1) plus an LLM-proxy ProxAnn-style fit score calibrated against the human labels (Q6).

### Gaps
- I found no study that directly correlates silhouette or Davies–Bouldin with human judgments of *text* cluster quality. The above is reasoning, not cited evidence.
- I did not verify the 2024 "Contextualized Topic Coherence" (Rahimi et al., EACL Findings 2024, arXiv 2305.14587) results this session. **[from memory, unverified]**

---

## Q4. Stability across seeds and subsamples: ARI/NMI between runs, bootstrap stability, consensus clustering, reporting

### Takeaway
Stability is necessary but not sufficient. A stable pipeline can be stably wrong, as the stability-overview literature stresses. Stability is still cheap and label-free, and it is especially relevant here because UMAP+HDBSCAN and LLM gist generation are both stochastic. Report: (i) pairwise ARI/NMI across ≥10 seeds, (ii) cluster-wise bootstrap Jaccard (Hennig-style) on 80% subsamples, flagging clusters with mean Jaccard <0.6, and (iii) a co-assignment (consensus) matrix used to find unstable comments, which then feed active labeling (Q5).

### Cited Findings
- von Luxburg, "Clustering Stability: An Overview" (arXiv Jul 2010; Foundations & Trends in ML): reviews protocols for computing stability and their use in model selection (choosing K). It notes that selecting the number of clusters by stability has become popular, and analyzes what stability can and cannot guarantee. **[verified existence/topic]** — [arXiv 1007.1075](https://arxiv.org/pdf/1007.1075)
- Ben-Hur, Elisseeff & Guyon (PSB 2002), "A stability based method for discovering structure in clustered data": cluster random subsamples repeatedly and compare the partitions on their overlap using pairwise similarity measures (e.g., Jaccard or matching coefficient on co-membership). The distribution of similarity values across K indicates the most stable K. **[verified via search summary]** — described in [von Luxburg 2010](https://arxiv.org/pdf/1007.1075) and [Cardiff bootstrap-stability paper](https://orca.cardiff.ac.uk/114553/1/Bootstrapping%20estimates%20of%20stability%20for%20clusters.pdf)
- Monti et al. (2003), "Consensus Clustering: A Resampling-Based Method for Class Discovery and Visualization of Gene Expression Microarray Data" (Machine Learning 52): builds a consensus (co-clustering frequency) matrix over resampled runs to assess stability and choose K. **[verified existence via Springer link]** — [Springer 10.1023/A:1023949509487](https://link.springer.com/article/10.1023/A:1023949509487)
- Hennig (2007), "Cluster-wise assessment of cluster stability" (Comput. Stat. & Data Analysis 52): bootstrap the data, recluster, and for each original cluster record the max Jaccard with any bootstrap cluster. Mean Jaccard ≤0.5 means "dissolved" and ≥0.75 means stable (as I recall it; implemented in R `fpc::clusterboot`). **[from memory, unverified this session]** — [DOI 10.1016/j.csda.2006.11.025](https://doi.org/10.1016/j.csda.2006.11.025)
- "Are Neural Topic Models Broken?" (Hoyle et al., 2022) examines run-to-run variability of topic models as an evaluation problem. **[existence verified; details from memory]** — [arXiv 2210.16162](https://arxiv.org/pdf/2210.16162)

### Inferences
- **Concrete stability protocol (my synthesis):**
  1. **Seed stability:** run each full pipeline variant for S=10 seeds (UMAP seed, HDBSCAN is deterministic given UMAP, k-means init, LLM gist temperature>0 if used). Report the median and IQR of pairwise ARI and NMI over the 45 seed pairs. Compute ARI both over all items and over items non-noise in both runs, and report the noise rate per run.
  2. **Subsample stability:** take 20 bootstrap or 80% subsamples, recluster, and compute per-cluster max-Jaccard against the full-data clustering. Report the fraction of comments that sit in clusters with Jaccard ≥0.75.
  3. **Gist stability (specific to this pipeline):** regenerate gists with 2–3 different seeds or temperatures and re-embed. ARI between gist runs isolates LLM-preprocessing variance from clustering variance.
  4. **Consensus/co-assignment matrix:** over all runs, P(i,j co-clustered). Pairs with P≈0.5 are the "unstable boundary". They are good candidates for human labeling and are useful for showing the reviewer which comments the codebook assignment is uncertain about.
  5. **Reporting:** a table per video (variant × {ARI median [IQR], NMI median, % items in stable clusters, noise %, K median [range]}). State explicitly that stability was used as a *filter*, e.g., drop variants with median ARI<0.5. Ranking among the surviving variants was decided by human-validated pairwise F1.
- **Caveat.** Gist-then-embed will probably look *more* stable than raw, because it reduces lexical variance. That doesn't mean it is more correct, which is why stability can't be the deciding metric.

### Gaps
- I couldn't verify the exact Hennig Jaccard thresholds or the specific findings of Hoyle et al. 2022 this session.
- I found no 2021–2026 paper that benchmarks stability metrics for LLM-gist + embedding + HDBSCAN pipelines specifically.

---

## Q5. Active labeling: which pairs or items to label; active evaluation and stratified/importance-sampling estimators

### Takeaway
Active *selection* must be paired with *correct weighting*, or the estimates become biased. Three families transfer to this problem:
- Importance sampling for F-measure under imbalance (OASIS, up to 83% fewer labels).
- Active statistical inference (Zrnic & Candès, ICML 2024): label where the model or LLM judge is uncertain, then reweight, which gives valid CIs.
- Confidence-driven inference (Gligorić et al., NAACL 2025): uses the LLM's verbalized confidence to choose human labels, with a guarantee of being no worse than human-only.

### Cited Findings
- Zrnic & Candès, "Active Statistical Inference" (arXiv Mar 2024; ICML 2024 oral): prioritize labels where the model is uncertain and rely on predictions where it is confident. It gives provably valid CIs and tests for any black-box model, and narrower CIs / more power than uniform sampling for the same budget. **[verified]** — [arXiv 2403.03208](https://arxiv.org/abs/2403.03208v2); [ICML 2024](https://icml.cc/virtual/2024/oral/35546)
- Follow-up, "Robust Sampling for Active Statistical Inference" (NeurIPS 2025): if the uncertainty estimates are inaccurate, active sampling can be *noisier than uniform sampling*, and the paper proposes robust variants. **[verified via search snippet/abstract]** — [NeurIPS 2025](https://papers.nips.cc/paper_files/paper/2025/hash/6389470564214983604d1ac81631c2c5-Abstract-Conference.html)
- Gligorić et al., "Can Unconfident LLM Annotations Be Used for Confident Conclusions?" (arXiv Aug 2024; NAACL 2025): **Confidence-Driven Inference** picks which items get human labels, guided by LLM annotations and the LLM's verbalized confidence. Results are guaranteed valid and no less accurate than human-only, and it cut human annotations needed by more than 25% on politeness, stance, and bias tasks. **[verified]** — [arXiv 2408.15204](https://arxiv.org/abs/2408.15204); [NAACL 2025](https://aclanthology.org/2025.naacl-long.179)
- OASIS (PVLDB 2017): adaptive importance sampling for P/R/F estimation under extreme imbalance, consistent estimates, up to 83% label reduction. **[verified]** — [arXiv 1703.00617](https://arxiv.org/abs/1703.00617v2)
- ClusterLLM: entropy-based selection of *informative* triplets for training/aligning, not for unbiased estimation. **[verified]** — [arXiv 2305.14871](https://arxiv.org/abs/2305.14871)
- Kossen et al., "Active Testing: Sample-Efficient Model Evaluation" (ICML 2021): actively select test points with an acquisition proposal, then correct the bias with importance weights. **[from memory, unverified this session]** — [arXiv 2103.05331](https://arxiv.org/abs/2103.05331)

### Inferences
- **Two goals, two pools (my synthesis):**
  - **Estimation pool (~60% of budget):** a probability sample of pairs with known π (stratified, Q1), or proposal-weighted sampling (OASIS / active inference) where π ∝ LLM-judge uncertainty × stratum weight, floored to keep variance bounded. Used for unbiased P/R/F and for PPI correction.
  - **Discovery pool (~40%):** non-random, targeted pairs and items for *improving* the codebook: cross-variant disagreements, consensus-matrix P≈0.5 pairs, items near cluster boundaries (2nd-nearest-centroid margin small), HDBSCAN outlier-score extremes, and Korean–English cross-lingual pairs. Do **not** use these in metric estimates unless they are weighted, or they will bias toward hard cases.
- **Selection signals ranked by expected value here:** (1) disagreement between pipeline variants, which is directly decision-relevant; (2) LLM-judge P(yes) near 0.5, from llama.cpp token logprobs for "Yes"/"No", which beats verbalized confidence for local models; (3) consensus instability; (4) cosine near the effective merge threshold.
- **Floor on π.** Per the robust-active-inference finding, mix the uncertainty proposal with uniform sampling, e.g., 0.5·uniform-within-stratum + 0.5·uncertainty, so a miscalibrated judge can't blow up variance.

### Gaps
- I found no paper that applies active statistical inference or PPI specifically to *clustering* evaluation metrics such as pairwise F1 or B³. Adapting them is straightforward, because pairwise TP/FP/FN are means of pair-level indicators, but nobody has published this for this setting.

---

## Q6. LLM-as-judge for same-issue pairs: reliability vs humans, calibration, and bias correction with a small human set (PPI)

### Takeaway
LLM judges vary a lot by task and are not a drop-in replacement (JUDGE-BENCH). Yet for the closest task, inferring a cluster's category and judging document fit, open-weight 32B-class models came close to GPT-4o and were statistically indistinguishable from a human annotator by the Alt-Test (ProxAnn, ACL 2025). An 8B model was clearly worse. Use the local LLM judge on *all* candidate pairs. Use 100–300 human labels to (a) validate it (Alt-Test needs only 50–100 doubly-annotated items) and (b) correct its bias with PPI/PPI++ so pairwise P/R/F estimates come with valid CIs.

### Cited Findings
- Bavaresco et al., "LLMs instead of Human Judges? A Large Scale Empirical Study across 20 NLP Evaluation Tasks" (JUDGE-BENCH; arXiv Jun 2024, ACL 2025): 11 LLMs, open and proprietary, on 20 human-annotated datasets. Every model shows large variance across datasets. Each correlates well on some tasks and poorly on others. LLMs agree more with non-expert than expert annotators. Conclusion: "not yet ready to systematically replace human judges." **[verified]** — [arXiv 2406.18403](https://arxiv.org/abs/2406.18403v1)
- Calderon, Reichart & Dror, "The Alternative Annotator Test for LLM-as-a-Judge: How to Statistically Justify Replacing Human Annotators with LLMs" (arXiv Jan 2025; ACL 2025): a leave-one-annotator-out test that asks whether the LLM represents the remaining humans better than the excluded human does. It needs **50–100 instances annotated by ≥2 humans**. Closed models (GPT-4o) sometimes pass, and open models less often. Code available. **[verified]** — [arXiv 2501.10970](https://arxiv.org/abs/2501.10970v3); [ACL 2025](https://aclanthology.org/2025.acl-long.782); [code](https://github.com/nitaytech/alttest)
- ProxAnn (ACL 2025), all **[verified]**:
  - LLM proxies vs humans, Kendall τ on doc-level fit/rank:
    - GPT-4o: 0.56–0.71
    - **Qwen-3-32B: 0.55–0.68**
    - **Llama-3.1-8B: 0.22 (Wiki Fit), 0.36 (Wiki Rank), i.e., poor**
  - Alt-Test advantage probability ρ: GPT-4o 0.56–0.65 at document level; Qwen-3-32B 0.67 (Bills). Models ≥32B generally had ρ>0.5 with significance.
  - Human–human leave-one-out τ was only about 0.09–0.41.
  - Krippendorff's α among humans: 0.31–0.74 on real topics, ≥0.80 on synthetic ground-truth clusters.
  — [arXiv 2507.00828](https://arxiv.org/html/2507.00828v1)
- Angelopoulos, Bates, Fannjiang, Jordan & Zrnic, "Prediction-Powered Inference" (Science 382, Nov 2023; arXiv Jan 2023): valid CIs for means, quantiles, and regression coefficients by combining many ML predictions with a small labeled set. A "rectifier" (mean of human − prediction on the labeled set) corrects the prediction-based estimate. **[from memory; PPI description consistent with verified search summaries below]** — [arXiv 2301.09633](https://arxiv.org/abs/2301.09633)
- PPI++ (Angelopoulos, Duchi & Zrnic, arXiv Nov 2023) adds a tuned weight λ on the prediction term so it is never worse than human-only. The estimator is the LLM-based estimate minus a bias correction measured on the gold set, and it is unbiased regardless of the judge's error profile. **[mechanism verified via search summary; paper from memory]** — [arXiv 2311.01453](https://arxiv.org/abs/2311.01453); [GLIDE PPI++ tutorial](https://glide-py.readthedocs.io/en/latest/tutorials/ppi/)
- Applied evidence of the gain size: PRECISE (arXiv Jan 2026; AAAI): about 100 human-annotated queries plus about 10,000 unlabeled items. On ESCI, 30 human labels plus Claude 3 Sonnet judgments cut the SE of Precision@4 from 4.45 to 3.50 (−21%). **[verified via search snippet]** — [arXiv 2601.18777](https://arxiv.org/pdf/2601.18777)
- Confidence-Driven Inference (NAACL 2025): more than 25% fewer human labels needed than baselines. **[verified]** — [arXiv 2408.15204](https://arxiv.org/abs/2408.15204)

### Inferences
- **Model size under the 16 GB GPU constraint.** ProxAnn's evidence points to ≥32B models being needed for human-like judgments. 8B was poor. On 16 GB VRAM a 32B model needs about Q3/Q4 quantization with partial CPU offload: slow, but feasible for a few thousand pairs per video at short prompts. A 14B model (e.g., Qwen-class) at Q4–Q5 fits fully. Run the Alt-Test on the 14B before trusting it, and fall back to 32B-offloaded if it fails. This is my extrapolation, since ProxAnn did not test 14B.
- **Judge prompt design (from C-STS and the Halterman & Keith findings).** Condition explicitly: "Do these two comments raise the same *issue/claim* about the video's topic, regardless of stance, tone, or language?" Ask for "Yes/No/Unclear". Take P(Yes) from logprobs. Randomize pair order and query both orders, then average, to remove position bias. Include the video title or context, but no gist, so the judge isn't scoring the gist step itself.
- **PPI protocol for pairwise P/R/F (my synthesis):**
  1. Let U be all pairs in the estimation strata (or a large weighted sample). The LLM judge labels all of them, giving ŷ.
  2. Humans label a random (or known-π) subset L with n=100–300, giving y.
  3. Precision of variant V is the mean of y over pairs co-clustered by V. PPI estimate = mean of ŷ over co-clustered pairs in U + mean of (y − ŷ) over co-clustered pairs in L. Use PPI++ λ-tuning. Recall works the same over true-positive pairs, a ratio estimand handled with PPI for ratios or the delta method, or simply bootstrap.
  4. Because all variants share U and L, compute PPI estimates of *differences* in F1 between variants for paired comparisons. That is the decision-relevant quantity.
- **Calibration check.** On the human-labeled set, plot a reliability diagram of P(Yes) and report ECE. If calibration is poor but ranking is good (high AUC), PPI still works, since PPI needs no calibration. Active-inference sampling does need reasonable uncertainty, though.
- Double-annotate about 50–100 of the human pairs with two humans to get human–human κ/α (ceiling) and to run the Alt-Test.

### Gaps
- I found no published reliability numbers for LLM judges on *same-issue pairs of short social-media comments*, nor on Korean or code-switched text. Treat ProxAnn and JUDGE-BENCH as indirect evidence.
- I didn't verify whether PPI has been applied to clustering pair metrics.

---

## Q7. Codebook-quality evaluation: intercoder reliability, coverage / no-issue rate, LLM-assisted qualitative coding (LACA, LLooM, TopicGPT)

### Takeaway
Evaluate the codebook as a *measurement instrument*. Can independent coders (humans, and the LLM applying the codebook) assign the same code to the same comment? Check Krippendorff's α per code and overall. What share of comments falls to "other / no issue"? Does each code cover something distinct? The LLM-assisted coding literature says to validate LLM codebook application on a human sample before trusting it, and that LLMs often fail to follow codebook exclusion rules zero-shot (Halterman & Keith). The extrinsic metrics TopicGPT and LLooM use (P1 = harmonic mean of purity and inverse purity, ARI, NMI, coverage of ground-truth topics) need a reference labeling, which the team doesn't have. They become feasible on a small double-coded sample.

### Cited Findings
- Chew et al., "LLM-Assisted Content Analysis: Using Large Language Models to Support Deductive Coding" (arXiv Jun 2023): LACA applies a predefined codebook with GPT-3.5. It "can often perform deductive coding at levels of agreement comparable to human coders", but performance varies by code. Recommended workflow: test agreement on a sample first, identify codes where the LLM does poorly, and refine prompts or codes iteratively. **[verified abstract-level; specific κ values not extracted]** — [arXiv 2306.14924](https://arxiv.org/abs/2306.14924)
- Halterman & Keith, "Codebook LLMs: Evaluating LLMs as Measurement Tools for Political Science Concepts" (arXiv Jul 2024; *Political Analysis* 2026), all **[verified]**:
  - Valid measurement requires a "codebook-construct label assumption": the LLM must follow the codebook's definitions *and exclusion criteria*.
  - Restructuring codebooks gave only modest zero-shot gains, and models "still struggle to comply with the constraints of the codebooks".
  - Instruction-tuning Mistral gave 0.76 vs 0.53 micro-F1 over zero-shot.
  — [arXiv 2407.10747](https://arxiv.org/abs/2407.10747v2); [Cambridge PDF](https://www.cambridge.org/core/services/aop-cambridge-core/content/view/7B323A0E47F782F2698A0AE849EA00DE/S104719872510017Xa.pdf/div-class-title-codebook-llms-evaluating-llms-as-measurement-tools-for-political-science-concepts-div.pdf)
- LLooM (Lam et al., "Concept Induction: Analyzing Unstructured Text with High-Level Concepts Using LLooM", CHI 2024), all **[verified]**:
  - Technical evaluation on Wikipedia (n=205 sampled from 14,290; 15 ground-truth topics) and Congressional Bills (n=213 of 32,661; 28 topics).
  - Metric: **coverage = proportion of ground-truth topics covered by generated concepts**. Matching was done by a GPT-3.5 few-shot judge, validated against manual matching on 16 trials (MAE 0.07).
  - Results:
    - Bills: LLooM 0.74, GPT-4 Turbo 0.56, BERTopic 0.29.
    - Wiki: LLooM 0.81, GPT-4 0.83, BERTopic 0.63.
  - Qualitatively, BERTopic topics were vaguer, and GPT-4 sometimes merged several ground-truth topics into one.
  - The technical evaluation reported no κ/α for concept scoring against humans.
  — [arXiv 2404.12259](https://arxiv.org/html/2404.12259v1); [UW IDL](https://idl.uw.edu/papers/lloom-concept-induction)
- TopicGPT (Pham et al., NAACL 2024): evaluates topic *assignment* against human labels using P1 (harmonic mean of purity and inverse purity), ARI, and NMI. Purity rewards clusters dominated by one gold class, and inverse purity rewards keeping each gold class together. It reports harmonic-mean purity 0.74 vs 0.64 for the best baseline on Wikipedia. **[verified]** — [arXiv 2311.01449](https://arxiv.org/html/2311.01449v2); [project page](https://chtmp223.github.io/topicGPT)
- ProxAnn's label→fit protocol is effectively a codebook-entry test: from exemplars, infer the category, then judge whether held-out documents fit. Human Krippendorff α on real topic models was 0.31–0.74, versus ≥0.80 on clean synthetic clusters. The authors read low agreement as "often indicat[ing] poor model outputs rather than misspecified tasks." **[verified]** — [arXiv 2507.00828](https://arxiv.org/html/2507.00828v1)
- Krippendorff's conventional thresholds are α ≥ 0.800 for reliable data and 0.667 ≤ α < 0.800 for tentative conclusions. **[from memory: Krippendorff, Content Analysis, 2004; unverified this session]**

### Inferences
- **Codebook evaluation protocol per video (my synthesis), about 120–200 judgments:**
  1. **Reliability sample:** draw 60–100 comments, stratified by assigned code with oversampling of small codes, plus about 15% from "other/no-issue". Two humans independently code them with the codebook (definition, include/exclude rules, examples) and may choose "other". The local LLM codes the same items with the same codebook text. That is 120–200 human codings, and it doubles as the Alt-Test data.
  2. **Metrics:**
     - Krippendorff's α (nominal), human–human and LLM–human. α handles missing data and more than 2 coders, so prefer it to Cohen's κ, but report κ for the LLM-vs-each-human pairs.
     - Per-code agreement: one-vs-rest α or κ, or per-code F1 vs human consensus. This flags codes whose include/exclude rules are ambiguous.
     - Confusion matrix: which codes get mixed up. Pairs of codes that are often confused are merge candidates, or need sharper exclusion rules.
  3. **Coverage / residual:** report the % of comments assigned "other/no-issue" by the pipeline and by humans, and the human-judged precision of "other" (what share of "other" comments actually express a real issue the codebook misses). A codebook can look reliable just by dumping hard items into "other", so always report α together with the residual rate.
  4. **Extrinsic agreement with the clusters:** on the double-coded sample, compute P1 (harmonic mean of purity and inverse purity), ARI, and NMI between the human consensus codes and the pipeline's cluster/code assignment, as TopicGPT does. With n≈100 these are noisy, so bootstrap the CIs.
  5. **Codebook-entry validity (ProxAnn-style):** for each code, the LLM proxy or a human sees only the definition plus examples and rates the fit (1–5) of several held-out cluster members and a few distractors from neighboring clusters. A good entry separates members from distractors (AUC). This tests whether the include/exclude rules actually discriminate.
  6. **Coverage of ground-truth issues (LLooM-style):** if a reviewer writes a quick independent issue list for a video, measure the % of those issues covered by codebook entries (LLM-matched, with spot-checked matching).
- **Thresholds for an operational go/no-go:** α ≥ 0.67 overall, with no code below about 0.5 per-code agreement, residual "other" ≤ 15–20% and "other" precision high. These are my suggested heuristics, derived from Krippendorff's conventions and ProxAnn's observed human α range. Treat them as starting points, not literature standards.
- **Local-LLM codebook adherence risk.** Halterman & Keith suggest zero-shot open models will under-apply exclusion rules. Mitigate with explicit per-rule checklists in the prompt and few-shot examples drawn from the codebook entry. If a model is fine-tuned, use the human-coded data, keeping held-out validation.

### Gaps
- I couldn't extract the specific κ/α values from Chew et al. (2023). The abstract only says "comparable to human coders".
- I found no CHI/CSCW paper that reports Krippendorff's α for *LLM-generated codebooks* applied to YouTube or social-media comments in Korean. Other qualitative-coding HCI work (e.g., Xiao et al., IUI 2023 companion, "Supporting Qualitative Analysis with LLMs: Combining Codebook with GPT-3 for Deductive Coding") was not verified this session.
- There is no established standard threshold for an acceptable "other/no-issue" rate in issue codebooks. The figures above are heuristics.
