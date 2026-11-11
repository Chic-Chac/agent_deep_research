---
id: R-0004
type: report
task: T-0002
title: "Score clusters with judged pairs, not geometry"
status: final
created: 2026-11-02
tags: [evaluation, llm-judge, clustering]
uses: [N-0011]
---

# Score clusters with judged pairs, not geometry

*Part 3 of 3: evaluation. Research notes dated 2026-11-02. PyPI versions checked 2026-11-02.*

## 1. Shortlist (ranked)

1. **Pairwise "same issue?" judgments on a stratified pair sample. The local LLM judges every pair, humans judge a subset, and PPI++ corrects the judge's bias.** A pair can be judged without a codebook, so one labeled pair set scores every variant (raw vs gist, embedder, clusterer) as a paired comparison. PPI++ turns cheap judge labels plus 100–200 human labels into pairwise precision, recall and F1 with valid confidence intervals ([Angelopoulos et al. 2023](https://arxiv.org/abs/2301.09633); [Binette et al. 2024](https://arxiv.org/abs/2404.05622)).
2. **Treat the codebook as a measurement instrument.** Two humans and the LLM independently code 50–60 comments. Report Krippendorff's α, per-code confusions, and the "other / no-issue" residual rate. The same doubly annotated items feed the Alternative Annotator Test (Alt-Test), which decides whether the local judge may stand in for humans ([Calderon et al. 2025](https://arxiv.org/abs/2501.10970v3); [Halterman & Keith 2024](https://arxiv.org/abs/2407.10747v2)).
3. **Use stability across seeds and subsamples as a gate, never as the ranking.** Seed-to-seed ARI and bootstrap Jaccard per cluster cost no labels. They remove fragile variants, and the consensus matrix supplies the boundary pairs for active labeling. A stable pipeline can still be consistently wrong, so human-validated pairwise F1 decides among the variants that survive ([von Luxburg 2010](https://arxiv.org/pdf/1007.1075)).

Intrinsic metrics (silhouette, Davies–Bouldin, NPMI coherence) do not appear on the shortlist. They are diagnostics only (§4.3).

## 2. Fit to constraints

| Constraint | How the protocol meets it |
|---|---|
| 16 GB GPU, llama.cpp | The judge is the team's existing local GGUF model served by llama.cpp. Embedding and judging run one after the other, never at the same time on the shared card. Each call is a short prompt that returns a single Yes/No/Unclear token, and P(Yes) is read from token logprobs. **Unverified:** the exact logprob options of the pinned llama.cpp server build, and throughput on this card. |
| Judge size | In ProxAnn, **Qwen-3-32B reached Kendall τ 0.55–0.68 against humans, close to GPT-4o's 0.56–0.71, while Llama-3.1-8B reached only 0.22–0.36** ([ProxAnn, ACL 2025](https://arxiv.org/html/2507.00828v1)). The team's 27B model sits between the two tested sizes, so it must pass the Alt-Test on their data before anyone trusts it. 14B and 27B were not tested in ProxAnn. |
| Python 3.12 + uv | Pin these with `uv add` and commit `uv.lock`: `scikit-learn==1.9.1` (ARI and NMI), `krippendorff==0.9.0`, `ppi-python==0.2.3`, `ER-Evaluation==2.3.0` (optional, for B³), and `llama-cpp-python==0.3.36` if you do not call the server binary directly. Pin the llama.cpp build tag and the GGUF file's SHA-256, and pin Alt-Test ([GitHub](https://github.com/nitaytech/alttest)) to a commit. **Unverified:** `ppi-python` 0.2.3 (Dec 2024) declares no Python requirement and has not been tested on 3.12. As a fallback, the PPI++ mean estimator is about 15 lines of numpy. |
| Korean + English | The judge prompt asks for the *issue* "regardless of language, stance, tone or target person". Agreement is reported separately for ko–ko, en–en and ko–en pairs. **No published judge-reliability numbers exist for Korean or code-switched comments**, so the Korean cells are the main open risk. |
| No external services | Everything above runs offline. Human annotation uses a local tool (a notebook or a CSV), not a hosted labeling service. |

## 3. Smallest confirming experiment

**Scope.** Use two videos: hQR (English, with prior EXP-005 labels) and one Korean-majority video. Compare two variants on each, for example raw text + Qwen3-Embedding-4B against gist + Qwen3-Embedding-4B, with both clusterings at matched K. The human budget is **300 judgments per video**.

**Pair pool U (LLM only).**
1. Embed all comments with a *reference* embedder that is not one of the variants being compared, so the sample does not favor either one.
2. Bin all pairs into 6 cosine-quantile strata, splitting the top 1% further.
3. Cross the strata with co-clustering status: same cluster in both variants, in one only (the disagreement stratum), or in neither.
4. Draw **3,000 pairs** and record each pair's inclusion probability π. Deliberately include a "same target person, different issue" cell. EXP-006 found people and content co-occurring in about 25% of content comments, and EXP-005 showed embedding clusters grouping by target and polarity rather than by issue.
5. The judge labels all 3,000 pairs in both orders (6,000 calls) to cancel position bias.

**Human labels.**
- **150 pairs** drawn from U with known π. Allocate 40% to disagreement strata, 40% to high-cosine strata and 20% to low and mid-cosine strata, with π mixed 0.5 uniform + 0.5 judge uncertainty (|P(Yes) − 0.5| small). The split and the mix are the notes' **heuristics**.
- **50 of those 150 pairs** get a second annotator (50 judgments).
- **50 comments** are coded against the draft codebook by two humans (100 judgments).
- Total: 300 judgments per video.

**Metrics and confidence intervals.**
- **Primary metric:** ΔF1, the difference in pairwise F1 between the two variants. Estimate it with PPI++ over Horvitz–Thompson-weighted pair counts, and get a **95% CI from a stratified bootstrap (2,000 resamples within strata)**.
- **Secondary metrics:**
  - Judge versus human: Cohen's κ and AUC of P(Yes), reported overall and per language cell.
  - Alt-Test advantage probability ρ on the 50 doubly annotated pairs.
  - Krippendorff α, human–human and LLM–human, on the 50 coded comments.

**Confirmation criteria.** The protocol counts as confirmed on a video if all three hold:
- (a) The judge passes the Alt-Test (ρ > 0.5, significant), or its κ with each human is at least the human–human κ.
- (b) The PPI++ CI on ΔF1 is narrower than the human-only CI from the same 150 pairs. As a reference point, PRECISE reported a **21% smaller standard error** from adding LLM judgments to 30 human labels ([PRECISE 2026](https://arxiv.org/pdf/2601.18777)).
- (c) The ko–ko and ko–en cells do not fall more than about 0.1 κ below en–en. This threshold is a **heuristic**.

**Expected resolution (the notes' arithmetic).** At n = 150 and a true proportion of 0.8, the binomial SE is about 0.033, giving a 95% CI of about ±0.065. Human labels alone can therefore separate variants whose F1 differs by roughly 0.08–0.10, but not by 0.02. Pairing the variants on the same pairs, and PPI, are the levers that tighten this.

## 4. Details on demand

### 4.1 Pairwise judgments versus cluster labels

Full cluster labels need an agreed label set. Here the label set is the codebook, which is the thing under evaluation, so using it as ground truth is circular. Pairwise judgments avoid that circularity.

The entity-resolution literature supplies the statistics. **Pairwise precision and recall can be estimated from a weighted sample with known inclusion probabilities**, and B³ precision and recall come from a sample of fully resolved clusters ([Binette et al. 2022](https://arxiv.org/pdf/2210.01230); [er-evaluation docs](https://er-evaluation.readthedocs.io/en/latest/er_evaluation.estimators.html)).

Uniform pair sampling fails because same-issue pairs are rare. With K = 10–30 issues per video, only about **3–10% of random pairs are positives** (the notes' arithmetic), so a uniform sample of 200 yields too few positives to estimate recall. Stratifying by cosine and by variant disagreement fixes this. OASIS-style adaptive importance sampling cut labels by **up to 83%** under this kind of imbalance ([Marchant & Rubinstein 2017](https://arxiv.org/abs/1703.00617v2)).

Precision is the best-resolved quantity. Recall rests on the thin low-cosine stratum and will have the widest CI, so say so when reporting it.

Granularity confounds pairwise metrics: precision rises and recall falls as K grows. Compare variants at matched K, or across the precision–recall curve over cut levels. ClusterLLM shows that the same pairwise oracle labels can pick the cut level ([Zhang et al. 2023](https://arxiv.org/abs/2305.14871)).

For B³, use the anchor approach. Take 10–20 anchors per video, and have a human mark which of each anchor's top 30–50 neighbors share its issue. This mirrors entity-centric sampling ([Binette et al. 2024](https://arxiv.org/abs/2404.05622)).

Triplets ("is B or C closer in issue to A?") suit the job of ranking embedders. They do not produce partition precision and recall.

### 4.2 Stability reporting

These are the notes' synthesized protocol. Thresholds marked † are **heuristics**.
- **Seeds:** run 10 seeds per variant and report the median and IQR of ARI and NMI over the 45 seed pairs. Compute ARI on all items and again on items that are non-noise in both runs, and report the noise rate and K range alongside.
- **Subsamples:** take 20 subsamples of 80% and compute each cluster's best Jaccard match against the full-data clustering. Flag clusters whose mean Jaccard is below 0.6†.
- **Gists:** regenerate gists 2–3 times to separate variance from the LLM step from variance in clustering.
- **Gate:** drop variants whose median ARI is below 0.5†.

Hennig's cut-offs (≤ 0.5 means the cluster "dissolved", ≥ 0.75 means it is stable) are **unverified**, recalled from memory ([Hennig 2007](https://doi.org/10.1016/j.csda.2006.11.025)).

Expect gist-then-embed to look *more* stable than raw text simply because it removes lexical variance. That is one more reason stability cannot decide the ranking. UMAP is also seed-sensitive at n < 1,000 ([UMAP docs](https://umap-learn.readthedocs.io/en/latest/clustering.html)).

### 4.3 Why intrinsic metrics only diagnose

In ProxAnn, **NPMI correlated with human fit judgments at Kendall τ between −0.18 and 0.01**, which is effectively zero ([ProxAnn](https://arxiv.org/html/2507.00828v1)). Earlier work had already found that automated coherence declares models different when human evaluations do not ([Hoyle et al. 2021](https://arxiv.org/abs/2107.02173)).

Silhouette and Davies–Bouldin are computed inside the space being judged. Gisting compresses paraphrases and inflates both scores almost by construction. HDBSCAN noise is usually dropped before scoring, which rewards variants that discard more comments. No study correlating these two metrics with human judgments of text clusters was found, so this argument is reasoning, not evidence.

Use them only to tune hyperparameters within one fixed embedding, and always report the noise rate next to them. Keyword-based coherence is weaker still for Korean, because the tokenizer choice shifts NPMI directly.

### 4.4 Active labeling without bias

Keep two pools.

**Estimation pool (about 60% of the budget).** Every pair has a known π, and π is floored by mixing in uniform sampling. Active statistical inference gives valid CIs with uncertainty-driven sampling ([Zrnic & Candès 2024](https://arxiv.org/abs/2403.03208v2)). However, when uncertainty estimates are poor, active sampling becomes **noisier than uniform sampling** ([NeurIPS 2025](https://papers.nips.cc/paper_files/paper/2025/hash/6389470564214983604d1ac81631c2c5-Abstract-Conference.html)). Confidence-Driven Inference cut the human labels needed by **more than 25%** on stance-like tasks ([Gligorić et al. 2025](https://aclanthology.org/2025.naacl-long.179)).

**Discovery pool (about 40% of the budget).** These items improve the codebook but never enter the metrics: variant disagreements, consensus pairs with P ≈ 0.5, HDBSCAN outliers, and ko–en pairs. The 60/40 split is a **heuristic**.

No published work applies PPI or active inference to clustering metrics specifically. The adaptation works because pairwise TP, FP and FN are means of pair-level indicators, but nobody has tested it.

### 4.5 Codebook checks

Report Krippendorff α overall and per code, together with the confusion matrix. Codes that are often confused with each other are merge candidates, or need sharper exclusion rules.

Always pair α with the residual "other" rate and with the precision of "other". A codebook can buy high agreement by dumping hard comments into "other".

Halterman & Keith found that open models "still struggle to comply with" codebook exclusion rules zero-shot. Instruction tuning raised micro-F1 from 0.53 to 0.76 ([Halterman & Keith 2024](https://arxiv.org/abs/2407.10747v2)). Write each include/exclude rule into the judge prompt as an explicit checklist item.

A ProxAnn-style entry test checks whether each codebook entry discriminates. Show only the definition and its examples, rate held-out cluster members against distractors from neighboring clusters, and compute the AUC.

The go/no-go thresholds below are **heuristics** drawn from Krippendorff's conventions (**unverified**, from memory) and from ProxAnn's observed human α of 0.31–0.74:
- overall α ≥ 0.67
- no code below about 0.5
- "other" ≤ 15–20%

## 5. Sources

| Source | Date | Status |
|---|---|---|
| [Binette et al., PatentsView ER evaluation](https://arxiv.org/pdf/2210.01230) | Oct 2022 | verified |
| [Binette et al., entity-centric ER evaluation](https://arxiv.org/abs/2404.05622) | Apr 2024 | verified |
| [er-evaluation estimators docs](https://er-evaluation.readthedocs.io/en/latest/er_evaluation.estimators.html) | v2.3.0, Nov 2023 | verified. Whether it accepts pair samples is unverified |
| [OASIS, Marchant & Rubinstein](https://arxiv.org/abs/1703.00617v2) | PVLDB 2017 | verified |
| [ClusterLLM, Zhang et al.](https://arxiv.org/abs/2305.14871) | EMNLP 2023 | verified |
| [C-STS, Deshpande et al.](https://aclanthology.org/2023.emnlp-main.345) | EMNLP 2023 | verified |
| [Hoyle et al., Incoherence of Coherence](https://arxiv.org/abs/2107.02173) | NeurIPS 2021 | verified |
| [Stammbach et al., LLM topic evaluation](https://arxiv.org/abs/2305.12152v2) | EMNLP 2023 | verified |
| [ProxAnn, Hoyle et al.](https://arxiv.org/html/2507.00828v1) | ACL 2025 (Jul 2025) | verified |
| [JUDGE-BENCH, Bavaresco et al.](https://arxiv.org/abs/2406.18403v1) | ACL 2025 | verified |
| [Alt-Test, Calderon et al.](https://arxiv.org/abs/2501.10970v3) | ACL 2025 | verified |
| [Prediction-Powered Inference](https://arxiv.org/abs/2301.09633) / [PPI++](https://arxiv.org/abs/2311.01453) | Science, Nov 2023 / Nov 2023 | mechanism consistent with search summaries; paper details recalled from memory |
| [PRECISE](https://arxiv.org/pdf/2601.18777) | Jan 2026 | verified via search snippet only |
| [Active Statistical Inference, Zrnic & Candès](https://arxiv.org/abs/2403.03208v2) | ICML 2024 | verified |
| [Robust active statistical inference](https://papers.nips.cc/paper_files/paper/2025/hash/6389470564214983604d1ac81631c2c5-Abstract-Conference.html) | NeurIPS 2025 | verified via abstract |
| [Confidence-Driven Inference, Gligorić et al.](https://aclanthology.org/2025.naacl-long.179) | NAACL 2025 | verified |
| [Halterman & Keith, Codebook LLMs](https://arxiv.org/abs/2407.10747v2) | Jul 2024; Political Analysis 2026 | verified |
| [TopicGPT, Pham et al.](https://arxiv.org/html/2311.01449v2) | NAACL 2024 | verified |
| [LLooM, Lam et al.](https://arxiv.org/html/2404.12259v1) | CHI 2024 | verified |
| [von Luxburg, Clustering Stability](https://arxiv.org/pdf/1007.1075) | Jul 2010 | verified (topic) |
| [Hennig, cluster-wise stability](https://doi.org/10.1016/j.csda.2006.11.025) | 2007 | **unverified** (thresholds recalled from memory) |
| [Amigó et al., extrinsic metric constraints (B³)](https://doi.org/10.1007/s10791-008-9066-8) | 2009 | **unverified** |
| [UMAP for clustering docs](https://umap-learn.readthedocs.io/en/latest/clustering.html) | accessed Nov 2026 | verified |
| PyPI: [scikit-learn 1.9.1](https://pypi.org/project/scikit-learn/1.9.1/), [krippendorff 0.9.0](https://pypi.org/project/krippendorff/0.9.0/), [ppi-python 0.2.3](https://pypi.org/project/ppi-python/0.2.3/), [ER-Evaluation 2.3.0](https://pypi.org/project/ER-Evaluation/2.3.0/), [llama-cpp-python 0.3.36](https://pypi.org/project/llama-cpp-python/0.3.36/) | released 2026-09-10, 2026-10-07, 2024-12-22, 2023-11-30, 2026-10-01 | versions checked 2026-11-02. Compatibility with Python 3.12 is untested for ppi-python |
| Requester context: EXP-005 and EXP-006 internal results | 2026-11-02 | internal, English-only |

**Labeled heuristics (not literature standards):**
- the 40/40/20 pair allocation
- the 0.5 uniform floor on π
- the 60/40 split between estimation and discovery pools
- Jaccard < 0.6 flag and ARI < 0.5 gate
- α ≥ 0.67, per-code ≥ 0.5, "other" ≤ 15–20%
- the ko-vs-en κ gap of 0.1
- the 3,000-pair judge pool size

Throughput of the local judge has not been measured.
