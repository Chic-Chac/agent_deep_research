---
id: N-0003
type: note
task: T-0001
title: "Prompt ensembling for LLM text generation: which variation dimensions help and which only add noise"
status: final
created: 2026-11-06
tags: [prompt-ensemble, text-generation]
---

# Prompt ensembling for LLM text generation: which variation dimensions help and which only add noise

Scope note: this file covers how the literature builds prompt ensembles, which variation dimensions give useful diversity versus noise, the trade-off between diversity and quality, prompt variance compared with sampling variance at temperature 0, and ensembling for short free-form outputs. Target setting: a Qwen3-family 27B instruction-tuned model at temperature 0 with JSON-schema output. It rewrites a short note block into one stance-neutral "issue question" of 4-12 words plus a has_issue flag. The outputs are then embedded and clustered.

Verification status: I read the full text of DIPPER and Jekyll & Hyde, and checked abstract-level claims through search or fetch for USC, GenQREnsemble, SPUQ, the persona studies (Zheng et al., Wharton PSR4) and DIPPER. Numbers marked "(abstract-level, not re-verified this session)" come from well-known abstracts I did not re-fetch. The report writer should treat those as needing a spot-check.

---

## Q1. Self-consistency and its prompt-ensemble variants: what each varied, how it aggregated, and what it gained

### Takeaway
Almost every published "prompt ensemble" was built and tested on tasks with a discrete answer (classification, multiple choice, math). Aggregation is usually majority or weighted vote, sometimes with a learned verifier or weak supervision. The methods that vary something substantive (exemplar sets, reasoning strategy, question format, specialised expert prompts) and filter or weight members by accuracy beat both pure sampling and naive paraphrase averaging. Only USC and MBR-style methods handle free-form outputs, and they work by *selecting* a consensus member, not by averaging.

### Cited Findings
- **Self-consistency (SC)** (Wang et al., ICLR 2023). One prompt; it samples diverse chain-of-thought paths and takes a majority vote over final answers. Gains with PaLM-540B: GSM8K +17.9, SVAMP +11.0, AQuA +12.2, StrategyQA +6.4, ARC-c +3.9 points (abstract-level, not re-verified this session). The paper also compares SC with *prompt-ensemble* baselines (permuting exemplar order, using multiple exemplar sets) and reports SC giving larger gains. I recall this qualitatively and did not re-check the numbers. — [arXiv 2203.11171](https://arxiv.org/abs/2203.11171)
- **Universal Self-Consistency (USC)** (Chen et al., 2023, arXiv preprint). Extends SC to free-form outputs: sample several responses, then prompt the LLM to "select the most consistent" one. Tested on math, code generation, long-context summarization and open-ended QA. It "matches the standard self-consistency performance" on math without needing identical answer formats, and matches execution-based voting on code. The abstract gives no numbers. — [arXiv 2311.17311](https://arxiv.org/abs/2311.17311)
- **DiVeRSe** (Li et al., ACL 2023). Varies the *few-shot exemplar set* (several prompts, each sampled several times) and aggregates with a trained step-aware verifier doing weighted voting. Reported GSM8K improvement with code-davinci-002 from 74.4% to 83.2% (abstract-level, not re-verified this session). — [arXiv 2206.02336](https://arxiv.org/abs/2206.02336)
- **Ask Me Anything (AMA)** (Arora et al., ICLR 2023). Varies the *question format*: an LLM chain recasts each input into open-ended QA prompts ("Who went to the park?") rather than restrictive formats ("John went to the park. True or False?"). It then aggregates the noisy predictions with *weak supervision*, a label model that estimates each prompt's accuracy and the dependencies between prompts, instead of a majority vote. Reported an average 10.2% lift over few-shot for GPT-J-6B, matching or beating few-shot GPT-3-175B on 15 of 20 benchmarks (abstract-level, not re-verified this session). — [arXiv 2210.02441](https://arxiv.org/abs/2210.02441)
- **Boosted Prompt Ensembles** (Pitis et al., 2023, preprint). Builds few-shot prompts one after another from "hard" training examples where the current ensemble is uncertain, in the style of boosting, and weights members. Reported to beat single-prompt output-space ensembles and bagged prompt ensembles on GSM8K and AQuA (qualitative; not re-verified). — [arXiv 2304.05970](https://arxiv.org/abs/2304.05970)
- **Mixture-of-Prompts / "One Prompt is not Enough"** (Wang et al., ICML 2024). Clusters demonstrations into semantic regions by embedding and gives each region its own "expert" prompt (instruction plus demos). At test time it *routes* each input to one expert by embedding proximity, so this is a *partitioning* of the input space, not averaging. Claimed an 81% average win rate over prior methods (abstract-level, not re-verified this session). — [arXiv 2407.00256](https://arxiv.org/abs/2407.00256)
- **PromptBoosting** (Hou et al., ICML 2023). AdaBoost over a small pool of prompts with verbalizers, for black-box *text classification* only. It does not apply to generation. — [arXiv 2212.09257](https://arxiv.org/abs/2212.09257)
- **PRomPTed** (Srivastava et al., "Instances Need More Care", 2023/2024). Not an ensemble. An LLM-in-the-loop *rewrites the prompt for each instance* to improve zero-shot performance. — [arXiv 2310.02107](https://arxiv.org/abs/2310.02107)
- **DIV-SE / IDIV-SE, "Diversity of Thought"** (Naik et al., 2023). The LLM itself is asked to propose distinct *approaches* and *personas* for the task. Those are combined into several prompts (DIV-SE, separate calls) or packed into one prompt (IDIV-SE), with majority vote. Reported better accuracy-cost trade-offs than SC on reasoning and planning benchmarks (qualitative; exact numbers not re-verified). — [arXiv 2310.07088](https://arxiv.org/abs/2310.07088)
- **DIPPER** (Lau et al., EMNLP 2025 main). This is the most direct study of *which prompt sets* to ensemble:
  - GPT-4o generates 200 task-agnostic "reasoning-behaviour" instructions.
  - A subset is chosen greedily to maximise "fidelity-adjusted semantic volume": the log-det of the Gram matrix of prompt embeddings (all-MiniLM-L6-v2) plus α times dev-set accuracy.
  - Aggregation is majority vote, or Best-of-N with a reward model.
  - Results on MATH: Qwen2-MATH-1.5B with 7 fixed prompts scored 57.31% vs 55.76% for a same-prompt self-ensemble. Qwen3-1.7B at n=3 went from 47.38 (self-ensemble) to 51.57 (FASV); at n=9 from 51.70 to 54.51. Qwen3-0.6B at n=9 went from 45.43 to 49.26.
  - The authors argue that self-ensembles "will nonetheless still be sampling from the same distribution" and that SC "typically yield[s] limited diversity".
  - Temperature was 0.7. Temperature 0 was not tested. — [arXiv 2412.15238 (HTML v2)](https://arxiv.org/html/2412.15238v2); [ACL Anthology EMNLP 2025](https://aclanthology.org/anthology-files/pdf/emnlp/2025.emnlp-main.1801.pdf)
- **MoRE, Mixture of Reasoning Experts** (Si et al., Findings EMNLP 2023). Four *specialised* prompts (factual with retrieval, multi-hop CoT, math with program-of-thought, commonsense with generated knowledge). A selector picks an answer, and *agreement between experts* is used as a signal for selective answering or abstention (qualitative; not re-verified). — [arXiv 2305.14628](https://arxiv.org/abs/2305.14628)
- **Multi-Perspective Self-Consistency (MPSC) for code** (Huang et al., ACL 2024). Varies the *output type or perspective*: solutions, specifications and test cases. Aggregation is graph-based, scoring candidates by consistency within and between perspectives (qualitative; exact gains not re-verified). — [arXiv 2309.17272](https://arxiv.org/abs/2309.17272)
- **GenQREnsemble** (Dhole & Agichtein, 2024). Uses *paraphrases of a zero-shot instruction* to generate several keyword sets for query reformulation. The union of keywords is appended to the query. Reported relative gains of up to 18% in nDCG@10 and 24% in MAP over the prior zero-shot state of the art on four IR benchmarks. — [arXiv 2404.03746](https://arxiv.org/abs/2404.03746)
- **Jiang et al. 2020, "How Can We Know What Language Models Know?"** (TACL). Mined and paraphrased cloze prompts, ensembled with learned weights for factual probing. The LAMA accuracy gain was roughly 31.1% to 39.6% (abstract-level, not re-verified). This is early evidence that paraphrase ensembles help *when outputs are aggregated in a discrete answer space*. — [arXiv 1911.12543](https://arxiv.org/abs/1911.12543)
- **Taxonomy.** The Prompt Report lists the ensembling family as DENSE, MoRE, Max Mutual Information, Meta-CoT, Self-Consistency, Universal Self-Consistency, Universal Self-Adaptive Prompting, DiVeRSe, COSP and Prompt Paraphrasing (not re-verified this session). — [Schulhoff et al. 2024, arXiv 2406.06608](https://arxiv.org/abs/2406.06608)
- **DENSE, demonstration ensembling** (Khalifa et al., 2023, ICLR ME-FoMo workshop). Splits demonstrations into buckets, prompts once per bucket and combines the output probabilities. Modest gains over concatenating all demos (qualitative; not re-verified). — [arXiv 2308.08780](https://arxiv.org/abs/2308.08780)

### Inferences
- Across these methods, a varied dimension paid off when it changed the *route* to the same target answer (exemplars, reasoning strategy, QA format, expert decomposition) while the *answer space stayed fixed*. A discrete answer space makes "agreement" well-defined. Our output is a free-form string, so agreement has to be defined through embeddings or an LLM judge (USC/MBR style).
- None of these papers use a 27B instruction-tuned model at temperature 0 on a canonicalisation task. Most results come from PaLM, GPT-3/codex, GPT-J or small Qwen math models on reasoning benchmarks.
- Transfer to our setting: the closest analogues are GenQREnsemble (short generated strings from paraphrased instructions) and USC/MBR (consensus selection over free-form outputs). GenQREnsemble's gain comes from a *union* of keywords used in retrieval, where recall benefits from diversity. Our use case benefits from *convergence* to a canonical string, so its success does not transfer directly.

### Gaps
- I could not confirm a separate published paper titled "Multi-prompt self-consistency" for generation apart from MPSC (code) and DIPPER. "PRomPTed" is a per-instance rewriting method, not an ensemble.
- No paper found ensembles prompts for a *normalisation* task whose output is later clustered.

---

## Q2. Evidence on specific variation dimensions

### Takeaway
Persona and role variation does not reliably improve accuracy on objective tasks, and its per-item effect looks largely random: diversity without a reliable way to select. Abstraction (step-back) and decomposition help *reasoning* when the abstraction is an intermediate step, not when it changes the final output. Format, wording and exemplar-order changes are the best-documented source of *spurious* variance. This variance is large, persists with scale and instruction tuning, is largest for open-ended generation, and is reduced by adding even one exemplar.

### Cited Findings
**Persona, role and perspective**
- Zheng et al. (final title: "When 'A Helpful Assistant' Is Not Really Helpful: Personas in System Prompts Do Not Improve Performances of LLMs", Findings EMNLP 2024). Tested 162 personas (6 relationship types, 8 expertise domains) on 4 open-source model families and 2,410 factual questions. Adding personas did not improve performance compared with no persona. The arXiv v1 had claimed the *opposite*, which is an example of how fragile this finding is. Abstract (v3, as recalled): aggregating the best persona per question would greatly improve accuracy, but automatically identifying that persona is about as good as random, and "the effect of each persona can be largely random". — [arXiv 2311.10054](https://arxiv.org/abs/2311.10054); [ACL Anthology 2024.findings-emnlp.888](https://preview.aclanthology.org/new-sigs/2024.findings-emnlp.888/)
- Wharton Prompting Science Report 4 (Basil, Mollick, Meincke et al., Dec 2025). Six models, GPQA Diamond and MMLU-Pro. Most persona conditions were statistically indistinguishable from baseline, and where effects were significant they were more often negative. Domain-matched expert personas had no significant effect except on Gemini 2.0 Flash. "Toddler" or "layperson" personas hurt accuracy in 4 of 6 models. Mismatched personas made Gemini 2.5 Flash refuse. — [arXiv 2512.05858](https://arxiv.org/abs/2512.05858)
- Kim et al., "Persona is a Double-edged Sword" / Jekyll & Hyde (Findings IJCNLP-AACL 2025).
  - Persona prompts degraded Llama-3-8B on 7 of 12 reasoning datasets.
  - Their ensemble pairs a persona solver with a neutral solver and uses an LLM evaluator run in both orders to cancel position bias. Average gain over the neutral baseline was +9.98 points for GPT-4 (76.22 to 86.20). The gain was about +6.9 for GPT-3.5 and about +2.2 for Llama-3-8B (my computation from the tables).
  - LLM-generated personas gave similar means but much lower variance than hand-crafted ones (AQuA SD 2.08 vs 6.11).
  - The ensemble beat self-consistency at equal call budget on AQuA and object tracking. — [arXiv 2408.08631v2](https://arxiv.org/html/2408.08631v2); [ACL Anthology 2025.findings-ijcnlp.51](https://aclanthology.org/2025.findings-ijcnlp.51)
- Hu & Collier, "Quantifying the Persona Effect in LLM Simulations" (ACL 2024). Persona variables explain little of the variance in human annotations on subjective NLP tasks (less than 10% in their analysis, as recalled; not re-verified). Persona prompting gives modest, mostly consistent shifts. — [arXiv 2402.10811](https://arxiv.org/abs/2402.10811)

**Abstraction level**
- Step-back prompting (Zheng et al., ICLR 2024). The model first answers a more abstract "step-back" question about principles or concepts, then reasons to the specific answer. PaLM-2L gains: MMLU Physics +7%, Chemistry +11%, TimeQA +27%, MuSiQue +7% (abstract-level, not re-verified this session). — [arXiv 2310.06117](https://arxiv.org/abs/2310.06117)

**Question format and framing**
- AMA (above): open-ended QA framing of the same task outperformed restrictive or claim-style framing for GPT-J-6B, and *aggregating across formats* with a weak-supervision label model added further gains. — [arXiv 2210.02441](https://arxiv.org/abs/2210.02441)

**Task decomposition and specialised experts**
- MoRE and MPSC (above): different reasoning types or output perspectives act as distinct experts, and cross-expert agreement is informative. — [arXiv 2305.14628](https://arxiv.org/abs/2305.14628); [arXiv 2309.17272](https://arxiv.org/abs/2309.17272)

**Demonstration-set and order variation**
- Lu et al., "Fantastically Ordered Prompts" (ACL 2022). Exemplar *order* alone can swing GPT-family few-shot classification from near state of the art to near chance. The effect does not disappear with model size. An entropy-based probing set picks good orders, with about 13% relative improvement across 11 tasks (abstract-level, not re-verified). — [arXiv 2104.08786](https://arxiv.org/abs/2104.08786)
- DiVeRSe and DENSE (above): ensembling *different exemplar sets* is one of the dimensions that most reliably gives positive gains. — [arXiv 2206.02336](https://arxiv.org/abs/2206.02336); [arXiv 2308.08780](https://arxiv.org/abs/2308.08780)

**Format, wording and paraphrase sensitivity (the "noise" dimension)**
- Sclar et al., FormatSpread (ICLR 2024). Semantically equivalent formatting changes (separators, casing, spacing) cause up to 76 accuracy points of spread for LLaMA-2-13B few-shot. Sensitivity persists with larger models, more shots and instruction tuning. They recommend reporting a *range* across formats (abstract-level, not re-verified this session). — [arXiv 2310.11324](https://arxiv.org/abs/2310.11324)
- Mizrahi et al., "State of What Art? A Call for Multi-Prompt LLM Evaluation" (TACL 2024). 6.5M instances, 20 LLMs, 39 tasks, using automatically paraphrased instructions. Absolute and relative model rankings change with the instruction paraphrase. They propose multi-prompt metrics (abstract-level, not re-verified). — [arXiv 2401.00595](https://arxiv.org/abs/2401.00595)
- POSIX (Chatterjee et al., Findings EMNLP 2024). A prompt sensitivity index based on log-likelihood shifts across intent-preserving variants. Findings: more parameters or instruction tuning does *not* necessarily reduce sensitivity, while adding even one few-shot exemplar does. Template changes drive sensitivity most for MCQ, and *paraphrasing drives it most for open-ended generation* (recalled from abstract; not re-verified this session). — [arXiv 2410.02185](https://arxiv.org/abs/2410.02185)
- ProSA (Zhuo et al., Findings EMNLP 2024). Sensitivity varies across datasets and models. Larger models tend to be more robust, few-shot examples reduce sensitivity, and higher model confidence goes with robustness (recalled; not re-verified). — [arXiv 2410.12405](https://arxiv.org/abs/2410.12405)
- "The Price of Format: Diversity Collapse in LLMs" (Findings EMNLP 2025). Structural template tokens (chat templates, structured formats) significantly constrain output diversity, and the collapse persists even at high temperature. — [ACL Anthology 2025.findings-emnlp.836](https://aclanthology.org/2025.findings-emnlp.836.pdf)

### Inferences
- The team's measurement fits this literature: surface paraphrases moved outputs by a median 1-cosine of about 0.10 and gave ARI 0.58-0.71. POSIX specifically finds that paraphrasing is the dominant sensitivity driver for *open-ended generation*, and that scale and instruction tuning do not remove it. A 27B model should not be expected to be paraphrase-invariant.
- Persona variation acts like a high-variance, mostly random perturbation on objective tasks (Zheng; PSR4). In our setting a "perspective" persona (an economist, an ethicist, and so on) would likely *inject topical framing into the issue string*, which conflicts with the stance-neutral, canonical target. The "oracle persona per item" headroom shows that personas *do* create diversity, but no reliable selector has been shown. Jekyll & Hyde-style *paired* ensembles (persona plus neutral, adjudicated by a judge) are the only positive persona-ensemble design found, and their gains shrink sharply on the smaller open model (+2.2 points on Llama-3-8B).
- Step-back helps when abstraction is an *intermediate* reasoning step. Transfer to our setting: rather than ensembling outputs at different abstraction levels, which would produce issue strings at different granularity that cluster apart by construction, use abstraction *inside* one prompt. For example: "first state the general policy domain, then write the specific issue question at the required level". Fix the target granularity with explicit spec and exemplars.
- Exemplars are the lever with the best evidence for *reducing* paraphrase variance (POSIX, ProSA) and the best evidence as a *positive* ensemble dimension (DiVeRSe, DENSE). Transfer to our setting: varying few-shot demonstration sets of canonical issue questions is probably a stronger ensemble axis than wording, and also likely lowers each member's variance. This is inferred, not tested on this task.

### Gaps
- No study found that measures persona or perspective prompting on *normalisation or rewriting* outputs, as opposed to accuracy.
- No direct evidence found on "question vs claim framing" of the *output* (e.g. "whether X" vs "X should..."). AMA concerns input/prompt framing for classification.
- Instruction *specificity* as an ensemble axis has no dedicated study I could locate. It appears only indirectly: Mizrahi paraphrases, and The Prompt Report's general advice.
- Most sensitivity numbers are for classification or MCQ with base or older models (LLaMA-2, GPT-3). Qwen3-class instruction models may be less sensitive in accuracy terms, but no study measures *embedding-level output drift* for them.

---

## Q3. Diversity vs quality trade-offs: when member diversity helps, when it only adds noise, and how to select a subset

### Takeaway
Classical ensemble theory and DIPPER agree: diversity helps only among members that are each accurate and *target the same quantity*. Selection should combine a fidelity (accuracy) term with a diversity term. Diversity alone admits poor or irrelevant prompts. The pattern across Q1 is that learned or weighted aggregation (weak supervision, verifiers, accuracy weighting, LLM judges) beats unweighted voting once members differ in quality.

### Cited Findings
- **Ambiguity decomposition** (Krogh & Vedelsby, NeurIPS 1994/1995). For a weighted-average regression ensemble, ensemble error = average member error − average "ambiguity" (spread around the ensemble mean). Diversity lowers error only if it does not raise average member error by more than it adds ambiguity. — [NeurIPS proceedings](https://papers.nips.cc/paper/1994/hash/b8c37e33defde51cf91e1e03e51657da-Abstract.html)
- **Kuncheva & Whitaker** (Machine Learning, 2003). Across ten diversity measures, the link between measured diversity and ensemble accuracy was weak and unclear in practice (recalled; not re-verified). — [Springer](https://link.springer.com/article/10.1023/A:1022859003006)
- **DIPPER's prompt-subset selection.**
  - Diversity-only selection (α=0) "may select a set of diverse prompts, but potentially some irrelevant or poor performing prompts".
  - The correlation between the selection metric and ensemble accuracy *rises and then falls* as α (the fidelity weight) grows, so both terms are needed.
  - More *unique* prompts generally raised accuracy and reduced variance compared with repeating prompts.
  - "Random+" (sampling prompts in proportion to predicted accuracy) beat the self-ensemble but lost to FASV. — [arXiv 2412.15238 (HTML v2)](https://arxiv.org/html/2412.15238v2)
- **Prompt selection without labels.** Max Mutual Information (Sorensen et al., ACL 2022) picks the template with the highest mutual information between input and output, without labels (recalled; not re-verified). Lu et al. use entropy-based probing for order selection. — [arXiv 2203.11364](https://arxiv.org/abs/2203.11364); [arXiv 2104.08786](https://arxiv.org/abs/2104.08786)
- **Weak-supervision aggregation** (AMA). Explicitly models that prompt outputs are *correlated* and of unequal accuracy, so the gain over majority vote comes from weighting and handling dependencies. — [arXiv 2210.02441](https://arxiv.org/abs/2210.02441)
- **Persona search** (Zheng et al.). Large oracle headroom, but learned or heuristic selection is close to random. Diversity exists, but nothing exploits it. — [arXiv 2311.10054](https://arxiv.org/abs/2311.10054)
- **Free-form consensus aggregation.**
  - MBR decoding (Suzgun, Melas-Kyriazi & Jurafsky, Findings ACL 2023) picks the candidate with the highest expected similarity to the rest (e.g. BERTScore) and beats greedy or beam decoding across several generation tasks (recalled; not re-verified). — [arXiv 2211.07634](https://arxiv.org/abs/2211.07634)
  - Bertsch et al. frame self-consistency and similar methods as instances of MBR. — [arXiv 2310.01387](https://arxiv.org/abs/2310.01387)
  - USC uses an LLM judge as the consistency function. — [arXiv 2311.17311](https://arxiv.org/abs/2311.17311)

### Inferences
- In our setting each prompt member must estimate the *same* canonical issue question. Members that deliberately change perspective or abstraction change the *estimand*. That gives "diversity" in Krogh-Vedelsby terms but also raises each member's error against the canonical target, so it will likely hurt.
- "Meaningful" diversity for this task should be diversity of *route* with the target fixed: different exemplar sets, a decompose-then-write chain (extract the contested claim first, then neutralise it into "whether..."), or a direct rewrite. It should not be diversity of *perspective on the content*.
- Averaging *embeddings* of members (the team's current aggregation) is not what the free-form ensemble literature does. MBR, USC and medoid selection pick an actual member near the consensus, and AMA or verifier-based methods weight members by estimated reliability. An embedding mean of paraphrase-noisy outputs gives a centroid that need not correspond to any well-formed issue string. It also cannot down-weight a systematically off member. A medoid or MBR choice using the clustering's own embedding model, plus weighting members by agreement with the others, is the literature-aligned alternative. This is inferred and should be tested.
- A practical subset-selection recipe adapted from DIPPER: generate many candidate prompts, score each on a small labelled set of synthetic note blocks (fidelity), embed the prompts (or their outputs) for diversity, and select greedily with a fidelity-weighted log-det objective. Untested on generation tasks.

### Gaps
- No study found measures "semantic diversity of ensemble members" for *generated strings* that are then clustered. DIPPER measures diversity of the *prompts*, not of the outputs.
- Kuncheva-style evidence is for classifiers. I found no equivalent analysis for LLM prompt ensembles on generation.

---

## Q4. Temperature 0: prompt variance vs sampling variance, and whether paraphrase variance is noise or signal

### Takeaway
At temperature 0, prompt variation is the main controllable source of output variance. Several works argue that prompt variation gives *more* diversity than sampling: DIPPER, and the format-collapse paper, which shows templates constrain diversity even at high temperature. The literature splits on how to treat paraphrase variance. Evaluation work (Sclar, Mizrahi, POSIX) treats it as *spurious noise* to average over or report as a range. Uncertainty-quantification work (SPUQ) treats disagreement under perturbation as a *signal* of epistemic uncertainty.

### Cited Findings
- DIPPER: same-prompt sampling ("self-ensemble") "will nonetheless still be sampling from the same distribution"; prompt diversity added gains beyond it. Tested at temperature 0.7 only. — [arXiv 2412.15238](https://arxiv.org/html/2412.15238v2)
- "The Price of Format: Diversity Collapse in LLMs" (Findings EMNLP 2025): structured templates collapse diversity even under high-temperature sampling, so prompt and template structure bounds what temperature can recover. — [ACL Anthology](https://aclanthology.org/2025.findings-emnlp.836.pdf)
- SPUQ (Gao et al., EACL 2024).
  - Perturbs the prompt (paraphrase, random dummy tokens, altered system message, temperature) to capture epistemic uncertainty, and samples to capture aleatoric uncertainty.
  - Aggregates agreement across perturbed outputs, e.g. by ROUGE-L, with up to 5 variants.
  - Reports roughly 50% average reduction in expected calibration error.
  - This is direct precedent for using *disagreement across paraphrased prompts as a confidence signal* for free-form outputs. — [ACL Anthology 2024.eacl-long.143](https://aclanthology.org/2024.eacl-long.143); [GitHub](https://github.com/intuit-ai-research/SPUQ)
- Sensitivity as noise: FormatSpread and multi-prompt evaluation recommend reporting distributions across prompt variants rather than one prompt. — [arXiv 2310.11324](https://arxiv.org/abs/2310.11324); [arXiv 2401.00595](https://arxiv.org/abs/2401.00595)
- Temperature 0 is not fully deterministic. Atil et al. report run-to-run accuracy variation at "deterministic" settings across several APIs (up to about 15% in some cases, recalled; not re-verified) — [arXiv 2408.04667](https://arxiv.org/abs/2408.04667). Thinking Machines Lab attributes inference nondeterminism mainly to batch-size-dependent (non-batch-invariant) kernels, not just floating-point concurrency (Sept 2025; recalled, not re-verified) — [blog](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/).
- Temperature has limited effect on accuracy. Renze & Guven found no statistically significant change in problem-solving accuracy across temperatures 0.0-1.0 for several models on MCQA (recalled; not re-verified). — [arXiv 2402.05201](https://arxiv.org/abs/2402.05201)
- Preprint (status unverified): "Stochastic Sampling is Epistemically Shallow" compares one model at temperature 1 with 24 models at temperature 0. It reports that within one model at most one variation dimension rises above noise, while cross-model ensembles show more structure. This concerns model diversity, not prompt diversity. — [arXiv 2607.20464](https://arxiv.org/abs/2607.20464)

### Inferences
- In our setting, the four-paraphrase disagreement at temperature 0 should be treated *both* ways:
  - as noise to reduce for the clustering output, through MBR/medoid aggregation and stronger exemplars;
  - as a *per-item confidence signal*, SPUQ-style. Items whose paraphrase outputs diverge (high 1-cosine) are likely ambiguous note blocks: multi-issue, vague, or borderline has_issue. They could be flagged, split, or routed to a second pass rather than averaged.
- A cheap control experiment would separate the variance sources: rerun the same prompt several times at temperature 0 with fixed batch settings to measure the inference-nondeterminism floor, before attributing all drift to wording.

### Gaps
- I found no study that directly compares, at temperature 0, prompt-paraphrase variance against sampling variance for *short generated strings*. DIPPER used temperature 0.7, and sensitivity studies measure accuracy or log-likelihood, not embedding drift.

---

## Q5. Ensembling for short, free-form normalised outputs (paraphrase, query rewriting, headlines, label generation)

### Takeaway
Evidence is thin. Query-rewriting ensembles (GenQREnsemble, multi-query RAG) work because retrieval rewards a *union* of diverse terms. For outputs that should converge to one canonical string, the only well-founded aggregation tools are MBR, USC and medoid selection. LLM-based clustering pipelines handle prompt instability mostly by *label consolidation* (generate, merge, then assign) rather than prompt ensembles.

### Cited Findings
- GenQREnsemble: paraphrased-instruction ensemble for keyword generation, up to 18% relative nDCG@10 gain. Motivated by single reformulations sometimes *hurting* compared with the original query. — [arXiv 2404.03746](https://arxiv.org/abs/2404.03746)
- MBR decoding for summarization, data-to-text and similar tasks (Suzgun et al., Findings ACL 2023). Consensus selection among candidates beats greedy (recalled; not re-verified). — [arXiv 2211.07634](https://arxiv.org/abs/2211.07634)
- USC covers summarization and open QA through LLM-judged consistency. — [arXiv 2311.17311](https://arxiv.org/abs/2311.17311)
- SPUQ uses ROUGE-L agreement across perturbed prompts for free-form outputs, as a calibration signal. — [ACL Anthology](https://aclanthology.org/2024.eacl-long.143)
- "Text Clustering as Classification with LLMs" (Huang & He, 2024): generate candidate labels in mini-batches, *merge* semantically redundant labels, then assign each item to one label from the induced inventory. The authors flag the stability and consistency of responses to different prompts as an open challenge. — [arXiv 2410.00927](https://arxiv.org/pdf/2410.00927)
- LLM-based few-shot clustering (Viswanathan et al., TACL 2024) uses LLM keyphrase expansion of inputs before embedding (recalled; not re-verified). — [arXiv 2307.00524](https://arxiv.org/abs/2307.00524)

### Inferences
- Transfer to our setting: two literature-supported architectures fit "convergent short string, then clustering" better than averaging paraphrase-prompt embeddings.
  - (a) Generate K candidates from route-diverse prompts (different exemplar sets, a decompose-then-write chain), then choose the MBR/medoid candidate. Optionally let a USC-style judge pick "the most consistent / most canonical" issue question.
  - (b) Generate first, then canonicalise in a second pass against an induced inventory of issue questions (the generate-merge-assign pattern), so equivalent paraphrases collapse to one label before clustering.
- has_issue is a discrete output, so ordinary majority or weighted vote across members applies directly. This is the part of the task where the classic SC and AMA evidence transfers most cleanly.

### Gaps
- I found no published prompt-ensemble study on headline generation or label generation that reports stability metrics (ARI, embedding dispersion). This is a genuine gap, and the team's experiment would itself be novel evidence.
- Several clustering-pipeline details above come from abstracts or search snippets. The 2025 ACL/EMNLP topic-clustering papers turned up by search ([ACL 2025.acl-long.902](https://aclanthology.org/2025.acl-long.902.pdf), [EMNLP 2025.emnlp-main.613](https://aclanthology.org/2025.emnlp-main.613.pdf)) were not read in full, and I could not confirm whether they test prompt variation.
