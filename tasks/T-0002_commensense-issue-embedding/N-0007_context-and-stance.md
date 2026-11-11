---
id: N-0007
type: note
task: T-0002
title: "Context injection for short-comment embeddings, and issue-vs-stance (and issue-vs-target-person) separation in clustering"
status: final
created: 2026-11-02
tags: [context-injection, stance-separation, embeddings]
---

# Context injection for short-comment embeddings, and issue-vs-stance (and issue-vs-target-person) separation in clustering

Scope note: the pipeline is LLM open coding, then a short "issue" gist per comment, then embedding, clustering and a codebook. It runs on YouTube interview/debate comments, about half Korean, on a 16 GB local GPU. Research budget was about 15 tool calls. Items marked **[from memory, not re-fetched this session]** are well-known papers whose identity and URL I am confident of, but whose specific numbers I did not re-verify in this session. Items marked **[UNVERIFIED]** came only from search snippets.

## Q1. Context injection: what each option does and what evidence covers the trade-offs

### Takeaway
All published context-injection methods were built and tested for *retrieval of document chunks*: Anthropic contextual retrieval, Jina late chunking and CDE. None was evaluated on clustering short social-media replies, so any transfer to this task is an inference. The approach with the best fit to the evidence is the decontextualization route. Use context only in the LLM gist step and produce a stand-alone issue gist, then embed only the gist. The proposition and decontextualization literature supports this, and it avoids the "context dominates the vector" failure by construction. Use vector mixing or concatenation only as a gated fallback.

### Cited Findings
**Anthropic "Contextual Retrieval" (blog, 19 Sep 2024)** [from memory, not re-fetched this session]
- Method: an LLM writes a short context of about 50–100 tokens that situates each chunk within its document. The context is prepended to the chunk before both embedding and BM25 indexing — [Anthropic](https://www.anthropic.com/news/contextual-retrieval)
- Reported results:
  - Contextual embeddings alone cut top-20 retrieval failure by 35% (5.7% → 3.7%).
  - Adding contextual BM25 cut failure by 49% (→ 2.9%).
  - Adding a reranker cut failure by 67% (→ 1.9%).
  - Source: [Anthropic](https://www.anthropic.com/news/contextual-retrieval)
- Relevance: the prepended text is a *short, LLM-written situating summary*, not raw context. This is the closest published analogue to "LLM writes the gist using context". The text that reaches the embedder is short and targeted, which limits dominance. Note that Anthropic used Claude for this step; a local LLM would be needed under the project constraints.

**Jina "Late Chunking" (Günther et al., arXiv 2409.04701, Sep 2024)** [from memory, not re-fetched this session]
- Method: encode the whole long text with a long-context embedder, then mean-pool the token embeddings within each chunk span. Each chunk vector is then conditioned on its surroundings through attention, without extra text — [arXiv 2409.04701](https://arxiv.org/abs/2409.04701)
- Reported gains were on retrieval benchmarks and grew with document length — [arXiv 2409.04701](https://arxiv.org/abs/2409.04701)
- Applying it here would mean encoding "[video brief][parent][comment]" and pooling only over the comment's tokens. That is a mechanism-level analogue, not something the paper tested.

**Contextual Document Embeddings, CDE (Morris & Rush, arXiv 2410.02525, Oct 2024; ICLR 2025)** [from memory, not re-fetched this session]
- Method: a two-stage encoder. The first stage embeds a sample of the *corpus*. The second stage embeds each document conditioned on those corpus embeddings. It also uses contrastive training with batches clustered by neighbour similarity — [arXiv 2410.02525](https://arxiv.org/abs/2410.02525)
- The context here is *corpus-level* (dataset adaptation, an IDF-like effect), not a per-item parent. In principle it could down-weight the features that every comment on one video shares, such as the interviewee's name or the video topic — [arXiv 2410.02525](https://arxiv.org/abs/2410.02525). That application is an inference.

**Decontextualization (Choi et al., TACL 2021, arXiv 2102.05169)** [from memory, not re-fetched this session]
- Defines rewriting a sentence so it is interpretable out of context while keeping its meaning. Edit types include:
  - name completion
  - pronoun → noun-phrase swaps
  - removal of discourse markers
  - bridging anaphora
  - adding scope or background information
- Some sentences are marked *infeasible* to decontextualize — [arXiv 2102.05169](https://arxiv.org/abs/2102.05169)
- The "infeasible" category maps directly onto replies like "this ^^", "agree" or "ㅋㅋㅋ". These have no content of their own to embed. Their issue is the parent's issue, and their own contribution is stance only.

**Propositions as stand-alone units ("Dense X Retrieval", Chen et al., arXiv 2312.06648, Dec 2023)** [from memory, not re-fetched this session]
- Decomposing text into atomic, *decontextualized* propositions and embedding those improved dense retrieval over passage or sentence units — [arXiv 2312.06648](https://arxiv.org/abs/2312.06648)
- This supports the "LLM rewrites to stand-alone, then embed" design.

**Conversational query rewriting** [from memory, not re-fetched this session]
- CANARD (Elgohary et al., EMNLP 2019) and QReCC (Anantha et al., NAACL 2021) rewrite context-dependent utterances into self-contained ones before retrieval. This is the established pattern for resolving ellipsis or coreference in turn *n* using turns 1..n−1 — [CANARD](https://aclanthology.org/D19-1605/), [QReCC](https://aclanthology.org/2021.naacl-main.44/)

**Instruction-conditioned embedding as a context or aspect knob**
- See Q2. InBedder found that instruction-tuned embedders (INSTRUCTOR, e5) barely change behaviour with instructions — [Peng et al., ACL 2024](https://arxiv.org/abs/2402.09642)

### Inferences
**Preferred design: context only in the gist step.**
- The LLM sees the video brief, the parent, and the transcript segment if it passes a gate. It outputs a stand-alone, stance-neutral issue gist, and only the gist is embedded.
- The gist should name the *issue* explicitly. For example, a reply "ㅇㅈ" (agree) under a parent about tuition policy should become "tuition policy (agreement with parent)", not "agrees".
- This mirrors Anthropic contextual retrieval and the proposition/decontextualization line. Context influences the vector only through what the LLM chooses to write, so it cannot dominate the vector unless the LLM copies it.

**Content-free replies ("this ^^", "agree", "ㅋㅋ", emoji-only).**
- Treat these as Choi et al.'s "infeasible" case.
- Do not embed them independently. Give them the parent's issue cluster, or put them in a "reaction to parent" bucket, and keep their stance in the stance facet.
- This prevents a large "agree/lol" cluster, which is otherwise a classic polarity-or-genre cluster.

**Vector mixing (v = v_comment + α·v_parent).**
- I found no published evaluation of this for comment clustering.
- The mechanics follow from geometry. Siblings under one parent all receive the same α·v_parent offset, so a larger α pulls whole threads together regardless of issue. That is a thread cluster, not an issue cluster.
- If used, α should be small, about 0.1–0.3. Better still, apply it only when the comment's own gist is low-information, for example when the gist is short or the LLM flags it as "context-dependent".

**Concatenating raw context into the embedding input.**
- This is the riskiest option for short comments. A 300-token transcript segment next to a 15-token comment will dominate mean-pooled or last-token embeddings.
- The late-chunking analogue (pool only the comment tokens) is a middle ground that can be tested.

**Low-confidence segment links (median cosine 0.58).**
- No paper speaks to this exact threshold, so it should be set empirically. Two practical options:
  - (a) Pass the segment to the LLM only when cosine is at least τ, with τ chosen from a small hand-labelled sample where "the link is correct" can be judged, or when cosine beats the second-best segment by a margin.
  - (b) Always pass the top-k (k = 2–3) segments plus their scores, and tell the LLM it may ignore them.
- Never put the segment into the embedding input. Absolute cosine values are model-specific; 0.58 may be high for one model and low for another. Rank and margin are therefore more transferable than a fixed threshold.

**Corpus-level context (CDE-style).**
- This is the one context method that might *reduce* video-specific confounds, such as every comment naming the host.
- CDE is English-centric (cde-small), so its Korean coverage is doubtful [UNVERIFIED].

### Gaps
- I found no study that measures "context dominance" in embeddings of short social-media texts, or that compares parent-mixing, concatenation and gist-only for reply clustering.
- I could not verify whether late chunking or contextual retrieval was evaluated on Korean.
- I found no published threshold guidance for transcript-to-comment linking.
- Exact effect sizes for Dense X Retrieval and late chunking were not re-fetched this session.

## Q2. Issue vs stance (and vs target person): evidence that embeddings split by polarity, and techniques to prevent it

### Takeaway
The evidence conflicts on which way embeddings fail.
- Vaccine-opinion work shows off-the-shelf SBERT-style models split same-aspect texts by stance.
- Stance-aware-ST work shows the opposite default: topic dominates and stance is weakly encoded.
- Separately, there is direct evidence that **person names dominate embedding similarity**. This matches the requester's "clusters by target person" observation.

Instructions are a weak lever. Instruction-tuned embedders largely ignore instructions on aspect-conditioned tests, though newer, instruction-trained LLM embedders such as Qwen3 may be better [partly unverified]. Stronger levers, in rough order of cost:
1. A stance-neutral, entity-anonymized gist.
2. LEACE erasure of stance and person directions, using the labels the pipeline already produces.
3. Contrastive fine-tuning on "same issue, opposite stance" pairs.

### Cited Findings
**Embeddings split same-aspect opinions by stance**
- Zhu, Zhao, Pergola & He, *Findings of ACL 2023*, "Disentangling Aspect and Stance via a Siamese Autoencoder for Aspect Clustering of Vaccination Opinions" (DOC).
  - Their example: "mRNA vaccines are poison" and "The Pfizer vaccine is safe" are texts "that the majority of existing methodologies are prone to cluster into different groups due to the opposite stances manifested, despite the fact that both of them are targeting safety issues" — [ACL Anthology 2023.findings-acl.115](https://aclanthology.org/2023.findings-acl.115.pdf)
  - The DOC method has four parts:
    - It starts from out-of-the-box SBERT embeddings.
    - A disentangling cross-attention, supervised by stance labels, separates stance from aspect.
    - A swapping autoencoder exchanges stance between pairs.
    - The resulting aspect representation is clustered with k-means, k-medoids or agglomerative clustering (AHC).
    - Source: [ACL Anthology](https://aclanthology.org/2023.findings-acl.115.pdf)
  - Results on the CMF dataset (K = 7) and the VAD dataset (K = 8), with metrics clustering accuracy (CA), NMI and AvgBS:

    | Setting | CA | NMI | AvgBS |
    |---|---|---|---|
    | CMF, SBERT-AHC | 51.7 | 48.5 | 18.9 |
    | CMF, DOC-k-medoids | 54.2 | 51.0 | 20.7 |
    | VAD, SBERT-AHC | 64.4 | 61.2 | 20.9 |
    | VAD, DOC-AHC | 66.7 | 63.6 | 22.8 |

    Plain RoBERTa or DeBERTa with k-means was far worse (CA 35–48). Source: [ACL Anthology](https://aclanthology.org/2023.findings-acl.115.pdf)
  - Gains over plain SBERT are modest, about 2–3 points. Cosine-based k-medoids or AHC beat k-means on SBERT vectors — [ACL Anthology](https://aclanthology.org/2023.findings-acl.115.pdf)

**The opposite default: topic dominates and stance is under-encoded**
- Ghafouri et al., EMNLP 2024, "I love pineapple on pizza != I hate pineapple on pizza: Stance-Aware Sentence Transformers for Opinion Mining".
  - Sentence transformers group topically similar texts well but struggle to separate opposing viewpoints on the same topic.
  - They fine-tune with pro/con arguments on claims to *add* stance sensitivity.
  - Source: [ACL Anthology 2024.emnlp-main.1171](https://aclanthology.org/2024.emnlp-main.1171)
- Implication: the same contrastive recipe can push toward *either* axis, depending on which pairs are labelled positive.

**Argument similarity and aspect corpora**
- Reimers et al., ACL 2019, "Classification and Clustering of Arguments with Contextualized Word Embeddings" (arXiv 1906.09821).
  - Introduced the UKP ASPECT corpus of argument-pair similarity.
  - Clustering improved by 7.8 percentage points over strong baselines on that corpus and by 12.3 points on the Argument Facet Similarity (AFS) corpus.
  - Classification improved by 20.8 points on the UKP corpus and 7.4 points on IBM Debater.
  - Source: [arXiv 1906.09821](https://arxiv.org/abs/1906.09821)
  - From memory, not re-fetched this session: about 3.6k sentence pairs over 28 topics, labelled on a 4-level similarity scale (high / some / no similarity / different topic). The pairs are about whether two arguments address the same aspect.
- The BWS Argument Similarity corpus was introduced with Augmented SBERT (Thakur et al., NAACL 2021) [from memory, not re-fetched this session] — [arXiv 2010.08240](https://arxiv.org/abs/2010.08240)
- Same Side Stance Classification (Stein et al., ArgMining 2021) asks whether two same-topic arguments share a stance. It treats stance as a pairwise relation that is separable from topic [from memory, not re-fetched this session] — [ACL Anthology 2021.argmining-1.15](https://aclanthology.org/2021.argmining-1.15/)

**Target-person dominance (name bias), which bears on the requester's extra concern**
- "What is in a name? Mitigating Name Bias in Text Embeddings via Anonymization" (Findings of ACL 2025, arXiv 2502.02903).
  - Embedding models favour name overlap over content. Stories with the same plot but different character names score *lower* similarity than stories with similar names but different plots.
  - Inference-time anonymization, which detects names and replaces them, significantly reduces this bias.
  - The authors flag clustering as an affected application.
  - Sources: [arXiv 2502.02903](https://arxiv.org/abs/2502.02903), [ACL Anthology 2025.findings-acl.914](https://aclanthology.org/2025.findings-acl.914)
- EASE (entity-aware contrastive sentence embeddings, 2022) notes that entity-aware training makes embeddings more sensitive to *topical* (entity) similarity, sometimes too sensitive — [arXiv 2205.04260](https://arxiv.org/abs/2205.04260)

**Do instruction prompts change what clusters capture?**
- InBedder (Peng et al., ACL 2024, arXiv 2402.09642) built "instruction awareness" tests:
  - IntentEmotion: triplets where the same utterances must rank differently under an intent instruction than under an emotion instruction.
  - InstructSTSB.
  - NYTClustering: the same NYT corpus clustered by topic versus by location.
  - Source: [arXiv 2402.09642](https://arxiv.org/abs/2402.09642)
- Results (InstructSTSB / IntentEmotion / NYT / average):

  | Model | InstructSTSB | IntentEmotion | NYT | Average |
  |---|---|---|---|---|
  | e5-large-v2 (no instruction) | 0.00 | 30.24 | 50.07 | 26.77 |
  | instructor-large | −15.02 | 47.96 | 49.96 | 27.63 |
  | llama-2-7b-InBedder | 22.07 | 89.68 | 64.65 | 58.80 |

  Source: [arXiv 2402.09642](https://arxiv.org/abs/2402.09642)
- What the table shows:
  - INSTRUCTOR's instructions barely changed NYT topic-versus-location clustering compared with no instruction.
  - INSTRUCTOR's InstructSTSB score was negative.
  - Embedding the *answer* to the instruction (generate a short answer, then embed it) followed instructions far better.
  - Source: [arXiv 2402.09642](https://arxiv.org/abs/2402.09642)
- InBedder's idea is close to the existing "LLM gist → embed" pipeline. The gist *is* the answer to "what issue does this comment discuss?" That gives empirical support to the current design over relying on embedder instructions.
- GSTransform (Feng et al., ACL 2025, arXiv 2505.24754) found that "instruction-relevant information is inherently encoded in generic embeddings but remains underutilized".
  - It proposes a lightweight transformation of precomputed embeddings, guided by instruction-specific clustering and labels, to adapt them to an instruction.
  - It was tested on 9 datasets across 3 tasks, with 6–300× speedups over re-encoding.
  - Source: [arXiv 2505.24754](https://arxiv.org/abs/2505.24754)
- Myntti, Kanerva, Laippala & Ginter (arXiv 2610.10508, 7 Oct 2026), "Effects of Retrieval Instructions in Embedding Models":
  - Embedding models "can fail to follow even simple task instructions when query-side distractors are included".
  - Fine-tuning with such distractors helps.
  - Source: [arXiv 2610.10508](https://arxiv.org/abs/2610.10508)
- [UNVERIFIED, search snippet only; attribution uncertain, possibly 2610.10508] Qwen3-Embedding, trained with instructions, follows them in nearly all cases, while embedders trained without instructions do not — [search result pointing to arXiv 2610.10508](https://arxiv.org/abs/2610.10508)
- The same snippet said E5-Mistral ranks 4th on MTEB but 10th on instruction following [UNVERIFIED] — [arXiv 2505.24754](https://arxiv.org/pdf/2505.24754)

**C-STS, conditional similarity (Deshpande et al., EMNLP 2023, arXiv 2305.15093)**
- Sentence pairs are rated for similarity under natural-language conditions (for example "the issue discussed"). The same pair is rated under two different conditions — [HF papers 2305.15093](https://huggingface.co/papers/2305.15093)
- From memory, not re-fetched this session: the original paper found both fine-tuned and instruction-tuned models scored low (Spearman well under 0.5).
- Follow-ups:
  - Tu et al. 2024 (arXiv 2406.03673) found the C-STS dataset has many annotation errors. They reformulated the task as QA and built an error-detection pipeline with over 80% F1 — [arXiv 2406.03673](https://arxiv.org/abs/2406.03673v1)
  - Zhang et al., EMNLP 2025, re-annotated C-STS with LLMs and gained 5.4% Spearman (statistically significant) from training on the cleaned data — [Cardiff ORCA](https://orca.cardiff.ac.uk/id/eprint/187574/), [Underline](https://underline.io/lecture/131125-annotating-training-data-for-conditional-semantic-textual-similarity-measurement-using-large-language-models)
- Takeaway: conditional similarity is hard and noisily benchmarked. Do not assume a condition string will cleanly remove stance.

**Linear concept erasure** [from memory, not re-fetched this session]
- INLP (Ravfogel et al., ACL 2020): repeatedly trains a linear classifier for the concept and projects the embeddings onto its nullspace — [arXiv 2004.07667](https://arxiv.org/abs/2004.07667)
- LEACE (Belrose et al., NeurIPS 2023): a closed-form projection that provably stops *every* linear classifier from recovering the concept while changing the representation as little as possible (in least-squares terms) — [arXiv 2306.03819](https://arxiv.org/abs/2306.03819)
- Both methods remove only *linearly* decodable information. Nonlinear stance signal can remain.

### Inferences
**1. A stance-neutral, entity-anonymized gist is the cheapest and highest-leverage fix.**
- Have the open-coding LLM output the issue as a neutral noun phrase. Examples:
  - "safety of X policy", not "X policy is dangerous".
  - "host's remark on military service", not "the host is a hypocrite".
- Strip evaluative adjectives, emoji, and sarcasm markers.
- Replace person names with a role placeholder ("[GUEST]", "[HOST]") *or drop them*. Put target person into a separate facet, just as stance is already separate.
- The name-bias anonymization paper directly supports the name step. DOC and stance-aware-ST support the idea that evaluative wording drives polarity splits.

**2. Diagnose before fixing.**
- On the existing clusters, compute how much cluster membership is predicted by:
  - LLM stance labels
  - target-person labels
  - a sentiment classifier
- Use NMI or the accuracy of a linear probe on the embeddings.
- This tells you which confound is actually present. The literature conflicts on the default, so measure it.

**3. LEACE on gist embeddings.**
- The pipeline already produces stance labels and can produce target-person labels. Fit LEACE (open-source, cheap, CPU-feasible) to erase "stance" and "target person" from the embeddings before clustering.
- Caveat: some issues are inherently one-sided, or tied to one person. Erasure will also remove that correlated issue signal.
- Check that clusters stay issue-coherent, for example with a small labelled set of same-issue/different-stance pairs.

**4. Instruction prompts in Qwen3-Embedding or e5-instruct.**
- Prompts such as "Represent the issue discussed, ignoring the speaker's stance" are cheap to try.
- The evidence (InBedder, 2610.10508) says not to rely on them. Treat them as an ablation, not as the mechanism.
- Embedding the LLM's *answer* (the gist) is the empirically better version of the same idea.

**5. Contrastive fine-tuning, if needed.**
- Build pairs with the local LLM: the same issue rewritten with the opposite stance, and with a different target person, as positives. A different issue with the same stance, as hard negatives.
- Fine-tune a small multilingual embedder, for example with sentence-transformers MultipleNegativesRankingLoss. This fits in 16 GB for models under 1B parameters.
- This is the stance-aware-ST recipe (EMNLP 2024) inverted, and the DOC idea without the autoencoder.

**6. Clustering algorithm.**
- DOC's results suggest cosine-based k-medoids or AHC beat k-means on SBERT vectors. That is a small, free change.

### Gaps
- **Korean-specific evidence: none found** on stance-versus-topic confounding, name bias, or instruction-following in Korean embeddings.
- The search on Korean comment clustering returned only generic BERTopic studies of Korean news comments (for example klue/roberta-base with HDBSCAN), with no stance or entity analysis — [koreascience](https://koreascience.kr/article/JAKO202417243235084.page?lang=ko), [KRNewsBERT repo](https://github.com/Navy10021/Parallel_Clustering_based_TM)
- Korean sarcasm and particle-heavy short replies (ㅇㅈ, ㄹㅇ, ㅋㅋ) are unstudied in this context.
- Whether name anonymization works for Korean honorific or nickname references, for example "OO형" or nicknames for a guest, is untested.
- I could not verify the Qwen3-Embedding instruction-following claim from a primary source. The Qwen3-Embedding model card says instructions improve retrieval by roughly 1–5% [from memory, not re-fetched this session], but that is not evidence about aspect control in clustering.
- I found no paper that applies LEACE or INLP specifically to remove stance or sentiment before *topic clustering*; this application is my inference.
- The full C-STS original numbers and the specific model scores in the follow-ups were not re-fetched this session.
