---
id: R-0002
type: report
task: T-0002
title: "Embed a stance-free issue gist, not the comment"
status: final
created: 2026-11-02
tags: [issue-normalization, stance-separation, text-preprocessing]
uses: [N-0006, N-0007, N-0008]
---

# Embed a stance-free issue gist, not the comment

**Bottom line.** For the issue facet, embed a short, English, stance-neutral issue phrase in which names are replaced by roles. Have the local LLM write it from the comment plus its context, and keep the raw comment, the stance and the target person in separate fields that are never embedded into the issue vector. This directly targets the EXP-005 failure, where clusters grouped by polarity and target. The strongest analogue is PAKPA: it clusters only on the aspect and keeps sentiment in a separate field, and it beat sentence-embedding clustering on quantification precision (0.95 vs 0.76) with an all-open 7B option ([PAKPA, ACL 2024](https://aclanthology.org/2024.acl-long.576.pdf)). The direct evidence that LLM rewriting before clustering helps is real but modest and English-only: roughly 0 to +12 accuracy points, once negative ([Viswanathan et al., TACL 2024](https://aclanthology.org/2024.tacl-1.18.pdf)). **No study tests any of this on Korean or Korean–English code-switched comments**, so the 300-comment experiment in section 3 is not optional.

## 1. Shortlist (ranked)

**#1 Context-conditioned, stance-neutral, role-anonymized issue gist; embed only `issue`.** This applies the field separation that worked in PAKPA. It also follows InBedder's finding that embedding an LLM's *answer* to "what is this about?" obeys the intended axis far better than an instruction-tuned embedder does ([InBedder, ACL 2024](https://arxiv.org/abs/2402.09642)). Name anonymization addresses the documented tendency of embeddings to match on names over content ([name bias, Findings ACL 2025](https://arxiv.org/abs/2502.02903)).

*Gist schema* (one JSON object per deduplicated comment, constrained to the schema with llama.cpp's JSON-schema/grammar output; that feature is from general knowledge, not the notes):

```json
{
  "has_issue": true,
  "no_issue_type": null,            // reaction | meme | praise_or_insult_only | off_topic | spam | unclear
  "context_dependent": false,       // true if meaning comes from parent or timestamp
  "issues": [                       // 1-2 items max; one issue per item
    {
      "issue": "handling of the military service question",   // EMBEDDED. 4-10 word English noun phrase
      "target_role": "[GUEST]",     // [GUEST] [HOST] [GUEST2] [OTHER_PERSON] [CHANNEL] [NONE]
      "target_name": "<canonical name from video brief>",      // target facet only
      "stance": "oppose",           // support | oppose | mixed | none | unclear
      "sarcasm_suspected": true,
      "evidence": "병역 얘기는 왜 피함?"                        // verbatim substring of the comment
    }
  ],
  "lang": "ko",                     // ko | en | mixed
  "confidence": "high"              // high | medium | low
}
```

*Prompt rules.* All of these are design inferences assembled from the cited work, not tested on this data.

1. **State the clustering goal.** "We group comments by the issue they raise, not by who they are about, and not by whether they agree." The prompt should have an instruction plus about 4 demonstrations. Viswanathan et al. found that demonstrations without instructions drift to the wrong axis, and instructions without demonstrations produce inconsistent output.
2. **Make the issue stance-neutral.** Phrase it as "a debate topic someone could argue either side of". Ban evaluative words in both languages (liar, hypocrite, great, 대단, 어이없, 논리 박살 …). Stance and sarcasm go in their own fields.
3. **Replace names with role placeholders, and drop the person entirely unless the issue *is* that person's conduct.** For example, use "tuition-free policy cost", not "[GUEST]'s tuition-free policy". The video brief supplies the cast list so the LLM can map nicknames and honorific forms (OO형, 쌤) to roles. Otherwise the shared "[GUEST]" token re-creates target clusters.
4. **Use paired pro/con demonstrations.** In each pair, two comments with opposite stance (one of them sarcastic) map to the *identical* `issue` string. Add one pair with a different target and the same issue. Include PAKPA-style invalid examples with corrections: too specific, too general, and stance leaking into the issue. Draw demonstrations from a *different* video, and mix Korean, English and code-switched comments. Never list candidate issues during open coding, because that leaks taxonomy.
5. **Abstain explicitly.** Set `has_issue=false` with a type for reactions, memes and unreadable comments, and use `unclear` rather than guessing. This follows Claimify's rule of extracting a claim "only when there is high confidence" ([Claimify, ACL 2025](https://arxiv.org/abs/2502.10855)).
6. **Enforce faithfulness.** `evidence` must be a verbatim substring of the comment, or the gist is regenerated. Use temperature 0 with a fixed seed, and cache all outputs, because LLooM reported run-to-run variability and had no verification step ([LLooM, CHI 2024](https://arxiv.org/html/2404.12259)).

*What gets embedded.* Only `issue`, after L2 normalization. When a comment has two issues, it gets two vectors that share the comment ID.

*Where context goes.* Context goes into the gist prompt only, never into the embedded string. That includes the video brief, the parent comment for replies, and a transcript segment when there is a timestamp or a confident link.

*Filter.* After normalization, apply a rule pre-filter, then the LLM's `has_issue`. Excluded items stay in a counted reaction bucket. HDBSCAN noise is kept separate from that bucket.

*Dedup.* Before gisting, deduplicate in three tiers and carry `count` forward as a prevalence weight. Never deduplicate identical *gists*, because they are the signal.

**#2 Same gist, concatenated with an anonymized raw-text embedding (gist ⊕ raw).** This is the most robust published recipe. Concatenating LLM keyphrases with the raw embedding lifted Tweet clustering from 57.5 to 69.8 accuracy with agglomerative clustering, though it cost 1.5 points on CLINC with k-means ([Viswanathan et al.](https://aclanthology.org/2024.tacl-1.18.pdf)). IDAS and LiSA also combine LLM labels with the original text ([IDAS](https://arxiv.org/abs/2305.19783); [LiSA, ACL 2025](https://aclanthology.org/2025.acl-long.902.pdf)). Prefer it over #1 if issue-only gists homogenize ("controversy about guest's remarks"). The cost is that raw text re-imports polarity cues, so the confound check matters even more.

**#3 #1 or #2 plus linear erasure of stance and target (LEACE).** The pipeline already produces stance and target labels. LEACE projects out every linearly decodable direction for those labels in closed form, cheaply, on CPU ([LEACE, NeurIPS 2023](https://arxiv.org/abs/2306.03819), from memory). Use it only if the confound check still shows residual stance or target structure. Two caveats: erasure also removes issue signal for one-sided or person-bound issues, and no one has published this exact use, so it is an inference.

## 2. Fit to constraints

Everything runs locally. Gisting uses the Qwen3.8-27B GGUF already running in llama.cpp. That is above the 7–8B scale at which open models produced usable per-item labels (LiSA with Llama-3-8B; PAKPA with Mistral-7B end to end, Soft-F1 0.71 vs 0.70 for a GPT-3.5 hybrid). It is still below the GPT-4-class models that the *taxonomy-generation* results depend on: TopicGPT's Mistral-7B generator produced 1,418 topics vs 151 ([TopicGPT, NAACL 2024](https://arxiv.org/html/2311.01449)). So the LLM writes per-comment gists, and clustering plus human review does the merging.

The LLM and the embedder share the 16 GB card, so run gisting and embedding as separate passes. Qwen3-Embedding-4B, the team's current winner, fits easily once the LLM is unloaded.

Preprocessing needs only pure-Python packages, each pinned in `uv.lock`:

- **soynlp**: use `repeat_normalize`. Avoid `only_hangle`, which deletes English, and `emoticon_normalize`, which splits syllables into jamo ([soynlp](https://github.com/lovit/soynlp)).
- **unicodedata**, for NFKC normalization.
- **datasketch**, optional, for MinHash.
- **concept-erasure** (EleutherAI's LEACE package), for option #3 only. The package name is from memory and unverified.

The notes do not give a current soynlp release or confirm that it supports Python 3.12, so check both at install. No Korean morphological analyzer is needed for embedding. Kiwi is only useful later for c-TF-IDF keyword labels.

Writing every `issue` in English removes language as a clustering axis, since multilingual embedders partly cluster by language ([LAReQA](https://arxiv.org/pdf/2004.05484)). It also relies on stronger English instruction-following, which TopicGPT notes degrades outside English. This is an inference. A Korean pivot is the alternative if reviewers prefer Korean codes. Raw Korean text stays in the record for review.

## 3. Smallest confirming experiment

**Data.** Use 300 comments from 2–3 interview videos: 150 Korean (including about 40 code-switched) and 150 English. Stratify toward replies and timestamped comments.

**Labels.** One annotator labels three things per comment: issue (free text, then consolidated to 15–30 codes), stance (support / oppose / none), and target role. The annotator labels **before seeing any gist**, because reviewers anchor on LLM suggestions and that inflates measured accuracy ([Just Put a Human in the Loop?, Jul 2025](https://arxiv.org/abs/2507.15821)). A second annotator double-codes 100 comments to estimate agreement. This takes about one person-day.

**Variants.** All variants use the same embedder (Qwen3-Embedding-4B) and the same clusterers: cosine agglomerative clustering cut at the gold K, plus an HDBSCAN `min_cluster_size` sweep.

| Variant | What is embedded |
|---|---|
| A | Raw comment, normalized |
| B | The current free-form open-coding gist |
| C | Gist ⊕ anonymized raw (#2) |
| D | `issue` only (#1) |
| D+ | D after LEACE (#3), fit with cross-validation |

**Primary metric.** AMI and ARI against the gold issue codes at matched K, with bootstrap 95% confidence intervals. Report Korean and English halves separately. With 300 items, only differences of about 0.05 or more are resolvable. This threshold is an inference.

**Confounding check.** The goal is that clusters track issue, not stance or target.

1. Compute NMI(clusters, stance) and NMI(clusters, target role). Lower is better. Compare each against the gold issue labels' own NMI with stance and target, which is the floor that genuinely one-sided issues force.
2. Compute a **stance-crossed AUC**. Take same-issue/opposite-stance pairs as positives and different-issue/same-stance (or same-target) pairs as negatives, and measure how well cosine separates them. Raw embeddings should do worst here if EXP-005's diagnosis holds.
3. For each cluster, check whether it is more than 90% one language when the corpus is mixed.

**Gist health checks.**

1. The share of `issue` strings that contain banned evaluative words.
2. The evidence-substring pass rate.
3. Distinct issues per 100 comments, to catch homogenization.
4. A 50-item human read of Korean sarcastic gists.

**Decision rule.** Adopt D if its issue AMI is within 0.02 of C's and its stance and target NMI are clearly lower. Otherwise adopt C. Add LEACE (D+) only if NMI with stance or target stays above the gold floor.

## 4. Details on demand

### Context injection: context in the prompt, never in the vector

Every published context-injection method was tested for chunk *retrieval*, not for clustering short replies. Anthropic's contextual retrieval prepends an LLM-written situating summary of 50–100 tokens and cut retrieval failures by 35%, from 5.7% to 3.7% ([Anthropic, Sep 2024](https://www.anthropic.com/news/contextual-retrieval), from memory). Late chunking pools only the chunk's tokens from a long encoding ([arXiv 2409.04701](https://arxiv.org/abs/2409.04701), from memory).

The design here goes one step further: context shapes *what the LLM writes*, so it can reach the vector only if the LLM copies it. This follows the decontextualization and proposition line of work ([Choi et al., TACL 2021](https://arxiv.org/abs/2102.05169); [Dense X, EMNLP 2024](https://arxiv.org/html/2312.06648)).

**Transcript segments.** Concatenating a roughly 300-token segment next to a 15-token comment would dominate the pooled vector. The safer approach is to resolve timestamps to a ±15–30 s window, and for unlinked comments pass the top 2–3 segments with their scores and permission to ignore them. Gate on rank or margin rather than on the 0.58 median cosine, because absolute cosine values are model-specific. All of this is inference.

**Content-free replies** ("ㅇㅈ", "this ^^", "ㅋㅋ"). These match Choi et al.'s "infeasible to decontextualize" case. Set `context_dependent=true`, inherit the parent's issue, and record only the reply's stance. Otherwise they form a large "agree/lol" polarity cluster.

**Parent vector mixing.** If you use v + α·v_parent at all, keep α at 0.1–0.3 and apply it only to low-information gists. Larger α pulls whole threads together regardless of issue (geometric inference; no published test).

### Issue vs stance vs person: the evidence points both ways, so measure

Off-the-shelf SBERT splits "mRNA vaccines are poison" from "Pfizer is safe" even though both are about safety. A stance-disentangled representation gained about 2–3 clustering-accuracy points ([DOC, Findings ACL 2023](https://aclanthology.org/2023.findings-acl.115.pdf)). Other work finds the opposite default: topic dominates and stance is under-encoded ([Stance-aware ST, EMNLP 2024](https://aclanthology.org/2024.emnlp-main.1171)). Name overlap outweighs plot similarity in embeddings, and anonymization reduces this ([arXiv 2502.02903](https://arxiv.org/abs/2502.02903)).

Instruction prefixes on the embedder are a weak lever. INSTRUCTOR barely changed topic-vs-location clustering ([InBedder](https://arxiv.org/abs/2402.09642)), and embedders fail simple instructions when distractors are present ([Myntti et al., Oct 2026](https://arxiv.org/abs/2610.10508)). Test a Qwen3-Embedding "regardless of stance" instruction only as an ablation.

If #3 is not enough, the next step is contrastive fine-tuning of a sub-1B embedder. Positives are LLM-generated same-issue pairs with flipped stance or a swapped target. This is the stance-aware-ST recipe inverted, and it is an inference.

**Korean evidence: none** on stance or name confounding in embeddings, and nicknames and honorific references are untested for anonymization.

### Short texts, code-switching, sarcasm

**Normalization:**

1. NFKC.
2. Strip zero-width characters.
3. `repeat_normalize(t, num_repeats=2)` (ㅋㅋㅋㅋ → ㅋㅋ).
4. Collapse repeated emoji to at most 2, but keep emoji for the LLM because 🙄 and 🤡 flip stance.
5. Replace URLs with `<URL>` and timestamps with `<TS>`, storing the parsed seconds.

**Rule pre-filter.** After stripping, a comment with fewer than 2 Hangul syllables or fewer than 2 Latin words goes to the reaction bucket, as do regex memes (1빠, first, 알고리즘이 데려옴). Transformer pipelines need little more cleaning than this ([BERTopic guide, Apr 2026](https://www.analyticsvidhya.com/blog/2026/04/bertopic-guide/), secondary source).

**Code-switching.** Do not machine-translate raw comments, because MT discards slang and sarcasm cues. The English-pivot gist translates and abstracts in one step (inference). Code-switched text is "anchored" to its framing language inside multilingual LLMs ([arXiv 2606.19668, Jun 2026](https://arxiv.org/abs/2606.19668), abstract only).

**Sarcasm.** LLMs still trail supervised models on sarcasm ([SarcasmBench, IEEE TAC 2025](https://www.computer.org/csdl/journal/ta/2025/04/11146812/29GBxQiBx9m)). Korean sarcasm depends heavily on context ([KoCoSa, LREC-COLING 2024](https://aclanthology.org/2024.lrec-main.864/)). However, sarcasm usually flips *stance*, not *issue*: "와 논리 대단하시네요 ㅋㅋ" is about the guest's reasoning either way. A stance-neutral issue field is therefore robust to sarcasm misreads, which is the main reason to prefer #1 over gists that carry stance.

**Korean resources.** [KorSarcasm](https://github.com/SpellOnYou/korean-sarcasm) (9,319 tweets) can sanity-check the 27B model's Korean sarcasm recall. [KOLD](https://arxiv.org/pdf/2205.11315) (40k NAVER and YouTube comments with titles) is a source of realistic few-shot inputs. **No evidence exists on open 8–30B models' Korean sarcasm accuracy.**

### Dedup: before gisting, with counts

Deduplicate within each video before gisting, in three tiers:

1. Exact hash of the normalized text.
2. Character 3-gram Jaccard ≥ 0.8. This works for Korean without tokenization, and MinHash is optional at 2k comments.
3. Raw-embedding cosine ≥ 0.95, grouped by connected components, for copy-paste with small edits.

Keep one representative with `count` and the member IDs. Cluster the representatives, and use `count` only to report prevalence. This avoids artificial density peaks, since HDBSCAN has no sample weights. Flag groups with count ≥ 5 as possible brigading or memes. Do not deduplicate around cosine 0.85, because different people making the same point is the prevalence signal. SemDeDup's k-means partitioning is unnecessary at this scale ([SemDeDup, ICML 2023](https://arxiv.org/abs/2303.09540)). All thresholds are untested on Korean comments.

## 5. Sources

Status key: **V** = primary source read by the research team; **M** = from memory, not re-fetched; **U** = snippet or abstract only, unverified. 🇰🇷 marks Korean-specific evidence.

| Source | Date | Used for | Status |
|---|---|---|---|
| [PAKPA, Tang et al., ACL 2024](https://aclanthology.org/2024.acl-long.576.pdf) | Aug 2024 | Aspect/sentiment separation; 0.95 vs 0.76 precision; open 7B | V |
| [Viswanathan et al., TACL 2024](https://aclanthology.org/2024.tacl-1.18.pdf) | Jul 2023 / 2024 | Gist ⊕ raw gains; instruction vs demonstration ablation | V |
| [InBedder, Peng et al., ACL 2024](https://arxiv.org/abs/2402.09642) | Feb 2024 | Embedding LLM answers beats embedder instructions | V |
| [Name bias in text embeddings, Findings ACL 2025](https://arxiv.org/abs/2502.02903) | Feb 2025 | Anonymization reduces name dominance | V (abstract level) |
| [DOC, Zhu et al., Findings ACL 2023](https://aclanthology.org/2023.findings-acl.115.pdf) | Jul 2023 | SBERT splits same aspect by stance | V |
| [Stance-aware ST, EMNLP 2024](https://aclanthology.org/2024.emnlp-main.1171) | Nov 2024 | Topic dominates stance by default | V (abstract level) |
| [LLooM, CHI 2024](https://arxiv.org/html/2404.12259) | Apr 2024 | Distill-then-cluster; no verification; re-run variability | V |
| [TopicGPT, NAACL 2024](https://arxiv.org/html/2311.01449) | Nov 2023 | 7B generation failure; non-English degradation | V |
| [LiSA, ACL 2025](https://aclanthology.org/2025.acl-long.902.pdf) | 2025 | Open 8B labels + embeddings on tweets | V |
| [IDAS](https://arxiv.org/abs/2305.19783) | May 2023 | Abstractive labels before clustering, up to +7.42 | U (model/venue) |
| [Claimify, ACL 2025](https://arxiv.org/abs/2502.10855) | Feb 2025 | Abstain when interpretation uncertain | U (pipeline details) |
| [Dense X Retrieval, EMNLP 2024](https://arxiv.org/html/2312.06648) | Dec 2023 | Decontextualized propositions | V |
| [Choi et al., Decontextualization, TACL 2021](https://arxiv.org/abs/2102.05169) | Feb 2021 | "Infeasible" replies | M |
| [Anthropic Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) | Sep 2024 | Context written by an LLM, −35% failures | M |
| [Late chunking](https://arxiv.org/abs/2409.04701) | Sep 2024 | Pooling only the comment's tokens | M |
| [LEACE, NeurIPS 2023](https://arxiv.org/abs/2306.03819) | Jun 2023 | Linear concept erasure | M; use for clustering is an inference |
| [Myntti et al., Retrieval instructions](https://arxiv.org/abs/2610.10508) | Oct 2026 | Embedders ignore instructions when distractors are present | V (abstract) |
| [Just Put a Human in the Loop?](https://arxiv.org/abs/2507.15821) | Jul 2025 | Reviewer anchoring | V (abstract) |
| [LAReQA](https://arxiv.org/pdf/2004.05484) | Apr 2020 | Same-language bias | V (abstract) |
| [Code-switching anchoring](https://arxiv.org/abs/2606.19668) | Jun 2026 | Code-switched representations | U (abstract only) |
| [soynlp](https://github.com/lovit/soynlp) 🇰🇷 | n/a | Repeat normalization; `only_hangle` pitfall | V; version and Python 3.12 support unverified |
| [KoCoSa, LREC-COLING 2024](https://aclanthology.org/2024.lrec-main.864/) 🇰🇷 | Feb 2024 | Korean sarcasm needs context | V (abstract) |
| [KorSarcasm / KoCasm](https://github.com/SpellOnYou/korean-sarcasm) 🇰🇷 | n/a | Sarcasm sanity set | V |
| [KOLD, EMNLP 2022](https://arxiv.org/pdf/2205.11315) 🇰🇷 | May 2022 | YouTube/NAVER comments with context | V |
| [Lee, Woo & Jun, DH2026](https://zenodo.org/records/22107263) 🇰🇷 | Aug 2026 | Korean YouTube BERTopic; language-specific handling matters | U (no preprocessing details) |
| [SarcasmBench, IEEE TAC](https://www.computer.org/csdl/journal/ta/2025/04/11146812/29GBxQiBx9m) | 2025 | LLMs trail supervised models on sarcasm | U |
| [SemDeDup, ICML 2023](https://arxiv.org/abs/2303.09540) | Mar 2023 | Semantic dedup | V (abstract) |
| [BERTopic guide](https://www.analyticsvidhya.com/blog/2026/04/bertopic-guide/) | Apr 2026 | Light preprocessing | Secondary source |

**Korean gap.** There is no study, Korean or code-switched, of raw vs gist clustering, stance or name confounding in embeddings, open-LLM sarcasm handling, or dedup thresholds. The only Korean evidence in this part is about normalization tooling and datasets. Every Korean-side recommendation above is an inference that the section 3 experiment must confirm.
