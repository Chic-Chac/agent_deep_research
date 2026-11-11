---
id: N-0019
type: note
task: T-0004
title: "D. Downstream effects of speaker labels and sentence-boundary quality on retrieval and LLM grounding over spoken-dialogue transcripts, and dialogue chunking"
status: final
created: 2026-11-03
tags: [comment-linking, retrieval, speaker-diarization]
---

# D. Downstream effects of speaker labels and sentence-boundary quality on retrieval and LLM grounding over spoken-dialogue transcripts, and dialogue chunking

Scope note: the earlier report and notes ("Transcript rechunking for comment linking") already cover chunk size and length bias (Bhat 2025, Chroma 2024, Jina size bias), semantic and topic segmentation (Qu 2024, YTSeg, Meta-Chunking), Dense X propositions, late chunking (Günther 2024), Anthropic Contextual Retrieval headline numbers, TREC Podcasts 2020 2-minute windows, and Korean/English segmenters (SaT, Kiwi, punctuation models). These notes do not repeat them. They add speaker-label and attribution effects, ASR-noise robustness, boundary and punctuation effects, dialogue-specific unit granularity (turn vs session vs segment), and new evidence on contextual prefixes.

## Q1. Do speaker labels and speaker-attribution accuracy matter for retrieval, QA and summarization over meetings, interviews and podcasts? How robust are LLMs to ASR and diarization errors?

### Takeaway
For generation and summarization, speaker labels clearly matter. Speaker confusion is the leading cause of factual errors in pre-LLM dialogue summarizers, and adding speaker identity raised ROUGE. Modern LLMs make few attribution errors when labels are correct. No study was found that measures how speaker labels or diarization accuracy affect *retrieval* specifically. LLM robustness to ASR word noise varies widely by model size: GPT-4o holds up, while 8B-class open models degrade sharply. Evidence that diarization errors leave summary quality unaffected is thin and unverified.

### Cited Findings
**Speaker information in summarization (pre-LLM, fine-tuned models)**
- Up to 56.4% of factual errors in abstractive dialogue summaries came directly from "speaker confusion" or "speaker missing". Pre-trained models "cannot identify different speakers well". The fix was speaker-aware supervised contrastive learning at token, turn and global level, which improved factual consistency — [Geng et al., COLING 2022](https://aclweb.org/anthology/2022.coling-1.569.pdf). Single source for the 56.4% figure, which is from SAMSum-style chat summarization, not ASR.
- Adding speaker identity as an input embedding to reduce "who said what" errors substantially increased ROUGE and human ratings on dialogue summarization, including MediaSum — [Kim et al., LREC 2022](https://aclanthology.org/2022.lrec-1.31). Exact deltas were not extracted.
- MediaSum (463.6K NPR/CNN interviews) builds its input by concatenating turns, each prefixed with the speaker name. This "Name: utterance" format has become the de facto serialization for interview transcripts — [MediaSum, arXiv 2103.06410](https://arxiv.org/pdf/2103.06410).

**LLM-era meeting summarization**
- Kirstein et al. (2024) built an error taxonomy over AMI/QMSum-style meetings. "Wrong references" (misattributing statements to participants) appeared in only 4/35 (11%) encoder-decoder and 3/35 (9%) decoder-only summaries. The "speaker dynamics" challenge did not correlate with wrong references, but it did correlate strongly (p ≤ 0.01) with *structural disorganization* errors. Inter-annotator agreement on challenges was 0.79 — [Kirstein et al., arXiv 2404.11124](https://arxiv.org/html/2404.11124). The sample is small (35 per model family), so treat the finding as indicative.
- Later work from the same group (FRAME, 2025) treats speaker attribution of extracted facts as a design goal. It reports hallucination and omission reduced by 2 of 5 points on QMSum and FAME — [arXiv 2509.15901](https://arxiv.org/pdf/2509.15901). This is not an ablation of speaker labels.

**QA over meetings and interviews**
- MeetingQA (ACL 2023), on AMI meetings: answers to participants' questions are 40% multi-span (non-consecutive sentences) and 48% multi-speaker, ~70% of multi-speaker answers contain disagreement, and 30% of questions are unanswerable. The best models reached F1 57.3 against 84.6 for humans — [MeetingQA, ACL 2023](https://aclanthology.org/2023.acl-long.837); [project page](https://archiki.github.io/meetingqa.html). In dialogue, the "answer" to a question is often spread over several turns and speakers, which bears directly on Q-A pair chunking (see Q3).
- ELITR-Bench (COLING 2025) has 271 questions over long ELITR meeting transcripts (Czech and English, 10 min to more than 2 h). The transcripts are ASR output that was manually corrected. Speakers are de-identified as "(PERSON3)", and "Who" questions "heavily depend on correctly generating anonymized entities". Synthetic noise was applied at target WER 20-100%, built from substitution rules mined from more than 500k ASR transcripts. GPT-4o stayed above a score of 6 (on a 10-point LLM-judge scale) even at 100% target WER. LLaMA-3.1-8B and Phi-3-small degraded substantially, and the gap "widens significantly as noise increases" — [ELITR-Bench, arXiv 2403.20262](https://arxiv.org/html/2403.20262v3). The achieved effective WER was lower than the target. Speaker-label noise was not manipulated.
- MISeD (Google, EMNLP Findings 2024): information-seeking dialogs over meeting transcripts, with human-annotated *attribution spans*. Fine-tuning on it improved attribution quality, and response quality was comparable to fully manual data — [arXiv 2405.01121](https://arxiv.org/abs/2405.01121v3); [data](https://github.com/google-research-datasets/MISeD). It is a resource for grounding evaluation, not a speaker-label ablation.
- Classic spoken QA: on Spoken-SQuAD, average F1 fell from 75.42 (text documents) to 55.4 (ASR transcripts, WER ~22.7%) — [Li et al., arXiv 1804.00320](https://arxiv.org/pdf/1804.00320). These are pre-LLM extractive models, so it is an upper bound on sensitivity.
- Sentence encoders: pre-trained neural sentence encoders were found fairly robust to ASR-like word substitutions on STS, while unweighted word-vector averages degraded quickly — [Voleti et al., arXiv 1811.07021](https://arxiv.org/pdf/1811.07021). This is older (2018) work.

**Diarization errors downstream**
- A student course project (Princeton COS514) reportedly found that summary quality dropped as ASR errors increased, but that "speaker diarization errors did not affect the quality". This comes from a search snippet only, and the PDF returned 404 — [search-result source](https://tanushreebanerjee.github.io/pdfs/cos514_final.pdf). **Unverified, single, non-peer-reviewed.**
- A vendor blog (Gladia, an ASR vendor, so biased) says diarization errors propagate into wrong action-item owners and per-speaker sentiment, and that "no published research yet quantifies how upstream errors compound" — [Gladia blog](https://gladia.io/blog/meeting-transcription-common-mistakes-what-meeting-assistant-builders-get-wrong). This is opinion, not measurement.

### Inferences
- For the requester's task (matching a viewer comment to a passage), speaker labels matter mainly in three places: (a) comments that name or refer to a person ("the host asked...", "when Kim said..."), (b) the LLM coder's interpretation of a shown passage (who is asserting vs. asking), and (c) question/answer boundary detection. Topic retrieval itself depends little on speaker labels. Summarization evidence (Kirstein) suggests that capable LLMs rarely misattribute when labels are right. The reverse risk, wrong labels causing confidently wrong "X said Y" readings, is plausible but unmeasured. The observed 14/33 questions attributed to the guest is exactly this risk.
- The ELITR-Bench result implies that a local ~27B model is likely intermediate in robustness. Word-level ASR noise matters more for small models than for speaker noise. This is extrapolated and model-specific.
- Given the soft-cue treatment of speaker tags, a defensible policy is: show the speaker label to the LLM coder only when confidence is high, or render it as a hedged role ("Host?" / "Guest?"). Never put a low-confidence name into retrieval text (see Q3).

### Gaps
- No controlled study found that varies speaker-label accuracy (DER or WDER) and measures retrieval recall or LLM QA accuracy on meetings or interviews. QMSum, MeetingBank, AMI and MediaSum papers did not ablate speaker labels for retrieval.
- TREC Podcasts / Spotify segment retrieval runs did not use speaker labels, as far as found.
- No 2025-2026 benchmark found that injects diarization (as opposed to WER) noise into LLM meeting QA. The COS514 claim could not be verified.

## Q2. Do clean sentence boundaries and punctuation improve retrieval or LLM grounding?

### Takeaway
Older spoken-language-processing literature consistently shows that segmentation quality affects summarization, parsing, IE and MT. Direct evidence that punctuation or sentence boundaries change *dense-retrieval or LLM-grounding* outcomes is essentially absent. The 2024-2025 paragraph-segmentation work explicitly does not measure downstream tasks. For this pipeline, boundary quality matters mostly through what it does to unit construction (fragmentary units, cross-speaker splits) rather than through the embedding of punctuation itself.

### Cited Findings
- A classic overview (Favre et al., IEEE SPM 2008): many NLP techniques "suffer when applied to speech recognition output because they assume the presence of reliable punctuation and structure". Proper sentence segmentation is "essential to achieve good performance" for summarization, and optimizing comma prediction for IE improved entity/relation extraction — [Favre et al. 2008](https://pageperso.lis-lab.fr/benoit.favre/papers/favre_spm2008.pdf). Pre-neural era.
- In speech translation, the sentence segmentation of ASR output strongly affects MT quality — [Matusov et al., IWSLT 2006](https://aclanthology.org/www.mt-archive.info/IWSLT-2006-Matusov.pdf). Old but often replicated. Numbers not extracted here.
- "Paragraph Segmentation Revisited" (Retkowski & Waibel, arXiv 2512.24517) introduces TEDPara (5,193 TED talks) and YTSegPara (YouTube, LLM-labelled). It claims paragraphs "support downstream tasks, including summarization and information retrieval" but provides **no** downstream evaluation — [arXiv 2512.24517](https://arxiv.org/html/2512.24517).
- Fu et al. (WNUT 2021) worked on punctuation restoration for noisy phone-conversation ASR. Their motivation is that unpunctuated ASR output "limits the performance of downstream NLP tasks". Their evaluation is punctuation F1, not downstream — [ACL Anthology 2021.wnut-1.19](https://aclanthology.org/2021.wnut-1.19).
- In LLM conversational memory, turn-level units underperform because "the retrieved contexts can be fragmentary and fail to encompass the complete request-related information flow" (SeCom; numbers in Q3). This is indirect evidence that fragmentary units, which bad boundaries produce, hurt both retrieval and grounding — [SeCom, arXiv 2502.05589](https://arxiv.org/html/2502.05589v3).
- SeCom also found that compression-based denoising (LLMLingua-2 at 75% compression) of memory units improved retrieval. Removing it cost 9.46 GPT4Score points on LOCOMO and 1.30 on Long-MT-Bench+ — [SeCom](https://arxiv.org/html/2502.05589v3). Spoken filler and disfluency act as noise for retrieval. This is single-source, on chat data rather than ASR.

### Inferences
- The requester's 14 speaker-boundary spans without terminal punctuation ("So is that a five-alarm fire to" | "you?") are mainly a *unit-integrity* problem. A 1-3-word fragment like "you?" is an unmatchable retrieval unit, and it can be displayed under the wrong speaker. Repairing it by merging the tail fragment into the previous sentence (or the head into the next), as covered in areas A/B, likely helps more than restoring punctuation in general.
- Modern embedders tolerate missing punctuation within a sentence reasonably well (Voleti et al. 2018 for word noise; no direct punctuation study found). BM25 is unaffected by punctuation after tokenization. The measurable gain from clean boundaries should therefore show up in the fraction of comments whose gold passage is a fragment, not in average cosine.
- For LLM grounding (the coder reading the passage), well-formed sentences with correct speaker attribution plausibly reduce misreadings such as a question read as the guest's assertion. This is untested and a reasoned extrapolation from Q1 (Geng 2022 speaker confusion).

### Gaps
- No study found that measures dense or BM25 retrieval quality on the same spoken corpus with and without punctuation or sentence restoration.
- No LLM-era study found on punctuation or segmentation effects on QA or grounding over ASR transcripts.
- A cheap local test would fill this: compare comment-linking accuracy on the ~100 labelled links before and after fragment repair.

## Q3. Chunking for dialogue: turn-based, Q-A pairs, speaker-turn windows, topic segments, fixed windows, sentence-window, small-to-big; measured comparisons on conversational data; whether to put speaker names in chunk text

### Takeaway
The best measured evidence on conversational data (SeCom, ICLR 2025; LongMemEval, ICLR 2025) says that topically coherent multi-turn segments beat both single turns and whole sessions for retrieval-plus-QA. Single turns beat whole sessions, and adding extracted facts to the original text (not replacing it) raises recall. MeetingQA shows that answers often span several turns and speakers, which argues for Q-A units that include the follow-up turns, not single turns. No measured evidence was found for or against including speaker names in the text that is embedded or BM25-indexed.

### Cited Findings
**Unit granularity on conversational data**
- SeCom (Pan et al., Microsoft, ICLR 2025), GPT4Score on QA with retrieved memory:

  | Unit | LOCOMO BM25 | LOCOMO MPNet | Long-MT-Bench+ BM25 | Long-MT-Bench+ MPNet |
  |---|---|---|---|---|
  | Turn-level | 65.58 | 57.99 | 82.85 | 84.91 |
  | Session-level | 63.16 | 51.18 | 81.27 | 73.38 |
  | Segment-level (SeCom) | 71.57 | 69.33 | 86.67 | 88.81 |

  Segments come from an LLM-based conversation segmenter that also scored well on DialSeg711, TIAGE and SuperDialSeg. Single turns are "fragmentary"; sessions mix topics — [SeCom, arXiv 2502.05589](https://arxiv.org/html/2502.05589v3); [ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/e56f394bbd4f0ec81393d767caa5a31b-Abstract-Conference.html). The data is user-assistant chat, not ASR interviews.
- LongMemEval (Wu et al., ICLR 2025): decomposing sessions into rounds (one user-assistant exchange, i.e. a Q-A pair) "significantly enhances reading performance with GPT-4o". Compressing values into summaries or facts *hurt* QA through information loss, except on multi-session reasoning. With the Stella-v5 retriever and round values, key = value + extracted facts raised Recall@5 from 0.582 to 0.644 and Recall@10 from 0.692 to 0.784, and GPT-4o QA accuracy from 0.615 to 0.657. Averaged, that is +9.4% recall@k and +5.4% accuracy. Keys made of facts, keyphrases or summaries *alone* did not improve recall over the raw value. Llama-3.1-8B QA "drops sharply beyond 3k retrieved tokens", while GPT-4o kept improving past 20k — [LongMemEval, arXiv 2410.10813](https://arxiv.org/html/2410.10813v2).
- MeetingQA: 40% of answers are multi-span and 48% multi-speaker — [ACL 2023](https://aclanthology.org/2023.acl-long.837). A Q-A pair cut at the first answering turn will often miss part of the answer.
- ORDER (Pellet, Perez & Puren, arXiv 2609.17012, 2026) compares 14 chunking strategies on parliamentary transcripts, including hybrids that split agenda-item topic segments into speaker-turn windows. It uses LCS-based coverage recall@3 — [arXiv 2609.17012](https://arxiv.org/pdf/2609.17012). The PDF could not be text-extracted in this session, so **no numbers were verified**. Worth reading if parliamentary turn-window results matter.
- Vendor and tutorial guidance (Cohere docs and others) recommends turn-level chunks with speaker metadata and keeping Q-A pairs together for transcripts. This is advice without measurements — [Cohere chunking strategies](https://docs.cohere.com/page/chunking-strategies). Vendor source, not evidence.

**Late chunking and contextual retrieval re-evaluated (new since the earlier report)**
- Merola & Singh (ECIR 2025 workshop, arXiv 2504.19754) used 512-character fixed chunks on an NFCorpus subset. Contextual retrieval beat early chunking with Jina-v3 (NDCG@5 0.317 vs 0.303, ~+4.6%) and beat late chunking (0.309). Late chunking gave only +0.5% (Stella-v5) and +1.6% (Jina-v3) over early chunking, and **with BGE-M3 early chunking substantially outperformed late chunking**. Contextualization with an LLM needed ~20 GB VRAM — [arXiv 2504.19754](https://arxiv.org/html/2504.19754). Single study on biomedical documents, not dialogue. It matters here because BGE-M3 is the requester's current embedder.

**Speaker names in chunk text**
- MediaSum's convention prefixes each turn with the speaker name for *generation* inputs — [arXiv 2103.06410](https://arxiv.org/pdf/2103.06410).
- No retrieval study was found that ablates "Name: text" vs bare text in embedded or BM25-indexed chunks. (Searched: meeting QA, podcast retrieval, conversational RAG.)

### Inferences
- Mapping to the requester's data: hQR has 46 turns over 287 spans, so turns average about 6 spans, and turns longer than 1-3 sentences are a natural middle granularity. The SeCom and LongMemEval pattern (segment ≥ round/turn > session; raw text plus derived keys > derived keys alone) suggests:
  1. a matching unit of 1-3 repaired sentences, never crossing a speaker change (the earlier report already recommends this);
  2. an additional *Q-A unit* (interviewer question + the start of the guest's answer, capped at ~100-150 tokens) indexed alongside, small-to-big style, so that comments reacting to a question ("great question about X") have a target;
  3. a shown unit that extends forward to cover multi-span answers (MeetingQA).
- Speaker names in the embedded text: names add identical tokens to every chunk from one speaker. For BM25 that lowers IDF for those tokens, but they still match comments that name a person. For dense embeddings of 5-35-token units, a "Name:" prefix is a non-trivial fraction of the text and pulls same-speaker chunks together, which hurts the margin and distinctiveness gating the requester relies on. Recommendation (inference, untested): keep speaker in metadata. Add it to BM25 only as a separate field, or add the name only when the comment mentions a person. Show it to the LLM coder.
- Using a fact or summary key *instead of* the raw span is contraindicated (LongMemEval). Adding it alongside as a second key is supported.

### Gaps
- No measured comparison found on *interview or podcast ASR* data of turn vs Q-A pair vs fixed window vs topic segment for retrieval. SeCom and LongMemEval are chat logs. ORDER (parliament) numbers are unverified.
- No evidence found on speaker-name inclusion in retrieval text.
- No Korean dialogue-chunking evidence.

## Q4. Implications for contextual-chunk prefixes on short chunks

### Takeaway
The requester saw contextual prefixes hurt when the prefix is longer than the chunk. That observation fits the available evidence: derived text helps as an *added* key but hurts when it dominates or replaces the original (LongMemEval). Contextual gains in independent replications are modest, about 4-5% on 512-character chunks. No published study directly tests prefix length against chunk length.

### Cited Findings
- Anthropic's recipe adds 50-100 tokens of context to chunks that are typically several hundred tokens long. Its gains are measured on document corpora — covered in the earlier report; [Anthropic](https://www.anthropic.com/news/contextual-retrieval).
- An independent replication on 512-character chunks found only about +4.6% NDCG@5 for contextual prefixes (Jina-v3), at substantial GPU cost — [Merola & Singh, arXiv 2504.19754](https://arxiv.org/html/2504.19754).
- Keys made only of derived text (summaries, facts, keyphrases) did not beat the raw value, while raw + facts did (+9.4% recall) — [LongMemEval](https://arxiv.org/html/2410.10813v2). This is the closest analogue to "prefix dominates chunk". Single source.
- Compression-based denoising, i.e. *removing* tokens from dialogue units, improved retrieval in SeCom — [arXiv 2502.05589](https://arxiv.org/html/2502.05589v3). Shorter, cleaner keys helped, which is the opposite direction to adding long prefixes.
- Practitioner guides recommend keeping generated context to about two sentences ("retrieval handles, not summarizing") — [understandingdata.com](https://understandingdata.com/posts/contextual-retrieval/). Blog-level source.

### Inferences
- With p50 spans of 55 chars (~12 tokens) and 50-100-token prefixes, the embedding mostly represents the prefix. In a single-video corpus, prefixes for neighbouring spans are near-identical, so cosine margins between neighbours collapse. That explains the requester's result.
- Better options, in order:
  1. repair and merge fragments first, then use units of ≥ ~40-60 tokens;
  2. if context is wanted, index it as a *separate* key (multi-vector: raw span vector + contextualized vector, score = max or weighted), not as a prefix;
  3. keep any prefix much shorter than the unit (≤ ~20-30% of tokens) and make it *local*, e.g. the interviewer's question that this answer responds to, rather than a global document summary;
  4. given the Merola result for BGE-M3, do not expect late chunking to substitute.
- Option 3's "question as context" is effectively the Q-A unit of Q3 and is grounded in the dialogue structure.

### Gaps
- No study found that sweeps prefix length relative to chunk length, or contextual retrieval on dialogue or transcript corpora.
- The ratio threshold above (≤ ~20-30%) is a heuristic, not a sourced number. Validate it on the ~100 labelled links.
