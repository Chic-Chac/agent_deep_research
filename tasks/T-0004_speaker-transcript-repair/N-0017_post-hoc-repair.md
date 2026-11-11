---
id: N-0017
type: note
task: T-0004
title: "Post-hoc repair of speaker-attributed transcripts (rules + LLM correction on existing sentence spans)"
status: final
created: 2026-11-03
tags: [transcript-repair, llm-correction]
---

# Post-hoc repair of speaker-attributed transcripts (rules + LLM correction on existing sentence spans)

Context recap (from requester, not sourced): 2-speaker 24-min interview, 287 spans; 14 spans end mid-sentence at speaker boundaries with halves on different speakers; 14/33 questions attributed to guest; interjections absorbed into other speaker's sentence; 27 spans <=3 words. Spans carry text, start/end, speaker_id, boundary, first/last word IDs; downstream labels reference span IDs.

## 1. Rule-based fragment merging: heuristics, failure modes, open-source implementations

### Takeaway
The most-used open-source rule is whisper-diarization's punctuation realignment: when the speaker changes at a word that is not sentence-final, the whole sentence (bounded by `.?!` and a max-words cap) is re-labelled to its majority speaker. This directly targets the "So is that a five-alarm fire to | you?" pattern. Published diarization post-processing otherwise works on time segments (minimum durations, gap merging) rather than on punctuation. I found no peer-reviewed evaluation of punctuation-based merging rules on their own.

### Cited Findings
- whisper-diarization (MahmoudAshraf97, BSD-2-Clause) gives each word the speaker of the diarization turn that contains the word's start time (an option uses the mid or end time instead). It then runs `get_realigned_ws_mapping_with_punctuation(word_speaker_mapping, max_words_in_sentence=50)`. The README says it uses "punctuation models to compensate for minor time shifts". — [GitHub repo](https://github.com/MahmoudAshraf97/whisper-diarization); code read from [helpers.py](https://raw.githubusercontent.com/MahmoudAshraf97/whisper-diarization/main/helpers.py)
- Exact algorithm (helpers.py), with sentence-final marks `.?!`:
  - The trigger is a word k where `speaker[k] != speaker[k+1]` and word k does not end in `.?!`.
  - The sentence start is found by walking left while the speaker stays the same and no sentence-final mark appears, up to `max_words`. The sentence end is found by walking right, ignoring speaker, until a sentence-final word.
  - If neither bound is a real sentence boundary within the cap, the sentence is skipped and nothing changes. Otherwise the mode speaker of the words in that range is computed. If its count is below half the words (`len//2`), nothing changes; otherwise every word in the range gets the mode speaker.
  - Sentences are then rebuilt with NLTK Punkt (`text_contains_sentbreak`), and a new sentence also starts on every speaker change. — [helpers.py](https://raw.githubusercontent.com/MahmoudAshraf97/whisper-diarization/main/helpers.py)
  - Implication: the rule decides by word count, and a mid-sentence speaker change is always treated as an error. Genuine interruptions inside an unpunctuated run get overwritten by whichever speaker has more words.
- Time-segment post-processing used in diarization systems:
  - Same-speaker segments are merged when the gap is below a threshold, and short segments are merged to reduce "flicker".
  - Segments shorter than a minimum duration are discarded, and speakers whose total talk time is below a threshold are pruned.
  - Morphological closing (dilate segments, then shrink) is used as a practical smoothing step.
  - Single-word utterances are sometimes filtered because one word is too short to carry speaker information.
  - Source: these findings come from a search-result summary of several arXiv papers and a tutorial, not checked line by line. Treat them as general practice, not specific parameter values. — [Speaker Diarization with Lexical Information (arXiv 2004.06756)](https://arxiv.org/pdf/2004.06756); [Can We Really Repurpose Multi-Speaker ASR Corpus for Speaker Diarization? (arXiv 2507.09226)](https://arxiv.org/pdf/2507.09226); [non-peer-reviewed tutorial: merging short segments](https://theneuralbase.com/whisper/learn/beginner/merging-short-segments/)
- Backchannels ("yeah", "yes", "oh", "okay", "uhhuh", "mhm", "[laughter]") are treated as independent utterances in some diarization corpora and pipelines. This supports keeping them as separate spans rather than merging them. — [arXiv 2507.09226](https://arxiv.org/pdf/2507.09226) (via search snippet; single source)
- Lexical evidence: most diarization errors occur at speaker transitions and in overlapping speech. That is where both rule and LLM corrections pay off. — [Paturi et al., Lexical Speaker Error Correction, Interspeech 2023 (arXiv 2306.09313)](https://arxiv.org/abs/2306.09313); [SEAL arXiv 2501.08421](https://arxiv.org/html/2501.08421v1)

### Inferences
- Proposed deterministic merge rule for the observed failure (an inference; no published evaluation found). Merge span A with the next span B when all of the following hold:
  - A has no terminal `.?!…`;
  - B starts lowercase or B is very short (≤3 words);
  - the gap is under roughly 300–500 ms;
  - speaker(A) ≠ speaker(B) and boundary == "speaker".
  
  Then relabel the merged group by majority (see Q2). This is essentially the whisper-diarization rule applied at span level.
- Guards against false merges. Do not merge when:
  - B is a closed-class backchannel ("yeah", "right", "mm-hm", "okay", "sure", "exactly") followed by A's speaker continuing;
  - A ends with an em-dash or "—" (an interruption or self-repair marker);
  - the gap is over about 1 s (trailing off);
  - A ends with a list connective ("and", "or", a comma) and B is a new speaker's full sentence of more than about 5 words.
  
  These thresholds are untested guesses and should be calibrated on the 287-span file by hand-labelling the 14 + 27 cases.
- The "there's— Would you support it?" inside a guest sentence is the reverse case: one span needs splitting. The split point is an em-dash followed by a capitalised question. Splitting creates new span IDs (see Q4).

### Gaps
- I found no peer-reviewed paper that quantifies punctuation-based merge or relabel rules on their own: no WDER before and after for whisper-diarization realignment. WhisperX's `assign_word_speakers` was not inspected.
- The parameter values for minimum turn duration and merge gap in pyannote-style pipelines were not verified from primary sources in this pass.

## 2. Re-assigning the speaker of a merged group; turn smoothing; backchannel handling

### Takeaway
The standard re-assignment is word-count majority with a ≥50% requirement (whisper-diarization). Other options are DiarizationLM's locality gating (accept relabels only on short runs of ≤5 words, keep long acoustic runs), "ghost speaker" absorption, and lexical/LLM cues. Role priors such as "the interviewer asks the questions" are not covered in the literature I found as an explicit rule. They are an easy, cheap check for the 14/33 misattributed questions but carry a real false-positive risk.

### Cited Findings
- Majority vote rule: mode speaker over the sentence's words, applied only when the mode holds at least `len//2` words. Otherwise the span is left unchanged. — [whisper-diarization helpers.py](https://raw.githubusercontent.com/MahmoudAshraf97/whisper-diarization/main/helpers.py)
- DiarizationLM's newer code adds `apply_locality_gated_speaker_transfer` (LG-TPST). It takes the LLM's per-word speaker labels and accepts a change only if the contiguous run of changed words is at most `max_local_span_words=5`. Longer runs keep the acoustic (original) speaker. — [google/speaker-id DiarizationLM utils.py](https://raw.githubusercontent.com/google/speaker-id/master/DiarizationLM/diarizationlm/utils.py) (repo Apache-2.0, last push 2026-10-06 per GitHub API)
  - Docstring: "Accepts LLM speaker modifications on short lexical boundary/backchannel spans (1 <= L <= max_local_span_words) while preserving the acoustic speaker anchor on long monologue spans".
  - Ghost-speaker absorption: if more than 2 speakers exist, speakers with ≤ `min(12, max(2, 1% of words))` words are relabelled to the previous valid speaker, or else the next one. This matters for 2-person interviews where the diarizer invents a third speaker.
  - For 2-speaker training data, full oracle supervision is kept and gating is not applied in the oracle builder (`get_locality_preserving_oracle_speakers`). — [utils.py](https://raw.githubusercontent.com/google/speaker-id/master/DiarizationLM/diarizationlm/utils.py)
- Acoustic + lexical fusion:
  - NVIDIA's contextual beam search (Park, Dhawan, Koluguri, Balam, ICASSP 2024) jointly decodes acoustic speaker probabilities with LLM lexical probabilities. It reports up to 39.8% relative improvement over the acoustic-only baseline. Applying it needs the diarizer's per-word speaker posteriors, not just final labels. — [arXiv 2309.05248 / ICASSP 2024](https://www.emergentmind.com/papers/2309.05248); [dblp](https://dblp.org/rec/conf/icassp/ParkDKB24)
  - SEAL (Amazon, arXiv 2501.08421, Jan 2025) shows that acoustic confidence passed to an LLM as categorical tokens ("low/medium/high") works better than raw probabilities. — [SEAL](https://arxiv.org/html/2501.08421v1)
- Lexical speaker error correction (Paturi et al., Interspeech 2023): a second-pass word-level speaker classifier using a language model plus the first-pass diarization labels. It gives 15–30% relative WDER reduction on RT03-CTS, Callhome American English and Fisher. Trained on Fisher only, it generalised to the other two. This is a non-generative, word-preserving design. — [arXiv 2306.09313](https://arxiv.org/abs/2306.09313)

### Inferences
- Recommended relabel cascade for a merged group (an inference):
  1. Duration-weighted majority (sum of word durations per speaker), not word count. Short backchannels then count less.
  2. Sentence-final "?" combined with a role prior. If the group is a question and >50% of its duration falls inside the guest's acoustic run, flag it rather than auto-flip, unless the timing evidence is weak (for example the question sits at a turn boundary or is ≤ N words).
  3. Otherwise leave the label as is.
  
  The role prior should be a soft feature. Guests do ask rhetorical or clarifying questions, so a hard "questions → host" rule would create new errors. No source quantifies this.
- Smoothing over the span sequence (HMM/Viterbi with a switch penalty, median filtering) amounts to "minimum turn duration". In a 2-speaker interview it will wrongly absorb genuine short answers ("Yes.", "No.") unless backchannel or answer tokens are exempted. LG-TPST's ≤5-word gate is the published analogue: allow changes on short runs, protect long runs.
- Backchannel policy: keep backchannels as separate spans with their own speaker. Do not drop them, because downstream labels may reference them. Do not attach them to the other speaker's sentence. Mark them with a flag (e.g. `is_backchannel`) so analyses can filter them.

### Gaps
- I found no paper that evaluates role priors (interviewer → questions) for speaker re-attribution in interviews.
- I found no published comparison of HMM/Viterbi versus median smoothing at the word or span level after ASR.

## 3. LLM-based correction: methods, gains, open models, licenses, local feasibility

### Takeaway
Fine-tuned LLM correctors give large relative WDER reductions on telephone speech: DiarizationLM 44.9–55.5% (PaLM 2-S), SEAL 24–43%, Efstathiadis et al. ensemble about 29–57% deltaCP. Zero-shot general LLMs degraded results in two independent studies. They delete or alter text and roughly double WDER, unless outputs are constrained or post-aligned. The best locally runnable option in late 2026 is DiarizationLM-Gemma-4-E4B-v1 (Apache-2.0 per its card, Q4 GGUF about 5.3 GB). The Llama-3-8B Fisher-v2 adapter is the Google-published alternative.

### Cited Findings
- **DiarizationLM (Google; Wang, Huang, Zhao, Clark, Xia, Liao; Interspeech 2024; arXiv 2401.03506)**
  - Versions: v1 is from 2024-01-07; the latest is v13, dated 2026-10-11. — [arXiv abs](https://arxiv.org/abs/2401.03506)
  - Prompt format: `<spk:1> word word <spk:2> word ...`, with the suffix `" --> "`.
  - Long inputs are split by recursive binary partitioning to fit within 4,096 tokens.
  - Results: fine-tuned PaLM 2-S reduces WDER by 55.5% relative on Fisher (5.32% → 2.37%, cpWER 16.93%) and by 44.9% on Callhome (→ 4.25%, cpWER 20.22%).
  - Zero-shot PaLM 2-S made WDER worse (5.32% → 11.96%). Zero- and one-shot outputs "often delete large chunks of hypothesis text", so fine-tuning is needed. — [arXiv HTML](https://arxiv.org/html/2401.03506); [search snippet of paper](https://arxiv.org/html/2401.03506)
- **google/DiarizationLM-8b-Fisher-v2**
  - Base: Llama 3 8B with a LoRA adapter (rank 256, 671M trainable parameters), trained for 28,800 steps (about 9 epochs) on Fisher.
  - Limits: max input about 6,000 characters, 4,096-token sequence length.
  - Results on Fisher test: WDER 5.32% → 3.28%, cpWER 21.19% → 18.37%.
  - Usage: input ends with `" --> "`; requires the `diarizationlm` package. — [HF model card](https://huggingface.co/google/DiarizationLM-8b-Fisher-v2)
  - The card summary I retrieved did not state the license. As a Llama 3 derivative it is presumably under the Meta Llama 3 Community License (inference, unverified).
  - Community GGUF quantisations exist (mradermacher/DiarizationLM-8b-Fisher-v2-GGUF). There is also a 13B Fisher-v1 (Llama 2) model. — [HF API listing](https://huggingface.co/api/models?search=DiarizationLM)
- **DiarizationLM-Gemma-4-E4B-v1** (diarizers-community on HF; listed in the google/speaker-id README as "multi-domain, best overall")
  - Base and license: Gemma 4 E4B with a merged LoRA (rank 256), Apache 2.0.
  - Training: 71,825 prompt/completion pairs. Fisher contributed 51,063; Callhome, ICSI and AMI contributed 20,762. It uses locality-preserving oracle supervision (LG-TPST).
  - Reported results:

    | Dataset | WDER | cpWER |
    |---|---|---|
    | Fisher (USM + Turn-to-Diarize front end) | 2.99% | 17.62% |
    | Callhome | 4.92% | 20.69% |
    | ICSI | 14.10% | 43.32% |
    | AMI | 14.89% | 39.57% |

    The improvement column on the card is ambiguous: −2.33 on Fisher matches 5.32 → 2.99 absolute. Gains on ICSI and AMI are small (−0.6 / −0.79).
  - Memory: about 16 GB at bf16; Q4_K_M GGUF about 5.3 GB. Runs via transformers, llama.cpp or Ollama.
  - **Single source (community model card); the numbers have not been independently replicated.** — [HF model card](https://huggingface.co/diarizers-community/DiarizationLM-Gemma-4-E4B-v1); [google/speaker-id DiarizationLM README](https://github.com/google/speaker-id/tree/master/DiarizationLM)
- **Efstathiadis et al., "LLM-based speaker diarization correction" (arXiv 2406.04927; journal version "A generalizable approach", Speech Communication 2025)**
  - Models: Mistral-7B-Instruct-v0.2 fine-tuned on Fisher (11,699 conversations) with TPST-built oracle targets. Separate models were trained on AWS, Azure and WhisperX (Whisper large-v2 + pyannote 3.1.1) transcripts, then merged with TIES-Merging.
  - The ensemble reduced deltaCP by 32% (AWS), 57% (Azure) and 29% (WhisperX) on Fisher. On out-of-domain PriMock57 medical consultations it reduced deltaCP by 39–55%.
  - Failure modes:
    - Corrections are ASR-specific (a model trained on one ASR did little on another).
    - Models turn "conservative" under an unfamiliar error distribution.
    - Weaker results on unseen GCP transcripts; English only.
    - Zero-shot Mistral 7B increased deltaCP error by 1677% on AWS transcripts; Mixtral 8x7B zero-shot also degraded results.
  - Weights released at HF bklynhlth; integrated in the OpenWillis library. — [arXiv 2406.04927v2](https://arxiv.org/html/2406.04927v2); [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0167639325000391)
  - License of the weights was not verified (the fetch only reported arXiv's paper license).
- **SEAL (Amazon, arXiv 2501.08421, Jan 2025)**
  - Model: Mistral-7B-Instruct-v0.2 with QLoRA (rank 64).
  - Input: EEND speaker scores converted to "low/medium/high" acoustic tokens per word or turn.
  - Constrained decoding: at each step the LLM may only emit the next input word or a speaker label, so output stays word-for-word identical to the input.
  - Results: 24–43% relative speaker-error reduction across Fisher, Callhome and RT03-CTS versus first-pass diarization; outperforms a DiarizationLM-style baseline.
  - No code released in v1. Needs diarizer posteriors, which a span-only repair pipeline may not have. — [arXiv HTML](https://arxiv.org/html/2501.08421v1)
- **Contextual beam search (NVIDIA, ICASSP 2024)**: acoustic + LLM joint beam search, up to 39.8% relative improvement. Needs acoustic posteriors. — [emergentmind / arXiv 2309.05248](https://www.emergentmind.com/papers/2309.05248)
- **Iterative LLM correction for French clinical interviews (arXiv 2603.00086, Feb 2026, revised May 2026)**
  - Method: Qwen3-Next-80B in multi-pass alternating "speaker recognition" and "word recognition" passes, preserving original words and timestamps.
  - Results: statistically significant WDER reduction on suicide-prevention calls (n=18, p<0.05). Stable, i.e. no gain, on neurosurgery consultations (n=10). RTF 0.32, zero output failures.
  - Small n; an 80B model is not feasible on a 16 GB GPU. — [arXiv 2603.00086](https://arxiv.org/abs/2603.00086)
- **Deonise, Bajwa et al., "Speaker-attributed meeting transcription refinement with constrained open-weight language models"** (Future Generation Computer Systems, 2026)
  - Pipeline: pyannote + Whisper + Qwen-family instruct LLMs of several sizes, with constrained JSON post-processing.
  - Zero-shot prompt is instructions + input JSON; one-shot adds an example showing how wrongly assigned fragments are reassigned.
  - Quantitative results could not be retrieved (ScienceDirect and ResearchGate returned 403). — [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0167739X26002827); [ResearchGate](https://www.researchgate.net/publication/406697008_Speaker-attributed_meeting_transcription_refinement_with_constrained_open-weight_language_models)
- **Interactive in-meeting speaker correction (arXiv 2509.18377, v3 Sep 2026)**: human-in-the-loop. LLM summaries help the user spot speaker errors; feedback updates the transcript and enrols speakers online. On AMI headset: −31.99% DER versus a streaming baseline and −52.68% speaker substitution error. — [arXiv 2509.18377](https://arxiv.org/abs/2509.18377)
- **MLC-SLM 2025 (Interspeech 2025 challenge)**
  - Task II (diarization + ASR) top systems were pipelines (3D-Speaker diarization + LLM-based ASR), not post-hoc speaker-label correctors.
  - TEA-ASLP: tcpWER 17.49%; DKU 18.08%; a unified speech LLM 27.25%.
  - I found no MLC-SLM system that does post-hoc LLM speaker re-attribution over text. — [TEA-ASLP arXiv 2507.18051](https://arxiv.org/html/2507.18051v1); [emergentmind topic summary](https://www.emergentmind.com/topics/mlc-slm-2025-challenge)
- **End-to-end alternatives (not post-hoc)**: Speaker-Reasoner (arXiv 2604.03074, Apr 2026) is an end-to-end speech LLM with multi-turn temporal reasoning (AliMeeting, AISHELL-4). Diarization-aware multi-speaker ASR via LLMs is arXiv 2506.05796. Neither applies to fixing existing spans without re-running audio. — [2604.03074](https://arxiv.org/abs/2604.03074); [2506.05796](https://arxiv.org/abs/2506.05796)

### Inferences
- VRAM on a 16 GB RTX 5060 Ti (inference from sizes):
  - Gemma-4-E4B: bf16 (about 16 GB on the card) will not fit together with KV cache. Q8 or Q4 GGUF (about 5.3 GB at Q4) fits comfortably.
  - Llama-3-8B Fisher-v2: bf16 is about 16 GB and too tight; 8-bit (about 9 GB) or 4-bit GGUF fits.
  - Context: the 6,000-character input cap means a 24-minute interview (median span 55 characters, 287 spans, roughly 20–30k characters) needs about 4–6 windows. Use overlapping windows and resolve labels at window edges.
  - Hardware caveat: Blackwell (sm_120) support in the chosen runtime should be checked; not verified here.
- Domain mismatch risk: all open correctors are trained mainly on Fisher (unpunctuated, lowercased telephone speech, 8 kHz-era ASR errors). The requester's spans are punctuated Whisper-style text. Efstathiadis et al. show corrections are ASR-specific, so expect smaller gains than reported. Normalise the input (lowercase, strip punctuation) to resemble the training format. The DiarizationLM repo has `normalize_text`.
- Zero-shot frontier LLMs (GPT-4/Claude class): the only quantified zero-shot results found are PaLM 2-S and Mistral/Mixtral, all negative. If using Claude or GPT zero-shot, constrain the output to per-span speaker labels or edit ops, never rewritten text. Then apply TPST or LG-TPST, or accept only ≤5-word relabels. No published WDER for Claude or GPT-4 zero-shot speaker re-attribution was found.
- Cheapest high-value path for the observed defects: deterministic rules (Q1/Q2) fix the 14 split sentences and the ghost/backchannel issues. An LLM pass restricted to the question-attribution check (14/33) and to splitting absorbed interjections, emitting labels only and gated by locality, handles the rest.

### Gaps
- No independent replication of the Gemma-4-E4B card's numbers; the release date was not captured.
- No published zero-shot evaluation of Claude or GPT-4/5 for speaker re-attribution with WDER found. The FGCS 2026 Qwen paper's numbers were not accessible.
- Licenses of the bklynhlth (Efstathiadis) weights and of google/DiarizationLM-8b-Fisher-v2 were not confirmed from model cards.
- No results found for 2-person interview or podcast domains, which differ from telephone conversations and meetings.

## 4. Keeping word/timestamp alignment and span IDs stable after correction

### Takeaway
Treat the LLM or rules as producing only per-word speaker labels. Never accept new text. Map the output back onto the original word IDs with TPST: a Levenshtein word alignment, then a Hungarian assignment of speaker labels. LG-TPST additionally gates the changes. Rebuild spans from word IDs and emit an explicit old→new span map. Constrained decoding (SEAL) is the alternative when you control the decoder.

### Cited Findings
- `transcript_preserving_speaker_transfer(src_text, src_spk, tgt_text, tgt_spk)` in google/speaker-id: aligns the LLM output words to the original hypothesis words by edit-distance alignment, then solves the speaker-label permutation with the Hungarian algorithm. It returns one speaker label per *original* word, so the original words (and therefore word IDs and timestamps) are kept. — [utils.py](https://raw.githubusercontent.com/google/speaker-id/master/DiarizationLM/diarizationlm/utils.py); [arXiv HTML](https://arxiv.org/html/2401.03506)
- `transfer_llm_completion(llm_completion, hyp, max_local_span_words=None|5)` is the end-to-end helper. It strips the suffix, extracts text and speakers, applies TPST or LG-TPST against the hypothesis, and re-renders the diarized text using the hypothesis words. — [utils.py](https://raw.githubusercontent.com/google/speaker-id/master/DiarizationLM/diarizationlm/utils.py)
- The DiarizationLM code comments note that the TPST step "can arguably be skipped" only under the assumption that the LLM doesn't change the text much. Zero-shot LLMs violate this by deleting text, which is why TPST matters. — [utils.py](https://raw.githubusercontent.com/google/speaker-id/master/DiarizationLM/diarizationlm/utils.py); [arXiv HTML](https://arxiv.org/html/2401.03506)
- SEAL's constrained decoding restricts each step to the next input word or a speaker label, which guarantees verbatim output. — [SEAL](https://arxiv.org/html/2501.08421v1)
- Efstathiadis et al. also used TPST to build oracle training targets. — [arXiv 2406.04927v2](https://arxiv.org/html/2406.04927v2)
- The 2026 French clinical system explicitly preserves original words and timestamps across iterations. — [arXiv 2603.00086](https://arxiv.org/abs/2603.00086)

### Inferences (design; no source found for span-ID versioning in this literature)
- Pipeline:
  1. Flatten spans to words (w1…wN) with their speaker labels.
  2. Run rules and/or the LLM to produce new per-word speakers, plus optional split or merge boundary operations keyed by word ID.
  3. TPST or LG-TPST back to word IDs.
  4. Rebuild spans as maximal runs of (same speaker, same sentence).
  5. Emit `span_map` entries:

     ```
     {old_span_id: [new_span_ids], op: keep|relabel|merge|split, evidence: rule_id/llm, confidence}
     ```

- Make new span IDs deterministic from content, e.g. `s_{first_word_id}_{last_word_id}` such as `s_w380_w393`. Then unchanged spans keep their IDs, and merges or splits get predictable IDs.
- Keep a version number and both old and new span tables. Downstream labels are remapped:
  - relabel: same ID;
  - merge: union of the old labels, flagged for review;
  - split: the label goes to the part that overlaps most in words, or is duplicated and flagged.
- For LLM prompting with Claude or GPT, give each span an index and ask only for JSON edit operations. Examples: `{"op":"relabel","span":"s_w380_w393","speaker":"host"}`, `{"op":"merge","spans":[a,b]}`, `{"op":"split","after_word":"w391"}`. Validate that the referenced IDs exist and reject anything else. This avoids text drift altogether.

### Gaps
- I found no published standard for versioned span-ID mapping in diarization post-processing. The above is engineering inference.

## 5. Per-method summary (version/date, license, local feasibility, expected gain, cost, integration risk)

### Takeaway
Rules first: zero cost and directly target the observed defects. Then optionally a label-only LLM pass gated by TPST or LG-TPST. Fine-tuned open correctors are feasible on 16 GB with quantisation, but carry a domain mismatch.

### Cited Findings
- whisper-diarization punctuation realignment
  - License: BSD-2-Clause.
  - Gain: not quantified in any source found.
  - Cost/risk: CPU-only; it overwrites genuine mid-sentence interruptions by majority. — [repo](https://github.com/MahmoudAshraf97/whisper-diarization); [helpers.py](https://raw.githubusercontent.com/MahmoudAshraf97/whisper-diarization/main/helpers.py)
- DiarizationLM code (TPST, LG-TPST, metrics WER/WDER/cpWER/SpkCntMAE)
  - License: Apache-2.0; active as of 2026-10.
  - Note: "NOT an official Google product". — [repo README](https://github.com/google/speaker-id/tree/master/DiarizationLM); [utils.py](https://raw.githubusercontent.com/google/speaker-id/master/DiarizationLM/diarizationlm/utils.py)
- DiarizationLM-8b-Fisher-v2 (Llama 3 8B LoRA)
  - Gain: −38% relative WDER on Fisher (5.32 → 3.28).
  - Feasibility: fits 16 GB only when quantised.
  - Limits: 6k-character input cap; Fisher domain. — [HF](https://huggingface.co/google/DiarizationLM-8b-Fisher-v2)
- DiarizationLM-Gemma-4-E4B-v1
  - License: Apache-2.0.
  - Gain: Fisher WDER 2.99%, Callhome 4.92%.
  - Feasibility: Q4 about 5.3 GB.
  - Single-source community card. — [HF](https://huggingface.co/diarizers-community/DiarizationLM-Gemma-4-E4B-v1)
- Efstathiadis ensemble (Mistral 7B TIES)
  - Gain: −29% to −57% deltaCP.
  - Risk: ASR-specific behaviour; license unverified. — [arXiv 2406.04927v2](https://arxiv.org/html/2406.04927v2)
- SEAL
  - Gain: 24–43% relative.
  - Availability: no code; needs acoustic scores. — [arXiv 2501.08421](https://arxiv.org/html/2501.08421v1)
- Lexical Speaker Error Correction
  - Gain: 15–30% relative WDER.
  - Design: word-preserving classifier; no LLM generation. — [arXiv 2306.09313](https://arxiv.org/abs/2306.09313)
- Zero-shot general LLMs without constraints
  - Results: WDER worse (PaLM 2-S 5.32 → 11.96; Mistral zero-shot deltaCP +1677%).
  - Risk: high unless restricted to labels or edit ops plus TPST or gating. — [DiarizationLM](https://arxiv.org/html/2401.03506); [Efstathiadis](https://arxiv.org/html/2406.04927v2)

### Inferences
- Expected gain on the requester's data cannot be predicted from these benchmarks, which are telephone and meeting audio with Fisher-trained models. Hand-label the 287 spans (or at least the roughly 75 flagged spans: 14 + 33 + 27, with overlap) as a mini test set and measure span-level speaker accuracy before and after each stage.

### Gaps
- No benchmark on interview or podcast audio with punctuated Whisper-style spans.
- No cost or latency numbers for DiarizationLM models on consumer GPUs.
