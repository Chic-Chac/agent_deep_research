---
id: N-0013
type: note
task: T-0003
title: "ASR transcript sentence re-segmentation and punctuation restoration (Korean + English)"
status: final
created: 2026-11-03
tags: [sentence-segmentation, punctuation-restoration, korean]
---

# ASR transcript sentence re-segmentation and punctuation restoration (Korean + English)

Scope: post-processing upstream ASR spans (start/end s, sentence-boundary flag, optional speaker id) into sentence-like units before re-chunking for embedding linking. Checked 2026-11-03. PyPI metadata pulled live from pypi.org JSON API on that date. "KO" marks Korean-specific evidence.

## Q1. Korean sentence splitting (kss, kiwipiepy, others): versions, Py3.12, accuracy on spoken/unpunctuated text

### Takeaway
Both kss and kiwipiepy split on Korean sentence-final endings (eomi), not just punctuation, so they work on unpunctuated text. kiwipiepy 0.24.1 is the easier pin for Python 3.12: it ships abi3 wheels, is actively released (Oct 2026), and returns char offsets plus EF (final-ending) POS tags. kss 6.0.6 led on its own 2022 benchmark, but it ships no Linux wheel and pulls in a heavy dependency set. Nobody has published an evaluation on ASR transcripts.

### Cited Findings
- **kss**: latest 6.0.6, uploaded 2025-11-13. License BSD-3-Clause, requires_python ">=3". The only wheel is `cp310-macosx_14_0_arm64`, so Linux/Py3.12 installs build from the sdist. Hard deps include `emoji==1.2.0`, pecab, networkx, scipy, koparadigm, kollocate, bs4, cmudict, and even `pytest`. — [PyPI JSON](https://pypi.org/pypi/kss/json)
- kss 6.0 was released 2024-04-27 and became a multi-function toolkit (split_sentences, correct_spacing, normalize, g2p, ...). `split_sentences(text, backend=..., num_workers, strip, ignores)` accepts backends 'mecab', 'pecab', 'punct' and 'fast'. Mecab is no longer auto-installed and must be installed manually (e.g. python-mecab-kor) "to use Kss much faster". — [kss README (GitHub main)](https://github.com/hyunwoongko/kss) (raw README fetched)
- kss README example on unpunctuated spoken-style review text: "…맛있었어요 다만, … 했답니다 강남역 …" splits correctly at -요 and -답니다 with no periods. This shows it splits on endings, not punctuation. — [kss README](https://github.com/hyunwoongko/kss)
- kss backend semantics (4.x docs): 'auto' tries mecab → konlpy Mecab → pecab → punct. 'punct' = "split sentences only near punctuation marks" (useless for unpunctuated ASR). — [kss 4.5.4 PyPI description](https://pypi.org/project/kss/4.5.4/)
- **KO benchmark (kss 4.2.0 vs Kiwi 0.14.1, 2022)**: datasets were blogs_lee (170 sents), blogs_ko (346), sample (41), tweets (178), wikipedia (326), nested (91), and v_ending (30, "difficult eomi segmentation, various dialect sentences"). The average Exact Match scores were Kss-mecab 0.805, Kss-pecab 0.801, Kiwi 0.662, regex baseline 0.453. The average Normalized F1 scores were Kss-mecab 0.851, Kss-pecab 0.850, Kiwi 0.782, baseline 0.561. On v_ending, EM was Kss 0.367, Kiwi 0.200, baseline 0.000, and NF1 was Kss 0.482, Kiwi 0.342. On tweets, NF1 was Kss 0.814 and Kiwi 0.791. Koalanlp backends were all worse than Kiwi (avg NF1 0.52-0.67). — [kss 4.5.4 PyPI description, benchmark section](https://pypi.org/project/kss/4.5.4/)
- The kss author's caveats: the F1 (dice) metric "gives huge advantages to splitters which separate sentences too finely". NF1 = F1 * min(1, len(gold)/len(pred)) penalizes over-splitting. The benchmark was built by the kss author, so it is partisan, and Kiwi has had many releases since 0.14.1. — [same](https://pypi.org/project/kss/4.5.4/)
- **kiwipiepy**: latest 0.24.1, uploaded 2026-10-20, Apache-2.0, requires_python >=3.9. Wheels are `cp39-abi3` (manylinux2014 x86_64/aarch64, macOS, Windows) plus `cp314t`, so Python 3.12 is covered by abi3. Deps: `kiwipiepy_model>=0.24,<0.25`, numpy, tqdm. — [PyPI JSON](https://pypi.org/pypi/kiwipiepy/json)
- `Kiwi.split_into_sents(text, ..., return_tokens=False)` returns `Sentence(text, start, end, tokens)`. The README example splits the unpunctuated "여러 문장으로 구성된 텍스트네 이걸 분리해줘" at the final ending 네 (token tagged `EF`). Kiwi also has `Kiwi.space()` (spacing correction) and `Kiwi.glue(text_chunks, return_space_insertions=...)`, which decides whether to insert a space when joining chunks. glue is directly useful for joining ASR spans. — [kiwipiepy README](https://github.com/bab2min/kiwipiepy)
- A known Kiwi issue: it split wrongly when a particle followed a terminal ending, and the logic was later fixed. — [kiwi issue #101](https://github.com/bab2min/kiwi/issues/101) (search snippet only; not opened)
- **pecab** (kss pure-Python backend): 1.0.8, 2022-12-23, Apache-2.0 (unmaintained since 2022). **python-mecab-ko** 1.3.7, 2024-07-14, BSD-3, wheels cp37-cp312 (has a cp312 wheel). — [PyPI pecab](https://pypi.org/pypi/pecab/json), [PyPI python-mecab-ko](https://pypi.org/pypi/python-mecab-ko/json)
- **SaT (wtpsplit)** also covers Korean (see Q3). KO: on the 3-layer models, UD-Korean (Kaist) F1 was SaT+SM 100.0 and SaT 99.6, against Llama-3-8B 99.9 and spaCy-M 99.8. On the harder OPUS100-ko set, F1 was SaT 72.4, SaT+SM 78.9, SaT+LoRA 81.8, WtP-PUNCT 81.6, Llama-3-8B 70.3 and spaCy-DP 46.6. — [SaT paper, Table 29 (arXiv 2406.16678v2)](https://arxiv.org/html/2406.16678v2)
- **baikal / koalanlp**: in the kss benchmark, all seven koalanlp backends (OKT, HNN, KMR, RHINO, EUNJEON, ARIRANG, KKMA) scored at or below Kiwi. koalanlp needs a JVM. — [kss 4.5.4 benchmark](https://pypi.org/project/kss/4.5.4/)

### Inferences
- For ASR Korean, a final-ending rule is the strongest cheap boundary cue: the last morpheme of a span has tag EF (from Kiwi `tokenize`/`split_into_sents(return_tokens=True)`). This covers -다/-요/-죠/-거든요/-네/-습니다 and similar. Connective endings (EC, e.g. -는데/-고/-서) mean "continue". Using that tag as a merge feature is simple and robust. In spoken Korean, -거든요/-는데요 sometimes act as clause-final, so allow a boundary at EF only, and treat a trailing 요 on EC (-는데요) as a weak boundary.
- Prefer kiwipiepy over kss for a uv-pinned Py3.12 project: it has a binary wheel, a small dependency set, and active maintenance. kss 6.0.6 likely installs from the sdist on Linux (it appears pure-Python, but this is UNVERIFIED), and its pinned `emoji==1.2.0` plus many deps risk conflicts. kss-pecab has the best published scores (partisan benchmark). It is worth a side-by-side on the 173 Korean spans if installation succeeds.
- Because Korean spans are already short (median 40 chars), the task is mostly MERGING over-split spans, not splitting. Run the splitter on concatenated windows of spans, then map sentence char offsets back to span boundaries (Kiwi returns start/end).

### Gaps
- No published accuracy of kss or Kiwi on Korean ASR output (unpunctuated, disfluent, or Whisper-punctuated). Benchmarks are blogs, tweets and wiki. There is no newer kss-vs-Kiwi comparison after 2022.
- Did not verify that `uv pip install kss==6.0.6` builds cleanly on Linux Py3.12.
- No Korean-specific evaluation of "ending-only" heuristic boundaries against gold transcripts was found.

## Q2. Punctuation restoration models: Korean coverage and reported F1

### Takeaway
Two locally runnable options exist for Korean. 1-800-BAD-CODE/xlm-roberta_punctuation_fullstop_truecase lists `ko` among its 47 languages but publishes no Korean metrics and was trained on news. whooray/koen_punctuation is a small Korean+English spoken-text model, Apache-2.0, with period F1 0.92 on an undisclosed eval set. oliverguhr fullstop-multilang does not cover Korean.

### Cited Findings
- **1-800-BAD-CODE/xlm-roberta_punctuation_fullstop_truecase**: Apache-2.0. Language metadata lists 47 languages including `ko` (also ja, zh). It predicts punctuation (incl. CJK `？，。、・`), truecasing and sentence boundaries. English F1: '.' 92.42, ',' 78.79, '?' 80.47, sentence boundary 99.75. No Korean metrics are shown. Trained on WMT News Crawl, ~1M lines per language, ~7 h on A100. The card warns it "may not perform well on conversational or informal data" and is "not of production quality". Run via `punctuators` with `PunctCapSegModelONNX.from_pretrained(...).infer(texts, apply_sbd=True)`. — [HF model card](https://huggingface.co/1-800-BAD-CODE/xlm-roberta_punctuation_fullstop_truecase) (raw README grepped: `- ko` present in language list)
- **punctuators** package: 0.0.7, 2024-06-08, requires_python >=3.6, no license in metadata (stale since mid-2024). — [PyPI JSON](https://pypi.org/pypi/punctuators/json)
- **oliverguhr/fullstop-punctuation-multilang-large**: covers EN/DE/FR/IT only (no Korean). MIT. Trained on Europarl (political speeches, SEPP-NLG). English F1 is '.' 0.948, '?' 0.890, ',' 0.819, ':' 0.575, '-' 0.425, macro 0.775. — [HF model card](https://huggingface.co/oliverguhr/fullstop-punctuation-multilang-large)
- **deepmultilingualpunctuation** (wrapper for the above): 1.0.1, 2022-02-11, MIT, unmaintained. — [PyPI JSON](https://pypi.org/pypi/deepmultilingualpunctuation/json)
- **KO: whooray/koen_punctuation**: fine-tuned from Alibaba-NLP/gte-multilingual-mlm-base, Apache-2.0, last modified 2025-03-10, 32 downloads. It is described as "Punctuation restoration for spoken language" (KO+EN). Eval-set F1: O 0.992, comma 0.827, period 0.922, question 0.832, exclamation 0.556, macro 0.826. The training/eval data is "None"/undisclosed. Usage: `pip install spokentxt-punctuation-restoration` (0.1.1, 2025-03-10, deps transformers>=4.49,<5 and torch>=2.1,<3, no license in metadata). — [HF model card](https://huggingface.co/whooray/koen_punctuation), [PyPI JSON](https://pypi.org/pypi/spokentxt-punctuation-restoration/json)
- Korean punctuation restoration in the literature is mostly for classical or historical texts (BERT, F1 0.84 punctuation, 0.96 spacing). It is not applicable to modern speech. — [arXiv 2312.11881](https://arxiv.org/abs/2312.11881v1)

### Inferences
- For re-segmentation we only need sentence-boundary decisions, not commas. A segmenter (SaT, Kiwi EF) is a better fit than a punctuation restorer. Punctuation models add value mainly for readability of the SHOWN unit.
- whooray/koen_punctuation is the only spoken-KO model found. Treat its metrics as unverified: the eval data is unknown and it has a tiny user base. The `transformers<5` pin may conflict with an environment using transformers 5.x.
- The xlm-roberta model's Korean support is nominal (no metrics, news domain). Test on a handful of spans before relying on it.

### Gaps
- No Korean F1 for the 1-800-BAD-CODE model. Did not find KoBERT/KoELECTRA-based spoken-Korean punctuation models with published metrics (search returned none beyond whooray).
- Did not verify that `punctuators` installs on Py3.12 with current onnxruntime.

## Q3. English (and multilingual) segmentation for ASR: SaT/wtpsplit, punctuators, spaCy/pySBD limits

### Takeaway
wtpsplit 2.2.2 (SaT, EMNLP 2024, MIT) is the best local option for both languages. It was trained with punctuation and casing corruption to emulate ASR. It covers 85 languages including Korean, runs at 3-12 layers (small on a 16 GB GPU, ONNX optional), and has a threshold knob plus TED/ASR-style LoRA adapters. Rule-based tools (pySBD, spaCy sentencizer, NLTK punkt) split on punctuation and do poorly on unpunctuated text.

### Cited Findings
- **wtpsplit**: latest 2.2.2, uploaded 2026-09-25, MIT, requires_python >=3.9, pure-Python `py3` wheel. Deps: transformers>=4.22.2, huggingface-hub, scikit-learn, skops, pandas, mosestokenizer. Extras: onnx-gpu, onnx-cpu, adapters. — [PyPI JSON](https://pypi.org/pypi/wtpsplit/json)
- wtpsplit supports 85 languages including Korean. The general recommendation is `-sm` models (e.g. `sat-3l-sm`; 12-layer for max quality). The `threshold` parameter makes segmentation more conservative when raised. LoRA modules cover "TED (i.e., ASR-style transcribed speeches) sentence styles in 81 languages". It also offers paragraph segmentation, ONNX inference, and length-constrained segmentation (Viterbi). English F1 ranges from 88.5 (sat-1l) to 97.4 (sat-12l-sm). — [wtpsplit GitHub](https://github.com/segment-any-text/wtpsplit)
- SaT paper (Frohmann et al., EMNLP 2024 main): "outperforms all baselines - including strong LLMs - across 8 corpora". It has a 3x speedup over WtP and a limited-lookahead mechanism for short sequences. — [arXiv 2406.16678](https://arxiv.org/abs/2406.16678)
- SaT+SM training corrupts data by lowercasing and removing all punctuation, which "aims to roughly emulate the output of an automatic speech recognition (ASR) system". It also uses random punctuation duplication, removal and spacing noise. The ASR evaluation uses TED2020 transcripts in 76 languages as unpunctuated, lowercased sentence pairs. — [SaT paper HTML](https://arxiv.org/html/2406.16678v2)
- SaT+SM set a new SOTA on the SEPP-NLG surprise test set (500 transcribed TED talks per language, 4 European languages, no casing or punctuation), beating the shared-task winners. The paper's TED (ASR-style) ablation column for 3-layer models reads WtP 9.8-12.6, SaT 16.9-28.8, SaT+SM 28.4-41.7. This metric is a strict short-sequence measure, so low absolute values reflect the task's difficulty. — [SaT paper HTML](https://arxiv.org/html/2406.16678v2) (table parsed from HTML; column semantics as read, moderately confident)
- English per-corpus table: PySBD is far weaker on some sets (e.g. 37.4 on one column), while SaT+SM scored ~80-97 across columns. Rule-based Moses, spaCy and PySBD "split on punctuation characters, except for predefined exceptions". — [SaT paper HTML](https://arxiv.org/html/2406.16678v2)
- **pySBD** 0.3.4, last release 2021-02-11, MIT (unmaintained). It is rule-based and needs punctuation. — [PyPI JSON](https://pypi.org/pypi/pysbd/json), [SaT paper](https://arxiv.org/html/2406.16678v2)

### Inferences
- Recommended English path: SaT (`sat-3l-sm` or `sat-12l-sm`) over concatenated span text, keeping the existing ASR sentence flag as a prior. Whisper-family English output is usually already punctuated, so the ASR flag plus terminal punctuation may already be adequate, and SaT mainly fixes splits at commas and fillers. The same SaT model can be run on Korean as a second opinion to Kiwi EF.
- The SaT `threshold` should be tuned upward (more conservative), since the problem is over-splitting.

### Gaps
- No per-language Korean TED/ASR SaT scores were found in the paper tables (Korean appears only under UD and OPUS100).
- No head-to-head of SaT vs kss/Kiwi on Korean.

## Q4. Merging spans using pause gaps, timing and speaker ids (WhisperX, stable-ts, subtitle heuristics)

### Takeaway
Established Whisper post-processors use punctuation first, then pause gaps, then length caps. stable-ts's default splits at sentence punctuation, then at gaps > 0.5 s, then at commas only if the segment exceeds 50 chars, then caps at 70 chars. WhisperX re-splits aligned segments into sentences with NLTK punkt, and Korean is not in its punkt language map, so it falls back to English punkt. A practical merge rule for this data combines text cues, a ~0.5 s pause threshold, speaker changes and a length cap.

### Cited Findings
- **stable-ts** 2.19.1, 2025-08-16, MIT, Python >=3.8. Default regroup ('da') = `ignore_special_periods → clamp_max → split_by_punctuation(['. ', '。', '?', '？']) → split_by_gap(0.5) → split_by_punctuation([', ', '，'], min_chars=50) → split_by_length(70) → clamp_max`. It also offers `merge_by_gap` (README example `merge_by_gap(.2, 10, lock=True)`, i.e. merge when gap < 0.2 s up to 10 words) and `split_by_gap(max_gap, default 0.1)`. — [stable-ts README](https://github.com/jianfch/stable-ts), [PyPI JSON](https://pypi.org/pypi/stable-ts/json)
- **WhisperX** 3.8.6, 2026-05-25, BSD-2-Clause, Python >=3.10,<3.14. It offers "Sentence-level segments (nltk toolbox)". The alignment code loads `punkt_tab/{lang}`, and for unsupported languages it falls back to English. The `PUNKT_LANGUAGES` map has cs, da, de, el, en, es, et, fi, fr, it, nl, no, pl, pt, sl, sv, tr, ml, ru, with no Korean. — [WhisperX README](https://github.com/m-bain/whisperX), [whisperx/alignment.py and utils.py](https://github.com/m-bain/whisperX/blob/main/whisperx/alignment.py), [PyPI JSON](https://pypi.org/pypi/whisperx/json)
- whisper-webui (aadnk) VAD merging uses `max_silent_period`, `max_merge_size`, a `merge_window` (default 5 s) and padding to combine speech segments. This is subtitle/VAD merging, not sentence logic. — [HF space src/vad.py](https://huggingface.co/spaces/aadnk/whisper-webui/commit/5bbbb16ea57e0a0b0b0ba075fa7d5fe1b91503bc) (search snippet)
- whisper-timestamped 1.15.9, 2025-09-09, GPLv3 (copyleft, so note it if you vendor code). — [PyPI JSON](https://pypi.org/pypi/whisper-timestamped/json)

### Inferences (recipe, not sourced as a whole; components cite above)
A greedy left-to-right merge over spans within a section:
1. Hard break if: section boundary; speaker change (EN); gap to next span >= ~1.0 s; or the merged unit would exceed a max length (EN ~350 chars / ~60-80 words; KO ~200 chars).
2. Soft break (a boundary is allowed) if gap >= 0.5 s (the stable-ts default) AND a text cue says sentence end. EN text cues: terminal `.?!` or SaT prob > threshold. KO text cues: last morpheme tag EF via Kiwi, or SaT boundary.
3. Force merge if the current unit is shorter than a minimum (EN < ~40 chars / KO < ~25 chars), or the span ends in a connective ending (KO EC: -고/-는데/-서/-면) or with no terminal punctuation and a gap < 0.3 s.
4. Keep the upstream sentence flag as one feature (it over-fires, so do not trust it alone). Store merged-unit start = first span start, end = last span end, and the list of original span ids so linking back to timestamps stays exact.
5. Tune the two thresholds (gap and min length) against the CE-08 labeled links, not by eye. Sweep gap in {0.3, 0.5, 0.8, 1.0} s.
- Korean span gaps for this video are unknown. Compute the gap distribution first (p50/p90). If Whisper-style segments abut with ~0 s gaps (common with segment-level timestamps), the gap cue is uninformative and text cues dominate.

### Gaps
- No peer-reviewed source found giving an optimal pause threshold for sentence boundaries in Whisper output. 0.5 s is a tool default, not an evaluated optimum. No Korean-specific pause statistics were found.

## Q5. Whisper-specific artifacts

### Takeaway
Whisper produces punctuated, cased text, but its segment boundaries follow its 30 s decoding windows and timestamp tokens rather than sentences. That over- or under-segments and can produce hallucinated repeats in long or silent stretches. VAD-based chunking with `condition_on_prev_text=False` (the WhisperX default) is the known mitigation upstream. Downstream we can only dedupe.

### Cited Findings
- WhisperX: "VAD preprocessing, reduces hallucination & batching with no WER degradation". `--condition_on_prev_text` defaults to False because it "reduces hallucination". It uses VAD-based segment transcription "unlike the buffered transcription of openai's". Its limitations include that words without dictionary characters (e.g. "2014.", "£13.60") get no timing, and that overlapping speech is handled poorly. — [WhisperX README](https://github.com/m-bain/whisperX)
- stable-ts adds `ignore_special_periods` and `clamp_max` before regrouping, which implies raw Whisper segment punctuation and durations need cleanup. — [stable-ts README](https://github.com/jianfch/stable-ts)

### Inferences
- Detect hallucinated repeats cheaply: flag consecutive spans with identical or near-identical normalized text (e.g. character 3-gram Jaccard > 0.9) or a span whose text is one repeated n-gram. Drop or collapse them before merging, because repeats inflate cosine matches.
- If the upstream ASR is Whisper, English spans probably already end in `.?!`. Whisper's Korean output often includes periods too, but less consistently (UNVERIFIED, no source found). Check the punctuation rate per language on the actual spans before choosing a strategy.

### Gaps
- Could not confirm which ASR produces the requester's spans. No sourced statistics on how often Whisper Korean output lacks sentence-final punctuation.
- Speaker diarization (pyannote) is out of scope. WhisperX bundles pyannote diarization but notes "Diarization is far from perfect" — [WhisperX README](https://github.com/m-bain/whisperX). Not worth adding for the Korean video (single-speaker presumption unverified).

## Q6. Target unit length for sentence-like units (KO vs EN)

### Takeaway
No source gives an evidence-based optimal length for embedding-matched spoken sentences. The tool defaults that exist (stable-ts 70-char subtitle lines; SaT's sentence definition) are for subtitles and sentences, not retrieval. Pick lengths to tune empirically.

### Cited Findings
- stable-ts caps segments at 70 chars (`split_by_length(70)`) and splits at commas only above 50 chars. These are subtitle-oriented limits. — [stable-ts README](https://github.com/jianfch/stable-ts)
- Current data: EN spans 3-280 chars (median 55), KO 8-99 (median 40). Current chunks are 5-sentence windows of 166-535 chars. — [requester brief, local file 00_requester_context.md]

### Inferences
- Suggested starting targets for the sentence-like unit (the MATCHING atom): EN ~60-200 chars (~12-40 words), KO ~40-120 chars. Korean packs more meaning per character (roughly 2-3x English chars per idea, UNVERIFIED rule of thumb). Merge anything shorter than ~25 chars (KO) or ~40 chars (EN), such as fillers like "네", "맞아요" or "Yeah.", into a neighbor. Matching windows then group 1-3 units.
- Express limits in embedding-model tokens if possible (BGE-M3 / Qwen3 tokenizers both handle Korean). Then measure tokens per unit on both videos rather than assuming char ratios.

### Gaps
- No empirical study found on optimal unit length for short-comment-to-transcript matching in Korean or English. This must come from the CE-08 sweep.
