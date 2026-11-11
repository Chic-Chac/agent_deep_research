---
id: N-0018
type: note
task: T-0004
title: "C. Evaluation metrics and low-cost evaluation protocols for speaker-attributed transcripts"
status: final
created: 2026-11-03
tags: [evaluation, speaker-diarization]
---

# C. Evaluation metrics and low-cost evaluation protocols for speaker-attributed transcripts

Context: word-timestamped ASR + 2-speaker diarization + sentence spans, YouTube interviews. Known failure modes: speaker change lands mid-sentence or a few words off; questions/interjections given to the wrong speaker; spans of uneven length. Research done 2026-11-03. Package versions and dates come from live PyPI JSON queries on that day (pypi.org/pypi/<pkg>/json).

## 1. Metric definitions, tools, versions, licenses, pitfalls

### Takeaway
Time-based metrics (DER/JER) come from pyannote.metrics and dscore. Word-based speaker-attribution metrics come from DiarizationLM's `metrics.py` (WDER, cpWER) and MeetEval (cpWER, tcpWER, ORC-WER, DI-cpWER, plus DER through md-eval). Boundary metrics come from pyannote.metrics (time-based change-point precision/recall with a tolerance) and segeval (Pk, WindowDiff, Boundary Similarity on unit sequences). All of them are pip-installable, run on CPU and work on Python 3.12. None needs the GPU.

### Cited Findings
**DER, with and without collar or overlap**
- DER = (false alarm + missed detection + confusion) / total reference speech. pyannote's `DiarizationErrorRate` maps speakers optimally with the Hungarian algorithm. A faster `GreedyDiarizationErrorRate` also exists. **Defaults: collar = 0.0 s, skip_overlap = False.** — [pyannote.metrics reference](https://pyannote.github.io/pyannote-metrics/reference.html)
- pyannote.metrics also provides Purity/Coverage, Homogeneity/Completeness, speaker-count accuracy/error and `JaccardErrorRate`. — [pyannote.metrics reference](https://pyannote.github.io/pyannote-metrics/reference.html)
- DIHARD III defines DER as the share of reference speaker time not correctly attributed under an optimal mapping. It uses JER, first introduced in DIHARD II and based on the Jaccard index, as its second metric. JER weights each speaker equally regardless of how long they talk (pyannote doc). — [DIHARD III eval plan](https://ar5iv.arxiv.org/html/2006.05815); [pyannote.metrics](https://pyannote.github.io/pyannote-metrics/reference.html)
- Collar conventions differ by source. A 0.25 s "no score" collar around reference boundaries is described as standard to absorb annotation inconsistency (search summary of DIHARD/diarization reviews). A 2025 benchmark paper used collar 0.25 s with skip_overlap=False through pyannote.metrics. pyannote.ai's 2026-10-30 blog says "commonly 0.25 seconds". It also warns that overlap handling varies ("some DER implementations ignore overlaps, while others count them strictly"), so numbers from different protocols cannot be compared. — [DIHARD III plan](https://ar5iv.arxiv.org/html/2006.05815); [Benchmarking Diarization Models, arXiv 2509.26177](https://ar5iv.labs.arxiv.org/html/2509.26177); [pyannote.ai blog](https://www.pyannote.ai/blog/how-to-evaluate-speaker-diarization-performance)
- Related pitfall: the 2509.26177 benchmark found that systems fail mainly on boundary precision of longer segments and on segments <0.5 s. — [arXiv 2509.26177](https://ar5iv.labs.arxiv.org/html/2509.26177)

**WDER (word diarization error rate)**
- DiarizationLM definition: **WDER = (S_IS + C_IS) / (S + C)**. S_IS is ASR substitutions carrying the wrong speaker, C_IS is correctly recognized words carrying the wrong speaker, S is all substitutions and C is all correct words. Insertions and deletions are excluded because they have no reference/hypothesis speaker pair. The original definition is from Shafey et al. 2019 (arXiv 1907.05337). — [DiarizationLM paper (Interspeech 2024)](https://arxiv.org/html/2401.03506); [DiarizationLM README](https://github.com/google/speaker-id/tree/master/DiarizationLM)
- Speaker labels are transferred over a Levenshtein alignment, with Hungarian matching over speaker permutations (TPST). — [DiarizationLM](https://arxiv.org/html/2401.03506)
- Implementation: `pip install diarizationlm` (v0.1.5, 2025-03-31, Apache-2.0). `compute_metrics_on_json_dict()` returns WER, WDER and cpWER. The README stresses that its reported metrics are **micro** averages (numerator and denominator are summed over all utterances before dividing). — [DiarizationLM README](https://github.com/google/speaker-id/tree/master/DiarizationLM); PyPI
- Reference numbers on 2-speaker telephone data (baseline before LLM correction): Fisher WDER 5.32%, cpWER 21.19%; Callhome English WDER 7.72%, cpWER 24.39%. The finetuned PaLM 2-S reduced WDER by 55.5% (Fisher) and 44.9% (Callhome) relative. — [DiarizationLM](https://arxiv.org/html/2401.03506)

**cpWER / tcpWER / ORC-WER / DI-cpWER / SA-WER (MeetEval)**
- cpWER: concatenate each speaker's words in reference and hypothesis, compute WER for every speaker permutation and keep the lowest. — [DiarizationLM](https://arxiv.org/html/2401.03506)
- MeetEval implements SISO WER, cpWER, ORC-WER, greedy ORC, MIMO-WER, tcMIMO, **tcpWER**, tcORC, greedy tcORC, **DI-cpWER** (diarization-invariant cpWER) and DER via an md-eval wrapper. Input formats: STM, CTM, SegLST (JSON, used in CHiME) and RTTM (DER only). MIT license. `pip install meeteval`; latest is **v0.4.3, 2025-09-08**. — [fgnt/meeteval](https://github.com/fgnt/meeteval); PyPI
- tcpWER adds a temporal constraint to cpWER, so that "only words are identified as correct when the temporal alignment is plausible". This penalizes bad timestamps. When only segment-level times exist, word times are approximated from segment times (a length-based heuristic), with results close to exact annotations. Paper: von Neumann, Boeddeker, Delcroix, Haeb-Umbach, CHiME-2023 workshop; arXiv 2307.11394 (rev. 2024-01-25). — [arXiv 2307.11394](https://arxiv.org/abs/2307.11394); [ISCA archive](https://www.isca-archive.org/chime_2023/neumann23_chime.html)
- CHiME-8 DASR adopted tcpWER with a **5 s collar** to allow for word-boundary inaccuracy. — [CHiME-7/8 review, arXiv 2507.18161](https://arxiv.org/pdf/2507.18161) (via search snippet)
- SA-WER (Kanda et al.) counts words that are misrecognized or attributed to the wrong speaker under best speaker permutation. A 2022 paper (arXiv 2211.16112) analyses how SA-WER, cpWER, ORC-WER and MIMO-WER relate. — [arXiv 2211.16112](https://arxiv.org/pdf/2211.16112); [emergentmind summary (secondary)](https://www.emergentmind.com/topics/speaker-attributed-asr-saa)
- Newer: tcpSemER (semantic-similarity version of tcpWER) and tcpWER split into overlap and non-overlap parts (Tawara, Cornell, Polok, Delcroix, Burget, Watanabe, arXiv 2603.22709, Mar 2026). It finds LLM-based systems competitive in **two-speaker** settings but degrading as speaker count and overlap rise. Single source; no code link seen. — [arXiv 2603.22709](https://arxiv.org/abs/2603.22709)

**Speaker-change detection and segmentation boundaries**
- pyannote.metrics `SegmentationPrecision` and `SegmentationRecall` score boundaries with a tolerance (default **0.5 s**). `SegmentationPurity`, `SegmentationCoverage` and `SegmentationPurityCoverageFMeasure` do not depend on a tolerance. The docs recommend purity/coverage for that reason. — [pyannote.metrics reference](https://pyannote.github.io/pyannote-metrics/reference.html)
- Speaker-change detection (SCD) literature: a predicted change counts as correct if it is within a collar of a reference change, commonly ±0.5 s. — [Collar-aware SCD, arXiv 2205.07086](https://arxiv.org/pdf/2205.07086); search summary. Token-level SCD work (transformer-transducer, arXiv 2211.06482) also exists. I did not open it, so I have no detail on its metric.
- Text-segmentation metrics: **Pk** and **WindowDiff** slide a window over reference and hypothesis and count windows whose boundary counts differ. Pevzner and Hearst (2002) proposed WindowDiff to fix Pk over-penalizing false negatives and transposed boundaries. **Boundary Similarity (B)** and Boundary Edit Distance (Fournier 2013) give partial credit for near-miss boundaries and yield confusion matrices (boundary P/R/F). Implementation: `segeval` v2.0.11 (last release **2013-12-21**, BSD; unmaintained, so check that it imports on Python 3.12). — [segeval PyPI](https://pypi.org/project/segeval); [Fournier, Segmentation Similarity and Agreement](https://ar5iv.labs.arxiv.org/html/1204.2847)

**Tool inventory (PyPI on 2026-11-03)**
| Tool | Version / date | License | Notes |
|---|---|---|---|
| pyannote.metrics | 4.2.0 / 2026-10-16 | MIT (repo; PyPI field empty) | requires Python >=3.11; DER, JER, purity/coverage, segmentation P/R |
| pyannote.audio | 4.0.7 / 2026-06-30 | MIT code (gated HF models) | Python >=3.10 |
| meeteval | 0.4.3 / 2025-09-08 | MIT | cpWER/tcpWER/ORC/DI-cpWER/DER |
| diarizationlm | 0.1.5 / 2025-03-31 | Apache-2.0 | WDER, cpWER, TPST alignment |
| spy-der (spyder) | 0.4.1 / 2023-06-29 | not shown in PyPI metadata | fast DER (C++). License unverified |
| dscore (PyPI) | 0.4.5 / 2022-12-20 | PyPI shows "GPL3+" text but BSD classifier | **may not be the NIST-style nryant/dscore GitHub tool**, which is normally cloned. Unverified |
| segeval | 2.0.11 / 2013-12-21 | BSD | Pk, WindowDiff, B, S |
| jiwer | 4.0.0 / 2025-06-19 | (Apache-2.0, not re-verified) | plain WER and alignments |
| label-studio | 1.23.2 / 2026-09-29 | Apache-2.0 | Python 3.10–3.x |
| pympi-ling | 1.71 / 2025-08-18 | (MIT, not re-verified) | read/write ELAN .eaf |

### Inferences
- All scorers are CPU-only. A few hour-long videos score in seconds to minutes. On RTX 5060 Ti / WSL2 / Py3.12 the only constraints are pyannote.metrics 4.x (needs Py >=3.11, fine) and old segeval (pure Python; may need small fixes).
- Use the DER collar consistently. For this use case run both collar 0 and collar 0.25 s with skip_overlap=False. Report the difference, since that difference is roughly the boundary-jitter component.

### Gaps
- Exact NeMo diarization scoring script names and versions not checked. NeMo is known to wrap pyannote.metrics for DER (from memory, unverified).
- License of spyder and the identity of PyPI `dscore` not confirmed.
- NIST RT md-eval specifics (default 0.25 s collar in md-eval) not fetched. They are commonly cited, but the claim rests on secondary sources here.

## 2. Which metric captures "speaker change 1–3 words off" and "sentence split at wrong place"?

### Takeaway
Use **WDER**, which counts words with the wrong speaker. Pair it with a **word-indexed speaker-change boundary F1 at tolerance k words** (k = 0, 1, 2, 3) and a near-miss-tolerant segmentation metric (**Boundary Similarity** or boundary F1 with tolerance, plus WindowDiff) over the word sequence for sentence spans. DER at 0.25 s collar largely hides 1–3-word shifts. cpWER mixes speaker errors with ASR errors, and tcpWER's 5 s collar is too loose to see them.

### Cited Findings
- WDER counts only correct and substituted words with the wrong speaker label (word-level, independent of time). — [DiarizationLM](https://arxiv.org/html/2401.03506)
- cpWER adds ASR errors and speaker errors together (whole-speaker-stream WER under best permutation). Fisher baseline cpWER 21.19% vs WDER 5.32% shows how much ASR error dominates cpWER. — [DiarizationLM](https://arxiv.org/html/2401.03506)
- tcpWER in CHiME-8 used a 5 s collar. — [arXiv 2507.18161](https://arxiv.org/pdf/2507.18161)
- Time-tolerance boundary P/R: pyannote's default is 0.5 s. — [pyannote.metrics](https://pyannote.github.io/pyannote-metrics/reference.html)
- Boundary Similarity gives partial credit to near misses, while Pk/WindowDiff are window-based. — [segeval](https://pypi.org/project/segeval); [Fournier](https://ar5iv.labs.arxiv.org/html/1204.2847)

### Inferences
- Speech runs at about 2.5–3 words/s, so 1–3 words is about 0.3–1.2 s. A 0.25 s DER collar removes part of this error, and the remainder is a small share of total speech time. For a turn-boundary error of 2 words in a 20-word turn, DER changes by only a few tenths of a percent per change point. WDER changes by the same word count but normalized by words, and a word-level boundary metric registers it directly as a miss at k<2.
- Recommended metric set for this pipeline:
  1. **WDER** (diarizationlm, or own code after aligning hypothesis words to reference words).
  2. **Speaker-change boundary P/R/F1 by word index**, reported at k = 0/1/2/3 words. A curve across k separates "missing turns" from "off by a couple of words".
  3. **Question/interjection attribution accuracy** over a labelled list of short turns (<=5 words) and question sentences. The targeted error class is invisible in global averages, which are dominated by long turns.
  4. **Sentence-span boundary F1 with tolerance** plus WindowDiff/B on word indices, and a **"speaker-pure span rate"**: fraction of sentence spans whose words all share one reference speaker (detects mid-sentence speaker splits).
  5. DER (collar 0 and 0.25) and cpWER only as sanity or comparison numbers.
- Using word indices requires aligning hypothesis words to reference words (Levenshtein, as in TPST). With imperfect references such as captions, compute boundaries on aligned anchor words only.

### Gaps
- No published metric was found that is specifically "speaker-change F1 at k-word tolerance". It is a straightforward construction (my inference). Token-level SCD papers (arXiv 2211.06482) may define a similar metric but were not read.

## 3. Cheap evaluation without full manual transcripts

### Takeaway
The cheapest reliable protocol is **partial, targeted annotation**: mark only speaker-change points and the speaker of short turns/questions in sampled windows, with model-assisted correction rather than labelling from scratch. Add reference-free proxy metrics and synthetic two-speaker concatenations for regression testing. Creator captions can serve as a *word* reference, but they need caution: style differences inflate WER, and captions usually lack speaker labels. LLM-as-judge for speaker attribution has **no published human-agreement validation that I found**, so treat it as a triage tool, not a metric.

### Cited Findings
**Partial and model-assisted annotation**
- Manual diarization annotation runs at **15–30× real time**. One estimate puts full transcription plus speaker identity at **about 9 h per hour** of spontaneous speech. Verifying automatic output took about **2× duration** in an audio-visual setup (VoxConverse "Spot the conversation"). — [arXiv 2007.01216](https://arxiv.org/pdf/2007.01216); [DIHARD III plan](https://arxiv.org/pdf/2006.05815) (figures from search summaries; exact attributions between the two papers not verified line by line)
- voxmap-studio (Yamaguchi, arXiv 2606.26842, June 2026, CC BY 4.0): unassisted manual annotation was "the costliest and least accurate". Automatic initialization shifts work "from creating turns to correcting them". Highlighting uncertain segments gave the lowest cost. Single source, preliminary (nine AMI files, 3-page paper). — [arXiv 2606.26842](https://arxiv.org/abs/2606.26842)
- Diarization confidence estimation: the best confidence measures put about **30% of diarization errors inside the lowest-confidence ~10% of segments**. This supports targeting annotation at low-confidence regions. — [arXiv 2406.17124](https://arxiv.org/html/2406.17124v1)

**Captions or other transcripts as reference**
- Transcription style matters a lot. On Rev16 podcasts, one system scored 11.21% WER against verbatim references, 9.62% against non-verbatim, 5.45% against multi-reference and 4.43% "GOLD" (only spans where both transcribers agree). Rev16 and Earnings-22 subset multi-reference sets are CC BY-SA 4.0. — [Style-agnostic ASR eval, arXiv 2412.07937](https://arxiv.org/html/2412.07937v1)
- Inference: creator captions are usually non-verbatim, sometimes edited, and their timing is caption-block-level. Use them for word alignment and for "GOLD-style" scoring restricted to aligned regions, not as an exact reference. Captions lacking speaker labels give no speaker reference unless the creator adds ">>"/names (not verified as a common practice).

**LLM-as-judge**
- No study was found that validates LLM judges on speaker-attribution or segmentation correctness of transcripts against human labels. Related findings: LLM-judge consistency checks can pass while still disagreeing with experts (2026 studies, search snippet). Voice-agent judge benchmarking (arXiv 2608.24314) finds attribute-dependent reliability. DiarizationLM shows LLMs can *correct* speaker labels (44.9–55.5% relative WDER reduction), which suggests text-only semantic cues carry attribution information. — [arXiv 2608.24314](https://arxiv.org/html/2608.24314); [DiarizationLM](https://arxiv.org/html/2401.03506)
- Inference: if LLM judging is used, validate it on the hand-labelled sample first (Cohen's κ against human labels on, say, 100–200 short turns) before trusting it on unlabelled videos. Use a different model than any LLM inside the repair pipeline, to avoid self-preference.

**Reference-free proxies (my design, no source validates them)**
- Fraction of sentence spans without terminal punctuation; fraction of spans with mixed speaker labels (where the pipeline keeps word-level labels); share of question sentences ("?") attributed to the guest vs host; count of turns < N words, especially 1–2-word turns flanked by the same other speaker (A-B-A sandwiches, a typical misattributed backchannel or mid-sentence split); turn count per minute vs expected; span-length distribution (CV, p95). These detect symptoms but cannot tell right from wrong, so use them as monitors, not as headline metrics.

**Synthetic tests**
- NeMo Speech Data Simulator builds multispeaker sessions from single-speaker aligned speech (e.g., LibriSpeech with alignments). Silence and overlap distributions are controlled by parameters. Optional RIR/multi-mic. — [NeMo simulator, arXiv 2310.12371](https://arxiv.org/pdf/2310.12371v1.pdf); [CHiME 2023 paper](https://www.isca-archive.org/chime_2023/park23b_chime.html)
- Inference: concatenating two known speakers' utterances gives exact word-level speaker truth, so WDER and boundary-F1 can be measured exactly. Read speech lacks interview prosody, backchannels and overlap, though, so use it for regression and unit tests only.

**Small public benchmarks with 2-speaker or interview-like data**
- **CALLHOME**: 2-speaker telephone, ~12.6% overlap (multilingual set 98.7 h). LDC-licensed (paid). **VoxConverse v0.3**: 63.8 h, YouTube-sourced, 2 to >20 speakers, 3.5% overlap, CC BY 4.0 for research; use v0.3 because of fixed test RTTMs. **AMI**: 4 speakers, ~16% overlap. — [arXiv 2509.26177](https://ar5iv.labs.arxiv.org/html/2509.26177); [VoxConverse README](https://github.com/joonson/voxconverse)
- **Fisher / Callhome English**: used for WDER in DiarizationLM. Both LDC (paid). — [DiarizationLM](https://arxiv.org/html/2401.03506)
- **Earnings-21** (39 h) and **Earnings-22** (125 files, ~119 h): transcripts with speaker labels, punctuation and RTTMs, CC BY-SA 4.0. Mostly monologue plus Q&A, which makes them useful for the "question attributed to wrong speaker" failure (analyst Q&A). — [Rev speech-datasets README](https://raw.githubusercontent.com/revdotcom/speech-datasets/main/README.md); [Earnings-21 paper](https://arxiv.org/pdf/2104.11348)
- **Rev16**: 16 h of podcast episodes, multi-reference transcripts, CC BY-SA 4.0. Whether it includes speaker labels was not verified. — [arXiv 2412.07937](https://arxiv.org/html/2412.07937v1)
- Podcast corpora (e.g., SPoRC) use pyannote-derived *automatic* speaker labels; about 39% of episodes have two speakers. These are not gold references. — [SPoRC, arXiv 2411.07892](https://arxiv.org/pdf/2411.07892)

### Inferences
- Best value for effort: one VoxConverse two-speaker file subset or Earnings-21 Q&A sections as public sanity checks, plus a small in-domain labelled set from the actual YouTube videos.

### Gaps
- No published validation of LLM-as-judge for speaker-attribution or segmentation accuracy was found.
- No published validation of the proxy metrics was found.
- Whether YouTube creator captions commonly contain speaker markers was not researched.

## 4. How much data, annotation time, tooling

### Takeaway
No source gives a minimum-minutes rule for diarization comparisons. Per-recording variance is large, and errors within a recording are correlated, so use a **paired, recording-level or block bootstrap** over change points or minutes. Practical budget (inference): roughly **30–60 labelled change-point-rich minutes across 3–5 videos**. At model-assisted correction speeds of about 2–5× real time for change-point-only labelling, that costs roughly 2–5 hours of annotator time.

### Cited Findings
- Speech errors are dependent, so plain bootstrap underestimates variance. Blockwise bootstrap has been used for confidence intervals on error-rate differences. — [arXiv 1912.09508](https://arxiv.org/pdf/1912.09508) (from search snippet; not read in full)
- Annotation cost: 15–30× real time for manual speaker-boundary annotation, about 2× for verifying automatic output, about 9 h per hour for full transcription with speaker identity. — [arXiv 2007.01216](https://arxiv.org/pdf/2007.01216); [DIHARD III plan](https://arxiv.org/pdf/2006.05815)
- Tools:
  - Prodigy `audio.manual` saves labelled regions (start/end/label), and pyannote.audio has offered Prodigy recipes for model-assisted diarization review. Prodigy is commercial. — [Prodigy audio docs](https://prodi.gy/docs/audio-video); [pyannote-audio 2.1.1 PyPI](https://pypi.org/project/pyannote-audio/2.1.1/)
  - Label Studio 1.23.2 (2026-09-29, Apache-2.0) supports audio region labelling. — PyPI
  - ELAN files can be read and written via pympi-ling 1.71. — PyPI
  - voxmap-studio (CC BY 4.0) is pyannote-integrated and logs edit counts and time. — [arXiv 2606.26842](https://arxiv.org/abs/2606.26842)

### Inferences
- Cheapest workflow: export the pipeline's word-timestamped transcript and change points as an Audacity label track (TSV start/end/label) or into Label Studio. Have the annotator only move or confirm change points and fix the speaker of short turns and questions, with each snapped to word indices. Text-first review (reading the transcript with speaker tags and playing audio only around a change) is likely faster than waveform-first review for 2-speaker interviews.
- Power intuition: a 1-hour interview has about 200–400 speaker changes. Three to five videos give about 600–1,500 change points, which is enough for paired comparisons of boundary F1 to detect differences of a few points, if analysed per change point with video-level block bootstrap. This is not derived from a source.
- Track kappa between two annotators on about 10 minutes to establish the noise floor of "where exactly the change is" (±1 word ambiguity at interruptions is expected). That noise floor decides whether a k=0 tolerance is meaningful.

### Gaps
- No sourced guidance on the minimum minutes or number of videos for diarization A/B comparisons.
- No sourced timing for "change-point-only" annotation. The 2–5× real-time figure is an extrapolation from the about 2× verification figure.
- Audacity label-track workflow and ELAN timings not sourced.
