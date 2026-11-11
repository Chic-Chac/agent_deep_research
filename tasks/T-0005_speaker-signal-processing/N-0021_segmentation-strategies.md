---
id: N-0021
type: note
task: T-0005
title: "Do segmentation choices (VAD, speaker change detection, window length and overlap, multi-scale, overlap detection) raise diarization accuracy?"
status: final
created: 2026-11-11
tags: [speaker-diarization, audio-segmentation, voice-activity-detection]
uses: [N-0015]
---

# Do segmentation choices (VAD, speaker change detection, window length and overlap, multi-scale, overlap detection) raise diarization accuracy?

Prior work: model-level DER and pyannote 4.x options are in N-0015 (not repeated). Budget-limited pass (15 calls); several primary PDFs did not render, so some claims are [A].

## 1. VAD and overlap detection: DER contribution

**Takeaway:** VAD quality is the largest single segmentation factor when VAD is poor: miss + false alarm (FA) were most of the DER. Once VAD is decent (VAD error under ~5%), further gains are 1-1.5 DER points; overlap detection adds more than speaker-assignment tricks.

BUT VoxConverse dev, no collar (Table 6/2/5) — [Landini et al. 2020](https://ar5iv.labs.arxiv.org/html/2010.11718) [V]:

| VAD | VAD err | DER | Miss | FA | Spk err | JER |
|---|---|---|---|---|---|---|
| Baseline VAD | 12.05 | 17.58 | 11.22 | 2.26 | 4.10 | 39.73 |
| Energy VAD | 5.34 | 5.69 | 2.99 | 1.78 | 0.92 | 21.52 |
| Proposed full VAD | 2.98 | 4.30 | 3.10 | 0.47 | 0.73 | 19.81 |
| Oracle VAD | 0 | 3.09 | 2.30 | 0 | 0.79 | 17.26 |

- Baseline-to-energy VAD (6.7 pt VAD error) = about 12 DER points; full-to-oracle VAD = 1.21 DER points. [V]
- Clustering: AHC 5.73 DER (spk err 2.16) vs VBx 4.36 (0.79) vs VBx + reclustering 4.30 (0.73); miss/FA fixed by VAD. VBx cut speaker error about 63% relative. [V]
- Overlap handling (dev, VBx, system VAD): no OVD 4.30; system OVD + second-speaker heuristic 4.00 (miss 3.10 -> 2.34, FA 0.47 -> 0.75, spk err 0.73 -> 0.91); system OVD + oracle speaker 3.56; oracle OVD + heuristic 2.70. Detection matters more than choosing the second speaker. [V]
- The paper's text says oracle speaker assignment gives 15% extra reduction, but its table gives about 33% (1.50 -> 1.01, oracle VAD). Table values used. [V]
- VoxConverse eval of that system: 8.12% DER, 18.35% JER; dev-to-eval gap unexplained (3.1 on dev oracle). [V]
- DIHARD III: Track 1 gives reference SAD, Track 2 requires system SAD; DIHARD scoring has no collar, so boundary errors count fully. A blog summary reports baseline DER 20.65 (reference segmentation) vs 27.34 (from scratch), unchecked against [overview](https://arxiv.org/abs/2012.01477). [A]
- Hitachi-JHU DIHARD III dev ablation: removing FA and recovering misses lowered DER 18.61 -> 13.13 (baseline VAD type unstated). [slides](https://dihardchallenge.github.io/dihard3workshop/slide/Hitachi-JHU%20System%20for%20the%20Third%20DIHARD%20Speech%20Diarization%20Challenge.pdf) [A]

**Gaps:** No fetched head-to-head of Silero vs pyannote segmentation vs MarbleNet as the diarization front end. Effect of collar/padding on DER not quantified beyond the no-collar note. Overlap detection gains are on VoxConverse only (broadcast/web debate); not 2-person interviews.

## 2. Embedding segment length, multi-scale

**Takeaway:** Short segments degrade embeddings sharply below 2 s. Multi-scale gains are real but small (about 0.3-1.2 DER points, collar-forgiving scoring) and are measured with oracle VAD only.

- Speaker verification, VoxCeleb1 cropped test: baseline EER about 20.4% (1 s), 8.9% (2 s); Thin-ResNet34 + GhostVLAD 13.20% (1 s), 7.69% (2 s), 5.47% (3 s), 3.22% (full). All rows from that paper's Tables 3-4 (it retested the Thin-ResNet34 system). — [arXiv 2005.03329](https://arxiv.org/pdf/2005.03329) [V]
- ERes2NetV2 VoxCeleb1-O: EER 1.48% (2 s trials), 0.98% (3 s), 0.61% (full). — [arXiv 2406.02167](https://arxiv.org/pdf/2406.02167) [A]
- Inference: 1 s windows give about 2-3x the EER of 2 s windows; 2-3 s is the practical floor for single-scale embeddings.
- NeMo MSDD scales: telephonic 5 scales 1.5/1.25/1.0/0.75/0.5 s; meeting 6 scales 3.0 to 0.5 s; shift = half window; finest shift 0.25 s. [V]
- DER (0.25 s collar, overlap ignored, oracle VAD, estimated speaker count; [Park et al. 2022](https://ar5iv.labs.arxiv.org/html/2203.15974) [V]):

| System | CALLHOME | CH109 | AMI-MH test |
|---|---|---|---|
| Single-scale clustering | 5.38 | 1.42 | 1.89 |
| Multi-scale, equal weights | 4.57 | 1.19 | 1.06 |
| Multi-scale, optimized weights | 4.18 | 0.70 | 1.06 |
| MSDD, optimized weights | 3.92 | 0.73 | 1.06 |

- MSDD with 0 s collar and overlap scored: CALLHOME 20.14, AMI-MH test 21.19 (vs Landini et al. 21.77, 22.85). [V]
- Reducing the base scale below 0.5 s increased error; temporal resolution limited by 0.25 s finest shift. [A, search summary]
- NVIDIA NeMo's telephonic MSDD checkpoint is trained on telephone speech only and needs initial clustering. — [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/diar_msdd_telephonic) [A]
- No speaker-change-point segmentation vs uniform window comparison was found.

**Gaps:** No Korean or interview data; single-source (NVIDIA authors) for MSDD gains.

## 3. Local segmentation windows, EEND chunking

**Takeaway:** For EEND-VC style pipelines, longer chunks (30-50 s) are better globally for strong models; pyannote's 10 s window is fixed by the trained model.

- pyannote segmentation-3.0: 10 s mono 16 kHz input, powerset output, 3 speakers per chunk, max 2 simultaneous (7 classes). — [model mirrors](https://huggingface.co/DroolingPanda/segmentation-3.0/blob/main/config.yaml) [A]
- Powerset vs multilabel PIT: better mainly on overlap, more domain-robust on 9 benchmarks, removes the detection threshold hyperparameter. No DER figures read. — [Plaquet & Bredin 2023](https://arxiv.org/abs/2310.13025) [A]
- EEND-VC study of 120 configurations, chunks 5/10/30/50 s at 20% overlap: local DER rises with chunk size; global DER improves with longer chunks for strong models; best system WavLM Base+ + Conformer, multilabel loss, 50 s chunks. Finetuned vs frozen WavLM about 11% relative gain. No DER values or stride effect found. — [alphaXiv overview of arXiv 2506.11605](https://www.alphaxiv.org/overview/2506.11605) [A]
- Original EEND-VC gains over EEND are largest on recordings over about 5 min. [A, search summary]

**Gaps:** Stride effect on DER not found; pyannote chunk-step ratio default not verified (N-0015 gap persists).

## 4. Segment filtering, embedding averaging, min cluster size

- No fetched source quantifies dropping short/overlapped segments before clustering. Related: in pyannote pipelines, embeddings are skipped for frames with overlap; not verified here. [M]
- Community grid search (Bengali, community-1, 10 files about 10 h): tuning min_duration_off, clustering threshold, fa, fb lowered DER 0.199 -> 0.194. — [arXiv 2603.19256](https://arxiv.org/pdf/2603.19256) [A]
- Inference: tuning gains are small (about 0.5 DER point); out-of-the-box defaults are near a plateau.

**Gaps:** Filtering and min-cluster-size effects: no numbers found.

## 5. Settings changeable without training

| Tool | Setting | Value / note |
|---|---|---|
| NeMo VAD (`diarizer.vad`) | onset / offset | 0.4 / 0.7 |
| | pad_onset / pad_offset | 0.05 / -0.1 s |
| | min_duration_on / off | 0.2 / 0.2 s |
| | window / shift / smoothing | 0.15 s / 0.01 s / median |
| NeMo embeddings | window / shift | 1.5 / 0.75 s single-scale; multiscale lists with `multiscale_weights` |
| NeMo clustering | oracle_num_speakers, max_num_speakers | False, 20 (2 speakers: set oracle count 2) |

Source: [NeMo 24.09 config docs](https://docs.nvidia.com/nemo-framework/user-guide/24.09/nemotoolkit/asr/speaker_diarization/configs.html) [V]. Defaults were tuned on CH109 (11 sessions) [V]; not tuned for music/SFX interviews.

- pyannote pipeline: `min_duration_off` fills non-speech gaps shorter than value; `min_duration_on` drops speech shorter than value. Community-tuned values range 0.0-0.58 s with thresholds like 0.444, i.e. per-dataset. — [HF forum](https://discuss.huggingface.co/t/what-are-min-duration-off-and-threshold-means-segmentation/35006) [A]
- `num_speakers`/min/max and `exclusive_speaker_diarization`: see N-0015.

**Recommendation (inference):**
- Fix `num_speakers=2` for interviews.
- Prioritize VAD quality over embedding tweaks: baseline-to-good VAD was worth 12 DER points, good-to-oracle 1.2.
- Add overlap detection or use powerset/EEND models for overlap: up to about 0.3 DER from a system OVD at modest cost.
- Keep min_duration_off small (about 0.1-0.2 s) so backchannels survive (pause-filling effects: see N-0015).
- Tune on a few labeled minutes of your own audio; expect about 0.5 DER point.
- For music/SFX, no evidence found; check VAD false alarms by listening, since FA falls directly into DER.

## Open gaps
- Silero/MarbleNet/pyannote VAD comparison; DIHARD III overview numbers (PDF did not render); Korean data; stride and filtering effects; pyannote 4.x defaults.
