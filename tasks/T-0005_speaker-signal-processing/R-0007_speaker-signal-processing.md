---
id: R-0007
type: report
task: T-0005
title: "VAD and in-domain fine-tuning help; transforms unproven, generic masking risky"
status: final
created: 2026-11-11
tags: [speaker-diarization, audio-preprocessing, speaker-extraction]
uses: [N-0020, N-0021, N-0022, N-0023, N-0024, N-0015]
---

# VAD and in-domain fine-tuning help; transforms unproven, generic masking risky

*Evidence as of 2026-11-11; revised the same day after a source check (Demucs figure was WER, not DER; fine-tuning and wavelet wording weakened). Benchmarks are English/Mandarin meetings, telephone, VoxCeleb and VoxConverse; no Korean or 2-speaker-interview result exists for any method. N-0024 is mostly [A] and single-source.*

## Conclusion

- **Segmentation is the only signal-side lever with large measured gains, and only when VAD is poor.** A weak-to-decent VAD swap was worth about 12 DER points; decent-to-oracle 1.2. Overlap detection, multi-scale embeddings and threshold tuning each add 0.3-1.2.
- **No transform or decomposition front-end has a demonstrated gain over log-mel for modern embedding models, and all need retraining.** Learnable and raw-waveform front-ends were measured at parity. Wavelet, CQT and scattering inputs were never compared on such models: untested, not disproven. SSL features (WavLM) are the exception, usable only through a prebuilt model.
- **Masking at inference is a risk, not a free gain, for speaker attribution.** Generic denoising raised speaker-verification EER 1.3-4.9x; separation helps ASR on overlap-heavy meetings but barely moves DER; target-speaker extraction has no measured DER or WER effect on real recordings.
- **Training-side processing pays when the domain is mismatched.** Fine-tuning on real in-domain data cut 2-speaker EEND DER from 15.09 to 8.44; two pyannote segmentation studies report larger cuts but are weak sources. Simulated conversations beat simulated mixtures.
- For the WhisperX/pyannote pipeline, these items come after R-0006's assignment-rule fixes: they reduce missed speech, false alarms and speaker confusion, not word-boundary misassignment.

## Evidence

### Transforms and decompositions: parity or untested against fbank (N-0020)

| Front-end | Result | Retrain? |
|---|---|---|
| Wavelet / wavelet packet / scattering | Beat MFCC only with GMM/HMM back-ends on small corpora; no numbers vs fbank + ECAPA/ResNet ([arXiv 2512.18902](https://arxiv.org/abs/2512.18902)) [A] | Yes |
| Raw waveform (TDF) / SincNet | TDF: Vox1-O EER 1.98 vs mel 2.04; mismatched VOiCES 8.38 vs 7.10, i.e. worse. SincNet: 2.37 and 8.55 ([Zhu et al.](https://ar5iv.labs.arxiv.org/html/2110.04265)) [A] | Yes |
| Learnable MFCC | Up to 6.7% relative EER gain on VoxCeleb1 ([arXiv 2102.10322](https://arxiv.org/abs/2102.10322)) [A] | Yes |
| WavLM Large (SUPERB, frozen + small head) | DER **3.24 vs 10.05** for fbank; the fbank baseline is weak ([WavLM](https://arxiv.org/html/2110.13900v5)) [A] | Yes, or prebuilt (DiariZen; N-0015) |

No results were found for EMD/VMD, CQT, harmonic/percussive, pitch or phase features.

### Segmentation: VAD dominates when it is poor (N-0021)

- VoxConverse dev, no collar: baseline VAD DER 17.58, energy VAD 5.69, best VAD 4.30, oracle VAD 3.09 ([Landini et al. 2020](https://ar5iv.labs.arxiv.org/html/2010.11718)) [V].
- In the same system, overlap detection with a second-speaker heuristic took DER 4.30 to 4.00 [V].
- Multi-scale embeddings: CALLHOME DER 5.38 to 4.18. Measured with oracle VAD and a 0.25 s collar ([Park et al. 2022](https://ar5iv.labs.arxiv.org/html/2203.15974)) [V].
- Embedding segments below 2 s degrade sharply: EER 13.20% at 1 s vs 7.69% at 2 s ([arXiv 2005.03329](https://arxiv.org/pdf/2005.03329)) [V].
- Grid-tuning pyannote `min_duration_off` and thresholds: DER 0.199 to 0.194 ([arXiv 2603.19256](https://arxiv.org/pdf/2603.19256)) [A].
- Inference: defaults sit near a plateau. The 12-point gain applies only if your VAD misses speech or fires on music or SFX.

### Masking, enhancement, separation, extraction at inference (N-0022, N-0024)

| Method | Gain | Hurts or none | Retrain? |
|---|---|---|---|
| Generic denoiser before embeddings | None | ECAPA EER on VOiCES rose from 2.70 to 10.57, and on VoxCeleb1 from 2.18 to 2.73 ([LC4SV](https://arxiv.org/html/2311.16604v1)) [V]. On clean input every variant was slightly worse [V] | No |
| Noisy + enhanced blend (LC4SV) | 0-4% relative EER [V] | 0 on unseen model [V] | Small module |
| Vocal-stem extraction (Demucs) | ASR only: WER 0.389 to 0.365 in one Bengali competition entry ([arXiv 2602.21183](https://arxiv.org/html/2602.21183)) [V] | DER unchanged in the same entry: 0.2882 with vs 0.2886 without [V]. Artifacts; separated vocals reported to hurt long-form Whisper (secondhand) ([arXiv 2506.15514](https://arxiv.org/pdf/2506.15514)) [A] | No |
| Separation (PixIT) | AMI-SDM cpWER 38.3 to 31.7 and AliMeeting cpCER 65.1 to 48.3 ([PixIT](https://arxiv.org/html/2403.02288v2)) [V] | Tuning for cpWER worsened DER to 25.3 / 28.3 [V]; trained on 2 meeting corpora only | Yes for new domains |
| Target-speaker extraction | Libri2Mix SI-SDRi only 9-17 dB ([USEF-TSE](https://arxiv.org/html/2409.02615)) [A] | Fails to transfer to real conversation ([REAL-TSE](https://arxiv.org/pdf/2607.15198)) [A]; extracts the wrong speaker when voices are similar ([arXiv 2204.01355](https://arxiv.org/abs/2204.01355v1)) [A] | Yes; needs 5-15 s of enrollment |
| TS-VAD, Seq2Seq-TSVAD | CHiME-6 DER 63.42 to 32.84 ([TS-VAD](https://ar5iv.labs.arxiv.org/html/2005.07272)) [A] | Over a strong first pass the gain is small: VoxConverse 5.35 to 4.55 ([Seq2Seq-TSVAD](https://ar5iv.labs.arxiv.org/html/2210.16127)) [V] | Yes; code not located |

- Inference: SI-SDR gains do not predict speaker-attribution gains. On 2-speaker audio with under 5% overlap, separation and TSE have little to remove and add artifacts, so their expected gain is near zero (unmeasured).

### Training-side processing (N-0023)

- Pitch-changing speed perturbation: Vox1-O EER 0.947 to 0.861. SpecAugment made it worse, 0.947 to 1.064 ([SRE system](https://ar5iv.labs.arxiv.org/html/2211.00815)) [V].
- WavLM's utterance-mixing pretraining: SUPERB DER 6.03 to 4.55 ([WavLM](https://ar5iv.labs.arxiv.org/html/2110.13900)) [V].
- Simulated conversations vs simulated mixtures, 2-speaker EEND: CALLHOME DER 8.64 vs 15.09 ([Landini et al. 2022](https://ar5iv.labs.arxiv.org/html/2204.00890)) [V].
- Fine-tuning 2-speaker EEND on real CALLHOME data: DER 15.09 to 8.44 from simulated mixtures, 8.64 to 7.28 from simulated conversations (same source) [V].
- Fine-tuning pyannote segmentation-3.0 on 54 h of accented English: DER 21.30 to 10.65 ([arXiv 2509.21554](https://arxiv.org/html/2509.21554v1)) [V]. Every fine-tuned figure in that table is exactly half its baseline; treat the size as unconfirmed.
- On Indonesian, 2 h of synthetic data: DER 53.47 to 34.81. Single source, and its abstract contradicts its table ([arXiv 2601.03684](https://arxiv.org/html/2601.03684)) [V].
- Adding Korean data to ECAPA training improved every test set; no figures ([KoreaScience](https://koreascience.kr/article/JAKO202414143303118.pub)) [A].
- Test-time scoring with AS-norm and calibration: Vox1-O EER 1.058 to 0.861 [V]. Inference: this applies to identifying enrolled speakers across videos, not to within-file clustering.

## Ranked list: what to try first

| # | Action | Retrain? | Evidence basis |
|---|---|---|---|
| 1 | Score miss and false-alarm rates of the current VAD on R-0006's labelled minutes. If music/SFX false alarms or missed backchannels are visible, tune onset/offset and `min_duration_on/off` | No | VAD swing of 12 DER points [V]; tuning about 0.5 [A] |
| 2 | Run Demucs vocal stems on music-flagged regions only, for ASR. Keep diarization and embeddings on the original audio (untested design) | No | WER gain, no DER change, one entry [V]; enhancement hurts embeddings [V] |
| 3 | Do not denoise globally before embedding. If noise is a problem, use enhancement for VAD only or blend noisy and enhanced audio | No | LC4SV [V] |
| 4 | Fine-tune pyannote segmentation on 2-25 h of labelled or synthetic Korean/English interview audio (fits 16 GB) | Yes | EEND fine-tuning gain [V]; pyannote-specific gains from two weak sources |
| 5 | For recurring hosts: enrol embeddings and use cosine + AS-norm scoring for naming | No | 0.2 EER absolute [V] |
| 6 | Separation, TSE, TS-VAD: only if labelled data show overlap well above 5% | Yes / code missing | Gains limited to overlap-heavy or weak-first-pass audio |
| 7 | Wavelet, CQT, learnable front-ends | Yes | Parity (learnable) or untested (wavelet, CQT) |

## Open questions and gaps

- None of the methods has been measured on Korean audio or on low-overlap 2-speaker interviews.
- No direct comparison of Silero, pyannote and MarbleNet VAD within diarization was found.
- No DER result exists for DeepFilterNet, MetricGAN or WPE as a front-end.
- Demucs: the only DER comparison found (one competition entry) shows no change; its measured gain is on WER.
- The two pyannote fine-tuning sources are weak: one reports figures that are exactly half the baseline, the other contradicts its own abstract.
- Wavelet, CQT, scattering, EMD/VMD and phase features have no published comparison on VoxCeleb-scale neural models; their effect is unknown.
- TSE and TS-VAD evidence (N-0024) is [A] from model-summarized pages and partly single-source. No DER/WER effect of TSE on real recordings has been reported.
- Multi-scale gains come from a single source (NVIDIA authors) and were measured with oracle VAD.
- No curve of DER against hours of fine-tuning data exists.
- How DER gains translate to WDER, the metric R-0006 uses, has not been measured.
