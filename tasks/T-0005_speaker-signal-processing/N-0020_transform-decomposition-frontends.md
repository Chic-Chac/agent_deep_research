---
id: N-0020
type: note
task: T-0005
title: "Do transform and decomposition front-ends (wavelet, multi-resolution spectral, learnable filterbank, SSL features) raise speaker embedding and diarization accuracy?"
status: final
created: 2026-11-11
tags: [speaker-diarization, audio-preprocessing, wavelet-transform]
---

# Do transform and decomposition front-ends (wavelet, multi-resolution spectral, learnable filterbank, SSL features) raise speaker embedding and diarization accuracy?

Budget-limited pass (about 15 calls). PDF full text was unreadable in this environment, so numbers come from arXiv HTML/abstract pages; markers reflect that.

## Q1. Wavelet-based front-ends (DWT, wavelet packet, scattering, CWT, denoising)

**Takeaway:** Wavelet gains were found only against MFCC with GMM/HMM back-ends on small corpora. No source found compares wavelet features with fbank on VoxCeleb with ECAPA/ResNet-class models.

- Wavelet-packet features (Siafarikas et al.) are reported to beat MFCC and earlier wavelet features for verification on NIST data with a classical back-end; no numbers in the snippet — [Interspeech 2004](https://www.isca-archive.org/interspeech_2004/siafarikas04_interspeech.html) [A]
- WP-MFC (wavelet packet + MFCC, Dec 2025) is reported to beat MFCC with GMM (identification) and HMM (verification) on VoxForge and CSTR US KED TIMIT with added noise; abstract gives no numbers — [arXiv 2512.18902](https://arxiv.org/abs/2512.18902) [A]
- Scattering transform on DWT coefficients feeding a TDNN was tested on VoxCeleb1/2 (2020, Russian journal); EER not visible in the snippet — [journal.asu.ru](https://journal.asu.ru/ptzi/article/view/13934) [A]
- A CNN on stacked mel, gammatone and CWT spectrograms is reported to improve over classical spectrograms; venue, dataset and numbers unconfirmed (secondary summary) — [BRACIS summary](https://distantreader.org/stacks/journals-ojs/bracis/bracis-33571.htm) [A]
- Searches for scalogram-vs-spectrogram on VoxCeleb with ResNet, and for wavelet denoising with ECAPA, returned nothing direct.

**Inference:** Wavelet gains likely do not transfer to modern embeddings (embedding networks learn their own multi-band filtering from log-mel), but this is untested in the sources found. Every wavelet claim here is single-source, small-corpus and weak-baseline.

**Gaps:** No ECAPA/ResNet + wavelet/scattering ablation on VoxCeleb found. No wavelet-denoising result on diarization found.

## Q2. Other decompositions and learnable front-ends

**Takeaway:** On VoxCeleb, raw-waveform and learnable filters roughly match mel features; gains are small (relative 3-7% at most) and robustness to mismatched data is often worse. Not found: results for EMD/VMD, CQT, harmonic/percussive, formant, pitch or phase features.

- Raw-waveform vs mel under PLDA (EER %, Vox1-O / Vox1-E / Vox1-H / VOiCEs mismatched): Kaldi x-vector MFCC 2.26/2.37/4.14/6.79; mel-fbank 2.04/2.17/3.79/7.10; SincNet 2.37/2.32/4.02/8.55; TDF waveform 1.98/2.19/3.85/8.38; TDF+analytic+variational dropout 1.99/2.26/3.93/7.40. Waveform matches mel on matched data, degrades on mismatched VOiCEs. Kaldi baseline uses VAD, so comparison is not fully fair — [Zhu et al. 2021 (arXiv 2110.04265)](https://ar5iv.labs.arxiv.org/html/2110.04265) [A]
- Learnable MFCC: up to 6.7% relative EER gain on VoxCeleb1 and 9.7% on SITW over static MFCC; architecture and absolute EER not in abstract — [arXiv 2102.10322](https://arxiv.org/abs/2102.10322) [A]
- LearnSF (learnable sparse filterbank, Interspeech 2022) is claimed to beat Mel-fbank and other parameterized learnable filters on VoxCeleb and CNCeleb; numbers not read (PDF unreadable) — [BUT Interspeech 2022](https://www.fit.vut.cz/research/group/speech/public/publi/2022/peng22e_interspeech2022_learnable.pdf) [A]
- In the LEAF paper SincNet beat LEAF on VoxCeleb speaker identification, 43.5% vs 33.1% accuracy; LEAF led on other audio tasks — [LEAF, arXiv 2101.08596](https://arxiv.org/pdf/2101.08596) [A]
- RawNet2 reaches 4.89% EER on Vox1-H vs 4.93% for Thin-ResNet-34 + GhostVLAD on spectrograms: parity — [arXiv 2004.00526](https://arxiv.org/pdf/2004.00526) [A]
- Reference ECAPA-TDNN (VoxCeleb2 train): 3.06% EER Vox1-O, 6.71% Vox1-H on the standard fbank front-end — [arXiv 2205.01304](https://arxiv.org/pdf/2205.01304) [A]

**Speech enhancement as front-end:**
- Enhancement models not built for speaker verification can remove speaker cues; proposed fix is joint training of the U-Net with the speaker model — [Kim et al. Interspeech 2022](https://www.isca-archive.org/interspeech_2022/kim22b_interspeech.html) [A]
- Diffusion-based enhancement jointly tuned with a pretrained ECAPA-TDNN using an EER loss (2023) — [Dowerah et al.](https://www.isca-archive.org/interspeech_2023/dowerah23_interspeech.pdf) [A]
- DNN enhancement front-end reduced EER on RedDots and real customer data for a commercial system (2018, classical back-end) — [Microsoft Research](https://www.microsoft.com/en-us/research/?p=662895) [A]

**Inference:** Learnable/raw front-ends give no consistent gain over fbank on matched data and can hurt under domain mismatch. Enhancement helps in noise only with a back-end trained or tuned for it; no clean-condition degradation numbers were found.

**Gaps:** No numbers for EMD/VMD, CQT, harmonic/percussive, formant/pitch or phase features. No multi-resolution spectrogram result read. LearnSF and LEAF-on-SV tables unread.

## Q3. Self-supervised representations

**Takeaway:** SSL features give large gains over fbank on SUPERB with a light downstream head; WavLM is best. Speaker information sits in lower layers (base) or middle layers (large).

- SUPERB speaker verification EER (%) / diarization DER (%): FBank 9.56 / 10.05; wav2vec 2.0 Large 5.65 / 5.62; HuBERT Large 5.98 / 5.75; WavLM Base+ 4.07 / 3.50; WavLM Large 3.77 / 3.24. Relative gain of WavLM Large over FBank: 61% EER, 68% DER. SUPERB uses frozen upstream and a small head, so the FBank baseline is a weak, lightly parameterized one — [WavLM, arXiv 2110.13900](https://arxiv.org/html/2110.13900v5) [A]
- Layer weights: bottom layers dominate SV/SD/SID for base models; middle layers dominate for large models; top layers favor content tasks — same source [A]
- SSL features reach FBank-level or better SV with a downstream model with 97.51% fewer parameters (29.93% average SUPERB gain); VoxCeleb EER not in abstract. All SSL models but vq-wav2vec beat FBank — [arXiv 2402.00340](https://arxiv.org/abs/2402.00340) [A]
- Diarization: DiariZen (WavLM-Large front-end + pyannote-style pipeline) cuts no-collar DER vs pyannote 3.1, e.g. AMI-SDM 22.4 to 13.9, DIHARD3 21.7 to 14.5, VoxConverse 11.3 to 9.1 — see N-0015. The gain mixes SSL features with different training data and architecture, so SSL contribution is not isolated.

**Inference:** SSL features are the only replacement front-end with a large, replicated gain. Part of the SUPERB margin reflects weak FBank + small-head baseline; fully trained fbank ECAPA/ResNet on VoxCeleb is much stronger (Q2), so the practical margin there is smaller (not measured in sources fetched).

**Gaps:** Not fetched: WavLM-based fine-tuned SV on VoxCeleb (full fine-tune vs ECAPA fbank), WeSpeaker/3D-Speaker SSL recipes, VoxSRC/DIHARD system descriptions, Korean-language results.

## Q4. Applicability without retraining the speaker model

Inference from Q1-Q3, not separately sourced:

| Option | Retraining needed | Note |
|---|---|---|
| Wavelet denoising / generic enhancement before embedding | No (but speaker cues may be stripped; Kim 2022) | Test on own audio; no clean-audio degradation numbers found |
| Wavelet, CQT, scattering, phase features as model input | Yes | Embedding networks are trained on a fixed input format |
| Learnable / raw front-ends | Yes (end-to-end) | Parity with fbank on matched data |
| SSL features (WavLM) | Yes (new head) or use a prebuilt SSL-based model (DiariZen) | Largest gain; DiariZen weights non-commercial (N-0015) |
