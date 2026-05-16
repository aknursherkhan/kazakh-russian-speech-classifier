# Kazakh-Russian Speech Classifier

An end-to-end ML pipeline for distinguishing Kazakh and Russian speech, built as a personal project using my own voice recordings and the Google FLEURS dataset. Achieved **89% test accuracy** with an Attentional Bi-LSTM on a low-resource, bilingual classification task.

---

## Overview

**Task:** Binary audio classification - Kazakh (0) vs. Russian (1)  
**Data:** ~4,800 two-second audio chunks from two sources (personal voice messages + Google FLEURS)  
**Best model:** Attentional Bi-LSTM with Bahdanau attention - 89% test accuracy  
**Key insight:** Single-speaker data caused severe overfitting; the critical fix was a data-centric pivot to integrate multi-speaker FLEURS data, not more model capacity.

---

## Project Journey (Iterative Design)

This project went through three major iterations, each motivated by a concrete failure.

### Draft 1 - Baseline
- Personal voice messages only (~single speaker)
- CNN and standard BiLSTM trained on MFCC features
- Best result: ~62–64% test accuracy despite high validation accuracy → clear overfitting to speaker identity

### Draft 2 - More Epochs, Bigger Models (Experiment 1 & 2)
- Increased training from 15 → 2,000 epochs
- Added deeper CNN layers and larger hidden dimensions
- Result: validation accuracy improved, but test accuracy plateaued at ~64% with widening train/val gap
- **Conclusion:** The model was memorizing microphone noise, not learning language features. More capacity made it worse.

### Draft 3 (Final) - Data-Centric Pivot + Architectural Upgrade
- Integrated Google FLEURS (300+ speakers, studio audio) via streaming pipeline
- Combined with personal recordings → composite dataset with acoustic diversity
- Replaced standard BiLSTM with Attentional Bi-LSTM (Bahdanau attention)
- Added Seq2Seq autoencoder for unsupervised representation learning
- **Result: 89% test accuracy, +27% generalization improvement**

---

## Pipeline

```
Raw audio (.wav)
    │
    ├─ load_audio()         16 kHz mono resampling
    ├─ chunk_signal()       2-second chunks, 25% overlap
    ├─ extract_mfcc()       40 MFCC coefficients per chunk → (40, T) matrix
    └─ GroupShuffleSplit    leak-free train/val/test split by file ID
            │
            ├─ ImprovedAudioCNN       treats MFCC as 2D image (1, 40, T)
            └─ AttentionalBiLSTM     reads MFCC as sequence + Bahdanau attention
```

### Data Engineering

**Sources:**
- **Personal recordings** - WhatsApp voice messages, 16 kHz, single speaker, high noise
- **Google FLEURS** (`kk_kz` and `ru_ru` splits) - 44.1 kHz studio audio, 300+ speakers, resampled to 16 kHz

**Streaming pipeline:** FLEURS is too large for Colab RAM, so audio is streamed from Hugging Face, resampled on the fly, and chunked.

**Feature extraction:** Each 2-second chunk is converted to a 40-coefficient MFCC matrix:
1. Short-Time Fourier Transform (STFT): time domain → frequency domain
2. Mel-scaling: linear frequencies → perceptual Mel scale
3. DCT compression: Mel energies → compact coefficient vector

**Preventing data leakage:** `GroupShuffleSplit` on `file_id` ensures all chunks from the same recording stay in the same split - prevents the model from "cheating" by recognizing background noise.

**SpecAugment:** Random frequency and time masking applied during training for regularization.

---

## Models

### 1. Improved CNN

Treats the MFCC spectrogram as a single-channel 2D image `(1, 40, T)`.

Architecture: 2 convolutional blocks (Conv2d → BatchNorm → ReLU → MaxPool) → Global Average Pooling → FC classifier

Limitation: CNNs are translation-invariant but miss temporal ordering - a key feature that distinguishes Kazakh from Russian.

**Test accuracy: ~80%** | Russian recall: 0.59 (poor)

### 2. Attentional Bi-LSTM ← Best model

Reads the MFCC matrix as a sequence of 40-dimensional frames.

**Why attention?** A standard BiLSTM compresses the entire sequence into one final hidden state `h_T`, losing information from earlier time steps. Bahdanau attention computes a weighted sum over all hidden states, letting the model focus on the most linguistically informative frames (e.g., specific phonemes or rhythmic patterns).

```
Attention score:    e_t = tanh(W · h_t)
Attention weights:  α_t = softmax(e_t)
Context vector:     c   = Σ α_t · h_t
```

Architecture: BiLSTM (hidden_dim=128, 2 layers, dropout=0.3) → Bahdanau attention → context vector → FC classifier

**Test accuracy: 89%** | Russian recall: 0.83 (↑ from CNN's 0.59)

### 3. Seq2Seq Autoencoder (Unsupervised)

Encoder-Decoder LSTM trained to reconstruct MFCC sequences without labels.

- Encoder: BiLSTM → fixed "thought vector"
- Decoder: LSTM → reconstructed MFCC sequence with teacher-forcing decay
- Loss: MSE on reconstructed vs. original MFCC frames

**Result:** MSE ≈ 1.08 on test set. PCA of latent vectors shows significant cluster overlap - the model learned some structure but cannot cleanly separate the two languages without supervision. Generated audio sounded like noise (expected: MFCCs are lossy and Griffin-Lim inversion produces artifacts).

---

## Results

| Model | Val Accuracy | Test Accuracy | Russian Recall |
|---|---|---|---|
| CNN (Draft 1, 15 epochs) | 73.8% | 62.4% | - |
| CNN (2000 epochs, small data) | ~80% | ~64% | - |
| Improved CNN (FLEURS data) | - | ~80% | 0.59 |
| **Attentional Bi-LSTM (FLEURS data)** | - | **89%** | **0.83** |

The +9% accuracy gain from CNN → AttBiLSTM is meaningful, but the more important improvement is **Russian recall**: the CNN missed 170/287 Russian test samples; the LSTM missed only 75. This validates the hypothesis that Kazakh-Russian discrimination is a **temporal** problem - the languages share similar spectral frequencies but have distinct phoneme rhythms.

---

## Key Findings

1. **Data > model capacity.** Training a bigger model for 2,000 epochs on single-speaker data gave ~64% test accuracy. Integrating multi-speaker FLEURS data with the same architecture gave 89%. The bottleneck was data variance, not model depth.

2. **Spectrograms ≠ images for speech.** The CNN's low Russian recall exposed the flaw of treating audio as a 2D image. Temporal ordering of phonemes is linguistically meaningful - an architecture that respects sequence order outperformed convolution.

3. **GroupShuffleSplit is critical for chunk-based audio.** Random splitting would have inflated test accuracy by letting the model recognize speaker-specific noise across splits.

---

## Limitations

- **Single speaker in personal data.** Even with FLEURS, the personal recordings add only one voice. Future work: collect more speakers or weight FLEURS samples higher.
- **Autoencoder audio quality.** MFCCs discard phase; Griffin-Lim inversion introduces severe artifacts. Replacing MFCCs with Mel-Spectrograms + a vocoder (e.g., HiFi-GAN) would produce cleaner reconstructions.
- **Code-switching.** The pipeline assumes monolingual clips. Kazakh and Russian are commonly mixed in real speech; a chunk-level classifier would fail on code-switched utterances.

---

## Repository Structure

```
kazakh-russian-speech-classifier/
├── kazakh_russian_speech_classifier.ipynb   # Full pipeline + experiments in appendix
└── README.md
```

The notebook appendix contains Experiment 1 (effect of training duration) and Experiment 2 (effect of model depth/capacity), with links to the original Colab runs.

> **Note:** Audio data is not included (personal voice messages + FLEURS is streamed from Hugging Face). To reproduce, replace the Google Drive paths with your own audio directory and run the streaming section to load FLEURS.

---

## Dependencies

```
torch
librosa
datasets          # Hugging Face - for FLEURS streaming
scikit-learn
numpy
pandas
matplotlib
seaborn
soundfile
```

---

## References

- Bahdanau, D., Cho, K., & Bengio, Y. (2015). [Neural machine translation by jointly learning to align and translate.](https://arxiv.org/abs/1409.0473) ICLR 2015.
- Conneau, A., et al. (2023). [FLEURS: Few-shot learning evaluation of universal representations of speech.](https://arxiv.org/abs/2205.12446) IEEE SLT 2022.
- McFee, B., et al. (2015). [librosa: Audio and music signal analysis in Python.](https://doi.org/10.25080/Majora-7b98e3ed-003) SciPy 2015.
- Park, D. S., et al. (2019). [SpecAugment: A simple data augmentation method for automatic speech recognition.](https://arxiv.org/abs/1904.08779) Interspeech 2019.
