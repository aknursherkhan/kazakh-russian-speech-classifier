# Kazakh–Russian Speech Classifier

A deep learning pipeline for **language identification** from raw audio, built on personal voice message data augmented with Google FLEURS. The project distinguishes between Kazakh and Russian speech using a combination of MFCC-based feature extraction, convolutional networks, and an attentional bidirectional LSTM.

---

## 🎯 Project Overview

I speak both Kazakh and Russian fluently, often switching between them depending on who I'm talking to. My international friends struggle to tell which language I'm speaking — so I built a classifier to see if a model could do better.

**Final result:** The Attentional Bi-LSTM achieved **89% test accuracy**, outperforming the CNN baseline (+9%) and the logistic regression baseline by a wide margin.

---

## 🏗️ Architecture

The pipeline has three parallel modeling streams, all trained on the same MFCC-based feature representation:

```
Raw Audio (.wav)
      │
      ▼
  Preprocessing
  (16kHz mono → 2s chunks, 25% overlap)
      │
      ▼
  MFCC Extraction (40 coefficients, normalized)
      │
  ┌───┴──────────────────────────────┐
  │                                  │
  ▼                                  ▼
Improved CNN                 Attentional Bi-LSTM
(2D conv on MFCC "image")    (BiLSTM + Bahdanau attention)
  │                                  │
  └────────────┬─────────────────────┘
               │
               ▼
    Seq2Seq Autoencoder
    (unsupervised latent space analysis)
```

---

## 📊 Results

| Model | Test Accuracy | Russian Recall |
|---|---|---|
| Logistic Regression | 34.4% | — |
| Improved CNN | 80% | ~50% |
| **Attentional Bi-LSTM** | **89%** | **88%** |

The key differentiator was **recall on Russian** — the CNN missed 170 out of 334 Russian test chunks. The attention mechanism in the LSTM allowed the model to focus on the most discriminative phonetic frames, dramatically reducing false negatives.

---

## 🔬 Key Technical Decisions

### Preventing Data Leakage
Standard `train_test_split` would be dangerous here — chunks from the same recording could appear in both train and test sets, allowing the model to "cheat" by recognizing speaker-specific background noise. I used `GroupShuffleSplit` from scikit-learn, assigning **entire recordings** to either train or test (never both).

### Data Augmentation
- **SpecAugment-style masking**: random time and frequency masks applied to MFCC matrices during training
- **Gaussian noise injection**: light noise added to training chunks

### Why BiLSTM + Attention?
Standard LSTMs suffer from context collapse — by the end of a 2-second chunk, early phonemes are largely forgotten. A Bidirectional LSTM processes the sequence in both directions (0→T and T→0), and the **Bahdanau attention mechanism** learns a weighted sum over all hidden states, letting the model focus on the frames that matter most for language discrimination.

### Unsupervised Representation Learning
As an additional experiment, I trained a **Seq2Seq autoencoder** (BiLSTM encoder + LSTM decoder) entirely without labels to see if the latent representations naturally separate by language. PCA visualization of the resulting "thought vectors" showed partial but meaningful clustering between Kazakh and Russian chunks — suggesting the model captures genuine acoustic structure.

---

## 📁 Repository Structure

```
kazakh-russian-speech-classifier/
├── README.md
├── kazakh_russian_speech_classifier.ipynb   # Full pipeline
├── requirements.txt
└── data/
    └── README.md                            # Dataset setup instructions
```

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

### Data Setup

The personal voice messages cannot be shared for privacy reasons. To reproduce this project:

1. **Personal data**: Collect `.wav` voice message clips in two folders: `data/kazakh/` and `data/russian/`
2. **FLEURS data**: Download the Kazakh and Russian subsets from [Google FLEURS](https://huggingface.co/datasets/google/fleurs) via HuggingFace

See `data/README.md` for the expected directory structure.

### Running the Notebook

The notebook is designed to run in **Google Colab** with GPU acceleration. Open it and update the `DATA_PATH` variable to point to your Google Drive folder:

```python
kazakh_path = "/content/drive/MyDrive/data/kazakh/"
russian_path = "/content/drive/MyDrive/data/russian/"
```

---

## 🧱 Tech Stack

| Category | Tools |
|---|---|
| Audio processing | `librosa`, `soundfile`, `FFmpeg` |
| Feature engineering | MFCCs, ZCR, Spectral Centroid, RMS |
| Deep learning | `PyTorch` (CNN, BiLSTM, Seq2Seq) |
| Classical ML | `scikit-learn` (Logistic Regression, GridSearchCV) |
| Data | Google FLEURS, personal WhatsApp/Telegram voice messages |
| Visualization | `matplotlib`, `seaborn` |

---

## 💡 Key Learnings

1. **Model capacity ≠ data variance.** Training a large model for 2,000 epochs on 40 personal recordings resulted in high validation accuracy (~97%) but catastrophic failure on out-of-distribution data — all predictions collapsed to one class. Adding FLEURS broke this degeneracy.

2. **Feature aggregation discards temporal information.** Taking the mean of MFCC coefficients (standard in classical pipelines) loses the rhythm and cadence that distinguish Kazakh from Russian. The CNN and LSTM operate on the full 2D MFCC matrix, which is why they dramatically outperform logistic regression.

3. **MFCCs are lossy.** The Seq2Seq autoencoder produced audio with severe phase artifacts. MFCCs discard phase information, making faithful waveform reconstruction impossible without additional techniques (e.g., Griffin-Lim, WaveNet decoder).

---

## 📚 References

- Bahdanau, D., Cho, K., & Bengio, Y. (2015). [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)
- Conneau, A., et al. (2023). [FLEURS: Few-Shot Learning Evaluation of Universal Representations of Speech](https://arxiv.org/abs/2205.12446)
- Librosa Development Team. [librosa documentation](https://librosa.org/doc/latest/index.html)
- Logan, B. (2000). [Mel Frequency Cepstral Coefficients for Music Modeling](https://ismir2000.ismir.net/papers/logan_paper.pdf)
