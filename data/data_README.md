# Data Setup

## Personal Voice Messages

The personal voice message data used in this project cannot be shared for privacy reasons. To replicate the personal data portion:

1. Export voice messages from WhatsApp or Telegram
2. Select clips where you speak clearly in one language (no code-switching within a clip)
3. Convert all files to `.wav` using FFmpeg:
   ```bash
   ffmpeg -i input.ogg -ar 16000 -ac 1 output.wav
   ```
4. Place files in the following structure:
   ```
   data/
   ├── kazakh/
   │   ├── kazakh_01.wav
   │   ├── kazakh_02.wav
   │   └── ...
   └── russian/
       ├── russian_01.wav
       ├── russian_02.wav
       └── ...
   ```

A minimum of 20 clips per language is recommended. More clips = better generalization.

## Google FLEURS Dataset

The FLEURS dataset provides additional out-of-distribution variance. Download it via HuggingFace:

```python
from datasets import load_dataset

# Kazakh
kk_train = load_dataset("google/fleurs", "kk_kz", split="train")
kk_val   = load_dataset("google/fleurs", "kk_kz", split="validation")

# Russian
ru_train = load_dataset("google/fleurs", "ru_ru", split="train")
ru_val   = load_dataset("google/fleurs", "ru_ru", split="validation")
```

Then save the audio arrays as `.wav` files into `data/kazakh/` and `data/russian/` following the same naming convention.

See the notebook (Section 3.1) for the exact preprocessing code used to load and merge the two sources.
