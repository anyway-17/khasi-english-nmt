# Data Augmentation for Khasi–English Machine Translation

A controlled ablation study of data augmentation strategies for **Khasi → English neural machine translation**, built on a customized **mBART-large-50**. Completed as part of the GRISHMA Undergraduate Summer Research Programme at **IIT Kharagpur**, under the mentorship of **Prof. Sudeshna Sarkar**.

> Full write-up: [`report/Project_report.pdf`](report/Project_report.pdf)

---

## Overview

Khasi is an indigenous language of Meghalaya, India, with very little parallel text available. This project asks a practical question: **which data augmentation techniques actually help when parallel data is scarce?**

A baseline Khasi→English system was fine-tuned from mBART-large-50 (extended with a custom `kha_IN` language token), and three augmentation strategies were then evaluated **in isolation** under identical settings:

1. **Back-translation** using monolingual English text
2. **Lexical substitution** using LaBSE nearest-neighbour word embeddings
3. **Token-level noise injection** (dropout, swap, replacement)

## Key Results

Evaluated on a held-out test set of ~2,600 sentence pairs (beam size 4).

| System | Training pairs | BLEU ↑ | chrF ↑ | TER ↓ |
|---|---:|---:|---:|---:|
| Baseline | 20,800 | 22.78 | 43.27 | 75.98 |
| + Back-translation | 69,492 | **27.31** | **46.85** | **72.48** |
| + Lexical substitution | 62,400 | 21.97 | 42.85 | 77.27 |
| + Noise injection | 41,600 | 21.97 | 42.85 | 77.27 |

**Change vs. baseline**

| System | ΔBLEU | ΔchrF | ΔTER |
|---|---:|---:|---:|
| + Back-translation | +4.53 | +3.58 | −3.50 |
| + Lexical substitution | −0.81 | −0.42 | +1.29 |
| + Noise injection | −0.81 | −0.42 | +1.29 |

**Takeaways**

- **Back-translation helped on every metric**, adding +4.53 BLEU over the baseline.
- **Lexical substitution and noise injection both hurt.** Source-side perturbation without a denoising objective appears to add noise rather than useful variation in this low-resource setting. For lexical substitution, LaBSE nearest neighbours in a small Khasi vocabulary were often not true synonyms, which altered sentence meaning.
- Augmentation that introduces **new, semantically consistent sentence pairs** outperformed augmentation that merely **perturbs existing ones**.

For context, Sayed et al. (2024) reported 10.47 BLEU for Khasi→English using NLLB-3.3B with LoRA on ~126k pairs. That work used different data and a different test set, so the numbers are not directly comparable.

---

## Method

### Model
- **Base model:** mBART-large-50 (encoder–decoder transformer)
- **Khasi support:** a new `kha_IN` language token was added to the tokenizer and model. Its embedding was initialised from the English token (`en_XX`) rather than Hindi (`hi_IN`), since Khasi and English both use the Latin script.

### Data
- ~26,000 English–Khasi parallel sentence pairs, provided by the project mentor. The corpus is primarily biblical text with a smaller set of general-domain sentences.
- Split 80/10/10 with a fixed random seed into ~20,800 train / ~2,600 validation / ~2,600 test pairs. The same splits are used for every experiment.
- A separate monolingual English corpus (~49,500 sentences) was used only for back-translation.
- Original data was in `.xlsx` and converted to JSONL for the Hugging Face Transformers pipeline.

### Augmentation strategies

| Strategy | How it works | Resulting training set |
|---|---|---|
| **Back-translation** | Train an English→Khasi model, translate ~49k monolingual English sentences into Khasi, filter by Khasi/English length ratio, merge with original data | ~69.5k pairs |
| **Lexical substitution** | Embed every unique Khasi word with LaBSE, find the 10 nearest neighbours, replace ~20% of words in each sentence with a random neighbour, generate 2 augmented copies per sentence | ~62.4k pairs |
| **Noise injection** | For each sentence, randomly apply one of token dropout, token swap, or token replacement (15% rate), 1 augmented copy per sentence | ~41.6k pairs |

### Training setup

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 3e-5 (linear schedule) |
| Batch size (per device) | 8 |
| Epochs | 5 |
| Beam size | 4 |
| Max source / target length | 1024 / 128 |

Hardware: NVIDIA V100 (32 GB) GPUs on the IIT Kharagpur HPC cluster. Framework: Hugging Face Transformers + PyTorch.

### Evaluation
BLEU (primary), chrF (character-level, better suited to Khasi's morphology) and TER, computed with sacreBLEU.

---

## Repository Structure

```
.
├── data/
│   ├── raw/                  # original English–Khasi corpus (.xlsx)
│   ├── train.jsonl
│   ├── val.jsonl
│   ├── test.jsonl
│   ├── train_bt_augmented.jsonl
│   ├── train_lexsub.jsonl
│   └── train_noise.jsonl
├── scripts/                  # data prep, augmentation, training, evaluation
├── results/                  # predictions and metric files for each experiment
├── report/
│   └── Project_report.pdf
└── README.md
```

> **Model checkpoints are not included.** Each fine-tuned mBART-large-50 model is ~2.4 GB, which exceeds GitHub's file size limits.

## Example: Fine-tuning

Training uses the Hugging Face `run_translation.py` example script. This command shows the hyperparameters used; adjust paths to your setup.

```bash
python run_translation.py \
  --model_name_or_path <path-to-mbart-with-kha_IN> \
  --source_lang kha_IN \
  --target_lang en_XX \
  --forced_bos_token en_XX \
  --train_file data/train.jsonl \
  --validation_file data/val.jsonl \
  --test_file data/test.jsonl \
  --do_train --do_eval --do_predict \
  --predict_with_generate \
  --per_device_train_batch_size 8 \
  --learning_rate 3e-5 \
  --num_train_epochs 5 \
  --num_beams 4 \
  --max_source_length 1024 \
  --max_target_length 128 \
  --output_dir <output-dir>
```

Each JSONL line follows the Hugging Face translation format:

```json
{"translation": {"kha": "<Khasi sentence>", "en": "<English sentence>"}}
```

---
