# NLI Model Lab 🔬

**Natural Language Inference on SNLI — Classical ML vs. Fine-tuned Transformer**

A command-line application that classifies a premise–hypothesis sentence pair into one of three relationships:
**Entailment** · **Neutral** · **Contradiction**

Four models are compared side-by-side: Logistic Regression, LinearSVC, LightGBM, and a fine-tuned DistilBERT.

---

## Results at a glance

| Model | Val Accuracy | Test Accuracy | Test Macro-F1 | Train Time |
|---|---|---|---|---|
| Logistic Regression ⭐ | 72.30% | 71.86% | 71.80% | ~3 min |
| LinearSVC | 72.17% | — | — | ~1 min |
| LightGBM | 70.43% | — | — | ~17 s |
| DistilBERT (fine-tuned) | 81.91% | **81.90%** | **81.88%** | ~4.4 min |

> Classical models trained on a shared stratified 50 K subset (seed 42).  
> DistilBERT: 1 epoch, lr 5×10⁻⁵, effective batch 64, max length 128.  
> Best classical model selected on validation accuracy (Logistic Regression).

---

## Project structure

```
NLI_Project/
├── 01_eda_local.py               # EDA and data cleaning
├── 02_classical_local.py         # Train LogReg / LinearSVC / LightGBM
├── 03_distilbert_local.py        # Fine-tune DistilBERT (local / MPS / CPU)
├── 03_classical_kaggle.py        # Classical training — Kaggle version
├── 04_distilbert_kaggle.py       # DistilBERT — Kaggle GPU version
├── eda_kaggle_fixed.py           # EDA — Kaggle version
│
├── nli_demo.py                   # ← CLI entry point
├── demo/
│   ├── inference.py              # Feature engineering + model loading
│   ├── results.py                # Metrics, confusion matrices, error analysis
│   └── display.py                # Terminal rendering (ANSI colours, tables)
│
├── models/
│   ├── classical/                # Saved .pkl files (gitignored — see below)
│   └── distilbert/               # Tokenizer configs + model weights (weights gitignored)
│
└── results/
    ├── classical_results.json    # Hyperparameters + val/test metrics
    ├── distilbert_results.json   # Hyperparameters + val/test metrics
    ├── classical_test_predictions.csv
    ├── distilbert_test_predictions.csv
    ├── eda_summary.json
    └── *.png                     # EDA plots
```

---

## Setup

### Prerequisites

- Python 3.10+
- [SNLI dataset](https://nlp.stanford.edu/projects/snli/) (run `01_eda_local.py` to produce the cleaned CSVs)

### Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install scikit-learn lightgbm transformers datasets torch joblib pandas numpy scipy
```

### Reproduce model artifacts

> ⚠️ Model weight files (`.pkl`, `model.safetensors`) are gitignored due to size.  
> Run the scripts below to regenerate them locally.

```bash
python 01_eda_local.py           # Clean SNLI → results/cleaned/
python 02_classical_local.py     # Train classical models → models/classical/
python 03_distilbert_local.py    # Fine-tune DistilBERT → models/distilbert/
```

---

## Running the demo

```bash
python nli_demo.py               # Interactive menu (recommended for viva)
```

### Direct subcommands

```bash
python nli_demo.py predict       # Single-model prediction
python nli_demo.py compare       # All four models on the same input
python nli_demo.py results       # Saved evaluation metrics + confusion matrices
python nli_demo.py errors        # Explore misclassified test examples
python nli_demo.py info          # Dataset and preprocessing details
python nli_demo.py training      # Training configuration and recorded metrics
python nli_demo.py status        # Check which model artifacts are present
python nli_demo.py train         # Re-run the full training pipeline
```

### Interactive menu

```
  1.  Predict with one model
  2.  Compare all models on the same input
  3.  View evaluation results
  4.  Explore misclassified examples
  5.  Dataset and preprocessing information
  6.  Training information
  7.  Check artifact status
  8.  Exit
```

---

## Feature engineering (classical models)

Two feature matrices are constructed per split — no data leakage, all transformers fitted on training data only.

**Linear matrix** (LogReg + LinearSVC)
- TF-IDF unigrams + bigrams on premise and hypothesis (50 K features each)
- Element-wise absolute difference of TF-IDF vectors
- Cosine similarity between premise and hypothesis
- Novel-word TF-IDF (hypothesis tokens absent from premise)
- Cross-pair hashed features (dropped × novel token combinations)
- 14 scaled dense features: token counts, length ratio, Jaccard/Dice overlap, bigram overlap, negation indicators, subsumption flag

**Tree matrix** (LightGBM)
- Same 14 dense features (unscaled)
- Cosine similarity
- Binary bag-of-words for hypothesis (top 3 K)
- Binary bag-of-words for novel tokens (top 3 K)

---

## Device support (DistilBERT)

The inference adapter auto-selects the best available device:

```
Apple Silicon MPS → CUDA → CPU
```

---

## Label mapping

| Index | Label | Meaning |
|---|---|---|
| 0 | Entailment | The hypothesis follows from the premise |
| 1 | Neutral | The premise neither confirms nor rules out the hypothesis |
| 2 | Contradiction | The hypothesis conflicts with the premise |

---

## What is gitignored

| Path | Reason |
|---|---|
| `models/classical/*.pkl` | Binary weights — up to ~25 MB total |
| `models/distilbert/model.safetensors` | ~250 MB — too large for GitHub |
| `results/cleaned/` | Raw SNLI CSVs — ~60 MB, regenerated by `01_eda_local.py` |
| `.venv/` | Virtual environment |
| `checkpoints/` | HuggingFace Trainer intermediate checkpoints |

---

## Roadmap

- [x] Classical ML baselines (LogReg, LinearSVC, LightGBM)
- [x] DistilBERT fine-tuning
- [x] Interactive CLI demo with model comparison and error analysis
- [ ] Fine-tuned LLM (LoRA / QLoRA)
- [ ] Prompted LLM (zero-shot / few-shot)
- [ ] Llama LoRA integration (`05_llama_lora_kaggle.py`)
