# NLI Demo — CLI App

A single-launch, menu-driven interactive CLI demo for all 5 NLI models.

## Quick Start

```bash
cd cli_demo
source .venv/bin/activate
python demo.py
```

Or simply:
```bash
bash cli_demo/run.sh
```

## What's Inside (5 screens)

| Menu | What it does |
|------|------|
| 🔮 Live Predict | Run any model on custom input or curated examples. Compare all models side-by-side. |
| 📊 Benchmark Results | Pre-computed accuracy/F1 tables across all datasets. Per-class breakdown + confusion matrix. |
| 🔍 Error Analysis | Hard vs Easy examples, paginated error sample browser, confusion matrix viewer. |
| 📈 Analysis Deep-Dive | Accuracy by linguistic property, genre breakdown (MNLI), statistical significance. |
| ⚡ Compute & Efficiency | Parameter count, training time, GPU memory, inference speed comparison. |

## Models

| ID | Model | SNLI Test Acc |
|---|---|---|
| `lr` | Logistic Regression (TF-IDF + cross features) | 74.6% |
| `distilbert` | DistilBERT baseline | 85.8% |
| `distilbert_kd` | DistilBERT + Knowledge Distillation (generic pairs) | 87.4% |
| `distilbert_kd_extra` | DistilBERT + KD + Extra SNLI labels | 87.9% |
| `llama` | Llama-3.2-1B Instruct + LoRA ★ | **91.2%** |

## Notes

- **LR, DistilBERT models**: work immediately, no extra setup needed.
- **Llama**: requires `transformers`, `torch`, `peft` installed + HuggingFace login (`huggingface-cli login`) with Llama access.
- All benchmark result screens work **without loading any model** (pre-computed data).

## Install Llama dependencies (optional)

```bash
source .venv/bin/activate
pip install transformers torch peft
huggingface-cli login
```
