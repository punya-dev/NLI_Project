"""
Central model registry + lazy loader with in-session cache.
Models are loaded on first request and cached for the rest of the session.
"""
import os
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
BASE  = os.path.normpath(os.path.join(_HERE, "..", ".."))


def _p(*parts):
    return os.path.join(BASE, *parts)


# ── Registry ──────────────────────────────────────────────────────────────────
MODEL_REGISTRY = {
    "lr": {
        "type":         "lr",
        "display_name": "LR — Logistic Regression (TF-IDF)",
        "short_name":   "LR",
        "artifact":     _p("NB1_LR/results-3/lr/lr_model.joblib"),
        "params_M":     2.0,
        "snli_acc":     74.6,
        "load_time":    "< 1s",
    },
    "distilbert": {
        "type":         "hf_classifier",
        "display_name": "DistilBERT Baseline",
        "short_name":   "DistilBERT",
        "artifact":     _p("NB2_distilbert/results-3/results/distilbert/distilbert_s42_model"),
        "params_M":     67.0,
        "snli_acc":     85.8,
        "load_time":    "~3–5s",
    },
    "distilbert_kd": {
        "type":         "hf_classifier",
        "display_name": "DistilBERT + KD (generic pairs)",
        "short_name":   "DB+KD",
        "artifact":     _p("NB7_GENERIC_PAIRS/results-9/results/distilbert_variants/distilbert_kd_generic_s42_model"),
        "params_M":     67.0,
        "snli_acc":     87.4,
        "load_time":    "~3–5s",
    },
    "distilbert_kd_extra": {
        "type":         "hf_classifier",
        "display_name": "DistilBERT + KD + Extra SNLI",
        "short_name":   "DB+KD+Extra",
        "artifact":     _p("NB7_GENERIC_PAIRS/results-10/results/distilbert_variants/distilbert_kd_extra_s42_model"),
        "params_M":     67.0,
        "snli_acc":     87.9,
        "load_time":    "~3–5s",
    },
    "llama": {
        "type":         "llama_lora",
        "display_name": "Llama-3.2-1B LoRA  ★ BEST",
        "short_name":   "Llama",
        "base_model":   "meta-llama/Llama-3.2-1B-Instruct",
        "adapter":      _p("NB3_LORA_LLAMA/results-4/results/llama1b_lora/llama1b_lora_adapter"),
        "params_M":     1247.1,
        "snli_acc":     91.2,
        "load_time":    "~15–30s (GPU needed for fast inference)",
        "label_token_ids": {"entailment": 87092, "neutral": 21277, "contradiction": 50859},
    },
}

# In-session model cache
_cache: dict = {}


def get_model(model_id: str):
    """Return a loaded model, loading it on first call."""
    if model_id in _cache:
        return _cache[model_id]

    import warnings
    warnings.filterwarnings("ignore")

    cfg = MODEL_REGISTRY[model_id]
    mtype = cfg["type"]

    if mtype == "lr":
        from models.lr_model import LRModel
        m = LRModel(cfg["artifact"])
    elif mtype == "hf_classifier":
        from models.hf_model import HFClassifierModel
        m = HFClassifierModel(cfg["artifact"])
    elif mtype == "llama_lora":
        from models.llama_model import LlamaLoRAModel
        m = LlamaLoRAModel(cfg["base_model"], cfg["adapter"], cfg["label_token_ids"])
    else:
        raise ValueError(f"Unknown model type: {mtype}")

    _cache[model_id] = m
    return m


def is_loaded(model_id: str) -> bool:
    return model_id in _cache


def clear_cache():
    _cache.clear()
