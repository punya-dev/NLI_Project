"""
Loads all pre-computed results, CSVs, and analysis files.
No model loading here — purely reading from disk.
"""
import json
import os
import pandas as pd

# ── Base paths ────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
BASE  = os.path.normpath(os.path.join(_HERE, "..", ".."))   # NLP_PHASE_3/


def _p(*parts):
    return os.path.join(BASE, *parts)


# ── Result JSON paths ─────────────────────────────────────────────────────────
RESULT_FILES = {
    "lr": {
        "snli_val":  _p("NB1_LR/results-3/lr/results_lr_snli_val.json"),
        "snli_test": _p("NB1_LR/results-3/lr/results_lr_snli_test.json"),
        "mnli_m":    _p("NB1_LR/results-3/lr/results_lr_mnli_m.json"),
        "mnli_mm":   _p("NB1_LR/results-3/lr/results_lr_mnli_mm.json"),
    },
    "lr_hyponly": {
        "snli_val":  _p("NB1_LR/results-3/lr/results_lr_hyponly_snli_val.json"),
        "snli_test": _p("NB1_LR/results-3/lr/results_lr_hyponly_snli_test.json"),
        "mnli_m":    _p("NB1_LR/results-3/lr/results_lr_hyponly_mnli_m.json"),
        "mnli_mm":   _p("NB1_LR/results-3/lr/results_lr_hyponly_mnli_mm.json"),
    },
    "distilbert": {
        "snli_val":  _p("NB2_distilbert/results-3/results/distilbert/results_distilbert_s42_snli_val.json"),
        "snli_test": _p("NB2_distilbert/results-3/results/distilbert/results_distilbert_s42_snli_test.json"),
        "mnli_m":    _p("NB2_distilbert/results-3/results/distilbert/results_distilbert_s42_mnli_m.json"),
        "mnli_mm":   _p("NB2_distilbert/results-3/results/distilbert/results_distilbert_s42_mnli_mm.json"),
    },
    "distilbert_kd": {
        "snli_val":  _p("NB7_GENERIC_PAIRS/results-8/results/distilbert_variants/results_distilbert_kd_generic_s42_snli_val.json"),
        "snli_test": _p("NB7_GENERIC_PAIRS/results-8/results/distilbert_variants/results_distilbert_kd_generic_s42_snli_test.json"),
        "mnli_m":    _p("NB7_GENERIC_PAIRS/results-8/results/distilbert_variants/results_distilbert_kd_generic_s42_mnli_m.json"),
        "mnli_mm":   _p("NB7_GENERIC_PAIRS/results-8/results/distilbert_variants/results_distilbert_kd_generic_s42_mnli_mm.json"),
    },
    "distilbert_kd_extra": {
        "snli_val":  _p("NB7_GENERIC_PAIRS/results-10/results/distilbert_variants/results_distilbert_kd_extra_s42_snli_val.json"),
        "snli_test": _p("NB7_GENERIC_PAIRS/results-10/results/distilbert_variants/results_distilbert_kd_extra_s42_snli_test.json"),
        "mnli_m":    _p("NB7_GENERIC_PAIRS/results-10/results/distilbert_variants/results_distilbert_kd_extra_s42_mnli_m.json"),
        "mnli_mm":   _p("NB7_GENERIC_PAIRS/results-10/results/distilbert_variants/results_distilbert_kd_extra_s42_mnli_mm.json"),
    },
    "llama": {
        "snli_val":  _p("NB3_LORA_LLAMA/results-4/results/llama1b_lora/results_llama1b_lora_snli_val.json"),
        "snli_test": _p("NB3_LORA_LLAMA/results-4/results/llama1b_lora/results_llama1b_lora_snli_test.json"),
        "mnli_m":    _p("NB3_LORA_LLAMA/results-4/results/llama1b_lora/results_llama1b_lora_mnli_m.json"),
        "mnli_mm":   _p("NB3_LORA_LLAMA/results-4/results/llama1b_lora/results_llama1b_lora_mnli_mm.json"),
    },
}

ANALYSIS_DIR = _p("NB_ANALYSIS/analysis-2")


def load_result(model_id: str, dataset: str) -> dict:
    path = RESULT_FILES[model_id][dataset]
    with open(path) as f:
        return json.load(f)


def load_main_results() -> pd.DataFrame:
    return pd.read_csv(os.path.join(ANALYSIS_DIR, "main_results.csv"))


def load_compute() -> pd.DataFrame:
    return pd.read_csv(os.path.join(ANALYSIS_DIR, "compute.csv"))


def load_accuracy_by_property() -> pd.DataFrame:
    return pd.read_csv(os.path.join(ANALYSIS_DIR, "accuracy_by_property.csv"))


def load_hard_easy() -> pd.DataFrame:
    return pd.read_csv(os.path.join(ANALYSIS_DIR, "hard_easy.csv"))


def load_errors_sample() -> pd.DataFrame:
    return pd.read_csv(os.path.join(ANALYSIS_DIR, "errors_sample.csv"))


def load_genre(dataset: str = "mnli_m") -> pd.DataFrame:
    fname = f"genre_{dataset}.csv"
    return pd.read_csv(os.path.join(ANALYSIS_DIR, fname), index_col=0)


def load_significance() -> pd.DataFrame:
    return pd.read_csv(os.path.join(ANALYSIS_DIR, "significance_vs_baseline.csv"))


def load_gap_closure() -> pd.DataFrame:
    return pd.read_csv(os.path.join(ANALYSIS_DIR, "gap_closure.csv"))
