"""
demo/inference.py
-----------------
Feature-engineering and model-loading logic for the NLI demo.

Mirrors the exact preprocessing used in 02_classical_local.py and
03_distilbert_local.py so that inference results match training predictions.

Label mapping (matches both training scripts):
    0 → entailment
    1 → neutral
    2 → contradiction
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import scipy.sparse as sp

# ── constants copied verbatim from 02_classical_local.py ─────────────────────
LABELS = ["entailment", "neutral", "contradiction"]  # indices 0, 1, 2
LABEL_DESCRIPTIONS = {
    "entailment":    "The hypothesis is supported by / follows from the premise.",
    "neutral":       "The premise neither confirms nor rules out the hypothesis.",
    "contradiction": "The hypothesis conflicts with the premise.",
}
LABEL_EMOJI = {
    "entailment": "✅",
    "neutral":    "🔷",
    "contradiction": "❌",
}

TOKEN_PATTERN = r"[a-z0-9]+(?:'[a-z]+)?"
TOKEN_RE = re.compile(TOKEN_PATTERN)
NEG_WORDS = {"not", "no", "nobody", "nothing", "never", "none",
             "nowhere", "neither", "nor", "cannot"}

CROSS_MAX_NOVEL, CROSS_MAX_DROPPED = 10, 15

PROJECT_DIR = Path(__file__).resolve().parent.parent
CLASSICAL_MODEL_DIR = PROJECT_DIR / "models" / "classical"
DISTILBERT_MODEL_DIR = PROJECT_DIR / "models" / "distilbert"


# ── result container ─────────────────────────────────────────────────────────
@dataclass
class PredictionResult:
    model_name: str
    predicted_label: str
    predicted_index: int
    scores: Dict[str, float]          # label → score/probability
    score_type: str                    # "probability", "decision_score", "softmax"
    load_time_s: float
    infer_time_s: float
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None


# ── feature engineering (identical to 02_classical_local.py) ─────────────────
def _tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())


def _has_neg(tokens: List[str]) -> bool:
    return any(w in NEG_WORDS or w.endswith("n't") for w in tokens)


def _analyze_pair(premise: str, hypothesis: str) -> dict:
    """Return the same feature dict as analyze_pairs() for a single row."""
    p, h = _tokenize(premise), _tokenize(hypothesis)
    ps, hs = set(p), set(h)
    novel   = [w for w in dict.fromkeys(h) if w not in ps]
    dropped = [w for w in dict.fromkeys(p) if w not in hs]
    inter, union = len(ps & hs), len(ps | hs)
    pb, hb = set(zip(p, p[1:])), set(zip(h, h[1:]))
    p_neg, h_neg = _has_neg(p), _has_neg(h)
    dense = np.array([[
        len(p), len(h), len(h) - len(p),
        min(len(h) / max(1, len(p)), 10.0),
        inter / max(1, len(hs)),
        inter / max(1, len(ps)),
        inter / max(1, union),
        len(novel), len(dropped),
        len(hb & pb) / max(1, len(hb)),
        float(h_neg), float(p_neg),
        float(h_neg != p_neg),
        float(len(hs) > 0 and hs <= ps),
    ]], dtype=np.float64)
    novel_text = " ".join(novel)
    cross_text = " ".join(
        f"{a}__{b}"
        for a in dropped[:CROSS_MAX_DROPPED]
        for b in novel[:CROSS_MAX_NOVEL]
    )
    return {"novel_text": [novel_text], "cross_text": [cross_text], "dense": dense}


def _build_classical_features(premise: str, hypothesis: str, comps: dict):
    """Return (linear_matrix, tree_matrix) for a single pair."""
    import pandas as pd

    df = pd.DataFrame({"premise": [premise], "hypothesis": [hypothesis]})
    info = _analyze_pair(premise, hypothesis)

    Xp = comps["tfidf"].transform(df["premise"])
    Xh = comps["tfidf"].transform(df["hypothesis"])
    cos = sp.csr_matrix(np.asarray(Xp.multiply(Xh).sum(axis=1)))

    linear = sp.hstack([
        Xp, Xh, abs(Xp - Xh), cos,
        comps["novel_tfidf"].transform(info["novel_text"]),
        sp.csr_matrix(comps["scaler"].transform(info["dense"])),
        comps["cross_hash"].transform(info["cross_text"]).tocsr(),
    ], format="csr")

    tree = sp.hstack([
        sp.csr_matrix(info["dense"]), cos,
        comps["hyp_bin"].transform(df["hypothesis"]),
        comps["novel_bin"].transform(info["novel_text"]),
    ], format="csr").astype(np.float32)

    return linear, tree


# ── model loader (lazy, cached) ───────────────────────────────────────────────
class ModelRegistry:
    """Loads each model once and keeps it in memory."""

    def __init__(self):
        self._comps = None          # classical feature components
        self._comps_loaded = False
        self._clf: Dict[str, object] = {}
        self._distilbert_tokenizer = None
        self._distilbert_model = None
        self._distilbert_device = None

    # ── classical components ──────────────────────────────────────────────────
    def _load_comps(self) -> dict:
        if not self._comps_loaded:
            path = CLASSICAL_MODEL_DIR / "feature_components.pkl"
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing feature_components.pkl at {path}\n"
                    "Run 02_classical_local.py first."
                )
            self._comps = joblib.load(path)
            self._comps_loaded = True
        return self._comps

    def _load_clf(self, name: str):
        if name not in self._clf:
            path = CLASSICAL_MODEL_DIR / f"{name}_model.pkl"
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing {path.name}. Run 02_classical_local.py first."
                )
            self._clf[name] = joblib.load(path)
        return self._clf[name]

    # ── DistilBERT ────────────────────────────────────────────────────────────
    def _load_distilbert(self):
        if self._distilbert_model is None:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            if not DISTILBERT_MODEL_DIR.exists() or not any(DISTILBERT_MODEL_DIR.iterdir()):
                raise FileNotFoundError(
                    f"DistilBERT model directory is empty: {DISTILBERT_MODEL_DIR}\n"
                    "Run 03_distilbert_local.py first to fine-tune and save the model."
                )

            self._distilbert_tokenizer = AutoTokenizer.from_pretrained(
                str(DISTILBERT_MODEL_DIR)
            )
            # device: MPS → CUDA → CPU
            if torch.backends.mps.is_available():
                self._distilbert_device = torch.device("mps")
            elif torch.cuda.is_available():
                self._distilbert_device = torch.device("cuda")
            else:
                self._distilbert_device = torch.device("cpu")

            self._distilbert_model = AutoModelForSequenceClassification.from_pretrained(
                str(DISTILBERT_MODEL_DIR)
            ).to(self._distilbert_device).eval()

        return self._distilbert_tokenizer, self._distilbert_model, self._distilbert_device

    # ── public predict methods ────────────────────────────────────────────────
    def predict_logreg(self, premise: str, hypothesis: str) -> PredictionResult:
        t_load = time.perf_counter()
        comps = self._load_comps()
        clf   = self._load_clf("logreg")
        t_load = time.perf_counter() - t_load

        t_inf = time.perf_counter()
        linear, _ = _build_classical_features(premise, hypothesis, comps)
        idx   = int(clf.predict(linear)[0])
        proba = clf.predict_proba(linear)[0]
        t_inf = time.perf_counter() - t_inf

        return PredictionResult(
            model_name="Logistic Regression",
            predicted_label=LABELS[idx],
            predicted_index=idx,
            scores={LABELS[i]: float(proba[i]) for i in range(3)},
            score_type="probability",
            load_time_s=t_load,
            infer_time_s=t_inf,
        )

    def predict_linearsvc(self, premise: str, hypothesis: str) -> PredictionResult:
        t_load = time.perf_counter()
        comps = self._load_comps()
        clf   = self._load_clf("linearsvc")
        t_load = time.perf_counter() - t_load

        t_inf = time.perf_counter()
        linear, _ = _build_classical_features(premise, hypothesis, comps)
        idx    = int(clf.predict(linear)[0])
        scores = clf.decision_function(linear)[0]
        t_inf  = time.perf_counter() - t_inf

        return PredictionResult(
            model_name="LinearSVC",
            predicted_label=LABELS[idx],
            predicted_index=idx,
            scores={LABELS[i]: float(scores[i]) for i in range(3)},
            score_type="decision_score",
            load_time_s=t_load,
            infer_time_s=t_inf,
        )

    def predict_lightgbm(self, premise: str, hypothesis: str) -> PredictionResult:
        t_load = time.perf_counter()
        comps = self._load_comps()
        clf   = self._load_clf("lightgbm")
        t_load = time.perf_counter() - t_load

        t_inf = time.perf_counter()
        _, tree = _build_classical_features(premise, hypothesis, comps)
        proba = clf.predict_proba(tree)[0]
        idx   = int(np.argmax(proba))
        t_inf = time.perf_counter() - t_inf

        return PredictionResult(
            model_name="LightGBM",
            predicted_label=LABELS[idx],
            predicted_index=idx,
            scores={LABELS[i]: float(proba[i]) for i in range(3)},
            score_type="probability",
            load_time_s=t_load,
            infer_time_s=t_inf,
        )

    def predict_distilbert(self, premise: str, hypothesis: str) -> PredictionResult:
        import torch
        import torch.nn.functional as F

        t_load = time.perf_counter()
        try:
            tok, model, device = self._load_distilbert()
        except FileNotFoundError as e:
            return PredictionResult(
                model_name="DistilBERT",
                predicted_label="N/A",
                predicted_index=-1,
                scores={},
                score_type="softmax",
                load_time_s=0.0,
                infer_time_s=0.0,
                error=str(e),
            )
        t_load = time.perf_counter() - t_load

        t_inf = time.perf_counter()
        enc = tok(
            premise, hypothesis,
            truncation=True, max_length=128,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            logits = model(**enc).logits
        proba = F.softmax(logits, dim=-1).squeeze().cpu().numpy()
        idx   = int(np.argmax(proba))
        t_inf = time.perf_counter() - t_inf

        return PredictionResult(
            model_name="DistilBERT",
            predicted_label=LABELS[idx],
            predicted_index=idx,
            scores={LABELS[i]: float(proba[i]) for i in range(3)},
            score_type="softmax",
            load_time_s=t_load,
            infer_time_s=t_inf,
        )

    def artifact_status(self) -> Dict[str, bool]:
        """Return which saved artifacts exist."""
        status = {}
        for name in ("logreg", "linearsvc", "lightgbm"):
            status[name] = (CLASSICAL_MODEL_DIR / f"{name}_model.pkl").exists()
        status["feature_components"] = (CLASSICAL_MODEL_DIR / "feature_components.pkl").exists()
        status["distilbert"] = (
            DISTILBERT_MODEL_DIR.exists()
            and any(DISTILBERT_MODEL_DIR.iterdir())
        )
        return status


# singleton used by the CLI
registry = ModelRegistry()


MODEL_NAMES = {
    "1": "logreg",
    "2": "linearsvc",
    "3": "lightgbm",
    "4": "distilbert",
}
MODEL_DISPLAY = {
    "logreg":     "Logistic Regression",
    "linearsvc":  "LinearSVC",
    "lightgbm":   "LightGBM",
    "distilbert": "DistilBERT",
}

PREDICT_FN = {
    "logreg":     registry.predict_logreg,
    "linearsvc":  registry.predict_linearsvc,
    "lightgbm":   registry.predict_lightgbm,
    "distilbert": registry.predict_distilbert,
}
