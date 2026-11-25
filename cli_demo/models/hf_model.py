"""
HuggingFace DistilBERT classifier wrapper.
Works for all three DistilBERT variants (baseline, kd, kd_extra).
"""
import os
import numpy as np

LABELS = ["entailment", "neutral", "contradiction"]


class HFClassifierModel:
    def __init__(self, model_path: str):
        if not os.path.isdir(model_path):
            raise FileNotFoundError(f"Model directory not found: {model_path}")

        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._tokenizer = AutoTokenizer.from_pretrained(model_path)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self._model.to(self._device)
        self._model.eval()

        # Label mapping from config.json
        cfg = self._model.config
        self._id2label = {int(k): v for k, v in cfg.id2label.items()}
        self._torch = torch

    def predict(self, premise: str, hypothesis: str) -> dict:
        """
        Returns:
            {
                "label": str,
                "probs": {"entailment": float, "neutral": float, "contradiction": float},
                "confidence": float
            }
        """
        import torch
        inputs = self._tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self._model(**inputs).logits

        probs = torch.softmax(logits, dim=-1).squeeze().cpu().tolist()

        # Build prob dict using id2label mapping
        prob_dict = {self._id2label[i]: float(p) for i, p in enumerate(probs)}
        # Normalize key names
        prob_dict = {k.lower(): v for k, v in prob_dict.items()}
        # Ensure all three keys
        for lbl in LABELS:
            if lbl not in prob_dict:
                prob_dict[lbl] = 0.0

        best_label = max(prob_dict, key=prob_dict.get)
        return {
            "label":      best_label,
            "probs":      prob_dict,
            "confidence": prob_dict[best_label],
        }
