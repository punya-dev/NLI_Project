"""
Llama-3.2-1B-Instruct + LoRA adapter wrapper.
Uses next-token logits over 3 label tokens to classify.
"""
import os
import numpy as np

LABELS = ["entailment", "neutral", "contradiction"]

PROMPT_TEMPLATE = (
    "Premise: {premise}\n"
    "Hypothesis: {hypothesis}\n"
    "Does the premise entail the hypothesis, contradict it, or is it neutral?\n"
    "Answer with one word: entailment, neutral, or contradiction.\n"
    "Answer:"
)


def _softmax_dict(logit_dict: dict) -> dict:
    keys = list(logit_dict.keys())
    vals = np.array([logit_dict[k] for k in keys])
    vals = vals - vals.max()  # numerical stability
    exp = np.exp(vals)
    s = exp.sum()
    return {k: float(exp[i] / s) for i, k in enumerate(keys)}


class LlamaLoRAModel:
    def __init__(self, base_model_id: str, adapter_path: str, label_token_ids: dict):
        if not os.path.isdir(adapter_path):
            raise FileNotFoundError(f"LoRA adapter not found: {adapter_path}")

        import glob
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._label_token_ids = label_token_ids  # {label_str: token_id}
        self._torch = torch

        # Check if base model is already cached locally to avoid 401 gate errors
        target_path = base_model_id
        cache_pattern = os.path.expanduser("~/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B*/snapshots/*")
        cached_snapshots = glob.glob(cache_pattern)
        if cached_snapshots and os.path.isdir(cached_snapshots[0]):
            target_path = cached_snapshots[0]

        self._tokenizer = AutoTokenizer.from_pretrained(target_path)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # Load base + LoRA
        dtype = torch.float16 if self._device == "cuda" else torch.float32
        base = AutoModelForCausalLM.from_pretrained(
            target_path,
            dtype=dtype,
            device_map="auto" if self._device == "cuda" else None,
        )
        self._model = PeftModel.from_pretrained(base, adapter_path)
        if self._device != "cuda":
            self._model.to(self._device)
        self._model.eval()

    def predict(self, premise: str, hypothesis: str) -> dict:
        import torch
        prompt = PROMPT_TEMPLATE.format(premise=premise, hypothesis=hypothesis)
        inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)

        # Next-token logits
        next_token_logits = outputs.logits[0, -1, :]

        # Extract logits for the 3 label tokens
        label_logits = {
            lbl: next_token_logits[tid].item()
            for lbl, tid in self._label_token_ids.items()
        }

        prob_dict = _softmax_dict(label_logits)
        best_label = max(prob_dict, key=prob_dict.get)
        return {
            "label":      best_label,
            "probs":      prob_dict,
            "confidence": prob_dict[best_label],
        }
