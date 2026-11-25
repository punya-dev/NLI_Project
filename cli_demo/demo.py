#!/usr/bin/env python3
"""
NLI Demo — Natural Language Inference
======================================
Run:  python demo.py

A single-launch, menu-driven interactive CLI demo for all 5 NLI models
trained in NLP Phase 3:
  • Logistic Regression (TF-IDF)
  • DistilBERT baseline
  • DistilBERT + Knowledge Distillation (generic pairs)
  • DistilBERT + KD + Extra SNLI labels
  • Llama-3.2-1B Instruct + LoRA  ★ Best
"""
import sys, os, warnings
warnings.filterwarnings("ignore")

# Ensure cli_demo directory is on the path so imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import run

if __name__ == "__main__":
    run()
