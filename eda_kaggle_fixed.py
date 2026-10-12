"""SNLI: Complete Data Loading, Cleaning and EDA (single script).



Usage:

    python src/eda.py                         # defaults: <project_root>/data/cleaned, <project_root>/results

    python src/eda.py --data-dir DIR --out-dir DIR

    python src/eda.py --force-download        # re-download from Hugging Face and overwrite CSVs

    python src/eda.py --skip-hyp-only         # skip the slow-ish artifact baselines

    python src/eda.py --skip-tokenizer        # skip the DistilBERT token-length section



Sections:

  1. Load SNLI (Hugging Face, or cached CSVs), drop label == -1, export cleaned CSVs

  2. Data integrity (nulls, empties, duplicates, unique premises, hypotheses per premise)

  3. Class balance (table + plot)

  4. Length statistics (all splits) + length distribution plot (train)

  5. Random sample of training examples

  6. Per-class statistics in train (hypothesis length, negation, word coverage) + plot

  7. Cross-split leakage check (premise / pair overlap)

  8. Vocabulary and OOV statistics

  9. DistilBERT token-length check (justifies max_length)

 10. Annotation-artifact baselines: hypothesis-only and premise-only TF-IDF + LR

 11. Before/after preprocessing examples (matches the Phase 2 TF-IDF settings)



Outputs:

  results/cleaned/{train,validation,test}.csv    (columns: premise, hypothesis, label)

  results/class_balance.png

  results/sentence_length_distribution.png

  results/per_class_statistics.png

  results/eda_summary.json

"""

import argparse

import json

import re

from pathlib import Path

from typing import Any, Dict, List, Optional, Tuple, cast



import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sns



# ----------------------------------------------------------------------------- config

LABEL_MAP = {0: "entailment", 1: "neutral", 2: "contradiction"}

LABELS = [LABEL_MAP[i] for i in range(3)]

SPLITS = ["train", "validation", "test"]

BERT_CKPT = "distilbert-base-uncased"  # change if you use a different checkpoint

NEG = {"not", "no", "nobody", "nothing", "never", "none", "nowhere", "neither", "nor"}



# Must match the Phase 2 classical baseline so the "after preprocessing" view is truthful

TFIDF_KWARGS: Dict[str, Any] = dict(max_features=20000, ngram_range=(1, 2), sublinear_tf=True, norm="l2")



SEP = "=" * 75





def header(title: str):

    print("\n" + SEP)

    print(title)

    print(SEP)





def words(s):

    """Simple lowercase word tokenizer used for EDA statistics."""

    return re.findall(r"[a-z]+(?:'[a-z]+)?", str(s).lower())





def has_neg(ws):

    return any(w in NEG or w.endswith("n't") for w in ws)





# ----------------------------------------------------------------------------- paths

def get_project_root() -> Path:

    try:

        return Path(__file__).resolve().parent.parent

    except NameError:  # notebooks / Kaggle

        return Path.cwd()





def parse_args():
    root = get_project_root()
    p = argparse.ArgumentParser(description="SNLI EDA + cleaning")
    p.add_argument("--data-dir", type=Path, default=root / "data" / "cleaned",
                   help="Read-only directory containing train.csv, validation.csv, and test.csv")
    p.add_argument("--out-dir", type=Path, default=root / "results",
                   help="Writable directory for cleaned CSVs, plots, and summary JSON")
    p.add_argument("--force-download", action="store_true",
                   help="Download SNLI from Hugging Face instead of using local CSVs")
    p.add_argument("--skip-hyp-only", action="store_true")
    p.add_argument("--skip-tokenizer", action="store_true")
    args, _ = p.parse_known_args()  # tolerant of extra notebook args

    # Do not create or write to data_dir: on Kaggle, /kaggle/input is read-only.
    args.out_dir.mkdir(parents=True, exist_ok=True)
    return args



# ----------------------------------------------------------------------------- 1. load & clean

def load_and_clean(data_dir: Path, out_dir: Path, force_download: bool) -> dict:
    """Load local CSV splits when present, otherwise download SNLI; write cleaned splits to out_dir/cleaned."""
    header("1. LOADING SNLI AND CLEANING INVALID LABELS")

    cleaned_dir = out_dir / "cleaned"
    cleaned_dir.mkdir(parents=True, exist_ok=True)
    cached = all((data_dir / f"{split}.csv").is_file() for split in SPLITS)

    if cached and not force_download:
        print(f"Found CSV splits in {data_dir}; loading them.")
        dfs = {}
        for split in SPLITS:
            df = pd.read_csv(data_dir / f"{split}.csv")
            required = {"premise", "hypothesis", "label"}
            missing = required - set(df.columns)
            if missing:
                raise ValueError(
                    f"{data_dir / f'{split}.csv'} is missing required columns: {sorted(missing)}"
                )

            # Keep valid SNLI labels and normalize missing sentence values.
            df["label"] = pd.to_numeric(df["label"], errors="coerce")
            df = df[df["label"].isin([0, 1, 2])].copy()
            df["label"] = df["label"].astype(int)
            df["premise"] = df["premise"].fillna("").astype(str)
            df["hypothesis"] = df["hypothesis"].fillna("").astype(str)
            df = df[["premise", "hypothesis", "label"]].reset_index(drop=True)
            dfs[split] = df
            print(f"  {split:<12}: {len(df):>9,d} cleaned rows")

    else:
        from datasets import load_dataset

        print("Loading SNLI from Hugging Face...")
        dataset = load_dataset("stanfordnlp/snli")
        dfs = {}
        print(f"\\n{'Split':<15} {'Raw Count':<15} {'Cleaned Count':<15} {'Removed (invalid label)':<22}")
        print("-" * 70)
        for split in SPLITS:
            raw = cast(pd.DataFrame, dataset[split].to_pandas())
            clean = raw[raw["label"].isin([0, 1, 2])].copy()
            clean["premise"] = clean["premise"].fillna("").astype(str)
            clean["hypothesis"] = clean["hypothesis"].fillna("").astype(str)
            clean = clean[["premise", "hypothesis", "label"]].reset_index(drop=True)
            clean["label"] = clean["label"].astype(int)
            dfs[split] = clean
            print(
                f"{split:<15} {len(raw):<15,d} {len(clean):<15,d} "
                f"{len(raw) - len(clean):<22,d}"
            )

    print(f"\\nExporting cleaned splits to {cleaned_dir}:")
    for split, df in dfs.items():
        path = cleaned_dir / f"{split}.csv"
        df.to_csv(path, index=False)
        print(f"[Saved] {split:<12} -> {path} ({len(df):,d} rows)")
    return dfs



# ----------------------------------------------------------------------------- 2. integrity

def data_integrity(splits: dict) -> dict:

    header("2. DATA INTEGRITY")

    integ = {}

    for name, df in splits.items():

        prem = df["premise"].fillna("").astype(str)

        hyp = df["hypothesis"].fillna("").astype(str)

        counts = df["label"].value_counts().sort_index()

        n_prem = int(prem.nunique())

        integ[name] = {

            "rows": len(df),

            "invalid_labels": int((~df["label"].isin([0, 1, 2])).sum()),

            "null_premise": int(df["premise"].isna().sum()),

            "null_hypothesis": int(df["hypothesis"].isna().sum()),

            "empty_premise": int((prem.str.strip() == "").sum()),

            "empty_hypothesis": int((hyp.str.strip() == "").sum()),

            "duplicate_pairs": int(df.duplicated(["premise", "hypothesis"]).sum()),

            "unique_premises": n_prem,

            "unique_hypotheses": int(hyp.nunique()),

            "avg_hypotheses_per_premise": round(len(df) / max(1, n_prem), 2),

            "class_counts": {LABELS[i]: int(counts.get(i, 0)) for i in range(3)},

            "class_pct": {LABELS[i]: round(100 * counts.get(i, 0) / len(df), 2) for i in range(3)},

        }

        print(name, json.dumps(integ[name], indent=2))

    return integ





# ----------------------------------------------------------------------------- 3. class balance

def class_balance(splits: dict, out_dir: Path):

    header("3. CLASS DISTRIBUTION")

    rows = []

    for name, df in splits.items():

        names = df["label"].map(LABEL_MAP)

        counts = names.value_counts()

        pcts = names.value_counts(normalize=True) * 100

        print(f"\n--- {name} (Total: {len(df):,}) ---")

        print(pd.DataFrame({"Count": counts, "Percentage (%)": pcts.round(2)}).to_string())

        for lab, c in counts.items():

            rows.append({"Split": name.capitalize(), "Label": lab.capitalize(),

                         "Count": c, "Percentage": pcts[lab]})

    plot_df = pd.DataFrame(rows)



    sns.set_theme(style="whitegrid", palette="muted")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.barplot(data=plot_df, x="Split", y="Count", hue="Label", ax=axes[0])

    axes[0].set_title("Class Counts by Split", fontsize=13, fontweight="bold", pad=10)

    axes[0].set_ylabel("Number of Samples")

    sns.barplot(data=plot_df, x="Split", y="Percentage", hue="Label", ax=axes[1])

    axes[1].set_title("Class Percentages by Split", fontsize=13, fontweight="bold", pad=10)

    axes[1].set_ylabel("Percentage (%)")

    axes[1].set_ylim(0, 40)

    axes[1].axhline(33.33, color="gray", linestyle="--", alpha=0.7, label="Balanced (33.3%)")

    axes[1].legend(title="Label")

    plt.tight_layout()

    path = out_dir / "class_balance.png"

    plt.savefig(path, dpi=300)

    plt.close()

    print(f"\n[Saved] {path}")





# ----------------------------------------------------------------------------- 4. lengths

def length_statistics(splits: dict, out_dir: Path) -> list:

    header("4. LENGTH STATISTICS (whitespace word counts)")

    rows = []

    lens = {}

    for name, df in splits.items():

        for col in ["premise", "hypothesis"]:

            L = df[col].fillna("").astype(str).str.split().map(len)

            lens[(name, col)] = L

            rows.append({"split": name, "field": col, "mean": round(L.mean(), 2),

                         "median": int(L.median()), "p95": round(L.quantile(0.95), 1),

                         "p99": round(L.quantile(0.99), 1), "max": int(L.max())})

    print(pd.DataFrame(rows).to_string(index=False))



    pl, hl = lens[("train", "premise")], lens[("train", "hypothesis")]

    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, series, color, title in [(axes[0], pl, "#3470a3", "Premise"),

                                     (axes[1], hl, "#e07a5f", "Hypothesis")]:

        bins = cast(Any, range(1, int(series.quantile(0.99)) + 2))

        sns.histplot(series, bins=bins, kde=True, color=color, ax=ax)

        ax.axvline(series.mean(), color="red", linestyle="--", label=f"Mean: {series.mean():.1f}")

        ax.axvline(series.median(), color="green", linestyle="-", label=f"Median: {series.median():.0f}")

        ax.axvline(series.quantile(0.95), color="orange", linestyle=":",

                   label=f"95th %ile: {series.quantile(0.95):.1f}")

        ax.set_title(f"{title} Word-Count Distribution (train)", fontsize=13, fontweight="bold", pad=10)

        ax.set_xlabel("Word Count")

        ax.legend()

    axes[0].set_ylabel("Frequency")

    plt.tight_layout()

    path = out_dir / "sentence_length_distribution.png"

    plt.savefig(path, dpi=300)

    plt.close()

    print(f"\n[Saved] {path}")

    return rows





# ----------------------------------------------------------------------------- 5. samples

def sample_examples(train: pd.DataFrame, n: int = 15):

    header(f"5. RANDOM SAMPLE OF {n} TRAINING EXAMPLES")

    for i, (_, r) in enumerate(train.sample(n=n, random_state=42).iterrows(), 1):

        lab_id = int(float(str(r["label"])))

        print(f"\n[{i}/{n}]")

        print(f"  Premise    : {r['premise']}")

        print(f"  Hypothesis : {r['hypothesis']}")

        print(f"  Label      : {LABEL_MAP[lab_id].upper()} (ID: {lab_id})")

        print("-" * 70)





# ----------------------------------------------------------------------------- 6. per-class stats

def per_class_stats(train: pd.DataFrame, out_dir: Path) -> dict:

    header("6. PER-CLASS STATISTICS (train)")

    tw_p = train["premise"].fillna("").map(words)

    tw_h = train["hypothesis"].fillna("").map(words)

    tr = train.assign(

        hyp_len=tw_h.map(len),

        prem_len=tw_p.map(len),

        hyp_has_negation=tw_h.map(has_neg).astype(float),

        hyp_word_coverage=[len(set(h) & set(p)) / max(1, len(set(h))) for h, p in zip(tw_h, tw_p)],

    )

    cols = ["hyp_len", "prem_len", "hyp_has_negation", "hyp_word_coverage"]

    grouped = pd.DataFrame(tr.groupby("label")[cols].mean())

    pc = grouped.rename(index=LABEL_MAP).round(3)

    print(pc.to_string())

    print("(hyp_word_coverage = fraction of hypothesis word types that also appear in the premise)")



    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for ax, col, title in zip(axes, ["hyp_len", "hyp_has_negation", "hyp_word_coverage"],

                              ["Mean hypothesis length (words)", "Negation rate in hypothesis",

                                "Hypothesis word coverage by premise"]):

        sns.barplot(x=list(pc.index.str.capitalize()), y=list(pc[col]), ax=ax)

        ax.set_title(title, fontsize=12, fontweight="bold")

        ax.set_xlabel("")

    plt.tight_layout()

    path = out_dir / "per_class_statistics.png"

    plt.savefig(path, dpi=300)

    plt.close()

    print(f"\n[Saved] {path}")

    return pc.to_dict()





# ----------------------------------------------------------------------------- 7. leakage

def leakage_check(splits: dict) -> dict:

    header("7. CROSS-SPLIT LEAKAGE CHECK")

    prem = {s: set(df["premise"].fillna("").astype(str)) for s, df in splits.items()}

    pair = {s: set(zip(df["premise"].fillna("").astype(str), df["hypothesis"].fillna("").astype(str)))

            for s, df in splits.items()}

    res = {}

    for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:

        res[f"{a}_vs_{b}"] = {

            "shared_premises": len(prem[a] & prem[b]),

            "shared_pairs": len(pair[a] & pair[b]),

            "pct_of_{}_premises_in_{}".format(b, a):

                round(100 * len(prem[a] & prem[b]) / max(1, len(prem[b])), 3),

        }

        print(f"{a} vs {b}: {json.dumps(res[f'{a}_vs_{b}'])}")

    print("(SNLI keeps premises disjoint across splits, so ~0 shared premises is expected)")

    return res





# ----------------------------------------------------------------------------- 8. vocab / OOV

def vocab_oov(splits: dict) -> dict:

    header("8. VOCABULARY AND OOV STATISTICS")

    def toks(df):

        return [w for t in pd.concat([df["premise"], df["hypothesis"]]).fillna("") for w in words(t)]



    train_tokens = toks(splits["train"])

    train_vocab = set(train_tokens)

    res: Dict[str, Any] = {"train_vocab_size": len(train_vocab), "train_tokens": len(train_tokens)}

    print(f"Train vocabulary (types): {len(train_vocab):,} | tokens: {len(train_tokens):,}")

    for s in ["validation", "test"]:

        t = toks(splits[s])

        types = set(t)

        oov_tok = sum(1 for w in t if w not in train_vocab)

        oov_typ = len(types - train_vocab)

        res[s] = {

            "vocab_size": len(types),

            "oov_token_rate_pct": round(100 * oov_tok / max(1, len(t)), 3),

            "oov_type_rate_pct": round(100 * oov_typ / max(1, len(types)), 3),

        }

        print(f"{s}: types={len(types):,} | OOV tokens={res[s]['oov_token_rate_pct']}% "

              f"| OOV types={res[s]['oov_type_rate_pct']}%")

    return res





# ----------------------------------------------------------------------------- 9. token lengths



def token_lengths(train: pd.DataFrame):

    header(f"9. TOKEN LENGTHS ({BERT_CKPT}, premise+hypothesis pair)")



    try:

        from transformers import AutoTokenizer, PreTrainedTokenizerBase



        tok = AutoTokenizer.from_pretrained(BERT_CKPT)



        # Narrow the type before calling the tokenizer

        if not isinstance(tok, PreTrainedTokenizerBase):

            raise TypeError("Expected a Hugging Face tokenizer")



        samp = train.sample(

            n=min(50000, len(train)),

            random_state=42,

        )



        enc = tok(

            samp["premise"].fillna("").astype(str).tolist(),

            text_pair=samp["hypothesis"].fillna("").astype(str).tolist(),

            add_special_tokens=True,

            truncation=False,

        )



        tl = np.array([len(x) for x in enc["input_ids"]])



        stats = {

            "mean": float(tl.mean()),

            "p95": float(np.percentile(tl, 95)),

            "p99": float(np.percentile(tl, 99)),

            "max": int(tl.max()),

            "frac_over_64": float((tl > 64).mean()),

            "frac_over_128": float((tl > 128).mean()),

        }



        print(json.dumps(stats, indent=2))

        return tok, stats



    except Exception as e:

        print("Skipped (tokenizer not available):", e)

        return None, None







# ----------------------------------------------------------------------------- 10. artifact baselines

def artifact_baselines(splits: dict) -> dict:

    header("10. ANNOTATION-ARTIFACT BASELINES (single-sentence TF-IDF + LR)")

    from sklearn.feature_extraction.text import TfidfVectorizer

    from sklearn.linear_model import LogisticRegression

    from sklearn.metrics import accuracy_score, f1_score



    print("Settings: TF-IDF (1,2)-grams, max_features=20000, sublinear_tf, min_df=2; LR C=1.0, max_iter=1000")

    print("(These are EDA diagnostics and differ slightly from the full Phase 2 baseline.)")

    out = {}

    for field in ["hypothesis", "premise"]:

        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=20000, sublinear_tf=True)

        Xtr = vec.fit_transform(splits["train"][field].fillna(""))

        Xte = vec.transform(splits["test"][field].fillna(""))

        clf = LogisticRegression(max_iter=1000, C=1.0).fit(Xtr, splits["train"]["label"])

        pred = clf.predict(Xte)

        res: Dict[str, Any] = {

            "accuracy": float(accuracy_score(splits["test"]["label"], pred)),

            "macro_f1": float(f1_score(splits["test"]["label"], pred, average="macro"))

        }

        print(f"\n{field.capitalize()}-only on test: acc={res['accuracy']:.4f}, macro-F1={res['macro_f1']:.4f}")

        if field == "hypothesis":

            print("(chance = ~0.33; well above that means the hypothesis alone leaks the label)")

            coefs = pd.DataFrame(clf.coef_.T, index=vec.get_feature_names_out(), columns=LABELS)

            res["top_cues"] = {c: pd.Series(coefs[c]).nlargest(10).index.tolist() for c in LABELS}

            for c in LABELS:

                print(f"  top hypothesis cues for {c}: {res['top_cues'][c]}")

        else:

            print("(premise-only should be near chance, since each premise appears with all 3 labels)")

        out[f"{field}_only"] = res

    return out





# ----------------------------------------------------------------------------- 11. before/after

def preprocessing_examples(test: pd.DataFrame, tok):

    header("11. BEFORE / AFTER PREPROCESSING EXAMPLES")

    from sklearn.feature_extraction.text import TfidfVectorizer

    analyzer = TfidfVectorizer(**TFIDF_KWARGS).build_analyzer()

    print("Classical pipeline: lowercase, default token pattern (drops 1-char tokens like 'a'), "

          "unigrams + bigrams,\nno stopword removal / stemming (negations and function words carry NLI signal).")



    ex = pd.concat([test[test["label"] == c].sample(1, random_state=42) for c in range(3)])

    for _, r in ex.iterrows():

        p, h = str(r["premise"]), str(r["hypothesis"])

        lab_id = int(float(str(r["label"])))

        print(f"\nLABEL: {LABEL_MAP[lab_id]}")

        print(f"  RAW premise      : {p}")

        print(f"  RAW hypothesis   : {h}")

        print(f"  TF-IDF premise   : {analyzer(p)}")

        print(f"  TF-IDF hypothesis: {analyzer(h)}")

        if tok is not None:

            e = tok(p, h)

            print(f"  DISTILBERT toks  : {tok.convert_ids_to_tokens(e['input_ids'])}")

            print(f"  DISTILBERT ids   : {e['input_ids']}  (len={len(e['input_ids'])})")

        print(f"  LLM user message : Premise: {p}\\\nHypothesis: {h}\\\nAnswer:")





# ----------------------------------------------------------------------------- main

def main():

    args = parse_args()

    summary = {}



    splits = load_and_clean(args.data_dir, args.out_dir, args.force_download)

    summary["integrity"] = data_integrity(splits)

    class_balance(splits, args.out_dir)

    summary["lengths"] = length_statistics(splits, args.out_dir)

    sample_examples(splits["train"])

    summary["per_class_train"] = per_class_stats(splits["train"], args.out_dir)

    summary["leakage"] = leakage_check(splits)

    summary["vocab_oov"] = vocab_oov(splits)



    tok = None

    if not args.skip_tokenizer:

        tok, tok_stats = token_lengths(splits["train"])

        if tok_stats:

            summary["token_lengths"] = tok_stats



    if not args.skip_hyp_only:

        summary["artifact_baselines"] = artifact_baselines(splits)



    preprocessing_examples(splits["test"], tok)



    json_path = args.out_dir / "eda_summary.json"

    with open(json_path, "w") as f:

        json.dump(summary, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))



    header("EDA AND CLEANING COMPLETED SUCCESSFULLY")

    print(f"[Saved] {json_path}")





if __name__ == "__main__":

    main()