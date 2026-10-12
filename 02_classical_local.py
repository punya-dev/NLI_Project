"""Classical ML baselines for SNLI: LogisticRegression, LinearSVC, LightGBM on TF-IDF + engineered features.

Trains on the shared 50K subset (created here on first run and reused by DistilBERT).
Run this script FIRST. One fixed C per linear model, no grid search. Best model (by validation accuracy)
is evaluated once on the test set.
"""
import json
import re
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

SEED = 42
N_TRAIN = 50_000
LABELS = ["entailment", "neutral", "contradiction"]   # 0, 1, 2

PROJECT_DIR = Path(__file__).resolve().parent
CLEAN_DIR = PROJECT_DIR / "results" / "cleaned"
TRAIN_50K = CLEAN_DIR / "train_50k_seed42.csv"
MODELS_DIR = PROJECT_DIR / "models" / "classical"
OUT_DIR = PROJECT_DIR / "results"

TOKEN_PATTERN = r"[a-z0-9]+(?:'[a-z]+)?"
TOKEN_RE = re.compile(TOKEN_PATTERN)
NEG_WORDS = {"not", "no", "nobody", "nothing", "never", "none", "nowhere", "neither", "nor", "cannot"}

TFIDF_PARAMS = dict(max_features=50000, ngram_range=(1, 2), min_df=2, sublinear_tf=True, norm="l2",
                    lowercase=True, token_pattern=TOKEN_PATTERN)
NOVEL_TFIDF_PARAMS = dict(ngram_range=(1, 1), min_df=2, sublinear_tf=True, norm="l2",
                          lowercase=False, token_pattern=r"\S+")
CROSS_HASH_PARAMS = dict(n_features=2 ** 19, binary=True, norm=None, alternate_sign=False,
                         lowercase=False, token_pattern=r"\S+")
CROSS_MAX_NOVEL, CROSS_MAX_DROPPED = 10, 15
TREE_HYP_BOW_PARAMS = dict(max_features=3000, binary=True, lowercase=True, token_pattern=TOKEN_PATTERN)
TREE_NOVEL_BOW_PARAMS = dict(max_features=3000, binary=True, lowercase=False, token_pattern=r"\S+")

LR_C = 1.0
SVC_C = 0.1
LR_PARAMS = dict(solver="lbfgs", max_iter=1000, tol=1e-4, random_state=SEED)
SVC_PARAMS = dict(penalty="l2", loss="squared_hinge", dual=False, tol=1e-4, max_iter=2000, random_state=SEED)
LGBM_PARAMS = dict(objective="multiclass", n_estimators=1500, learning_rate=0.08, num_leaves=63,
                   min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.6,
                   reg_lambda=1.0, random_state=SEED, n_jobs=-1, verbosity=-1)
LGBM_EARLY_STOPPING = 50


# ------------------------------------------------------------------ data
def read_split(path):
    df = pd.read_csv(path, usecols=["premise", "hypothesis", "label"])
    df["premise"] = df["premise"].fillna("").astype(str)
    df["hypothesis"] = df["hypothesis"].fillna("").astype(str)
    df["label"] = df["label"].astype(int)
    if not df["label"].isin([0, 1, 2]).all():
        raise ValueError(f"{path} has labels outside {{0,1,2}}")
    return df


def pairs(df):
    return set(zip(df["premise"], df["hypothesis"]))


def load_data():
    val_df = read_split(CLEAN_DIR / "validation.csv")
    test_df = read_split(CLEAN_DIR / "test.csv")
    test_df.insert(0, "id", np.arange(len(test_df)))          # row index = stable test id

    if not TRAIN_50K.exists():                                 # create the shared subset ONCE
        full = read_split(CLEAN_DIR / "train.csv")
        if len(full) < N_TRAIN:
            raise ValueError(f"Cleaned train has only {len(full)} rows, need {N_TRAIN}")
        sub, _ = train_test_split(full, train_size=N_TRAIN, random_state=SEED, stratify=full["label"])
        sub = sub.reset_index(drop=True)
        sub.to_csv(TRAIN_50K, index=False)
        print(f"[Saved] {TRAIN_50K} (stratified random sample, seed {SEED})")
    train_df = read_split(TRAIN_50K)
    assert len(train_df) == N_TRAIN, f"{TRAIN_50K} has {len(train_df)} rows"

    print("Train class counts:", train_df["label"].value_counts().sort_index().to_dict())
    print(f"Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")
    print(f"Pair overlap (not removed): train-val {len(pairs(train_df) & pairs(val_df))}, "
          f"train-test {len(pairs(train_df) & pairs(test_df))}, val-test {len(pairs(val_df) & pairs(test_df))}")
    return train_df, val_df, test_df


# ------------------------------------------------------------------ features
def tokenize(text):
    return TOKEN_RE.findall(text.lower())


def has_neg(tokens):
    return any(w in NEG_WORDS or w.endswith("n't") for w in tokens)


def analyze_pairs(df):
    """Novel words (in hypothesis, not premise), cross-pair tokens, and 14 dense numeric features."""
    novel_texts, cross_texts, dense = [], [], []
    for prem, hyp in zip(df["premise"], df["hypothesis"]):
        p, h = tokenize(prem), tokenize(hyp)
        ps, hs = set(p), set(h)
        novel = [w for w in dict.fromkeys(h) if w not in ps]
        dropped = [w for w in dict.fromkeys(p) if w not in hs]
        inter, union = len(ps & hs), len(ps | hs)
        pb, hb = set(zip(p, p[1:])), set(zip(h, h[1:]))
        p_neg, h_neg = has_neg(p), has_neg(h)
        novel_texts.append(" ".join(novel))
        cross_texts.append(" ".join(f"{a}__{b}" for a in dropped[:CROSS_MAX_DROPPED] for b in novel[:CROSS_MAX_NOVEL]))
        dense.append([len(p), len(h), len(h) - len(p), min(len(h) / max(1, len(p)), 10.0),
                      inter / max(1, len(hs)), inter / max(1, len(ps)), inter / max(1, union),
                      len(novel), len(dropped), len(hb & pb) / max(1, len(hb)),
                      float(h_neg), float(p_neg), float(h_neg != p_neg), float(len(hs) > 0 and hs <= ps)])
    return {"novel_text": novel_texts, "cross_text": cross_texts, "dense": np.asarray(dense, dtype=np.float64)}


def fit_components(train_df, info):
    corpus = pd.concat([train_df["premise"], train_df["hypothesis"]], ignore_index=True)
    return {
        "tfidf": TfidfVectorizer(**TFIDF_PARAMS).fit(corpus),
        "novel_tfidf": TfidfVectorizer(**NOVEL_TFIDF_PARAMS).fit(info["novel_text"]),
        "cross_hash": HashingVectorizer(**CROSS_HASH_PARAMS),
        "scaler": StandardScaler().fit(info["dense"]),
        "hyp_bin": CountVectorizer(**TREE_HYP_BOW_PARAMS).fit(train_df["hypothesis"]),
        "novel_bin": CountVectorizer(**TREE_NOVEL_BOW_PARAMS).fit(info["novel_text"]),
    }


def build_features(df, info, c):
    """Returns (linear feature matrix, tree feature matrix)."""
    Xp, Xh = c["tfidf"].transform(df["premise"]), c["tfidf"].transform(df["hypothesis"])
    cos = sp.csr_matrix(np.asarray(Xp.multiply(Xh).sum(axis=1)))
    linear = sp.hstack([Xp, Xh, abs(Xp - Xh), cos,
                        c["novel_tfidf"].transform(info["novel_text"]),
                        sp.csr_matrix(c["scaler"].transform(info["dense"])),
                        c["cross_hash"].transform(info["cross_text"]).tocsr()], format="csr")
    tree = sp.hstack([sp.csr_matrix(info["dense"]), cos,
                      c["hyp_bin"].transform(df["hypothesis"]),
                      c["novel_bin"].transform(info["novel_text"])], format="csr").astype(np.float32)
    return linear, tree


# ------------------------------------------------------------------ evaluation
def evaluate(y, pred):
    rep = classification_report(y, pred, target_names=LABELS, output_dict=True, digits=4)
    return {"accuracy": float(accuracy_score(y, pred)),
            "f1_macro": float(f1_score(y, pred, average="macro")),
            "f1_weighted": float(f1_score(y, pred, average="weighted")),
            "per_class": {l: {"precision": rep[l]["precision"], "recall": rep[l]["recall"],
                              "f1": rep[l]["f1-score"]} for l in LABELS}}


# ------------------------------------------------------------------ main
def main():
    t_start = time.time()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    if not all((CLEAN_DIR / f"{split}.csv").is_file() for split in ("train", "validation", "test")):
        raise FileNotFoundError(
            f"Cleaned SNLI CSVs are missing in {CLEAN_DIR}. Run 01_eda_local.py first."
        )

    train_df, val_df, test_df = load_data()
    y_tr, y_val, y_te = train_df["label"].to_numpy(), val_df["label"].to_numpy(), test_df["label"].to_numpy()

    print("Building features...")
    info = {k: analyze_pairs(d) for k, d in [("train", train_df), ("val", val_df), ("test", test_df)]}
    comps = fit_components(train_df, info["train"])
    joblib.dump(comps, MODELS_DIR / "feature_components.pkl", compress=3)
    Xl_tr, Xt_tr = build_features(train_df, info["train"], comps)
    Xl_val, Xt_val = build_features(val_df, info["val"], comps)
    Xl_te, Xt_te = build_features(test_df, info["test"], comps)

    # name -> (model, X_train, X_val, X_test)
    models = {
        "logreg": (LogisticRegression(C=LR_C, **LR_PARAMS), Xl_tr, Xl_val, Xl_te),
        "linearsvc": (LinearSVC(C=SVC_C, **SVC_PARAMS), Xl_tr, Xl_val, Xl_te),
    }
    try:
        import lightgbm as lgb
        models["lightgbm"] = (lgb.LGBMClassifier(**LGBM_PARAMS), Xt_tr, Xt_val, Xt_te)
    except ImportError:
        print("lightgbm not installed, skipping it")

    fitted = {}
    for name, (clf, Xtr, Xv, Xte) in models.items():
        t0 = time.time()
        if name == "lightgbm":
            clf.fit(Xtr, y_tr, eval_set=[(Xv, y_val)],
                    callbacks=[lgb.early_stopping(LGBM_EARLY_STOPPING, verbose=False)])
        else:
            clf.fit(Xtr, y_tr)
        train_time = time.time() - t0
        val_m = evaluate(y_val, clf.predict(Xv))
        fitted[name] = {"model": clf, "val": val_m, "train_time_s": train_time}
        joblib.dump(clf, MODELS_DIR / f"{name}_model.pkl", compress=3)
        print(f"{name:<10} val acc={val_m['accuracy']:.4f} macro-F1={val_m['f1_macro']:.4f} | {train_time:.0f}s")

    best = max(fitted, key=lambda n: fitted[n]["val"]["accuracy"])      # selected on VALIDATION only
    print(f"\nSelected model: {best}")
    clf = fitted[best]["model"]
    X_test = models[best][3]
    pred = clf.predict(X_test)
    test_m = evaluate(y_te, pred)
    print(classification_report(y_te, pred, target_names=LABELS, digits=4))

    out = test_df[["id", "premise", "hypothesis", "label"]].copy()
    out["pred"] = pred
    out.to_csv(OUT_DIR / "classical_test_predictions.csv", index=False)

    results = {
        "model": f"classical ({best})",
        "train_file": str(TRAIN_50K), "n_train": len(train_df), "seed": SEED,
        "hyperparameters": {"logreg_C": LR_C, "linearsvc_C": SVC_C, "tfidf": {k: str(v) for k, v in TFIDF_PARAMS.items()},
                            "lightgbm": {k: str(v) for k, v in LGBM_PARAMS.items()}},
        "validation_all_models": {n: f["val"] for n, f in fitted.items()},
        "train_time_s": {n: f["train_time_s"] for n, f in fitted.items()},
        "selected": best, "validation": fitted[best]["val"], "test": test_m,
        "total_runtime_s": time.time() - t_start,
    }
    with open(OUT_DIR / "classical_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Test acc {test_m['accuracy']:.4f} | macro-F1 {test_m['f1_macro']:.4f} | "
          f"runtime {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
