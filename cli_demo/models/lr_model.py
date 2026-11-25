"""
Logistic Regression model wrapper.

The LR model was trained with a custom feature builder class 'Feats'
defined in the training notebook. We re-define it here (identical code)
so that joblib can unpickle the saved bundle correctly.

Bundle structure: {"feats": Feats, "clf": LogisticRegression, "group": str}
"""
import re
import os
import numpy as np
import scipy.sparse as sp

# ── Re-define exactly what the notebook defined so pickle can find it ─────────
TOKEN_PATTERN = r"[a-z0-9]+(?:'[a-z]+)?"
TOKEN_RE = re.compile(TOKEN_PATTERN)
NEG_WORDS = {"not", "no", "nobody", "nothing", "never", "none", "nowhere",
             "neither", "nor", "cannot"}
CROSS_MAX_NOVEL, CROSS_MAX_DROPPED = 10, 15


def tokenize(t):
    return TOKEN_RE.findall(t.lower())


def has_neg(toks):
    return any(w in NEG_WORDS or w.endswith("n't") for w in toks)


def _analyze_pair(premise: str, hypothesis: str):
    """Analyze a single premise/hypothesis pair — returns (novel_str, cross_str, dense_array)."""
    p, h = tokenize(premise), tokenize(hypothesis)
    ps, hs = set(p), set(h)
    novel   = [w for w in dict.fromkeys(h) if w not in ps]
    dropped = [w for w in dict.fromkeys(p) if w not in hs]
    inter, union = len(ps & hs), len(ps | hs)
    pb, hb = set(zip(p, p[1:])), set(zip(h, h[1:]))
    pn, hn = has_neg(p), has_neg(h)
    novel_t = " ".join(novel)
    cross_t = " ".join(
        f"{a}__{b}"
        for a in dropped[:CROSS_MAX_DROPPED]
        for b in novel[:CROSS_MAX_NOVEL]
    )
    dense = np.array([
        len(p), len(h), len(h) - len(p),
        min(len(h) / max(1, len(p)), 10.0),
        inter / max(1, len(hs)),
        inter / max(1, len(ps)),
        inter / max(1, union),
        len(novel), len(dropped),
        len(hb & pb) / max(1, len(hb)),
        float(hn), float(pn), float(hn != pn),
        float(len(hs) > 0 and hs <= ps),
    ], dtype=np.float64)
    return novel_t, cross_t, dense


def _analyze_df(df):
    """Batch-analyze a DataFrame — matches the original notebook's analyze()."""
    import pandas as _pd
    novel_t_list, cross_t_list, dense_list = [], [], []
    for prem, hyp in zip(df["premise"], df["hypothesis"]):
        n, c, d = _analyze_pair(prem, hyp)
        novel_t_list.append(n)
        cross_t_list.append(c)
        dense_list.append(d)
    return novel_t_list, cross_t_list, np.array(dense_list, dtype=np.float64)


GROUPS = {
    "concat":       ["P", "H"],
    "+diff_cos":    ["P", "H", "diff", "cos"],
    "+dense":       ["P", "H", "diff", "cos", "dense"],
    "+novel_cross": ["P", "H", "diff", "cos", "dense", "novel", "cross"],
}


class Feats:
    """Feature builder — must match the class saved in the joblib exactly."""
    def fit(self, df):
        from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
        from sklearn.preprocessing import StandardScaler
        import pandas as _pd
        novel, _, dense = _analyze_df(df)
        tfidf_params = dict(max_features=50_000, ngram_range=(1, 2), min_df=2,
                            sublinear_tf=True, token_pattern=TOKEN_PATTERN)
        self.tfidf = TfidfVectorizer(**tfidf_params).fit(
            _pd.concat([df["premise"], df["hypothesis"]], ignore_index=True)
        )
        self.novel = TfidfVectorizer(
            lowercase=False, token_pattern=r"\S+", min_df=2, sublinear_tf=True
        ).fit(novel)
        self.scaler = StandardScaler().fit(dense)
        self.cross = HashingVectorizer(
            n_features=2**19, binary=True, norm=None, alternate_sign=False,
            lowercase=False, token_pattern=r"\S+"
        )
        return self

    def blocks(self, df):
        novel, cross, dense = _analyze_df(df)
        P = self.tfidf.transform(df["premise"])
        H = self.tfidf.transform(df["hypothesis"])
        return {
            "P":     P,
            "H":     H,
            "diff":  abs(P - H),
            "cos":   sp.csr_matrix(np.asarray(P.multiply(H).sum(axis=1))),
            "dense": sp.csr_matrix(self.scaler.transform(dense)),
            "novel": self.novel.transform(novel),
            "cross": self.cross.transform(cross).tocsr(),
        }


def _assemble(blocks, group):
    return sp.hstack([blocks[k] for k in GROUPS[group]], format="csr")


# ── The actual model wrapper ──────────────────────────────────────────────────
LABELS = ["entailment", "neutral", "contradiction"]
# The classifier stores classes as integers 0,1,2 → map to string labels
_IDX_TO_LABEL = {0: "entailment", 1: "neutral", 2: "contradiction"}


class LRModel:
    def __init__(self, artifact_path: str):
        if not os.path.exists(artifact_path):
            raise FileNotFoundError(f"LR model not found: {artifact_path}")

        import warnings
        import sys
        import __main__

        # Inject Feats so pickle can find it
        __main__.Feats = Feats

        import joblib
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bundle = joblib.load(artifact_path)

        self._feats = bundle["feats"]
        self._clf   = bundle["clf"]
        self._group = bundle["group"]

        # Determine label ordering from clf.classes_
        # classes_ may be [0,1,2] or ['entailment','neutral','contradiction']
        raw_classes = list(self._clf.classes_)
        if all(isinstance(c, (int, np.integer)) or (isinstance(c, str) and c.isdigit()) for c in raw_classes):
            self._classes = [_IDX_TO_LABEL[int(c)] for c in raw_classes]
        else:
            self._classes = [str(c).lower() for c in raw_classes]

    def predict(self, premise: str, hypothesis: str) -> dict:
        import pandas as pd

        df = pd.DataFrame([{"premise": premise, "hypothesis": hypothesis}])
        blocks = self._feats.blocks(df)
        X = _assemble(blocks, self._group)
        probs = self._clf.predict_proba(X)[0]

        prob_dict = {lbl: float(p) for lbl, p in zip(self._classes, probs)}
        # Ensure all three keys exist
        for lbl in LABELS:
            if lbl not in prob_dict:
                prob_dict[lbl] = 0.0

        best_label = max(prob_dict, key=prob_dict.get)
        return {
            "label":      best_label,
            "probs":      prob_dict,
            "confidence": prob_dict[best_label],
        }
