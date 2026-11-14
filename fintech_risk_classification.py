#!/usr/bin/env python3
"""
FinTech Agile User Stories -> EU AI Act Risk Classification  (complete project, one file)
=========================================================================================

Dataset : "FinTech Software Product Requirements, Agile User Stories, and EU AI Act
          Risk Classification Dataset" (Mendeley Data, DOI 10.17632/ww3wmkw466.1)
Task    : read a FinTech user story  ->  predict its EU AI Act risk tier
          (Minimal-Risk / Limited-Risk / High-Risk / Unacceptable)

HOW TO READ THIS FILE
---------------------
The file is organised in numbered parts that mirror the data-science workflow.
`main()` at the bottom calls them in order, so read it first.

    PART 0  Configuration, colours, small helpers
    PART 1  Load the data and detect its columns
    PART 2  Exploratory data analysis (EDA)
    PART 3  Data cleaning + a data-quality audit
    PART 4  Encode categories and split the data
    PART 5  Feature engineering experiments (which text representation works best?)
    PART 6  Train, tune and predict with several models
    PART 7  Evaluate: accuracy, F1, confidence intervals, significance tests
    PART 8  Extra analysis: clustering, per-domain results, error analysis
    PART 9  Visualisations (every figure is saved as a PNG)
    PART 10 Auto-generated results report (outputs/RESULTS.md)
    PART 11 Predict the tier of a new user story with the saved model

RUN IT
------
    python fintech_risk_classification.py --data data/<file>.csv
    python fintech_risk_classification.py --demo          # synthetic smoke test, NOT real results
    python fintech_risk_classification.py --predict "As a borrower, I want an AI to approve my loan ..."

Nothing in the results is hard-coded: every number is computed from the data.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # draw to files, no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import binomtest
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    normalized_mutual_info_score,
    precision_score,
    recall_score,
    silhouette_score,
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    learning_curve,
    train_test_split,
)
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import LabelEncoder, Normalizer
from sklearn.svm import LinearSVC

# =========================================================================== #
# PART 0 - CONFIGURATION, COLOURS, SMALL HELPERS
# =========================================================================== #
SEED = 42  # fixed seed => the same data always gives the same results

# The four EU AI Act tiers, ordered from least to most serious.
TIERS = ["Minimal-Risk", "Limited-Risk", "High-Risk", "Unacceptable"]
HIGH_RISK = "High-Risk"  # the tier we care most about not missing

# One fixed colour per tier, used in every figure (colour follows the class).
TIER_COLORS = {"High-Risk": "#2a78d6", "Limited-Risk": "#eb6834", "Minimal-Risk": "#1baf7a", "Unacceptable": "#eda100"}
CLUSTER_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e3e2dd"

# Common English words are ignored; "want/able" appear in every "As a..I want.." story.
STOP_WORDS = sorted(ENGLISH_STOP_WORDS | {"want", "wants", "able", "like", "would"})


@dataclass
class Config:
    """All settings in one place (filled from the command line)."""

    data: str | None
    text_cols: list[str] | None
    label_col: str | None
    domain_col: str | None
    out: Path
    test_size: float
    cv_folds: int
    demo: bool
    bootstrap: int = 1000


class Tee:
    """Send everything that is printed to the screen AND to outputs/run_log.txt."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for s in self.streams:
            s.write(text)

    def flush(self):
        for s in self.streams:
            s.flush()


def banner(title: str):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def tier_color(name: str) -> str:
    return TIER_COLORS.get(name, "#6250d6")


def style_axes(ax, title=None, xlabel=None, ylabel=None):
    """Clean, minimal chart style shared by every figure."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, loc="left", fontsize=11, color=INK, fontweight="bold")
    if xlabel is not None:
        ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
    if ylabel is not None:
        ax.set_ylabel(ylabel, color=MUTED, fontsize=9)


def horizontal_grid(ax):
    """For horizontal bar charts: light vertical grid instead of horizontal."""
    ax.grid(axis="x", color=GRID)
    ax.grid(axis="y", visible=False)


def save_fig(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)
    print(f"   figure saved: {path.name}")


def df_to_markdown(df: pd.DataFrame, float_fmt="{:.3f}") -> str:
    """Tiny DataFrame -> Markdown table converter (avoids an extra dependency)."""
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.iterrows():
        cells = ["-" if isinstance(v, (float, np.floating)) and np.isnan(v)
                 else float_fmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# =========================================================================== #
# PART 1 - LOAD THE DATA AND DETECT ITS COLUMNS
# =========================================================================== #
def load_data(path: Path) -> pd.DataFrame:
    """Read CSV / Excel / JSON. CSVs are tried in several encodings."""
    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xls", ".xlsm"):
        return pd.read_excel(path)
    if suffix == ".json":
        return pd.read_json(path)
    last_error = None
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(path, encoding=encoding, sep=None, engine="python")
        except Exception as err:  # noqa: BLE001 - try the next encoding
            last_error = err
    raise RuntimeError(f"Could not read {path}: {last_error}")


def find_data_file(explicit: str | None) -> Path:
    """Use --data if given, otherwise the first dataset file found in ./data or next to this script."""
    if explicit:
        path = Path(explicit)
        if not path.exists():
            sys.exit(f"Data file not found: {path}")
        return path
    here = Path(__file__).resolve().parent
    for folder in (here / "data", Path.cwd() / "data", here):
        if folder.exists():
            for pattern in ("*.csv", "*.xlsx", "*.xls", "*.json"):
                hits = [h for h in sorted(folder.glob(pattern)) if "metrics" not in h.name and "predictions" not in h.name]
                if hits:
                    return hits[0]
    sys.exit("No dataset found. Download it from Mendeley Data (DOI 10.17632/ww3wmkw466.1), put it in ./data/ or pass --data <file>.")


def detect_columns(df: pd.DataFrame, text_cols, label_col, domain_col):
    """
    Work out which column is the user story, which is the risk label and which is
    the FinTech domain, by looking at column names and contents.
    Any of them can be forced from the command line.
    """
    cols = list(df.columns)
    simple = {c: re.sub(r"[^a-z0-9]+", "_", str(c).lower()).strip("_") for c in cols}

    if label_col is None:  # a small column (<= 8 distinct values) named like "risk", "label", "tier"...
        candidates = [c for c in cols if re.search(r"risk|label|tier|class|level", simple[c]) and df[c].nunique(dropna=True) <= 8]
        candidates.sort(key=lambda c: (0 if "risk" in simple[c] else 1, df[c].nunique(dropna=True)))
        if not candidates:
            sys.exit(f"Could not detect the label column. Columns: {cols}. Use --label-col.")
        label_col = candidates[0]
    elif label_col not in cols:
        sys.exit(f"--label-col '{label_col}' not in columns: {cols}")

    if text_cols:
        missing = [c for c in text_cols if c not in cols]
        if missing:
            sys.exit(f"--text-cols {missing} not in columns: {cols}")
    else:  # text columns are named like "story"/"requirement"/"description", else take the longest text column
        text_cols = [c for c in cols if c != label_col and pd.api.types.is_string_dtype(df[c])
                     and re.search(r"story|requirement|description|feature|text", simple[c])]
        if not text_cols:
            texty = [c for c in cols if c != label_col and pd.api.types.is_string_dtype(df[c])]
            if not texty:
                sys.exit(f"Could not detect a text column. Columns: {cols}. Use --text-cols.")
            text_cols = [max(texty, key=lambda c: df[c].astype(str).str.len().mean())]

    if domain_col is None:  # optional: used for analysis only, never as a model feature
        candidates = [c for c in cols if c not in text_cols and c != label_col
                      and re.search(r"domain|area|sector|segment|category|product|module", simple[c])
                      and 2 <= df[c].nunique(dropna=True) <= 12]
        domain_col = candidates[0] if candidates else None
    return list(text_cols), label_col, domain_col


# =========================================================================== #
# PART 2 - EXPLORATORY DATA ANALYSIS (on the raw, uncleaned data)
# =========================================================================== #
def canonical_label(value) -> str | None:
    """Map any spelling of a label ('high risk', 'HIGH-RISK', 'Unacceptable/Blocked') to one of the 4 tiers."""
    if pd.isna(value):
        return None
    v = re.sub(r"[^a-z]+", " ", str(value).lower()).strip()
    if any(word in v for word in ("unacceptable", "prohibit", "blocked", "banned")):
        return "Unacceptable"
    if "high" in v:
        return "High-Risk"
    if "limited" in v or "transparen" in v:
        return "Limited-Risk"
    if "minimal" in v or v in ("low", "low risk", "none", "no risk"):
        return "Minimal-Risk"
    return None


def run_eda(df, text_cols, label_col, domain_col, fig_dir: Path) -> dict:
    """Describe the raw data: size, types, missing values, duplicates, class balance, text length."""
    banner("PART 2 - EXPLORATORY DATA ANALYSIS")
    text = df[text_cols].fillna("").astype(str).agg(" ".join, axis=1)
    words = text.str.split().str.len()

    info = {
        "rows": int(len(df)),
        "columns": [str(c) for c in df.columns],
        "missing_per_column": {str(k): int(v) for k, v in df.isna().sum().items()},
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_texts": int(text.str.lower().str.strip().duplicated().sum()),
        "raw_label_counts": {str(k): int(v) for k, v in df[label_col].value_counts(dropna=False).items()},
        "words_per_story": {k: round(float(v), 2) for k, v in words.describe().items()},
    }
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Story column(s): {text_cols} | label column: '{label_col}' | domain column: {domain_col}")
    print("\nMissing values per column:\n" + df.isna().sum().to_string())
    print(f"\nDuplicate rows: {info['duplicate_rows']} | duplicate story texts: {info['duplicate_texts']}")
    print("\nRaw label counts:\n" + df[label_col].value_counts(dropna=False).to_string())
    print("\nWords per story:\n" + words.describe().round(1).to_string())

    # Figure 01 - how many stories per (raw) label?
    counts = df[label_col].fillna("(missing)").astype(str).value_counts()
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(counts.index, counts.values, color=[tier_color(c) for c in counts.index], width=0.6)
    for bar, value in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value} ({value / counts.sum():.0%})", ha="center", va="bottom", fontsize=9, color=INK)
    style_axes(ax, "Risk-tier distribution (raw labels)", ylabel="User stories")
    ax.set_ylim(0, counts.max() * 1.15)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    save_fig(fig, fig_dir / "01_class_distribution_raw.png")

    # Figure 02 - are risky stories longer/shorter? (spelling variants grouped for this plot)
    shown = df[label_col].map(lambda v: canonical_label(v) or ("(missing)" if pd.isna(v) else str(v).strip())).astype(str)
    table = pd.DataFrame({"words": words, "label": shown})
    unique = sorted(table["label"].unique())
    order = [c for c in TIERS if c in unique] + [c for c in unique if c not in TIERS]
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(data=table, x="label", y="words", order=order, hue="label", legend=False, ax=ax, width=0.5, linewidth=1,
                palette={c: tier_color(c) for c in order})
    style_axes(ax, "Story length by risk tier", xlabel="", ylabel="Words per story")
    save_fig(fig, fig_dir / "02_text_length_by_class.png")

    # Figure 03 - does the FinTech domain relate to the risk tier?
    if domain_col is not None:
        crosstab = pd.crosstab(df[domain_col].astype(str), df[label_col].astype(str))
        info["domain_vs_label"] = {str(k): {str(a): int(b) for a, b in v.items()} for k, v in crosstab.to_dict().items()}
        print(f"\nDomain ('{domain_col}') vs risk tier:\n" + crosstab.to_string())
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        sns.heatmap(crosstab, annot=True, fmt="d", cmap="Blues", cbar=False, linewidths=2, linecolor="white", ax=ax)
        ax.set_title("Domain vs risk tier (counts)", loc="left", fontsize=11, fontweight="bold", color=INK)
        ax.set_xlabel("")
        ax.set_ylabel("")
        save_fig(fig, fig_dir / "03_domain_vs_risk.png")

    # Figure 04 - which words characterise each tier? (mean TF-IDF weight per tier)
    labels = df[label_col].map(canonical_label).fillna(df[label_col].astype(str))
    classes = [c for c in TIERS if c in set(labels)] or sorted(labels.unique())
    if len(classes) >= 2 and len(df) >= 10:
        try:
            vec = TfidfVectorizer(stop_words=STOP_WORDS, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
            matrix = vec.fit_transform(text.str.lower())
            terms = np.array(vec.get_feature_names_out())
            ncol, nrow = 2, int(np.ceil(len(classes) / 2))
            fig, axes = plt.subplots(nrow, ncol, figsize=(10, 3.2 * nrow), squeeze=False)
            for ax in axes.ravel():
                ax.axis("off")
            for ax, cls in zip(axes.ravel(), classes):
                ax.axis("on")
                mean_weight = np.asarray(matrix[(labels == cls).values].mean(axis=0)).ravel()
                top = mean_weight.argsort()[-8:]
                ax.barh(terms[top], mean_weight[top], color=tier_color(cls), height=0.65)
                style_axes(ax, str(cls), xlabel="mean TF-IDF weight")
                horizontal_grid(ax)
            fig.suptitle("Most characteristic terms per risk tier", x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
            save_fig(fig, fig_dir / "04_top_terms_per_class.png")
        except ValueError:
            print("   (top-terms figure skipped: vocabulary too small)")
    return info


# =========================================================================== #
# PART 3 - DATA CLEANING + DATA-QUALITY AUDIT
# =========================================================================== #
def clean_text(s: str) -> str:
    """lower-case, drop URLs, punctuation and standalone numbers, collapse spaces."""
    s = str(s).lower()
    s = re.sub(r"https?://\S+", " ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\b\d+\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def clean_data(df, text_cols, label_col, min_class_size: int):
    """
    Turn the raw table into a clean table with three new columns:
        text_raw   original story (kept for reports / error analysis)
        text_clean normalised story (used by the models)
        label      one of the canonical EU AI Act tiers
    Every removal is counted in `log` so nothing disappears silently.
    """
    banner("PART 3 - DATA CLEANING")
    log = {"rows_in": int(len(df))}
    out = df.copy()
    out["text_raw"] = out[text_cols].fillna("").astype(str).apply(lambda row: " ".join(x.strip() for x in row if x.strip()), axis=1)

    # a) rows with no story or no label cannot be used
    n = len(out)
    out = out[(out["text_raw"].str.len() > 0) & out[label_col].notna()]
    log["dropped_missing_text_or_label"] = int(n - len(out))

    # b) one spelling per tier
    out["label"] = out[label_col].map(canonical_label)
    if out["label"].notna().sum() == 0:  # dataset uses its own label names -> keep them as they are
        out["label"] = out[label_col].astype(str).str.strip()
        log["label_mapping"] = "kept original labels (no EU AI Act tier names recognised)"
    else:
        unknown = out.loc[out["label"].isna(), label_col].astype(str).unique().tolist()
        if unknown:
            print(f"   WARNING: unrecognised label values will be dropped: {unknown}")
        log["label_mapping"] = "mapped to canonical EU AI Act tiers"
    n = len(out)
    out = out[out["label"].notna()]
    log["dropped_unrecognised_label"] = int(n - len(out))

    # c) normalise the story text
    out["text_clean"] = out["text_raw"].map(clean_text)
    n = len(out)
    out = out[out["text_clean"].str.len() > 0]
    log["dropped_empty_after_cleaning"] = int(n - len(out))

    # d) the SAME story with DIFFERENT labels is ambiguous -> remove every copy
    labels_per_text = out.groupby("text_clean")["label"].nunique()
    conflicting = labels_per_text[labels_per_text > 1].index
    n = len(out)
    out = out[~out["text_clean"].isin(conflicting)]
    log["dropped_conflicting_duplicates"] = int(n - len(out))

    # e) identical stories with the same label -> keep one (duplicates would leak between train and test)
    n = len(out)
    out = out.drop_duplicates(subset="text_clean", keep="first")
    log["dropped_duplicates"] = int(n - len(out))

    # f) a class with only a handful of rows cannot be split or cross-validated
    counts = out["label"].value_counts()
    tiny = counts[counts < min_class_size].index.tolist()
    if tiny:
        print(f"   WARNING: classes with < {min_class_size} stories removed: {tiny}")
    out = out[~out["label"].isin(tiny)]
    log["dropped_tiny_classes"] = {str(k): int(counts[k]) for k in tiny}

    out = out.reset_index(drop=True)
    log["rows_out"] = int(len(out))
    log["class_counts"] = {str(k): int(v) for k, v in out["label"].value_counts().items()}
    for key, value in log.items():
        print(f"   {key}: {value}")
    return out, log


def audit_near_duplicates(df: pd.DataFrame, threshold: float = 0.90, max_rows: int = 4000) -> pd.DataFrame:
    """
    Data-quality audit: find pairs of stories that are almost identical (cosine similarity >= threshold).
    Near-duplicates inflate accuracy when one copy lands in train and the other in test, and
    pairs with DIFFERENT labels point to possible annotation errors.
    """
    if len(df) > max_rows or len(df) < 2:
        return pd.DataFrame()
    matrix = TfidfVectorizer(stop_words=STOP_WORDS, sublinear_tf=True).fit_transform(df["text_clean"])
    sim = cosine_similarity(matrix)
    np.fill_diagonal(sim, 0)
    i, j = np.where(np.triu(sim >= threshold))
    pairs = pd.DataFrame({
        "similarity": sim[i, j].round(3),
        "label_a": df["label"].values[i], "label_b": df["label"].values[j],
        "story_a": df["text_raw"].values[i], "story_b": df["text_raw"].values[j],
    })
    pairs["labels_differ"] = pairs["label_a"] != pairs["label_b"]
    return pairs.sort_values(["labels_differ", "similarity"], ascending=False).reset_index(drop=True)


# =========================================================================== #
# PART 4 - ENCODE CATEGORIES AND SPLIT
# =========================================================================== #
def encode_labels(df: pd.DataFrame):
    """Turn tier names into numbers 0..k-1 in a meaningful order (least -> most serious)."""
    present = list(df["label"].unique())
    classes = [t for t in TIERS if t in present] + sorted(t for t in present if t not in TIERS)
    encoder = LabelEncoder()
    encoder.classes_ = np.array(classes)
    return encoder, classes, encoder.transform(df["label"])


# =========================================================================== #
# PART 5 - FEATURE ENGINEERING EXPERIMENT
# =========================================================================== #
def make_word_tfidf(ngram=(1, 2)):
    """Words (and word pairs) weighted by TF-IDF: rare, informative words count more."""
    return TfidfVectorizer(stop_words=STOP_WORDS, ngram_range=ngram, sublinear_tf=True, strip_accents="unicode")


def make_char_tfidf():
    """Character 3-5-grams: robust to typos, plurals and word-forms (e.g. score/scoring/scored)."""
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, strip_accents="unicode")


def compare_feature_sets(X, y, cv) -> pd.DataFrame:
    """Same classifier, three different text representations -> which one separates the tiers best?"""
    banner("PART 5 - FEATURE ENGINEERING: WHICH TEXT REPRESENTATION WORKS BEST?")
    clf = LogisticRegression(max_iter=3000, C=10, class_weight="balanced", random_state=SEED)
    variants = {
        "Words (1-gram)": make_word_tfidf((1, 1)),
        "Words (1-2 grams)": make_word_tfidf((1, 2)),
        "Words + characters": FeatureUnion([("word", make_word_tfidf((1, 2))), ("char", make_char_tfidf())]),
    }
    rows = []
    for name, vectorizer in variants.items():
        scores = cross_val_score(Pipeline([("tfidf", vectorizer), ("clf", clf)]), X, y, cv=cv, scoring="f1_macro", n_jobs=-1)
        rows.append({"features": name, "macro_f1_mean": scores.mean(), "macro_f1_std": scores.std()})
        print(f"   {name:<20} macro-F1 = {scores.mean():.3f} +/- {scores.std():.3f}")
    return pd.DataFrame(rows)


# =========================================================================== #
# PART 6 - TRAIN, TUNE AND PREDICT
# =========================================================================== #
def candidate_models(seed: int) -> dict:
    """
    name -> (pipeline, grid of settings to try).
    The TF-IDF step is INSIDE the pipeline, so for every cross-validation fold it is learned on the
    training part only - the test part never influences the features (no data leakage).
    """
    def make(clf):
        return Pipeline([("tfidf", make_word_tfidf()), ("clf", clf)])

    return {
        "Logistic Regression": (make(LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed)),
                                {"clf__C": [1, 10, 100]}),
        "Linear SVM": (make(LinearSVC(class_weight="balanced", max_iter=10000, random_state=seed)),
                       {"clf__C": [0.1, 1, 10]}),
        "Multinomial NB": (make(MultinomialNB()), {"clf__alpha": [0.05, 0.3, 1.0]}),
        "Complement NB": (make(ComplementNB()), {"clf__alpha": [0.05, 0.3, 1.0]}),
        "Random Forest": (make(RandomForestClassifier(class_weight="balanced_subsample", random_state=seed, n_jobs=-1)),
                          {"clf__n_estimators": [200, 400], "clf__max_depth": [None, 30]}),
    }


class KeywordRuleBaseline:
    """
    The original v1.0 approach: a hand-written list of keywords. Kept ONLY as a baseline so we can
    measure how much the trained models actually add. It is not a trained model.
    """

    HIGH = ("creditworthiness", "credit score", "loan approval", "lending", "insurance underwriting", "risk pricing", "biometric", "fraud freeze")
    LIMITED = ("chatbot", "virtual assistant", "conversational", "generated")
    UNACCEPTABLE = ("social scoring", "subliminal manipulation")

    def predict_tier(self, story: str) -> str:
        s = story.lower()
        if any(k in s for k in self.UNACCEPTABLE):
            return "Unacceptable"
        if any(k in s for k in self.HIGH):
            return "High-Risk"
        if any(k in s for k in self.LIMITED):
            return "Limited-Risk"
        return "Minimal-Risk"


def tune_and_predict(X, y, train_idx, test_idx, classes, inner_cv, outer_cv, models) -> dict:
    """
    For every model:
      1. tune its settings with cross-validation on the TRAINING rows only,
      2. predict the held-out TEST rows (one honest estimate),
      3. run NESTED cross-validation over ALL rows (a second, more stable estimate): inside each outer
         fold the settings are tuned again, and the outer fold's rows are predicted. Collecting those
         predictions gives one "out-of-fold" (OOF) prediction for every story.
    """
    banner("PART 6 - TRAINING, TUNING AND PREDICTION")
    print(f"Tuning: {inner_cv.get_n_splits()}-fold CV on train set | stability check: nested {outer_cv.get_n_splits()}-fold CV on all rows")
    results = {}
    for name, (pipeline, grid) in models.items():
        print(f" - {name} ...", flush=True)
        search = GridSearchCV(pipeline, grid, cv=inner_cv, scoring="f1_macro", n_jobs=-1)
        search.fit(X.iloc[train_idx], y[train_idx])

        oof = np.full(len(y), -1)
        fold_acc, fold_f1 = [], []
        for tr, va in outer_cv.split(X, y):
            k = max(2, min(3, int(np.bincount(y[tr]).min())))
            inner = GridSearchCV(pipeline, grid, cv=StratifiedKFold(k, shuffle=True, random_state=SEED), scoring="f1_macro", n_jobs=-1)
            inner.fit(X.iloc[tr], y[tr])
            oof[va] = inner.predict(X.iloc[va])
            fold_acc.append(accuracy_score(y[va], oof[va]))
            fold_f1.append(f1_score(y[va], oof[va], average="macro", zero_division=0))
        results[name] = {
            "kind": "ML (TF-IDF)", "search": search, "test_pred": search.predict(X.iloc[test_idx]), "oof": oof,
            "fold_acc": fold_acc, "fold_f1": fold_f1,
            "best_params": {k.split("__")[1]: v for k, v in search.best_params_.items()},
        }

    # --- baseline 1: always predict the most common tier ---
    oof = np.full(len(y), -1)
    for tr, va in outer_cv.split(X, y):
        oof[va] = DummyClassifier(strategy="most_frequent").fit(X.iloc[tr], y[tr]).predict(X.iloc[va])
    dummy = DummyClassifier(strategy="most_frequent").fit(X.iloc[train_idx], y[train_idx])
    results["Majority class"] = {"kind": "Baseline", "test_pred": dummy.predict(X.iloc[test_idx]), "oof": oof, "best_params": {}}

    return results


def add_rule_baseline(results, df, y, test_idx, classes, outer_cv):
    """Score the keyword rules on the same rows/folds (they need no training)."""
    rules = KeywordRuleBaseline()
    guesses = np.array([classes.index(p) if p in classes else -1 for p in (rules.predict_tier(s) for s in df["text_raw"])])
    results["Keyword rules (v1.0)"] = {"kind": "Baseline", "test_pred": guesses[test_idx], "oof": guesses, "best_params": {}}


# =========================================================================== #
# PART 7 - EVALUATION: ACCURACY, F1, CONFIDENCE INTERVALS, SIGNIFICANCE
# =========================================================================== #
def metric_set(y_true, y_pred, n_classes: int, high_idx: int | None) -> dict:
    """The accuracy-type numbers reported for every model."""
    result = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }
    # recall on High-Risk = "of all truly high-risk stories, how many did we catch?" (missing one is the costly error)
    result["high_risk_recall"] = (recall_score(y_true, y_pred, labels=[high_idx], average=None, zero_division=0)[0]
                                  if high_idx is not None else np.nan)
    return result


def bootstrap_ci(y_true, y_pred, n_boot: int, seed: int = SEED) -> dict:
    """95% confidence interval: resample the stories with replacement and recompute the score many times."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    acc, f1 = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        acc.append(accuracy_score(y_true[idx], y_pred[idx]))
        f1.append(f1_score(y_true[idx], y_pred[idx], average="macro", zero_division=0))
    return {"accuracy": (float(np.percentile(acc, 2.5)), float(np.percentile(acc, 97.5))),
            "macro_f1": (float(np.percentile(f1, 2.5)), float(np.percentile(f1, 97.5)))}


def mcnemar_test(y_true, pred_a, pred_b) -> dict:
    """
    Is model A really better than model B, or is the gap just luck?
    McNemar's exact test looks only at stories where exactly one model is right.
    p < 0.05  => the difference is unlikely to be chance.
    """
    a_right, b_right = pred_a == y_true, pred_b == y_true
    only_a, only_b = int((a_right & ~b_right).sum()), int((~a_right & b_right).sum())
    n = only_a + only_b
    p = 1.0 if n == 0 else float(binomtest(min(only_a, only_b), n, 0.5).pvalue)
    return {"only_first_correct": only_a, "only_second_correct": only_b, "p_value": p}


def summarise_models(results, y, y_test, classes) -> pd.DataFrame:
    """One row per model: pooled out-of-fold metrics, fold spread and held-out test metrics."""
    high_idx = classes.index(HIGH_RISK) if HIGH_RISK in classes else None
    rows = []
    for name, r in results.items():
        cv = metric_set(y, r["oof"], len(classes), high_idx)
        te = metric_set(y_test, r["test_pred"], len(classes), high_idx)
        row = {"model": name, "type": r["kind"], "best_params": json.dumps(r["best_params"], default=str)}
        row.update({f"cv_{k}": v for k, v in cv.items()})
        row["cv_accuracy_fold_std"] = float(np.std(r["fold_acc"])) if "fold_acc" in r else np.nan
        row["cv_macro_f1_fold_std"] = float(np.std(r["fold_f1"])) if "fold_f1" in r else np.nan
        row.update({f"test_{k}": v for k, v in te.items()})
        rows.append(row)
    table = pd.DataFrame(rows)
    ml = table[table["type"] == "ML (TF-IDF)"].sort_values(["cv_macro_f1", "test_macro_f1"], ascending=False)
    table["selected"] = table["model"].eq(ml.iloc[0]["model"])
    return table.sort_values(["cv_macro_f1", "test_macro_f1"], ascending=False).reset_index(drop=True)


# =========================================================================== #
# PART 8 - EXTRA ANALYSIS: CLUSTERING, PER-DOMAIN RESULTS, ERRORS
# =========================================================================== #
def unsupervised_check(X, y, n_classes):
    """
    Do the stories fall into natural groups WITHOUT using the labels?
    KMeans clusters are compared with the true tiers: ARI/NMI close to 1 means the text alone
    already separates the tiers; close to 0 means the tiers are not obvious from vocabulary.
    """
    matrix = make_word_tfidf().fit_transform(X)
    k_dims = int(min(50, matrix.shape[0] - 1, matrix.shape[1] - 1))
    svd = TruncatedSVD(n_components=max(2, k_dims), random_state=SEED)
    reduced = Normalizer().fit_transform(svd.fit_transform(matrix))
    clusters = KMeans(n_clusters=n_classes, n_init=10, random_state=SEED).fit_predict(reduced)
    plane = TruncatedSVD(n_components=2, random_state=SEED).fit_transform(matrix)  # 2-D map for plotting
    stats = {
        "adjusted_rand_index": float(adjusted_rand_score(y, clusters)),
        "normalized_mutual_info": float(normalized_mutual_info_score(y, clusters)),
        "silhouette": float(silhouette_score(reduced, clusters)) if len(set(clusters)) > 1 else float("nan"),
        "explained_variance_top_components": float(svd.explained_variance_ratio_.sum()),
    }
    print(f"   KMeans (k={n_classes}) vs true tiers: ARI={stats['adjusted_rand_index']:.3f}  NMI={stats['normalized_mutual_info']:.3f}  silhouette={stats['silhouette']:.3f}")
    return stats, plane, clusters


def per_domain_accuracy(df, domain_col, y, oof) -> pd.DataFrame:
    """Does the best model work equally well in every FinTech domain?"""
    if domain_col is None or domain_col not in df.columns:
        return pd.DataFrame()
    frame = pd.DataFrame({"domain": df[domain_col].astype(str).values, "correct": (oof == y)})
    out = frame.groupby("domain")["correct"].agg(accuracy="mean", stories="size").reset_index()
    return out.sort_values("accuracy").reset_index(drop=True)


def error_analysis(df, y, oof, classes) -> tuple[pd.DataFrame, pd.DataFrame]:
    """List every misclassified story and count which tier pairs are confused most often."""
    wrong = oof != y
    errors = pd.DataFrame({
        "story": df["text_raw"].values[wrong],
        "true_label": [classes[i] for i in y[wrong]],
        "predicted_label": [classes[i] if i >= 0 else "?" for i in oof[wrong]],
    })
    pairs = (errors.groupby(["true_label", "predicted_label"]).size().reset_index(name="count")
             .sort_values("count", ascending=False).reset_index(drop=True))
    return errors, pairs


# =========================================================================== #
# PART 9 - VISUALISATIONS
# =========================================================================== #
def plot_clean_distribution(log, fig_dir):
    order = [c for c in TIERS if c in log["class_counts"]] or list(log["class_counts"])
    values = [log["class_counts"][c] for c in order]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(order, values, color=[tier_color(c) for c in order], width=0.6)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v} ({v / sum(values):.0%})", ha="center", va="bottom", fontsize=9, color=INK)
    style_axes(ax, f"Class distribution after cleaning (n={sum(values)})", ylabel="User stories")
    ax.set_ylim(0, max(values) * 1.15)
    save_fig(fig, fig_dir / "05_class_distribution_clean.png")


def plot_embedding(plane, y, clusters, classes, fig_dir):
    """2-D map of the stories (TF-IDF -> SVD). Left: true tiers. Right: unsupervised clusters."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for i, name in enumerate(classes):
        m = y == i
        axes[0].scatter(plane[m, 0], plane[m, 1], s=22, color=tier_color(name), label=name, edgecolor="white", linewidth=0.4, alpha=0.9)
    for c in sorted(set(clusters)):
        m = clusters == c
        axes[1].scatter(plane[m, 0], plane[m, 1], s=22, color=CLUSTER_COLORS[c % len(CLUSTER_COLORS)], label=f"Cluster {c}", edgecolor="white", linewidth=0.4, alpha=0.9)
    for ax, title in zip(axes, ["True risk tier", "KMeans clusters (labels not used)"]):
        style_axes(ax, title, xlabel="SVD component 1", ylabel="SVD component 2")
        ax.grid(axis="x", color=GRID)
        ax.legend(frameon=False, fontsize=8, loc="best")
    save_fig(fig, fig_dir / "06_story_map_svd_clusters.png")


def plot_feature_sets(table, fig_dir):
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.barh(table["features"], table["macro_f1_mean"], xerr=table["macro_f1_std"], color=BLUE, height=0.5, ecolor=MUTED, capsize=3)
    for i, v in enumerate(table["macro_f1_mean"]):
        ax.text(min(v + 0.02, 1.02), i, f"{v:.3f}", va="center", fontsize=9, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.12)
    style_axes(ax, "Text representation comparison (5-fold macro-F1)", xlabel="Macro-F1")
    horizontal_grid(ax)
    save_fig(fig, fig_dir / "07_feature_set_comparison.png")


def plot_model_comparison(table, fig_dir):
    order = table["model"].tolist()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharey=True)
    for ax, metric, title in zip(axes, ["accuracy", "macro_f1"], ["Accuracy", "Macro-F1"]):
        pos = np.arange(len(order))
        cv_vals, test_vals = table[f"cv_{metric}"].values, table[f"test_{metric}"].values
        b1 = ax.barh(pos - 0.19, cv_vals, height=0.36, color=BLUE, label="Nested CV (all stories)")
        b2 = ax.barh(pos + 0.19, test_vals, height=0.36, color=ORANGE, label="Held-out test set")
        for bars in (b1, b2):
            for b in bars:
                ax.text(b.get_width() + 0.01, b.get_y() + b.get_height() / 2, f"{b.get_width():.2f}", va="center", fontsize=8, color=INK)
        ax.set_yticks(pos)
        ax.set_yticklabels(order)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.12)
        style_axes(ax, title)
        horizontal_grid(ax)
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, fontsize=9, loc="upper right", ncol=2)
    fig.suptitle("Model comparison", x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    save_fig(fig, fig_dir / "08_model_comparison.png")


def plot_fold_spread(results, fig_dir):
    names = [n for n, r in results.items() if "fold_f1" in r]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    box = ax.boxplot([results[n]["fold_f1"] for n in names], vert=False, patch_artist=True, widths=0.5, tick_labels=names)
    for patch in box["boxes"]:
        patch.set(facecolor=BLUE, alpha=0.35, edgecolor=BLUE)
    for median in box["medians"]:
        median.set(color=INK, linewidth=1.5)
    ax.invert_yaxis()
    style_axes(ax, "Macro-F1 across nested-CV folds (stability)", xlabel="Macro-F1")
    horizontal_grid(ax)
    save_fig(fig, fig_dir / "09_cv_fold_spread.png")


def plot_confusion_best(y, oof, classes, best, fig_dir):
    labels_idx = list(range(len(classes)))
    counts = confusion_matrix(y, oof, labels=labels_idx)
    share = counts / counts.sum(axis=1, keepdims=True).clip(min=1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, data, fmt, title in [(axes[0], counts, "d", "Counts"), (axes[1], share, ".0%", "Row-normalised (recall per tier)")]:
        sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues", cbar=False, xticklabels=classes, yticklabels=classes, linewidths=2, linecolor="white", ax=ax)
        ax.set_title(title, loc="left", fontsize=10, color=INK, fontweight="bold")
        ax.set_xlabel("Predicted", color=MUTED)
        ax.set_ylabel("Actual", color=MUTED)
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.suptitle(f"Confusion matrix - {best} (nested-CV predictions for every story)", x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    save_fig(fig, fig_dir / "10_confusion_matrix_best.png")


def plot_confusion_all(results, y_test, classes, fig_dir):
    names = list(results)
    ncol, nrow = 3, int(np.ceil(len(names) / 3))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.9 * nrow), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    short = [c[:4] for c in classes]
    for ax, name in zip(axes.ravel(), names):
        ax.axis("on")
        cm = confusion_matrix(y_test, results[name]["test_pred"], labels=list(range(len(classes))))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=short, yticklabels=short, linewidths=1.5, linecolor="white", ax=ax)
        ax.set_title(f"{name} ({accuracy_score(y_test, results[name]['test_pred']):.0%})", loc="left", fontsize=9, color=INK, fontweight="bold")
    save_fig(fig, fig_dir / "11_confusion_matrices_all_models_test.png")


def plot_per_class_f1(results, y, classes, fig_dir):
    labels_idx = list(range(len(classes)))
    table = pd.DataFrame({n: f1_score(y, r["oof"], labels=labels_idx, average=None, zero_division=0) for n, r in results.items()}, index=classes).T
    fig, ax = plt.subplots(figsize=(7.5, 0.55 * len(table) + 1.8))
    sns.heatmap(table, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1, cbar=False, linewidths=2, linecolor="white", ax=ax)
    ax.set_title("Per-tier F1 (nested-CV predictions)", loc="left", fontsize=11, fontweight="bold", color=INK)
    ax.set_xlabel("")
    ax.set_ylabel("")
    save_fig(fig, fig_dir / "12_per_class_f1_heatmap.png")


def plot_learned_terms(best_search, best_name, classes, fig_dir):
    """Which words does the best model use to decide each tier?"""
    pipeline = best_search.best_estimator_
    terms = np.array(pipeline.named_steps["tfidf"].get_feature_names_out())
    clf = pipeline.named_steps["clf"]
    weights = None
    if hasattr(clf, "coef_") and clf.coef_.shape[0] == len(classes):
        weights = clf.coef_
    elif hasattr(clf, "feature_log_prob_"):
        weights = clf.feature_log_prob_ - clf.feature_log_prob_.mean(axis=0)
    if weights is not None:
        ncol, nrow = 2, int(np.ceil(len(classes) / 2))
        fig, axes = plt.subplots(nrow, ncol, figsize=(10, 3.3 * nrow), squeeze=False)
        for ax in axes.ravel():
            ax.axis("off")
        for ax, (i, cls) in zip(axes.ravel(), enumerate(classes)):
            ax.axis("on")
            top = weights[i].argsort()[-10:]
            ax.barh(terms[top], weights[i][top], color=tier_color(cls), height=0.65)
            style_axes(ax, cls, xlabel="weight")
            horizontal_grid(ax)
        fig.suptitle(f"Terms that push a story toward each tier - {best_name}", x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
        save_fig(fig, fig_dir / "13_top_features_best_model.png")
    elif hasattr(clf, "feature_importances_"):
        top = clf.feature_importances_.argsort()[-15:]
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.barh(terms[top], clf.feature_importances_[top], color=BLUE, height=0.65)
        style_axes(ax, f"Top 15 terms - {best_name}", xlabel="Importance")
        horizontal_grid(ax)
        save_fig(fig, fig_dir / "13_top_features_best_model.png")


def plot_domain_accuracy(table, fig_dir):
    fig, ax = plt.subplots(figsize=(7, 0.6 * len(table) + 1.6))
    ax.barh(table["domain"], table["accuracy"], color=BLUE, height=0.55)
    for i, (acc, n) in enumerate(zip(table["accuracy"], table["stories"])):
        ax.text(min(acc + 0.01, 1.0), i, f"{acc:.0%} (n={n})", va="center", fontsize=9, color=INK)
    ax.set_xlim(0, 1.18)
    style_axes(ax, "Best model: accuracy per FinTech domain", xlabel="Accuracy (nested-CV predictions)")
    horizontal_grid(ax)
    save_fig(fig, fig_dir / "14_accuracy_per_domain.png")


def plot_learning_curve(estimator, X, y, cv, best, fig_dir):
    """If validation F1 is still rising at the right edge, collecting more labelled stories would help."""
    try:
        sizes, train_scores, val_scores = learning_curve(estimator, X, y, cv=cv, scoring="f1_macro", train_sizes=np.linspace(0.2, 1.0, 5),
                                                         n_jobs=-1, shuffle=True, random_state=SEED)
    except Exception as err:  # noqa: BLE001
        print(f"   (learning curve skipped: {err})")
        return
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for values, color, label in [(train_scores, BLUE, "Training"), (val_scores, ORANGE, "Validation")]:
        mean, std = values.mean(axis=1), values.std(axis=1)
        ax.plot(sizes, mean, color=color, linewidth=2, marker="o", markersize=6, markeredgecolor="white", label=label)
        ax.fill_between(sizes, mean - std, mean + std, color=color, alpha=0.12)
    style_axes(ax, f"Learning curve - {best}", xlabel="Training stories", ylabel="Macro-F1")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=9)
    save_fig(fig, fig_dir / "15_learning_curve.png")


# =========================================================================== #
# PART 10 - AUTO-GENERATED RESULTS REPORT
# =========================================================================== #
def write_report(path: Path, ctx: dict):
    """Write outputs/RESULTS.md: every number below comes from this run."""
    t, best = ctx["table"], ctx["best"]
    best_row = t[t["model"] == best].iloc[0]
    L = []
    add = L.append
    add("# Results report\n")
    if ctx["demo"]:
        add("> **DEMO RUN on synthetic data - these numbers do not describe the real dataset.**\n")
    add(f"_Generated by `fintech_risk_classification.py` (random seed {SEED}). Dataset: DOI 10.17632/ww3wmkw466.1._\n")

    add("## 1. Data and cleaning")
    add(f"- Story column(s): `{ctx['cols']['text']}` | label column: `{ctx['cols']['label']}` | domain column: `{ctx['cols']['domain']}`")
    c = ctx["clean_log"]
    add(f"- Rows in: **{c['rows_in']}** -> rows used: **{c['rows_out']}**")
    add(f"- Removed: missing text/label {c['dropped_missing_text_or_label']}, unrecognised label {c['dropped_unrecognised_label']}, "
        f"conflicting duplicates {c['dropped_conflicting_duplicates']}, exact duplicates {c['dropped_duplicates']}")
    add(f"- Class counts after cleaning: {c['class_counts']}")
    nd = ctx["near_dups"]
    if len(nd):
        add(f"- Near-duplicate audit (cosine >= 0.90): **{len(nd)}** pairs, of which **{int(nd['labels_differ'].sum())}** have different labels "
            "(see `near_duplicate_pairs.csv` - possible annotation inconsistencies and train/test leakage risk).")
    else:
        add("- Near-duplicate audit (cosine >= 0.90): no pairs found.")
    add("\n![](figures/01_class_distribution_raw.png)\n![](figures/05_class_distribution_clean.png)\n")

    add("## 2. Exploratory analysis")
    add(f"- Words per story: {ctx['eda']['words_per_story']}")
    u = ctx["unsup"]
    add(f"- Unsupervised check (KMeans on TF-IDF/SVD vs true tiers): ARI **{u['adjusted_rand_index']:.3f}**, NMI **{u['normalized_mutual_info']:.3f}**, "
        f"silhouette **{u['silhouette']:.3f}**. Values near 1 mean tiers are separable from vocabulary alone; near 0 means they are not.")
    add("\n![](figures/02_text_length_by_class.png)\n![](figures/04_top_terms_per_class.png)\n![](figures/06_story_map_svd_clusters.png)\n")
    if ctx["eda"].get("domain_vs_label"):
        add("![](figures/03_domain_vs_risk.png)\n")

    add("## 3. Feature representation experiment")
    add(df_to_markdown(ctx["feature_table"]))
    add("\n![](figures/07_feature_set_comparison.png)\n")

    add("## 4. Model comparison")
    add(f"Split: {ctx['split']['train']} train / {ctx['split']['test']} test (stratified); nested CV: {ctx['split']['folds']} folds over all {c['rows_out']} stories.\n")
    view = t[["model", "cv_accuracy", "cv_accuracy_fold_std", "cv_macro_f1", "cv_high_risk_recall", "test_accuracy", "test_macro_f1", "test_high_risk_recall", "test_mcc"]]
    add(df_to_markdown(view))
    add(f"\n**Selected model (highest nested-CV macro-F1 among trained models): {best}** with settings `{best_row['best_params']}`.\n")
    add("![](figures/08_model_comparison.png)\n![](figures/09_cv_fold_spread.png)\n")

    add("## 5. Accuracy of the selected model")
    ci = ctx["ci"]
    add(f"- Nested-CV accuracy **{best_row['cv_accuracy']:.3f}** (95% bootstrap CI {ci['accuracy'][0]:.3f} - {ci['accuracy'][1]:.3f})")
    add(f"- Nested-CV macro-F1 **{best_row['cv_macro_f1']:.3f}** (95% bootstrap CI {ci['macro_f1'][0]:.3f} - {ci['macro_f1'][1]:.3f})")
    add(f"- Held-out test accuracy **{best_row['test_accuracy']:.3f}**, macro-F1 **{best_row['test_macro_f1']:.3f}**, High-Risk recall **{best_row['test_high_risk_recall']:.3f}**\n")
    add("Per-tier report (held-out test set):\n")
    add("```\n" + ctx["class_report"] + "\n```\n")
    add("![](figures/10_confusion_matrix_best.png)\n![](figures/11_confusion_matrices_all_models_test.png)\n![](figures/12_per_class_f1_heatmap.png)\n")

    add("## 6. Is the winner really better? (McNemar exact test on nested-CV predictions)")
    add(df_to_markdown(ctx["mcnemar"], float_fmt="{:.4f}"))
    add("\n`p_value < 0.05` means the accuracy gap is unlikely to be chance.\n")

    add("## 7. Error analysis")
    pairs = ctx["error_pairs"]
    add(f"- Misclassified stories (nested-CV): **{ctx['n_errors']}** of {c['rows_out']} - full list in `misclassified_stories.csv`.")
    if len(pairs):
        add("\nMost frequent confusions:\n")
        add(df_to_markdown(pairs.head(8)))
    add("\n![](figures/13_top_features_best_model.png)\n")
    if len(ctx["domain_acc"]):
        add("\n## 8. Accuracy per FinTech domain\n")
        add(df_to_markdown(ctx["domain_acc"]))
        add("\n![](figures/14_accuracy_per_domain.png)\n")
    add("## 9. Learning curve\n\n![](figures/15_learning_curve.png)\n")

    add("## 10. Files produced")
    add("`model_comparison.csv`, `classification_report_best_model.csv`, `test_predictions.csv`, `misclassified_stories.csv`, "
        "`near_duplicate_pairs.csv`, `summary.json`, `run_log.txt`, `figures/*.png`, `best_model.joblib` (not committed).")
    path.write_text("\n".join(L), encoding="utf-8")
    print(f"   report saved: {path}")


# =========================================================================== #
# PART 11 - PREDICT A NEW STORY WITH THE SAVED MODEL
# =========================================================================== #
def predict_story(model_path: Path, story: str):
    if not model_path.exists():
        sys.exit(f"No saved model at {model_path}. Run the pipeline on the dataset first.")
    bundle = joblib.load(model_path)
    model, classes = bundle["model"], bundle["classes"]
    text = clean_text(story)
    guess = classes[int(model.predict([text])[0])]
    print(f"\nStory     : {story}\nPredicted : {guess}")
    clf = model.named_steps["clf"]
    if hasattr(clf, "predict_proba"):
        for name, p in sorted(zip(classes, model.predict_proba([text])[0]), key=lambda x: -x[1]):
            print(f"   {name:<14} {p:6.1%}")


# =========================================================================== #
# SYNTHETIC DATA (smoke test only - never quote results from this)
# =========================================================================== #
def make_demo_data(n: int = 350, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    personas = {"Lending": ["borrower", "credit officer"], "Fraud & Payments": ["payment analyst", "merchant"],
                "Wealth Management": ["investor", "portfolio manager"], "Insurance": ["policyholder", "claims handler"],
                "Customer Support": ["customer", "support agent"]}
    actions = {
        "High-Risk": ["automatically score applicant creditworthiness and decide approval", "price premiums with a model trained on personal health data",
                      "verify identity through facial biometric matching before onboarding", "rank applicants for eligibility using an automated scoring model",
                      "freeze accounts automatically when the model flags suspicious behaviour"],
        "Limited-Risk": ["talk to a chatbot that answers questions about my account", "receive replies from a virtual assistant that discloses it is an AI",
                         "get generated summaries of my statements labelled as machine produced", "use a conversational agent to explain product options"],
        "Minimal-Risk": ["sort my monthly expenses by date and category", "export my transaction history to a spreadsheet", "set a reminder before a bill is due",
                         "change the colour theme of the dashboard", "search past invoices by keyword"],
        "Unacceptable": ["assign citizens a social scoring rank based on behaviour", "nudge users with subliminal manipulation to take on more debt",
                         "exploit vulnerabilities of elderly users to sell products"],
    }
    weights = {"High-Risk": 0.38, "Limited-Risk": 0.22, "Minimal-Risk": 0.34, "Unacceptable": 0.06}
    rows = []
    for _ in range(n):
        domain = rng.choice(list(personas))
        tier = rng.choice(list(weights), p=list(weights.values()))
        rows.append({"Domain": domain,
                     "User_Story": f"As a {rng.choice(personas[domain])}, I want to {rng.choice(actions[tier])} so that I can {rng.choice(['save time', 'reduce errors', 'stay compliant', 'improve trust'])}.",
                     "Risk_Level": tier})
    df = pd.DataFrame(rows)
    df.loc[rng.choice(n, 4, replace=False), "Risk_Level"] = ["high risk", "HIGH-RISK", "Unacceptable/Blocked", " minimal risk "]  # messy labels
    df.loc[rng.choice(n, 3, replace=False), "User_Story"] = None  # missing text
    return pd.concat([df, df.sample(5, random_state=seed)], ignore_index=True)  # exact duplicates


# =========================================================================== #
# MAIN - THE WHOLE WORKFLOW IN ORDER
# =========================================================================== #
def run(cfg: Config) -> dict:
    fig_dir = cfg.out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # PART 1 ---------------------------------------------------------------
    banner("PART 1 - LOADING DATA")
    if cfg.demo:
        print("!! DEMO MODE: synthetic data. These numbers say nothing about the real dataset. !!")
        raw = make_demo_data()
    else:
        path = find_data_file(cfg.data)
        raw = load_data(path)
        print(f"Loaded {path} -> {raw.shape[0]} rows x {raw.shape[1]} columns")
    text_cols, label_col, domain_col = detect_columns(raw, cfg.text_cols, cfg.label_col, cfg.domain_col)

    # PART 2 ---------------------------------------------------------------
    eda = run_eda(raw, text_cols, label_col, domain_col, fig_dir)

    # PART 3 ---------------------------------------------------------------
    df, clean_log = clean_data(raw, text_cols, label_col, min_class_size=max(cfg.cv_folds, 4))
    if df["label"].nunique() < 2:
        sys.exit("Fewer than two classes remain after cleaning; cannot train a classifier.")
    plot_clean_distribution(clean_log, fig_dir)
    near_dups = audit_near_duplicates(df)
    near_dups.to_csv(cfg.out / "near_duplicate_pairs.csv", index=False)
    print(f"   near-duplicate pairs (cosine >= 0.90): {len(near_dups)}"
          + (f" | with different labels: {int(near_dups['labels_differ'].sum())}" if len(near_dups) else ""))

    # PART 4 ---------------------------------------------------------------
    banner("PART 4 - CATEGORIES AND TRAIN/TEST SPLIT")
    encoder, classes, y = encode_labels(df)
    X = df["text_clean"]
    print("Encoded classes:", {c: i for i, c in enumerate(classes)})
    idx_all = np.arange(len(df))
    train_idx, test_idx = train_test_split(idx_all, test_size=cfg.test_size, stratify=y, random_state=SEED)
    y_test = y[test_idx]
    print(f"Train: {len(train_idx)} stories | Test: {len(test_idx)} stories (stratified, test_size={cfg.test_size})")
    smallest = int(np.bincount(y).min())
    outer_cv = StratifiedKFold(n_splits=max(2, min(cfg.cv_folds, smallest)), shuffle=True, random_state=SEED)
    inner_cv = StratifiedKFold(n_splits=max(2, min(3, int(np.bincount(y[train_idx]).min()))), shuffle=True, random_state=SEED)

    # PART 5 ---------------------------------------------------------------
    feature_table = compare_feature_sets(X, y, outer_cv)
    plot_feature_sets(feature_table, fig_dir)

    # PART 6 ---------------------------------------------------------------
    results = tune_and_predict(X, y, train_idx, test_idx, classes, inner_cv, outer_cv, candidate_models(SEED))
    add_rule_baseline(results, df, y, test_idx, classes, outer_cv)

    # PART 7 ---------------------------------------------------------------
    banner("PART 7 - EVALUATION")
    table = summarise_models(results, y, y_test, classes)
    best = table[table["selected"]].iloc[0]["model"]
    show = table[["model", "cv_accuracy", "cv_macro_f1", "test_accuracy", "test_macro_f1", "test_high_risk_recall", "test_mcc"]]
    print(show.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nBest trained model (highest nested-CV macro-F1): {best}")
    table.to_csv(cfg.out / "model_comparison.csv", index=False)

    best_oof = results[best]["oof"]
    ci = bootstrap_ci(y, best_oof, cfg.bootstrap)
    print(f"95% bootstrap CI -> accuracy {ci['accuracy'][0]:.3f}-{ci['accuracy'][1]:.3f} | macro-F1 {ci['macro_f1'][0]:.3f}-{ci['macro_f1'][1]:.3f}")

    ml_sorted = [n for n in table["model"] if results[n]["kind"] == "ML (TF-IDF)"]
    rivals = [n for n in ["Keyword rules (v1.0)", "Majority class"] if n in results] + [n for n in ml_sorted if n != best][:1]
    mc_rows = []
    for rival in rivals:
        r = mcnemar_test(y, best_oof, results[rival]["oof"])
        mc_rows.append({"best_model": best, "compared_with": rival, **r})
        print(f"McNemar {best} vs {rival}: only-best-correct={r['only_first_correct']}, only-rival-correct={r['only_second_correct']}, p={r['p_value']:.4f}")
    mcnemar = pd.DataFrame(mc_rows)

    banner(f"DETAILED ACCURACY - {best} (held-out test set)")
    labels_idx = list(range(len(classes)))
    best_test = results[best]["test_pred"]
    report_text = classification_report(y_test, best_test, labels=labels_idx, target_names=classes, zero_division=0)
    print(report_text)
    pd.DataFrame(classification_report(y_test, best_test, labels=labels_idx, target_names=classes, zero_division=0, output_dict=True)).T.to_csv(
        cfg.out / "classification_report_best_model.csv")
    pd.DataFrame({"story": df["text_raw"].values[test_idx], "true_label": [classes[i] for i in y_test],
                  "predicted_label": [classes[i] for i in best_test]}).assign(
        correct=lambda d: d["true_label"] == d["predicted_label"]).to_csv(cfg.out / "test_predictions.csv", index=False)

    # PART 8 ---------------------------------------------------------------
    banner("PART 8 - EXTRA ANALYSIS")
    unsup, plane, clusters = unsupervised_check(X, y, len(classes))
    domain_acc = per_domain_accuracy(df, domain_col, y, best_oof)
    errors, error_pairs = error_analysis(df, y, best_oof, classes)
    errors.to_csv(cfg.out / "misclassified_stories.csv", index=False)
    print(f"   misclassified stories (nested-CV): {len(errors)} of {len(df)} -> misclassified_stories.csv")
    if len(domain_acc):
        print("   accuracy per domain:\n" + domain_acc.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # PART 9 ---------------------------------------------------------------
    banner("PART 9 - FIGURES")
    plot_embedding(plane, y, clusters, classes, fig_dir)
    plot_model_comparison(table, fig_dir)
    plot_fold_spread(results, fig_dir)
    plot_confusion_best(y, best_oof, classes, best, fig_dir)
    plot_confusion_all(results, y_test, classes, fig_dir)
    plot_per_class_f1(results, y, classes, fig_dir)
    plot_learned_terms(results[best]["search"], best, classes, fig_dir)
    if len(domain_acc):
        plot_domain_accuracy(domain_acc, fig_dir)
    plot_learning_curve(results[best]["search"].best_estimator_, X, y, outer_cv, best, fig_dir)

    # Save the final model: best settings, refit on ALL cleaned stories
    best_clf = results[best]["search"].best_estimator_.named_steps["clf"]
    final = Pipeline([("tfidf", make_word_tfidf()), ("clf", best_clf)]).fit(X, y)
    joblib.dump({"model": final, "classes": classes}, cfg.out / "best_model.joblib")

    # PART 10 --------------------------------------------------------------
    banner("PART 10 - REPORT")
    best_cv = table[table["model"] == best].iloc[0]
    summary = {
        "demo_data": cfg.demo, "columns": {"text": text_cols, "label": label_col, "domain": domain_col},
        "cleaning": clean_log, "classes": classes, "best_model": best,
        "best_model_nested_cv": {"accuracy": float(best_cv["cv_accuracy"]), "macro_f1": float(best_cv["cv_macro_f1"]),
                                 "accuracy_ci95": ci["accuracy"], "macro_f1_ci95": ci["macro_f1"]},
        "best_model_test": {k.replace("test_", ""): float(best_cv[k]) for k in best_cv.index if str(k).startswith("test_")},
        "clustering": unsup, "eda": eda,
    }
    (cfg.out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    write_report(cfg.out / "RESULTS.md", {
        "demo": cfg.demo, "cols": {"text": text_cols, "label": label_col, "domain": domain_col}, "clean_log": clean_log, "eda": eda,
        "near_dups": near_dups, "unsup": unsup, "feature_table": feature_table, "table": table, "best": best, "ci": ci,
        "split": {"train": len(train_idx), "test": len(test_idx), "folds": outer_cv.get_n_splits()},
        "class_report": report_text, "mcnemar": mcnemar, "error_pairs": error_pairs, "n_errors": len(errors), "domain_acc": domain_acc,
    })

    print(f"\nBest model: {best} | nested-CV accuracy {best_cv['cv_accuracy']:.3f}, macro-F1 {best_cv['cv_macro_f1']:.3f} | "
          f"test accuracy {best_cv['test_accuracy']:.3f}, High-Risk recall {best_cv['test_high_risk_recall']:.3f}")
    print(f"All outputs are in: {cfg.out.resolve()}")
    if cfg.demo:
        print("(synthetic demo data - do not cite these numbers)")
    return summary


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Classify FinTech user stories into EU AI Act risk tiers.")
    p.add_argument("--data", help="Dataset path (.csv/.xlsx/.json). Default: first file in ./data")
    p.add_argument("--text-cols", nargs="+", help="Column(s) with the user story (auto-detected)")
    p.add_argument("--label-col", help="Column with the risk label (auto-detected)")
    p.add_argument("--domain-col", help="Optional domain column (analysis only)")
    p.add_argument("--out", default="outputs", help="Output folder (default: outputs)")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--demo", action="store_true", help="Run on synthetic data to test the code (NOT real results)")
    p.add_argument("--predict", metavar="STORY", help="Predict the tier of one story using the saved model, then exit")
    return p.parse_args(argv)


def main(argv=None):
    warnings.filterwarnings("ignore")
    args = parse_args(argv)
    out = Path(args.out)
    if args.predict:
        predict_story(out / "best_model.joblib", args.predict)
        return
    sns.set_theme(style="white")
    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": "white"})
    out.mkdir(parents=True, exist_ok=True)
    cfg = Config(args.data, args.text_cols, args.label_col, args.domain_col, out, args.test_size, args.cv_folds, args.demo)

    original_stdout = sys.stdout
    with open(out / "run_log.txt", "w", encoding="utf-8") as log_file:
        sys.stdout = Tee(original_stdout, log_file)  # everything printed is also saved
        try:
            run(cfg)
        finally:
            sys.stdout = original_stdout


if __name__ == "__main__":
    main()
