#!/usr/bin/env python3
"""
FinTech Agile User Stories -> EU AI Act risk-tier classification pipeline.

End-to-end, reproducible workflow for the Mendeley dataset
"FinTech Software Product Requirements, Agile User Stories, and EU AI Act
Risk Classification Dataset" (DOI: 10.17632/ww3wmkw466.1):

    1. Load data            (CSV / XLSX / JSON, encoding-safe, columns auto-detected)
    2. Exploratory analysis (shape, dtypes, missing, duplicates, class balance,
                             text length, domain-vs-risk, discriminative terms)
    3. Cleaning             (labels canonicalised, empty/duplicate/conflicting
                             rows removed, text normalised) with a cleaning log
    4. Categories           (label encoding, stratified train/test split)
    5. Training             (TF-IDF + 5 ML models, hyper-parameters tuned with
                             inner cross-validation on the training set only)
    6. Prediction           (held-out test set + nested cross-validation)
    7. Comparison           (ML models vs. keyword rule baseline vs. majority class)
    8. Accuracy             (accuracy, balanced accuracy, macro-F1, weighted-F1,
                             MCC, High-Risk recall, per-class report)
    9. Visualisation        (PNG figures saved to <out>/figures)

Usage
-----
    python fintech_risk_pipeline.py --data data/fintech_stories.csv
    python fintech_risk_pipeline.py --data file.xlsx --text-cols story --label-col risk_level
    python fintech_risk_pipeline.py --demo        # synthetic smoke test ONLY (not real results)

Every number in the report is computed from the data; nothing is hard-coded.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    learning_curve,
    train_test_split,
)
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

# The keyword baseline lives in the package (src/fintech_ai_act). Make it
# importable when the script is run straight from a fresh clone.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
try:
    from fintech_ai_act import EUAIActRiskClassifier
except ImportError:  # pragma: no cover
    EUAIActRiskClassifier = None

SEED = 42
HIGH_RISK = "High-Risk"
CANONICAL_ORDER = ["Minimal-Risk", "Limited-Risk", "High-Risk", "Unacceptable"]

# Fixed colour per risk tier (colour follows the entity, not its rank).
CLASS_COLORS = {
    "High-Risk": "#2a78d6",
    "Limited-Risk": "#eb6834",
    "Minimal-Risk": "#1baf7a",
    "Unacceptable": "#eda100",
}
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e3e2dd"

STOP_WORDS = sorted(ENGLISH_STOP_WORDS | {"want", "wants", "able", "like", "would"})


# --------------------------------------------------------------------------- #
# Plot helpers
# --------------------------------------------------------------------------- #
def style_axes(ax, title: str | None = None, xlabel: str | None = None, ylabel: str | None = None):
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


def save_fig(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)
    print(f"   saved {path}")


def class_color(name: str) -> str:
    return CLASS_COLORS.get(name, "#6250d6")


# --------------------------------------------------------------------------- #
# 1. Load data
# --------------------------------------------------------------------------- #
def load_data(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xls", ".xlsm"):
        return pd.read_excel(path)
    if suffix == ".json":
        return pd.read_json(path)
    last_err = None
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc, sep=None, engine="python")
        except Exception as err:  # noqa: BLE001
            last_err = err
    raise RuntimeError(f"Could not read {path}: {last_err}")


def find_data_file(explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.exists():
            sys.exit(f"Data file not found: {p}")
        return p
    here = Path(__file__).resolve().parent
    for folder in (here / "data", here, Path.cwd() / "data", Path.cwd()):
        for pattern in ("*.csv", "*.xlsx", "*.xls", "*.json"):
            hits = sorted(folder.glob(pattern)) if folder.exists() else []
            hits = [h for h in hits if "predictions" not in h.name and "metrics" not in h.name]
            if hits:
                return hits[0]
    sys.exit(
        "No data file found. Download the dataset from Mendeley Data "
        "(DOI 10.17632/ww3wmkw466.1), put it in ./data/, or pass --data <file>."
    )


def detect_columns(df: pd.DataFrame, text_cols, label_col, domain_col):
    """Pick the user-story text column(s), the risk label and the domain column."""
    cols = list(df.columns)
    norm = {c: re.sub(r"[^a-z0-9]+", "_", str(c).lower()).strip("_") for c in cols}

    # --- label ---
    if label_col is None:
        cands = [c for c in cols if re.search(r"risk|label|tier|class|level", norm[c]) and df[c].nunique(dropna=True) <= 8]
        cands.sort(key=lambda c: (0 if "risk" in norm[c] else 1, df[c].nunique(dropna=True)))
        if not cands:
            sys.exit(f"Could not detect the label column. Columns: {cols}. Use --label-col.")
        label_col = cands[0]
    elif label_col not in cols:
        sys.exit(f"--label-col '{label_col}' not in columns: {cols}")

    # --- text ---
    if text_cols:
        missing = [c for c in text_cols if c not in cols]
        if missing:
            sys.exit(f"--text-cols {missing} not in columns: {cols}")
    else:
        text_cols = [c for c in cols if c != label_col and re.search(r"story|requirement|description|feature|text", norm[c])
                     and pd.api.types.is_string_dtype(df[c])]
        if not text_cols:  # fall back to the longest free-text column
            obj = [c for c in cols if c != label_col and pd.api.types.is_string_dtype(df[c])]
            if not obj:
                sys.exit(f"Could not detect a text column. Columns: {cols}. Use --text-cols.")
            text_cols = [max(obj, key=lambda c: df[c].astype(str).str.len().mean())]

    # --- domain (optional, used for EDA only; never as a feature) ---
    if domain_col is None:
        cands = [c for c in cols if c not in text_cols and c != label_col
                 and re.search(r"domain|area|sector|segment|category|product|module", norm[c])
                 and 2 <= df[c].nunique(dropna=True) <= 12]
        domain_col = cands[0] if cands else None
    return list(text_cols), label_col, domain_col


# --------------------------------------------------------------------------- #
# 2. EDA
# --------------------------------------------------------------------------- #
def run_eda(df: pd.DataFrame, text_cols, label_col, domain_col, fig_dir: Path) -> dict:
    print("\n" + "=" * 70 + "\n2. EXPLORATORY DATA ANALYSIS\n" + "=" * 70)
    text = df[text_cols].fillna("").astype(str).agg(" ".join, axis=1)
    lengths_words = text.str.split().str.len()

    summary = {
        "rows": int(len(df)),
        "columns": list(map(str, df.columns)),
        "dtypes": {str(k): str(v) for k, v in df.dtypes.items()},
        "missing_per_column": {str(k): int(v) for k, v in df.isna().sum().items()},
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_texts": int(text.str.lower().str.strip().duplicated().sum()),
        "raw_label_counts": {str(k): int(v) for k, v in df[label_col].value_counts(dropna=False).items()},
        "words_per_text": {k: round(float(v), 2) for k, v in lengths_words.describe().items()},
    }
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Text column(s): {text_cols} | Label column: '{label_col}' | Domain column: {domain_col}")
    print("\nMissing values per column:")
    print(df.isna().sum().to_string())
    print(f"\nDuplicate rows: {summary['duplicate_rows']} | duplicate story texts: {summary['duplicate_texts']}")
    print("\nRaw label counts:")
    print(df[label_col].value_counts(dropna=False).to_string())
    print("\nWords per story:")
    print(lengths_words.describe().round(1).to_string())

    # Fig 1: raw class distribution
    counts = df[label_col].fillna("(missing)").astype(str).value_counts()
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(counts.index, counts.values, color=[class_color(c) for c in counts.index], width=0.6)
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v} ({v / counts.sum():.0%})", ha="center", va="bottom", fontsize=9, color=INK)
    style_axes(ax, "Risk-tier distribution (raw labels)", ylabel="User stories")
    ax.set_ylim(0, counts.max() * 1.15)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    save_fig(fig, fig_dir / "01_class_distribution_raw.png")

    # Fig 2: story length by class
    # group spelling variants ("high risk", "HIGH-RISK") under their canonical tier for this plot
    disp = df[label_col].map(lambda v: canonical_label(v) or ("(missing)" if pd.isna(v) else str(v).strip()))
    tmp = pd.DataFrame({"words": lengths_words, "label": disp.astype(str)})
    uniq = sorted(tmp["label"].unique())
    order = [c for c in CANONICAL_ORDER if c in uniq] + [c for c in uniq if c not in CANONICAL_ORDER]
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(data=tmp, x="label", y="words", order=order, ax=ax, width=0.5, linewidth=1,
                palette={c: class_color(c) for c in order}, hue="label", legend=False)
    style_axes(ax, "Story length by risk tier", xlabel="", ylabel="Words per story")
    save_fig(fig, fig_dir / "02_text_length_by_class.png")

    # Fig 3: domain vs risk
    if domain_col is not None:
        ct = pd.crosstab(df[domain_col].astype(str), df[label_col].astype(str))
        summary["domain_vs_label"] = ct.to_dict()
        print(f"\nDomain ('{domain_col}') vs risk tier:")
        print(ct.to_string())
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        sns.heatmap(ct, annot=True, fmt="d", cmap="Blues", cbar=False, linewidths=2, linecolor="white", ax=ax)
        ax.set_title("Domain vs risk tier (counts)", loc="left", fontsize=11, fontweight="bold", color=INK)
        ax.set_xlabel("")
        ax.set_ylabel("")
        save_fig(fig, fig_dir / "03_domain_vs_risk.png")

    # Fig 4: most distinctive terms per class (mean TF-IDF)
    labels = df[label_col].astype(str)
    classes = [c for c in CANONICAL_ORDER if c in set(labels)] or sorted(labels.unique())
    if len(classes) >= 2 and len(df) >= 10:
        vec = TfidfVectorizer(stop_words=STOP_WORDS, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        try:
            mat = vec.fit_transform(text.str.lower())
            terms = np.array(vec.get_feature_names_out())
            ncol = 2
            nrow = int(np.ceil(len(classes) / ncol))
            fig, axes = plt.subplots(nrow, ncol, figsize=(10, 3.2 * nrow), squeeze=False)
            for ax in axes.ravel():
                ax.axis("off")
            for ax, cls in zip(axes.ravel(), classes):
                ax.axis("on")
                m = np.asarray(mat[(labels == cls).values].mean(axis=0)).ravel()
                top = m.argsort()[-8:]
                ax.barh(terms[top], m[top], color=class_color(cls), height=0.65)
                style_axes(ax, str(cls), xlabel="mean TF-IDF")
                ax.grid(axis="x", color=GRID)
                ax.grid(axis="y", visible=False)
            fig.suptitle("Most characteristic terms per risk tier", x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
            save_fig(fig, fig_dir / "04_top_terms_per_class.png")
        except ValueError:
            print("   (skipped top-terms plot: vocabulary too small)")
    return summary


# --------------------------------------------------------------------------- #
# 3. Cleaning
# --------------------------------------------------------------------------- #
def canonical_label(value) -> str | None:
    """Map label spellings to the four EU AI Act tiers (None if unrecognised)."""
    if pd.isna(value):
        return None
    v = re.sub(r"[^a-z]+", " ", str(value).lower()).strip()
    if any(k in v for k in ("unacceptable", "prohibit", "blocked", "banned")):
        return "Unacceptable"
    if "high" in v:
        return "High-Risk"
    if "limited" in v or "transparen" in v:
        return "Limited-Risk"
    if "minimal" in v or v in ("low", "low risk", "none", "no risk"):
        return "Minimal-Risk"
    return None


def clean_text(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r"https?://\S+", " ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)       # punctuation -> space
    s = re.sub(r"\b\d+\b", " ", s)            # standalone numbers
    return re.sub(r"\s+", " ", s).strip()


def clean_data(df: pd.DataFrame, text_cols, label_col, domain_col, min_class_size: int):
    print("\n" + "=" * 70 + "\n3. DATA CLEANING\n" + "=" * 70)
    log = {"rows_in": int(len(df))}
    out = df.copy()

    out["text_raw"] = out[text_cols].fillna("").astype(str).apply(lambda r: " ".join(x.strip() for x in r if x.strip()), axis=1)

    # a) drop rows with no text or no label
    n = len(out)
    out = out[out["text_raw"].str.len() > 0]
    out = out[out[label_col].notna()]
    log["dropped_missing_text_or_label"] = int(n - len(out))

    # b) canonicalise labels
    out["label"] = out[label_col].map(canonical_label)
    unmapped = out[out["label"].isna()][label_col].astype(str).unique().tolist()
    if unmapped and out["label"].notna().sum() > 0:
        print(f"   WARNING: unrecognised label values dropped: {unmapped}")
    if out["label"].notna().sum() == 0:  # nothing matched the EU tiers: keep the dataset's own labels
        out["label"] = out[label_col].astype(str).str.strip()
        log["label_mapping"] = "kept original labels (no EU AI Act tier names recognised)"
    else:
        log["label_mapping"] = "mapped to canonical EU AI Act tiers"
    n = len(out)
    out = out[out["label"].notna()]
    log["dropped_unrecognised_label"] = int(n - len(out))

    # c) normalise text
    out["text_clean"] = out["text_raw"].map(clean_text)
    n = len(out)
    out = out[out["text_clean"].str.len() > 0]
    log["dropped_empty_after_cleaning"] = int(n - len(out))

    # d) conflicting duplicates (same text, different labels) are ambiguous -> remove entirely
    grp = out.groupby("text_clean")["label"].nunique()
    conflict = grp[grp > 1].index
    n = len(out)
    out = out[~out["text_clean"].isin(conflict)]
    log["dropped_conflicting_duplicates"] = int(n - len(out))

    # e) exact duplicates -> keep first
    n = len(out)
    out = out.drop_duplicates(subset="text_clean", keep="first")
    log["dropped_duplicates"] = int(n - len(out))

    # f) classes too small to stratify
    vc = out["label"].value_counts()
    tiny = vc[vc < min_class_size].index.tolist()
    if tiny:
        print(f"   WARNING: classes with < {min_class_size} samples removed: {tiny}")
    out = out[~out["label"].isin(tiny)]
    log["dropped_tiny_classes"] = {str(k): int(vc[k]) for k in tiny}

    out = out.reset_index(drop=True)
    log["rows_out"] = int(len(out))
    log["class_counts"] = {str(k): int(v) for k, v in out["label"].value_counts().items()}
    for k, v in log.items():
        print(f"   {k}: {v}")

    # Fig: before/after class counts
    fig, ax = plt.subplots(figsize=(7, 4))
    order = [c for c in CANONICAL_ORDER if c in log["class_counts"]] or list(log["class_counts"])
    vals = [log["class_counts"][c] for c in order]
    bars = ax.bar(order, vals, color=[class_color(c) for c in order], width=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v} ({v / sum(vals):.0%})", ha="center", va="bottom", fontsize=9, color=INK)
    style_axes(ax, f"Class distribution after cleaning (n={len(out)})", ylabel="User stories")
    ax.set_ylim(0, max(vals) * 1.15)
    return out, log, fig


# --------------------------------------------------------------------------- #
# 4-6. Models
# --------------------------------------------------------------------------- #
def tfidf():
    return TfidfVectorizer(stop_words=STOP_WORDS, ngram_range=(1, 2), min_df=1, sublinear_tf=True, strip_accents="unicode")


def build_models(seed: int):
    """name -> (pipeline, param_grid). TF-IDF sits inside the pipeline => no leakage."""
    def pipe(clf):
        return Pipeline([("tfidf", tfidf()), ("clf", clf)])

    return {
        "Logistic Regression": (
            pipe(LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed)),
            {"clf__C": [1, 10, 100]},
        ),
        "Linear SVM": (
            pipe(LinearSVC(class_weight="balanced", random_state=seed, max_iter=10000)),
            {"clf__C": [0.1, 1, 10]},
        ),
        "Multinomial NB": (pipe(MultinomialNB()), {"clf__alpha": [0.05, 0.3, 1.0]}),
        "Complement NB": (pipe(ComplementNB()), {"clf__alpha": [0.05, 0.3, 1.0]}),
        "Random Forest": (
            pipe(RandomForestClassifier(class_weight="balanced_subsample", random_state=seed, n_jobs=-1)),
            {"clf__n_estimators": [200, 400], "clf__max_depth": [None, 30]},
        ),
    }


def metrics_block(y_true, y_pred, labels, high_label=HIGH_RISK) -> dict:
    m = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }
    if high_label in labels:
        idx = labels.index(high_label)
        m["high_risk_recall"] = recall_score(y_true, y_pred, labels=[idx], average=None, zero_division=0)[0]
    else:
        m["high_risk_recall"] = np.nan
    return m


def rule_baseline_predict(texts: pd.Series) -> list[str] | None:
    if EUAIActRiskClassifier is None:
        return None
    clf = EUAIActRiskClassifier()
    return [clf.predict_tier(t) for t in texts]


# --------------------------------------------------------------------------- #
# Main pipeline
# --------------------------------------------------------------------------- #
def run_pipeline(df_raw: pd.DataFrame, args, out_dir: Path) -> dict:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    text_cols, label_col, domain_col = detect_columns(df_raw, args.text_cols, args.label_col, args.domain_col)

    # 2. EDA (on raw data)
    eda = run_eda(df_raw, text_cols, label_col, domain_col, fig_dir)

    # 3. Cleaning
    n_splits = args.cv_folds
    df, clean_log, fig_clean = clean_data(df_raw, text_cols, label_col, domain_col, min_class_size=max(n_splits, 4))
    save_fig(fig_clean, fig_dir / "05_class_distribution_clean.png")
    if df["label"].nunique() < 2:
        sys.exit("Fewer than two classes left after cleaning; cannot train a classifier.")

    # 4. Categories / encoding / split
    print("\n" + "=" * 70 + "\n4. CATEGORIES & TRAIN/TEST SPLIT\n" + "=" * 70)
    le = LabelEncoder().fit(df["label"])
    labels = list(le.classes_)
    labels = [c for c in CANONICAL_ORDER if c in labels] + [c for c in labels if c not in CANONICAL_ORDER]
    le.classes_ = np.array(labels)  # keep a meaningful, stable class order
    y = le.transform(df["label"])
    X = df["text_clean"]
    print("Encoded classes:", {c: int(i) for i, c in enumerate(labels)})

    X_tr, X_te, y_tr, y_te, idx_tr, idx_te = train_test_split(
        X, y, df.index, test_size=args.test_size, stratify=y, random_state=SEED
    )
    print(f"Train: {len(X_tr)} | Test: {len(X_te)} (stratified, test_size={args.test_size})")

    inner_k = max(2, min(3, int(np.bincount(y_tr).min())))
    inner_cv = StratifiedKFold(n_splits=inner_k, shuffle=True, random_state=SEED)
    outer_k = max(2, min(n_splits, int(np.bincount(y).min())))
    outer_cv = StratifiedKFold(n_splits=outer_k, shuffle=True, random_state=SEED)

    # 5-6. Train, tune, predict
    print("\n" + "=" * 70 + "\n5-6. TRAINING, TUNING & PREDICTION\n" + "=" * 70)
    print(f"Hyper-parameter search: {inner_k}-fold CV on the training set | robustness check: nested {outer_k}-fold CV on all data")
    rows, fitted, test_pred, fold_scores = [], {}, {}, {}
    for name, (pipe, grid) in build_models(SEED).items():
        print(f" - {name} ...", flush=True)
        gs = GridSearchCV(pipe, grid, cv=inner_cv, scoring="f1_macro", n_jobs=-1, refit=True)
        gs.fit(X_tr, y_tr)
        y_hat = gs.predict(X_te)
        fitted[name], test_pred[name] = gs, y_hat

        # Nested CV: tuning happens inside every outer training fold -> unbiased estimate.
        fold_acc, fold_f1 = [], []
        for tr, va in outer_cv.split(X, y):
            inner = GridSearchCV(pipe, grid, cv=StratifiedKFold(max(2, min(3, int(np.bincount(y[tr]).min()))), shuffle=True, random_state=SEED),
                                 scoring="f1_macro", n_jobs=-1)
            inner.fit(X.iloc[tr], y[tr])
            p = inner.predict(X.iloc[va])
            fold_acc.append(accuracy_score(y[va], p))
            fold_f1.append(f1_score(y[va], p, average="macro", zero_division=0))
        fold_scores[name] = {"accuracy": fold_acc, "macro_f1": fold_f1}

        row = {"model": name, "type": "ML (TF-IDF)", "best_params": json.dumps({k.split("__")[1]: v for k, v in gs.best_params_.items()}),
               "cv_accuracy_mean": np.mean(fold_acc), "cv_accuracy_std": np.std(fold_acc),
               "cv_macro_f1_mean": np.mean(fold_f1), "cv_macro_f1_std": np.std(fold_f1)}
        row.update({f"test_{k}": v for k, v in metrics_block(y_te, y_hat, labels).items()})
        rows.append(row)

    # Baselines (same split, same folds)
    maj = DummyClassifier(strategy="most_frequent").fit(X_tr, y_tr)
    maj_cv_acc, maj_cv_f1 = [], []
    for tr, va in outer_cv.split(X, y):
        p = DummyClassifier(strategy="most_frequent").fit(X.iloc[tr], y[tr]).predict(X.iloc[va])
        maj_cv_acc.append(accuracy_score(y[va], p))
        maj_cv_f1.append(f1_score(y[va], p, average="macro", zero_division=0))
    row = {"model": "Majority class", "type": "Baseline", "best_params": "{}",
           "cv_accuracy_mean": np.mean(maj_cv_acc), "cv_accuracy_std": np.std(maj_cv_acc),
           "cv_macro_f1_mean": np.mean(maj_cv_f1), "cv_macro_f1_std": np.std(maj_cv_f1)}
    row.update({f"test_{k}": v for k, v in metrics_block(y_te, maj.predict(X_te), labels).items()})
    rows.append(row)
    test_pred["Majority class"] = maj.predict(X_te)

    raw_text = df["text_raw"]
    rule_pred = rule_baseline_predict(raw_text)
    if rule_pred is not None and set(rule_pred) & set(labels):
        rp = np.array([labels.index(p) if p in labels else -1 for p in rule_pred])  # -1 = tier not in data
        cv_acc, cv_f1 = [], []
        for _, va in outer_cv.split(X, y):
            cv_acc.append(accuracy_score(y[va], rp[va]))
            cv_f1.append(f1_score(y[va], rp[va], average="macro", zero_division=0))
        row = {"model": "Keyword rules (v1.0 baseline)", "type": "Baseline", "best_params": "{}",
               "cv_accuracy_mean": np.mean(cv_acc), "cv_accuracy_std": np.std(cv_acc),
               "cv_macro_f1_mean": np.mean(cv_f1), "cv_macro_f1_std": np.std(cv_f1)}
        row.update({f"test_{k}": v for k, v in metrics_block(y_te, rp[idx_te], labels).items()})
        rows.append(row)
        test_pred["Keyword rules (v1.0 baseline)"] = rp[idx_te]

    # 7. Comparison
    print("\n" + "=" * 70 + "\n7. MODEL COMPARISON\n" + "=" * 70)
    res = pd.DataFrame(rows)
    ml = res[res["type"] == "ML (TF-IDF)"]
    best_name = ml.sort_values(["cv_macro_f1_mean", "test_macro_f1"], ascending=False).iloc[0]["model"]
    res["selected"] = res["model"].eq(best_name)
    res = res.sort_values(["cv_macro_f1_mean", "test_macro_f1"], ascending=False).reset_index(drop=True)
    show = res[["model", "cv_accuracy_mean", "cv_macro_f1_mean", "test_accuracy", "test_macro_f1", "test_high_risk_recall", "test_mcc"]]
    print(show.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nBest ML model (highest nested-CV macro-F1): {best_name}")
    res.to_csv(out_dir / "model_comparison.csv", index=False)

    # 8. Accuracy detail for the best model
    print("\n" + "=" * 70 + f"\n8. DETAILED ACCURACY - {best_name}\n" + "=" * 70)
    best = fitted[best_name]
    yb = test_pred[best_name]
    report = classification_report(y_te, yb, labels=list(range(len(labels))), target_names=labels, zero_division=0, output_dict=True)
    print(classification_report(y_te, yb, labels=list(range(len(labels))), target_names=labels, zero_division=0))
    pd.DataFrame(report).T.to_csv(out_dir / "classification_report_best_model.csv")

    pred_df = pd.DataFrame({
        "story": df.loc[idx_te, "text_raw"].values,
        "true_label": le.inverse_transform(y_te),
        "predicted_label": le.inverse_transform(yb),
    })
    pred_df["correct"] = pred_df["true_label"] == pred_df["predicted_label"]
    pred_df.to_csv(out_dir / "test_predictions.csv", index=False)
    errors = pred_df[~pred_df["correct"]]
    errors.to_csv(out_dir / "misclassified_test_stories.csv", index=False)
    print(f"Misclassified test stories: {len(errors)} of {len(pred_df)} -> {out_dir / 'misclassified_test_stories.csv'}")

    # ----------------------------- figures ------------------------------- #
    print("\n9. FIGURES")
    # 06 model comparison (CV vs held-out)
    order = res["model"].tolist()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, (metric, title) in zip(axes, [("accuracy", "Accuracy"), ("macro_f1", "Macro-F1")]):
        x = np.arange(len(order))
        cv_vals = res[f"cv_{metric}_mean"].values if f"cv_{metric}_mean" in res else res["cv_macro_f1_mean"].values
        te_vals = res[f"test_{metric}"].values
        b1 = ax.barh(x - 0.19, cv_vals, height=0.36, color=BLUE, label="Nested CV (mean)")
        b2 = ax.barh(x + 0.19, te_vals, height=0.36, color=ORANGE, label="Held-out test")
        for bars in (b1, b2):
            for b in bars:
                ax.text(b.get_width() + 0.01, b.get_y() + b.get_height() / 2, f"{b.get_width():.2f}", va="center", fontsize=8, color=INK)
        ax.set_yticks(x)
        ax.set_yticklabels(order)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.12)
        style_axes(ax, title)
        ax.grid(axis="x", color=GRID)
        ax.grid(axis="y", visible=False)
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, fontsize=9, loc="upper right", ncol=2)
    fig.suptitle("Model comparison", x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    save_fig(fig, fig_dir / "06_model_comparison.png")

    # 07 CV fold spread
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ml_order = [m for m in order if m in fold_scores]
    data = [fold_scores[m]["macro_f1"] for m in ml_order]
    bp = ax.boxplot(data, vert=False, patch_artist=True, widths=0.5, tick_labels=ml_order)
    for patch in bp["boxes"]:
        patch.set(facecolor=BLUE, alpha=0.35, edgecolor=BLUE)
    for med in bp["medians"]:
        med.set(color=INK, linewidth=1.5)
    ax.invert_yaxis()
    style_axes(ax, f"Macro-F1 across {outer_k} nested-CV folds", xlabel="Macro-F1")
    ax.grid(axis="x", color=GRID)
    ax.grid(axis="y", visible=False)
    save_fig(fig, fig_dir / "07_cv_fold_spread.png")

    # 08 confusion matrix of best model (counts + row-normalised)
    cm = confusion_matrix(y_te, yb, labels=list(range(len(labels))))
    cmn = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, data_, fmt, ttl in [(axes[0], cm, "d", "Counts"), (axes[1], cmn, ".0%", "Row-normalised (recall)")]:
        sns.heatmap(data_, annot=True, fmt=fmt, cmap="Blues", cbar=False, xticklabels=labels, yticklabels=labels,
                    linewidths=2, linecolor="white", ax=ax)
        ax.set_title(ttl, loc="left", fontsize=10, color=INK, fontweight="bold")
        ax.set_xlabel("Predicted", color=MUTED)
        ax.set_ylabel("Actual", color=MUTED)
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.suptitle(f"Confusion matrix - {best_name} (test set)", x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    save_fig(fig, fig_dir / "08_confusion_matrix_best.png")

    # 09 confusion matrices, all models
    names = list(test_pred)
    ncol = 3
    nrow = int(np.ceil(len(names) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.9 * nrow), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, nm in zip(axes.ravel(), names):
        ax.axis("on")
        c = confusion_matrix(y_te, test_pred[nm], labels=list(range(len(labels))))
        sns.heatmap(c, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=[l[:4] for l in labels],
                    yticklabels=[l[:4] for l in labels], linewidths=1.5, linecolor="white", ax=ax)
        acc = accuracy_score(y_te, test_pred[nm])
        ax.set_title(f"{nm} ({acc:.0%})", loc="left", fontsize=9, color=INK, fontweight="bold")
    save_fig(fig, fig_dir / "09_confusion_matrices_all_models.png")

    # 10 per-class F1 heatmap
    f1_tbl = pd.DataFrame(
        {nm: f1_score(y_te, test_pred[nm], labels=list(range(len(labels))), average=None, zero_division=0) for nm in names},
        index=labels,
    ).T
    fig, ax = plt.subplots(figsize=(7.5, 0.55 * len(names) + 1.8))
    sns.heatmap(f1_tbl, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1, cbar=False, linewidths=2, linecolor="white", ax=ax)
    ax.set_title("Per-class F1 on the test set", loc="left", fontsize=11, fontweight="bold", color=INK)
    ax.set_xlabel("")
    ax.set_ylabel("")
    save_fig(fig, fig_dir / "10_per_class_f1_heatmap.png")

    # 11 top features of the best model (if it exposes weights)
    est = best.best_estimator_
    vec, clf = est.named_steps["tfidf"], est.named_steps["clf"]
    terms = np.array(vec.get_feature_names_out())
    weights = None
    if hasattr(clf, "coef_") and clf.coef_.shape[0] == len(labels):
        weights = clf.coef_
    elif hasattr(clf, "feature_log_prob_"):
        weights = clf.feature_log_prob_ - clf.feature_log_prob_.mean(axis=0)
    elif hasattr(clf, "feature_importances_"):
        weights = None
        top = clf.feature_importances_.argsort()[-15:]
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.barh(terms[top], clf.feature_importances_[top], color=BLUE, height=0.65)
        style_axes(ax, f"Top 15 terms - {best_name}", xlabel="Importance")
        ax.grid(axis="x", color=GRID)
        ax.grid(axis="y", visible=False)
        save_fig(fig, fig_dir / "11_top_features_best_model.png")
    if weights is not None:
        ncol = 2
        nrow = int(np.ceil(len(labels) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(10, 3.3 * nrow), squeeze=False)
        for ax in axes.ravel():
            ax.axis("off")
        for ax, (i, cls) in zip(axes.ravel(), enumerate(labels)):
            ax.axis("on")
            top = weights[i].argsort()[-10:]
            ax.barh(terms[top], weights[i][top], color=class_color(cls), height=0.65)
            style_axes(ax, cls, xlabel="weight")
            ax.grid(axis="x", color=GRID)
            ax.grid(axis="y", visible=False)
        fig.suptitle(f"Terms that push a story toward each tier - {best_name}", x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
        save_fig(fig, fig_dir / "11_top_features_best_model.png")

    # 12 learning curve (does more data help?)
    try:
        sizes, tr_s, va_s = learning_curve(
            fitted[best_name].best_estimator_, X, y, cv=outer_cv, scoring="f1_macro",
            train_sizes=np.linspace(0.2, 1.0, 5), n_jobs=-1, random_state=SEED, shuffle=True,
        )
        fig, ax = plt.subplots(figsize=(7, 4.2))
        for vals, color, lab in [(tr_s, BLUE, "Training"), (va_s, ORANGE, "Validation")]:
            m, s = vals.mean(axis=1), vals.std(axis=1)
            ax.plot(sizes, m, color=color, linewidth=2, marker="o", markersize=6, markeredgecolor="white", label=lab)
            ax.fill_between(sizes, m - s, m + s, color=color, alpha=0.12)
        style_axes(ax, f"Learning curve - {best_name}", xlabel="Training samples", ylabel="Macro-F1")
        ax.set_ylim(0, 1.05)
        ax.legend(frameon=False, fontsize=9)
        save_fig(fig, fig_dir / "12_learning_curve.png")
    except Exception as err:  # noqa: BLE001
        print(f"   (learning curve skipped: {err})")

    # Persist the final model: best configuration refit on ALL cleaned data
    final = Pipeline([("tfidf", tfidf()), ("clf", best.best_estimator_.named_steps["clf"])]).fit(X, y)
    joblib.dump({"model": final, "label_encoder": le, "classes": labels}, out_dir / "best_model.joblib")

    summary = {
        "data_columns": {"text": text_cols, "label": label_col, "domain": domain_col},
        "eda": eda,
        "cleaning": clean_log,
        "classes": labels,
        "split": {"train": int(len(X_tr)), "test": int(len(X_te)), "outer_cv_folds": outer_k},
        "best_model": best_name,
        "best_model_test_metrics": {k: float(v) for k, v in metrics_block(y_te, yb, labels).items()},
        "best_model_nested_cv": {
            "accuracy_mean": float(np.mean(fold_scores[best_name]["accuracy"])),
            "accuracy_std": float(np.std(fold_scores[best_name]["accuracy"])),
            "macro_f1_mean": float(np.mean(fold_scores[best_name]["macro_f1"])),
            "macro_f1_std": float(np.std(fold_scores[best_name]["macro_f1"])),
        },
        "demo_data": bool(args.demo),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


# --------------------------------------------------------------------------- #
# Synthetic data (smoke test ONLY - never report results from this)
# --------------------------------------------------------------------------- #
def make_demo_data(n: int = 350, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    domains = {
        "Lending": ["loan applicants", "borrowers", "credit officers"],
        "Fraud & Payments": ["payment analysts", "merchants", "compliance officers"],
        "Wealth Management": ["investors", "portfolio managers", "advisors"],
        "Insurance": ["policyholders", "claims handlers", "underwriters"],
        "Customer Support": ["customers", "support agents", "new users"],
    }
    tiers = {
        "High-Risk": ["automatically score applicant creditworthiness and decide approval", "price premiums from a model trained on personal health data",
                      "verify identity through facial biometric matching before onboarding", "rank applicants for eligibility using an automated scoring model",
                      "freeze accounts automatically when the model flags suspicious behaviour"],
        "Limited-Risk": ["talk to a chatbot that answers questions about my account", "receive replies from a virtual assistant that must disclose it is an AI",
                         "get generated summaries of my statements labelled as machine produced", "use a conversational agent to explain product options"],
        "Minimal-Risk": ["sort my monthly expenses by date and category", "export my transaction history to a spreadsheet", "set a reminder before a bill is due",
                         "change the colour theme of the dashboard", "search past invoices by keyword"],
        "Unacceptable": ["assign citizens a social scoring rank based on behaviour", "nudge users with subliminal manipulation to take on more debt",
                         "exploit vulnerabilities of elderly users to sell products"],
    }
    weights = {"High-Risk": 0.38, "Limited-Risk": 0.22, "Minimal-Risk": 0.34, "Unacceptable": 0.06}
    rows = []
    for _ in range(n):
        dom = rng.choice(list(domains))
        tier = rng.choice(list(weights), p=list(weights.values()))
        persona = rng.choice(domains[dom])
        action = rng.choice(tiers[tier])
        benefit = rng.choice(["save time", "reduce errors", "stay compliant", "get clearer information", "improve trust"])
        rows.append({"Domain": dom, "User_Story": f"As a {persona.rstrip('s')}, I want to {action} so that I can {benefit}.", "Risk_Level": tier})
    df = pd.DataFrame(rows)
    # inject realistic mess so cleaning is exercised
    df.loc[rng.choice(n, 4, replace=False), "Risk_Level"] = ["high risk", "HIGH-RISK", "Unacceptable/Blocked", " minimal risk "]
    df.loc[rng.choice(n, 3, replace=False), "User_Story"] = None
    return pd.concat([df, df.sample(5, random_state=seed)], ignore_index=True)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="EU AI Act risk-tier classification of FinTech user stories.")
    p.add_argument("--data", help="Path to the dataset (.csv/.xlsx/.json). Default: first file found in ./data")
    p.add_argument("--text-cols", nargs="+", help="Column(s) holding the story/requirement text (auto-detected)")
    p.add_argument("--label-col", help="Column with the risk label (auto-detected)")
    p.add_argument("--domain-col", help="Optional domain column, used for EDA only")
    p.add_argument("--out", default="outputs", help="Output directory (default: outputs)")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--demo", action="store_true", help="Use synthetic data to smoke-test the pipeline (NOT real results)")
    return p.parse_args(argv)


def main(argv=None):
    warnings.filterwarnings("ignore")
    args = parse_args(argv)
    sns.set_theme(style="white")
    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": "white"})
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70 + "\n1. LOADING DATA\n" + "=" * 70)
    if args.demo:
        print("!! DEMO MODE: synthetic data. These numbers say nothing about the real dataset. !!")
        df = make_demo_data()
    else:
        path = find_data_file(args.data)
        df = load_data(path)
        print(f"Loaded {path} -> {df.shape[0]} rows x {df.shape[1]} columns")
    print(df.head(3).to_string())

    summary = run_pipeline(df, args, out_dir)

    m, cv = summary["best_model_test_metrics"], summary["best_model_nested_cv"]
    print("\n" + "=" * 70 + "\nDONE\n" + "=" * 70)
    print(f"Best model      : {summary['best_model']}")
    print(f"Test accuracy   : {m['accuracy']:.3f}  | macro-F1 {m['macro_f1']:.3f}  | High-Risk recall {m['high_risk_recall']:.3f}")
    print(f"Nested CV       : accuracy {cv['accuracy_mean']:.3f} +/- {cv['accuracy_std']:.3f} | macro-F1 {cv['macro_f1_mean']:.3f} +/- {cv['macro_f1_std']:.3f}")
    print(f"Outputs in      : {out_dir.resolve()}")
    if args.demo:
        print("(synthetic demo data - do not cite these numbers)")


if __name__ == "__main__":
    main()
