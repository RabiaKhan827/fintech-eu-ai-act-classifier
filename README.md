# FinTech Agile User Stories & EU AI Act Risk Classification

A complete, single-file data-science project that classifies FinTech Agile user stories into EU AI Act risk tiers (`Minimal-Risk`, `Limited-Risk`, `High-Risk`, `Unacceptable`).

Dataset (350 annotated stories across Lending, Fraud & Payments, Wealth Management, Insurance, Customer Support):

> Khan, H. S., **Khan, Rabia**, & Khan, M. W. (2026). *FinTech Software Product Requirements, Agile User Stories, and EU AI Act Risk Classification Dataset*. **Mendeley Data**, V1. DOI: `10.17632/ww3wmkw466.1`

**Author / maintainer:** Rabia Khan (University of Haripur; PAF-IAST) - ORCID `0009-0007-5575-1564`

## Everything is in one file: `fintech_risk_classification.py`

The file is divided into numbered parts that follow the data-science workflow; `main()` / `run()` call them in order.

| Part | What it does |
|---|---|
| 1 Load | Reads CSV/XLSX/JSON and auto-detects the story, label and domain columns. |
| 2 EDA | Shape, types, missing values, duplicates, class balance, story length, domain-vs-risk, characteristic words per tier. |
| 3 Cleaning + audit | Normalises label spellings, drops empty / conflicting / duplicate rows (all counted in a log), normalises text, and audits **near-duplicate stories** (possible label errors and train/test leakage). |
| 4 Categories | Label encoding in a meaningful order, stratified train/test split. |
| 5 Feature engineering | Compares word, word+bigram and word+character TF-IDF representations. |
| 6 Train / tune / predict | Logistic Regression, Linear SVM, Multinomial NB, Complement NB, Random Forest. Settings tuned by cross-validation on the training set; nested CV over all stories gives an out-of-fold prediction for every story. |
| 7 Evaluation | Accuracy, balanced accuracy, macro/weighted F1, MCC, High-Risk recall, **95% bootstrap confidence intervals**, **McNemar significance tests** vs. the keyword baseline, a majority-class baseline and the runner-up model. |
| 8 Extra analysis | Unsupervised check (SVD + KMeans vs. true tiers: ARI/NMI/silhouette), per-domain accuracy, error analysis (most confused tier pairs, every misclassified story). |
| 9 Figures | 15 PNG figures. |
| 10 Report | Writes `outputs/RESULTS.md`, a full results report generated from the run. |
| 11 Predict | `--predict "<story>"` classifies a new story with the saved model. |

The v1.0 keyword rules are kept only as a baseline (`KeywordRuleBaseline`) so the trained models have something honest to beat. No numbers are hard-coded.

## Run it
```bash
pip install -r requirements.txt

# 1) put the Mendeley file in ./data/ (or pass --data), then run:
python fintech_risk_classification.py --data data/<your_file>.csv

# if column auto-detection picks the wrong ones:
python fintech_risk_classification.py --data data/<file>.csv --text-cols "User Story" --label-col "Risk Level" --domain-col "Domain"

# 2) classify a new story with the saved model:
python fintech_risk_classification.py --predict "As a borrower, I want an AI model to approve my loan automatically"

# smoke-test the code on synthetic data (results are meaningless):
python fintech_risk_classification.py --demo

python -m unittest discover -s tests -v
```

## Outputs (`outputs/`)

| File | Contents |
|---|---|
| `RESULTS.md` | Full report: cleaning log, EDA findings, model comparison, confidence intervals, significance tests, error and per-domain analysis, with all figures embedded. |
| `run_log.txt` | Complete console output of the run. |
| `model_comparison.csv`, `classification_report_best_model.csv`, `summary.json` | Machine-readable metrics. |
| `test_predictions.csv`, `misclassified_stories.csv`, `near_duplicate_pairs.csv` | Row-level predictions, errors and data-quality findings. |
| `figures/*.png` | 15 figures. |
| `best_model.joblib` | Final model refit on all stories (git-ignored). |

## Reading the results
- With about 350 stories a single split is noisy, so each model is also scored with nested cross-validation and a bootstrap confidence interval.
- Only the story text is used as a model feature; the domain column is used for analysis only.
- High-Risk recall is reported separately because missing a high-risk story is the costly error.
