# FinTech Agile User Stories & EU AI Act Risk Classification (`fintech-eu-ai-act-classifier`)

Python pipeline accompanying the **FinTech Software Product Requirements, Agile User Stories, and EU AI Act Risk Classification Dataset** (350 annotated FinTech user stories across Lending, Fraud & Payments, Wealth Management, Insurance, and Customer Support):

> Khan, H. S., **Khan, Rabia**, & Khan, M. W. (2026). *FinTech Software Product Requirements, Agile User Stories, and EU AI Act Risk Classification Dataset*. **Mendeley Data**, V1. DOI: `10.17632/ww3wmkw466.1`

## Author
- **Co-Author & Maintainer:** **Rabia Khan** (Lecturer in Information Technology, University of Haripur | M.Sc. in Data Science, PAF-IAST | B.Sc. in Computer Engineering, COMSATS Abbottabad)
- **ORCID:** `0009-0007-5575-1564`

## What the pipeline does (`fintech_risk_pipeline.py`)

Every number is computed from the data; nothing is hard-coded.

| Step | What happens |
|---|---|
| 1. Load | CSV / XLSX / JSON, encoding-safe. Story, label and domain columns are auto-detected (override with flags). |
| 2. EDA | Shape, dtypes, missing values, duplicates, class balance, story length, domain-vs-risk table, most characteristic terms per tier. |
| 3. Clean | Label spellings mapped to the 4 EU AI Act tiers, missing/empty rows, conflicting and exact duplicates removed, text normalised. A cleaning log is printed and saved. |
| 4. Categories | Label encoding and a stratified train/test split. |
| 5. Train | TF-IDF (1-2 grams) with Logistic Regression, Linear SVM, Multinomial NB, Complement NB and Random Forest. Hyper-parameters tuned with inner CV on the training set only. |
| 6. Predict | Held-out test predictions plus nested stratified k-fold CV on all data. |
| 7. Compare | ML models vs. the v1.0 keyword-rule baseline vs. a majority-class baseline, on identical splits. |
| 8. Accuracy | Accuracy, balanced accuracy, macro/weighted F1, MCC, High-Risk recall, per-class report, misclassified stories. |
| 9. Visualise | 12 PNG figures: class balance, length, domain heatmap, top terms, model comparison, CV spread, confusion matrices, per-class F1, learned terms, learning curve. |

## Quickstart
```bash
pip install -r requirements.txt

# put the Mendeley file in ./data/ (or pass --data), then:
python fintech_risk_pipeline.py --data data/your_file.csv

# if auto-detection picks the wrong columns:
python fintech_risk_pipeline.py --data data/your_file.csv --text-cols "User Story" --label-col "Risk Level"

# smoke-test the pipeline on synthetic data (results are meaningless):
python fintech_risk_pipeline.py --demo

python -m unittest discover -s tests -v
```

Outputs are written to `outputs/`: `model_comparison.csv`, `classification_report_best_model.csv`, `test_predictions.csv`, `misclassified_test_stories.csv`, `summary.json`, `best_model.joblib` and `figures/`.

## Notes on interpreting results
- With ~350 rows, a single split is noisy. Prefer the nested-CV mean +/- std reported next to the test score.
- Only the story text is used as a feature; the domain column is used for EDA only.
- `EUAIActRiskClassifier` (in `src/fintech_ai_act`) is a keyword-rule baseline, kept so the trained models have something honest to beat.
