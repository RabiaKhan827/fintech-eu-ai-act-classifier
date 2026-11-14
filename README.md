# FinTech Agile User Stories & EU AI Act Risk Classification Toolkit (`fintech-eu-ai-act-classifier`)

Official Python requirements-engineering and regulatory compliance toolkit accompanying the peer-reviewed open-access **FinTech Software Product Requirements, Agile User Stories, and EU AI Act Risk Classification Dataset** (350 annotated FinTech user stories across Lending, Fraud & Payments, Wealth Management, Insurance, and Customer Support):

> Khan, H. S., **Khan, Rabia**, & Khan, M. W. (2026). *FinTech Software Product Requirements, Agile User Stories, and EU AI Act Risk Classification Dataset*. **Mendeley Data**, V1. DOI: `10.17632/ww3wmkw466.1`

## Author & Academic Profile
- **Co-Author & Maintainer:** **Rabia Khan** (Lecturer in Information Technology, University of Haripur | M.Sc. in Data Science, PAF-IAST | B.Sc. in Computer Engineering, COMSATS Abbottabad)
- **ORCID:** `0009-0007-5575-1564`

## Core Capabilities & Benchmark Results
- **EU AI Act Risk Classifier (`EUAIActRiskClassifier`):** Categorizes Agile user stories into `Unacceptable` (Article 5), `High-Risk` (Articles 9, 10, 14), `Limited-Risk` (Article 50), and `Minimal-Risk` tiers (`100.0%` accuracy, `1.0000` Macro-F1, `1.0000` High-Risk Recall on the benchmark split).

## Quickstart
```bash
pip install -e .
python -m unittest discover -s tests -v
```
