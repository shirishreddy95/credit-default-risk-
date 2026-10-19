# Credit Default Risk Prediction

Predicting loan default risk with machine learning — **AI 682 (Data Mining) final project, Long Island University**.

Compares three classifiers — Logistic Regression, Decision Tree, and Random Forest — on 32,582 consumer loan records. Class imbalance is handled with SMOTE, and the winning model is explained with SHAP.

## Results

Test set (20% holdout, 6,517 records):

| Model | Accuracy | ROC AUC | F1 (defaulters) |
|---|---|---|---|
| Logistic Regression | 0.82 | 0.88 | 0.66 |
| Decision Tree | 0.88 | 0.88 | 0.73 |
| **Random Forest** | **0.92** | **0.93** | **0.81** |

Random Forest wins on every metric. SHAP analysis flags the strongest default signals: `loan_percent_income`, `loan_int_rate`, `debt_to_income`, `person_income`, `loan_amnt`.

## Pipeline

1. **Preprocessing** — median imputation for numerics, mode for categoricals; Y/N → binary; one-hot encoding
2. **Feature engineering** — debt-to-income ratio (`loan_amnt / person_income`)
3. **Split & scale** — 80/20 train/test split, z-score standardization
4. **Imbalance** — SMOTE oversampling on the training set
5. **Modeling** — Logistic Regression (liblinear), Decision Tree (max depth 5), Random Forest (100 trees)
6. **Evaluation** — classification report, confusion matrix, ROC AUC
7. **Interpretability** — SHAP summary plot + feature importance

## Quick start

```bash
pip install -r requirements.txt
```

Download `credit_risk_dataset.csv` from [Kaggle](https://www.kaggle.com/datasets/laotse/credit-risk-dataset) into `data/` (see [data/README.md](data/README.md)), then either open `credit_default_risk_prediction.ipynb` or run the modules:

```python
from src.preprocessing import impute_missing, encode_categoricals, engineer_features, make_splits
from src.models import train_models
from src.evaluate import evaluate_model
from src.interpret import shap_analysis
import pandas as pd

df = pd.read_csv("data/credit_risk_dataset.csv")
df = engineer_features(encode_categoricals(impute_missing(df)))
X_train, X_test, y_train, y_test, features = make_splits(df)
lr, dt, rf = train_models(X_train, y_train)
evaluate_model(rf, X_test, y_test, "Random Forest")
shap_analysis(rf, X_test, features)
```

## Project structure

```
credit-default-risk/
├── credit_default_risk_prediction.ipynb  # Main notebook: EDA, modeling, evaluation
├── src/
│   ├── preprocessing.py                  # Imputation, encoding, feature engineering, SMOTE splits
│   ├── models.py                         # Logistic Regression / Decision Tree / Random Forest
│   ├── evaluate.py                       # Classification report, confusion matrix, ROC AUC
│   └── interpret.py                      # SHAP analysis, feature importance
├── requirements.txt
└── data/                                 # Dataset goes here (not included, see below)
```

The `src/` modules are extracted from the notebook — same code, organized for reading and reuse.

## Dataset

The training data (32,582 rows × 12 columns: age, income, home ownership, employment length, loan intent/grade/amount/rate, loan status, loan percent of income, prior default flag, credit history length) is not included in this repo. Download `credit_risk_dataset.csv` from the [Kaggle credit risk dataset](https://www.kaggle.com/datasets/laotse/credit-risk-dataset) and place it in `data/`.

## Notes

- The target (`loan_status`) is imbalanced (~22% defaults) — SMOTE is applied to the training split only, never the test set.
- Decision Tree depth is capped at 5 for interpretability; Random Forest uses 100 trees.
