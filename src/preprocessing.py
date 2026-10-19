"""Preprocessing: missing-value imputation, categorical encoding, feature engineering, scaling, SMOTE.

Extracted from credit_default_risk_prediction.ipynb — logic unchanged, organized into functions.
"""
import warnings

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def impute_missing(df):
    """Impute numerical features with median, categoricals with mode (notebook cell 8)."""
    df = df.copy()
    num_cols = df.select_dtypes(include=np.number).columns
    imputer_num = SimpleImputer(strategy="median")
    df[num_cols] = imputer_num.fit_transform(df[num_cols])

    cat_cols = df.select_dtypes(include="object").columns
    imputer_cat = SimpleImputer(strategy="most_frequent")
    df[cat_cols] = imputer_cat.fit_transform(df[cat_cols])
    return df


def encode_categoricals(df):
    """Binary-encode cb_person_default_on_file; one-hot encode home ownership, intent, grade (cell 10)."""
    df = df.copy()
    df['cb_person_default_on_file'] = df['cb_person_default_on_file'].map({'Y': 1, 'N': 0})
    df = pd.get_dummies(df, columns=["person_home_ownership", "loan_intent", "loan_grade"], drop_first=True)
    return df


def engineer_features(df):
    """Debt-to-income ratio (cell 13)."""
    df = df.copy()
    df['debt_to_income'] = df['loan_amnt'] / df['person_income']
    return df


def make_splits(df, test_size=0.2, random_state=42):
    """80/20 split, z-score standardization, SMOTE on the training set (cell 13)."""
    warnings.filterwarnings("ignore", category=FutureWarning)
    X = df.drop(columns=['loan_status'])
    y = df['loan_status']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    smote = SMOTE(random_state=random_state)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    return X_train_res, X_test, y_train_res, y_test, X.columns
