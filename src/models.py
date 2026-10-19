"""Model training: Logistic Regression, Decision Tree, Random Forest (notebook cell 18).

Extracted from credit_default_risk_prediction.ipynb — logic unchanged, organized into functions.
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


def train_models(X_train, y_train, random_state=42):
    """Train all three classifiers; return (lr, dt, rf)."""
    lr = LogisticRegression(random_state=random_state, solver="liblinear")
    lr.fit(X_train, y_train)

    dt = DecisionTreeClassifier(random_state=random_state, max_depth=5)
    dt.fit(X_train, y_train)

    rf = RandomForestClassifier(random_state=random_state, n_estimators=100)
    rf.fit(X_train, y_train)
    return lr, dt, rf
