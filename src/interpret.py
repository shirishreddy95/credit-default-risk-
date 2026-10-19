"""Interpretability: SHAP summary + Random Forest feature importance (notebook cells 22, 25).

Extracted from credit_default_risk_prediction.ipynb — logic unchanged, organized into functions.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap


def shap_analysis(rf, X_test, feature_names, sample_size=100, random_state=42):
    """SHAP summary plot for the Random Forest + top-5 feature explanations."""
    rng = np.random.default_rng(random_state)
    sample_indices = rng.choice(X_test.shape[0], sample_size, replace=False)
    X_sample = X_test[sample_indices]
    X_sample_df = pd.DataFrame(X_sample, columns=feature_names)

    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_sample)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    shap.summary_plot(shap_values, X_sample_df, feature_names=feature_names)

    importance = pd.DataFrame({"Feature": feature_names, "Importance": rf.feature_importances_})
    importance = importance.sort_values(by="Importance", ascending=False)
    print("\nDetailed explanations for top features:")
    for feature in importance["Feature"].head(5):
        print(f"\nFeature: {feature}")
        print(f"- High SHAP values for {feature} indicate a strong impact on the model's prediction.")
        print("- Direction of SHAP value tells whether it pushes the prediction towards default or not.")
    return importance


def plot_feature_importance(rf, feature_names):
    """Bar plot of Random Forest feature importances."""
    importances = rf.feature_importances_
    plt.figure(figsize=(15, 8))
    sns.barplot(x=importances, y=list(feature_names))
    plt.title('Feature Importance')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.show()
