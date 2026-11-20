"""Explainability: SHAP (tree + gradient), LIME, per-test contribution."""
import numpy as np
import shap
import lime.lime_tabular


def shap_tree(model, X_flat, feature_names, plot=True):
    ex = shap.TreeExplainer(model)
    sv = ex.shap_values(X_flat)
    if plot:
        shap.summary_plot(sv, X_flat, feature_names=feature_names, show=False)
    return sv


def shap_deep(model, X_seq, background, plot=True):
    ex = shap.GradientExplainer(model, background)
    sv = ex.shap_values(X_seq)
    if isinstance(sv, list):
        sv = sv[0]
    return sv


def per_test_contribution(shap_vals):
    return np.abs(shap_vals).mean(axis=2)


def lime_explain(predict_fn, X_train, feature_names, instance, num_features=10):
    expl = lime.lime_tabular.LimeTabularExplainer(
        X_train, feature_names=feature_names, mode="regression", discretize_continuous=True)
    exp = expl.explain_instance(instance, predict_fn, num_features=num_features)
    return exp


def permutation_importance(model, X, y, feature_names, n_repeats=5):
    from sklearn.inspection import permutation_importance as pi
    r = pi(model, X, y, n_repeats=n_repeats, random_state=42)
    order = np.argsort(r.importances_mean)[::-1]
    return [(feature_names[i], float(r.importances_mean[i])) for i in order]