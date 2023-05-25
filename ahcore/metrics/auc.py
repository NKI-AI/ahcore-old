""" AUC-based metrics """

import numpy as np
from sklearn import metrics


def standardized_auc(fpr: np.ndarray, tpr: np.ndarray, **kwargs):
    """

    Parameters
    ----------
    fpr: np.ndarray, FP ratios
    tpr: np.ndarray, TP ratios

    Returns
    -------
    standardized AUC; for values below 0.5, 1 - AUC will be returned, so that the result is always >= 0.5
    """
    auc = metrics.auc(fpr, tpr)
    if auc < 0.5:
        auc = 1.0 - auc  # equivalently, return the AUC for the opposing class
    return auc


def robustness_auc(X: np.ndarray, y: np.ndarray, **kwargs) -> np.ndarray:
    """calculate the robustness AUC; i.e. the AUC resulting from using feature values as category predictors.
    Args:
        X: array-like or sparse matrix, shape (n_samples, n_features), feature matrix
        y: array-like of shape (n_samples,), Target vector, has to be discrete
        **kwargs: to be passed to mutual_info_classif
    Returns: auc, ndarray, shape (n_features,) for each combination of feature and the target category.
    """

    nr_features = X.shape[1]
    auc_per_feature = []
    for feature_index in range(nr_features):
        fpr, tpr, thresholds = metrics.roc_curve(y, X[:, feature_index], pos_label=True)
        auc = standardized_auc(fpr, tpr, **kwargs)
        auc_per_feature.append(auc)
    return np.array(auc_per_feature)
