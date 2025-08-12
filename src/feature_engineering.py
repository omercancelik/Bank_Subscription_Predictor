# src/feature_engineering.py
import numpy as np
import pandas as pd

def select_important_features(model, X_train, threshold_percentile=20):
    importances = pd.Series(model.get_feature_importance(), index=X_train.columns)
    threshold = np.percentile(importances, threshold_percentile)
    selected_features = importances[importances > threshold].index
    return selected_features
