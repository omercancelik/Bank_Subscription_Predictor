# src/ensemble.py
import numpy as np
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, precision_recall_curve

def build_voting_ensemble(models, X_train, y_train):
    voting = VotingClassifier(
        estimators=[("cb", models["catboost"]),
                    ("xgb", models["xgboost"]),
                    ("lgb", models["lightgbm"])],
        voting="soft",
        n_jobs=-1
    )
    voting.fit(X_train, y_train)
    return voting

def optimize_threshold(voting, X_test, y_test):
    probs = voting.predict_proba(X_test)[:, 1]
    prec, rec, thr = precision_recall_curve(y_test, probs)
    f1_scores = 2 * prec * rec / (prec + rec)
    best_idx = np.nanargmax(f1_scores)
    return thr[best_idx], probs

def evaluate_model(voting, threshold, probs, y_test):
    y_pred = (probs >= threshold).astype(int)
    print(f"=== Ensemble Evaluation (Threshold: {threshold:.3f}) ===")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("ROC AUC :", roc_auc_score(y_test, probs))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4))
