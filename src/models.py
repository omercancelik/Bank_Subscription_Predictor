# src/models.py
import optuna
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

def tune_catboost(X_train, y_train, X_test, y_test, scale_pos_weight):
    def objective(trial):
        params = {
            "iterations": trial.suggest_int("iterations", 100, 500),
            "learning_rate": trial.suggest_loguniform("learning_rate", 1e-3, 1e-1),
            "depth": trial.suggest_int("depth", 3, 10),
            "l2_leaf_reg": trial.suggest_int("l2_leaf_reg", 1, 10),
            "verbose": False,
            "class_weights": {0: 1, 1: scale_pos_weight}
        }
        model = CatBoostClassifier(**params)
        model.fit(X_train, y_train)
        preds = model.predict_proba(X_test)[:, 1]
        return roc_auc_score(y_test, preds)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)
    return CatBoostClassifier(**study.best_params, verbose=False).fit(X_train, y_train)

def tune_xgboost(X_train, y_train, X_test, y_test, scale_pos_weight):
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "learning_rate": trial.suggest_loguniform("learning_rate", 1e-3, 1e-1),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_uniform("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.5, 1.0),
            "scale_pos_weight": scale_pos_weight,
            "use_label_encoder": False,
            "eval_metric": "logloss",
        }
        model = XGBClassifier(**params)
        model.fit(X_train, y_train)
        preds = model.predict_proba(X_test)[:, 1]
        return roc_auc_score(y_test, preds)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)
    return XGBClassifier(**study.best_params, use_label_encoder=False, eval_metric="logloss").fit(X_train, y_train)

def tune_lightgbm(X_train, y_train, X_test, y_test, scale_pos_weight):
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "learning_rate": trial.suggest_loguniform("learning_rate", 1e-3, 1e-1),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "num_leaves": trial.suggest_int("num_leaves", 20, 100),
            "class_weight": {0: 1, 1: scale_pos_weight},
        }
        model = LGBMClassifier(**params)
        model.fit(X_train, y_train)
        preds = model.predict_proba(X_test)[:, 1]
        return roc_auc_score(y_test, preds)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)
    return LGBMClassifier(**study.best_params).fit(X_train, y_train)
