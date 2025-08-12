# main.py
from src.data_preprocessing import load_and_clean_data
from src.models import tune_catboost, tune_xgboost, tune_lightgbm
from src.feature_engineering import select_important_features
from src.ensemble import build_voting_ensemble, optimize_threshold, evaluate_model


def main():
    X_train, X_test, y_train, y_test, feature_names = load_and_clean_data("data/bank-full.csv")

    scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()

    # Train models
    print("Tuning CatBoost...")
    cb_model = tune_catboost(X_train, y_train, X_test, y_test, scale_pos_weight)
    print("Tuning XGBoost...")
    xgb_model = tune_xgboost(X_train, y_train, X_test, y_test, scale_pos_weight)
    print("Tuning LightGBM...")
    lgb_model = tune_lightgbm(X_train, y_train, X_test, y_test, scale_pos_weight)

    # Feature selection
    selected_features = select_important_features(cb_model, X_train)
    X_train_sel = X_train[selected_features]
    X_test_sel = X_test[selected_features]

    # Ensemble and evaluation
    models = {"catboost": cb_model, "xgboost": xgb_model, "lightgbm": lgb_model}
    voting = build_voting_ensemble(models, X_train_sel, y_train)
    threshold, probs = optimize_threshold(voting, X_test_sel, y_test)
    evaluate_model(voting, threshold, probs, y_test)


if __name__ == "__main__":
    main()
