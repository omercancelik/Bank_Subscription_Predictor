# README.md

# Bank Subscription Predictor

This project predicts whether a customer will subscribe to a term deposit using an ensemble of machine learning models.

## Features
- Data cleaning and label encoding
- Visual analysis (cat and num distributions)
- Hyperparameter tuning with Optuna
- Feature selection with model importance
- Ensemble learning with CatBoost, XGBoost, and LightGBM
- Threshold optimization using F1 score

## Usage
1. Clone the repo and install dependencies:
```bash
pip install -r requirements.txt
```

2. Add the dataset to `data/bank-full.csv`

3. Run the pipeline:
```bash
python main.py
```

## Project Structure
```
bank-subscription-predictor/
├── data/
│   └── bank-full.csv
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── visualization.py
│   ├── models.py
│   └── ensemble.py
├── main.py
├── README.md
├── requirements.txt
└── .gitignore
```

