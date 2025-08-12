# src/data_preprocessing.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def grab_col_names(dataframe, cat_th=10, car_th=20):
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]
    return cat_cols, num_cols, cat_but_car

def load_and_clean_data(path):
    df = pd.read_csv(path, delimiter=';')
    df.drop(["day", "default", "education", "job", "marital", "loan"], axis=1, inplace=True)

    cat_cols, num_cols, _ = grab_col_names(df)

    for col in cat_cols:
        df[col] = df[col].apply(lambda x: x.replace('unknown', 'others'))

    le = LabelEncoder()
    df = df.apply(le.fit_transform)

    y = df["y"]
    X = df.drop(["y"], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    return X_train, X_test, y_train, y_test, X.columns
