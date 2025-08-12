# src/visualization.py
import matplotlib.pyplot as plt
import seaborn as sns

def cat_plot(dataframe, col):
    plt.figure(figsize=(12, 6))
    sns.countplot(data=dataframe, x=col, hue='y', order=dataframe[col].value_counts().index)
    plt.title(f'Subscription Outcome by {col}')
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.legend(title="Subscribed")
    plt.tight_layout()
    plt.show()

def num_plot(dataframe, col):
    plt.figure(figsize=(8, 5))
    sns.histplot(dataframe[col], kde=True, bins=20)
    plt.title(f'{col} Distribution')
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()
