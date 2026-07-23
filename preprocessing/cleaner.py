import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
import seaborn as sns

def perform_eda(df):
    df.hist(figsize=(15, 12), bins=30, edgecolor='black', color='skyblue')
    plt.suptitle("Histogram", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    plt.figure(figsize=(15, 10))
    sns.boxplot(data=df.drop("Potability", axis=1), orient="h", palette="Set2")
    plt.title("Boxplot", fontsize=16)
    plt.show()

    plt.figure(figsize=(10,10))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix")
    plt.show()


def split_data(df, target_col, test_size):
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    y = y.reset_index(drop=True)

    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)

def impute_missing_value(X_train,X_test, n_neighbors=5):
    imputer = KNNImputer(n_neighbors=n_neighbors)
   
    X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
    X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

    return X_train_imputed, X_test_imputed


def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    return X_train_scaled, X_test_scaled, scaler



