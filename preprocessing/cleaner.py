import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
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

