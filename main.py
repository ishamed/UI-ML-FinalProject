import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from preprocessing.cleaner import split_data, impute_missing_value, scale_features

try:
    from my_impliment_models.adaBoost import get_model as get_adaboost
    from my_impliment_models.bayesNaiveGaussian import get_model as get_nb
    from my_impliment_models.gradientBoosting import get_model as get_gb
    from my_impliment_models.MLP import get_model as get_mlp
    from my_impliment_models.randomForest import get_model as get_rf

    from models.logistic_regression import get_model as get_lr
    from models.knn import get_model as get_knn
    from models.svm import get_model as get_svm
    from models.decision_tree import get_model as get_dt
    from models.xgboost import get_model as get_xgb
except ImportError as e:
    print(f"Import Error: {e}")


def load_data(file_path="water_potability.csv"):
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        np.random.seed(42)
        X_dummy = np.random.randn(200, 5)
        y_dummy = np.random.randint(0, 2, 200)
        df = pd.DataFrame(X_dummy, columns=['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate'])
        df['Potability'] = y_dummy
        df.loc[10:20, 'ph'] = np.nan
        return df


def main():
    df = load_data("water_potability.csv")

    target_col = "Potability"
    if target_col not in df.columns:
        print(f"Target column '{target_col}' not found in dataset!")
        return

    X_train, X_test, y_train, y_test = split_data(df, target_col=target_col, test_size=0.2)
    X_train, X_test = impute_missing_value(X_train, X_test, n_neighbors=5)
    X_train, X_test, scaler = scale_features(X_train, X_test)

    X_train_np = X_train.values
    X_test_np = X_test.values
    y_train_np = y_train.values
    y_test_np = y_test.values

    models = {
        "AdaBoost": get_adaboost(),
        "Gaussian Naive Bayes": get_nb(),
        "Gradient Boosting": get_gb(),
        "Multi-Layer Perceptron": get_mlp(),
        "Random Forest": get_rf(),
        "Logistic Regression": get_lr(),
        "K-Nearest Neighbors": get_knn(),
        "Support Vector Machine": get_svm(),
        "Decision Tree": get_dt(),
        "XGBoost": get_xgb()
    }

    results = []

    for name, model in models.items():
        print(f"-> Training and Evaluating {name}...")
        start_time = time.time()
        try:
            model.fit(X_train_np, y_train_np)
            y_pred = model.predict(X_test_np)

            elapsed_time = time.time() - start_time

            # محاسبه شاخص‌های ارزیابی
            acc = accuracy_score(y_test_np, y_pred)
            prec = precision_score(y_test_np, y_pred, zero_division=0)
            rec = recall_score(y_test_np, y_pred, zero_division=0)
            f1 = f1_score(y_test_np, y_pred, zero_division=0)

            plt.figure(figsize=(6, 5))
            cm = confusion_matrix(y_test_np, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                        xticklabels=['Predicted 0', 'Predicted 1'],
                        yticklabels=['Actual 0', 'Actual 1'])
            plt.title(f'Confusion Matrix - {name}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()

            file_name = f"confusion_matrix_{name.replace(' ', '_').lower()}.png"
            plt.savefig(file_name, dpi=300)
            plt.close()

            results.append({
                "Model": name,
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1-Score": f1,
                "Time (s)": elapsed_time
            })

        except Exception as e:
            print(f"Error in {name}: {e}")

    if results:
        print("\n" + "=" * 60)
        print("FINAL RESULTS SUMMARY (Sorted by Accuracy)")
        print("=" * 60)
        results_df = pd.DataFrame(results).sort_values(by="Accuracy", ascending=False)

        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        pd.set_option('display.float_format', '{:.4f}'.format)

        print(results_df.to_string(index=False))


if __name__ == '__main__':
    main()