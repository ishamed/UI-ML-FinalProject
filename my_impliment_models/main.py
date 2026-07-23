import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from cleaner import perform_eda, split_data, impute_missing_value, scale_features

try:
    from adaBoost import get_model as get_adaboost
    from bayesNaiveGaussian import get_model as get_nb
    from gradientBoosting import get_model as get_gb
    from MLP import get_model as get_mlp
    from randomForest import get_model as get_rf
except ImportError as e:
    print(f"Warning: Failed to import some models. Ensure all .py files are in the directory.\nError: {e}")


def load_data(file_path="water_potability.csv"):
    """بارگذاری مجموعه داده"""
    try:
        df = pd.read_csv(file_path)
        print(f"Dataset loaded successfully with shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: {file_path} not found. Generating a dummy dataset for testing purposes...")
        np.random.seed(42)
        X_dummy = np.random.randn(200, 5)
        y_dummy = np.random.randint(0, 2, 200)
        df = pd.DataFrame(X_dummy, columns=['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate'])
        df['Potability'] = y_dummy
        df.loc[10:20, 'ph'] = np.nan
        return df


def main():
    print("=== Starting ML Pipeline for From-Scratch Models ===\n")

    # ۱. بارگذاری داده‌ها
    df = load_data("water_potability.csv")

    # ۲. تحلیل اکتشافی داده‌ها (EDA)
    # print("Performing EDA...")
    # perform_eda(df)

    # ۳. تقسیم داده‌ها
    target_col = "Potability"
    if target_col not in df.columns:
        print(f"Target column '{target_col}' not found in dataset!")
        return

    print("Splitting data...")
    X_train, X_test, y_train, y_test = split_data(df, target_col=target_col, test_size=0.2)

    print("Imputing missing values using KNNImputer...")
    X_train, X_test = impute_missing_value(X_train, X_test, n_neighbors=5)

    print("Scaling features using StandardScaler...")
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
        "Random Forest": get_rf()
    }

    print("\n=== Training and Evaluation ===")
    results = []

    for name, model in models.items():
        print(f"\n-> Training {name}...")
        start_time = time.time()

        try:
            # آموزش مدل
            model.fit(X_train_np, y_train_np)

            # پیش‌بینی
            y_pred = model.predict(X_test_np)

            # محاسبه زمان و دقت
            elapsed_time = time.time() - start_time
            acc = accuracy_score(y_test_np, y_pred)

            print(f"{name} trained in {elapsed_time:.2f} seconds.")
            print(f"Accuracy: {acc:.4f}")
            print(classification_report(y_test_np, y_pred, zero_division=0))

            # --- رسم و ذخیره ماتریس درهم‌ریختگی با Seaborn ---
            plt.figure(figsize=(6, 5))
            cm = confusion_matrix(y_test_np, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                        xticklabels=['Predicted 0', 'Predicted 1'],
                        yticklabels=['Actual 0', 'Actual 1'])
            plt.title(f'Confusion Matrix - {name}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()

            # ذخیره تصویر باکیفیت برای استفاده در گزارش LaTeX
            file_name = f"confusion_matrix_{name.replace(' ', '_').lower()}.png"
            plt.savefig(file_name, dpi=300)
            plt.close()
            # -----------------------------------------------

            results.append({"Model": name, "Accuracy": acc, "Time (s)": elapsed_time})

        except Exception as e:
            print(f"An error occurred while training/testing {name}: {e}")

    if results:
        print("\n=== Final Results Summary ===")
        results_df = pd.DataFrame(results).sort_values(by="Accuracy", ascending=False)
        print(results_df.to_string(index=False))


if __name__ == '__main__':
    main()