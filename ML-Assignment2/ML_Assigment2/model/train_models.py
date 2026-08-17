"""
Train and evaluate 5 classification models on the Breast Cancer Wisconsin dataset.
Dataset source: UCI Machine Learning Repository
https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

Models:
    1. Logistic Regression
    2. Decision Tree Classifier
    3. K-Nearest Neighbor Classifier
    4. Gaussian Naive Bayes
    5. Random Forest (Ensemble)

Each model is evaluated using: Accuracy, AUC, Precision, Recall, F1, MCC
Trained models and scaler are saved as .pkl files for use in the Streamlit app.
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)


def load_and_prepare_data():
    """Load the Breast Cancer Wisconsin dataset and split into train/test."""
    data = load_breast_cancer()
    feature_names = [str(f) for f in data.feature_names]
    X = pd.DataFrame(data.data, columns=feature_names)
    y = pd.Series(data.target, name="target")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test, feature_names


def scale_features(X_train, X_test):
    """Standardize features using StandardScaler."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return scaler, X_train_scaled, X_test_scaled


def get_models():
    """Return a dictionary of model name -> model instance."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=10000, random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5, random_state=42
        ),
        "kNN": KNeighborsClassifier(n_neighbors=7),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(
            n_estimators=150, random_state=42
        ),
    }


def evaluate_model(model, X_test, y_test):
    """Compute all 6 evaluation metrics for a trained model."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_prob),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "MCC": matthews_corrcoef(y_test, y_pred),
    }
    return metrics, y_pred


def main():
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)

    # Load data
    print("Loading Breast Cancer Wisconsin dataset...")
    X_train, X_test, y_train, y_test, feature_names = load_and_prepare_data()
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"Number of features: {len(feature_names)}")

    # Save test data as CSV (features + target)
    test_df = pd.DataFrame(X_test.values, columns=feature_names)
    test_df["target"] = y_test.values
    test_csv_path = os.path.join(project_dir, "test_data.csv")
    test_df.to_csv(test_csv_path, index=False)
    print(f"Test data saved to {test_csv_path}")

    # Also save train data for reference
    train_df = pd.DataFrame(X_train.values, columns=feature_names)
    train_df["target"] = y_train.values
    train_csv_path = os.path.join(project_dir, "train_data.csv")
    train_df.to_csv(train_csv_path, index=False)
    print(f"Train data saved to {train_csv_path}")

    # Scale features
    scaler, X_train_scaled, X_test_scaled = scale_features(X_train, X_test)
    scaler_path = os.path.join(script_dir, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")

    # Train and evaluate each model
    models = get_models()
    model_filenames = {
        "Logistic Regression": "logistic_regression.pkl",
        "Decision Tree": "decision_tree.pkl",
        "kNN": "knn.pkl",
        "Naive Bayes": "naive_bayes.pkl",
        "Random Forest": "random_forest.pkl",
    }

    all_results = {}
    target_names = ["Malignant (0)", "Benign (1)"]

    for name, model in models.items():
        print(f"\n{'='*60}")
        print(f"Training {name}...")

        # Train
        model.fit(X_train_scaled, y_train)

        # Save model
        model_path = os.path.join(script_dir, model_filenames[name])
        joblib.dump(model, model_path)
        print(f"Model saved to {model_path}")

        # Evaluate
        metrics, y_pred = evaluate_model(model, X_test_scaled, y_test)
        all_results[name] = metrics

        # Print metrics
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}")

        # Print classification report
        print(f"\nClassification Report for {name}:")
        print(classification_report(y_test, y_pred, target_names=target_names))

        # Print confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"Confusion Matrix:\n{cm}")

    # Print comparison table
    print(f"\n{'='*60}")
    print("COMPARISON TABLE - All Models")
    print(f"{'='*60}")
    results_df = pd.DataFrame(all_results).T
    results_df = results_df.round(4)
    print(results_df.to_string())

    # Find best model for each metric
    print(f"\n{'='*60}")
    print("BEST MODEL PER METRIC")
    print(f"{'='*60}")
    for col in results_df.columns:
        best = results_df[col].idxmax()
        print(f"  {col}: {best} ({results_df.loc[best, col]:.4f})")

    # Overall winner (by average rank across metrics)
    ranks = results_df.rank(ascending=False)
    avg_ranks = ranks.mean(axis=1)
    winner = avg_ranks.idxmin()
    print(f"\nOverall Winner (by average rank): {winner}")

    # Save results to CSV
    results_path = os.path.join(project_dir, "model_results.csv")
    results_df.to_csv(results_path)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
