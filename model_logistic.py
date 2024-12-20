import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Paths to datasets and iterations
iterations = ["iteration_1", "iteration_2"]
datasets = {
    "dataset_transaction_type.json": "amount transaction_hour transaction_type fraudulent",
    "dataset_device_used.json": "amount device_used high_risk_country fraudulent",
    "dataset_recurring_payment.json": "amount recurring_payment velocity fraudulent",
    "dataset_age_group.json": "amount customer_age_group geo_mismatch fraudulent",
    "dataset_previous_transaction.json": "amount previous_transaction_amount unusual_merchant fraudulent",
    "dataset_transaction_day.json": "amount transaction_day high_velocity fraudulent",
}

# Helper function to encode categorical features
def encode_categorical(data, features):
    for feature in features:
        if data[feature].dtype == 'object':
            data[feature] = pd.factorize(data[feature])[0]
    return data

# Store results
results = []

# Iterate through each iteration folder and dataset
for iteration in iterations:
    print(f"Processing datasets in {iteration}...")
    for dataset_name, feature_string in datasets.items():
        print(f"Processing {dataset_name}...")

        # Load dataset
        file_path = os.path.join(iteration, dataset_name)
        if not os.path.exists(file_path):
            print(f"File {file_path} not found. Skipping...")
            continue

        data = pd.read_json(file_path)

        # Specify features and target
        features = feature_string.split()
        target = "fraudulent"
        X = data[features[:-1]]
        y = data[target]

        # Encode categorical features
        X = encode_categorical(X, X.columns)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train a Logistic Regression Model
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # Evaluate metrics
        metrics = {
            "iteration": iteration,
            "dataset": dataset_name,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_prob),
        }

        results.append(metrics)
        print(f"Finished processing {dataset_name} in {iteration}.")

# Save results to DataFrame and display
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="precision", ascending=False)
results_df.to_csv("./Comparison/model_comparison_results_logistic_regression.csv", index=False)
print("Results saved to 'model_comparison_results_logistic_regression.csv'.")
print(results_df)
