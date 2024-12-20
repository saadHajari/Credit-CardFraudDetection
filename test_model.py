import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.utils import resample

# Helper function to encode categorical features
def encode_categorical(data, features):
    for feature in features:
        if data[feature].dtype == 'object':
            data[feature] = pd.factorize(data[feature])[0]
    return data

# Function to create test scenarios
def create_test_cases(data):
    test_cases = {}

    # 1. Balanced Data
    fraud = data[data['fraudulent'] == 1]
    non_fraud = data[data['fraudulent'] == 0]
    balanced_data = pd.concat([
        resample(fraud, replace=True, n_samples=len(non_fraud), random_state=42),
        non_fraud
    ])
    test_cases['Balanced Data'] = balanced_data

    # 2. Imbalanced Data
    imbalanced_data = pd.concat([
        fraud,
        non_fraud.sample(frac=0.1, random_state=42)
    ])
    test_cases['Imbalanced Data'] = imbalanced_data

    # 3. Transaction Amount Extremes
    extremes_data = data.copy()
    extremes_data.loc[0, 'amount'] = 0.01  # Extremely low
    extremes_data.loc[1, 'amount'] = 10000  # Extremely high
    test_cases['Transaction Amount Extremes'] = extremes_data

    # 4. Edge Hours for Transactions
    edge_hours_data = data.copy()
    edge_hours_data.loc[0, 'transaction_hour'] = 0  # Midnight
    edge_hours_data.loc[1, 'transaction_hour'] = 23  # End of day
    test_cases['Edge Hours for Transactions'] = edge_hours_data

    # 5. Rare Transaction Types
    rare_types_data = data.copy()
    rare_types_data.loc[0, 'transaction_type'] = 'rare_type_1'
    rare_types_data.loc[1, 'transaction_type'] = 'rare_type_2'
    test_cases['Rare Transaction Types'] = rare_types_data

    # 6. Missing or Corrupted Data
    missing_data = data.copy()
    missing_data.loc[0, 'amount'] = np.nan
    missing_data.loc[1, 'transaction_hour'] = -1
    test_cases['Missing or Corrupted Data'] = missing_data

    # 7. Synthetic Fraudulent Patterns
    synthetic_data = data.copy()
    synthetic_data.loc[0, ['amount', 'transaction_hour', 'fraudulent']] = [9999, 3, 1]
    test_cases['Synthetic Fraudulent Patterns'] = synthetic_data

    return test_cases

# Function to evaluate the model on test cases
def evaluate_test_cases(model, test_cases, features, target):
    results = []
    for case_name, case_data in test_cases.items():
        print(f"Testing {case_name}...")

        # Preprocess data
        case_data = encode_categorical(case_data, features)
        X = case_data[features]
        y = case_data[target]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the model
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # Evaluate metrics
        metrics = {
            "Test Case": case_name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1 Score": f1_score(y_test, y_pred, zero_division=0),
            "ROC AUC": roc_auc_score(y_test, y_prob)
        }

        results.append(metrics)

    return pd.DataFrame(results)

# Main script
if __name__ == "__main__":
    # Load dataset
    file_path = "enhanced_datasets_iteration_2/dataset_transaction_type.json"
    data = pd.read_json(file_path)

    # Define features and target
    features = ["amount", "transaction_hour", "transaction_type"]
    target = "fraudulent"

    # Create test cases
    test_cases = create_test_cases(data)

    # Initialize model
    model = RandomForestClassifier(random_state=42)

    # Evaluate test cases
    results_df = evaluate_test_cases(model, test_cases, features, target)

    # Save and display results
    results_df.to_csv("test_case_results.csv", index=False)
    print(results_df)
