import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Load the dataset from JSON
def load_data(filename='augmented_fraud_dataset.json'):
    """
    Load the dataset from a JSON file.
    """
    return pd.read_json(filename, lines=True)

# Step 2: Preprocess the data
def preprocess_data(df):
    """
    Preprocess the data by scaling the features and splitting into training/testing sets.
    """
    # Separate features and target
    X = df.drop(['label', 'predicted_fraud'], axis=1)
    y = df['label']

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, scaler

# Step 3: Enhanced Data Augmentation
def augment_data(df, fraud_cases=100, noise_level=0.1):
    """
    Enhance the dataset by adding noise and generating additional synthetic fraud cases with more variability.
    """
    # Add noise to continuous features
    for feature in ['amount', 'distance_from_home', 'average_monthly_spending']:
        noise = np.random.normal(0, noise_level * df[feature].std(), df[feature].shape)
        df[feature] = np.clip(df[feature] + noise, a_min=0, a_max=None)  # Ensure no negative values

    # Generate diverse synthetic fraud cases
    fraud_data = {
        'amount': np.random.uniform(5000, 20000, fraud_cases),  # Wider range for fraud amounts
        'distance_from_home': np.random.uniform(1000, 10000, fraud_cases),  # Longer distances for fraud
        'high_risk_country': np.random.choice([0, 1], fraud_cases, p=[0.3, 0.7]),  # More weight to high-risk countries
        'average_monthly_spending': np.random.uniform(2000, 10000, fraud_cases),
        'transaction_hour': np.random.choice(range(24), fraud_cases),  # Random distribution across all hours
        'label': [1] * fraud_cases,
        'predicted_fraud': [1] * fraud_cases
    }
    fraud_df = pd.DataFrame(fraud_data)

    # Concatenate the original data with synthetic fraud cases
    augmented_df = pd.concat([df, fraud_df], ignore_index=True)

    # Shuffle the dataset to mix synthetic and real data
    augmented_df = augmented_df.sample(frac=1, random_state=42).reset_index(drop=True)

    return augmented_df

# Step 4: Resample the data using SMOTE
def resample_data(X_train, y_train):
    """
    Apply SMOTE to handle class imbalance in the training data.
    """
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    return X_resampled, y_resampled

# Step 5: Train the model
def train_model(X_train, y_train):
    """
    Train a RandomForestClassifier with class weights to handle class imbalance.
    """
    model = RandomForestClassifier(n_estimators=100, class_weight={0: 1, 1: 10}, random_state=42)
    model.fit(X_train, y_train)
    return model

# Step 6: Evaluate the model
def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model using classification report and confusion matrix.
    """
    y_pred = model.predict(X_test)

    # Print classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Normal", "Fraud"], yticklabels=["Normal", "Fraud"])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# Step 7: Save the model and scaler
def save_model(model, scaler, model_filename='fraud_detection_model.pkl', scaler_filename='scaler.pkl'):
    """
    Save the trained model and scaler to disk.
    """
    import joblib
    joblib.dump(model, model_filename)
    joblib.dump(scaler, scaler_filename)
    print(f"Model saved to {model_filename} and scaler saved to {scaler_filename}")

# Main function
def main():
    # Load the dataset
    df = load_data('augmented_fraud_dataset.json')

    # Augment the dataset
    df = augment_data(df, fraud_cases=200, noise_level=0.15)  # Increased fraud cases and noise level

    # Preprocess the data
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)

    # Resample the training data
    X_resampled, y_resampled = resample_data(X_train, y_train)

    # Train the model
    model = train_model(X_resampled, y_resampled)

    # Evaluate the model
    evaluate_model(model, X_test, y_test)

    # Save the model and scaler
    save_model(model, scaler)

if __name__ == "__main__":
    main()