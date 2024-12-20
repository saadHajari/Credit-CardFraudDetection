import pandas as pd
import numpy as np
from datetime import datetime
import random
from sklearn.preprocessing import StandardScaler

# Step 1: Generate Synthetic Dataset
def generate_synthetic_data(num_samples=1000):
    """
    Generate a synthetic dataset with random values for fraud detection.
    """
    data = {
        'amount': np.random.uniform(10, 10000, num_samples),
        'distance_from_home': np.random.uniform(1, 2000, num_samples),
        'high_risk_country': np.random.choice([0, 1], num_samples, p=[0.8, 0.2]),
        'average_monthly_spending': np.random.uniform(100, 5000, num_samples),
        'transaction_hour': np.random.choice(range(24), num_samples),
        'label': np.random.choice([0, 1], num_samples, p=[0.95, 0.05])  # 95% normal, 5% fraud
    }
    return pd.DataFrame(data)

# Step 2: Data Augmentation Functions
def add_noise_to_feature(data, feature, noise_level=0.1):
    """
    Add noise to a feature to augment the dataset. The noise is normally distributed.
    """
    noisy_data = data.copy()
    noisy_data[feature] = noisy_data[feature] + np.random.normal(0, noise_level * noisy_data[feature].std(), noisy_data[feature].shape)
    return noisy_data

def generate_fraudulent_cases(num_cases):
    """
    Generate fraudulent cases based on high risk parameters (e.g., high amount, high distance, high risk country).
    """
    fraud_data = {
        'amount': np.random.uniform(5000, 10000, num_cases),
        'distance_from_home': np.random.uniform(1000, 5000, num_cases),
        'high_risk_country': [1] * num_cases,  # Fraudulent cases come from high-risk countries
        'average_monthly_spending': np.random.uniform(1000, 5000, num_cases),
        'transaction_hour': np.random.choice([0, 1, 2, 3, 4], num_cases),  # Transactions at night
        'label': [1] * num_cases  # Label 1 indicates fraud
    }
    return pd.DataFrame(fraud_data)

# Step 3: Apply Business Rules for Fraud Detection
def apply_business_rules(row):
    """
    Apply business rules to determine if a transaction is potentially fraudulent.
    """
    risk_score = 0
    if row['distance_from_home'] > 1000:
        risk_score += 2
    if row['transaction_hour'] < 6:  # Fraudulent transactions often occur at night
        risk_score += 2
    if row['amount'] > 5000:
        risk_score += 3
    if row['high_risk_country'] == 1:
        risk_score += 3
    return risk_score

def label_fraud_based_on_rules(df):
    """
    Apply business rules to the dataset and label transactions with a high risk score as fraudulent.
    """
    df['risk_score'] = df.apply(apply_business_rules, axis=1)
    df['predicted_fraud'] = df['risk_score'].apply(lambda x: 1 if x >= 5 else 0)
    return df

# Step 4: Create and Augment the Dataset
def create_and_augment_dataset(num_samples=1000, num_fraud_cases=100, noise_level=0.1):
    """
    Create a dataset and apply data augmentation.
    """
    # Generate initial dataset
    df = generate_synthetic_data(num_samples)
    
    # Add noise to some features for data augmentation
    df = add_noise_to_feature(df, 'amount', noise_level)
    df = add_noise_to_feature(df, 'distance_from_home', noise_level)
    
    # Generate synthetic fraud cases and add them to the dataset
    fraud_cases = generate_fraudulent_cases(num_fraud_cases)
    df = pd.concat([df, fraud_cases], ignore_index=True)
    
    # Apply business rules to label fraud and normal transactions
    df = label_fraud_based_on_rules(df)
    
    # Round the 'amount' column to 4 decimal places
    df['amount'] = df['amount'].round(4)
    
    # Drop the temporary 'risk_score' column
    df = df.drop(columns=['risk_score'])
    
    return df

# Step 5: Save Dataset to JSON
def save_to_json(df, filename='augmented_fraud_dataset.json'):
    """
    Save the augmented dataset to a JSON file.
    """
    df.to_json(filename, orient='records', lines=True)
    print(f"Dataset saved to {filename}")

# Step 6: Main Function to Run All Steps
def main():
    # Create and augment the dataset
    augmented_df = create_and_augment_dataset(num_samples=1000, num_fraud_cases=100, noise_level=0.1)
    
    # Save the augmented dataset to a JSON file
    save_to_json(augmented_df, 'augmented_fraud_dataset.json')

    # Display the first few rows of the dataset
    print(augmented_df.head())

if __name__ == "__main__":
    main()
