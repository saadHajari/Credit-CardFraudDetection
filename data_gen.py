import numpy as np
import pandas as pd



def base_data_generation(n_samples=1000):
    """Generate base transaction data."""
    np.random.seed(42)  # For reproducibility
    
    # Generate basic attributes for the transactions
    amount = np.random.uniform(1, 1000, n_samples)  # Transaction amount between 1 and 1000
    amount = np.round(amount, 2)
    transaction_hour = np.random.randint(0, 24, n_samples)  # Hour of the transaction
    fraudulent = np.random.choice([0, 1], n_samples, p=[0.95, 0.05])  # 5% fraud transactions
    high_risk_country = np.random.choice([0, 1], n_samples, p=[0.85, 0.15])  # 15% high-risk country
    
    # Combine into a dictionary
    data = {
        'amount': amount,
        'transaction_hour': transaction_hour,
        'fraudulent': fraudulent,
        'high_risk_country': high_risk_country
    }
    
    return data


def add_fraud_patterns(data):
    """Add fraud patterns to the base data."""
    # Add a pattern where high-value transactions (greater than 800) are more likely to be fraudulent
    data['fraudulent'] = np.where(data['amount'] > 800, 1, data['fraudulent'])
    
    # Add a pattern where transactions made during non-working hours (00:00 - 06:00) have a higher fraud probability
    data['fraudulent'] = np.where((data['transaction_hour'] < 6), 1, data['fraudulent'])
    
    # Introduce fraud for high-risk countries
    data['fraudulent'] = np.where(data['high_risk_country'] == 1, 1, data['fraudulent'])
    
    # Add some noise by flipping fraud label for a few random entries
    flip_indices = np.random.choice(len(data['fraudulent']), size=int(len(data) * 0.05), replace=False)
    data['fraudulent'][flip_indices] = 1 - data['fraudulent'][flip_indices]  # Flip fraud status for noise
    
    return data


# Function to add new features to the dataset
def add_transaction_type(data):
    n_samples = len(data['amount'])
    transaction_types = ['online', 'in-store', 'mobile']
    data['transaction_type'] = np.random.choice(transaction_types, n_samples)
    return data

def add_device_used(data):
    n_samples = len(data['amount'])
    devices = ['desktop', 'mobile', 'tablet']
    data['device_used'] = np.random.choice(devices, n_samples)
    return data

def add_recurring_payment(data):
    n_samples = len(data['amount'])
    data['recurring_payment'] = np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
    return data

def add_customer_age_group(data):
    n_samples = len(data['amount'])
    age_groups = ['18-25', '26-35', '36-50', '50+']
    data['customer_age_group'] = np.random.choice(age_groups, n_samples)
    return data

def add_previous_transaction_amount(data):
    n_samples = len(data['amount'])
    data['previous_transaction_amount'] = np.random.uniform(0, 1000, n_samples)
    return data

def add_transaction_day(data):
    n_samples = len(data['amount'])
    data['transaction_day'] = np.random.choice(['weekday', 'weekend'], n_samples, p=[0.7, 0.3])
    return data

# Enhanced Data Generation with Multiple Cases

def add_geolocation_inconsistencies(data):
    n_samples = len(data['amount'])
    data['billing_country'] = np.random.choice(['US', 'UK', 'CA', 'FR', 'IN'], n_samples)
    data['transaction_country'] = np.random.choice(['US', 'UK', 'CA', 'FR', 'IN', 'CN', 'RU'], n_samples)
    
    # Flag inconsistencies where the billing country and transaction country differ
    data['geo_mismatch'] = np.where(data['billing_country'] != data['transaction_country'], 1, 0)
    return data


def add_transaction_velocity(data):
    n_samples = len(data['amount'])
    data['velocity'] = np.random.randint(1, 10, n_samples)  # Number of transactions in a short time frame
    
    # Flag high velocity as suspicious
    data['high_velocity'] = np.where(data['velocity'] > 5, 1, 0)
    return data

def add_merchant_codes(data):
    n_samples = len(data['amount'])
    common_merchants = ['grocery', 'electronics', 'clothing', 'restaurant']
    unusual_merchants = ['luxury_goods', 'gambling', 'cryptocurrency', 'escort_service']
    
    # Assign a mix of common and unusual merchant codes
    all_merchants = common_merchants + unusual_merchants
    data['merchant_code'] = np.random.choice(all_merchants, n_samples)
    
    # Use np.isin to check for unusual merchant codes
    data['unusual_merchant'] = np.where(np.isin(data['merchant_code'], unusual_merchants), 1, 0)
    return data

# Example usage
def generate_enhanced_datasets(output_folder="iteration_1"):
    import os
    os.makedirs(output_folder, exist_ok=True)

    # Generate base data
    base_data = base_data_generation(n_samples=1000)
    
    # Add fraud patterns to the base data
    fraud_data = add_fraud_patterns(base_data)
    fraud_data = add_geolocation_inconsistencies(fraud_data)
    fraud_data = add_transaction_velocity(fraud_data)
    fraud_data = add_merchant_codes(fraud_data)
    
    # Continue with adding new features and saving datasets (as previously defined)
    cases = [
        (add_transaction_type, "dataset_transaction_type.json"),
        (add_device_used, "dataset_device_used.json"),
        (add_recurring_payment, "dataset_recurring_payment.json"),
        (add_customer_age_group, "dataset_age_group.json"),
        (add_previous_transaction_amount, "dataset_previous_transaction.json"),
        (add_transaction_day, "dataset_transaction_day.json"),
    ]

    for feature_fn, file_name in cases:
        modified_data = feature_fn(fraud_data.copy())
        df = pd.DataFrame(modified_data)
        df.to_json(f"{output_folder}/{file_name}", orient="records", lines=False, indent=4)

        print(f"Generated {file_name}")

generate_enhanced_datasets()

# Call the data generation function

# # Define the feature sets for testing
# test_cases = [
#     {
#         "file": "enhanced_datasets/dataset_transaction_type.json",
#         "features": ['amount', 'transaction_hour', 'transaction_type', 'fraudulent']
#     },
#     {
#         "file": "enhanced_datasets/dataset_device_used.json",
#         "features": ['amount', 'device_used', 'high_risk_country', 'fraudulent']
#     },
#     {
#         "file": "enhanced_datasets/dataset_recurring_payment.json",
#         "features": ['amount', 'recurring_payment', 'distance_from_home', 'fraudulent']
#     },
#     {
#         "file": "enhanced_datasets/dataset_age_group.json",
#         "features": ['amount', 'customer_age_group', 'average_monthly_spending', 'fraudulent']
#     },
#     {
#         "file": "enhanced_datasets/dataset_previous_transaction.json",
#         "features": ['amount', 'previous_transaction_amount', 'high_risk_country', 'fraudulent']
#     },
#     {
#         "file": "enhanced_datasets/dataset_transaction_day.json",
#         "features": ['amount', 'transaction_day', 'transaction_hour', 'fraudulent']
#     },
# ]

# # Train and test each case
# for case in test_cases:
#     train_and_test_model(case["file"], case["features"])