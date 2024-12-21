# Fraud Detection with Synthetic Data and Machine Learning

This project focuses on detecting **fraudulent transactions** using **synthetic** **datasets** generated through **Python scripts**. The datasets are iteratively created to simulate various **fraud scenarios**, and two machine learning models, **Logistic Regression and Random Forest**, are trained to classify transactions as fraudulent or not.
![project_linkedin](https://github.com/user-attachments/assets/638d7f9f-6a6e-43bb-891a-16a8a7025269)

---

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Datasets](#datasets)
- [Models](#models)
- [Results](#results)
- [Technologies Used](#technologies-used)
- [Future Enhancements](#future-enhancements)
- [Contact](#contact)
- [License](#license)

---

### Overview
Fraud detection is a critical application of data science and machine learning. This project generates synthetic datasets representing different fraud scenarios, trains models on these datasets, and evaluates their performance. The aim is to provide a robust pipeline for detecting fraud in financial transactions.

---

### Features
- **Synthetic Data Generation**: Python scripts to generate datasets for various fraud detection scenarios.
- **Iterative Modeling**: Models are trained on two iterations of the datasets to assess consistency.
- **Machine Learning Models**: Implementation of Logistic Regression and Random Forest for classification.
- **Metrics Evaluation**: Evaluation of models using metrics like accuracy, precision, recall, F1-score, and ROC-AUC.
- **CSV Output**: Results are saved in a CSV file for easy analysis.

---

### Installation

## 1. Clone the repository:
   ```bash
   git clone https://github.com/saadHajari/Credit-CardFraudDetection.git
   cd Credit-CardFraudDetection
 ```

## 2. pip install -r requirements.txt

## 3. fraud-detection/
```bash
├── iteration1/
├── iteration2/
├── Comparison/
├── data_gen.py
├── data_gen2.py
├── model_train.py
├── README.md
└── requirements.txt
 ```

### Usage 

## Step 1: Generate Synthetic Datasets

```bash
python data_gen.py   # Generates datasets in iteration1/
 ```
```bash
python data_gen2.py  # Generates datasets in iteration2/
 ```

### Step 2: Train and Evaluate Models

```bash
python model_logistic.py #load datasets from iteration1/ and iteration2/ and train LogisticRegression and test the model
 ```
```bash
python model_random.py   #load datasets from iteration1/ and iteration2/ and train RandomForest and test the model 
 ```

The evaluation metrics will be saved in

**Comparison/model_comparison_results_logistic_regression.csv** 

**Comparison/random_forest_model_comparison_results.csv** 

### Datasets

Each dataset simulates a specific fraud scenario with features such as:

**Transaction Day:** Day of the transaction and velocity.

**Transaction Type:** Type and time of the transaction.

**Device Used:** Device type and location risk.

**Recurring Payment:** Frequency and velocity of payments.

**Customer Age Group:** Customer demographics and geo-mismatch.

**Previous Transaction:** Details of past transactions and unusual merchants

**Example** : 
```bash 
    {
        "amount":375.17,
        "transaction_hour":14,
        "fraudulent":1,
        "high_risk_country":1,
        "billing_country":"FR",
        "transaction_country":"US",
        "geo_mismatch":1,
        "velocity":2,
        "high_velocity":0,
        "merchant_code":"gambling",
        "unusual_merchant":1,
        "transaction_type":"mobile"
    },
```

### Models

**Logistic Regression**
A linear model suitable for binary classification tasks.

**Random Forest**
An ensemble model that combines decision trees for better accuracy and robustness.

### Results

The results are saved in **Comparison/nameofthemodel.csv** and include the following metrics:

**Accuracy**
**Precision**
**Recall**
**F1-score**
**ROC-AUC** 

Example of output : 

![image](https://github.com/user-attachments/assets/ab76ff9a-5e43-49b4-8528-5152d492bcb9)

### Technologies Used

**Python:** Core programming language.

**scikit-learn:** Machine learning library for model implementation.

**Pandas:** Data manipulation and analysis.

**JSON:** Dataset format.

**Git:** Version control.

### Future Enhancements

Add more fraud scenarios to datasets.✅

Experiment with additional machine learning models (e.g., Gradient Boosting, Neural Networks).✅

Implement hyperparameter tuning for better model performance.✅

Visualize results using dashboards.✅

### Contact 

**For Contact** : Send me an email in : **saadhajari10@gmail.com**

**You Can Buy Me a coffee Here** -----> **https://www.paypal.com/donate/?hosted_button_id=5URJR262Y77BQ**
