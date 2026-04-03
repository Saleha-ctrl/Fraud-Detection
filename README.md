# Fraud Detection System

## Overview
This project is a Machine Learning-based Fraud Detection System that identifies potentially fraudulent transactions using historical data patterns. The model classifies transactions as either **fraudulent or legitimate**.

## Tech Stack
- Python
- Pandas
- Scikit-learn

## Problem Statement
Financial fraud causes major losses in digital transactions. The goal is to build a model that can accurately detect fraudulent activities based on transaction behavior.

## Workflow
1. Load dataset
2. Data preprocessing
3. Feature scaling using StandardScaler
4. Train ML models (KNN / SVM / Logistic Regression)
5. Select best model
6. Save model using pickle
7. Test predictions

## Approach
- Data cleaning and preprocessing
- Handling imbalanced dataset
- Feature selection
- Model training (Logistic Regression / Random Forest / etc.)
- Evaluation using accuracy, precision, recall, F1-score

## Model Files
- scaler.pkl → Used for feature scaling
- fraud_model.pkl → Trained ML model saved using joblib/pickle

## Results
- High accuracy achieved on test data
- Strong performance on detecting fraud cases (recall-focused)

## How to Run
```bash
python code file.py
