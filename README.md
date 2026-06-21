# Customer Churn Prediction Platform using Machine Learning
Customer Churn Prediction System using Machine Learning to identify customers likely to leave a telecom service. Built with Logistic Regression, Random Forest, and XGBoost, the project includes an interactive Streamlit dashboard for real-time predictions, risk analysis, and business insights.

## Overview

Coustomer churn is one of the most significant challenges faced by subscription-based businesses. Losing existing customers directly impacts revenue, profitablility, and long-term growth.

This project leverages Machine Learning techniques to predict whether a customer is likely to churn based on demographic information, service subscriptions, contract details, and billing behavior. The system enables businesses to identify high-risk customers proactively and implement targeted retention strategies.

## Problem Statement

Telecommunication companies face significant revenue loss due to customer attrition. Traditional retention strategies are often reactive and fail to identify at-risk customers before they leave.

This project aims to build an intelligent predictive system capable of:

  - Predicting customer churn probability
  - Identifying key churn drivers
  - Segmenting customers by risk level
  - Providing actionable business insights

## Dataset

Dataset used "Telco Customer Churn Dataset".

### Dataset Statistics

| Attribute | Value | 
| --- | --- |
| Records | 7,043 | 
| Features | 19 | 
| Target Variable | Churn | 
| Problem type | Binary Classification | 

 ## Tools and Technologies

| Category | Technologies | 
| --- | --- |
| Programming Language | Python | 
| Data Analysis | Pandas, NumPy | 
| Visualization | Matplotlib, Seaborn | 
| Machine Learning | Scikit-Learn | 
| Gradient Boosting | XGBoost | 
| Deployment | Streamlit | 
| Model Serialization | Joblib | 

## Machine Learning Models

### 1. Logistic Regression

#### Purpose
  - Used as a baseline classification model.
  - Predicts the probability of customer churn using a linear relationship between features and the   - target variable.
#### How it is used?
  - Customer data is preprocessed using the pipeline.
  - Logistic Regression predicts whether a customer is likely to churn or not.
  - Class balancing is applied using class_weight='balanced'.
  - Provides interpretable and fast predictions.
#### Advantages
  - Simple and easy to understand.
  - Fast training and prediction.
  - Works well as a benchmark model.
  - Generates churn probabilities directly.

### 2. Random Forest

#### Purpose
  - Ensemble learning algorithm that combines multiple decision trees.
  - Captures complex patterns and feature interactions.
#### How it is used?
  - Trained on preprocessed customer data.
  - Uses multiple decision trees to predict churn.
  - Feature importance analysis is performed using Random Forest.
  - Hyperparameter tuning is applied using GridSearchCV.
#### Advantages
  - Handles non-linear relationships effectively.
  - Reduces overfitting through ensemble learning.
  - Provides feature importance scores.
  - Delivers robust and stable predictions.

### 3.XGBoost

#### Purpose
  - Advanced gradient boosting algorithm.
  - Designed for high-performance predictive modeling.
#### How it is used?
  - Trained using optimized boosting techniques.
  - Uses scale_pos_weight to handle class imbalance.
  - Learns from previous prediction errors iteratively.
  - Produces highly accurate churn predictions.
#### Advantages
  - Excellent predictive performance.
  - Handles imbalanced datasets efficiently.
  - Captures complex customer behavior patterns.
  - Widely used in industry and machine learning competitions.

## Model Comparison

| Model | ROC-AUC | Precision | Recall | F1 Score | Accuracy | 
| --- | --- | --- | --- | --- |  --- |
| Logistic Regression | 0.76 | 0.51 | 0.79 | 0.62 | 0.75 |
| Random Forest | 0.75 | 0.54 | 0.74 | 0.62 | 0.76 |
| XGBoost | 0.76 | 0.53 | 0.76 | 0.62 | 0.75 |

- Compares Logistic Regression, Random Forest, and XGBoost models
- Uses key metrics like Accuracy, Precision, Recall, F1-score, and ROC-AUC
- Helps identify the best-performing model for churn prediction
- Supports model selection for deployment

### Best Performing Model

- XGBoost achieved the highest overall predictive performance with a ROC-AUC score of 0.76 and F1-score of 0.62.

## Dashboard Features

The project includes an interactive Streamlit dashboard with two primary modules.

### 🔮 Prediction Module/Dashboard
  - Customer information input form
  - Model selection
  - Real-time churn prediction
  - Churn probability score
  - Risk classification
  - Business recommendations

### 📈 Insights Module/Dashboard
  - Model performance comparison
  - About Models Used
  - Feature importance visualization
  - Precision–Recall–F1 Tradeoff
  - ROC Curve analysis
  - Business insights
  - Summary

Project dashboard : https://customer-churn-prediction-platform-using-ml.streamlit.app/


## How to run this project?

### Installation Prerequisites
  - Python 3.11
  - pip
  - Git
### Clone Repository
  - git clone https://github.com/yourusername/customer-churn-prediction.git
  - cd customer-churn-prediction
### Create Virtual Environment
  - python -m venv churn_env
#### Activate Environment
  - Windows
     - churn_env\Scripts\activate
  - Linux/Mac
     - source churn_env/bin/activate
### Install Dependencies
  - pip install -r requirements.txt
### Run Application
  - streamlit run churn_app.py
  - Application will open automatically at:
     - http://localhost:8501

## Business Impact

The developed system enables organizations to:

  - Identify high-risk customers early
  - Reduce customer attrition
  - Improve retention campaigns
  - Increase customer lifetime value
  - Support data-driven decision making

## Future Enhancements

- Cloud Deployment (AWS/Azure/GCP)
- Real-Time Prediction API
- CRM Integration
- Automated Retention Recommendations
- Deep Learning Models
- Customer Lifetime Value Prediction
- Real-Time Monitoring Dashboard

 ## Author

Pola Santhosh

Gmail : santhoshpola2003@gmail.com

Project dashboard : https://customer-churn-prediction-platform-using-ml.streamlit.app/


Machine Learning | Data Science | Artificial Intelligence
