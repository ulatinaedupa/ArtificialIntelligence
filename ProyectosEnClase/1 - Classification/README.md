# Project 1: Customer Churn Classification

## Overview
A comprehensive classification project that predicts customer churn using machine learning. This project includes a complete pipeline from data analysis to model deployment with a Streamlit web application.

## Project Structure
```
1 - Classification/
├── Customer Churn [Classification Project].ipynb  # Main analysis notebook
├── classificationdata.csv                          # Customer dataset
├── app/                                            # Streamlit web application
│   ├── app.py                                      # Main application file
│   ├── utils.py                                    # Utility functions
│   └── schema.json                                 # Data schema
├── models/                                         # Trained ML models
├── imgs/                                           # Visualizations and plots
└── requirements.txt                                # Python dependencies
```

## Learning Objectives
- Binary classification problem-solving
- Feature engineering for customer data
- Model training and evaluation
- Deployment with Streamlit
- Creating interactive ML applications

## Getting Started

### Prerequisites
```bash
pip install -r requirements.txt
```

### How to Run

**1. Jupyter Notebook Analysis:**
```bash
jupyter notebook "Customer Churn [Classification Project].ipynb"
```

**2. Streamlit Web Application:**
```bash
cd app
streamlit run app.py
```

## Key Features
- **Data Analysis**: Comprehensive EDA on customer behavior
- **Classification Models**: Multiple algorithms compared
- **Web Interface**: Interactive prediction tool
- **Model Persistence**: Saved models for deployment
- **Visualization**: Customer insights and model performance metrics

## Models Used
- Logistic Regression
- Decision Trees
- Random Forest
- Gradient Boosting
- (And other classification algorithms)

## Dataset
Customer churn dataset with features including:
- Customer demographics
- Account information
- Service usage patterns
- Churn labels (target variable)

## Business Value
Predicting customer churn helps businesses:
- Identify at-risk customers
- Implement retention strategies
- Reduce customer acquisition costs
- Improve customer satisfaction
