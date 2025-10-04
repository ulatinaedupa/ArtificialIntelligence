# Lab 1: Classification - Customer Churn Prediction

## Overview
Comprehensive classification laboratory where you'll predict customer churn using various machine learning algorithms. This lab includes model training, deployment, and a complete web application.

## Project Structure
```
Lab1/
├── Laboratorio 3 - Clasificacion.ipynb           # Main lab assignment
├── Predicting Customer Churn - Example.ipynb     # Reference solution
├── classificationdata.csv                         # Customer dataset
├── app/                                           # Streamlit application
│   ├── app.py                                     # Main app file
│   ├── utils.py                                   # Helper functions
│   └── schema.json                                # Data schema
├── models/                                        # Trained models
├── doc/                                           # Documentation
└── img/                                           # Visualizations
```

## Learning Objectives
- Binary classification problem solving
- Feature engineering techniques
- Model comparison and selection
- Hyperparameter tuning
- Model evaluation metrics
- Deployment with Streamlit
- Creating production-ready ML apps

## Getting Started

### Prerequisites
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
pip install streamlit joblib
```

### How to Run

**1. Complete the Lab Assignment:**
```bash
jupyter notebook "Laboratorio 3 - Clasificacion.ipynb"
```

**2. Review the Example Solution:**
```bash
jupyter notebook "Predicting Customer Churn - Example.ipynb"
```

**3. Run the Web Application:**
```bash
cd app
streamlit run app.py
```

## Lab Tasks

### Part 1: Data Exploration
- Load and inspect customer data
- Analyze feature distributions
- Identify missing values
- Explore target variable balance

### Part 2: Data Preprocessing
- Handle missing values
- Encode categorical variables
- Feature scaling/normalization
- Train/test split

### Part 3: Model Training
Train and compare multiple classifiers:
- Logistic Regression
- Decision Trees
- Random Forest
- Gradient Boosting
- Support Vector Machines

### Part 4: Model Evaluation
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix analysis
- ROC-AUC curves
- Feature importance

### Part 5: Hyperparameter Tuning
- Grid Search
- Random Search
- Cross-validation

### Part 6: Deployment
- Save trained models
- Build Streamlit interface
- Create prediction API
- Test deployment

## Dataset
**Customer Churn Dataset** features include:
- **Demographics**: Age, gender, location
- **Account Info**: Tenure, contract type
- **Services**: Phone, internet, streaming
- **Billing**: Monthly charges, total charges
- **Target**: Churn (Yes/No)

## Evaluation Metrics
Focus on:
- **Accuracy**: Overall correctness
- **Precision**: Avoiding false positives
- **Recall**: Catching actual churners
- **F1-Score**: Balance between precision/recall
- **ROC-AUC**: Model discrimination ability

## Web Application Features
- Input customer information
- Real-time churn prediction
- Probability scores
- Feature importance visualization
- Model performance dashboard

## Deliverables
1. Completed Jupyter notebook with analysis
2. Trained and saved models
3. Working Streamlit application
4. Performance comparison report
5. Documentation of approach

## Business Context
**Why Customer Churn Matters:**
- Acquiring new customers is 5-25x more expensive than retaining existing ones
- Predicting churn enables proactive retention strategies
- Improves customer lifetime value
- Optimizes marketing spend

## Tips for Success
- Start with exploratory data analysis
- Handle class imbalance if present
- Try multiple models before choosing
- Focus on interpretable features
- Validate on holdout set
- Document your decisions

## Common Challenges
- Imbalanced classes (more non-churners)
- Feature engineering from raw data
- Choosing the right evaluation metric
- Overfitting on training data
- Deploying models effectively
