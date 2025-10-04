# Lab 2: Regression - Bike Rental & Housing Prediction

## Overview
Regression laboratory covering two comprehensive projects: bike rental demand prediction and end-to-end machine learning for housing prices. Learn to build, evaluate, and deploy regression models.

## Project Structure
```
Lab2/
├── Laboratorio 2 - Alquiler de Bicicletas.ipynb    # Bike rental assignment
├── 02_end_to_end_machine_learning_project.ipynb    # Housing project
├── datasets/                                        # Project data
├── models/                                          # Trained models
└── visualizations/                                  # Analysis plots
```

## Learning Objectives
- Regression problem formulation
- Feature engineering for continuous targets
- Model training and evaluation
- Hyperparameter optimization
- End-to-end ML pipeline creation
- Model deployment strategies
- Performance metrics for regression

## Getting Started

### Prerequisites
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
pip install scipy statsmodels jupyter
```

### How to Run
```bash
# Bike rental lab
jupyter notebook "Laboratorio 2 - Alquiler de Bicicletas.ipynb"

# End-to-end ML project
jupyter notebook "02_end_to_end_machine_learning_project.ipynb"
```

## Project 1: Bike Rental Demand Prediction

### Problem Statement
Predict hourly bike rental demand based on:
- **Temporal Features**: Hour, day, month, season
- **Weather Conditions**: Temperature, humidity, windspeed
- **Calendar Info**: Weekday, holiday, working day

### Lab Tasks
1. **Data Exploration**
   - Temporal pattern analysis
   - Weather impact on rentals
   - Seasonal trends
   - Holiday effects

2. **Feature Engineering**
   - Time-based features (hour of day, day of week)
   - Weather interaction terms
   - Lag features
   - Rolling averages

3. **Model Training**
   - Linear Regression
   - Ridge/Lasso Regression
   - Decision Trees
   - Random Forest
   - Gradient Boosting

4. **Evaluation**
   - RMSE (Root Mean Squared Error)
   - MAE (Mean Absolute Error)
   - R² Score
   - Residual analysis

## Project 2: End-to-End Machine Learning (Housing Prices)

### Complete ML Pipeline
This project demonstrates a professional ML workflow:

1. **Problem Framing**
   - Define business objective
   - Choose performance metrics
   - Identify assumptions

2. **Data Acquisition**
   - Load data
   - Initial inspection
   - Data profiling

3. **Exploratory Data Analysis**
   - Univariate analysis
   - Correlation analysis
   - Geographic visualization
   - Identify patterns

4. **Data Preparation**
   - Handle missing values
   - Feature scaling
   - Categorical encoding
   - Custom transformers
   - Pipeline creation

5. **Model Selection**
   - Try multiple algorithms
   - Cross-validation
   - Compare performance

6. **Hyperparameter Tuning**
   - Grid Search
   - Randomized Search
   - Model ensemble

7. **Model Evaluation**
   - Test set performance
   - Error analysis
   - Feature importance

8. **Deployment Preparation**
   - Save final model
   - Create prediction pipeline
   - Documentation

## Regression Algorithms Covered
- **Linear Models**: Linear, Ridge, Lasso, ElasticNet
- **Tree-Based**: Decision Trees, Random Forest
- **Boosting**: Gradient Boosting, XGBoost
- **Support Vector Regression**: SVR

## Evaluation Metrics

### Primary Metrics
- **RMSE**: Penalizes large errors
- **MAE**: Robust to outliers
- **R² Score**: Proportion of variance explained
- **MAPE**: Percentage error

### Residual Analysis
- Plot residuals vs predictions
- Check for patterns
- Verify normality
- Identify outliers

## Advanced Topics
- Feature engineering techniques
- Handling outliers
- Missing value strategies
- Feature selection methods
- Regularization (L1, L2)
- Ensemble methods
- Pipeline creation

## Business Applications

### Bike Rental System
- **Operations**: Staff scheduling, bike distribution
- **Inventory**: Predict bike demand at stations
- **Maintenance**: Plan based on usage patterns
- **Expansion**: Identify high-demand areas

### Housing Prices
- **Real Estate**: Property valuation
- **Investment**: Identify undervalued properties
- **Market Analysis**: Understand price drivers
- **Development**: Location assessment

## Deliverables
1. Completed Jupyter notebooks with analysis
2. Trained regression models
3. Performance evaluation reports
4. Feature importance analysis
5. Residual plots and diagnostics
6. Documentation of methodology

## Tips for Success
- Visualize data before modeling
- Check for outliers and handle appropriately
- Create meaningful features from raw data
- Use cross-validation for robust evaluation
- Compare multiple models
- Analyze residuals to improve models
- Document assumptions and decisions

## Common Challenges
- Non-linear relationships
- Outlier handling
- Feature scaling importance
- Overfitting vs underfitting
- Choosing the right metric
- Interpreting results

## Extension Ideas
- Time series forecasting
- Online learning
- A/B testing for models
- Real-time prediction API
- Dashboard creation
