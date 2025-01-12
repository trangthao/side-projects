# Predicting Credit Card Default

## Business Context

Banks make money by lending to people who repay on time with interest. The more they lend to reliable borrowers, the more money they earn. If banks can identify borrowers likely to have trouble repaying loans, they can reduce risks, helping them earn more money, avoid losses, and maintain a good reputation.

### Key Terms
- **Delinquent**: A borrower is behind on payments for a few months but might still pay.
- **Default**: A borrower has not paid for a long time and is unlikely to repay.

### Data Overview

We have information about borrowers, including:
- **Personal details**: Age, monthly income, and number of dependents.
- **Financial history**: Debt-to-income ratio, how much they owe compared to their credit limit, and how often they missed payments in the last few months.

### Objective

The goal is to use this data to predict whether a borrower will fall behind on payments in the next two years. This will assist banks in making better lending decisions and reducing their risk exposure.

## Libraries

The following libraries are required to run the notebook for Python 3.10.12:

- `imbalanced_learn==0.10.1`
- `imblearn==0.0`
- `lightgbm==3.3.5`
- `numpy==1.24.4`
- `pandas==1.5.3`
- `scikit_learn==1.2.2`
- `matplotlib==3.7.3`
- `scipy==1.10.1`
- `shap==0.42.1`
- `lime==0.2.0.1`
- `seaborn==0.12.2`
- `keras==2.11.0`
- `tensorflow==2.11.1`
- `xgboost==1.7.6`
- `catboost==1.2`

## Notebook Overview

1. **Business Context**: The notebook starts by providing an understanding of the business context of the project.
2. **Data Preparation**: Data is cleaned, missing values are handled, and relevant features are extracted.
3. **Exploratory Data Analysis (EDA)**: EDA is performed to visualize relationships between variables and understand the dataset.
4. **Modeling**: The notebook explores various machine learning models to predict credit card defaults, including LightGBM, XGBoost, and others.
5. **Evaluation**: The models are evaluated using precision, recall, and other performance metrics to determine the best model for predicting defaults.
