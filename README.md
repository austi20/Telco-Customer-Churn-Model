# Telco Customer Churn Prediction

This project predicts customer churn for a telecom provider using the IBM Telco Customer Churn dataset. It is built as a portfolio-ready machine learning case study with clear preprocessing, exploratory analysis, feature engineering, model comparison, and business interpretation.

## Project Goal

The objective is to identify which customers are most likely to churn and explain the main risk drivers so a business can improve retention strategy, reduce revenue leakage, and target interventions more effectively.

## Dataset

- Source file: `Telco-Customer-Churn.csv`
- Records: 7,043 customers
- Target: `Churn`
- Feature groups:
  - Demographics: gender, senior citizen, partner, dependents
  - Customer lifecycle: tenure, contract type
  - Service usage: phone, internet, streaming, security, backup, support
  - Billing and payment: monthly charges, total charges, paperless billing, payment method

## Methods

- Cleaned the dataset and fixed `TotalCharges`, which was stored as text with blank-string missing values
- Encoded categorical variables with one-hot encoding
- Scaled numeric variables for logistic regression
- Engineered portfolio-friendly features:
  - `tenure_band`
  - `total_services`
  - `avg_monthly_charge_from_total`
  - `monthly_to_total_ratio`
  - `is_new_customer`
  - `has_streaming_bundle`
- Trained and compared:
  - Logistic Regression
  - Random Forest

## Results

Test-set performance:

| Model | Accuracy | Precision | Recall | F1 Score | ROC AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.735 | 0.500 | 0.794 | 0.614 | 0.847 |
| Random Forest | 0.772 | 0.556 | 0.703 | 0.621 | 0.841 |

Key interpretation:

- Random Forest delivered the best overall classification balance by accuracy and F1 score.
- Logistic Regression achieved the best ROC AUC, making it a strong benchmark for ranking churn risk.
- The most important churn signals were consistently tied to:
  - Month-to-month contracts
  - Short tenure / new customers
  - Fiber optic internet service
  - Electronic check payments
  - Weak retention-oriented add-ons such as missing online security or tech support
- Two-year contracts were strongly associated with lower churn risk.

## Business Value

This analysis can support a retention program by helping a telecom business:

- Prioritize outreach for newer month-to-month customers
- Bundle security and support services for at-risk accounts
- Encourage migration from month-to-month plans to longer contracts
- Target payment-friction segments such as electronic-check users

## Project Files

- `Customer_Churn.py`: reusable end-to-end pipeline for cleaning, feature engineering, modeling, and chart generation
- `Customer_Churn_Portfolio.ipynb`: notebook version organized for presentation
- `outputs/model_metrics.csv`: exported model comparison table
- `outputs/logistic_coefficients.csv`: top logistic regression coefficient drivers
- `outputs/random_forest_importance.csv`: top random forest feature importances
- `outputs/figures/`: saved charts for EDA and model evaluation

## How to Run

```powershell
.\.venv\Scripts\pip.exe install -r requirements.txt
.\.venv\Scripts\python.exe Customer_Churn.py
.\.venv\Scripts\python.exe build_notebook.py
```

To open the notebook locally:

```powershell
.\.venv\Scripts\jupyter-lab.exe
```

## Resume Summary

Built an end-to-end customer churn prediction project in Python using pandas, scikit-learn, matplotlib, and Jupyter Notebook. Cleaned telecom customer data, engineered retention-focused features, and compared logistic regression with random forest using ROC AUC, F1, confusion matrices, and feature interpretation to translate model output into business retention actions.
