from __future__ import annotations

from pathlib import Path

import nbformat as nbf


NOTEBOOK_PATH = Path("Customer_Churn_Portfolio.ipynb")


def markdown_cell(text: str):
    return nbf.v4.new_markdown_cell(text)


def code_cell(code: str):
    return nbf.v4.new_code_cell(code)


def build_notebook() -> None:
    nb = nbf.v4.new_notebook()
    nb.metadata.update(
        {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.14",
            },
        }
    )

    nb.cells = [
        markdown_cell(
            "# Customer Churn Prediction\n"
            "A portfolio-ready end-to-end machine learning project using pandas, scikit-learn, matplotlib, and Jupyter Notebook.\n\n"
            "**Objective:** predict customer churn and identify the strongest retention risk drivers in the Telco customer base."
        ),
        markdown_cell(
            "## 1. Imports and Setup\n"
            "This notebook uses a reusable analysis module so the workflow is easy to rerun as a script or present in a notebook."
        ),
        code_cell(
            "from pathlib import Path\n\n"
            "import pandas as pd\n"
            "from IPython.display import Image, display\n\n"
            "from Customer_Churn import (\n"
            "    clean_data,\n"
            "    engineer_features,\n"
            "    run_analysis,\n"
            ")\n\n"
            "pd.set_option('display.max_columns', None)\n"
            "DATA_PATH = Path('Telco-Customer-Churn.csv')"
        ),
        markdown_cell("## 2. Data Loading\nRead the raw customer churn dataset and inspect the structure."),
        code_cell(
            "raw_df = pd.read_csv(DATA_PATH)\n"
            "print(f'Shape: {raw_df.shape[0]:,} rows x {raw_df.shape[1]} columns')\n"
            "raw_df.head()"
        ),
        code_cell(
            "raw_df.info()\n\n"
            "display(raw_df.describe(include='all').transpose().head(10))"
        ),
        markdown_cell(
            "## 3. Preprocessing and Cleaning\n"
            "Key tasks:\n"
            "- fix `TotalCharges` data type and blank-string missing values\n"
            "- standardize categorical values\n"
            "- create a binary target column for modeling"
        ),
        code_cell(
            "cleaned_df = clean_data(raw_df)\n"
            "cleaned_df[['tenure', 'MonthlyCharges', 'TotalCharges', 'Churn', 'ChurnLabel']].head()"
        ),
        code_cell(
            "cleaned_df.isna().sum().sort_values(ascending=False).head(10)"
        ),
        markdown_cell(
            "## 4. Feature Engineering\n"
            "Business-oriented features improve interpretability and model signal:\n"
            "- `tenure_band`\n"
            "- `total_services`\n"
            "- `avg_monthly_charge_from_total`\n"
            "- `monthly_to_total_ratio`\n"
            "- simple customer state flags such as `is_new_customer`"
        ),
        code_cell(
            "featured_df = engineer_features(cleaned_df)\n"
            "featured_df[\n"
            "    [\n"
            "        'tenure', 'tenure_band', 'MonthlyCharges', 'TotalCharges',\n"
            "        'total_services', 'avg_monthly_charge_from_total',\n"
            "        'monthly_to_total_ratio', 'is_new_customer', 'has_streaming_bundle'\n"
            "    ]\n"
            "].head()"
        ),
        markdown_cell(
            "## 5. Exploratory Analysis\n"
            "Run the full pipeline once to create saved visual assets and model outputs."
        ),
        code_cell(
            "results = run_analysis(DATA_PATH)\n"
            "metrics_df = results['metrics_df']\n"
            "logistic_df = results['logistic_df']\n"
            "tree_df = results['tree_df']\n"
            "featured_df = results['featured_df']\n"
            "metrics_df"
        ),
        code_cell(
            "display(Image(filename='outputs/figures/churn_distribution.png'))\n"
            "display(Image(filename='outputs/figures/contract_vs_churn.png'))\n"
            "display(Image(filename='outputs/figures/tenure_and_charges.png'))"
        ),
        markdown_cell(
            "## 6. Model Training and Evaluation\n"
            "Two models are compared:\n"
            "- Logistic Regression for a strong baseline with interpretable coefficients\n"
            "- Random Forest for a nonlinear tree-based benchmark"
        ),
        code_cell(
            "metrics_df.style.format({\n"
            "    'accuracy': '{:.3f}',\n"
            "    'precision': '{:.3f}',\n"
            "    'recall': '{:.3f}',\n"
            "    'f1_score': '{:.3f}',\n"
            "    'roc_auc': '{:.3f}'\n"
            "})"
        ),
        code_cell(
            "display(Image(filename='outputs/figures/model_performance.png'))\n"
            "display(Image(filename='outputs/figures/confusion_matrices.png'))\n"
            "display(Image(filename='outputs/figures/roc_curves.png'))"
        ),
        markdown_cell(
            "## 7. Model Interpretation\n"
            "Interpretability matters for portfolio projects and real retention strategy. "
            "The logistic model shows directional impact, while the random forest highlights overall predictive importance."
        ),
        code_cell(
            "logistic_df[['feature', 'coefficient']].head(10)"
        ),
        code_cell(
            "tree_df.head(10)"
        ),
        code_cell(
            "display(Image(filename='outputs/figures/feature_insights.png'))"
        ),
        markdown_cell(
            "## 8. Conclusion\n"
            "Use the saved metrics and feature rankings to summarize churn drivers, model tradeoffs, and the business actions suggested by the analysis."
        ),
        code_cell(
            "best_model = metrics_df.sort_values('roc_auc', ascending=False).iloc[0]\n"
            "top_positive = logistic_df.sort_values('coefficient', ascending=False).head(5)[['feature', 'coefficient']]\n"
            "top_tree = tree_df.head(5)\n\n"
            "print(f\"Best model by ROC AUC: {best_model['model']} ({best_model['roc_auc']:.3f})\")\n"
            "print('\\nTop churn-increasing logistic coefficients:')\n"
            "display(top_positive)\n"
            "print('\\nTop random forest importance drivers:')\n"
            "display(top_tree)"
        ),
    ]

    nbf.write(nb, NOTEBOOK_PATH)


if __name__ == "__main__":
    build_notebook()
