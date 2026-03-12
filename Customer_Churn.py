from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42
DATA_PATH = Path("Telco-Customer-Churn.csv")
OUTPUT_DIR = Path("outputs")
FIGURES_DIR = OUTPUT_DIR / "figures"


def ensure_output_dirs() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_data(data_path: Path = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(data_path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()

    object_columns = cleaned.select_dtypes(include=["object", "string"]).columns
    cleaned[object_columns] = cleaned[object_columns].apply(lambda col: col.str.strip())

    cleaned["TotalCharges"] = pd.to_numeric(cleaned["TotalCharges"], errors="coerce")
    missing_total = cleaned["TotalCharges"].isna()
    cleaned.loc[missing_total, "TotalCharges"] = (
        cleaned.loc[missing_total, "MonthlyCharges"] * cleaned.loc[missing_total, "tenure"]
    )

    cleaned["SeniorCitizen"] = cleaned["SeniorCitizen"].map({0: "No", 1: "Yes"})
    cleaned["ChurnLabel"] = cleaned["Churn"].map({"No": 0, "Yes": 1})
    return cleaned


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    featured = df.copy()

    featured["tenure_band"] = pd.cut(
        featured["tenure"],
        bins=[-1, 12, 24, 48, 60, 72],
        labels=["0-12 months", "13-24 months", "25-48 months", "49-60 months", "61-72 months"],
    )

    service_columns = [
        "PhoneService",
        "MultipleLines",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]
    positive_service_values = {"Yes"}
    featured["total_services"] = featured[service_columns].apply(
        lambda row: sum(value in positive_service_values for value in row), axis=1
    )

    tenure_for_ratio = featured["tenure"].replace(0, 1)
    total_for_ratio = featured["TotalCharges"].replace(0, np.nan)

    featured["avg_monthly_charge_from_total"] = featured["TotalCharges"] / tenure_for_ratio
    featured["monthly_to_total_ratio"] = featured["MonthlyCharges"] / total_for_ratio
    featured["monthly_to_total_ratio"] = featured["monthly_to_total_ratio"].fillna(0)
    featured["is_new_customer"] = np.where(featured["tenure"] <= 12, "Yes", "No")
    featured["has_streaming_bundle"] = np.where(
        (featured["StreamingTV"] == "Yes") & (featured["StreamingMovies"] == "Yes"),
        "Yes",
        "No",
    )

    return featured


def create_feature_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    features = df.drop(columns=["customerID", "Churn", "ChurnLabel"])
    target = df["ChurnLabel"]
    return features, target


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    categorical_columns = features.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    numeric_columns = features.select_dtypes(include=["number"]).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_columns),
            ("cat", categorical_pipeline, categorical_columns),
        ]
    )


def build_models(preprocessor: ColumnTransformer) -> Dict[str, Pipeline]:
    return {
        "Logistic Regression": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=400,
                        min_samples_leaf=4,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def evaluate_models(
    models: Dict[str, Pipeline],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, np.ndarray | Pipeline]]]:
    rows = []
    fitted_models: Dict[str, Dict[str, np.ndarray | Pipeline]] = {}

    for model_name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        probabilities = pipeline.predict_proba(X_test)[:, 1]
        matrix = confusion_matrix(y_test, predictions)

        rows.append(
            {
                "model": model_name,
                "accuracy": accuracy_score(y_test, predictions),
                "precision": precision_score(y_test, predictions),
                "recall": recall_score(y_test, predictions),
                "f1_score": f1_score(y_test, predictions),
                "roc_auc": roc_auc_score(y_test, probabilities),
            }
        )
        fitted_models[model_name] = {
            "pipeline": pipeline,
            "predictions": predictions,
            "probabilities": probabilities,
            "confusion_matrix": matrix,
        }

    metrics_df = pd.DataFrame(rows).sort_values(by="roc_auc", ascending=False).reset_index(drop=True)
    return metrics_df, fitted_models


def get_top_logistic_coefficients(pipeline: Pipeline, top_n: int = 15) -> pd.DataFrame:
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()
    coefficients = model.coef_[0]

    coefficient_frame = pd.DataFrame(
        {"feature": feature_names, "coefficient": coefficients, "abs_coefficient": np.abs(coefficients)}
    )
    return coefficient_frame.sort_values("abs_coefficient", ascending=False).head(top_n)


def get_top_tree_importances(pipeline: Pipeline, top_n: int = 15) -> pd.DataFrame:
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()

    importance_frame = pd.DataFrame({"feature": feature_names, "importance": model.feature_importances_})
    return importance_frame.sort_values("importance", ascending=False).head(top_n)


def save_dataset_snapshot(df: pd.DataFrame) -> None:
    profile = pd.DataFrame(
        {
            "column": df.columns,
            "dtype": df.dtypes.astype(str).values,
            "missing_values": df.isna().sum().values,
            "unique_values": [df[column].nunique() for column in df.columns],
        }
    )
    profile.to_csv(OUTPUT_DIR / "dataset_profile.csv", index=False)


def plot_churn_distribution(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    order = ["No", "Yes"]
    sns.countplot(
        data=df,
        x="Churn",
        order=order,
        hue="Churn",
        palette=["#5B8E7D", "#D95D39"],
        legend=False,
        ax=ax,
    )
    ax.set_title("Customer Churn Distribution")
    ax.set_xlabel("Churn")
    ax.set_ylabel("Customers")
    for patch in ax.patches:
        ax.annotate(
            f"{int(patch.get_height()):,}",
            (patch.get_x() + patch.get_width() / 2, patch.get_height()),
            ha="center",
            va="bottom",
        )
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "churn_distribution.png", dpi=300)
    plt.close(fig)


def plot_contract_vs_churn(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    contract_rates = (
        df.groupby("Contract", observed=False)["ChurnLabel"].mean().sort_values(ascending=False).mul(100).reset_index()
    )
    sns.barplot(
        data=contract_rates,
        x="Contract",
        y="ChurnLabel",
        hue="Contract",
        palette="crest",
        legend=False,
        ax=ax,
    )
    ax.set_title("Churn Rate by Contract Type")
    ax.set_xlabel("Contract")
    ax.set_ylabel("Churn Rate (%)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "contract_vs_churn.png", dpi=300)
    plt.close(fig)


def plot_tenure_and_charges(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.histplot(data=df, x="tenure", hue="Churn", bins=30, kde=True, multiple="stack", ax=axes[0])
    axes[0].set_title("Tenure Distribution by Churn")
    axes[0].set_xlabel("Tenure (months)")

    sns.boxplot(
        data=df,
        x="Churn",
        y="MonthlyCharges",
        hue="Churn",
        palette=["#5B8E7D", "#D95D39"],
        legend=False,
        ax=axes[1],
    )
    axes[1].set_title("Monthly Charges by Churn")
    axes[1].set_xlabel("Churn")
    axes[1].set_ylabel("Monthly Charges")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "tenure_and_charges.png", dpi=300)
    plt.close(fig)


def plot_model_performance(metrics_df: pd.DataFrame) -> None:
    metrics_long = metrics_df.melt(id_vars="model", var_name="metric", value_name="score")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=metrics_long, x="metric", y="score", hue="model", palette="Set2", ax=ax)
    ax.set_title("Model Performance Comparison")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Metric")
    ax.set_ylabel("Score")
    ax.legend(title="Model")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "model_performance.png", dpi=300)
    plt.close(fig)


def plot_confusion_matrices(fitted_models: Dict[str, Dict[str, np.ndarray | Pipeline]], y_test: pd.Series) -> None:
    fig, axes = plt.subplots(1, len(fitted_models), figsize=(12, 5))
    axes = np.atleast_1d(axes)

    for ax, (model_name, payload) in zip(axes, fitted_models.items()):
        ConfusionMatrixDisplay(payload["confusion_matrix"]).plot(ax=ax, colorbar=False)
        ax.set_title(model_name)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "confusion_matrices.png", dpi=300)
    plt.close(fig)


def plot_roc_curves(fitted_models: Dict[str, Dict[str, np.ndarray | Pipeline]], y_test: pd.Series) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))

    for model_name, payload in fitted_models.items():
        probabilities = payload["probabilities"]
        fpr, tpr, _ = roc_curve(y_test, probabilities)
        auc_score = roc_auc_score(y_test, probabilities)
        ax.plot(fpr, tpr, linewidth=2, label=f"{model_name} (AUC = {auc_score:.3f})")

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_title("ROC Curve Comparison")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "roc_curves.png", dpi=300)
    plt.close(fig)


def plot_feature_insights(logistic_df: pd.DataFrame, tree_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    logistic_plot = logistic_df.sort_values("coefficient")
    colors = ["#D95D39" if value > 0 else "#5B8E7D" for value in logistic_plot["coefficient"]]
    axes[0].barh(logistic_plot["feature"], logistic_plot["coefficient"], color=colors)
    axes[0].set_title("Logistic Regression Coefficients")
    axes[0].set_xlabel("Coefficient")

    tree_plot = tree_df.sort_values("importance")
    axes[1].barh(tree_plot["feature"], tree_plot["importance"], color="#4F6D7A")
    axes[1].set_title("Random Forest Feature Importance")
    axes[1].set_xlabel("Importance")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "feature_insights.png", dpi=300)
    plt.close(fig)


def save_model_artifacts(
    metrics_df: pd.DataFrame,
    logistic_df: pd.DataFrame,
    tree_df: pd.DataFrame,
) -> None:
    metrics_df.to_csv(OUTPUT_DIR / "model_metrics.csv", index=False)
    logistic_df.to_csv(OUTPUT_DIR / "logistic_coefficients.csv", index=False)
    tree_df.to_csv(OUTPUT_DIR / "random_forest_importance.csv", index=False)


def run_analysis(data_path: Path = DATA_PATH) -> Dict[str, pd.DataFrame | Dict[str, Dict[str, np.ndarray | Pipeline]]]:
    ensure_output_dirs()
    sns.set_theme(style="whitegrid", context="talk")

    raw_df = load_data(data_path)
    cleaned_df = clean_data(raw_df)
    featured_df = engineer_features(cleaned_df)

    save_dataset_snapshot(featured_df)
    plot_churn_distribution(featured_df)
    plot_contract_vs_churn(featured_df)
    plot_tenure_and_charges(featured_df)

    X, y = create_feature_matrix(featured_df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    preprocessor = build_preprocessor(X)
    models = build_models(preprocessor)
    metrics_df, fitted_models = evaluate_models(models, X_train, X_test, y_train, y_test)

    plot_model_performance(metrics_df)
    plot_confusion_matrices(fitted_models, y_test)
    plot_roc_curves(fitted_models, y_test)

    logistic_df = get_top_logistic_coefficients(fitted_models["Logistic Regression"]["pipeline"])
    tree_df = get_top_tree_importances(fitted_models["Random Forest"]["pipeline"])
    plot_feature_insights(logistic_df, tree_df)
    save_model_artifacts(metrics_df, logistic_df, tree_df)

    return {
        "raw_df": raw_df,
        "featured_df": featured_df,
        "metrics_df": metrics_df,
        "logistic_df": logistic_df,
        "tree_df": tree_df,
        "fitted_models": fitted_models,
        "y_test": y_test,
    }


def print_summary(results: Dict[str, pd.DataFrame | Dict[str, Dict[str, np.ndarray | Pipeline]]]) -> None:
    metrics_df = results["metrics_df"]
    logistic_df = results["logistic_df"]
    tree_df = results["tree_df"]

    print("Model comparison")
    print(metrics_df.to_string(index=False))
    print("\nTop logistic regression coefficients")
    print(logistic_df[["feature", "coefficient"]].to_string(index=False))
    print("\nTop random forest features")
    print(tree_df.to_string(index=False))


if __name__ == "__main__":
    analysis_results = run_analysis()
    print_summary(analysis_results)
