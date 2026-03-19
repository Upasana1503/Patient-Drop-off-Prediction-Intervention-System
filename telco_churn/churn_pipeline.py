#!/usr/bin/env python3
"""Customer Churn Prediction & Retention Prioritization Engine."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    XGBOOST_IMPORT_ERROR = None
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except Exception as xgb_import_error:
    XGBOOST_AVAILABLE = False
    XGBOOST_IMPORT_ERROR = xgb_import_error


RANDOM_STATE = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train churn models and prioritize retention actions.")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("WA_Fn-UseC_-Telco-Customer-Churn.csv"),
        help="Path to the Telco churn CSV file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory to save reports and plots.",
    )
    parser.add_argument(
        "--churn-threshold",
        type=float,
        default=0.60,
        help="Threshold for high churn risk in gate 1.",
    )
    parser.add_argument(
        "--ltv-quantile",
        type=float,
        default=0.75,
        help="Quantile cutoff for high LTV in gate 2.",
    )
    return parser.parse_args()


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    tenure_safe = df["tenure"].replace(0, 1)
    df["avg_revenue_per_tenure"] = df["TotalCharges"] / tenure_safe
    df["charges_tenure_ratio"] = df["MonthlyCharges"] / tenure_safe
    df["is_new_customer"] = (df["tenure"] <= 6).astype(int)
    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=[-1, 6, 24, 72],
        labels=["New", "Mid", "Loyal"],
    )

    service_cols = [
        "PhoneService",
        "MultipleLines",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]
    existing_service_cols = [col for col in service_cols if col in df.columns]
    if existing_service_cols:
        service_yes = df[existing_service_cols].eq("Yes")
        df["total_services_count"] = service_yes.sum(axis=1)
    else:
        df["total_services_count"] = 0

    return df


def load_and_prepare_data(csv_path: Path) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame]:
    df_raw = pd.read_csv(csv_path)
    df = add_engineered_features(df_raw)

    y = df["Churn"].map({"Yes": 1, "No": 0})

    context_cols = [
        "customerID",
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        "Contract",
        "InternetService",
        "PaymentMethod",
        "tenure_group",
        "total_services_count",
        "Churn",
    ]
    context_cols = [c for c in context_cols if c in df.columns]
    scoring_context = df[context_cols].copy()

    X = df.drop(columns=["Churn", "customerID"])

    return X, y, scoring_context, df


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )
    return preprocessor


def get_models() -> dict[str, object]:
    models: dict[str, object] = {
        "LogisticRegression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(
            n_estimators=400,
            max_depth=12,
            min_samples_split=5,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    return models


def evaluate_model(y_true: pd.Series, y_pred: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
    }


def feature_importance_from_pipeline(model_name: str, pipeline: Pipeline) -> pd.DataFrame:
    preprocessor = pipeline.named_steps["preprocessor"]
    estimator = pipeline.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()

    if model_name == "LogisticRegression":
        importances = np.abs(estimator.coef_[0])
    elif hasattr(estimator, "feature_importances_"):
        importances = estimator.feature_importances_
    else:
        return pd.DataFrame(columns=["feature", "importance"])

    feat_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    return feat_df.sort_values("importance", ascending=False)


def apply_dual_gate_logic(
    scoring_df: pd.DataFrame,
    churn_threshold: float,
    ltv_threshold: float,
) -> pd.DataFrame:
    scored = scoring_df.copy()

    high_risk = scored["churn_probability"] >= churn_threshold
    high_value = scored["estimated_ltv"] >= ltv_threshold

    segment_conditions = [
        high_risk & high_value,
        high_risk & (~high_value),
        (~high_risk) & high_value,
    ]
    segment_choices = [
        "High Risk / High Value",
        "High Risk / Low Value",
        "Low Risk / High Value",
    ]
    scored["retention_segment"] = np.select(segment_conditions, segment_choices, default="Low Risk / Low Value")

    recommendation_conditions = [
        scored["retention_segment"] == "High Risk / High Value",
        scored["retention_segment"] == "High Risk / Low Value",
        scored["retention_segment"] == "Low Risk / High Value",
    ]
    recommendation_choices = [
        "Retain",
        "Let Churn",
        "Monitor",
    ]
    scored["retention_recommendation"] = np.select(
        recommendation_conditions,
        recommendation_choices,
        default="Minimal Priority",
    )

    scored["priority_rank"] = scored["retention_recommendation"].map(
        {
            "Retain": 1,
            "Monitor": 2,
            "Minimal Priority": 3,
            "Let Churn": 4,
        }
    )

    return scored


def save_business_insights(
    df_model: pd.DataFrame,
    scored_df: pd.DataFrame,
    output_dir: Path,
    churn_threshold: float,
    ltv_threshold: float,
) -> None:
    insights = []

    df = df_model.copy()
    df["churn_flag"] = df["Churn"].map({"Yes": 1, "No": 0})

    contract_churn = df.groupby("Contract", observed=False)["churn_flag"].mean().sort_values(ascending=False)
    insights.append("Churn rate by contract type:")
    for contract_name, churn_rate in contract_churn.items():
        insights.append(f"- {contract_name}: {churn_rate:.2%}")

    month_to_month = contract_churn.get("Month-to-month", np.nan)
    if not np.isnan(month_to_month):
        insights.append(f"Month-to-month churn rate: {month_to_month:.2%}")

    q75 = df["MonthlyCharges"].quantile(0.75)
    high_paying = df[df["MonthlyCharges"] >= q75]["churn_flag"].mean()
    non_high_paying = df[df["MonthlyCharges"] < q75]["churn_flag"].mean()
    insights.append(f"High-paying churn rate (top 25% MonthlyCharges): {high_paying:.2%}")
    insights.append(f"Other customers churn rate: {non_high_paying:.2%}")

    new_customers_churn = df[df["tenure"] <= 6]["churn_flag"].mean()
    established_customers_churn = df[df["tenure"] > 6]["churn_flag"].mean()
    insights.append(f"New customers (<=6 months) churn rate: {new_customers_churn:.2%}")
    insights.append(f"Established customers (>6 months) churn rate: {established_customers_churn:.2%}")

    insights.append("")
    insights.append("Dual-gate thresholds:")
    insights.append(f"- High churn threshold: {churn_threshold:.2f}")
    insights.append(f"- High LTV threshold: {ltv_threshold:.2f}")

    segment_summary = (
        scored_df.groupby(["retention_segment", "retention_recommendation"], observed=False)
        .agg(
            customers=("customerID", "count"),
            avg_churn_probability=("churn_probability", "mean"),
            avg_estimated_ltv=("estimated_ltv", "mean"),
        )
        .reset_index()
        .sort_values(["customers", "avg_estimated_ltv"], ascending=False)
    )
    segment_summary.to_csv(output_dir / "retention_segment_summary.csv", index=False)

    high_risk_high_value = scored_df[scored_df["retention_segment"] == "High Risk / High Value"]
    revenue_at_risk = high_risk_high_value["estimated_ltv"].sum()
    insights.append(
        f"High Risk / High Value customers: {len(high_risk_high_value)} | Estimated LTV at risk: {revenue_at_risk:.2f}"
    )

    insights.append("")
    insights.append("Retention strategy suggestions:")
    insights.append("- Retain: proactive outreach, contract upgrade offers, and service quality checks.")
    insights.append("- Monitor: personalized bundle nudges and loyalty reminders.")
    insights.append("- Minimal Priority / Let Churn: avoid costly incentives and monitor aggregate trend.")

    report_path = output_dir / "business_insights.txt"
    report_path.write_text("\n".join(insights), encoding="utf-8")


def plot_business_views(df_model: pd.DataFrame, output_dir: Path) -> None:
    df = df_model.copy()
    df["churn_flag"] = df["Churn"].map({"Yes": 1, "No": 0})

    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(8, 5))
    contract_churn = df.groupby("Contract", observed=False)["churn_flag"].mean().sort_values(ascending=False)
    sns.barplot(x=contract_churn.index, y=contract_churn.values)
    plt.title("Churn Rate by Contract Type")
    plt.ylabel("Churn Rate")
    plt.xlabel("Contract")
    plt.tight_layout()
    plt.savefig(output_dir / "churn_by_contract.png", dpi=140)
    plt.close()

    plt.figure(figsize=(7, 5))
    tenure_churn = df.groupby("tenure_group", observed=False)["churn_flag"].mean().reindex(["New", "Mid", "Loyal"])
    sns.barplot(x=tenure_churn.index, y=tenure_churn.values)
    plt.title("Churn Rate by Tenure Group")
    plt.ylabel("Churn Rate")
    plt.xlabel("Tenure Group")
    plt.tight_layout()
    plt.savefig(output_dir / "churn_by_tenure_group.png", dpi=140)
    plt.close()


def plot_retention_matrix(
    scored_df: pd.DataFrame,
    output_dir: Path,
    churn_threshold: float,
    ltv_threshold: float,
) -> None:
    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        data=scored_df,
        x="churn_probability",
        y="estimated_ltv",
        hue="retention_segment",
        alpha=0.65,
        s=40,
    )
    plt.axvline(churn_threshold, color="red", linestyle="--", label="Churn Threshold")
    plt.axhline(ltv_threshold, color="black", linestyle="--", label="LTV Threshold")
    plt.title("Retention Matrix: Churn Probability vs Estimated LTV")
    plt.xlabel("Churn Probability")
    plt.ylabel("Estimated LTV")
    plt.tight_layout()
    plt.savefig(output_dir / "retention_matrix.png", dpi=140)
    plt.close()


def main() -> None:
    args = parse_args()
    data_path = args.data
    output_dir = args.output_dir
    churn_threshold = args.churn_threshold
    ltv_quantile = args.ltv_quantile

    if not 0 < churn_threshold < 1:
        raise ValueError("--churn-threshold must be between 0 and 1.")
    if not 0 < ltv_quantile < 1:
        raise ValueError("--ltv-quantile must be between 0 and 1.")

    output_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    X, y, scoring_context, df_model = load_and_prepare_data(data_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    preprocessor = build_preprocessor(X)
    models = get_models()

    if not XGBOOST_AVAILABLE:
        print("[Info] XGBoost unavailable. Comparing Logistic Regression and Random Forest only.")
        if XGBOOST_IMPORT_ERROR is not None:
            print(f"[Info] XGBoost import error: {XGBOOST_IMPORT_ERROR}")

    model_results = []
    fitted_models: dict[str, Pipeline] = {}

    for model_name, estimator in models.items():
        model_preprocessor = clone(preprocessor)
        pipeline = Pipeline(
            steps=[
                ("preprocessor", model_preprocessor),
                ("model", estimator),
            ]
        )
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        metrics = evaluate_model(y_test, y_pred, y_prob)
        metrics["model"] = model_name
        model_results.append(metrics)
        fitted_models[model_name] = pipeline

        report_file = output_dir / f"classification_report_{model_name}.txt"
        report_file.write_text(classification_report(y_test, y_pred), encoding="utf-8")

    results_df = pd.DataFrame(model_results).sort_values(["roc_auc", "f1"], ascending=False)
    results_df.to_csv(output_dir / "model_comparison.csv", index=False)

    best_model_name = results_df.iloc[0]["model"]
    best_pipeline = fitted_models[best_model_name]

    y_test_pred = best_pipeline.predict(X_test)
    test_cm = confusion_matrix(y_test, y_test_pred)
    cm_df = pd.DataFrame(test_cm, index=["Actual_No", "Actual_Yes"], columns=["Pred_No", "Pred_Yes"])
    cm_df.to_csv(output_dir / f"confusion_matrix_{best_model_name}.csv")

    all_prob = best_pipeline.predict_proba(X)[:, 1]
    all_pred = (all_prob >= 0.5).astype(int)

    scoring_df = scoring_context.copy()
    scoring_df["actual_churn"] = scoring_df["Churn"].map({"Yes": 1, "No": 0})
    scoring_df = scoring_df.drop(columns=["Churn"])
    scoring_df["churn_probability"] = all_prob
    scoring_df["churn_prediction"] = all_pred
    scoring_df["churn_risk_band"] = pd.cut(
        scoring_df["churn_probability"],
        bins=[-0.01, 0.33, 0.66, 1.0],
        labels=["Low Risk", "Medium Risk", "High Risk"],
    )

    scoring_df["estimated_ltv"] = scoring_df["MonthlyCharges"] * scoring_df["tenure"]
    ltv_threshold = float(scoring_df["estimated_ltv"].quantile(ltv_quantile))
    scoring_df["ltv_tier"] = np.where(scoring_df["estimated_ltv"] >= ltv_threshold, "High LTV", "Low LTV")

    scored_with_actions = apply_dual_gate_logic(scoring_df, churn_threshold=churn_threshold, ltv_threshold=ltv_threshold)

    scored_with_actions = scored_with_actions.sort_values(
        ["priority_rank", "churn_probability", "estimated_ltv"],
        ascending=[True, False, False],
    )

    scored_with_actions.to_csv(output_dir / "customer_retention_prioritization.csv", index=False)
    scored_with_actions.to_csv(output_dir / "customer_risk_scoring.csv", index=False)

    feat_imp_df = feature_importance_from_pipeline(best_model_name, best_pipeline)
    if not feat_imp_df.empty:
        feat_imp_df.to_csv(output_dir / f"feature_importance_{best_model_name}.csv", index=False)

        top_n = feat_imp_df.head(20)
        plt.figure(figsize=(10, 8))
        sns.barplot(data=top_n, x="importance", y="feature")
        plt.title(f"Top 20 Feature Importances - {best_model_name}")
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.tight_layout()
        plt.savefig(output_dir / f"feature_importance_{best_model_name}.png", dpi=140)
        plt.close()

    save_business_insights(
        df_model=df_model,
        scored_df=scored_with_actions,
        output_dir=output_dir,
        churn_threshold=churn_threshold,
        ltv_threshold=ltv_threshold,
    )
    plot_business_views(df_model=df_model, output_dir=output_dir)
    plot_retention_matrix(
        scored_df=scored_with_actions,
        output_dir=output_dir,
        churn_threshold=churn_threshold,
        ltv_threshold=ltv_threshold,
    )

    print("Training complete.")
    print(f"Best model: {best_model_name}")
    print(f"LTV high-value threshold: {ltv_threshold:.2f}")
    print(f"Output directory: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
