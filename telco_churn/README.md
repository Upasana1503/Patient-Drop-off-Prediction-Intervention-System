# Patient Drop-off Prediction & Intervention System

This project predicts patient drop-off risk (e.g., missed follow-ups, treatment non-adherence) and prioritizes clinical interventions using a dual-gate decision framework.

## Clinical Objective

- Predict each patient's risk of disengagement from treatment or follow-up
- Estimate patient priority score (proxy using engagement and treatment duration features)
- Recommend intervention strategies only when clinically justified

## Dual-Gate Decision Framework

- Gate 1 (Risk): Identify patients with high risk of treatment drop-off
- Gate 2 (Value): Among them, prioritize patients with high clinical importance / engagement

## Decision Logic

- High risk + High priority -> Immediate Intervention
- High risk + Low priority -> Basic Follow-up
- Low risk + High priority -> Monitor Closely
- Low risk + Low priority -> Routine Care

## Pipeline Overview

- Data preprocessing and cleaning
- Feature engineering:
  - Patient engagement metrics (e.g., visit frequency, tenure)
  - Treatment duration indicators
  - Derived ratios (engagement consistency, service utilization)
- Patient segmentation (New, Ongoing, Long-term)
- Model development and comparison:
  - Logistic Regression
  - Random Forest
  - XGBoost
- Clinical insight layer:
  - Risk variation across patient groups
  - Early-stage vs long-term patient drop-off patterns
  - High-engagement vs low-engagement patient behavior
- Decision support outputs:
  - Risk score per patient
  - Priority score
  - Intervention recommendation category

## Key Features

- Predictive modeling for patient adherence and engagement risk
- Data-driven care prioritization using a dual-gate framework
- Interpretable outputs for clinical decision support
- Modular pipeline adaptable to electronic health record (EHR) systems

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python churn_pipeline.py --data patient_data.csv --output-dir outputs
```

Optional thresholds:

```bash
python churn_pipeline.py \
  --data patient_data.csv \
  --output-dir outputs \
  --risk-threshold 0.60 \
  --priority-quantile 0.75
```

## Output Artifacts

- Patient risk scoring reports
- Care prioritization recommendations
- Model performance metrics
- Feature importance and interpretability outputs
- Visualization of patient risk segmentation

## Quick Interpretation

- Use model comparison metrics (ROC-AUC, F1) to select best model
- Use prioritization outputs to target high-risk patients for intervention
- Use visualizations to support clinical and operational decision-making
