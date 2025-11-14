# Insurance Health-Risk Classifier
A reproducible ML pipeline that classifies insurance applicants as **healthy** or **unhealthy** to inform eligibility and premium pricing decisions. The project covers data cleaning, domain-aware feature engineering, model training, evaluation, and artifact export suitable for integration with pricing systems.

---
## Overview
Anova Insurance aims to optimize premium pricing by assessing applicant health risk. This repository implements a supervised binary classification workflow built on a 10,000-row dataset with mixed numeric and categorical columns. The pipeline emphasizes data hygiene, interpretability, and business-aligned evaluation so models can safely inform underwriting decisions.

## Approach & Key Steps
1. **Data validation & cleaning**
   * Detect and correct obvious data-entry errors (e.g., negative ages handled via rule-based correction or removal).
   * Enforce types and ranges; flag outliers for domain review.
2. **Missing-value strategy**
   * Apply targeted imputation depending on feature semantics (median for continuous clinical measures, mode/conditional imputation for categorical lifestyle fields).
   * Maintain a missingness indicator for features where absence itself is informative.
3. **Feature engineering**
   * Normalized BMI and blood-pressure buckets.
   * Aggregate lifestyle score combining Smoking / Alcohol / Diet / PhysicalActivity.
   * Interaction features (e.g., Age × MedicalHistory severity).
4. **Training & model selection**
   * Baseline models (logistic regression) and tree ensembles (XGBoost / RandomForest).
   * Cross-validation and hyperparameter tuning for robust selection.
   * Class imbalance handled with stratified sampling, class weights, or resampling where appropriate.
5. **Evaluation & business metrics**
   * Standard metrics: Precision/Recall, F1, ROC-AUC.
   * Calibration checks and threshold selection for operational trade-offs (eligibility vs. premium sensitivity).
   * Business-impact metrics: estimated premium lift, false-negative cost analysis.
6. **Explainability**
   * Feature importance and SHAP-compatible exports for per-claim explanations.
     
---
## Features / Engineering (high level)
* BMI normalization & categorical bins
* Blood-pressure and cholesterol risk bands
* Lifestyle composite scores (smoking/alcohol/diet/activity)
* Missingness indicator features
* Interaction features for age and medical history severity

---
## Model Training & Evaluation
* Models trained with scikit-learn and XGBoost.
* Hyperparameter tuning via cross-validation (GridSearch or randomized search).
* Output evaluation split includes train/validation/test with stratified sampling.
* Key metrics produced: ROC-AUC, Precision@k, Recall, F1, calibration plots, confusion matrices, and cost-weighted error analysis.

---
## Outputs & Artifacts

* Trained model files (pickled or joblib) and metadata (features, preprocessing pipeline).
* Evaluation report (CSV/HTML) with metric summaries and recommended decision thresholds.
* Feature-importance report suitable for SHAP analysis.
* Inference script / notebook to score new applicants and produce eligibility + suggested premium adjustments.

---
## Limitations & Assumptions

* Negative ages are treated as data errors and corrected/removed per documented rules — this may discard ambiguous records.
* Labels reflect the provided `Target` column; label noise or target-definition bias will affect model performance.
* The model is trained on historical data and may not generalize to new population distributions without recalibration.

---
## Future improvements

* Add probabilistic calibration and conformal prediction for safer decision-making.
* Deploy a lightweight API or batch-scoring service for production integration.
* Expand feature set with longitudinal records or external risk signals.
* Add fairness audits across demographics and automatic monitoring pipelines for data drift.
