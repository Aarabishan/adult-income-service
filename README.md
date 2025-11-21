# Adult Income Classification - ML Coursework

End-to-end pipeline for predicting whether a person’s income is >50K using the UCI Adult Census dataset.
Includes preprocessing, model training & comparison, explainability (Permutation Importance + SHAP), and a production-style FastAPI service containerized with Docker.

## 📌 Project Objectives

- Build a reproducible ML pipeline for binary classification (<=50K vs >50K).

- Compare four model families: Logistic Regression, Random Forest, CatBoost, MLP.

- Apply threshold tuning on a validation set and evaluate on a held-out test set.

- Provide interpretability (Permutation Importance, SHAP) and error analysis.

- Package a FastAPI inference service and run it via Docker.

## 🧰 Environment & Reproducibility

- Python: 3.11 (service image: python:3.11-slim)

- Key libs: scikit-learn, catboost, pandas, numpy, fastapi, uvicorn

- MLflow for experiment tracking (mlruns file not added due to large file size)

## 🧮 Data & Features

- Dataset: UCI Adult (Census Income)

- After cleaning/feature engineering, we used:

- Numerical (continuous): age, education-num, capital-gain, capital-loss, hours-per-week

- Binary numeric: net_Capital, has_capital, marital_simple, is_us

- Categorical (OHE): workclass, occupation, relationship, race, gender, age_group, work_hours_category, education_level

- Target: income → <=50K (0), >50K (1)

- Split: 70% train / 15% val / 15% test (stratified)
  
## 🔧 Modeling

Trained with a shared ColumnTransformer:

- Numeric → impute (median) + StandardScaler

- Binary numeric → impute (most_frequent), no scaling

- Categorical → impute (most_frequent) + OneHotEncoder(handle_unknown="ignore")

## 🆚 Models compared:

- Logistic Regression

- Random Forest

- CatBoost (best)

- MLP (Neural Net)

Threshold tuning on validation set (optimize F1 for >50K), then final evaluation on test.

## 📊 Results

Model comparison

          Model  Accuracy  Precision  Recall      F1  ROC-AUC  PR-AUC
2      CatBoost    0.8735     0.7315  0.7449  0.7381   0.9316  0.8336
1  RandomForest    0.8602     0.6872  0.7637  0.7234   0.9218  0.8149
3           MLP    0.8503     0.6592  0.7751  0.7125   0.9193  0.7969
0        LogReg    0.8438     0.6425  0.7837  0.7061   0.9150  0.7868


Chosen best model: CatBoost

- Accuracy: 0.873

- Precision (>50K): 0.732

- Recall (>50K): 0.745

- F1 (>50K): 0.738

- ROC-AUC: 0.932, PR-AUC: 0.834

## 🔍 Interpretability

- Permutation Importance (aggregated to original features):
Top drivers included education-num, hours-per-week, age, capital-gain / loss, and marital status simplification.

- SHAP (CatBoost):
Confirmed education-num, hours-per-week, and age as strong contributors; positive capital-gain pushes probability upward.

## 💡 Practical insights

- Policies that encourage skills/education and professional roles align with higher earning potential.

- Longer work hours correlate with >50K, but this should be balanced with well-being.

- High capital-gain events are strong signals; risk-based rules might flag such records for manual review in a decision workflow.

## 🧪 Error Analysis (CatBoost)

- False Positives: often managerial/professional roles with moderate losses or gov/private work; predicted >50K but actual <=50K.

- False Negatives: younger workers, service/moving roles, or low capital-gain, predicted <=50K though actual >50K.
Action: targeted thresholding or cost-sensitive adjustment depending on downstream use (e.g., minimize FN if missing high-income customers is costly).

## 🚀 Running the API (Docker)

Packaging & Project Layout:

- Created a lightweight inference service with FastAPI and Uvicorn.
- Service code under service/: 
    
    - app.py — API server, computes the same feature engineering used in training for 
      example, net_capital, has_capital, marital_simple, age_group, and aligns columns 
      to the pipeline, loads model and tuned thresholds, and serves predictions. 
    - models.joblib — serialized dictionary of trained pipelines (preprocess + estimator). 
    - thresholds.json — per-model decision thresholds tuned on the validation set. 
    - requirements.txt —Python libraries for reproducible builds. 
    - Dockerfile — container recipe to build a self-contained image.

 🔴 Note: The trained model file model.joblib is not included in this repository due to its large file size.

- Versioning: Git tags for releases (e.g., v1.0.0)

🎓 Acadamice context: This repository documents one of the applied projects completed for the Machine Learning module in my MSc in Data Science programme (Octorber 2025). All results, code, and design choices are provided for educational purposes; actual performance may vary across environments and dataset revisions.
