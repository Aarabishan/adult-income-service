# Adult Income Classification

End-to-end pipeline for predicting whether a person’s income is >50K using the UCI Adult Census dataset.
Includes preprocessing, model training & comparison, explainability (Permutation Importance + SHAP), and a production-style FastAPI service containerized with Docker.

## Project Objectives

- Build a reproducible ML pipeline for binary classification (<=50K vs >50K).

- Compare four model families: Logistic Regression, Random Forest, CatBoost, MLP.

- Apply threshold tuning on a validation set and evaluate on a held-out test set.

- Provide interpretability (Permutation Importance, SHAP) and error analysis.

- Package a FastAPI inference service and run it via Docker.
