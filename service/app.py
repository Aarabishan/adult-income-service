# app.py
# FastAPI service for Adult Income model (multi-model with thresholds)
# - Computes engineered features at inference (net_capital, has_capital, etc.)
# - Aligns request columns to the pipeline's expected inputs
# - Predicts probability and applies tuned threshold

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# --------------------------------------------------------------------------------------
# Paths (match what you COPY into the Docker image in your Dockerfile)
# --------------------------------------------------------------------------------------
MODELS_PATH = os.getenv("MODELS_PATH", "models.joblib")        # dict[str, sklearn Pipeline]
THRESH_PATH = os.getenv("THRESH_PATH", "thresholds.json")      # {"CatBoost": 0.47, ...}
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "CatBoost")

# --------------------------------------------------------------------------------------
# Load artifacts
# --------------------------------------------------------------------------------------
try:
    models: Dict[str, Any] = joblib.load(MODELS_PATH)
    if not isinstance(models, dict) or not models:
        raise ValueError("models.joblib did not contain a non-empty dict of models")
except Exception as e:
    raise RuntimeError(f"Failed to load models from {MODELS_PATH}: {e}")

try:
    with open(THRESH_PATH, "r") as f:
        thresholds: Dict[str, float] = json.load(f)
except Exception:
    thresholds = {}  # fallback to 0.5 later

# --------------------------------------------------------------------------------------
# FastAPI app
# --------------------------------------------------------------------------------------
app = FastAPI(title="Adult Income Inference API", version="1.0.0")


# --------------------------------------------------------------------------------------
# Request/Response schema
# (Keep it simple: we accept a free-form record dict; FE happens server-side)
# --------------------------------------------------------------------------------------
class PredictPayload(BaseModel):
    model_name: Optional[str] = Field(
        default=None, description="One of the trained models (e.g., 'CatBoost'). If omitted, uses DEFAULT_MODEL."
    )
    record: Dict[str, Any] = Field(
        ...,
        description="Raw Adult record (e.g., age, workclass, education, education-num, capital-gain/loss, etc.)."
    )


# --------------------------------------------------------------------------------------
# Feature engineering (must mirror your training-time features)
# --------------------------------------------------------------------------------------
def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    # 1) net_capital = capital-gain - capital-loss
    if "net_capital" not in df.columns:
        cg = df.get("capital-gain", 0)
        cl = df.get("capital-loss", 0)
        df["capital-gain"] = pd.to_numeric(cg, errors="coerce").fillna(0)
        df["capital-loss"] = pd.to_numeric(cl, errors="coerce").fillna(0)
        df["net_capital"] = df["capital-gain"] - df["capital-loss"]

    # 2) has_capital (binary)
    if "has_capital" not in df.columns:
        cg = pd.to_numeric(df.get("capital-gain", 0), errors="coerce").fillna(0)
        cl = pd.to_numeric(df.get("capital-loss", 0), errors="coerce").fillna(0)
        df["has_capital"] = ((cg > 0) | (cl > 0)).astype(int)

    # 3) is_us (binary)
    if "is_us" not in df.columns:
        nat = df.get("native-country", pd.Series([""] * len(df)))
        nat = nat.astype(str).str.strip()
        df["is_us"] = (nat == "United-States").astype(int)

    # 4) marital_simple (1=married, 0=not)
    if "marital_simple" not in df.columns:
        married_set = {"Married-civ-spouse", "Married-AF-spouse"}
        ms = df.get("marital-status", pd.Series([""] * len(df))).astype(str).str.strip()
        df["marital_simple"] = np.where(ms.isin(married_set), 1, 0)

    # 5) age_group (categorical)
    if "age_group" not in df.columns:
        age = pd.to_numeric(df.get("age", 0), errors="coerce")
        df["age_group"] = pd.cut(
            age,
            bins=[0, 25, 35, 45, 55, 100],
            labels=["Young", "Adult", "Middle-aged", "Senior", "Elderly"],
            include_lowest=True,
            ordered=True,
        )

    # 6) work_hours_category (categorical)
    if "work_hours_category" not in df.columns:
        h = pd.to_numeric(df.get("hours-per-week", 0), errors="coerce")
        df["work_hours_category"] = pd.cut(
            h,
            bins=[0, 20, 40, 60, 100],
            labels=["Part-time", "Full-time", "Overtime", "Extreme"],
            include_lowest=True,
            ordered=True,
        )

    # 7) education_level (Low/Medium/High)
    if "education_level" not in df.columns:
        education_mapping = {
            "Preschool": "Low", "1st-4th": "Low", "5th-6th": "Low", "7th-8th": "Low",
            "9th": "Low", "10th": "Low", "11th": "Medium", "12th": "Medium",
            "HS-grad": "Medium", "Some-college": "High", "Assoc-voc": "High",
            "Assoc-acdm": "High", "Bachelors": "High", "Masters": "High",
            "Prof-school": "High", "Doctorate": "High"
        }
        edu = df.get("education", pd.Series([""] * len(df))).astype(str).str.strip()
        df["education_level"] = edu.map(education_mapping)

    return df


def align_to_preprocessor_inputs(pipe: Any, df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures df has the exact columns (names) the pipeline's preprocessor expects.
    Adds missing columns as NaN; extra columns are ignored by ColumnTransformer.
    """
    pre = getattr(pipe, "named_steps", {}).get("preprocess", None)
    if pre is None:
        # No preprocessor? Return df as-is.
        return df

    # sklearn >= 1.0 exposes feature_names_in_ on transformers after .fit
    expected_cols = getattr(pre, "feature_names_in_", None)
    if expected_cols is None:
        # Fallback: try ColumnTransformer's remainder or assume current df is OK
        return df

    expected = list(expected_cols)
    missing = set(expected) - set(df.columns)
    if missing:
        for c in missing:
            df[c] = np.nan
        # order columns as expected
        df = df[expected]
    else:
        df = df[expected]
    return df


def pick_threshold(model_name: str) -> float:
    # Use tuned threshold if available; else 0.5
    return float(thresholds.get(model_name, 0.5))


def get_model(model_name: Optional[str]) -> tuple[str, Any]:
    name = (model_name or DEFAULT_MODEL).strip()
    if name not in models:
        raise HTTPException(status_code=400, detail=f"Unknown model '{name}'. Available: {list(models.keys())}")
    return name, models[name]


# --------------------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------------------
@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "models_available": list(models.keys()),
        "default_model": DEFAULT_MODEL,
    }


@app.get("/models")
def available_models() -> Dict[str, Any]:
    return {
        "models": list(models.keys()),
        "thresholds": thresholds,
        "default_model": DEFAULT_MODEL,
    }


@app.post("/predict")
def predict(payload: PredictPayload) -> Dict[str, Any]:
    # 1) pick model
    model_name, pipe = get_model(payload.model_name)

    # 2) make 1-row DataFrame from request and run FE
    try:
        df = pd.DataFrame([payload.record])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid record payload: {e}")

    df = apply_feature_engineering(df)

    # 3) align input columns to preprocessor expectations
    X = align_to_preprocessor_inputs(pipe, df)

    # 4) predict proba and apply threshold
    try:
        proba = pipe.predict_proba(X)[:, 1]
    except AttributeError:
        # Some models may not implement predict_proba; fallback to decision_function or predict
        if hasattr(pipe, "decision_function"):
            # map decision scores to [0,1] via logistic; this is a rough fallback
            from scipy.special import expit
            scores = pipe.decision_function(X)
            proba = expit(scores)
        else:
            preds = pipe.predict(X).astype(float)
            proba = np.clip(preds, 0.0, 1.0)

    prob = float(proba[0])
    thr = pick_threshold(model_name)
    pred = int(prob >= thr)

    return {
        "model": model_name,
        "threshold_used": thr,
        "prob_gt_50k": prob,
        "pred_gt_50k": pred,
        "engineered_columns_added": [c for c in ["net_capital", "has_capital", "is_us",
                                                 "marital_simple", "age_group",
                                                 "work_hours_category", "education_level"]
                                     if c in df.columns],
    }
