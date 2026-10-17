"""Request and response models for the prediction API.

Validation ranges are deliberately wide (they reject typos and unit errors,
not unusual patients). Category codes match the UCI documentation exactly.
``ca`` and ``thal`` are optional because they are genuinely missing in the
source data; the model's imputers handle them.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class Patient(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "age": 61, "sex": 1, "cp": 4, "trestbps": 140, "chol": 207,
                "fbs": 0, "restecg": 2, "thalach": 138, "exang": 1,
                "oldpeak": 1.9, "slope": 1, "ca": 1, "thal": 7,
            }
        },
    )

    age: int = Field(..., ge=18, le=110, description="Age in years")
    sex: Literal[0, 1] = Field(..., description="1 = male, 0 = female")
    cp: Literal[1, 2, 3, 4] = Field(
        ..., description="Chest pain: 1 typical angina, 2 atypical, 3 non-anginal, 4 asymptomatic")
    trestbps: float = Field(..., ge=60, le=250, description="Resting blood pressure (mm Hg)")
    chol: float = Field(..., ge=80, le=700, description="Serum cholesterol (mg/dl)")
    fbs: Literal[0, 1] = Field(..., description="Fasting blood sugar > 120 mg/dl")
    restecg: Literal[0, 1, 2] = Field(
        ..., description="Resting ECG: 0 normal, 1 ST-T abnormality, 2 LV hypertrophy")
    thalach: float = Field(..., ge=50, le=230, description="Max heart rate achieved")
    exang: Literal[0, 1] = Field(..., description="Exercise-induced angina")
    oldpeak: float = Field(..., ge=-3, le=10, description="ST depression induced by exercise")
    slope: Literal[1, 2, 3] = Field(
        ..., description="Slope of peak exercise ST: 1 up, 2 flat, 3 down")
    ca: Optional[Literal[0, 1, 2, 3]] = Field(
        None, description="Major vessels coloured by fluoroscopy (optional)")
    thal: Optional[Literal[3, 6, 7]] = Field(
        None, description="Thallium test: 3 normal, 6 fixed defect, 7 reversible (optional)")


class BatchRequest(BaseModel):
    patients: list[Patient] = Field(..., min_length=1, max_length=100)


class Prediction(BaseModel):
    prediction: Literal[0, 1]
    label: Literal["heart_disease", "no_heart_disease"]
    probability_disease: float = Field(..., ge=0, le=1)
    confidence: float = Field(..., ge=0, le=1,
                              description="Probability of the predicted class")


class PredictionResponse(Prediction):
    # allow field names starting with "model_" without pydantic warnings
    model_config = ConfigDict(protected_namespaces=())

    api_version: str = Field(..., description="Response schema version")
    model_version: str
    decision_threshold: float
    request_id: str


class BatchResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    predictions: list[Prediction]
    api_version: str
    model_version: str
    decision_threshold: float
    request_id: str
