from pathlib import Path
from typing import Any, Literal
import os

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field, ConfigDict, field_validator

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = BASE_DIR / "model.pkl"
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH)))

app = FastAPI(
    title="D2C Churn Scoring API",
    version="1.0.0",
    description="Internal API for churn probability scoring and retention prioritization.",
)


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


ALLOWED_CITY_TIER = {"Tier 1", "Tier 2", "Tier 3"}
ALLOWED_AGE_GROUP = {"18-24", "25-34", "35-44", "45+"}
ALLOWED_CHANNEL = {"Google Search", "Instagram", "Influencer", "Referral", "Marketplace", "Organic"}
ALLOWED_LOYALTY = {"Silver", "Gold", "Platinum", "Unknown", None}
ALLOWED_CATEGORY = {"Skin Care", "Hair Care", "Makeup", "Fragrance", "Wellness", "Baby Care"}
ALLOWED_CONSENT = {"Yes", "No"}


class CustomerFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")

    customer_id: str | None = Field(default=None, description="Optional trace ID; not used as a model feature")
    city_tier: str
    age_group: str
    acquisition_channel: str
    loyalty_tier: str | None = None
    preferred_category: str
    marketing_consent: str
    recency_days: int = Field(ge=0)
    frequency_180d: int = Field(ge=0)
    monetary_180d: float = Field(ge=0)
    return_rate_180d: float = Field(ge=0, le=1)
    avg_discount_pct_180d: float = Field(ge=0, le=1)
    avg_rating_180d: float | None = Field(default=None, ge=1, le=5)
    category_diversity_180d: int = Field(ge=0)
    ticket_count_90d: int = Field(ge=0)
    negative_ticket_rate_90d: float = Field(ge=0, le=1)
    avg_resolution_hours_90d: float = Field(ge=0)
    days_since_signup: int = Field(ge=0)
    sessions_30d: int = Field(ge=0)
    product_views_30d: int = Field(ge=0)
    cart_adds_30d: int = Field(ge=0)
    wishlist_adds_30d: int = Field(ge=0)
    abandoned_carts_30d: int = Field(ge=0)
    email_opens_30d: int = Field(ge=0)
    campaign_clicks_30d: int = Field(ge=0)
    last_visit_days_ago: int = Field(ge=0)

    @field_validator("city_tier")
    @classmethod
    def validate_city_tier(cls, v: str) -> str:
        if v not in ALLOWED_CITY_TIER:
            raise ValueError(f"city_tier must be one of {sorted(ALLOWED_CITY_TIER)}")
        return v

    @field_validator("age_group")
    @classmethod
    def validate_age_group(cls, v: str) -> str:
        if v not in ALLOWED_AGE_GROUP:
            raise ValueError(f"age_group must be one of {sorted(ALLOWED_AGE_GROUP)}")
        return v

    @field_validator("acquisition_channel")
    @classmethod
    def validate_channel(cls, v: str) -> str:
        if v not in ALLOWED_CHANNEL:
            raise ValueError(f"acquisition_channel must be one of {sorted(ALLOWED_CHANNEL)}")
        return v

    @field_validator("loyalty_tier")
    @classmethod
    def validate_loyalty(cls, v: str | None) -> str | None:
        if v not in ALLOWED_LOYALTY:
            raise ValueError("loyalty_tier must be Silver, Gold, Platinum, Unknown, or null")
        return v

    @field_validator("preferred_category")
    @classmethod
    def validate_category(cls, v: str) -> str:
        if v not in ALLOWED_CATEGORY:
            raise ValueError(f"preferred_category must be one of {sorted(ALLOWED_CATEGORY)}")
        return v

    @field_validator("marketing_consent")
    @classmethod
    def validate_consent(cls, v: str) -> str:
        if v not in ALLOWED_CONSENT:
            raise ValueError("marketing_consent must be Yes or No")
        return v


class PredictionResponse(BaseModel):
    customer_id: str | None
    churn_probability: float
    predicted_class: Literal[0, 1]
    threshold: float
    risk_band: Literal["low", "medium", "high"]
    risk_explanation: str


def load_artifact() -> dict[str, Any]:
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model file not found at {MODEL_PATH}. Run `python src/train_model_for_api.py` first."
        )
    return joblib.load(MODEL_PATH)


ARTIFACT: dict[str, Any] | None = None


@app.on_event("startup")
def startup_load_model() -> None:
    global ARTIFACT
    ARTIFACT = load_artifact()


def get_artifact() -> dict[str, Any]:
    global ARTIFACT
    if ARTIFACT is None:
        ARTIFACT = load_artifact()
    return ARTIFACT


def to_model_frame(payload: CustomerFeatures, feature_columns: list[str]) -> pd.DataFrame:
    data = payload.model_dump()
    data.pop("customer_id", None)
    missing = [c for c in feature_columns if c not in data]
    if missing:
        raise HTTPException(status_code=422, detail=f"Missing model features: {missing}")
    return pd.DataFrame([{c: data.get(c) for c in feature_columns}])


def risk_band(probability: float) -> str:
    if probability >= 0.70:
        return "high"
    if probability >= 0.40:
        return "medium"
    return "low"


def explain(payload: CustomerFeatures, probability: float) -> str:
    reasons = []
    if payload.recency_days >= 90:
        reasons.append("last purchase is stale")
    if payload.sessions_30d <= 2 or payload.last_visit_days_ago >= 14:
        reasons.append("recent app/web engagement is low")
    if payload.return_rate_180d >= 0.25:
        reasons.append("return rate is high")
    if payload.ticket_count_90d >= 1 or payload.negative_ticket_rate_90d >= 0.5:
        reasons.append("recent support friction exists")
    if payload.abandoned_carts_30d >= 2:
        reasons.append("cart abandonment suggests purchase friction")
    if payload.frequency_180d >= 3 and payload.monetary_180d >= 1000:
        reasons.append("customer has high value, so review before aggressive discounting")
    if not reasons:
        reasons.append("risk is based on combined RFM, support, and engagement signals")
    return "; ".join(reasons) + f". Model probability={probability:.3f}."


def predict_one(payload: CustomerFeatures) -> PredictionResponse:
    artifact = get_artifact()
    model = artifact["model"]
    threshold = float(artifact["threshold"])
    feature_columns = artifact["feature_columns"]
    frame = to_model_frame(payload, feature_columns)
    prob = float(model.predict_proba(frame)[0, 1])
    pred = int(prob >= threshold)
    return PredictionResponse(
        customer_id=payload.customer_id,
        churn_probability=round(prob, 4),
        predicted_class=pred,
        threshold=round(threshold, 4),
        risk_band=risk_band(prob),
        risk_explanation=explain(payload, prob),
    )


@app.get("/health")
def health() -> dict[str, Any]:
    artifact = get_artifact()
    return {
        "status": "ok",
        "model_loaded": True,
        "snapshot_date": artifact.get("snapshot_date"),
        "feature_count": len(artifact.get("feature_columns", [])),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: CustomerFeatures) -> PredictionResponse:
    return predict_one(payload)


@app.post("/batch_predict", response_model=list[PredictionResponse])
def batch_predict(payloads: list[CustomerFeatures]) -> list[PredictionResponse]:
    if len(payloads) == 0:
        raise HTTPException(status_code=422, detail="At least one payload is required")
    if len(payloads) > 1000:
        raise HTTPException(status_code=413, detail="Batch size too large; maximum is 1000")
    return [predict_one(p) for p in payloads]
