"""Small local HTTP interface for delayed label submissions and Stage 2 counts."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field, field_validator

from src.labels.store import (LabelValidationError, SubmissionIdConflict, UnauthorizedLabelSource,
                              get_label, get_prediction, get_stats, get_submissions, initialize, submit_label)

app = FastAPI(title="Fraud Detection Delayed Label API", version="Stage 2")


class LabelSubmission(BaseModel):
    transaction_id: str = Field(min_length=1, max_length=200)
    outcome: Literal["fraud", "legitimate"]
    label_source: str = Field(min_length=1, max_length=200)
    source_type: Literal["trusted_verified", "untrusted", "synthetic_test"]
    confirmed_at: datetime
    submission_id: str = Field(min_length=8, max_length=100)

    @field_validator("confirmed_at")
    @classmethod
    def timezone_required(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("confirmed_at must include a timezone")
        return value.astimezone(timezone.utc)


@app.on_event("startup")
def startup() -> None:
    initialize()


@app.get("/health")
def health() -> dict:
    counts = get_stats()
    return {"status": "healthy", "genuine_source_configured": counts["genuine_source_configured"],
            "genuine_labels_available": counts["genuine_labels_available"]}


@app.get("/stats")
def stats() -> dict:
    return get_stats()


@app.get("/predictions/{transaction_id}")
def prediction(transaction_id: str) -> dict:
    result = get_prediction(transaction_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Prediction not registered")
    return result


@app.get("/labels/{transaction_id}")
def label(transaction_id: str) -> dict:
    result = get_label(transaction_id)
    if result is None:
        raise HTTPException(status_code=404, detail="No label submitted")
    return result


@app.get("/submissions/{transaction_id}")
def submissions(transaction_id: str) -> list[dict]:
    return get_submissions(transaction_id)


@app.post("/labels", status_code=201)
def submit(request: LabelSubmission, x_label_token: str | None = Header(default=None)) -> dict:
    try:
        return submit_label(transaction_id=request.transaction_id, outcome=request.outcome,
                            label_source=request.label_source, source_type=request.source_type,
                            confirmed_at=request.confirmed_at.isoformat(), submission_id=request.submission_id,
                            authorization_token=x_label_token)
    except UnauthorizedLabelSource as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except SubmissionIdConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except LabelValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
