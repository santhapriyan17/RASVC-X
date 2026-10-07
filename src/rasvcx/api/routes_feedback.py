"""Feedback route for RASVC-X.

POST /feedback
    Records a user's rating of one /query response.

GET /feedback/summary
    Counts of recorded feedback by rating and decision.

What is stored (one JSON line per event, append-only):
    feedback_id, timestamp, query_id, rating, decision, kb_version_id,
    execution_mode, llm_model, config (retrieval mode / reranker / NLI)
    and, optionally, a short reason category.

What is deliberately NOT stored: the question text, the answer text, the
evidence, and free-text comments.  A medical question can identify a
person; an evaluation pipeline that needs the content can join on
query_id against a store with its own access controls.

Feedback never changes production behaviour.  It is written to a file
for offline review and evaluation; nothing in the query path reads it,
and there is no automatic retraining, re-ranking or threshold update.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from rasvcx.api.dependencies import get_settings, rate_limit, verify_auth
from rasvcx.config.settings import Settings
from rasvcx.schemas.decision import CANONICAL_DECISIONS

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/feedback", tags=["feedback"])

_DEFAULT_FEEDBACK_PATH = "data/feedback/feedback.jsonl"
_write_lock = threading.Lock()

REASONS = (
    "incorrect",
    "unsupported_claim",
    "wrong_citation",
    "missing_information",
    "outdated",
    "should_have_answered",
    "should_have_abstained",
    "other",
)


class FeedbackRequest(BaseModel):
    query_id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9._:\-]+$")
    rating: Literal["up", "down"]
    decision: str = Field(description="Canonical decision of the rated response.")
    kb_version_id: str | None = Field(default=None, max_length=64)
    reason: str | None = Field(default=None, description=f"One of: {', '.join(REASONS)}")


class FeedbackResponse(BaseModel):
    feedback_id: str
    recorded: bool = True
    feedback_collected: bool = True
    learning_loop: bool = False
    """Feedback is collected for offline review only; nothing is learned from it."""


def _feedback_path() -> Path:
    return Path(os.environ.get("RASVCX_FEEDBACK_PATH", _DEFAULT_FEEDBACK_PATH))


@router.post("", response_model=FeedbackResponse, status_code=201)
def submit_feedback(
    body: FeedbackRequest,
    settings: Annotated[Settings, Depends(get_settings)],
    _auth: Annotated[None, Depends(verify_auth)],
    _rate: Annotated[None, Depends(rate_limit)],
) -> FeedbackResponse:
    if body.decision not in CANONICAL_DECISIONS.values():
        raise HTTPException(
            status_code=422,
            detail=f"decision must be one of {sorted(CANONICAL_DECISIONS.values())}",
        )
    if body.reason is not None and body.reason not in REASONS:
        raise HTTPException(status_code=422, detail=f"reason must be one of {list(REASONS)}")

    event = {
        "feedback_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query_id": body.query_id,
        "rating": body.rating,
        "reason": body.reason,
        "decision": body.decision,
        "kb_version_id": body.kb_version_id,
        "execution_mode": settings.execution_mode.value,
        "llm_model": None if settings.llm.provider == "stub" else settings.llm.model_name,
        "config": {
            "retrieval_mode": settings.retrieval.mode,
            "reranker": settings.reranker.enabled,
            "nli": settings.nli.enabled,
        },
    }
    path = _feedback_path()
    try:
        with _write_lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(event, sort_keys=True) + "\n")
    except OSError as exc:
        logger.error("feedback could not be written: %s", exc)
        raise HTTPException(status_code=503, detail="Feedback store unavailable") from exc
    return FeedbackResponse(feedback_id=event["feedback_id"])


@router.get("/summary")
def feedback_summary(
    _auth: Annotated[None, Depends(verify_auth)],
) -> dict[str, Any]:
    path = _feedback_path()
    by_rating: dict[str, int] = {}
    by_decision: dict[str, dict[str, int]] = {}
    total = 0
    malformed = 0
    if path.is_file():
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                    rating = event["rating"]
                    decision = event["decision"]
                except (ValueError, KeyError, TypeError):
                    malformed += 1
                    continue
                total += 1
                by_rating[rating] = by_rating.get(rating, 0) + 1
                bucket = by_decision.setdefault(decision, {})
                bucket[rating] = bucket.get(rating, 0) + 1
    return {
        "total": total,
        "by_rating": by_rating,
        "by_decision": by_decision,
        "malformed_lines": malformed,
    }


__all__ = ["router"]
