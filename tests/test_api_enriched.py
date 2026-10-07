"""Tests for M16 enriched API — models_enriched, routes_eval, and
enriched=True query responses — File 23/38.

All tests are synchronous via TestClient.
Uses the same _mocked_client pattern as test_api.py.
"""

from __future__ import annotations

import json
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fastapi.testclient import TestClient

from rasvcx.api.main import create_app
from rasvcx.api.models_enriched import (
    ClaimVerificationModel,
    ConflictResolutionModel,
    EnrichedQueryResponse,
    EvalRunListModel,
    EvalRunSummaryModel,
    RiskFeatureScoresModel,
    RiskProfileEnrichedModel,
    ValidationSummaryModel,
    VerificationSummaryModel,
    pipeline_result_to_enriched,
    run_record_to_summary,
)
from rasvcx.config.settings import Settings
from rasvcx.pipeline.pipeline_result import PipelineResult
from rasvcx.retrieval.bridge import DisabledRerankingService
from rasvcx.schemas.common import (
    CandidateId,
    ClaimId,
    EvidenceRelationship,
    QueryId,
)
from rasvcx.schemas.decision import Decision, DecisionAction
from rasvcx.schemas.query import RiskFeatureScores, RiskProfile, ValidationDepth
from rasvcx.schemas.verification import (
    AnswerVerdict,
    ClaimVerificationResult,
    SupportLabel,
    VerificationStage,
    VerificationSummary,
)
from rasvcx.validation.verified_context import (
    EvidenceRelationshipResult,
    ValidationSummary,
)

_KEY_PIPELINE = "rasvcx_pipeline"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_verification_summary(
    verdict: AnswerVerdict = AnswerVerdict.VERIFIED,
    confidence: float = 0.85,
    n_claims: int = 1,
) -> VerificationSummary:
    claims = [
        ClaimVerificationResult(
            claim_id=ClaimId(f"c{i}"),
            label=SupportLabel.SUPPORTED,
            confidence=confidence,
            stage=VerificationStage.DETERMINISTIC,
            reason_code=None,
            rationale="test rationale",
            supporting_item_ids=[],
            contradicting_item_ids=[],
        )
        for i in range(n_claims)
    ]
    return VerificationSummary(
        answer_verdict=verdict,
        overall_confidence=confidence,
        claim_results=claims,
        citation_results=[],
        orphan_citation_ids=[],
        semantic_verification_calls=0,
        safety_critical_failure_count=0,
        budget_exhausted=False,
    )


def _make_validation_summary(
    n_conflicts: int = 0,
) -> ValidationSummary:
    resolutions = [
        EvidenceRelationshipResult(
            candidate_id=CandidateId(f"pair_{i}"),
            relationship=(
                EvidenceRelationship.GENUINE_CONFLICT
                if i % 2 == 0
                else EvidenceRelationship.COMPATIBLE
            ),
            confidence=0.8,
            contributing_claim_ids=[],
            rationale="test",
        )
        for i in range(n_conflicts)
    ]
    return ValidationSummary(
        candidates_generated=n_conflicts,
        validation_results={},
        resolutions=resolutions,
        nli_calls_used=n_conflicts,
    )


def _make_risk_profile(
    score: float = 0.5,
    depth: ValidationDepth = ValidationDepth.STANDARD,
) -> RiskProfile:
    return RiskProfile(
        overall_risk_score=score,
        feature_scores=RiskFeatureScores(),
        validation_depth=depth,
        retrieval_retry_budget=1,
        nli_call_allowance=3,
    )


def _make_pipeline_result(
    action: DecisionAction = DecisionAction.ANSWER,
    confidence: float = 0.85,
    generated_text: str = "Visiting hours are 9am to 8pm.",
    with_verification: bool = True,
    with_validation: bool = True,
    with_risk: bool = True,
) -> PipelineResult:
    vs = _make_verification_summary() if with_verification else None
    vsum = _make_validation_summary(n_conflicts=1) if with_validation else None
    rp = _make_risk_profile() if with_risk else None
    return PipelineResult(
        query_id=QueryId("test-enriched-id"),
        decision=Decision(
            action=action,
            confidence=confidence,
            rationale="test rationale",
        ),
        generated_text=generated_text,
        corrective_attempts=0,
        verification_summary=vs,
        validation_summary=vsum,
        risk_profile=rp,
    )


def _make_mock_pipeline(result: PipelineResult):
    from rasvcx.pipeline.orchestrator import PipelineOrchestrator
    mock = MagicMock(spec=PipelineOrchestrator)
    mock.run.return_value = result
    mock._reranking = DisabledRerankingService()
    mock._validation = None
    mock._sufficiency_config = None
    return mock


@contextmanager
def _mocked_client(result: PipelineResult, settings: Settings | None = None):
    s = settings or Settings()
    app = create_app(settings=s)
    with TestClient(app, raise_server_exceptions=False) as client:
        setattr(app.state, _KEY_PIPELINE, _make_mock_pipeline(result))
        yield client


# ---------------------------------------------------------------------------
# EnrichedQueryResponse model unit tests
# ---------------------------------------------------------------------------

class TestEnrichedQueryResponseModel:
    def test_construction_minimal(self):
        r = EnrichedQueryResponse(
            query_id="q1",
            success=True,
            decision="ANSWER",
            action="answer",
            confidence=0.85,
        )
        assert r.query_id == "q1"
        assert r.action == "answer"
        assert r.mock_llm is True
        assert r.retrieved_chunk_ids == []

    def test_limitations_always_present(self):
        # The notice is produced by the converter and depends on whether a
        # mock LLM answered: an offline answer must say so explicitly.
        mock = pipeline_result_to_enriched(_make_pipeline_result(), mock_llm=True)
        real = pipeline_result_to_enriched(_make_pipeline_result(), mock_llm=False)
        assert "MockLLMClient" in mock.limitations
        assert "OFFLINE TEST MODE" in mock.limitations
        assert "MockLLMClient" not in real.limitations
        assert "medical advice" in real.limitations

    def test_json_serialisable(self):
        r = EnrichedQueryResponse(
            query_id="q1", success=True, decision="ANSWER", action="answer",
            confidence=0.5,
        )
        d = r.model_dump()
        json.dumps(d)

    def test_verification_none_by_default(self):
        r = EnrichedQueryResponse(
            query_id="q1", success=True, decision="ABSTAIN", action="abstain",
            confidence=0.2,
        )
        assert r.verification is None
        assert r.validation is None
        assert r.risk_profile is None


class TestVerificationSummaryModel:
    def test_construction(self):
        cm = ClaimVerificationModel(
            claim_id="c1", label="supported",
            confidence=0.9, stage="deterministic",
        )
        vsm = VerificationSummaryModel(
            answer_verdict="verified",
            overall_confidence=0.9,
            claim_results=[cm],
        )
        assert vsm.answer_verdict == "verified"
        assert len(vsm.claim_results) == 1

    def test_confidence_bounds(self):
        with pytest.raises(Exception):
            VerificationSummaryModel(
                answer_verdict="verified",
                overall_confidence=1.5,
            )


class TestValidationSummaryModel:
    def test_construction(self):
        cr = ConflictResolutionModel(
            candidate_id="pair_1",
            relationship="genuine-conflict",
            confidence=0.8,
        )
        vsm = ValidationSummaryModel(
            candidates_generated=1,
            nli_calls_used=1,
            genuine_conflict_count=1,
            unresolved_count=0,
            resolutions=[cr],
        )
        assert vsm.genuine_conflict_count == 1
        assert len(vsm.resolutions) == 1


class TestRiskProfileEnrichedModel:
    def test_construction(self):
        fs = RiskFeatureScoresModel()
        rp = RiskProfileEnrichedModel(
            overall_risk_score=0.6,
            validation_depth="standard",
            retrieval_retry_budget=1,
            nli_call_allowance=3,
            feature_scores=fs,
        )
        assert rp.validation_depth == "standard"
        assert rp.feature_scores.numeric_content_score == 0.0


# ---------------------------------------------------------------------------
# pipeline_result_to_enriched conversion
# ---------------------------------------------------------------------------

class TestPipelineResultToEnriched:
    def test_answer_action(self):
        result = _make_pipeline_result(action=DecisionAction.ANSWER)
        enriched = pipeline_result_to_enriched(result, mock_llm=True)
        assert enriched.action == "answer"
        assert enriched.has_answer is True

    def test_abstain_action(self):
        result = _make_pipeline_result(action=DecisionAction.ABSTAIN)
        enriched = pipeline_result_to_enriched(result, mock_llm=True)
        assert enriched.action == "abstain"
        assert enriched.has_answer is False

    def test_warning_has_answer(self):
        result = _make_pipeline_result(action=DecisionAction.WARNING)
        enriched = pipeline_result_to_enriched(result, mock_llm=True)
        assert enriched.has_answer is True

    def test_verification_populated(self):
        result = _make_pipeline_result(with_verification=True)
        enriched = pipeline_result_to_enriched(result, mock_llm=True)
        assert enriched.verification is not None
        assert enriched.verification.answer_verdict == "verified"
        assert len(enriched.verification.claim_results) == 1

    def test_verification_none_when_absent(self):
        result = _make_pipeline_result(with_verification=False)
        enriched = pipeline_result_to_enriched(result, mock_llm=True)
        assert enriched.verification is None

    def test_validation_populated(self):
        result = _make_pipeline_result(with_validation=True)
        enriched = pipeline_result_to_enriched(result, mock_llm=True)
        assert enriched.validation is not None
        assert enriched.validation.candidates_generated == 1

    def test_validation_none_when_absent(self):
        result = _make_pipeline_result(with_validation=False)
        enriched = pipeline_result_to_enriched(result, mock_llm=True)
        assert enriched.validation is None

    def test_risk_profile_populated(self):
        result = _make_pipeline_result(with_risk=True)
        enriched = pipeline_result_to_enriched(result, mock_llm=True)
        assert enriched.risk_profile is not None
        assert enriched.risk_profile.validation_depth == "standard"

    def test_retrieved_chunk_ids_always_empty(self):
        result = _make_pipeline_result()
        enriched = pipeline_result_to_enriched(result, mock_llm=True)
        assert enriched.retrieved_chunk_ids == []

    def test_mock_llm_flagged(self):
        result = _make_pipeline_result()
        enriched = pipeline_result_to_enriched(result, mock_llm=True)
        assert enriched.mock_llm is True

    def test_request_id_passed_through(self):
        result = _make_pipeline_result()
        enriched = pipeline_result_to_enriched(
            result, request_id="req-abc", mock_llm=True
        )
        assert enriched.request_id == "req-abc"

    def test_json_serialisable(self):
        result = _make_pipeline_result()
        enriched = pipeline_result_to_enriched(result, mock_llm=True)
        d = enriched.model_dump()
        json.dumps(d)

    def test_conflict_resolutions_populated(self):
        result = _make_pipeline_result(with_validation=True)
        enriched = pipeline_result_to_enriched(result, mock_llm=True)
        assert enriched.validation is not None
        assert len(enriched.validation.resolutions) == 1
        assert enriched.validation.resolutions[0].relationship == "genuine-conflict"

    def test_genuine_conflict_count(self):
        result = _make_pipeline_result(with_validation=True)
        enriched = pipeline_result_to_enriched(result, mock_llm=True)
        assert enriched.validation.genuine_conflict_count == 1


# ---------------------------------------------------------------------------
# POST /query with enriched=True
# ---------------------------------------------------------------------------

class TestEnrichedQueryRoute:
    def test_enriched_false_returns_thin_response(self):
        result = _make_pipeline_result()
        with _mocked_client(result) as client:
            resp = client.post("/query", json={"query": "What are visiting hours?"})
        assert resp.status_code == 200
        data = resp.json()
        assert "decision" in data
        assert "action" not in data

    def test_enriched_true_returns_enriched_response(self):
        result = _make_pipeline_result()
        with _mocked_client(result) as client:
            resp = client.post(
                "/query",
                json={"query": "What are visiting hours?", "enriched": True},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert "action" in data
        assert "verification" in data
        assert "validation" in data
        assert "risk_profile" in data

    def test_enriched_true_limitations_present(self):
        result = _make_pipeline_result()
        with _mocked_client(result) as client:
            resp = client.post(
                "/query",
                json={"query": "test query", "enriched": True},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert "limitations" in data
        assert len(data["limitations"]) > 0

    def test_enriched_true_mock_llm_flagged(self):
        result = _make_pipeline_result()
        with _mocked_client(result) as client:
            resp = client.post(
                "/query",
                json={"query": "test query", "enriched": True},
            )
        assert resp.status_code == 200
        assert resp.json()["mock_llm"] is True

    def test_enriched_true_retrieved_chunk_ids_empty(self):
        result = _make_pipeline_result()
        with _mocked_client(result) as client:
            resp = client.post(
                "/query",
                json={"query": "test query", "enriched": True},
            )
        assert resp.status_code == 200
        assert resp.json()["retrieved_chunk_ids"] == []

    def test_enriched_true_action_present(self):
        result = _make_pipeline_result(action=DecisionAction.ABSTAIN)
        with _mocked_client(result) as client:
            resp = client.post(
                "/query",
                json={"query": "test query", "enriched": True},
            )
        assert resp.status_code == 200
        assert resp.json()["action"] == "abstain"

    def test_enriched_default_is_false(self):
        result = _make_pipeline_result()
        with _mocked_client(result) as client:
            resp = client.post(
                "/query",
                json={"query": "test query"},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert "decision" in data


# ---------------------------------------------------------------------------
# GET /eval/runs routes
# ---------------------------------------------------------------------------

class TestEvalRoutes:
    def _write_run_record(self, output_dir: Path, run_id: str) -> Path:
        data = {
            "run_id": run_id,
            "baseline_id": "B3_RASVCX",
            "dataset_id": "ds_test",
            "dataset_version": "1.0.0",
            "split": "test",
            "corpus_fingerprint": "abc123def456abcd",
            "corpus_condition": "clean",
            "execution_mode": "offline_test",
            "mock_llm": True,
            "accounting": {
                "total_cases": 10,
                "offered": 10,
                "accepted": 10,
                "completed": 9,
                "error": 1,
                "skipped": 0,
                "rejected_overload": 0,
                "cancelled_deadline": 0,
                "not_offered_deadline": 0,
            },
            "timing": {
                "baseline_init_seconds": 0.5,
                "total_run_wall_seconds": 5.0,
                "start_utc": "2026-10-01T10:00:00Z",
                "end_utc": "2026-10-01T10:00:05Z",
            },
            "results_jsonl_path": str(output_dir / f"cases_{run_id}.jsonl"),
            "integrity_error": None,
        }
        path = output_dir / f"run_{run_id}.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def _write_jsonl(self, output_dir: Path, run_id: str, n: int = 3) -> Path:
        path = output_dir / f"cases_{run_id}.jsonl"
        lines = [
            json.dumps({
                "case_id": f"c{i}",
                "status": "completed",
                "baseline_id": "B3_RASVCX",
            })
            for i in range(n)
        ]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def test_list_runs_empty(self, tmp_path, monkeypatch):
        monkeypatch.setenv("RASVCX_EVAL_OUTPUT_DIR", str(tmp_path))
        app = create_app(settings=Settings())
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.get("/eval/runs")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0
        assert data["runs"] == []

    def test_list_runs_returns_summaries(self, tmp_path, monkeypatch):
        monkeypatch.setenv("RASVCX_EVAL_OUTPUT_DIR", str(tmp_path))
        self._write_run_record(tmp_path, "run001")
        self._write_run_record(tmp_path, "run002")
        app = create_app(settings=Settings())
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.get("/eval/runs")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2
        assert len(data["runs"]) == 2

    def test_get_run_by_id(self, tmp_path, monkeypatch):
        monkeypatch.setenv("RASVCX_EVAL_OUTPUT_DIR", str(tmp_path))
        self._write_run_record(tmp_path, "run001")
        app = create_app(settings=Settings())
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.get("/eval/runs/run001")
        assert resp.status_code == 200
        data = resp.json()
        assert data["run_id"] == "run001"
        assert data["baseline_id"] == "B3_RASVCX"

    def test_get_run_404(self, tmp_path, monkeypatch):
        monkeypatch.setenv("RASVCX_EVAL_OUTPUT_DIR", str(tmp_path))
        app = create_app(settings=Settings())
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.get("/eval/runs/nonexistent_run_id")
        assert resp.status_code == 404

    def test_stream_cases(self, tmp_path, monkeypatch):
        monkeypatch.setenv("RASVCX_EVAL_OUTPUT_DIR", str(tmp_path))
        self._write_run_record(tmp_path, "run001")
        self._write_jsonl(tmp_path, "run001", n=3)
        app = create_app(settings=Settings())
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.get("/eval/runs/run001/cases")
        assert resp.status_code == 200
        lines = [l for l in resp.text.strip().split("\n") if l.strip()]
        assert len(lines) == 3
        for line in lines:
            obj = json.loads(line)
            assert "case_id" in obj

    def test_stream_cases_404_missing_jsonl(self, tmp_path, monkeypatch):
        monkeypatch.setenv("RASVCX_EVAL_OUTPUT_DIR", str(tmp_path))
        self._write_run_record(tmp_path, "run001")
        app = create_app(settings=Settings())
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.get("/eval/runs/run001/cases")
        assert resp.status_code == 404

    def test_list_runs_pagination(self, tmp_path, monkeypatch):
        monkeypatch.setenv("RASVCX_EVAL_OUTPUT_DIR", str(tmp_path))
        for i in range(5):
            self._write_run_record(tmp_path, f"run{i:03d}")
        app = create_app(settings=Settings())
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.get("/eval/runs?limit=2&offset=0")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 5
        assert len(data["runs"]) == 2

    def test_run_summary_mock_llm_field(self, tmp_path, monkeypatch):
        monkeypatch.setenv("RASVCX_EVAL_OUTPUT_DIR", str(tmp_path))
        self._write_run_record(tmp_path, "run001")
        app = create_app(settings=Settings())
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.get("/eval/runs/run001")
        assert resp.status_code == 200
        assert resp.json()["mock_llm"] is True

    def test_run_summary_limitations_present(self, tmp_path, monkeypatch):
        monkeypatch.setenv("RASVCX_EVAL_OUTPUT_DIR", str(tmp_path))
        self._write_run_record(tmp_path, "run001")
        app = create_app(settings=Settings())
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.get("/eval/runs/run001")
        assert resp.status_code == 200
        data = resp.json()
        assert "limitations" in data
        assert "MockLLMClient" in data["limitations"]