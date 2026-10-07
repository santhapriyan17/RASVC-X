"""Comprehensive tests for the RASVC-X HTTP API (Module 14).

Coverage:
  - All routes: /health, /ready, /query, /admin/config
  - Middleware: X-Request-ID, X-Response-Time-Ms, 413 enforcement
  - Auth: require_auth=False and require_auth=True (missing/wrong/correct token)
  - Query validation: empty, whitespace, too-long, missing field, max-length edge
  - Pipeline result mapping: ANSWER, WARNING, ABSTAIN, pipeline_error
  - Readiness: offline_test component states
  - Secrets: api_key and auth_token never appear in any response
"""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from rasvcx.api.main import create_app
from rasvcx.config.settings import APISettings, Settings
from rasvcx.pipeline.pipeline_result import PipelineError, PipelineResult, make_error_result
from rasvcx.retrieval.bridge import DisabledRerankingService
from rasvcx.schemas.common import QueryId
from rasvcx.schemas.decision import Decision, DecisionAction

_KEY_PIPELINE = "rasvcx_pipeline"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(
    action: DecisionAction = DecisionAction.ANSWER,
    confidence: float = 0.85,
    generated_text: str = "Visiting hours are 9am to 8pm. [E1]",
    corrective_attempts: int = 0,
    pipeline_error: PipelineError | None = None,
) -> PipelineResult:
    if pipeline_error is not None:
        return make_error_result(QueryId("test-id"), pipeline_error)
    return PipelineResult(
        query_id=QueryId("test-id"),
        decision=Decision(action=action, confidence=confidence, rationale="test"),
        generated_text=generated_text,
        corrective_attempts=corrective_attempts,
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
    """Context manager: starts the app lifespan, then injects mock pipeline."""
    s = settings or Settings()
    app = create_app(settings=s)
    with TestClient(app, raise_server_exceptions=False) as client:
        setattr(app.state, _KEY_PIPELINE, _make_mock_pipeline(result))
        yield client


@contextmanager
def _real_client(settings: Settings | None = None):
    """Context manager: full real pipeline (offline_test smoke corpus)."""
    s = settings or Settings()
    app = create_app(settings=s)
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def answer_client():
    with _mocked_client(_make_result(DecisionAction.ANSWER)) as c:
        yield c

@pytest.fixture
def abstain_client():
    with _mocked_client(_make_result(
            action=DecisionAction.ABSTAIN, confidence=0.2,
            generated_text="")) as c:
        yield c

@pytest.fixture
def warning_client():
    with _mocked_client(_make_result(
            action=DecisionAction.WARNING, confidence=0.6,
            generated_text="With caveats. [E1]")) as c:
        yield c

@pytest.fixture
def error_client():
    with _mocked_client(_make_result(
            pipeline_error=PipelineError(
                stage="sufficiency_gate",
                message="scored_item_count=0 < min_scored_items=1",
                is_retryable=False))) as c:
        yield c

@pytest.fixture
def real():
    with _real_client() as c:
        yield c


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_200(self, real):
        assert real.get("/health").status_code == 200

    def test_status_ok(self, real):
        assert real.get("/health").json()["status"] == "ok"

    def test_has_version(self, real):
        assert "version" in real.get("/health").json()

    def test_request_id_header(self, real):
        assert "x-request-id" in real.get("/health").headers

    def test_latency_header(self, real):
        assert "x-response-time-ms" in real.get("/health").headers

    def test_caller_request_id_preserved(self, real):
        r = real.get("/health", headers={"X-Request-ID": "trace-42"})
        assert r.headers.get("x-request-id") == "trace-42"

    def test_always_unauthenticated(self):
        s = Settings(api=APISettings(require_auth=True, _auth_token="secret"))
        with _real_client(s) as c:
            assert c.get("/health").status_code == 200

    def test_no_secrets_in_response(self, real):
        body = real.get("/health").text
        assert "api_key" not in body and "auth_token" not in body


# ---------------------------------------------------------------------------
# /ready
# ---------------------------------------------------------------------------

class TestReady:
    def test_200(self, real):
        assert real.get("/ready").status_code == 200

    def test_ready_true(self, real):
        assert real.get("/ready").json()["ready"] is True

    def test_execution_mode(self, real):
        assert real.get("/ready").json()["execution_mode"] == "offline_test"

    def test_has_components(self, real):
        components = real.get("/ready").json()["components"]
        assert isinstance(components, list) and len(components) > 0

    def test_bm25_loaded(self, real):
        states = {c["name"]: c["state"]
                  for c in real.get("/ready").json()["components"]}
        assert states["bm25_index"] == "loaded"

    def test_llm_reported_as_stub_not_loaded(self, real):
        # The offline stub must never be reported as a loaded real LLM.
        body = real.get("/ready").json()
        states = {c["name"]: c["state"] for c in body["components"]}
        assert states["llm"] == "stub"
        assert body["offline"] is True
        assert body["llm_provider"] == "stub"

    def test_offline_components_reported_disabled(self, real):
        states = {c["name"]: c["state"]
                  for c in real.get("/ready").json()["components"]}
        assert states["dense_index"] == "disabled"
        assert states["reranker"] == "disabled"
        assert states["nli"] == "disabled"
        assert states["knowledge_base"] == "loaded"

    def test_always_unauthenticated(self):
        s = Settings(api=APISettings(require_auth=True, _auth_token="secret"))
        with _real_client(s) as c:
            assert c.get("/ready").status_code == 200

    def test_no_secrets_in_response(self, real):
        body = real.get("/ready").text
        assert "api_key" not in body and "auth_token" not in body


# ---------------------------------------------------------------------------
# /query -- validation
# ---------------------------------------------------------------------------

class TestQueryValidation:
    def test_empty_query_422(self, answer_client):
        assert answer_client.post("/query", json={"query": ""}).status_code == 422

    def test_whitespace_only_422(self, answer_client):
        assert answer_client.post("/query", json={"query": "   "}).status_code == 422

    def test_missing_field_422(self, answer_client):
        assert answer_client.post("/query", json={}).status_code == 422

    def test_non_string_422(self, answer_client):
        assert answer_client.post("/query", json={"query": 999}).status_code == 422

    def test_too_long_422(self, answer_client):
        assert answer_client.post("/query", json={"query": "x" * 2049}).status_code == 422

    def test_max_length_accepted(self, answer_client):
        assert answer_client.post("/query", json={"query": "a" * 2048}).status_code == 200

    def test_valid_query_200(self, answer_client):
        assert answer_client.post(
            "/query", json={"query": "What are visiting hours?"}
        ).status_code == 200

    def test_no_body_422(self, answer_client):
        assert answer_client.post("/query").status_code == 422


# ---------------------------------------------------------------------------
# /query -- response structure
# ---------------------------------------------------------------------------

class TestQueryResponse:
    def test_has_query_id(self, answer_client):
        r = answer_client.post("/query", json={"query": "test"})
        assert "query_id" in r.json()

    def test_has_decision_action(self, answer_client):
        r = answer_client.post("/query", json={"query": "test"})
        assert "action" in r.json()["decision"]

    def test_has_confidence(self, answer_client):
        r = answer_client.post("/query", json={"query": "test"})
        assert "confidence" in r.json()["decision"]

    def test_answer_action(self, answer_client):
        r = answer_client.post("/query", json={"query": "test"})
        assert r.json()["decision"]["action"] == "answer"

    def test_answer_has_text(self, answer_client):
        r = answer_client.post("/query", json={"query": "test"})
        assert r.json()["generated_text"] != ""

    def test_answer_has_answer_true(self, answer_client):
        r = answer_client.post("/query", json={"query": "test"})
        assert r.json()["has_answer"] is True

    def test_abstain_action(self, abstain_client):
        r = abstain_client.post("/query", json={"query": "test"})
        assert r.json()["decision"]["action"] == "abstain"

    def test_abstain_empty_text(self, abstain_client):
        r = abstain_client.post("/query", json={"query": "test"})
        assert r.json()["generated_text"] == ""

    def test_abstain_has_answer_false(self, abstain_client):
        r = abstain_client.post("/query", json={"query": "test"})
        assert r.json()["has_answer"] is False

    def test_warning_has_answer_true(self, warning_client):
        r = warning_client.post("/query", json={"query": "test"})
        assert r.json()["has_answer"] is True

    def test_pipeline_error_present(self, error_client):
        r = error_client.post("/query", json={"query": "test"})
        assert r.status_code == 200
        body = r.json()
        assert body["pipeline_error"] is not None
        assert body["pipeline_error"]["stage"] == "sufficiency_gate"
        assert body["success"] is False

    def test_pipeline_error_is_retryable_false(self, error_client):
        r = error_client.post("/query", json={"query": "test"})
        assert r.json()["pipeline_error"]["is_retryable"] is False

    def test_success_true_for_answer(self, answer_client):
        r = answer_client.post("/query", json={"query": "test"})
        assert r.json()["success"] is True

    def test_corrective_attempts_present(self, answer_client):
        r = answer_client.post("/query", json={"query": "test"})
        assert "corrective_attempts" in r.json()

    def test_confidence_in_range(self, answer_client):
        r = answer_client.post("/query", json={"query": "test"})
        conf = r.json()["decision"]["confidence"]
        assert isinstance(conf, float) and 0.0 <= conf <= 1.0

    def test_caller_request_id_in_response(self, answer_client):
        r = answer_client.post("/query",
            json={"query": "test", "request_id": "my-id-123"})
        assert r.json()["request_id"] == "my-id-123"

    def test_auto_request_id_generated(self, answer_client):
        r = answer_client.post("/query", json={"query": "test"})
        rid = r.json().get("request_id")
        assert rid is not None and len(rid) > 0

    def test_no_secrets_in_response(self, answer_client):
        r = answer_client.post("/query", json={"query": "test"})
        assert "api_key" not in r.text and "auth_token" not in r.text


# ---------------------------------------------------------------------------
# /query -- real pipeline
# ---------------------------------------------------------------------------

class TestQueryRealPipeline:
    def test_200(self, real):
        r = real.post("/query", json={"query": "What are visiting hours?"})
        assert r.status_code == 200

    def test_valid_action(self, real):
        r = real.post("/query", json={"query": "What are visiting hours?"})
        valid = {"answer", "warning", "abstain", "repair", "regenerate"}
        assert r.json()["decision"]["action"] in valid

    def test_drug_conflict_query(self, real):
        r = real.post("/query", json={"query": "What is the dose of Drug C?"})
        assert r.status_code == 200
        assert r.json()["decision"]["action"] in {"answer", "warning", "abstain"}

    def test_no_sufficiency_gate_error(self, real):
        r = real.post("/query", json={"query": "visiting hours"})
        err = r.json().get("pipeline_error")
        if err:
            assert err["stage"] != "sufficiency_gate", (
                f"Unexpected sufficiency failure: {err['message']}"
            )


# ---------------------------------------------------------------------------
# /query -- middleware
# ---------------------------------------------------------------------------

class TestQueryMiddleware:
    def test_413_content_length(self, answer_client):
        r = answer_client.post(
            "/query",
            content=b"x" * 70_000,
            headers={"Content-Type": "application/json",
                     "Content-Length": "70000"},
        )
        assert r.status_code == 413

    def test_413_has_error_field(self, answer_client):
        r = answer_client.post(
            "/query",
            content=b"x" * 70_000,
            headers={"Content-Type": "application/json",
                     "Content-Length": "70000"},
        )
        assert "error" in r.json()

    def test_request_id_header(self, answer_client):
        r = answer_client.post("/query", json={"query": "test"})
        assert "x-request-id" in r.headers

    def test_latency_header(self, answer_client):
        r = answer_client.post("/query", json={"query": "test"})
        assert "x-response-time-ms" in r.headers

    def test_caller_request_id_preserved(self, answer_client):
        r = answer_client.post(
            "/query", json={"query": "test"},
            headers={"X-Request-ID": "trace-xyz-123"})
        assert r.headers.get("x-request-id") == "trace-xyz-123"


# ---------------------------------------------------------------------------
# /admin/config
# ---------------------------------------------------------------------------

class TestAdminConfig:
    def test_200(self, real):
        assert real.get("/admin/config").status_code == 200

    def test_execution_mode(self, real):
        assert real.get("/admin/config").json()["execution_mode"] == "offline_test"

    def test_reranker_disabled(self, real):
        assert real.get("/admin/config").json()["reranker_enabled"] is False

    def test_nli_disabled(self, real):
        assert real.get("/admin/config").json()["nli_enabled"] is False

    def test_llm_provider_stub(self, real):
        assert real.get("/admin/config").json()["llm_provider"] == "stub"

    def test_has_effective_routing(self, real):
        routing = real.get("/admin/config").json()["effective_routing"]
        assert "weights" in routing and "thresholds" in routing and "note" in routing

    def test_routing_weights_numeric(self, real):
        weights = real.get("/admin/config").json()["effective_routing"]["weights"]
        for k, v in weights.items():
            assert isinstance(v, (int, float)), f"{k} not numeric"

    def test_no_api_key(self, real):
        assert "api_key" not in real.get("/admin/config").text

    def test_no_auth_token(self, real):
        assert "auth_token" not in real.get("/admin/config").text

    def test_has_max_corrective_attempts(self, real):
        assert "max_corrective_attempts" in real.get("/admin/config").json()


# ---------------------------------------------------------------------------
# Auth enforcement
# ---------------------------------------------------------------------------

class TestAuth:
    @pytest.fixture
    def auth_c(self):
        s = Settings(api=APISettings(require_auth=True, _auth_token="test-secret"))
        with _real_client(s) as c:
            yield c

    def test_health_unauthenticated(self, auth_c):
        assert auth_c.get("/health").status_code == 200

    def test_ready_unauthenticated(self, auth_c):
        assert auth_c.get("/ready").status_code == 200

    def test_query_no_token_401(self, auth_c):
        assert auth_c.post("/query", json={"query": "test"}).status_code == 401

    def test_query_wrong_token_401(self, auth_c):
        r = auth_c.post("/query", json={"query": "test"},
                        headers={"Authorization": "Bearer wrong"})
        assert r.status_code == 401

    def test_query_correct_token_200(self, auth_c):
        r = auth_c.post("/query", json={"query": "test"},
                        headers={"Authorization": "Bearer test-secret"})
        assert r.status_code == 200

    def test_admin_no_token_401(self, auth_c):
        assert auth_c.get("/admin/config").status_code == 401

    def test_admin_wrong_token_401(self, auth_c):
        r = auth_c.get("/admin/config",
                       headers={"Authorization": "Bearer wrong"})
        assert r.status_code == 401

    def test_admin_correct_token_200(self, auth_c):
        r = auth_c.get("/admin/config",
                       headers={"Authorization": "Bearer test-secret"})
        assert r.status_code == 200

    def test_malformed_scheme_401(self, auth_c):
        r = auth_c.post("/query", json={"query": "test"},
                        headers={"Authorization": "Basic test-secret"})
        assert r.status_code == 401

    def test_token_never_in_response(self, auth_c):
        r = auth_c.get("/admin/config",
                       headers={"Authorization": "Bearer test-secret"})
        assert "test-secret" not in r.text

    def test_no_auth_false_allows_all(self):
        with _real_client() as c:
            assert c.get("/health").status_code == 200
            assert c.get("/ready").status_code == 200
            assert c.post("/query", json={"query": "test"}).status_code == 200
            assert c.get("/admin/config").status_code == 200


# ---------------------------------------------------------------------------
# API models unit tests
# ---------------------------------------------------------------------------

class TestAPIModels:
    def test_answer_to_response(self):
        from rasvcx.api.models import pipeline_result_to_response
        r = _make_result(DecisionAction.ANSWER, confidence=0.9,
                         generated_text="Answer text.")
        resp = pipeline_result_to_response(r, request_id="req-1")
        assert resp.decision.action.value == "answer"
        assert resp.generated_text == "Answer text."
        assert resp.has_answer is True
        assert resp.request_id == "req-1"
        assert resp.success is True

    def test_abstain_to_response(self):
        from rasvcx.api.models import pipeline_result_to_response
        r = _make_result(DecisionAction.ABSTAIN, confidence=0.1, generated_text="")
        resp = pipeline_result_to_response(r)
        assert resp.decision.action.value == "abstain"
        assert resp.has_answer is False

    def test_warning_has_answer_true(self):
        from rasvcx.api.models import pipeline_result_to_response
        r = _make_result(DecisionAction.WARNING, generated_text="Caveat answer.")
        resp = pipeline_result_to_response(r)
        assert resp.has_answer is True

    def test_error_result_to_response(self):
        from rasvcx.api.models import pipeline_result_to_response
        r = _make_result(pipeline_error=PipelineError(
            stage="generation", message="LLM failed", is_retryable=True))
        resp = pipeline_result_to_response(r)
        assert resp.pipeline_error is not None
        assert resp.pipeline_error.stage == "generation"
        assert resp.pipeline_error.is_retryable is True
        assert resp.success is False

    def test_query_request_valid(self):
        from rasvcx.api.models import QueryRequest
        q = QueryRequest(query="What are visiting hours?")
        assert q.query == "What are visiting hours?"
        assert q.request_id is None

    def test_query_request_whitespace_rejected(self):
        from rasvcx.api.models import QueryRequest
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            QueryRequest(query="   ")

    def test_component_state_values(self):
        from rasvcx.api.models import ComponentState
        assert ComponentState.CONFIGURED.value == "configured"
        assert ComponentState.LOADED.value == "loaded"
        assert ComponentState.REACHABLE.value == "reachable"
        assert ComponentState.UNAVAILABLE.value == "unavailable"

    def test_decision_action_enum_values(self):
        from rasvcx.api.models import DecisionActionEnum
        values = {e.value for e in DecisionActionEnum}
        assert values == {"answer", "warning", "abstain", "repair", "regenerate"}

    def test_error_response_model(self):
        from rasvcx.api.models import ErrorResponse
        e = ErrorResponse(error="overload", detail="at capacity", request_id="r1")
        assert e.error == "overload" and e.request_id == "r1"