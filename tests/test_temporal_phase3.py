"""Phase 3: temporal states (CURRENT / HISTORICAL / SUPERSEDED / FUTURE /
UNDATED / UNKNOWN), historical statements in answers, current-question
qualification, and Gemini provider accounting.

No test here needs the network, a GPU, Qdrant, or an API key.
"""

from __future__ import annotations

import dataclasses
from datetime import date
from types import SimpleNamespace

import pytest

from rasvcx.provenance.evidence_roles import TemporalStatus, assess_evidence, temporal_status
from rasvcx.schemas.common import UNKNOWN, ChunkId, EvidenceItemId, EvidenceRelationship, QueryId, SourceType
from rasvcx.schemas.evidence import EvidenceBundle, EvidenceItem, Provenance, SourceLifecycle, SourceRef

TODAY = date(2026, 10, 7)
NEW = "The maintenance dose of Nerolimab for adults with erosive arthropathy is 10 mg every two weeks."
OLD = "The maintenance dose of Nerolimab for adults with erosive arthropathy is 20 mg every week."


def _item(iid: str, text: str = "dose text", date_: object = "2024-01-01",
          lifecycle: SourceLifecycle | None = None, doc: str | None = None, rerank: float = 6.0) -> EvidenceItem:
    return EvidenceItem(
        item_id=EvidenceItemId(iid), chunk_id=ChunkId(f"{doc or iid}__chunk_0000"), text=text,
        retrieval_score=0.02, rerank_score=rerank,
        provenance=Provenance(source_type=SourceType.CLINICAL_GUIDELINE, date=date_, jurisdiction="US",
                              population="adults", dosage_context="subcutaneous"),
        source=SourceRef(doc_id=doc or iid, lifecycle=lifecycle),
    )


# ---------------------------------------------------------------------------
# Temporal state model
# ---------------------------------------------------------------------------

class TestTemporalStates:
    @pytest.mark.parametrize("date_,lc,expected", [
        ("2024-09-01", SourceLifecycle(status="current"), TemporalStatus.CURRENT),
        ("2011-05-01", SourceLifecycle(status="historical"), TemporalStatus.HISTORICAL),
        ("2011-05-01", SourceLifecycle(status="superseded"), TemporalStatus.SUPERSEDED),
        ("2011-05-01", SourceLifecycle(status="withdrawn"), TemporalStatus.WITHDRAWN),
        ("2024-09-01", SourceLifecycle(status="current", effective_date="2027-01-01"), TemporalStatus.FUTURE),
        ("2027-03-01", None, TemporalStatus.FUTURE),
        (UNKNOWN, None, TemporalStatus.UNDATED),
        ("2024-09-01", None, TemporalStatus.UNKNOWN),
        (UNKNOWN, SourceLifecycle(status="current"), TemporalStatus.CURRENT),
    ])
    def test_matrix(self, date_, lc, expected) -> None:
        assert temporal_status(_item("E1", date_=date_, lifecycle=lc), today=TODAY)[0] is expected

    def test_withdrawn_beats_future_effective_date(self) -> None:
        lc = SourceLifecycle(status="withdrawn", effective_date="2030-01-01")
        assert temporal_status(_item("E1", lifecycle=lc), today=TODAY)[0] is TemporalStatus.WITHDRAWN

    def test_partial_dates_are_not_called_future_prematurely(self) -> None:
        assert temporal_status(_item("E1", date_="2026"), today=TODAY)[0] is TemporalStatus.UNKNOWN
        assert temporal_status(_item("E1", date_="2026-10"), today=TODAY)[0] is TemporalStatus.UNKNOWN

    def test_same_document_different_versions(self) -> None:
        """v2 of a guideline declares it supersedes v1: retrieving only v1
        still shows it superseded, with v2 named."""
        from rasvcx.retrieval.corpus import CorpusStore

        store = CorpusStore()
        prov = _item("x").provenance
        store.add(ChunkId("guide_v1__chunk_0000"), OLD, prov,
                  SourceRef(doc_id="guide_v1", title="Guideline", lifecycle=SourceLifecycle(version="1")))
        store.add(ChunkId("guide_v2__chunk_0000"), NEW, prov,
                  SourceRef(doc_id="guide_v2", title="Guideline", lifecycle=SourceLifecycle(
                      status="current", version="2", supersedes=("guide_v1",))))
        v1 = dataclasses.replace(_item("E1"), source=store.lookup_source(ChunkId("guide_v1__chunk_0000")))
        status, reason = temporal_status(v1, today=TODAY)
        assert status is TemporalStatus.SUPERSEDED and "guide_v2" in reason


# ---------------------------------------------------------------------------
# Conflict resolution across temporal states
# ---------------------------------------------------------------------------

def _resolve(a: EvidenceItem, b: EvidenceItem):
    from rasvcx.schemas.validation import ValidationLabel, ValidationResult, ValidationStage
    from rasvcx.validation.resolution import EvidenceResolver

    bundle = EvidenceBundle(query_id=QueryId("q"), risk_profile=None)
    bundle.add_evidence_item(a)
    bundle.add_evidence_item(b)
    det = ValidationResult(candidate_id="c", stage=ValidationStage.DETERMINISTIC,
                           label=ValidationLabel.CONTRADICTION, confidence=0.9)
    return EvidenceResolver().resolve(bundle, "c", a.item_id, b.item_id, det, None, None)


class TestTemporalConflicts:
    def test_current_vs_historical_is_resolvable(self) -> None:
        r = _resolve(_item("E1", NEW, "2024-09-01", SourceLifecycle(status="current")),
                     _item("E2", OLD, "2011-05-01", SourceLifecycle(status="historical")))
        assert r.relationship is EvidenceRelationship.TEMPORAL_DIFF

    def test_current_vs_superseded_is_resolvable(self) -> None:
        r = _resolve(_item("E1", NEW, "2024-09-01", SourceLifecycle(status="current")),
                     _item("E2", OLD, "2011-05-01", SourceLifecycle(superseded_by="E1")))
        assert r.relationship is EvidenceRelationship.TEMPORAL_DIFF

    def test_newer_undeclared_conflicting_source_is_unresolved(self) -> None:
        r = _resolve(_item("E1", NEW, "2024-09-01"), _item("E2", OLD, "2011-05-01"))
        assert r.relationship is EvidenceRelationship.UNRESOLVED

    def test_two_current_sources_conflict(self) -> None:
        cur = SourceLifecycle(status="current")
        r = _resolve(_item("E1", NEW, "2024-09-01", cur), _item("E2", OLD, "2019-01-01", cur))
        assert r.relationship is EvidenceRelationship.GENUINE_CONFLICT

    def test_future_effective_guidance_does_not_override_current(self) -> None:
        r = _resolve(_item("E1", NEW, "2024-09-01", SourceLifecycle(status="current")),
                     _item("E2", OLD, "2024-09-01", SourceLifecycle(status="current", effective_date="2099-01-01")))
        assert r.relationship is EvidenceRelationship.TEMPORAL_DIFF

    def test_undated_conflicting_source_is_unresolved(self) -> None:
        r = _resolve(_item("E1", NEW, "2024-09-01"), _item("E2", OLD, UNKNOWN))
        assert r.relationship is EvidenceRelationship.UNRESOLVED


# ---------------------------------------------------------------------------
# Historical statements in generated answers (the temporal_current root cause)
# ---------------------------------------------------------------------------

def _verify(answer: str, lc_old: SourceLifecycle | None = SourceLifecycle(status="withdrawn")):
    from rasvcx.claims import AtomicClaimPipeline
    from rasvcx.routing import route_query
    from rasvcx.schemas.query import QueryRequest
    from rasvcx.validation import NLIService, NullNLIBackend, ValidationPipeline
    from rasvcx.verification import VerificationPipeline

    q = QueryRequest(query_id=QueryId("q"), raw_text="current dose", normalized_text="current dose")
    bundle = EvidenceBundle(query_id=QueryId("q"), risk_profile=route_query("what is the dose"))
    bundle.add_evidence_item(_item("E1", NEW + " The earlier weekly schedule was withdrawn.", "2024-09-01",
                                   SourceLifecycle(status="current"), doc="g2024"))
    bundle.add_evidence_item(_item("E2", OLD, "2011-05-01", lc_old, doc="g2011"))
    AtomicClaimPipeline().run(bundle)
    vsum = ValidationPipeline(NLIService(NullNLIBackend())).run(bundle, q, bundle.risk_profile)
    return VerificationPipeline().verify(answer, bundle, query=q, validation_summary=vsum), bundle


class TestHistoricalStatements:
    CURRENT_SENTENCE = ("The maintenance dose of Nerolimab for adults with erosive arthropathy is "
                        "10 mg every two weeks [E1].")

    def test_reported_history_is_not_a_contradiction(self) -> None:
        """Exact sentence observed in the failing live run (run 0)."""
        vs, bundle = _verify(self.CURRENT_SENTENCE + " An earlier maintenance dose of 20 mg every week "
                             "was previously used, but that weekly schedule has been withdrawn [E1, E2].")
        hist = vs.claim_results[1]
        assert hist.label.value == "supported" and hist.reason_code.value == "historical_statement"
        assert vs.safety_critical_failure_count == 0
        a = assess_evidence(bundle.evidence_items.values(), vs)
        assert a["E1"].supports_answer and not a["E2"].supports_answer  # history is not support

    # A claim that asserts the OLD value as the current one, citing only the
    # withdrawn source, is "supported" by that citation at M9 (pre-existing
    # direct-evidence match).  It is NOT excused as history, and the
    # withdrawn source is reported as support for the answer -- which the
    # decision layer then repairs / withholds (see
    # test_evidence_semantics.py::TestNonCurrentSupport).
    @pytest.mark.parametrize("answer", [
        "The maintenance dose of Nerolimab for adults with erosive arthropathy is 20 mg every week [E2].",
        "The previously used dose of 20 mg every week is still recommended for adults with erosive arthropathy [E2].",
    ])
    def test_old_value_asserted_as_current_is_not_excused_and_reaches_the_decision_rule(self, answer) -> None:
        vs, bundle = _verify(answer)
        r = vs.claim_results[0]
        assert r.reason_code is None or r.reason_code.value != "historical_statement"
        a = assess_evidence(bundle.evidence_items.values(), vs)
        assert a["E2"].supports_answer is True and a["E2"].temporal_status is TemporalStatus.WITHDRAWN

    def test_old_value_asserted_against_cited_current_source_stays_contradicted(self) -> None:
        vs, _ = _verify("The maintenance dose of Nerolimab for adults with erosive arthropathy is "
                        "currently 20 mg every week [E1, E2].")
        assert vs.claim_results[0].label.value == "contradicted"

    def test_history_is_only_excused_by_a_declared_non_current_source(self) -> None:
        """Same sentence, but the 2011 source declares nothing: the KB never
        said it was replaced, so it is not excused."""
        vs, _ = _verify(self.CURRENT_SENTENCE + " An earlier maintenance dose of 20 mg every week "
                        "was previously used, but that weekly schedule has been withdrawn [E1, E2].",
                        lc_old=None)
        assert vs.claim_results[1].reason_code is None or vs.claim_results[1].reason_code.value != "historical_statement"


class TestCurrentQuestion:
    @pytest.mark.parametrize("text,meta,expected", [
        ("What is the current maintenance dose of Nerolimab?", {}, True),
        ("What is the latest guidance on X?", {}, True),
        ("What is the dose of X?", {"time_sensitivity": "current"}, True),
        ("What is the dose of X?", {}, False),
    ])
    def test_detection(self, text, meta, expected) -> None:
        from rasvcx.pipeline.orchestrator import _asks_for_current
        from rasvcx.schemas.query import QueryRequest

        q = QueryRequest(query_id=QueryId("q"), raw_text=text, normalized_text=text.lower(), metadata=meta)
        assert _asks_for_current(q) is expected


# ---------------------------------------------------------------------------
# Published-KB fail-closed checks
# ---------------------------------------------------------------------------

class TestPublishedKBExpectation:
    EXPECT = {"kb_version_id": "v_76aefc96ddc1", "kb_source": "published_kb",
              "publication_status": "ACTIVE", "dense_index_origin": "persisted"}
    GOOD = {"version_id": "v_76aefc96ddc1", "kb_source": "published_kb",
            "publication_status": "ACTIVE", "dense_index_origin": "persisted"}

    def test_published_persisted_kb_passes(self) -> None:
        from evaluation.protocol import kb_mismatches
        assert kb_mismatches(self.EXPECT, self.GOOD) == []

    @pytest.mark.parametrize("change", [
        {"kb_source": "seed_fallback", "publication_status": "NOT_PUBLISHED"},
        {"dense_index_origin": "rebuilt_from_store"},
    ])
    def test_seed_fallback_or_rebuilt_dense_index_is_refused(self, change) -> None:
        from evaluation.protocol import kb_mismatches
        assert kb_mismatches(self.EXPECT, {**self.GOOD, **change})

    def test_runtime_reports_publication_status(self, tmp_path) -> None:
        from fastapi.testclient import TestClient
        from rasvcx.api.main import create_app
        from tests.test_integration_repair import _offline_settings, _wait_for_job

        with TestClient(create_app(settings=_offline_settings(tmp_path))) as c:
            d = c.post("/query", json={"query": "visiting hours", "enriched": True}).json()
            assert d["kb"]["kb_source"] == "smoke_test"
            job = c.post("/ingest/upload", files={"file": ("p.txt", b"Protocol PUBMARK text.\n\nMore text.\n",
                                                           "text/plain")}).json()
            assert _wait_for_job(c, job["job_id"])["status"] == "completed"
        with TestClient(create_app(settings=_offline_settings(tmp_path))) as c:   # clean process start
            snap = c.app.state.rasvcx_lease_tracker.current_snapshot()
            assert snap.describe()["publication_status"] == "ACTIVE"
            assert snap.describe()["kb_source"] == "published_kb"


class TestProviderRequestReport:
    def test_rate_and_quota_from_request_log(self) -> None:
        import time as _t
        from evaluation.protocol import provider_request_report
        from rasvcx.generation import gemini_client

        t0 = _t.time() + 10_000   # isolated window in the future
        entries = [{"t": t0 + i * 2.0, "model": "m", "status": 200, "seconds": 1.0,
                    "quota_id": None, "quota_value": None, "retry_delay_seconds": None} for i in range(20)]
        entries.append({"t": t0 + 41.0, "model": "m", "status": 429, "seconds": 0.2,
                        "quota_id": "GenerateRequestsPerMinutePerProjectPerModel-FreeTier",
                        "quota_value": "15", "retry_delay_seconds": 19.0})
        gemini_client.REQUEST_LOG.extend(entries)
        rep = provider_request_report(t0, t0 + 100)
        assert rep["http_attempts"] == 21 and rep["status_counts"] == {"200": 20, "429": 1}
        assert rep["peak_attempts_in_any_60s"] == 21
        assert rep["quota_limits_reported"] == ["15"] and rep["advised_retry_delays_s"] == [19.0]


# ---------------------------------------------------------------------------
# Provider accounting (Gemini)
# ---------------------------------------------------------------------------

class _QuotaError(Exception):
    def __init__(self, delay: str = "41s") -> None:
        super().__init__("429 RESOURCE_EXHAUSTED")
        self.code = 429
        self.status = "RESOURCE_EXHAUSTED"
        self.details = {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "message": "secret prompt echo",
                                  "details": [
                                      {"@type": "type.googleapis.com/google.rpc.QuotaFailure",
                                       "violations": [{"quotaMetric": "generativelanguage.googleapis.com/generate_content_free_tier_requests",
                                                       "quotaId": "GenerateRequestsPerMinutePerProjectPerModel-FreeTier",
                                                       "quotaValue": "15"}]},
                                      {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": delay},
                                  ]}}


class TestProviderAccounting:
    def _client(self, monkeypatch, outcomes, timeout=60.0):
        pytest.importorskip("google.genai")
        from rasvcx.generation import gemini_client

        sleeps: list[float] = []
        monkeypatch.setattr(gemini_client.time, "sleep", lambda s: sleeps.append(s))
        c = gemini_client.GeminiLLMClient(api_key="k", model_name="m", max_retries=2, timeout_seconds=timeout)
        calls = {"n": 0}

        class _M:
            def generate_content(self, **kw):
                calls["n"] += 1
                o = outcomes[calls["n"] - 1]
                if isinstance(o, Exception):
                    raise o
                return SimpleNamespace(text="ok [E1]", usage_metadata=None, candidates=[])

        c._client = SimpleNamespace(models=_M())
        return c, sleeps, calls

    def test_quota_details_parsed_without_message_text(self) -> None:
        from rasvcx.generation.gemini_client import _quota_details

        q = _quota_details(_QuotaError())
        assert q == {"quota_id": "GenerateRequestsPerMinutePerProjectPerModel-FreeTier",
                     "quota_metric": "generativelanguage.googleapis.com/generate_content_free_tier_requests",
                     "quota_value": "15", "retry_delay_seconds": 41.0}
        assert "secret" not in str(q)

    def test_retry_waits_the_advised_delay_and_stats_are_recorded(self, monkeypatch) -> None:
        from rasvcx.generation.generation_types import LLMConfig

        c, sleeps, calls = self._client(monkeypatch, [_QuotaError("5s"), "ok"])
        c.generate("s", "u", LLMConfig(model_id="m", timeout_seconds=60))
        st = c.last_call_stats()
        assert sleeps == [5.0] and calls["n"] == 2
        assert st["attempts"] == 2 and st["statuses"] == [429, 200] and st["retry_wait_seconds"] == 5.0
        assert st["quota"]["quota_value"] == "15" and st["failed"] is False

    def test_no_retry_when_advised_delay_exceeds_budget(self, monkeypatch) -> None:
        from rasvcx.generation.generation_types import LLMConfig
        from rasvcx.generation.llm_client import LLMClientError

        c, sleeps, calls = self._client(monkeypatch, [_QuotaError("41s"), "ok"])
        with pytest.raises(LLMClientError):
            c.generate("s", "u", LLMConfig(model_id="m", timeout_seconds=30))
        assert sleeps == [] and calls["n"] == 1     # no futile retry, no amplified load
        assert c.last_call_stats()["failed"] is True

    def test_request_log_records_every_attempt(self, monkeypatch) -> None:
        from rasvcx.generation import gemini_client
        from rasvcx.generation.generation_types import LLMConfig

        before = len(gemini_client.REQUEST_LOG)
        c, _, _ = self._client(monkeypatch, [_QuotaError("1s"), "ok"])
        c.generate("s", "u", LLMConfig(model_id="m", timeout_seconds=60))
        tail = list(gemini_client.REQUEST_LOG)[before:]
        assert [e["status"] for e in tail] == [429, 200]
        assert tail[0]["quota_id"].startswith("GenerateRequestsPerMinute")
