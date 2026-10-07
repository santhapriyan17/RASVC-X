"""Evidence roles, temporal states, lifecycle metadata and KB identity v2.

No test here needs the network, a GPU, Qdrant, or an API key.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from rasvcx.provenance.evidence_roles import (
    EvidenceRole,
    TemporalStatus,
    answer_basis,
    assess_evidence,
    temporal_status,
)
from rasvcx.retrieval.corpus import CorpusStore
from rasvcx.schemas.common import ChunkId, EvidenceItemId, SourceType, UNKNOWN
from rasvcx.schemas.evidence import EvidenceItem, Provenance, SourceLifecycle, SourceRef


def _prov(st: SourceType = SourceType.CLINICAL_GUIDELINE, date: object = "2024-01-01") -> Provenance:
    return Provenance(source_type=st, date=date, jurisdiction="US", population="adults", dosage_context="oral")


def _item(iid: str, rerank: float | None = 3.0, lifecycle: SourceLifecycle | None = None,
          st: SourceType = SourceType.CLINICAL_GUIDELINE, doc: str | None = None) -> EvidenceItem:
    return EvidenceItem(
        item_id=EvidenceItemId(iid), chunk_id=ChunkId(f"{doc or iid}__chunk_0000"), text=f"text {iid}",
        retrieval_score=0.02, rerank_score=rerank, provenance=_prov(st),
        source=SourceRef(doc_id=doc or iid, lifecycle=lifecycle),
    )


def _claim(label: str, supporting=(), contradicting=()):
    return SimpleNamespace(label=SimpleNamespace(value=label),
                           supporting_item_ids=frozenset(supporting),
                           contradicting_item_ids=frozenset(contradicting))


# ---------------------------------------------------------------------------
# Temporal state: declared lifecycle only
# ---------------------------------------------------------------------------

class TestTemporalStatus:
    @pytest.mark.parametrize("lc,expected", [
        (None, TemporalStatus.UNKNOWN),
        (SourceLifecycle(status="current"), TemporalStatus.CURRENT),
        (SourceLifecycle(status="withdrawn"), TemporalStatus.WITHDRAWN),
        (SourceLifecycle(status="superseded"), TemporalStatus.SUPERSEDED),
        (SourceLifecycle(superseded_by="newdoc"), TemporalStatus.SUPERSEDED),
        (SourceLifecycle(status="current", superseded_by="newdoc"), TemporalStatus.SUPERSEDED),
        (SourceLifecycle(status="historical"), TemporalStatus.HISTORICAL),
        (SourceLifecycle(version="3"), TemporalStatus.UNKNOWN),
    ])
    def test_rules(self, lc, expected) -> None:
        assert temporal_status(_item("E1", lifecycle=lc))[0] is expected

    def test_old_date_alone_is_unknown_not_superseded(self) -> None:
        it = dataclasses.replace(_item("E1"), provenance=_prov(date="1998-01-01"))
        assert temporal_status(it)[0] is TemporalStatus.UNKNOWN

    def test_invalid_status_rejected(self) -> None:
        with pytest.raises(ValueError):
            SourceLifecycle(status="obsolete")

    def test_supersession_is_applied_kb_wide(self) -> None:
        """The old document declares nothing; the new one declares it
        supersedes the old.  Retrieving ONLY the old one still shows it
        superseded, and the stored record is not rewritten."""
        store = CorpusStore()
        store.add(ChunkId("old__chunk_0000"), "dose 20 mg weekly", _prov(), SourceRef(doc_id="old"))
        store.add(ChunkId("new__chunk_0000"), "dose 10 mg fortnightly", _prov(),
                  SourceRef(doc_id="new", lifecycle=SourceLifecycle(status="current", supersedes=("old",))))
        src = store.lookup_source(ChunkId("old__chunk_0000"))
        assert src.lifecycle.superseded_by == "new"
        old_record = next(r for r in store.to_records() if r["doc_id"] == "old")
        assert "lifecycle" not in old_record
        reloaded = CorpusStore.from_records(store.to_records())
        assert reloaded.lookup_source(ChunkId("old__chunk_0000")).lifecycle.superseded_by == "new"


# ---------------------------------------------------------------------------
# Roles
# ---------------------------------------------------------------------------

class TestRoles:
    def test_retrieved_is_not_supporting(self) -> None:
        items = [_item("E1", 4.0), _item("E2", -3.0), _item("E3", None)]
        roles = {k: v.role for k, v in assess_evidence(items).items()}
        assert roles == {"E1": EvidenceRole.RELEVANT, "E2": EvidenceRole.IRRELEVANT,
                         "E3": EvidenceRole.RETRIEVED}

    def test_roles_from_verification(self) -> None:
        items = [_item("E1"), _item("E2"), _item("E3"), _item("E4", -1.0)]
        vs = SimpleNamespace(claim_results=[
            _claim("supported", supporting=["E1"]),
            _claim("contradicted", supporting=["E1"], contradicting=["E2"]),
            _claim("unsupported", supporting=["E3"]),  # unsupported claim: E3 is not support
        ])
        a = assess_evidence(items, vs)
        assert a["E1"].role is EvidenceRole.SUPPORTING and a["E1"].supports_answer
        assert a["E2"].role is EvidenceRole.CONTRADICTORY and a["E2"].contradicts_answer
        assert a["E3"].role is EvidenceRole.RELEVANT and not a["E3"].supports_answer
        assert a["E4"].role is EvidenceRole.IRRELEVANT

    def test_withdrawn_support_is_superseded_role_but_still_marked_as_used(self) -> None:
        items = [_item("E1", lifecycle=SourceLifecycle(status="withdrawn"))]
        vs = SimpleNamespace(claim_results=[_claim("supported", supporting=["E1"])])
        a = assess_evidence(items, vs)["E1"]
        assert a.role is EvidenceRole.SUPERSEDED and a.supports_answer
        assert a.temporal_status is TemporalStatus.WITHDRAWN

    def test_conflict_partner_of_supporting_item_is_contradictory(self) -> None:
        from rasvcx.schemas.common import EvidenceRelationship

        items = [_item("E1"), _item("E2")]
        vs = SimpleNamespace(claim_results=[_claim("supported", supporting=["E1"])])
        val = SimpleNamespace(resolutions=[SimpleNamespace(
            candidate_id="c1", relationship=EvidenceRelationship.GENUINE_CONFLICT)])
        a = assess_evidence(items, vs, val, {"c1": ("E1", "E2")})
        assert a["E2"].role is EvidenceRole.CONTRADICTORY

    def test_answer_basis_excludes_unused_and_stale(self) -> None:
        items = [_item("E1"), _item("E2", st=SourceType.DRUG_LABEL),
                 _item("E3", lifecycle=SourceLifecycle(status="superseded"))]
        vs = SimpleNamespace(claim_results=[_claim("supported", supporting=["E1", "E3"])])
        assert [str(i.item_id) for i in answer_basis(items, vs)] == ["E1"]
        assert answer_basis(items, None) == items  # no answer yet: all evidence

    def test_irrelevant_second_source_type_does_not_raise_diversity(self) -> None:
        from rasvcx.confidence.features import FeatureExtractor
        from rasvcx.schemas.common import QueryId
        from rasvcx.schemas.evidence import EvidenceBundle
        from rasvcx.routing import route_query
        from rasvcx.verification import SupportLabel

        items = [_item("E1", doc="a"), _item("E2", doc="a2"),
                 _item("E3", -6.0, st=SourceType.DRUG_LABEL, doc="b")]
        b = EvidenceBundle(query_id=QueryId("q"), risk_profile=route_query("dose"))
        for it in items:
            b.add_evidence_item(it)
        vs = SimpleNamespace(claim_results=[SimpleNamespace(
            label=SupportLabel.SUPPORTED, supporting_item_ids=frozenset({"E1", "E2"}),
            contradicting_item_ids=frozenset())])
        fx = FeatureExtractor()
        with_basis = fx._source_diversity(answer_basis(items, vs))
        all_items = fx._source_diversity(items)
        assert all_items == 1.0 and with_basis == 0.5


# ---------------------------------------------------------------------------
# Warnings are about the evidence the answer uses
# ---------------------------------------------------------------------------

class TestWarnings:
    def _run(self, supporting: list[str]):
        from rasvcx.pipeline.orchestrator import _Run
        from rasvcx.schemas.common import QueryId
        from rasvcx.schemas.evidence import EvidenceBundle
        from rasvcx.schemas.query import QueryRequest
        from rasvcx.routing import route_query

        q = QueryRequest(query_id=QueryId("q"), raw_text="dose", normalized_text="dose")
        run = _Run(q, lambda *a: None, None, None)
        run.bundle = EvidenceBundle(query_id=QueryId("q"), risk_profile=route_query("dose"))
        for iid in ("E1", "E2"):
            run.bundle.add_evidence_item(_item(iid))
        mismatch = SimpleNamespace(value="known_mismatch")
        ok = SimpleNamespace(value="unknown")
        run.provenance = SimpleNamespace(per_item={
            EvidenceItemId("E1"): SimpleNamespace(temporal=ok, population=ok, jurisdiction=ok, dosage_context=ok),
            EvidenceItemId("E2"): SimpleNamespace(temporal=mismatch, population=ok, jurisdiction=ok, dosage_context=ok),
        })
        run.verification_summary = SimpleNamespace(
            claim_results=[_claim("supported", supporting=supporting)])
        run.observability()
        return run.warnings

    def test_unused_outdated_item_does_not_warn(self) -> None:
        assert not any("outdated" in w for w in self._run(["E1"]))

    def test_supporting_outdated_item_warns(self) -> None:
        warnings = self._run(["E1", "E2"])
        assert any("supporting evidence may be outdated" in w and w.endswith(": E2") for w in warnings)


# ---------------------------------------------------------------------------
# Decision: an answer may not rest on withdrawn evidence
# ---------------------------------------------------------------------------

class TestContextMismatchedSupport:
    def test_only_supporting_items_with_known_context_mismatch_count(self) -> None:
        from rasvcx.pipeline.orchestrator import _context_mismatched_support

        mm, ok, unk = (SimpleNamespace(value=v) for v in ("known_mismatch", "known_match", "unknown"))
        run = SimpleNamespace(provenance=SimpleNamespace(per_item={
            EvidenceItemId("E1"): SimpleNamespace(population=mm, jurisdiction=ok, dosage_context=ok, temporal=ok),
            EvidenceItemId("E2"): SimpleNamespace(population=ok, jurisdiction=ok, dosage_context=ok, temporal=mm),
            EvidenceItemId("E3"): SimpleNamespace(population=mm, jurisdiction=unk, dosage_context=unk, temporal=ok),
        }))
        assessed = {iid: SimpleNamespace(supports_answer=iid != "E3") for iid in ("E1", "E2", "E3")}
        # E1: supporting + population mismatch -> counts.  E2: only temporal
        # (age is handled by lifecycle rules).  E3: not supporting.
        assert _context_mismatched_support(run, assessed) == ["E1"]


class TestNonCurrentSupport:
    def test_repair_then_withhold_at_high_risk(self) -> None:
        from rasvcx.decision import DecisionEngine
        from rasvcx.schemas.decision import Decision, DecisionAction
        from tests.test_pipeline import _build_orch, _query, _retrieval_fn

        stale = dataclasses.replace(
            _item("E1", 5.0, doc="old"), text="The maximum adult dose of Zorvex is 25 mg daily.",
            source=SourceRef(doc_id="old", lifecycle=SourceLifecycle(status="withdrawn")))
        other = dataclasses.replace(
            _item("E2", 4.0, st=SourceType.DRUG_LABEL, doc="lbl"),
            text="Zorvex tablets are stored at room temperature.")
        engine = DecisionEngine()
        engine.decide = lambda *a, **k: Decision(action=DecisionAction.ANSWER, confidence=0.9,
                                                 rationale="meets_answer_threshold")
        orch = _build_orch(retrieval_fn=_retrieval_fn(stale, other), decision_engine=engine,
                           canned_text="The maximum adult dose of Zorvex is 25 mg daily [E1].",
                           max_corrective_attempts=1)
        result = orch.run(_query("What is the maximum dose of Zorvex in mg for adults?"))
        assert result.decision.action is DecisionAction.ABSTAIN
        assert "withdrawn/superseded evidence E1" in result.decision.rationale
        assert result.corrective_attempts == 1  # a repair naming E1 was tried first
        assert result.evidence_assessments["E1"].role is EvidenceRole.SUPERSEDED


# ---------------------------------------------------------------------------
# KB identity v2
# ---------------------------------------------------------------------------

def _records(**changes) -> list[dict]:
    rec = {"chunk_id": "d__chunk_0000", "text": "dose is 10 mg", "doc_id": "d", "title": "T",
           "provenance": {"source_type": "drug_label", "date": "2024-01-01", "jurisdiction": "US",
                          "population": "adults", "dosage_context": "oral"},
           "ingestion_timestamp": "2026-01-01T00:00:00", "job_id": "j1"}
    rec.update(changes)
    return [rec]


class TestIdentity:
    def test_metadata_changes_identity_volatile_fields_do_not(self) -> None:
        from rasvcx.ingestion.corpus_builder import _fingerprint

        base = _fingerprint(_records())
        assert _fingerprint(_records(ingestion_timestamp="2027-05-05", job_id="j9")) == base
        assert _fingerprint(_records(heading="")) == _fingerprint(_records(heading=None))
        prov = dict(_records()[0]["provenance"], population="children")
        assert _fingerprint(_records(provenance=prov)) != base
        assert _fingerprint(_records(lifecycle={"status": "withdrawn"})) != base
        assert _fingerprint(_records(title="Other title")) != base
        assert _fingerprint(_records(text="dose is 20 mg")) != base

    def test_legacy_text_only_version_still_loads_labelled_legacy(self) -> None:
        from rasvcx.config.settings import RetrievalSettings
        from rasvcx.ingestion.corpus_builder import _fingerprint_v1
        from rasvcx.retrieval.bm25 import BM25Index
        from rasvcx.retrieval.knowledge_base import SnapshotLoader

        store = CorpusStore.from_records(_records())
        bm25 = BM25Index.build(store.to_documents_list())
        legacy_id = "v_" + _fingerprint_v1(store.to_records())[:12]
        snap = SnapshotLoader(RetrievalSettings(mode="bm25_only")).build(legacy_id, store, bm25)
        assert snap.identity_scheme == "kb-identity-v1-text-only"
        from rasvcx.pipeline.factory import seed_version_id
        assert SnapshotLoader(RetrievalSettings(mode="bm25_only")).build(
            seed_version_id(store), store, bm25).identity_scheme == "kb-identity-v2"


# ---------------------------------------------------------------------------
# Ingestion: lifecycle metadata end to end
# ---------------------------------------------------------------------------

class TestLifecycleIngestion:
    DOC = b"Protocol LCYMARKER sets the maintenance dose at 20 mg weekly.\n\nProtocol LCYMARKER applies to adults.\n"

    def test_metadata_update_publishes_and_reaches_query(self, tmp_path: Path) -> None:
        from rasvcx.api.main import create_app
        from tests.test_integration_repair import _offline_settings, _wait_for_job

        with TestClient(create_app(settings=_offline_settings(tmp_path))) as client:
            def upload(**data: str) -> dict:
                r = client.post("/ingest/upload", files={"file": ("p.txt", self.DOC, "text/plain")}, data=data)
                assert r.status_code == 202, r.text
                return _wait_for_job(client, r.json()["job_id"])

            first = upload(source_type="clinical_guideline", date="2011-05-01", status="current")
            assert first["status"] == "completed"
            v1 = client.get("/ingest/status").json()["serving_version_id"]

            same = client.post("/ingest/upload", files={"file": ("p.txt", self.DOC, "text/plain")},
                               data={"source_type": "clinical_guideline", "date": "2011-05-01", "status": "current"})
            assert same.json()["duplicate"] is True     # same file, same metadata

            withdrawn = upload(source_type="clinical_guideline", date="2011-05-01", status="withdrawn")
            assert withdrawn["status"] == "completed"
            v2 = client.get("/ingest/status").json()["serving_version_id"]
            assert v2 != v1                              # metadata-only change = new version

            d = client.post("/query", json={"query": "protocol LCYMARKER dose", "enriched": True}).json()
            hit = next(e for e in d["evidence"] if "LCYMARKER" in e["text"])
            assert hit["temporal_status"] == "WITHDRAWN"
            assert hit["lifecycle"]["status"] == "withdrawn"
            assert hit["role"] in ("SUPERSEDED", "CONTRADICTORY")

            bad = client.post("/ingest/upload", files={"file": ("p.txt", self.DOC, "text/plain")},
                              data={"status": "obsolete"})
            assert bad.status_code == 422
