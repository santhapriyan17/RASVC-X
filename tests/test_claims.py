"""Tests for rasvcx.schemas.claims."""

import pytest

from rasvcx.schemas.claims import Claim, ClaimRegistry, ClaimSource, ClaimType
from rasvcx.schemas.common import UNKNOWN, ChunkId, ClaimId


def make_claim(claim_id="c1", text="X causes Y", safety_critical=False):
    return Claim(
        claim_id=ClaimId(claim_id),
        text=text,
        source=ClaimSource.EVIDENCE_EXTRACTION,
        claim_type=ClaimType.FACTUAL,
        origin_chunk_id=ChunkId("chunk1"),
        is_safety_critical=safety_critical,
    )


def test_claim_construction():
    claim = make_claim()
    assert claim.claim_id == "c1"
    assert claim.text == "X causes Y"
    assert claim.source is ClaimSource.EVIDENCE_EXTRACTION


def test_claim_requires_nonempty_id():
    with pytest.raises(ValueError):
        Claim(
            claim_id=ClaimId(""),
            text="x",
            source=ClaimSource.EVIDENCE_EXTRACTION,
            claim_type=ClaimType.FACTUAL,
            origin_chunk_id=ChunkId("chunk1"),
        )


def test_claim_requires_nonempty_text():
    with pytest.raises(ValueError):
        Claim(
            claim_id=ClaimId("c1"),
            text="",
            source=ClaimSource.EVIDENCE_EXTRACTION,
            claim_type=ClaimType.FACTUAL,
            origin_chunk_id=ChunkId("chunk1"),
        )


def test_claim_is_immutable():
    claim = make_claim()
    with pytest.raises(Exception):
        claim.text = "mutated"


def test_claim_post_generation_origin_chunk_can_be_unknown():
    claim = Claim(
        claim_id=ClaimId("c2"),
        text="dose is 10mg",
        source=ClaimSource.POST_GENERATION,
        claim_type=ClaimType.DOSAGE,
        origin_chunk_id=UNKNOWN,
    )
    assert claim.origin_chunk_id is UNKNOWN


def test_registry_register_and_get():
    registry = ClaimRegistry()
    claim = make_claim()
    registry.register(claim)
    assert registry.get(ClaimId("c1")) is claim
    assert ClaimId("c1") in registry
    assert len(registry) == 1


def test_registry_idempotent_reregistration():
    registry = ClaimRegistry()
    claim = make_claim()
    registry.register(claim)
    registry.register(claim)
    assert len(registry) == 1


def test_registry_rejects_duplicate_id_with_different_content():
    registry = ClaimRegistry()
    registry.register(make_claim(text="original"))
    with pytest.raises(ValueError):
        registry.register(make_claim(text="different"))


def test_registry_get_missing_raises_keyerror():
    registry = ClaimRegistry()
    with pytest.raises(KeyError):
        registry.get(ClaimId("missing"))


def test_registry_ids_and_as_dict():
    registry = ClaimRegistry()
    claim1 = make_claim(claim_id="c1")
    claim2 = make_claim(claim_id="c2")
    registry.register(claim1)
    registry.register(claim2)
    assert registry.ids() == frozenset({ClaimId("c1"), ClaimId("c2")})
    snapshot = registry.as_dict()
    assert snapshot == {ClaimId("c1"): claim1, ClaimId("c2"): claim2}


def test_registry_no_cross_instance_state_leakage():
    registry_a = ClaimRegistry()
    registry_b = ClaimRegistry()
    registry_a.register(make_claim(claim_id="only-in-a"))
    assert ClaimId("only-in-a") not in registry_b
    assert len(registry_b) == 0
