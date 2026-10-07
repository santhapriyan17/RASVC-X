"""Query-side prompt-injection guard (pipeline stage: input_validation).

Retrieved documents are already isolated from instructions by the
generation prompt.  This guard covers the other entry point: a *question*
that is not a question but an attempt to redirect the model ("ignore all
previous instructions ...", "reveal your system prompt").

Such a request has no medical answer in the knowledge base.  Letting it
through is still harmful: lexical retrieval matches it to whatever
document happens to contain similar words (including a poisoned document
carrying the same phrases) and the pipeline then answers a question the
user never asked.  The guard rejects it before retrieval.

Deterministic and conservative: it matches explicit instruction-override
phrasing only, so ordinary clinical questions -- including ones that
mention instructions, prompts or systems in a medical sense -- pass.
A match is reported with the name of the rule, never by echoing the text.
"""

from __future__ import annotations

import re

_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("override_previous_instructions", re.compile(
        r"\b(ignore|disregard|forget|override|bypass)\b.{0,40}\b"
        r"(previous|prior|above|earlier|all|your|the|any)\b.{0,30}\b"
        r"(instruction|prompt|rule|guideline|constraint|direction)s?\b", re.IGNORECASE | re.DOTALL)),
    ("reveal_system_prompt", re.compile(
        r"\b(reveal|show|print|repeat|output|leak|tell me)\b.{0,40}\b"
        r"(system|hidden|initial|developer)\s+(prompt|message|instruction)s?\b", re.IGNORECASE | re.DOTALL)),
    ("role_reassignment", re.compile(
        r"\b(you are now|from now on you|act as|pretend to be|enter)\b.{0,40}\b"
        r"(developer mode|dan\b|jailbreak|unrestricted|no restrictions|without (any )?(rules|restrictions))",
        re.IGNORECASE | re.DOTALL)),
    ("forced_output", re.compile(
        r"\b(reply|respond|answer|output|say)\s+(only|just|exactly)\s+(with\s+)?(the\s+)?"
        r"(word|phrase|text|string)\b", re.IGNORECASE)),
    ("system_override_marker", re.compile(
        r"\b(system override|begin system prompt|end of (system )?prompt|###\s*instruction)\b", re.IGNORECASE)),
)


def detect_instruction_override(text: str) -> str | None:
    """Return the name of the first matching rule, or None if the text is
    an ordinary question."""
    for name, pattern in _RULES:
        if pattern.search(text):
            return name
    return None


__all__ = ["detect_instruction_override"]
