"""Extensive pattern list for extracting local causal triples.

Each entry is a tuple: (compiled_regex, relation_label, confidence_score).
Regexes use named groups `cause` and `effect` and are case-insensitive.
"""

from __future__ import annotations

import re
from typing import List, Tuple

PatternEntry = Tuple[re.Pattern, str, float]

# A comprehensive set of lexico-syntactic causal patterns.
# Confidence scores are heuristic; adjust as needed for your dataset.
CAUSAL_PATTERNS: List[PatternEntry] = [
    # direct causative verbs
    (re.compile(r"(?P<cause>.+?)\s+(?:causes|cause|caused)\s+(?P<effect>.+)", re.IGNORECASE), "causes", 0.96),
    (re.compile(r"(?P<cause>.+?)\s+(?:triggers|trigger|triggered)\s+(?P<effect>.+)", re.IGNORECASE), "causes", 0.93),
    (re.compile(r"(?P<cause>.+?)\s+(?:produces|produce|produced)\s+(?P<effect>.+)", re.IGNORECASE), "causes", 0.9),
    (re.compile(r"(?P<cause>.+?)\s+(?:generates|generate|generated)\s+(?P<effect>.+)", re.IGNORECASE), "causes", 0.9),
    (re.compile(r"(?P<cause>.+?)\s+(?:creates|create|created)\s+(?P<effect>.+)", re.IGNORECASE), "causes", 0.9),

    # leading/result patterns
    (re.compile(r"(?P<cause>.+?)\s+(?:leads to|lead to|led to|leading to)\s+(?P<effect>.+)", re.IGNORECASE), "leads_to", 0.95),
    (re.compile(r"(?P<cause>.+?)\s*,\s*(?:leading to|leading to the)\s+(?P<effect>.+)", re.IGNORECASE), "leads_to", 0.9),
    (re.compile(r"(?P<cause>.+?)\s+(?:results in|resulted in|result in)\s+(?P<effect>.+)", re.IGNORECASE), "results_in", 0.94),
    (re.compile(r"(?P<effect>.+?)\s+(?:results from|result from|is a result of)\s+(?P<cause>.+)", re.IGNORECASE), "results_from", 0.9),

    # increase/raise/elevate patterns
    (re.compile(r"(?P<cause>.+?)\s+(?:increases|increase|increased|raises|raise|raised|elevates|elevate|elevated)\s+(?P<effect>.+)", re.IGNORECASE), "increases", 0.89),
    (re.compile(r"(?:an\s+)?increase in\s+(?P<cause>.+?)\s+(?:leads to|is associated with|is linked to)\s+(?P<effect>.+)", re.IGNORECASE), "leads_to", 0.82),

    # passive causation (effect first)
    (re.compile(r"(?P<effect>.+?)\s+(?:is caused by|are caused by|was caused by|were caused by|is due to|was due to)\s+(?P<cause>.+)", re.IGNORECASE), "causes", 0.95),
    (re.compile(r"(?P<effect>.+?)\s+(?:is attributable to|are attributable to|can be attributed to)\s+(?P<cause>.+)", re.IGNORECASE), "causes", 0.9),

    # because / due to / owing to
    (re.compile(r"because of\s+(?P<cause>.+?),?\s*(?P<effect>.+)", re.IGNORECASE), "causes", 0.9),
    (re.compile(r"(?P<effect>.+?)\s+because\s+(?P<cause>.+)", re.IGNORECASE), "causes", 0.85),
    (re.compile(r"(?P<effect>.+?)\s+(?:due to|owing to|as a result of)\s+(?P<cause>.+)", re.IGNORECASE), "causes", 0.9),
    (re.compile(r"(?P<cause>.+?)\s+because\s+(?:of\s+)?(?P<effect>.+)", re.IGNORECASE), "causes", 0.8),

    # causal connectors and subordination
    (re.compile(r"(?P<cause>.+?)\s+(?:so that|so|thus|therefore),?\s*(?P<effect>.+)", re.IGNORECASE), "leads_to", 0.78),
    (re.compile(r"(?P<cause>.+?)\s+and\s+as a result\s+(?P<effect>.+)", re.IGNORECASE), "results_in", 0.82),
    (re.compile(r"(?P<cause>.+?)\s*,\s*which\s+caused\s+(?P<effect>.+)", re.IGNORECASE), "causes", 0.88),

    # bring about / give rise to / set off
    (re.compile(r"(?P<cause>.+?)\s+(?:brings about|bring about|brought about)\s+(?P<effect>.+)", re.IGNORECASE), "causes", 0.9),
    (re.compile(r"(?P<cause>.+?)\s+(?:gives rise to|give rise to|gave rise to)\s+(?P<effect>.+)", re.IGNORECASE), "causes", 0.9),
    (re.compile(r"(?P<cause>.+?)\s+(?:sets off|set off|sets in motion)\s+(?P<effect>.+)", re.IGNORECASE), "causes", 0.82),

    # responsible for / the reason for
    (re.compile(r"(?P<cause>.+?)\s+(?:is responsible for|are responsible for)\s+(?P<effect>.+)", re.IGNORECASE), "causes", 0.92),
    (re.compile(r"(?P<effect>.+?)\s+(?:is the reason for|are the reason for)\s+(?P<cause>.+)", re.IGNORECASE), "causes", 0.8),

    # causative nouns and noun phrases
    (re.compile(r"(?P<cause>.+?)\s+(?:increase|rise|surge|spike)\s+in\s+(?P<effect>.+)", re.IGNORECASE), "increases", 0.8),
    (re.compile(r"(?P<effect>.+?)\s+followed by\s+(?P<cause>.+)", re.IGNORECASE), "temporal_association", 0.5),

    # conditional phrasing implying causality
    (re.compile(r"if\s+(?P<cause>.+?),\s*(?:then\s+)?(?P<effect>.+)", re.IGNORECASE), "conditional_causes", 0.7),
    (re.compile(r"when\s+(?P<cause>.+?),\s*(?P<effect>.+)", re.IGNORECASE), "conditional_causes", 0.7),

    # patterns with 'due' and 'as' in front
    (re.compile(r"due to\s+(?P<cause>.+?),?\s*(?P<effect>.+)", re.IGNORECASE), "causes", 0.88),
    (re.compile(r"as a result,?\s*(?P<effect>.+)\s+of\s+(?P<cause>.+)", re.IGNORECASE), "results_from", 0.85),

    # 'lead' nominalizations
    (re.compile(r"(?P<cause>.+?)\s+is\s+linked\s+to\s+(?P<effect>.+)", re.IGNORECASE), "linked_to", 0.6),
    (re.compile(r"(?P<cause>.+?)\s+is\s+associated\s+with\s+(?P<effect>.+)", re.IGNORECASE), "associated_with", 0.55),

    # 'resulted from' variants
    (re.compile(r"(?P<effect>.+?)\s+(?:resulted from|was the result of)\s+(?P<cause>.+)", re.IGNORECASE), "results_from", 0.9),

    # punctuation-driven patterns: em-dash or colon
    (re.compile(r"(?P<cause>.+?)\s+[-—:]\s+(?:which )?(?:caused|resulted in|leading to)\s+(?P<effect>.+)", re.IGNORECASE), "causes", 0.86),

    # small/short patterns that may be noisy but useful
    (re.compile(r"(?P<cause>[^,;:.]+?)\s+-->\s+(?P<effect>.+)", re.IGNORECASE), "causes", 0.6),
    (re.compile(r"(?P<cause>.+?)\s+->\s+(?P<effect>.+)", re.IGNORECASE), "causes", 0.6),
]

__all__ = ["CAUSAL_PATTERNS"]
