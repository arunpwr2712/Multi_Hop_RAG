"""Candidate causal path selection from the knowledge graph."""

from __future__ import annotations

from typing import Dict, List, Optional

def choose_best_path(candidate_paths: List[List[str]], decomposition: Dict[str, Optional[str]]) -> List[str]:
    """Rank candidate paths using overlap with decomposed causal anchors."""

    anchors = {value.lower() for value in decomposition.values() if value}
    if not candidate_paths:
        return []

    def score(path: List[str]) -> tuple[int, int]:
        overlap = sum(1 for node in path if node.lower() in anchors)
        return overlap, -len(path)

    return max(candidate_paths, key=score)
