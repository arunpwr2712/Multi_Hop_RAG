"""Query intake, retrieval-query generation, and causal decomposition helpers."""

from __future__ import annotations

import re
from typing import Dict, List, Optional

from ..config import AppConfig, build_default_config
from ..llm_interface import LLMClient


def receive_query(raw_query: Optional[str] = None, prompt_text: str = "Enter your query: ") -> str:
    """Receive a user query directly or from standard input."""

    if raw_query is not None:
        query = raw_query.strip()
    else:
        query = input(prompt_text).strip()

    if not query:
        raise ValueError("A non-empty query is required")
    return query


def decompose_query_llm(query: str, config: Optional[AppConfig] = None) -> Dict[str, Optional[str]]:
    """Decompose a query into causal components using an LLM with heuristic fallback."""

    app_config = config or build_default_config()

    client = LLMClient(app_config.llm)
    require_ollama = bool(app_config.llm.require_ollama)

    if client.available:
        try:
            return _fill_missing_intermediates(_decompose_with_llm(query=query, client=client))
        except RuntimeError as exc:
            if require_ollama:
                raise RuntimeError(f"Ollama decomposition failed: {exc}") from exc
    elif require_ollama:
        raise RuntimeError("Ollama is required for decomposition but is not configured.")

    return _fill_missing_intermediates(_heuristic_decomposition(query))


def generate_retrieval_queries_llm(
    query: str,
    config: Optional[AppConfig] = None,
    min_queries: int = 5,
    max_queries: int = 10,
) -> List[str]:
    """Generate 5-10 LLM-authored subqueries for multi-hop causal retrieval."""

    app_config = config or build_default_config()
    require_ollama = bool(app_config.llm.require_ollama)
    if not app_config.llm.query_planning_enabled:
        return _heuristic_retrieval_queries(query=query, min_queries=min_queries, max_queries=max_queries)

    client = LLMClient(app_config.llm)
    if not client.available:
        if require_ollama:
            raise RuntimeError("Ollama is required for retrieval query generation but is not configured.")
        return _heuristic_retrieval_queries(query=query, min_queries=min_queries, max_queries=max_queries)

    min_queries = max(1, min_queries)
    max_queries = max(min_queries, max_queries)

    prompt = (
        f"Decompose the user question into {min_queries} to {max_queries} retrieval subqueries for multi-hop causal reasoning. "
        "Each subquery must target an intermediate causal link, mechanism, or bridge fact. "
        "Return ONLY the subqueries, one per line, no explanation, no headings.\n\n"
        f"User question: {query}"
    )

    try:
        response = client.generate(
            prompt=prompt,
            system_prompt="You produce concise retrieval queries for multi-hop causal evidence gathering.",
            max_tokens=420,
        )
    except RuntimeError as exc:
        if require_ollama:
            raise RuntimeError(f"Ollama retrieval query generation failed: {exc}") from exc
        return _heuristic_retrieval_queries(query=query, min_queries=min_queries, max_queries=max_queries)

    queries = _parse_generated_queries(response)

    if len(queries) < min_queries:
        expansion_prompt = (
            f"You generated too few retrieval subqueries. Produce exactly {min_queries} distinct subqueries now. "
            "Return one subquery per line only.\n\n"
            f"User question: {query}\n"
            f"Already generated:\n" + "\n".join(queries)
        )
        try:
            expanded = client.generate(
                prompt=expansion_prompt,
                system_prompt="You generate distinct causal retrieval subqueries.",
                max_tokens=420,
            )
            queries = queries + _parse_generated_queries(expanded)
        except RuntimeError as exc:
            if require_ollama:
                raise RuntimeError(f"Ollama retrieval query expansion failed: {exc}") from exc

    deduplicated = list(dict.fromkeys(candidate for candidate in queries if candidate.strip()))
    if query not in deduplicated:
        deduplicated.insert(0, query)
    return deduplicated[:max_queries]


def _heuristic_retrieval_queries(query: str, min_queries: int, max_queries: int) -> List[str]:
    """Generate lightweight retrieval queries without calling the LLM."""

    decomposition = _fill_missing_intermediates(_heuristic_decomposition(query))
    candidates = [
        query,
        f"{query} causes",
        f"{query} mechanism",
        f"{query} evidence",
    ]

    for key in ("A", "B", "C", "D"):
        value = decomposition.get(key)
        if value:
            candidates.append(str(value))
            candidates.append(f"{value} cause effect")

    deduplicated = [item for item in dict.fromkeys(part.strip() for part in candidates if part and part.strip())]
    if len(deduplicated) < min_queries:
        deduplicated.extend([query] * (min_queries - len(deduplicated)))
    return deduplicated[:max_queries]


def _decompose_with_llm(query: str, client: LLMClient) -> Dict[str, Optional[str]]:
    """Ask the LLM to produce causal components labelled A-D."""

    prompt = (
        "Decompose the following question into up to four causal components labelled A, B, C, and D. "
        "A should be the initial cause and D should be the final outcome. If an intermediate step is unknown, "
        "leave it blank. Return only four lines in the form 'A: ...'.\n\n"
        f"Query: {query}"
    )
    response = client.generate(
        prompt=prompt,
        system_prompt="You extract concise causal components for multi-hop reasoning.",
        max_tokens=180,
    )

    components: Dict[str, Optional[str]] = {"A": None, "B": None, "C": None, "D": None}
    for line in response.splitlines():
        match = re.match(r"^\s*([ABCD])\s*:\s*(.*)\s*$", line)
        if match:
            value = match.group(2).strip() or None
            components[match.group(1)] = value
    if components["A"] and components["D"]:
        return components
    return _heuristic_decomposition(query)


def _parse_generated_queries(response: str) -> List[str]:
    """Parse one-query-per-line LLM output into clean retrieval queries."""

    queries: List[str] = []
    for raw_line in response.splitlines():
        cleaned_line = raw_line.strip()
        if not cleaned_line:
            continue

        chunks = [cleaned_line]
        if ";" in cleaned_line:
            chunks = [chunk.strip() for chunk in cleaned_line.split(";") if chunk.strip()]

        for chunk in chunks:
            cleaned = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", chunk).strip()
            cleaned = cleaned.strip('"').strip("'")
            if cleaned:
                queries.append(cleaned)
    return queries


def _heuristic_decomposition(query: str) -> Dict[str, Optional[str]]:
    """Fallback decomposition that extracts the source and target causal anchors."""

    normalized = query.strip().rstrip("?")
    causal_verbs = "increase|raise|elevate|cause|lead to|trigger|worsen|contribute to|result in|affect|effects|effect|influence|impact|drive|shape"
    patterns = [
        rf"why does (?P<a>.+?) (?:{causal_verbs}) (?P<d>.+)$",
        rf"how does (?P<a>.+?) (?:{causal_verbs}) (?P<d>.+)$",
        r"what causes (?P<d>.+)$",
        r"why is (?P<a>.+?) associated with (?P<d>.+)$",
    ]

    for pattern in patterns:
        match = re.match(pattern, normalized, flags=re.IGNORECASE)
        if match:
            if "a" not in match.groupdict() and "d" in match.groupdict():
                return {
                    "A": None,
                    "B": None,
                    "C": None,
                    "D": match.group("d").strip(),
                }
            return {
                "A": match.group("a").strip(),
                "B": None,
                "C": None,
                "D": match.group("d").strip(),
            }

    connector_split = re.split(r"\b(?:to|into|toward|towards)\b", normalized, maxsplit=1, flags=re.IGNORECASE)
    if len(connector_split) == 2:
        left = re.sub(r"^(how|why)\s+does\s+", "", connector_split[0], flags=re.IGNORECASE).strip()
        right = connector_split[1].strip()
        if left and right:
            return {"A": left, "B": None, "C": None, "D": right}

    fragments = [fragment.strip() for fragment in re.split(r"\b(?:because|therefore|due to|leading to)\b", normalized, maxsplit=1, flags=re.IGNORECASE)]
    if len(fragments) == 2:
        return {"A": fragments[0] or normalized, "B": None, "C": None, "D": fragments[1] or normalized}

    return {"A": normalized, "B": None, "C": None, "D": None}


def _fill_missing_intermediates(components: Dict[str, Optional[str]]) -> Dict[str, Optional[str]]:
    """Ensure intermediate decomposition steps are populated when endpoints are known."""

    normalized: Dict[str, Optional[str]] = {
        "A": components.get("A"),
        "B": components.get("B"),
        "C": components.get("C"),
        "D": components.get("D"),
    }

    a = normalized.get("A")
    b = normalized.get("B")
    c = normalized.get("C")
    d = normalized.get("D")

    # Only infer intermediate links when both endpoints are known.
    if not a or not d:
        return normalized

    if b is None and c is None:
        normalized["B"] = f"biological and environmental mechanisms connecting {a} to {d}"
        normalized["C"] = f"downstream physiological effects that directly increase {d}"
        return normalized

    if b is None:
        normalized["B"] = f"intermediate mechanisms linking {a} to {c}"

    if c is None:
        normalized["C"] = f"proximal effects linking {b} to {d}"

    return normalized
