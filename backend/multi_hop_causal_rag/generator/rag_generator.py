"""RAG answer generation from the causal chain and supporting evidence."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Mapping, Optional, Sequence

try:
    from ..config import AppConfig, build_default_config
    from ..llm_interface import LLMClient
    from ..multi_hop.chain_builder import EvidenceChain
except ImportError:
    # Support direct script execution (python multi_hop_causal_rag/generator/rag_generator.py)
    workspace_root = Path(__file__).resolve().parents[2]
    if str(workspace_root) not in sys.path:
        sys.path.insert(0, str(workspace_root))
    from multi_hop_causal_rag.config import AppConfig, build_default_config
    from multi_hop_causal_rag.llm_interface import LLMClient
    from multi_hop_causal_rag.multi_hop.chain_builder import EvidenceChain


# No word count limits - allow full answers


def generate_answer(
    query: str,
    causal_chain: EvidenceChain,
    app_config: Optional[AppConfig] = None,
    chat_history: Optional[Sequence[Mapping[str, str]]] = None,
) -> str:
    """Generate a final answer from the query, causal chain, and evidence."""

    config = app_config or build_default_config()
    client = LLMClient(config.llm)
    require_ollama = bool(config.llm.require_ollama)
    chain_text = " -> ".join(causal_chain.nodes)
    chain_steps_text = "\n".join(f"{idx}. {node}" for idx, node in enumerate(causal_chain.nodes, 1))
    graph_edges_text = "\n".join(
        f"- {triple.cause} --[{triple.relation}]--> {triple.effect} "
        f"(confidence={triple.confidence:.2f}, source={triple.source_doc_id})"
        for triple in causal_chain.triples[:12]
    )
    history_text = _format_chat_history(chat_history)

    if not client.available:
        if require_ollama:
            raise RuntimeError(
                "Ollama is required but not configured. Set MHC_RAG_LLM_BASE_URL and MHC_RAG_LLM_MODEL."
            )
    else:
        prompt = (
            "You are generating the final response for a multi-hop causal RAG chatbot. "
            "Use the retrieved causal chain, graph edges, and evidence as authoritative context. "
            "Answer like a knowledgeable assistant speaking directly to the user in natural conversation. "
            "State only well-supported facts from the provided context and avoid mentioning internal pipeline steps.\n\n"
            f"Recent conversation:\n{history_text}\n\n"
            f"Query: {query}\n"
            f"Retrieved causal chain (linear): {chain_text or '(none)'}\n"
            f"Retrieved causal chain (steps):\n{chain_steps_text or '(none)'}\n\n"
            f"Retrieved causal graph edges:\n{graph_edges_text or '(no extracted edges)'}\n\n"
            "Write a comprehensive, well-structured answer that fully explains the causal relationship. "
            "Use clear sentences with a logical flow. "
            "Do not output causal chain notation like 'A -> B', and do not list raw chain nodes. "
            "Do not use headings, bullet points, or section labels. "
            "Incorporate key terms from the retrieved chain as proof words in your answer. "
            "Provide complete details - do not truncate or abbreviate explanations. "
            "If the evidence is incomplete, say so briefly and cautiously."
        )
        try:
            raw_answer = client.generate(
                prompt=prompt,
                system_prompt="You are a factual multi-hop causal RAG chatbot. Answer conversationally using only supported evidence.",
                max_tokens=max(180, min(config.llm.max_output_tokens, 350)),
            )
            return _enforce_answer_constraints(
                answer=raw_answer,
                query=query,
                chain_nodes=causal_chain.nodes,
                evidence_texts=[result.document.text for result in causal_chain.evidence],
            )
        except RuntimeError as exc:
            if require_ollama:
                raise RuntimeError(f"Ollama generation failed: {exc}") from exc

    fallback = _fallback_answer(
        query=query,
        chain_nodes=causal_chain.nodes,
        evidence_texts=[result.document.text for result in causal_chain.evidence],
        chat_history=chat_history,
    )
    return _enforce_answer_constraints(
        answer=fallback,
        query=query,
        chain_nodes=causal_chain.nodes,
        evidence_texts=[result.document.text for result in causal_chain.evidence],
    )


def _fallback_answer(
    query: str,
    chain_nodes: List[str],
    evidence_texts: List[str],
    chat_history: Optional[Sequence[Mapping[str, str]]] = None,
) -> str:
    """Generate a deterministic answer when no LLM API is configured."""

    if chain_nodes:
        path_narrative = _build_chain_narrative(chain_nodes)
        leading_evidence = _clip_text(evidence_texts[0], limit=360) if evidence_texts else "The available evidence suggests a causal relationship."
        return (
            f"Based on the retrieved causal evidence, {path_narrative}. "
            f"The strongest supporting evidence indicates that {leading_evidence}."
        )
    return f"Based on available evidence, a complete causal chain could not be constructed to answer your question. Please try rephrasing or provide additional context."


def _clip_text(text: str, limit: int = 400) -> str:
    """Trim long evidence snippets to keep the generation prompt focused."""

    clean = " ".join(text.split())
    if len(clean) <= limit:
        return clean
    # Clip at word boundary, not character boundary
    clipped = clean[:limit]
    last_space = clipped.rfind(" ")
    if last_space > limit * 0.8:  # Only use word break if it's relatively close
        return clipped[:last_space].rstrip() + "..."
    return clipped.rstrip() + "..."


def _format_chat_history(chat_history: Optional[Sequence[Mapping[str, str]]]) -> str:
    """Serialize recent chat turns for conversational answer generation."""

    if not chat_history:
        return "(no prior conversation)"

    lines: List[str] = []
    for turn in chat_history[-4:]:
        user_text = (turn.get("user") or "").strip()
        assistant_text = (turn.get("assistant") or "").strip()
        if user_text:
            lines.append(f"User: {_clip_text(user_text, limit=180)}")
        if assistant_text:
            lines.append(f"Assistant: {_clip_text(assistant_text, limit=240)}")
    return "\n".join(lines) if lines else "(no prior conversation)"


def _build_chain_narrative(chain_nodes: Sequence[str]) -> str:
    """Convert a chain node list into natural language instead of arrow notation."""

    cleaned_nodes = [node.strip() for node in chain_nodes if node and node.strip()]
    if not cleaned_nodes:
        return "the available evidence does not provide a clear causal progression"
    if len(cleaned_nodes) == 1:
        return f"the central factor is {cleaned_nodes[0]}"
    if len(cleaned_nodes) == 2:
        return f"{cleaned_nodes[0]} contributes to {cleaned_nodes[1]}"

    opening = cleaned_nodes[0]
    ending = cleaned_nodes[-1]
    middle = cleaned_nodes[1:-1]
    middle_clause = ", then ".join(middle[:3])
    if middle_clause:
        return f"{opening} influences {middle_clause}, which ultimately contributes to {ending}"
    return f"{opening} ultimately contributes to {ending}"


def _enforce_answer_constraints(
    answer: str,
    query: str,
    chain_nodes: Sequence[str],
    evidence_texts: Sequence[str],
) -> str:
    """Clean and format answer without imposing word limits."""

    paragraph = " ".join((answer or "").split())
    if not paragraph:
        paragraph = f"Based on retrieved evidence, the answer involves multiple causal steps that are interconnected."

    if paragraph and paragraph[-1] not in ".!?":
        paragraph = paragraph + "."
    return paragraph

