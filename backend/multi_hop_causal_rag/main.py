"""Command-line entry point for the multi-hop causal RAG system."""

from __future__ import annotations

import io
import os
import sys
import textwrap
import webbrowser
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, Dict, List

# Add parent directory to path so script can be run directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multi_hop_causal_rag.config import AppConfig, build_default_config
from multi_hop_causal_rag.graph.causal_graph import CausalKnowledgeGraph
from multi_hop_causal_rag.graph.graph_visualizer import CausalGraphVisualizer
from multi_hop_causal_rag.pipeline.multi_hop_pipeline import MultiHopCausalPipeline
from multi_hop_causal_rag.pipeline.query_processor import receive_query
from multi_hop_causal_rag.retrieval.retriever import CausalRetriever


# ── display constants ─────────────────────────────────────────────────────────
_WIDTH = 74


def _fmt_value(val: Any, depth: int = 0) -> str:
    """Recursively format a detail value into a readable string."""
    indent = "    " + "  " * depth
    if val is None:
        return "(none)"
    if isinstance(val, list):
        if not val:
            return "(empty)"
        items = []
        for i, item in enumerate(val, 1):
            if isinstance(item, dict):
                kv = "  ".join(f"{k}={v}" for k, v in item.items())
                items.append(f"{indent}[{i}] {kv}")
            else:
                items.append(f"{indent}- {item}")
        return "\n".join(items)
    if isinstance(val, dict):
        lines = []
        for k, v in val.items():
            formatted = _fmt_value(v, depth + 1)
            if "\n" in formatted:
                lines.append(f"{indent}{k}:")
                lines.append(formatted)
            else:
                lines.append(f"{indent}{k}: {formatted}")
        return "\n".join(lines)
    return str(val)


def _default_confidence_text(
    causal_path: List[str],
    source_ids: List[str],
    consistency: Dict[str, Any],
) -> str:
    """Generate a default confidence summary from provenance signals."""

    unsupported = consistency.get("unsupported_edges", [])
    contradictory = consistency.get("contradictory_edges", [])
    if consistency.get("is_consistent") and not unsupported and not contradictory:
        return "High: retrieved chain passed the consistency checks."
    if causal_path or source_ids:
        return "Medium: answer grounded in retrieved evidence, but the chain is incomplete or partially validated."
    return "Low: limited causal evidence was available for this query."


def _extract_answer_sections(answer: str, provenance: Dict[str, Any]) -> Dict[str, str]:
    """Normalize the model answer into stable display sections."""

    raw_answer = (answer or "").strip()
    section_aliases = {
        "final answer": "Final Answer",
        "causal reasoning": "Causal Reasoning",
        "evidence used": "Evidence Used",
        "confidence": "Confidence",
    }
    sections: Dict[str, str] = {}
    current_section: str | None = None

    for raw_line in raw_answer.splitlines():
        line = raw_line.strip()
        matched_section = None
        matched_value = ""
        for alias, canonical in section_aliases.items():
            prefix = f"{alias}:"
            if line.lower().startswith(prefix):
                matched_section = canonical
                matched_value = line[len(prefix):].strip()
                break

        if matched_section is not None:
            current_section = matched_section
            existing = sections.get(current_section, "")
            sections[current_section] = "\n".join(part for part in [existing, matched_value] if part).strip()
            continue

        if current_section is not None:
            existing = sections.get(current_section, "")
            sections[current_section] = "\n".join(part for part in [existing, line] if part).strip()

    causal_path = provenance.get("causal_path", []) or []
    source_ids = provenance.get("document_sources", []) or []
    validations = provenance.get("validations", {}) or {}
    consistency = validations.get("consistency", {}) if isinstance(validations, dict) else {}

    defaults = {
        "Final Answer": raw_answer or "(no answer generated)",
        "Causal Reasoning": " -> ".join(causal_path) if causal_path else "(no causal chain established)",
        "Evidence Used": ", ".join(f"[{source_id}]" for source_id in source_ids[:8]) if source_ids else "(no evidence sources recorded)",
        "Confidence": _default_confidence_text(causal_path, source_ids, consistency if isinstance(consistency, dict) else {}),
    }
    for title, fallback in defaults.items():
        if not sections.get(title):
            sections[title] = fallback

    return sections


def _print_chat_response(
    answer: str,
    provenance: Dict[str, Any],
    candidate_paths: List[List[str]] | None = None,
) -> None:
    """Render only the final user-facing response in chat format."""

    final_answer = _extract_answer_sections(answer, provenance).get("Final Answer", "").strip() or (answer or "(no answer generated)")
    wrapped_lines = textwrap.wrap(final_answer, width=_WIDTH - 8) or [final_answer]
    retrieved_chains = provenance.get("retrieved_causal_chains", []) or candidate_paths or []

    normalized_chains: List[List[str]] = []
    for chain in retrieved_chains:
        if isinstance(chain, list):
            nodes = [str(node).strip() for node in chain if str(node).strip()]
            if len(nodes) >= 2:
                normalized_chains.append(nodes)

    if not normalized_chains:
        for edge_text in provenance.get("evidence_chain", []) or []:
            if not isinstance(edge_text, str):
                continue
            if "->" not in edge_text:
                continue
            nodes = [part.strip() for part in edge_text.split("->") if part.strip()]
            if len(nodes) >= 2:
                normalized_chains.append(nodes)

    print()
    print("Assistant:")
    for line in wrapped_lines:
        print(f"  {line}")

    print("  Retrieved causal chains:")
    if normalized_chains:
        for idx, chain_nodes in enumerate(normalized_chains, 1):
            chain_text = " -> ".join(chain_nodes)
            wrapped_chain = textwrap.wrap(chain_text, width=_WIDTH - 8) or [chain_text]
            for line_idx, line in enumerate(wrapped_chain):
                prefix = f"    [{idx}] " if line_idx == 0 else " " * 8
                print(f"{prefix}{line}")
    else:
        fallback_path = [str(node).strip() for node in (provenance.get("causal_path", []) or []) if str(node).strip()]
        if len(fallback_path) >= 2:
            chain_text = " -> ".join(fallback_path)
            wrapped_chain = textwrap.wrap(chain_text, width=_WIDTH - 8) or [chain_text]
            for line_idx, line in enumerate(wrapped_chain):
                prefix = "    [1] " if line_idx == 0 else " " * 8
                print(f"{prefix}{line}")
        else:
            print("    (no causal chain retrieved)")

    # print("  Provenance:")
    # for key in (
    #     "causal_path",
    #     "retrieved_causal_chains",
    #     "evidence_chain",
    #     "document_sources",
    #     "source_details",
    #     "validations",
    # ):
    #     value = provenance.get(key)
    #     rendered = _fmt_value(value)
    #     if "\n" in rendered:
    #         print(f"    {key}:")
    #         for line in rendered.splitlines():
    #             print(line)
    #     else:
    #         print(f"    {key}: {rendered}")
    # print()


def _is_visualization_request(query: str) -> bool:
    """Return whether the user asked for dataset/graph visualization."""

    normalized = " ".join(query.lower().split())
    trigger_phrases = {
        "visualize dataset",
        "visualise graph",
        "show dataset graph",
        "show causal graph",
        "plot dataset",
    }
    return normalized in trigger_phrases or normalized.startswith("/visualize")


def _render_dataset_visualization(graph: CausalKnowledgeGraph) -> None:
    """Generate visualization files for the current causal graph and open HTML output."""

    if graph.graph.number_of_nodes() == 0:
        print("Assistant:")
        print("  The dataset graph is currently empty. Ask at least one causal question first, then request visualization.")
        return

    output_dir = Path(__file__).parent / "graph_outputs"
    html_path = output_dir / "causal_graph.html"
    json_path = output_dir / "causal_graph.json"

    visualizer = CausalGraphVisualizer(graph.graph)
    visualizer.save_json(str(json_path))
    html_created = False
    try:
        visualizer.save_interactive_html(str(html_path))
        html_created = True
    except RuntimeError:
        html_created = False

    print("Assistant:")
    if html_created:
        print(f"  Visualization generated: {html_path}")
        print(f"  Graph data exported: {json_path}")
        webbrowser.open(html_path.resolve().as_uri())
        print("  Opened the interactive graph in your browser.")
    else:
        print(f"  Interactive HTML visualization could not be generated (pyvis not installed).")
        print(f"  Graph data exported: {json_path}")


# ── main entry point ──────────────────────────────────────────────────────────


def run() -> None:
    """Run the multi-hop causal RAG chatbot from the terminal."""

    config = build_default_config()
    project_root = Path(__file__).resolve().parent.parent
    config.retrieval.faiss_index_path = str(project_root / "faiss_index_512" / "index.faiss")
    config.retrieval.faiss_docstore_path = str(project_root / "faiss_index_512" / "index.pkl")

    index_file = Path(config.retrieval.faiss_index_path)
    docstore_file = Path(config.retrieval.faiss_docstore_path)
    if not index_file.exists() or not docstore_file.exists():
        raise RuntimeError(
            "FAISS retrieval files are required. "
            "Expected files in faiss_index_512: index.faiss and index.pkl."
        )

    retriever = CausalRetriever(config=config.retrieval)
    retriever.initialize_with_documents([])

    pipeline = MultiHopCausalPipeline(retriever=retriever, config=config)
    state_path = Path(__file__).parent / "graph_outputs" / "causal_graph_state.json"
    if pipeline.knowledge_graph.load_state(str(state_path)):
        print(f"Loaded existing graph state: {state_path}")
    chat_history: List[Dict[str, str]] = []
    print("Multi-Hop Causal RAG Chatbot")
    if config.llm.base_url and config.llm.model:
        print(f"Mode: Ollama RAG chat ({config.llm.model} @ {config.llm.base_url})")
    else:
        print("Mode: fallback RAG chat (Ollama base URL/model missing)")
    print("Type 'exit' or 'quit' to stop.")
    print("Type '/visualize' (or 'visualize dataset') to open graph visualization.")

    while True:
        try:
            query = receive_query(prompt_text="You: ")
        except (EOFError, KeyboardInterrupt):
            print("\nSession ended.")
            break

        if query.lower() in {"exit", "quit"}:
            print("Session ended.")
            break

        if _is_visualization_request(query):
            _render_dataset_visualization(pipeline.knowledge_graph)
            continue

        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            result = pipeline.run(query, chat_history=chat_history)
        pipeline.knowledge_graph.save_state(str(state_path))
        _print_chat_response(result.answer, result.provenance, result.candidate_paths)
        chat_history.append({"user": query, "assistant": result.answer})
        if len(chat_history) > 6:
            chat_history = chat_history[-6:]


def run_benchmark() -> None:
    """Execute the benchmark workflow from the main entrypoint."""

    from .evaluation.benchmark_runner import run_benchmark as execute_benchmark

    execute_benchmark()


if __name__ == "__main__":
    if "--benchmark" in {arg.strip().lower() for arg in sys.argv[1:]}:
        run_benchmark()
    else:
        run()
