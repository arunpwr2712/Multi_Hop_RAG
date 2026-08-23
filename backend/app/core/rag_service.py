from __future__ import annotations

import sys
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from multi_hop_causal_rag.config import build_default_config
from multi_hop_causal_rag.graph.graph_visualizer import CausalGraphVisualizer
from multi_hop_causal_rag.pipeline.multi_hop_pipeline import MultiHopCausalPipeline
from multi_hop_causal_rag.retrieval.retriever import CausalRetriever
from multi_hop_causal_rag.llm_interface import LLMClient


class RAGService:
    """Stateful service that wraps the existing multi-hop pipeline without changing logic."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._chat_history: List[Dict[str, str]] = []
        self._last_result: Dict[str, Any] | None = None

        config = build_default_config()
        self._config = config  # Store config for readiness checks
        # Keep API responses available even when Ollama is down or returns 5xx.
        config.llm.require_ollama = False
        # Reduce avoidable latency in API mode when local LLM is unstable.
        config.llm.query_planning_enabled = False
        config.llm.timeout_seconds = min(int(config.llm.timeout_seconds), 8)
        config.retrieval.faiss_index_path = str(PROJECT_ROOT / "faiss_index_512" / "index.faiss")
        config.retrieval.faiss_docstore_path = str(PROJECT_ROOT / "faiss_index_512" / "index.pkl")

        retriever = CausalRetriever(config=config.retrieval)
        retriever.initialize_with_documents([])

        self._pipeline = MultiHopCausalPipeline(retriever=retriever, config=config)
        self._state_path = PROJECT_ROOT / "multi_hop_causal_rag" / "graph_outputs" / "causal_graph_state.json"
        self._pipeline.knowledge_graph.load_state(str(self._state_path))

    def run_query(self, query: str, keep_last_turns: int = 6) -> Dict[str, Any]:
        """Run a user query through the pipeline and return structured payload."""

        with self._lock:
            result = self._pipeline.run(query=query, chat_history=self._chat_history)
            self._pipeline.knowledge_graph.save_state(str(self._state_path))

            self._chat_history.append({"user": query, "assistant": result.answer})
            if len(self._chat_history) > keep_last_turns:
                self._chat_history = self._chat_history[-keep_last_turns:]

            self._last_result = {
                "query": result.query,
                "answer": result.answer,
                "candidate_paths": result.candidate_paths,
                "provenance": result.provenance,
                "trace_steps": result.trace_steps,
                "decomposition": result.decomposition,
                "retrieval_queries": result.retrieval_queries,
            }
            return self._last_result

    def get_last_result(self) -> Dict[str, Any] | None:
        return self._last_result

    def is_ready(self) -> Dict[str, bool]:
        """Check if the backend and Ollama are ready for querying."""
        backend_ready = True
        llm_client = LLMClient(self._config.llm)
        ollama_ready = llm_client.is_ollama_ready()
        
        return {
            "backend": backend_ready,
            "ollama": ollama_ready,
            "ready": backend_ready and ollama_ready,
        }

    def reset_chat(self) -> None:
        with self._lock:
            self._chat_history = []
            self._last_result = None

    def get_graph_payload(self) -> Dict[str, Any]:
        graph = self._pipeline.knowledge_graph.graph
        visualizer = CausalGraphVisualizer(graph)
        nodes = [{"id": str(node), "label": str(node)} for node in graph.nodes()]
        edges = []

        for source, target, data in graph.edges(data=True):
            edges.append(
                {
                    "source": str(source),
                    "target": str(target),
                    "relation": str(data.get("relation", "causes")),
                    "confidence": data.get("confidence"),
                }
            )

        return {
            "summary": visualizer.summary(),
            "nodes": nodes,
            "edges": edges,
        }


service = RAGService()
