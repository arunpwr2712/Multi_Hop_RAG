"""Visualization helpers for causal graphs.

This module is standalone and does not modify pipeline behavior.
Use it after running the pipeline, for example:

    from multi_hop_causal_rag.pipeline.multi_hop_pipeline import MultiHopCausalPipeline
    from multi_hop_causal_rag.graph.graph_visualizer import CausalGraphVisualizer

    # ... run pipeline ...
    visualizer = CausalGraphVisualizer(pipeline.knowledge_graph.graph)
    visualizer.save_static_png("causal_graph.png")
    visualizer.save_interactive_html("causal_graph.html")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import networkx as nx


class CausalGraphVisualizer:
    """Render and export a causal graph without changing retrieval logic."""

    def __init__(self, graph: nx.DiGraph) -> None:
        """Initialize with an already-built directed causal graph."""

        self.graph = graph

    def summary(self) -> Dict[str, Any]:
        """Return basic graph statistics."""

        return {
            "nodes": int(self.graph.number_of_nodes()),
            "edges": int(self.graph.number_of_edges()),
            "is_directed": bool(self.graph.is_directed()),
            "is_acyclic": bool(nx.is_directed_acyclic_graph(self.graph)),
        }

    def save_static_png(
        self,
        output_path: str,
        figsize: tuple[int, int] = (16, 10),
        with_labels: bool = True,
        node_size: int = 1400,
        font_size: int = 8,
        seed: int = 42,
    ) -> str:
        """Save a static PNG visualization using matplotlib."""

        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise RuntimeError("matplotlib is required for PNG visualization") from exc

        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=figsize)
        if self.graph.number_of_nodes() == 0:
            plt.text(0.5, 0.5, "Graph is empty", ha="center", va="center", fontsize=14)
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(target, dpi=180)
            plt.close()
            return str(target)

        pos = nx.spring_layout(self.graph, k=1.0, iterations=200, seed=seed)

        nx.draw_networkx_nodes(
            self.graph,
            pos,
            node_size=node_size,
            node_color="#d9edf7",
            edgecolors="#4a6fa5",
            linewidths=1.0,
        )
        nx.draw_networkx_edges(
            self.graph,
            pos,
            arrows=True,
            arrowstyle="-|>",
            arrowsize=16,
            width=1.2,
            edge_color="#37474f",
            connectionstyle="arc3,rad=0.05",
        )

        if with_labels:
            wrapped_labels = {node: self._truncate(str(node), 42) for node in self.graph.nodes()}
            nx.draw_networkx_labels(self.graph, pos, labels=wrapped_labels, font_size=font_size)

        edge_labels = {}
        for u, v, data in self.graph.edges(data=True):
            relation = str(data.get("relation", "causes"))
            confidence = data.get("confidence")
            if isinstance(confidence, (int, float)):
                edge_labels[(u, v)] = f"{relation} ({confidence:.2f})"
            else:
                edge_labels[(u, v)] = relation

        if edge_labels:
            nx.draw_networkx_edge_labels(
                self.graph,
                pos,
                edge_labels=edge_labels,
                font_size=max(6, font_size - 1),
                rotate=False,
                label_pos=0.55,
            )

        plt.title("Causal Knowledge Graph")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(target, dpi=180)
        plt.close()
        return str(target)

    def save_interactive_html(self, output_path: str, height: str = "800px", width: str = "100%") -> str:
        """Save an interactive HTML visualization using pyvis."""

        try:
            from pyvis.network import Network
        except ImportError as exc:
            raise RuntimeError("pyvis is required for interactive HTML visualization") from exc

        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)

        net = Network(height=height, width=width, directed=True, bgcolor="#ffffff", font_color="#111111")
        net.barnes_hut(gravity=-12000, central_gravity=0.1, spring_length=170, spring_strength=0.03, damping=0.09)

        for node in self.graph.nodes():
            label = self._truncate(str(node), 60)
            net.add_node(str(node), label=label, title=str(node), color="#8ecae6")

        for u, v, data in self.graph.edges(data=True):
            relation = str(data.get("relation", "causes"))
            confidence = data.get("confidence")
            title = relation
            edge_label = relation
            if isinstance(confidence, (int, float)):
                title = f"{relation} | confidence={confidence:.3f}"
                edge_label = f"{relation} ({confidence:.2f})"

            net.add_edge(str(u), str(v), label=edge_label, title=title, arrows="to", color="#455a64")

        net.save_graph(str(target))
        return str(target)

    def save_json(self, output_path: str) -> str:
        """Export graph data as JSON for custom visualization pipelines."""

        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "summary": self.summary(),
            "nodes": [{"id": str(node)} for node in self.graph.nodes()],
            "edges": [
                {
                    "source": str(u),
                    "target": str(v),
                    "relation": str(data.get("relation", "causes")),
                    "confidence": data.get("confidence"),
                    "evidence_count": len(data.get("evidence", [])) if isinstance(data.get("evidence", []), list) else 0,
                }
                for u, v, data in self.graph.edges(data=True)
            ],
        }

        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return str(target)

    @staticmethod
    def _truncate(text: str, limit: int) -> str:
        """Shorten long labels to keep plots readable."""

        clean = " ".join(text.split())
        if len(clean) <= limit:
            return clean
        return clean[: limit - 3] + "..."

