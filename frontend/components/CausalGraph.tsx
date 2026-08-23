"use client";

import { useEffect, useRef } from "react";
import { DataSet } from "vis-data";
import { Network } from "vis-network";

import { GraphResponse } from "@/types/rag";

type CausalGraphProps = {
  graph: GraphResponse | null;
};

export default function CausalGraph({ graph }: CausalGraphProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!containerRef.current || !graph) {
      return;
    }

    const nodes = new DataSet(
      graph.nodes.map((node) => ({
        id: node.id,
        label: node.label,
        title: node.label,
        shape: "dot",
        size: 18,
        color: {
          background: "#4fb0c6",
          border: "#0f4c5c",
          highlight: {
            background: "#f4d35e",
            border: "#0f4c5c",
          },
        },
        font: {
          color: "#0a1f24",
          size: 14,
        },
      }))
    );

    const edges = new DataSet(
      graph.edges.map((edge, index) => {
        const confidence = typeof edge.confidence === "number" ? edge.confidence.toFixed(2) : "NA";
        return {
          id: `${edge.source}-${edge.target}-${index}`,
          from: edge.source,
          to: edge.target,
          label: `${edge.relation} (${confidence})`,
          arrows: "to",
          color: {
            color: "#1b3a4b",
            highlight: "#ee964b",
          },
          font: {
            color: "#102a43",
            size: 11,
            align: "top",
          },
          width: 1.4,
        };
      })
    );

    const network = new Network(
      containerRef.current,
      { nodes, edges },
      {
        autoResize: true,
        layout: {
          improvedLayout: true,
        },
        interaction: {
          hover: true,
          navigationButtons: true,
          keyboard: true,
          tooltipDelay: 150,
        },
        physics: {
          barnesHut: {
            gravitationalConstant: -8000,
            springLength: 150,
            springConstant: 0.03,
            damping: 0.14,
          },
          stabilization: {
            iterations: 200,
          },
        },
      }
    );

    return () => {
      network.destroy();
    };
  }, [graph]);

  if (!graph || graph.summary.nodes === 0) {
    return <div className="graph-empty">No graph data yet. Ask a question to build causal edges.</div>;
  }

  return <div ref={containerRef} className="graph-canvas" />;
}
