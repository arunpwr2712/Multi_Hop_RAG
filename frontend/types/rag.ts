export type ChatResponse = {
  query: string;
  answer: string;
  decomposition: Record<string, string | null>;
  retrieval_queries: string[];
  candidate_paths: string[][];
  provenance: Record<string, unknown>;
  trace_steps: Array<Record<string, unknown>>;
};

export type GraphSummary = {
  nodes: number;
  edges: number;
  is_directed: boolean;
  is_acyclic: boolean;
};

export type GraphNode = {
  id: string;
  label: string;
};

export type GraphEdge = {
  source: string;
  target: string;
  relation: string;
  confidence: number | null;
};

export type GraphResponse = {
  summary: GraphSummary;
  nodes: GraphNode[];
  edges: GraphEdge[];
};

export type ReadinessResponse = {
  backend: boolean;
  ollama: boolean;
  ready: boolean;
};
