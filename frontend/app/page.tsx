"use client";

import { FormEvent, useEffect, useMemo, useRef, useState } from "react";

import CausalGraph from "@/components/CausalGraph";
import { checkReadiness, fetchGraph, resetSession, sendChat } from "@/lib/api";
import { ChatResponse, GraphResponse, ReadinessResponse } from "@/types/rag";

type Turn = {
  id: string;
  query: string;
  response: ChatResponse;
};

export default function HomePage() {
  const [query, setQuery] = useState("");
  const [turns, setTurns] = useState<Turn[]>([]);
  const [graph, setGraph] = useState<GraphResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [insightsOpen, setInsightsOpen] = useState(false);
  const [insightsFullscreen, setInsightsFullscreen] = useState(false);
  const [insightTab, setInsightTab] = useState<"chain" | "graph">("chain");
  const [readiness, setReadiness] = useState<ReadinessResponse | null>(null);
  const [readinessLoading, setReadinessLoading] = useState(true);
  const readinessChecked = useRef(false);

  const latest = turns.length ? turns[turns.length - 1].response : null;

  useEffect(() => {
    if (readinessChecked.current) return;
    readinessChecked.current = true;

    const checkStatus = async () => {
      setReadinessLoading(true);
      try {
        const status = await checkReadiness();
        setReadiness(status);
      } catch (err) {
        console.error("Failed to check readiness:", err);
        setReadiness({ backend: false, ollama: false, ready: false });
      } finally {
        setReadinessLoading(false);
      }
    };

    checkStatus();
  }, []);

  const causalPaths = useMemo(() => {
    if (!latest) {
      return [] as string[][];
    }
    return latest.candidate_paths ?? [];
  }, [latest]);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!query.trim()) {
      return;
    }

    if (!readiness?.ready) {
      setError("Backend or Ollama is not ready yet. Please wait...");
      return;
    }

    const q = query.trim();
    setBusy(true);
    setError(null);

    try {
      const response = await sendChat(q);
      setTurns((current) => [...current, { id: crypto.randomUUID(), query: q, response }]);
      setQuery("");

      const graphPayload = await fetchGraph();
      setGraph(graphPayload);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Unexpected error while querying backend";
      setError(message);
    } finally {
      setBusy(false);
    }
  }

  async function handleRefreshGraph() {
    setBusy(true);
    setError(null);
    try {
      const graphPayload = await fetchGraph();
      setGraph(graphPayload);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Could not refresh graph";
      setError(message);
    } finally {
      setBusy(false);
    }
  }

  async function handleReset() {
    setBusy(true);
    setError(null);
    try {
      await resetSession();
      setTurns([]);
      setGraph(null);
      setQuery("");
    } catch (err) {
      const message = err instanceof Error ? err.message : "Could not reset session";
      setError(message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="chat-shell">
      <header className="chat-header">
        <div>
          <p className="eyebrow">Causal Assistant</p>
          <h1>Multi-hop Causal RAG</h1>
        </div>
        
        <div className="readiness-status">
          {readinessLoading ? (
            <p style={{ color: "#999", fontSize: "0.9em" }}>Checking server status...</p>
          ) : readiness?.ready ? (
            <p style={{ color: "#4caf50", fontSize: "0.9em" }}>✓ Server Ready</p>
          ) : (
            <p style={{ color: "#ff6b6b", fontSize: "0.9em" }}>
              ✗ {!readiness?.backend ? "Backend " : ""}{!readiness?.ollama ? "Ollama " : ""}offline
            </p>
          )}
        </div>

        <div className="header-actions">
          <button
            onClick={() => {
              setInsightsOpen((open) => {
                if (open && insightsFullscreen) {
                  setInsightsFullscreen(false);
                }
                return !open;
              });
            }}
            className="ghost-button"
            disabled={busy}
            type="button"
          >
            {insightsOpen ? "Hide Insights" : "Show Insights"}
          </button>
          {insightsOpen ? (
            <button
              onClick={() => setInsightsFullscreen((full) => !full)}
              className="ghost-button"
              disabled={busy}
              type="button"
            >
              {insightsFullscreen ? "Exit Full Screen" : "Full Screen"}
            </button>
          ) : null}
          <button onClick={handleReset} className="ghost-button" disabled={busy} type="button">
            Reset Chat
          </button>
        </div>
      </header>

      <section
        className={`chat-layout ${insightsOpen ? "with-insights" : ""} ${insightsFullscreen ? "insights-fullscreen" : ""}`}
      >
        <article className={`chat-panel ${insightsFullscreen ? "hidden-in-fullscreen" : ""}`}>
          <div className="chat-scroll">
            {turns.length === 0 ? (
              <div className="empty-state">
                <h2>Ask anything causal</h2>
                <p>
                  Try: "How does increased atmospheric CO2 influence ocean acidity and marine ecosystems?"
                </p>
              </div>
            ) : null}

            {turns.map((turn) => (
              <div className="turn-stack" key={turn.id}>
                <div className="bubble user-bubble">{turn.query}</div>
                <div className="bubble assistant-bubble">{turn.response.answer}</div>
              </div>
            ))}

            {busy ? <div className="bubble assistant-bubble thinking">Thinking...</div> : null}
          </div>

          {error ? <p className="error-banner">{error}</p> : null}

          <form onSubmit={handleSubmit} className="composer">
            <textarea
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="Message the causal assistant..."
              rows={2}
              disabled={busy || !readiness?.ready}
            />
            <button type="submit" className="primary-button" disabled={busy || !query.trim() || !readiness?.ready}>
              Send
            </button>
          </form>
        </article>

        {insightsOpen ? (
          <aside className={`insights-panel ${insightsFullscreen ? "is-fullscreen" : ""}`}>
            <div className="insights-tabs" role="tablist" aria-label="Insights tabs">
              <button
                type="button"
                className={`tab-button ${insightTab === "chain" ? "active" : ""}`}
                onClick={() => setInsightTab("chain")}
              >
                Causal Chain
              </button>
              <button
                type="button"
                className={`tab-button ${insightTab === "graph" ? "active" : ""}`}
                onClick={() => setInsightTab("graph")}
              >
                Graph
              </button>
            </div>

            {insightTab === "chain" ? (
              <div className="insight-content">
                {!latest ? <p className="hint">Run a query to inspect candidate chains and retrieval planning.</p> : null}

                {latest ? (
                  <>
                    <div className="chip-row">
                      {(latest.retrieval_queries || []).map((item, index) => (
                        <span className="chip" key={`${item}-${index}`}>
                          {item}
                        </span>
                      ))}
                    </div>

                    <div className="path-list">
                      {causalPaths.length === 0 ? <p className="hint">No candidate path returned.</p> : null}
                      {causalPaths.map((path, index) => (
                        <p key={`${path.join("->")}-${index}`} className="path-item">
                          {index + 1}. {path.join(" -> ")}
                        </p>
                      ))}
                    </div>

                    <details>
                      <summary>Trace Steps</summary>
                      <pre>{JSON.stringify(latest.trace_steps, null, 2)}</pre>
                    </details>
                  </>
                ) : null}
              </div>
            ) : (
              <div className="insight-content">
                <div className="panel-header">
                  <h2>Knowledge Graph</h2>
                  <button onClick={handleRefreshGraph} className="ghost-button" disabled={busy} type="button">
                    Refresh
                  </button>
                </div>

                {graph ? (
                  <p className="graph-meta">
                    Nodes: {graph.summary.nodes} | Edges: {graph.summary.edges} | DAG: {graph.summary.is_acyclic ? "Yes" : "No"}
                  </p>
                ) : (
                  <p className="hint">Graph not loaded yet.</p>
                )}
                <CausalGraph graph={graph} />
              </div>
            )}
          </aside>
        ) : null}
      </section>
    </main>
  );
}
