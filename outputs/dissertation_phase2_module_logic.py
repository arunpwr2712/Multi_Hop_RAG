"""
Dissertation Phase-2 Implementation Snippets (Condensed)

This file captures only the core logic used in the project for:
4.3.1 Query Decomposition Module
4.3.2 Embedding and Retrieval Module
4.3.3 Causal Extraction Module
4.3.4 Causal Knowledge Graph Construction
4.3.5 Multi-Hop Retrieval Engine
4.3.6 Evidence Aggregation Module
4.3.7 Generation Module

Note: This is intentionally compact pseudocode-style Python derived from the
actual implementation, not the full runnable codebase.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence


# -----------------------------------------------------------------------------
# 4.3.1 Query Decomposition Module
# -----------------------------------------------------------------------------

def decompose_query_llm(query: str, llm_client) -> Dict[str, Optional[str]]:
	"""
	Core logic used in project:
	1) Try LLM-based decomposition into A, B, C, D.
	2) If LLM unavailable/fails, use heuristic decomposition.
	3) If A and D exist but B/C missing, infer intermediate placeholders.
	"""
	if llm_client.available:
		try:
			response = llm_client.generate(
				prompt=(
					"Decompose question into causal components A,B,C,D. "
					"A = root cause, D = final outcome. Return only lines: A:, B:, C:, D:."
				)
			)
			parts = {"A": None, "B": None, "C": None, "D": None}
			for line in response.splitlines():
				m = re.match(r"^\s*([ABCD])\s*:\s*(.*)\s*$", line)
				if m:
					parts[m.group(1)] = m.group(2).strip() or None
		except Exception:
			parts = _heuristic_decomposition(query)
	else:
		parts = _heuristic_decomposition(query)

	return _fill_missing_intermediates(parts)


def generate_retrieval_queries_llm(query: str, llm_client, min_q: int = 5, max_q: int = 10) -> List[str]:
	"""
	Project logic:
	- Generate 5-10 retrieval subqueries via LLM.
	- Parse one-query-per-line output.
	- Deduplicate and force-include original query.
	- Fallback to heuristic query expansion if needed.
	"""
	queries: List[str] = []
	if llm_client.available:
		try:
			raw = llm_client.generate(
				prompt=(
					f"Generate {min_q}-{max_q} retrieval subqueries for multi-hop causal reasoning. "
					"One per line, no explanation."
				)
			)
			queries = _parse_queries(raw)
		except Exception:
			queries = _heuristic_retrieval_queries(query)
	else:
		queries = _heuristic_retrieval_queries(query)

	unique = list(dict.fromkeys([q for q in queries if q.strip()]))
	if query not in unique:
		unique.insert(0, query)
	return unique[:max_q]


def _heuristic_decomposition(query: str) -> Dict[str, Optional[str]]:
	q = query.strip().rstrip("?")
	m = re.match(r"(?i)^how does (.+?) (?:cause|lead to|increase) (.+)$", q)
	if m:
		return {"A": m.group(1).strip(), "B": None, "C": None, "D": m.group(2).strip()}
	return {"A": q, "B": None, "C": None, "D": None}


def _fill_missing_intermediates(parts: Dict[str, Optional[str]]) -> Dict[str, Optional[str]]:
	a, b, c, d = parts.get("A"), parts.get("B"), parts.get("C"), parts.get("D")
	if a and d:
		if b is None:
			parts["B"] = f"intermediate mechanisms linking {a} to {d}"
		if c is None:
			parts["C"] = f"downstream effects leading to {d}"
	return parts


def _parse_queries(raw: str) -> List[str]:
	out: List[str] = []
	for line in raw.splitlines():
		cleaned = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line.strip())
		if cleaned:
			out.append(cleaned)
	return out


def _heuristic_retrieval_queries(query: str) -> List[str]:
	return [query, f"{query} causes", f"{query} mechanism", f"{query} evidence"]


# -----------------------------------------------------------------------------
# 4.3.2 Embedding and Retrieval Module
# -----------------------------------------------------------------------------

@dataclass
class RetrievalResult:
	doc_id: str
	text: str
	score: float


class EmbeddingModel:
	"""Sentence-transformer encoding with L2 normalization."""

	def __init__(self, model):
		self.model = model

	def encode_texts(self, texts: Sequence[str]):
		vectors = self.model.encode(list(texts))
		return self._normalize(vectors)

	def encode_query(self, query: str):
		return self.encode_texts([query])[0]

	@staticmethod
	def _normalize(vectors):
		# Cosine similarity via inner-product in FAISS IndexFlatIP.
		norms = (vectors**2).sum(axis=1, keepdims=True) ** 0.5
		norms[norms == 0.0] = 1.0
		return vectors / norms


class CausalRetriever:
	"""
	Core retrieval flow used in project:
	- Build/load FAISS index.
	- Retrieve top-k by dense similarity.
	- Hash-embedding fallback if model initialization fails.
	"""

	def __init__(self, vector_index, embedding_model=None):
		self.vector_index = vector_index
		self.embedding_model = embedding_model

	def retrieve_documents(self, query: str, top_k: int = 5):
		try:
			q_vec = self.embedding_model.encode_query(query)
		except Exception:
			q_vec = self._hash_query_embedding(query, dim=self.vector_index.dimension)
		return self.vector_index.search(q_vec, top_k)

	@staticmethod
	def _hash_query_embedding(query: str, dim: int):
		vec = [0.0] * dim
		for token in query.lower().split():
			digest = hashlib.md5(token.encode("utf-8")).digest()
			idx = int.from_bytes(digest[:4], byteorder="little") % dim
			sign = 1.0 if digest[4] % 2 == 0 else -1.0
			vec[idx] += sign
		return vec


# -----------------------------------------------------------------------------
# 4.3.3 Causal Extraction Module
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class CausalTriple:
	cause: str
	relation: str
	effect: str
	source_doc_id: str
	confidence: float


CAUSAL_PATTERNS = [
	(re.compile(r"(?P<cause>.+?)\s+(?:causes|caused)\s+(?P<effect>.+)", re.I), "causes", 0.92),
	(re.compile(r"(?P<cause>.+?)\s+(?:leads to|results in)\s+(?P<effect>.+)", re.I), "leads_to", 0.88),
	(re.compile(r"(?P<cause>.+?)\s+(?:increases|raises)\s+(?P<effect>.+)", re.I), "increases", 0.84),
	(re.compile(r"(?P<effect>.+?)\s+(?:is caused by|results from)\s+(?P<cause>.+)", re.I), "causes", 0.90),
]


def extract_local_causal_triples(text: str, source_doc_id: str) -> List[CausalTriple]:
	triples: List[CausalTriple] = []
	for sentence in split_sentences(text):
		s = sentence.strip(" .;")
		for pattern, relation, confidence in CAUSAL_PATTERNS:
			m = pattern.search(s)
			if not m:
				continue
			cause = _clean_span(m.group("cause"))
			effect = _clean_span(m.group("effect"))
			if cause and effect and cause != effect:
				triples.append(CausalTriple(cause, relation, effect, source_doc_id, confidence))
			break
	return triples


def deduplicate_triples(triples: Sequence[CausalTriple]) -> List[CausalTriple]:
	seen = set()
	out = []
	for t in triples:
		key = (t.cause, t.relation, t.effect)
		if key not in seen:
			seen.add(key)
			out.append(t)
	return out


def split_sentences(text: str) -> List[str]:
	return [x.strip() for x in re.split(r"[.!?]+", text) if x.strip()]


def _clean_span(span: str) -> str:
	return re.sub(r"\s+", " ", span).strip(" ,.;:")


# -----------------------------------------------------------------------------
# 4.3.4 Causal Knowledge Graph Construction
# -----------------------------------------------------------------------------

class CausalKnowledgeGraph:
	"""
	Core graph logic:
	- Directed graph nodes are canonicalized events.
	- Each edge stores relation, max confidence, and aggregated evidence list.
	"""

	def __init__(self):
		self.edges = {}  # key: (cause, effect), value: edge attributes

	def add_triple(self, triple: CausalTriple):
		u = canonicalize_event_text(triple.cause)
		v = canonicalize_event_text(triple.effect)
		key = (u, v)
		existing = self.edges.get(key, {"confidence": 0.0, "evidence": [], "relation": triple.relation})
		existing["confidence"] = max(existing["confidence"], triple.confidence)
		existing["relation"] = triple.relation
		existing["evidence"].append(
			{
				"source_doc_id": triple.source_doc_id,
				"confidence": triple.confidence,
			}
		)
		self.edges[key] = existing

	def update_graph(self, triples: Sequence[CausalTriple]):
		for t in triples:
			self.add_triple(t)

	def has_edge(self, cause: str, effect: str) -> bool:
		return (canonicalize_event_text(cause), canonicalize_event_text(effect)) in self.edges


def canonicalize_event_text(text: str) -> str:
	return re.sub(r"\s+", " ", text.lower().strip())


# -----------------------------------------------------------------------------
# 4.3.5 Multi-Hop Retrieval Engine
# -----------------------------------------------------------------------------

@dataclass
class EvidenceChain:
	nodes: List[str] = field(default_factory=list)
	triples: List[CausalTriple] = field(default_factory=list)
	evidence: List[RetrievalResult] = field(default_factory=list)
	validations: Dict[str, object] = field(default_factory=dict)


def run_multi_hop_pipeline(query: str, retriever: CausalRetriever, llm_client, max_hops: int = 6):
	"""
	Core iterative loop used in project:
	1) decompose query + create retrieval queries
	2) initial retrieval -> triple extraction -> graph update
	3) path selection A...D
	4) loop: detect missing link, retrieve next hop, update chain/graph
	5) validate chain (consistency + counterfactual), stop when complete
	6) send final chain to generator
	"""
	decomposition = decompose_query_llm(query, llm_client)
	retrieval_queries = generate_retrieval_queries_llm(query, llm_client)

	graph = CausalKnowledgeGraph()
	chain = initialize_chain(decomposition)

	initial_hits = merge_multi_query_hits(retriever, retrieval_queries, top_k=5)
	triples = deduplicate_triples(extract_triples_from_hits(initial_hits))
	graph.update_graph(triples)
	add_evidence_to_chain(chain, triples, initial_hits)

	source, target = decomposition.get("A"), decomposition.get("D")
	candidate_paths = select_candidate_paths(graph, source, target)
	if candidate_paths:
		chain.nodes = choose_best_path(candidate_paths, decomposition)

	hop = 0
	while hop < max_hops:
		missing = detect_missing_link(graph, chain.nodes, decomposition, query)
		next_hits = retrieve_next_iteration(retriever, query, retrieval_queries, chain, missing, target)
		if not next_hits:
			break

		hop_triples = deduplicate_triples(extract_triples_from_hits(next_hits))
		graph.update_graph(hop_triples)
		add_evidence_to_chain(chain, hop_triples, next_hits)

		chain.validations = {
			"consistency": check_chain_consistency(graph, chain),
			"counterfactual": validate_counterfactual_chain(graph, chain, source or "", target or ""),
		}

		if check_chain_complete(chain, source, target):
			break
		hop += 1

	return {
		"query": query,
		"decomposition": decomposition,
		"retrieval_queries": retrieval_queries,
		"chain": chain,
		"candidate_paths": candidate_paths,
	}


def merge_multi_query_hits(retriever: CausalRetriever, queries: Sequence[str], top_k: int) -> List[RetrievalResult]:
	merged: Dict[str, RetrievalResult] = {}
	for q in queries:
		for hit in retriever.retrieve_documents(q, top_k=top_k):
			if hit.doc_id not in merged or hit.score > merged[hit.doc_id].score:
				merged[hit.doc_id] = hit
	return sorted(merged.values(), key=lambda x: x.score, reverse=True)


def select_candidate_paths(graph: CausalKnowledgeGraph, source: Optional[str], target: Optional[str]) -> List[List[str]]:
	# In codebase this is done using bounded shortest_simple_paths over NetworkX graph.
	if not source or not target:
		return []
	return [[source, "...", target]]


def choose_best_path(candidate_paths: List[List[str]], decomposition: Dict[str, Optional[str]]) -> List[str]:
	anchors = {v.lower() for v in decomposition.values() if v}
	if not candidate_paths:
		return []

	def score(path: List[str]):
		overlap = sum(1 for n in path if n.lower() in anchors)
		return (overlap, -len(path))

	return max(candidate_paths, key=score)


def retrieve_next_iteration(retriever, query, retrieval_queries, chain, missing_link, target_node):
	if missing_link is not None:
		qlist = [
			missing_link["retrieval_query"],
			*[f"{q}. Focus on how {missing_link['left_node']} leads to {missing_link['right_node']}." for q in retrieval_queries],
		]
		return merge_multi_query_hits(retriever, qlist, top_k=4)

	if not chain.nodes:
		return []

	current = chain.nodes[-1]
	qlist = [query, *retrieval_queries, f"Evidence that {current} leads to {target_node}."]
	return merge_multi_query_hits(retriever, qlist, top_k=4)


def detect_missing_link(graph, chain_nodes, decomposition, original_query):
	for left, right in zip(chain_nodes, chain_nodes[1:]):
		if not graph.has_edge(left, right):
			return {
				"left_node": left,
				"right_node": right,
				"retrieval_query": f"{original_query}. Explain how {left} leads to {right}.",
			}
	return None


# -----------------------------------------------------------------------------
# 4.3.6 Evidence Aggregation Module
# -----------------------------------------------------------------------------

def initialize_chain(decomposition: Dict[str, Optional[str]]) -> EvidenceChain:
	ordered = [canonicalize_event_text(v) for v in decomposition.values() if v]
	unique = list(dict.fromkeys([x for x in ordered if x]))
	return EvidenceChain(nodes=unique)


def add_evidence_to_chain(chain: EvidenceChain, triples: Sequence[CausalTriple], hits: Sequence[RetrievalResult]):
	# Merge unique triples.
	existing = {(t.cause, t.relation, t.effect) for t in chain.triples}
	for t in triples:
		if (t.cause, t.relation, t.effect) not in existing:
			chain.triples.append(t)
			existing.add((t.cause, t.relation, t.effect))
		if t.cause not in chain.nodes:
			chain.nodes.append(t.cause)
		if t.effect not in chain.nodes:
			chain.nodes.append(t.effect)

	# Merge unique evidence docs.
	existing_ids = {h.doc_id for h in chain.evidence}
	for h in hits:
		if h.doc_id not in existing_ids:
			chain.evidence.append(h)
			existing_ids.add(h.doc_id)


def build_provenance_payload(answer: str, chain: EvidenceChain, retrieval_queries: Sequence[str]) -> Dict[str, object]:
	sources = []
	for hit in chain.evidence:
		if hit.doc_id not in sources:
			sources.append(hit.doc_id)

	return {
		"answer": answer,
		"retrieval_queries": list(retrieval_queries),
		"evidence_chain": [f"{t.cause} -> {t.effect}" for t in chain.triples],
		"document_sources": sources,
		"causal_path": chain.nodes,
		"validations": chain.validations,
	}


def check_chain_complete(chain: EvidenceChain, source_node: Optional[str], target_node: Optional[str]) -> bool:
	if not source_node or not target_node:
		return False
	s, t = canonicalize_event_text(source_node), canonicalize_event_text(target_node)
	has_endpoints = s in chain.nodes and t in chain.nodes
	has_edges = any((tr.cause == s or tr.effect == t) for tr in chain.triples)
	return has_endpoints and has_edges


def check_chain_consistency(graph: CausalKnowledgeGraph, chain: EvidenceChain) -> Dict[str, object]:
	unsupported = []
	contradictory = []
	negation_markers = {"not", "never", "without", "prevents"}
	for tr in chain.triples:
		if not graph.has_edge(tr.cause, tr.effect):
			unsupported.append(f"{tr.cause} -> {tr.effect}")
		if any(marker in f"{tr.cause} {tr.effect}".lower() for marker in negation_markers):
			contradictory.append(f"{tr.cause} -> {tr.effect}")
	return {
		"is_consistent": (not unsupported and not contradictory),
		"unsupported_edges": unsupported,
		"contradictory_edges": contradictory,
	}


def validate_counterfactual_chain(graph: CausalKnowledgeGraph, chain: EvidenceChain, source_node: str, target_node: str):
	# In project, this uses graph-ablation and path-existence checks.
	critical_nodes = [node for node in chain.nodes if node not in {source_node, target_node}]
	return {
		"is_counterfactually_supported": bool(critical_nodes) or source_node == target_node,
		"critical_nodes": critical_nodes,
	}


def extract_triples_from_hits(hits: Sequence[RetrievalResult]) -> List[CausalTriple]:
	triples: List[CausalTriple] = []
	for hit in hits:
		triples.extend(extract_local_causal_triples(hit.text, source_doc_id=hit.doc_id))
	return triples


# -----------------------------------------------------------------------------
# 4.3.7 Generation Module
# -----------------------------------------------------------------------------

def generate_answer(query: str, chain: EvidenceChain, llm_client) -> str:
	"""
	Core generation strategy:
	- Build prompt from query + chain nodes + extracted graph edges + recent chat.
	- Ask LLM for a grounded natural-language explanation.
	- If LLM unavailable, return deterministic fallback from strongest evidence.
	"""
	chain_text = " -> ".join(chain.nodes)
	edge_text = "\n".join(
		f"- {t.cause} --[{t.relation}]--> {t.effect} (conf={t.confidence:.2f}, src={t.source_doc_id})"
		for t in chain.triples[:12]
	)

	if llm_client.available:
		prompt = (
			f"Query: {query}\n"
			f"Retrieved causal chain: {chain_text or '(none)'}\n"
			f"Retrieved causal graph edges:\n{edge_text or '(no extracted edges)'}\n\n"
			"Write a factual, complete, conversational explanation using only supported evidence."
		)
		try:
			answer = llm_client.generate(prompt=prompt)
			return _enforce_answer_constraints(answer)
		except Exception:
			pass

	return _fallback_answer(query, chain)


def _fallback_answer(query: str, chain: EvidenceChain) -> str:
	if chain.nodes:
		lead = chain.evidence[0].text[:240] if chain.evidence else "available evidence"
		return (
			f"Based on retrieved causal evidence, {' then '.join(chain.nodes[:4])}. "
			f"The strongest supporting evidence indicates: {lead}."
		)
	return "A complete causal chain could not be constructed from available evidence."


def _enforce_answer_constraints(answer: str) -> str:
	paragraph = " ".join((answer or "").split())
	if not paragraph:
		paragraph = "Based on retrieved evidence, the answer involves multiple connected causal steps."
	if paragraph[-1] not in ".!?":
		paragraph += "."
	return paragraph


# # -----------------------------------------------------------------------------
# # Prompts and System Prompts Used in the Project
# # -----------------------------------------------------------------------------
# """
# This section lists the prompt strings and system prompt strings used across
# the codebase (LLM prompts, expansion prompts, and CLI input prompts). For each
# entry we include: the exact prompt text (as used in code), a short usage note,
# and a 100-200 word description explaining intent, design tradeoffs, and why the
# prompt is written as it is.

# Notes:
# - Prompts marked as 'system' are passed to the LLM as the system role.
# - Prompts marked as 'user' are the user-facing or generation prompt content.
# """

# 1) "Decompose the user question into {min_queries} to {max_queries} retrieval subqueries for multi-hop causal reasoning. Each subquery must target an intermediate causal link, mechanism, or bridge fact. Return ONLY the subqueries, one per line, no explanation, no headings.\n\nUser question: {query}"
# 	 - Usage: generate_retrieval_queries_llm -> ask LLM for concise retrieval subqueries.
# 	 - Description (≈120–150 words):
# 		 This prompt instructs the LLM to transform a single broad causal question
# 		 into multiple focused retrieval subqueries that target intermediate causal
# 		 relationships (bridging facts, mechanisms, or specific edges). The
# 		 requirement to return one subquery per line and to avoid explanations or
# 		 headings is deliberate: the downstream code expects a strict one-item-per-line
# 		 format that can be parsed deterministically into retrieval queries. By
# 		 constraining the LLM to target 'intermediate causal link, mechanism, or
# 		 bridge fact' the prompt nudges the model away from superficial paraphrases
# 		 and toward subqueries that facilitate multi-hop retrieval. This prompt is
# 		 intentionally minimal about style but specific about function, prioritizing
# 		 machine-parseable output and retrieval utility over rhetorical polish.

# 2) System: "You produce concise retrieval queries for multi-hop causal evidence gathering."
# 	 - Usage: system role paired with the retrieval-subquery prompt above.
# 	 - Description (≈110–140 words):
# 		 The system prompt frames the LLM's role and output style for the retrieval
# 		 subquery generation phase. By telling the model it 'produces concise
# 		 retrieval queries' we bias it toward terser, query-like outputs rather
# 		 than explanatory prose. The additional context 'for multi-hop causal
# 		 evidence gathering' instructs the model to prioritize causal mechanisms and
# 		 bridging facts rather than generic topical queries. Using a dedicated
# 		 system prompt reduces variability across different LLMs and provides a
# 		 stable intent signal the client stacks onto user prompts; this helps keep
# 		 parsing reliable and helps ensure that generated subqueries are retrieval-optimized.

# 3) "You generated too few retrieval subqueries. Produce exactly {min_queries} distinct subqueries now. Return one subquery per line only.\n\nUser question: {query}\nAlready generated:\n{existing}"
# 	 - Usage: expansion prompt when initial generation yielded too few subqueries.
# 	 - Description (≈110–150 words):
# 		 This corrective prompt is used when the initial LLM output contains fewer
# 		 subqueries than requested. It explicitly asks for exactly N distinct
# 		 subqueries and repeats the constraint (one per line) to ensure parsing
# 		 fidelity. The inclusion of 'Already generated' gives the model context to
# 		 avoid repeating items and encourages diversity. This style of iterative
# 		 correction is pragmatic: rather than re-prompting from scratch, the system
# 		 provides the model with a constrained task and example context to expand
# 		 coverage. The prompt trades verbosity for precision: it repeats constraints
# 		 so the downstream parser can remain very simple and deterministic.

# 4) System: "You generate distinct causal retrieval subqueries."
# 	 - Usage: paired system role for the expansion prompt above.
# 	 - Description (≈100–130 words):
# 		 This short system role statement reinforces the model's objective during
# 		 the expansion pass: produce distinct, causally-relevant retrieval queries.
# 		 It's intentionally terse to avoid introducing stylistic variation while
# 		 emphasizing uniqueness and causal focus. Because the expansion stage is a
# 		 remedial, mechanical step (filling missing items), the system prompt
# 		 focuses purely on outcome properties (distinctness, causal relevance)
# 		 rather than higher-level conversational goals. This keeps the model's
# 		 generation concentrated on generating concise retrieval strings suitable
# 		 for downstream search.

# 5) "Decompose the following question into up to four causal components labelled A, B, C, and D. A should be the initial cause and D should be the final outcome. If an intermediate step is unknown, leave it blank. Return only four lines in the form 'A: ...'.\n\nQuery: {query}"
# 	 - Usage: decompose_query_llm -> request structured causal components (A–D).
# 	 - Description (≈130–160 words):
# 		 This prompt asks the model to perform a structured causal decomposition of
# 		 the input question into a small set of labeled anchors A→D. The labels are
# 		 chosen to match a compact internal representation (A=root cause, D=outcome,
# 		 B/C=intermediates) used across retrieval and path-selection modules. The
# 		 instruction to 'Return only four lines in the form "A: ..."' is crucial:
# 		 it simplifies parsing and error handling by making the output predictable.
# 		 The prompt also allows blank intermediates, which aligns with graceful
# 		 degradation when the model cannot infer certain links. Overall the design
# 		 balances structured output needs with enough flexibility to handle partial
# 		 decompositions.

# 6) System: "You extract concise causal components for multi-hop reasoning."
# 	 - Usage: system prompt paired with the causal decomposition request.
# 	 - Description (≈110–140 words):
# 		 This system role narrows the model's objective to concise extraction rather
# 		 than lengthy explanation. By specifying 'for multi-hop reasoning' the
# 		 statement signals that the output should be oriented toward chain-building
# 		 (succinct causal anchors) and not general summarization. The brevity of
# 		 the system prompt reduces style variability and helps produce compact
# 		 anchor lines (e.g., single phrases rather than sentences), which the
# 		 pipeline expects when constructing retrieval queries and candidate paths.

# 7) Long generation user prompt (final answer):
# 	 "You are generating the final response for a multi-hop causal RAG chatbot. Use the retrieved causal chain, graph edges, and evidence as authoritative context. Answer like a knowledgeable assistant speaking directly to the user in natural conversation. State only well-supported facts from the provided context and avoid mentioning internal pipeline steps.\n\nRecent conversation:\n{history_text}\n\nQuery: {query}\nRetrieved causal chain (linear): {chain_text or '(none)'}\nRetrieved causal chain (steps):\n{chain_steps_text or '(none)'}\n\nRetrieved causal graph edges:\n{graph_edges_text or '(no extracted edges)'}\n\nWrite a comprehensive, well-structured answer that fully explains the causal relationship. Use clear sentences with a logical flow. Do not output causal chain notation like 'A -> B', and do not list raw chain nodes. Do not use headings, bullet points, or section labels. Incorporate key terms from the retrieved chain as proof words in your answer. Provide complete details - do not truncate or abbreviate explanations. If the evidence is incomplete, say so briefly and cautiously."
# 	 - Usage: generate_answer -> produce user-facing final answer grounded in retrieved evidence.
# 	 - Description (≈160–190 words):
# 		 This comprehensive prompt converts internal pipeline outputs (chain nodes,
# 		 extracted graph edges, evidence snippets, and recent chat history) into a
# 		 single, authoritative instruction for the LLM to produce the final human-
# 		 facing answer. The prompt emphasizes epistemic discipline ('State only
# 		 well-supported facts', 'use only supported evidence') to reduce hallucination
# 		 and to tie the answer explicitly to retrieved context. It also constrains
# 		 presentation (no headings, no arrow notation) so that the downstream UI
# 		 and evaluation parsers receive consistent prose. Including recent
# 		 conversation preserves conversational continuity while the explicit
# 		 requirement to avoid internal pipeline details preserves user-oriented
# 		 clarity. The prompt's many constraints are deliberate: the generator must
# 		 balance completeness with fidelity, so the prompt guides the model toward
# 		 careful, evidence-grounded explanations rather than speculative summaries.

# 8) System: "You are a factual multi-hop causal RAG chatbot. Answer conversationally using only supported evidence."
# 	 - Usage: system prompt paired with the final-answer generation prompt.
# 	 - Description (≈120–150 words):
# 		 This system-level statement sets the high-level role and epistemic standard
# 		 for the final-answer LLM call. It frames the model as a factual assistant
# 		 whose outputs must be grounded in provided evidence. The 'conversationally'
# 		 phrase instructs the model about tone, while 'using only supported
# 		 evidence' is an explicit guardrail against unfounded inferences. The system
# 		 prompt is intentionally concise so it can be reliably prepended to the
# 		 user prompt by simple client wrappers. This separation of role (system) and
# 		 task (user prompt) helps the client maintain consistent generation behavior
# 		 across different queries and models.

# 9) "Answer the user question using only the retrieved snippets. If evidence is incomplete, say so briefly.\n\nQuestion: {query}\nRetrieved snippets:\n{context}\n\nAnswer:"
# 	 - Usage: baseline RAG (run_baseline_rag) -> direct generation from retrieved snippets.
# 	 - Description (≈110–140 words):
# 		 The baseline prompt is a straightforward instruction used in experiments
# 		 that evaluate retrieval-augmented generation without multi-hop graph
# 		 machinery. It instructs the model to answer using only the retrieved
# 		 snippets and to acknowledge incomplete evidence, which provides a simple
# 		 fidelity check. The minimal nature of the prompt keeps the baseline
# 		 comparable across configurations and reduces confounding stylistic effects.
# 		 Because this prompt is shorter and less structured than the multi-hop
# 		 final-answer prompt, it can be useful as an ablation to measure the value
# 		 added by the structured chain-and-graph conditioning.

# 10) CLI input prompt (function default): "Enter your query: "
# 		- Usage: receive_query() default prompt when reading from stdin.
# 		- Description (≈100–120 words):
# 			This simple user-prompt string is used when the command-line tool asks
# 			for a query interactively. It is intentionally neutral and unobtrusive
# 			so it can be used in different contexts (interactive shells, scripts).
# 			The prompt is not an LLM instruction; it's a local UI affordance. Keeping
# 			the prompt concise reduces friction for interactive experiments and
# 			automated logging.

# 11) CLI input prompt override in main: "You: "
# 		- Usage: human-facing prompt printed before reading each turn in chat mode.
# 		- Description (≈100–120 words):
# 			This abbreviated prompt improves the terminal chat experience by mimicking
# 			chat UI conventions (user label 'You:'). The short, conversational form
# 			makes terminal sessions more readable and aligns the interactive
# 			interface with the chat-history format that is later fed into the
# 			generator prompt. It's a purely UX-oriented string and has no bearing on
# 			LLM behavior beyond making interactions clearer to human operators.

# -- End of prompts list --


