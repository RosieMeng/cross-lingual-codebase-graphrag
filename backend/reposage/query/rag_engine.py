"""
Code-Aware RAG Query Engine — The brain of RepoSage.

Handles the full query pipeline:
1. Receive natural language question
2. Retrieve relevant code chunks from ChromaDB (vector similarity)
3. Format chunks with file:line attribution
4. Construct code-aware prompts (system + user)
5. Stream response from local Ollama model
6. Return answer with source citations

The system prompt instructs the LLM to always cite file paths and line
numbers, explain code before technical details, and connect multi-file
answers.
"""

import os
import logging
import copy
from dataclasses import replace
from typing import Optional, AsyncGenerator

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.messages import SystemMessage, HumanMessage
import pickle
from .query_transformer import QueryTransformer
from .multilingual_query import MultilingualQueryPlanner, PlannerConfig, ollama_rewriter, detect_language
from .code_aware_query import CodeAwareQueryPlanner
from .evidence import evidence_from_docs, audit_citations, chunk_identity
from .source_graph import enrich as enrich_source_graph, load_graph as load_source_graph, graph_context, requested_direction
from .graph_answer import graph_answer

logger = logging.getLogger(__name__)

# ─── Prompts ───

SYSTEM_PROMPT = """You are RepoSage, an expert code assistant with deep knowledge \
of the {repo_name} codebase. You answer questions about code architecture, \
implementation details, and specific functions.

Rules:
1. Cite each factual claim using only the supplied source IDs, such as [S1].
2. Never invent a source ID. A source ID identifies code, not proof that a claim is true.
3. Explain only what the shown code supports; do not infer missing behavior.
4. If evidence is insufficient, explicitly say so instead of guessing.
5. Answer in the language of the user's question when possible.
6. Be concise, preserve exact code identifiers, and explain implementation details accurately.
7. Describe arithmetic and gradient assignments exactly as written. Do not invent equations, examples, or variable relationships. If the relevant operation is not shown, say it is unknown.

You're talking to a developer who wants to understand this codebase quickly."""

QUERY_PROMPT = """Here are the relevant code sections from the {repo_name} repository:

{context}

---

Developer's question: {question}

Provide a clear answer with source IDs such as [S1]. If the code does not support an answer, say so."""

CONDENSE_PROMPT = """Given the following conversation and a follow-up question, \
rephrase the follow-up question to be a standalone question that includes \
relevant context from the conversation history.

Chat History:
{chat_history}

Follow Up Question: {question}

Standalone question:"""


def _answer_language_instruction(question: str) -> str:
    language = detect_language(question).language
    return {"en": "Write the entire answer in English.",
            "fr": "Rédige toute la réponse en français, sauf les identifiants du code.",
            "zh": "请用中文回答，代码标识符保持原样。"}.get(language, "Use the user's language where clear; preserve code identifiers.")


class CodeRAGEngine:
    """RAG query engine for code-aware question answering.

    Retrieves relevant code chunks from ChromaDB and generates
    answers with file:line citations using a local Ollama model.

    Args:
        repo_name: Name of the indexed repository.
        persist_dir: ChromaDB persistence directory.
        ollama_base_url: Base URL for local Ollama service.
        ollama_model: Local Ollama model name.
        ollama_embed_model: Local Ollama embedding model name.
    """

    def __init__(
        self,
        repo_name: str,
        persist_dir: str = "./chroma_db",
        ollama_base_url: str = "http://localhost:11434",
        ollama_model: str = "llama3",
        ollama_embed_model: str = "nomic-embed-text",
        multilingual_config: PlannerConfig | None = None,
        code_aware_enabled: bool = False,
    ):
        embed = OllamaEmbeddings(
            base_url=ollama_base_url,
            model=ollama_embed_model,
        )

        collection_dir = os.path.join(persist_dir, repo_name)
        self.vectorstore = Chroma(
            persist_directory=collection_dir,
            embedding_function=embed,
            collection_name=repo_name,
        )
        self.repo_name = repo_name
        self._disable_model_thinking = ollama_model.lower().startswith("qwen3")
        self.llm = ChatOllama(
            base_url=ollama_base_url,
            model=ollama_model,
            temperature=0,
            num_predict=int(os.getenv("OLLAMA_MAX_ANSWER_TOKENS", "512")),
        )
        self.condense_llm = ChatOllama(
            base_url=ollama_base_url,
            model=ollama_model,
            temperature=0,
        )
        self.query_transformer = QueryTransformer(
            ollama_base_url=ollama_base_url,
            model=ollama_model,
        )
        config = multilingual_config or PlannerConfig()
        rewriter = None
        if config.enabled or code_aware_enabled:
            try:
                rewriter = ollama_rewriter(config, ollama_base_url, ollama_model)
            except Exception as e:
                logger.warning("Multilingual rewriter unavailable: %s", type(e).__name__)
        self.multilingual_planner = MultilingualQueryPlanner(config, rewriter)
        self.code_aware_planner = (CodeAwareQueryPlanner(MultilingualQueryPlanner(replace(config, enabled=True), rewriter))
                                   if code_aware_enabled else None)
        
        # Load BM25 index if available
        self.bm25_retriever = None
        bm25_path = os.path.join(collection_dir, "bm25_docs.pkl")
        if os.path.exists(bm25_path):
            try:
                with open(bm25_path, "rb") as f:
                    docs = pickle.load(f)
                self.bm25_retriever = BM25Retriever.from_documents(docs)
                logger.info(f"Loaded BM25 index with {len(docs)} documents.")
            except Exception as e:
                logger.error("Failed to load BM25 index: %s", type(e).__name__)

        # Load Knowledge Graph
        self.graph = None
        graph_path = os.path.join(collection_dir, "graph.pkl")
        if os.path.exists(graph_path):
            try:
                import networkx as nx
                with open(graph_path, "rb") as f:
                    self.graph = pickle.load(f)
                logger.info(f"Loaded Knowledge Graph with {self.graph.number_of_nodes()} nodes.")
            except ImportError:
                logger.warning("networkx not installed, skipping GraphRAG.")
            except Exception as e:
                logger.error("Failed to load Knowledge Graph: %s", type(e).__name__)

        # Existing graph.pkl remains compatible. A separately generated v2
        # sidecar takes precedence when present, without migrating the index.
        self.source_graph = None
        graph_dir = os.getenv("REPOSAGE_SOURCE_GRAPH_DIR")
        source_graph_path = (os.path.join(graph_dir, f"{repo_name}.json") if graph_dir
                             else os.path.join(collection_dir, "source_graph_v2.json"))
        self.disable_legacy_graph = bool(graph_dir) or os.path.isfile(source_graph_path)
        if os.path.isfile(source_graph_path):
            try:
                self.source_graph = load_source_graph(source_graph_path, repo_name)
                logger.info("Loaded qualified source graph")
            except (OSError, ValueError, TypeError, KeyError):
                logger.warning("Qualified source graph unavailable or invalid")
        elif graph_dir:
            logger.warning("Configured qualified source graph not found")


    def _format_context(self, docs) -> str:
        """Format code chunks with clear file:line attribution.

        Each chunk is displayed with:
        - Header: file_path:line_start (type: name)
        - Language-specific code fence
        - Full chunk content

        This formatting helps the LLM understand where code comes from
        and produce accurate citations in its response.
        """
        parts = []
        repo_name = getattr(self, "repo_name", None)
        evidence = {item["id"]: item for item in evidence_from_docs(docs, repo_name)}
        emitted = set()
        for doc in docs:
            identity = chunk_identity(doc, repo_name)
            if identity not in evidence or identity in emitted:
                continue
            emitted.add(identity)
            m = doc.metadata
            file_path = m.get("file", "unknown")
            line_start = m.get("line_start", "?")
            chunk_type = m.get("type", "unknown")
            name = m.get("name", "unknown")
            language = m.get("language", "")

            label = evidence[identity]["citation_id"]
            header = f"### [{label}] {file_path}:{line_start} ({chunk_type}: {name})"
            line_end = evidence[identity]["line_end"]
            if line_end is not None:
                header += f" [lines {line_start}-{line_end}]"
            code_fence = f"```{language}\n{doc.page_content}\n```"
            parts.append(f"{header}\n{code_fence}")

        return "\n\n".join(parts)

    def _condense_question(self, question: str, chat_history: list[dict]) -> str:
        """Condense a follow-up question using conversation history.

        Handles cases like "Tell me more about that function" by
        incorporating context from previous turns.
        """
        if not chat_history:
            return question

        history_str = "\n".join(
            f"{msg['role'].title()}: {msg['content']}" for msg in chat_history[-6:]
        )

        prompt = CONDENSE_PROMPT.format(
            chat_history=history_str,
            question=question,
        )

        response = self.condense_llm.invoke(prompt)
        condensed = response.content.strip()
        logger.info("Query condensation completed")
        return condensed

    def _reciprocal_rank_fusion(self, vector_docs: list, bm25_docs: list, k: int = 60) -> list:
        """Merge documents using Reciprocal Rank Fusion (RRF)."""
        fused_scores = {}
        doc_map = {}
        
        for rank, doc in enumerate(vector_docs):
            content_key = chunk_identity(doc, getattr(self, "repo_name", None))
            doc_map[content_key] = doc
            fused_scores[content_key] = fused_scores.get(content_key, 0) + 1 / (rank + k)
            
        for rank, doc in enumerate(bm25_docs):
            content_key = chunk_identity(doc, getattr(self, "repo_name", None))
            doc_map[content_key] = doc
            fused_scores[content_key] = fused_scores.get(content_key, 0) + 1 / (rank + k)
            
        # Sort by fused score
        sorted_docs = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_map[key] for key, _ in sorted_docs]

    def _retrieve_and_fuse(self, question: str, k: int, stage_observer=None,
                           original_question: str | None = None, plan_observer=None,
                           retrieval_plan=None, graph_observer=None) -> tuple[list, list[str]]:
        """Retrieve relevant code chunks and enrich context in parallel using a ThreadPoolExecutor."""
        # Expand queries once
        planner = getattr(self, "multilingual_planner", None)
        code_planner = getattr(self, "code_aware_planner", None)
        if retrieval_plan is not None or code_planner is not None:
            original = question if original_question is None else original_question
            plan = retrieval_plan if retrieval_plan is not None else code_planner.plan(original, k, contextual_question=question)
            if not plan.requests or any(request.candidate_limit != k or request.variant_index < 0
                    or request.variant_index >= len(plan.variants) or request.retriever not in {"vector", "bm25"}
                    for request in plan.requests):
                raise ValueError("Invalid retrieval plan or candidate budget")
            vector_queries = [plan.variants[r.variant_index].text for r in plan.requests if r.retriever == "vector"]
            bm25_queries = [plan.variants[r.variant_index].text for r in plan.requests if r.retriever == "bm25"]
            if plan_observer:
                plan_observer(plan)
            logger.info("Code-aware query plan: language=%s status=%s requests=%d", plan.detection.language, plan.status, len(plan.requests))
        elif planner is not None and planner.config.enabled:
            original = question if original_question is None else original_question
            plan = planner.plan(original, contextual_question=question)
            if plan_observer:
                plan_observer(plan)
            queries = [variant.text for variant in plan.variants]
            vector_queries = bm25_queries = queries
            logger.info("Multilingual query plan: language=%s status=%s variants=%d",
                        plan.detection.language, plan.status, len(queries))
        else:
            queries = self.query_transformer.expand_query(question)
            vector_queries = bm25_queries = queries
        
        vector_retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
        
        # Parallel retrieval
        from concurrent.futures import ThreadPoolExecutor
        
        def retrieve_vector(q):
            try:
                return vector_retriever.invoke(q)
            except Exception as e:
                logger.warning("Vector retrieval failed: %s", type(e).__name__)
                return []
                
        def retrieve_bm25(q):
            try:
                self.bm25_retriever.k = k
                return self.bm25_retriever.invoke(q)
            except Exception as e:
                logger.warning("BM25 retrieval failed: %s", type(e).__name__)
                return []

        with ThreadPoolExecutor(max_workers=10) as executor:
            # Concurrently fetch vector and BM25 matching documents across all queries
            vector_futures = [executor.submit(retrieve_vector, q) for q in vector_queries]
            bm25_futures = [executor.submit(retrieve_bm25, q) for q in bm25_queries] if self.bm25_retriever else []
            
            vector_results = [f.result() for f in vector_futures]
            bm25_results = [f.result() for f in bm25_futures]

        # Flatten and deduplicate lists
        vector_docs = []
        seen_vector = set()
        for docs_list in vector_results:
            for doc in docs_list:
                key = chunk_identity(doc, getattr(self, "repo_name", None))
                if key not in seen_vector:
                    seen_vector.add(key)
                    vector_docs.append(doc)
                    
        bm25_docs = []
        seen_bm25 = set()
        for docs_list in bm25_results:
            for doc in docs_list:
                key = chunk_identity(doc, getattr(self, "repo_name", None))
                if key not in seen_bm25:
                    seen_bm25.add(key)
                    bm25_docs.append(doc)

        # Reciprocal Rank Fusion
        docs = self._reciprocal_rank_fusion(vector_docs, bm25_docs)[:k]
        if stage_observer:
            stage_observer("rrf", docs)

        # GraphRAG Context Enrichment
        source_graph = getattr(self, "source_graph", None)
        if source_graph and self.bm25_retriever:
            try:
                graph_info = {}
                graph_context_docs = enrich_source_graph(source_graph, docs, self.bm25_retriever.docs,
                                                         getattr(self, "repo_name", None),
                                                         question if original_question is None else original_question,
                                                         diagnostics=graph_info)
                if graph_observer:
                    graph_observer(graph_info)
                if stage_observer:
                    stage_observer("graph_enrichment", graph_context_docs)
                docs.extend(graph_context_docs)
            except Exception as e:
                logger.warning("Qualified graph traversal failed: %s", type(e).__name__)
                if stage_observer:
                    stage_observer("error", {"stage": "graph_enrichment", "error_type": type(e).__name__})
        elif self.graph and self.bm25_retriever and not getattr(self, "disable_legacy_graph", False):
            doc_lookup = {d.metadata.get("name"): d for d in self.bm25_retriever.docs if d.metadata.get("name")}
            graph_context_docs = []
            graph_originals = []
            
            for doc in docs:
                name = doc.metadata.get("name")
                if name and self.graph.has_node(name):
                    try:
                        neighbors = list(self.graph.predecessors(name)) + list(self.graph.successors(name))
                        for n in neighbors[:3]:  # Top 3 neighbors per chunk
                            if n in doc_lookup and doc_lookup[n] not in docs and doc_lookup[n] not in graph_originals:
                                graph_originals.append(doc_lookup[n])
                                extra_doc = copy.copy(doc_lookup[n])
                                extra_doc.metadata = dict(extra_doc.metadata)
                                extra_doc.metadata["type"] = extra_doc.metadata.get("type", "") + " (Graph Context)"
                                graph_context_docs.append(extra_doc)
                    except Exception as e:
                        logger.warning("Graph traversal failed: %s", type(e).__name__)
                        if stage_observer:
                            stage_observer("error", {"stage": "graph_enrichment", "error_type": type(e).__name__})
                        
            if stage_observer:
                stage_observer("graph_enrichment", graph_context_docs)
            docs.extend(graph_context_docs)

        sources = [
            f"{d.metadata.get('file', '?')}:{d.metadata.get('line_start', '?')}"
            for d in docs
        ]
        return docs, sources

    def query(
        self,
        question: str,
        k: int = 6,
        chat_history: Optional[list[dict]] = None,
        retrieval_original_question: Optional[str] = None,
    ) -> dict:
        """Answer a question about the codebase using RAG.

        Args:
            question: Natural language question about the code.
            k: Number of chunks to retrieve (default: 6).
            chat_history: Optional list of previous messages for context.

        Returns:
            Dict with 'answer', 'sources', and 'chunks_used'.
        """
        # Condense follow-up questions using chat history
        standalone_question = self._condense_question(
            question, chat_history or []
        )

        # Retrieve documents concurrently
        graph_info = {}
        docs, sources = self._retrieve_and_fuse(standalone_question, k, original_question=question if retrieval_original_question is None else retrieval_original_question, graph_observer=graph_info.update)
        if (getattr(self, "source_graph", None) or getattr(self, "disable_legacy_graph", False)) and not graph_info:
            graph_info.update({"lookup_status": "unavailable", "direction": requested_direction(question if retrieval_original_question is None else retrieval_original_question)})

        evidence = evidence_from_docs(docs, self.repo_name)
        if not evidence:
            grounded = graph_answer(question if retrieval_original_question is None else retrieval_original_question, graph_info, evidence)
            return {
                "answer": grounded or "I couldn't find any relevant code in the indexed repository. "
                          "Try rephrasing your question or make sure the repository "
                          "has been indexed correctly.",
                "sources": [],
                "chunks_used": 0,
                "evidence": [],
                "citation_audit": audit_citations("", []),
                "graph": graph_info,
                "answer_mode": "graph_evidence" if grounded is not None else "model",
            }

        # Format context and build prompt
        context = self._format_context(docs) + graph_context(graph_info, evidence)
        grounded = graph_answer(question if retrieval_original_question is None else retrieval_original_question, graph_info, evidence)
        if grounded is not None:
            return {"answer": grounded, "sources": [f"{item['file']}:{item['line_start']}" for item in evidence],
                    "chunks_used": len(docs), "evidence": evidence, "citation_audit": audit_citations(grounded, evidence),
                    "graph": graph_info, "answer_truncated": False, "answer_mode": "graph_evidence"}

        messages = [
            SystemMessage(content=SYSTEM_PROMPT.format(repo_name=self.repo_name) + "\n" +
                          _answer_language_instruction(question if retrieval_original_question is None else retrieval_original_question)),
            HumanMessage(content=QUERY_PROMPT.format(
                repo_name=self.repo_name,
                context=context,
                question=standalone_question,
            )),
        ]

        response = self.llm.invoke(messages, **({"think": False} if getattr(self, "_disable_model_thinking", False) else {}))
        answer = response.content

        return {
            "answer": answer,
            "sources": [f"{item['file']}:{item['line_start']}" for item in evidence],
            "chunks_used": len(docs),
            "evidence": evidence,
            "citation_audit": audit_citations(answer, evidence),
            "graph": graph_info,
            "answer_truncated": getattr(response, "response_metadata", {}).get("done_reason") == "length",
        }

    async def aquery_stream(
        self,
        question: str,
        k: int = 6,
        chat_history: Optional[list[dict]] = None,
    ) -> AsyncGenerator[dict, None]:
        """Stream an answer token by token (for SSE endpoints).

        Yields dicts with either:
        - {"type": "token", "content": "..."} for each token
        - {"type": "sources", "sources": [...]} at the end
        - {"type": "done"} when streaming is complete

        Args:
            question: Natural language question.
            k: Number of chunks to retrieve.
            chat_history: Optional conversation history.
        """
        # Condense follow-up questions using chat history
        standalone_question = self._condense_question(
            question, chat_history or []
        )

        # Retrieve documents concurrently
        graph_info = {}
        docs, sources = self._retrieve_and_fuse(standalone_question, k, original_question=question, graph_observer=graph_info.update)
        if (getattr(self, "source_graph", None) or getattr(self, "disable_legacy_graph", False)) and not graph_info:
            graph_info.update({"lookup_status": "unavailable", "direction": requested_direction(question)})

        evidence = evidence_from_docs(docs, self.repo_name)
        if not evidence:
            grounded = graph_answer(question, graph_info, evidence)
            yield {
                "type": "token",
                "content": grounded or "I couldn't find any relevant code in the indexed repository.",
            }
            yield {"type": "sources", "sources": [], "evidence": [], "citation_audit": audit_citations("", []), "graph": graph_info}
            yield {"type": "done", "answer_mode": "graph_evidence" if grounded is not None else "model"}
            return

        context = self._format_context(docs) + graph_context(graph_info, evidence)
        grounded = graph_answer(question, graph_info, evidence)
        if grounded is not None:
            yield {"type": "token", "content": grounded}
            yield {"type": "sources", "sources": [f"{item['file']}:{item['line_start']}" for item in evidence],
                   "evidence": evidence, "citation_audit": audit_citations(grounded, evidence), "graph": graph_info}
            yield {"type": "done", "answer_truncated": False, "answer_mode": "graph_evidence"}
            return

        messages = [
            SystemMessage(content=SYSTEM_PROMPT.format(repo_name=self.repo_name) + "\n" + _answer_language_instruction(question)),
            HumanMessage(content=QUERY_PROMPT.format(
                repo_name=self.repo_name,
                context=context,
                question=standalone_question,
            )),
        ]

        # Stream tokens
        answer_parts = []
        truncated = False
        async for chunk in self.llm.astream(messages, **({"think": False} if getattr(self, "_disable_model_thinking", False) else {})):
            if getattr(chunk, "response_metadata", {}).get("done_reason") == "length":
                truncated = True
            if chunk.content:
                answer_parts.append(chunk.content)
                yield {"type": "token", "content": chunk.content}

        # Send sources at the end
        yield {"type": "sources", "sources": [f"{item['file']}:{item['line_start']}" for item in evidence], "evidence": evidence,
               "citation_audit": audit_citations("".join(answer_parts), evidence), "graph": graph_info}
        yield {"type": "done", "answer_truncated": truncated}

    def get_collection_stats(self) -> dict:
        """Get statistics about the indexed collection."""
        try:
            count = self.vectorstore._collection.count()
            return {
                "repo": self.repo_name,
                "chunks": count,
                "status": "ready",
            }
        except Exception as e:
            return {
                "repo": self.repo_name,
                "chunks": 0,
                "status": f"error: {e}",
            }
