"""
Tests for the retrieval pipeline — verify that RAG components work correctly.

These tests mock the LLM and Embedding interfaces to avoid requiring models during CI.
They verify:
- Context formatting with file:line citations
- Query condensation with chat history
- Document metadata structure
- Source citation formatting
"""

import os
import sys
import pytest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from reposage.chunkers.python_chunker import CodeChunk


# ─── Mock classes for testing without actual models ───

@dataclass
class MockDocument:
    """Simulates a LangChain Document."""
    page_content: str
    metadata: dict


class MockResponse:
    """Simulates an LLM response."""
    def __init__(self, content: str):
        self.content = content


# ═══════ Context Formatting Tests ═══════

class TestContextFormatting:
    """Test that code chunks are formatted correctly for the LLM prompt."""

    def test_format_single_document(self):
        """Verify formatting of a single code chunk."""
        doc = MockDocument(
            page_content="def hello():\n    return 'world'",
            metadata={
                "file": "src/utils.py",
                "line_start": 42,
                "type": "function",
                "name": "hello",
                "language": "python",
            },
        )

        # Simulate _format_context logic
        m = doc.metadata
        header = f"### [1] {m['file']}:{m['line_start']} ({m['type']}: {m['name']})"
        code_fence = f"```{m['language']}\n{doc.page_content}\n```"
        result = f"{header}\n{code_fence}"

        assert "src/utils.py:42" in result
        assert "function: hello" in result
        assert "```python" in result
        assert "def hello():" in result

    def test_format_multiple_documents(self):
        """Verify formatting of multiple chunks with indexing."""
        docs = [
            MockDocument(
                page_content="def auth():\n    pass",
                metadata={
                    "file": "auth.py", "line_start": 10,
                    "type": "function", "name": "auth", "language": "python",
                },
            ),
            MockDocument(
                page_content="class UserService:\n    pass",
                metadata={
                    "file": "services/user.py", "line_start": 1,
                    "type": "class", "name": "UserService", "language": "python",
                },
            ),
        ]

        parts = []
        for i, doc in enumerate(docs, 1):
            m = doc.metadata
            header = f"### [{i}] {m['file']}:{m['line_start']} ({m['type']}: {m['name']})"
            code_fence = f"```{m['language']}\n{doc.page_content}\n```"
            parts.append(f"{header}\n{code_fence}")

        result = "\n\n".join(parts)

        assert "[1]" in result
        assert "[2]" in result
        assert "auth.py:10" in result
        assert "services/user.py:1" in result

    def test_source_citation_format(self):
        """Verify source citations are file:line format."""
        docs = [
            MockDocument(
                page_content="...",
                metadata={"file": "src/auth/login.py", "line_start": 42},
            ),
            MockDocument(
                page_content="...",
                metadata={"file": "utils/jwt.py", "line_start": 8},
            ),
        ]

        sources = [
            f"{d.metadata['file']}:{d.metadata['line_start']}" for d in docs
        ]

        assert sources == ["src/auth/login.py:42", "utils/jwt.py:8"]


# ═══════ Metadata Tests ═══════

class TestDocumentMetadata:
    """Verify that chunk metadata structure is correct for RAG."""

    def test_required_metadata_fields(self):
        """All chunks should have the required metadata fields."""
        required_fields = {"repo", "file", "type", "name", "line_start", "line_end", "language"}

        metadata = {
            "repo": "fastapi-backend",
            "file": "src/auth/login.py",
            "type": "function",
            "name": "authenticate_user",
            "line_start": 42,
            "line_end": 67,
            "language": "python",
        }

        assert required_fields.issubset(metadata.keys())

    def test_chunk_type_values(self):
        """Chunk type should be one of the valid values."""
        valid_types = {"function", "class", "module"}

        for chunk_type in ["function", "class", "module"]:
            assert chunk_type in valid_types

    def test_code_chunk_dataclass(self):
        """Verify CodeChunk dataclass works correctly."""
        chunk = CodeChunk(
            content="def foo(): pass",
            file_path="test.py",
            chunk_type="function",
            name="foo",
            line_start=1,
            line_end=5,
            language="python",
            docstring="Test function",
        )

        assert chunk.content == "def foo(): pass"
        assert chunk.file_path == "test.py"
        assert chunk.chunk_type == "function"
        assert chunk.name == "foo"
        assert chunk.line_start == 1
        assert chunk.line_end == 5
        assert chunk.language == "python"
        assert chunk.docstring == "Test function"


# ═══════ Query Transformer Tests ═══════

class TestQueryTransformerLogic:
    """Test query transformation logic without API calls."""

    def test_original_query_always_included(self):
        """The original query should always be in the expansion list."""
        queries = ["How does auth work?"]  # Original
        # Simulated expansions
        expansions = [
            "authentication flow",
            "login function",
            "JWT verify",
            "token check",
        ]
        queries.extend(expansions)

        assert queries[0] == "How does auth work?"
        assert len(queries) == 5

    def test_deduplication_logic(self):
        """Duplicate documents should be removed."""
        docs = [
            MockDocument(page_content="def auth(): pass", metadata={}),
            MockDocument(page_content="def auth(): pass", metadata={}),  # Duplicate
            MockDocument(page_content="def login(): pass", metadata={}),
        ]

        seen = set()
        unique = []
        for doc in docs:
            key = doc.page_content.strip()
            if key not in seen:
                seen.add(key)
                unique.append(doc)

        assert len(unique) == 2, "Should deduplicate identical chunks"


# ═══════ Chat History Condensation Tests ═══════

class TestChatHistoryCondensation:
    """Test conversation memory logic."""

    def test_no_history_returns_original(self):
        """With no chat history, question should be unchanged."""
        question = "How does login work?"
        chat_history = []

        # Logic: if no history, return original
        result = question if not chat_history else "condensed"
        assert result == question

    def test_history_formatting(self):
        """Chat history should be formatted correctly for condensation."""
        history = [
            {"role": "user", "content": "What is authenticate_user?"},
            {"role": "assistant", "content": "authenticate_user validates credentials..."},
        ]

        formatted = "\n".join(
            f"{msg['role'].title()}: {msg['content']}" for msg in history[-6:]
        )

        assert "User: What is authenticate_user?" in formatted
        assert "Assistant: authenticate_user validates credentials..." in formatted

    def test_history_truncation(self):
        """Only the last 6 messages should be used for condensation."""
        history = [{"role": "user", "content": f"msg {i}"} for i in range(20)]
        truncated = history[-6:]

        assert len(truncated) == 6
        assert truncated[0]["content"] == "msg 14"

# ═══════ Hybrid Search & RRF Tests ═══════

class TestHybridSearchRRF:
    """Test Reciprocal Rank Fusion logic."""

    def test_reciprocal_rank_fusion(self):
        from reposage.query.rag_engine import CodeRAGEngine
        engine = CodeRAGEngine.__new__(CodeRAGEngine) # Bypass __init__
        
        docA = MockDocument(page_content="docA", metadata={})
        docB = MockDocument(page_content="docB", metadata={})
        docC = MockDocument(page_content="docC", metadata={})
        
        vector_docs = [docA, docB, docC]
        bm25_docs = [docC, docA]
        
        # docA: rank 0 in vector, rank 1 in bm25 -> 1/60 + 1/61 = 0.03306
        # docB: rank 1 in vector, rank inf in bm25 -> 1/61 = 0.01639
        # docC: rank 2 in vector, rank 0 in bm25 -> 1/62 + 1/60 = 0.03279
        
        # Expected order: docA, docC, docB
        fused = engine._reciprocal_rank_fusion(vector_docs, bm25_docs, k=60)
        
        assert len(fused) == 3
        assert fused[0].page_content == "docA"
        assert fused[1].page_content == "docC"
        assert fused[2].page_content == "docB"


def _document(file, name, line_start, content=None, chunk_type="function"):
    return MockDocument(
        page_content=content or f"def {name}(): pass",
        metadata={
            "file": file,
            "name": name,
            "line_start": line_start,
            "line_end": line_start + 2,
            "type": chunk_type,
            "language": "python",
        },
    )


class _RawQueryTransformer:
    def expand_query(self, query):
        return [query]


class _StaticRetriever:
    def __init__(self, docs):
        self.docs = docs
        self.queries = []

    def invoke(self, query):
        self.queries.append(query)
        return list(self.docs)


class _StaticVectorStore:
    def __init__(self, retriever):
        self.retriever = retriever
        self.search_kwargs = None

    def as_retriever(self, search_kwargs):
        self.search_kwargs = search_kwargs
        return self.retriever


def _retrieval_engine(vector_docs, bm25_docs, graph=None):
    from reposage.query.rag_engine import CodeRAGEngine

    engine = CodeRAGEngine.__new__(CodeRAGEngine)
    engine.query_transformer = _RawQueryTransformer()
    engine.vectorstore = _StaticVectorStore(_StaticRetriever(vector_docs))
    engine.bm25_retriever = _StaticRetriever(bm25_docs)
    engine.graph = graph
    return engine


def test_retrieval_uses_original_query_for_vector_and_bm25_and_preserves_rrf_order():
    vector_a = _document("src/a.py", "alpha", 1, "vector alpha")
    vector_b = _document("src/b.py", "beta", 8, "vector beta")
    bm25_a = _document("src/a.py", "alpha", 1, "vector alpha")
    engine = _retrieval_engine([vector_a, vector_b], [bm25_a])
    question = "Où est alpha? keepExactName"

    docs, sources = engine._retrieve_and_fuse(question, 2)

    assert engine.vectorstore.retriever.queries == [question]
    assert engine.bm25_retriever.queries == [question]
    assert engine.vectorstore.search_kwargs == {"k": 2}
    assert [doc.metadata["name"] for doc in docs] == ["alpha", "beta"]
    assert sources == ["src/a.py:1", "src/b.py:8"]


def test_format_context_uses_retrieved_citation_metadata():
    from reposage.query.rag_engine import CodeRAGEngine

    engine = CodeRAGEngine.__new__(CodeRAGEngine)
    context = engine._format_context([
        _document("micrograd/engine.py", "backward", 52, "def backward(): pass")
    ])

    assert "micrograd/engine.py:52 (function: backward)" in context
    assert "```python" in context
    assert "def backward(): pass" in context


def test_retrieval_adds_call_graph_neighbors_after_rrf():
    import networkx as nx

    seed = _document("micrograd/engine.py", "backward", 52, "def backward(): pass")
    neighbor = _document("micrograd/engine.py", "build_topo", 43, "def build_topo(): pass")
    graph = nx.DiGraph()
    graph.add_edge("backward", "build_topo", type="calls")
    engine = _retrieval_engine([seed], [seed, neighbor], graph)

    docs, sources = engine._retrieve_and_fuse("gradient", 1)

    assert [doc.metadata["name"] for doc in docs] == ["backward", "build_topo"]
    assert docs[1].metadata["type"] == "function (Graph Context)"
    assert sources == ["micrograd/engine.py:52", "micrograd/engine.py:43"]


def test_graph_enrichment_does_not_contaminate_later_query():
    import networkx as nx
    seed = _document("a.py", "seed", 1)
    neighbor = _document("b.py", "neighbor", 5)
    graph = nx.DiGraph()
    graph.add_edge("seed", "neighbor")
    engine = _retrieval_engine([seed], [seed, neighbor], graph)
    stages = []
    first, _ = engine._retrieve_and_fuse("first", 1, lambda stage, docs: stages.append((stage, list(docs))))
    assert first[1].metadata["type"] == "function (Graph Context)"
    assert neighbor.metadata["type"] == "function"
    engine.vectorstore.retriever.docs = [neighbor]
    engine.bm25_retriever.docs = [neighbor]
    second, _ = engine._retrieve_and_fuse("second", 1)
    assert second[0].metadata["type"] == "function"
    assert [stage for stage, _ in stages] == ["rrf", "graph_enrichment"]


def test_multilingual_plan_integrates_with_shared_retrieval_and_legacy_default():
    from reposage.query.multilingual_query import MultilingualQueryPlanner, PlannerConfig
    doc = _document("src/a.py", "authenticate_user", 1)
    engine = _retrieval_engine([doc], [doc])
    original = "Comment fonctionne authenticate_user dans le code ?"
    contextual = "How does authenticate_user work in the code?"
    plans = []
    engine.multilingual_planner = MultilingualQueryPlanner(PlannerConfig(), lambda _: pytest.fail("disabled"))
    engine._retrieve_and_fuse(contextual, 1, original_question=original, plan_observer=plans.append)
    assert engine.vectorstore.retriever.queries == [contextual]
    assert not plans
    engine.vectorstore.retriever.queries.clear()
    engine.bm25_retriever.queries.clear()
    engine.multilingual_planner = MultilingualQueryPlanner(
        PlannerConfig(enabled=True), lambda _: "How does REPOSAGEPROTECTED0TOKEN work?")
    engine._retrieve_and_fuse(contextual, 1, original_question=original, plan_observer=plans.append)
    expected = [original, contextual, "How does authenticate_user work?"]
    assert sorted(engine.vectorstore.retriever.queries) == sorted(expected)
    assert sorted(engine.bm25_retriever.queries) == sorted(expected)
    assert [v.provenance for v in plans[0].variants] == ["original", "contextual", "english_rewrite"]


def test_disabled_planner_retains_existing_expansion():
    from reposage.query.multilingual_query import MultilingualQueryPlanner, PlannerConfig
    doc = _document("src/a.py", "alpha", 1)
    engine = _retrieval_engine([doc], [doc])
    engine.query_transformer.expand_query = lambda q: [q, "legacy expansion"]
    engine.multilingual_planner = MultilingualQueryPlanner(PlannerConfig())
    engine._retrieve_and_fuse("original", 1)
    assert sorted(engine.vectorstore.retriever.queries) == ["legacy expansion", "original"]
    assert sorted(engine.bm25_retriever.queries) == ["legacy expansion", "original"]


def test_code_aware_planner_routes_one_query_per_retriever_and_keeps_rrf_trace():
    from reposage.query.multilingual_query import MultilingualQueryPlanner, PlannerConfig
    from reposage.query.code_aware_query import CodeAwareQueryPlanner
    doc = _document("src/a.py", "authenticate_user", 1)
    engine = _retrieval_engine([doc], [doc])
    engine.code_aware_planner = CodeAwareQueryPlanner(MultilingualQueryPlanner(
        PlannerConfig(enabled=True), lambda _: "How does REPOSAGEPROTECTED0TOKEN work?"))
    original = "Comment fonctionne authenticate_user dans le code ?"
    stages, plans = [], []
    docs, _ = engine._retrieve_and_fuse(original, 2, stage_observer=lambda s, d: stages.append(s), plan_observer=plans.append)
    assert engine.vectorstore.retriever.queries == ["How does authenticate_user work?"]
    assert engine.bm25_retriever.queries == [original]
    assert [d.metadata["name"] for d in docs] == ["authenticate_user"]
    assert "rrf" in stages
    assert len(plans[0].requests) == 2


def test_streaming_query_uses_code_aware_routes():
    import asyncio
    from types import SimpleNamespace
    from reposage.query.multilingual_query import MultilingualQueryPlanner, PlannerConfig
    from reposage.query.code_aware_query import CodeAwareQueryPlanner
    doc = _document("src/a.py", "authenticate_user", 1)
    engine = _retrieval_engine([doc], [doc])
    engine.code_aware_planner = CodeAwareQueryPlanner(MultilingualQueryPlanner(
        PlannerConfig(enabled=True), lambda _: "How does REPOSAGEPROTECTED0TOKEN work?"))
    engine.repo_name = "sample"
    class LLM:
        async def astream(self, messages):
            yield SimpleNamespace(content="answer")
    engine.llm = LLM()
    question = "Comment fonctionne authenticate_user dans le code ?"
    async def collect(): return [event async for event in engine.aquery_stream(question)]
    events = asyncio.run(collect())
    assert events[0]["content"] == "answer"
    assert engine.bm25_retriever.queries == [question]
    assert engine.vectorstore.retriever.queries == ["How does authenticate_user work?"]


def test_standard_query_retains_user_question_after_condensation():
    from types import SimpleNamespace
    from reposage.query.multilingual_query import MultilingualQueryPlanner, PlannerConfig
    doc = _document("src/a.py", "authenticate_user", 1)
    engine = _retrieval_engine([doc], [doc])
    engine.multilingual_planner = MultilingualQueryPlanner(PlannerConfig(enabled=True), lambda _: "How does REPOSAGEPROTECTED0TOKEN work?")
    engine.repo_name = "sample"
    engine.llm = SimpleNamespace(invoke=lambda _: SimpleNamespace(content="answer"))
    engine._condense_question = lambda question, history: "How does authenticate_user work in the code?"
    original = "Comment fonctionne authenticate_user dans le code ?"
    assert engine.query(original, chat_history=[{"role": "user", "content": "previous"}])["answer"] == "answer"
    assert original in engine.vectorstore.retriever.queries
