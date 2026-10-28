import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from reposage.query.evidence import evidence_from_docs, audit_citations


def test_evidence_identity_preserves_distinct_files_and_graph_metadata():
    def doc(path, kind="function"):
        return Document(page_content="same code", metadata={"file": path, "name": "run", "line_start": 3, "line_end": 7, "type": kind})
    evidence = evidence_from_docs([doc("a.py"), doc("b.py"), doc("a.py", "function (Graph Context)")])
    assert len(evidence) == 2
    assert evidence[0]["id"] != evidence[1]["id"]
    assert evidence[0]["file"] == "a.py"
    assert evidence[0]["line_end"] == 7
    assert evidence[0]["symbol"] == "run"


def test_citation_audit_checks_lines_and_unknown_references():
    evidence = [{"citation_id": "S1", "file": "src/a.py", "line_start": 3, "line_end": 7}]
    assert audit_citations("See [S1] and src/a.py:4-6.", evidence)["status"] == "matched"
    assert audit_citations("See [S4], src/a.py:8 and other.py:4.", evidence)["unsupported"] == ["[S4]", "src/a.py:8", "other.py:4"]
    assert audit_citations("See [S01]", evidence)["status"] == "unsupported"
    assert audit_citations("See [Sx]", evidence)["status"] == "unsupported"
    assert audit_citations("[Sources] and [See] are ordinary labels.", evidence)["status"] == "not_detected"
    assert audit_citations("See src/a.py:4.", evidence)["status"] == "not_detected"
    assert audit_citations("No citation", evidence)["status"] == "not_detected"


def test_query_response_preserves_graph_and_agent_traces():
    from main import QueryResponse
    graph = {"lookup_status": "matched", "unresolved": [{"status": "unresolved_target"}]}
    data = QueryResponse(answer="answer", sources=[], chunks_used=0, graph=graph, graph_traces=[graph]).model_dump()
    assert data["graph"] == graph
    assert data["graph_traces"] == [graph]


def test_agent_normal_semantic_tool_returns_retrieved_evidence(monkeypatch):
    import reposage.query.agent_engine as module
    item = evidence_from_docs([Document(page_content="def foo(): pass", metadata={"file": "a.py", "name": "foo", "line_start": 3, "line_end": 5, "type": "function"})])[0]
    class Model:
        def __init__(self, **kwargs): self.calls = 0
        def bind_tools(self, tools): return self
        def invoke(self, messages):
            self.calls += 1
            if self.calls == 1:
                return AIMessage(content="", tool_calls=[{"name": "semantic_search", "args": {"query": "foo"}, "id": "call1"}])
            return AIMessage(content="See [S1].")
    class Rag:
        def __init__(self, **kwargs): self.multilingual_planner = None
        def query(self, *args, **kwargs):
            return {"answer": "found", "sources": ["a.py:3"], "evidence": [item]}
    monkeypatch.setattr(module, "ChatOllama", Model)
    monkeypatch.setattr(module, "CodeRAGEngine", Rag)
    result = module.CodeAgentEngine("sample").query("find foo")
    assert result["tools_executed"] is True
    assert result["evidence"] == [item]
    assert result["sources"] == ["a.py:3"]
    assert result["citation_audit"]["status"] == "matched"


def test_agent_remaps_local_source_ids_across_tool_calls(monkeypatch):
    import reposage.query.agent_engine as module
    first = evidence_from_docs([Document(page_content="a", metadata={"file": "a.py", "name": "a", "line_start": 1, "line_end": 2})])[0]
    second = evidence_from_docs([Document(page_content="b", metadata={"file": "b.py", "name": "b", "line_start": 5, "line_end": 6})])[0]
    class Model:
        def __init__(self, **kwargs): self.calls = 0
        def bind_tools(self, tools): return self
        def invoke(self, messages):
            self.calls += 1
            if self.calls == 1:
                return AIMessage(content="", tool_calls=[
                    {"name": "semantic_search", "args": {"query": "a"}, "id": "one"},
                    {"name": "semantic_search", "args": {"query": "b"}, "id": "two"}])
            assert "[S2]" in str(messages[-1].content)
            return AIMessage(content="Both [S1] and [S2].")
    class Rag:
        def __init__(self, **kwargs): self.multilingual_planner = None
        def query(self, query, **kwargs):
            return {"answer": "Found [S1].", "sources": [], "evidence": [first if query == "a" else second]}
    monkeypatch.setattr(module, "ChatOllama", Model)
    monkeypatch.setattr(module, "CodeRAGEngine", Rag)
    result = module.CodeAgentEngine("sample").query("find both")
    assert [(e["citation_id"], e["file"]) for e in result["evidence"]] == [("S1", "a.py"), ("S2", "b.py")]
    assert result["citation_audit"]["cited"] == ["S1", "S2"]


def test_agent_preserves_original_question_and_remaps_graph_citations(monkeypatch):
    import reposage.query.agent_engine as module
    first = evidence_from_docs([Document(page_content="a", metadata={"file": "a.py", "name": "a", "line_start": 1, "line_end": 2})])[0]
    second = evidence_from_docs([Document(page_content="b", metadata={"file": "b.py", "name": "b", "line_start": 5, "line_end": 6})])[0]
    class Model:
        def __init__(self, **kwargs): self.calls = 0
        def bind_tools(self, tools): return self
        def invoke(self, messages):
            self.calls += 1
            if self.calls == 1:
                return AIMessage(content="", tool_calls=[{"name": "semantic_search", "args": {"query": "a"}, "id": "one"},
                    {"name": "semantic_search", "args": {"query": "b"}, "id": "two"}])
            assert "at b.py:5 [S2]" in messages[-1].content
            return AIMessage(content="See [S2].")
    class Rag:
        def __init__(self, **kwargs):
            self.multilingual_planner = None
            self.source_graph = {}
        def query(self, query, **kwargs):
            assert kwargs["retrieval_original_question"] == "Who calls A.run?"
            item = first if query == "a" else second
            return {"answer": "Found [S1].", "sources": [], "evidence": [item],
                    "graph": {"relationships": [{"caller_chunk_id": item["id"], "caller": "caller", "callee": "target",
                                                  "caller_file": item["file"], "call_line": item["line_start"]}], "unresolved": []}}
    monkeypatch.setattr(module, "ChatOllama", Model)
    monkeypatch.setattr(module, "CodeRAGEngine", Rag)
    result = module.CodeAgentEngine("sample").query("Who calls A.run?")
    assert result["tools_executed"] is True
    assert [g["relationships"][0]["caller_citation_id"] for g in result["graph_traces"]] == ["S1", "S2"]


def test_agent_reports_no_tools_for_evidence_first_graph_answer(monkeypatch):
    import reposage.query.agent_engine as module
    class Model:
        def __init__(self, **kwargs): pass
        def bind_tools(self, tools): return self
        def invoke(self, messages): pytest.fail("Agent model should not run for a grounded relationship answer")
    class Rag:
        def __init__(self, **kwargs): self.source_graph = {"schema": "v2"}
        def query(self, **kwargs):
            assert kwargs["question"] == "What does Value.backward call?"
            return {"answer": "Static reference [S1]", "sources": [], "evidence": [], "answer_mode": "graph_evidence"}
    import pytest
    monkeypatch.setattr(module, "ChatOllama", Model)
    monkeypatch.setattr(module, "CodeRAGEngine", Rag)
    result = module.CodeAgentEngine("sample").query("What does Value.backward call?")
    assert result["execution_mode"] == "graph_grounded_rag"
    assert result["tools_executed"] is False
    assert result["answer_mode"] == "graph_evidence"


def test_agent_no_tool_fallback_preserves_genuine_rag_evidence(monkeypatch):
    import reposage.query.agent_engine as module
    doc = Document(page_content="def foo(): pass", metadata={"repo": "sample", "file": "a.py", "name": "foo", "line_start": 3, "line_end": 5, "type": "function"})
    evidence = evidence_from_docs([doc], "sample")
    class Model:
        def __init__(self, **kwargs): pass
        def bind_tools(self, tools): return self
        def invoke(self, messages): return AIMessage(content="No tool call")
    class Rag:
        def __init__(self, **kwargs): pass
        def query(self, **kwargs):
            return {"answer": "See [S1].", "sources": ["a.py:3"], "chunks_used": 1,
                    "evidence": evidence, "citation_audit": audit_citations("See [S1].", evidence)}
    monkeypatch.setattr(module, "ChatOllama", Model)
    monkeypatch.setattr(module, "CodeRAGEngine", Rag)
    result = module.CodeAgentEngine("sample").query("find foo")
    assert result["execution_mode"] == "rag_fallback_no_tools"
    assert result["tools_executed"] is False
    assert result["evidence"] == evidence
    assert result["citation_audit"]["status"] == "matched"


def test_standard_and_streaming_return_evidence():
    import asyncio
    from reposage.query.rag_engine import CodeRAGEngine
    doc = Document(page_content="def foo(): pass", metadata={"file": "a.py", "name": "foo", "line_start": 3, "line_end": 5, "type": "function"})
    engine = CodeRAGEngine.__new__(CodeRAGEngine)
    engine.repo_name = "sample"
    engine._retrieve_and_fuse = lambda *args, **kwargs: ([doc], ["a.py:3"])
    class LLM:
        def invoke(self, messages): return SimpleNamespace(content="See [S1].", response_metadata={"done_reason": "length"})
        async def astream(self, messages): yield SimpleNamespace(content="See [S1].", response_metadata={"done_reason": "length"})
    engine.llm = LLM()
    answer = engine.query("find foo")
    async def collect(): return [event async for event in engine.aquery_stream("find foo")]
    source_event = next(e for e in asyncio.run(collect()) if e["type"] == "sources")
    assert answer["evidence"] == source_event["evidence"]
    assert answer["citation_audit"] == source_event["citation_audit"] == {"status": "matched", "unsupported": [], "cited": ["S1"], "claim_coverage": "not_assessed", "factual_support": "not_verified"}
    assert answer["answer_truncated"] is True
    assert asyncio.run(collect())[-1]["answer_truncated"] is True


def test_missing_source_metadata_causes_insufficient_evidence_instead_of_generation():
    from reposage.query.rag_engine import CodeRAGEngine
    engine = CodeRAGEngine.__new__(CodeRAGEngine)
    engine.repo_name = "sample"
    engine._retrieve_and_fuse = lambda *args, **kwargs: ([Document(page_content="unattributed code", metadata={})], [])
    engine.llm = SimpleNamespace(invoke=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not generate")))
    result = engine.query("question")
    assert result["evidence"] == []
    assert result["sources"] == []


def test_answer_language_instruction_reuses_existing_detection():
    from reposage.query.rag_engine import _answer_language_instruction
    assert "français" in _answer_language_instruction("Comment fonctionne le code ?")
    assert "中文" in _answer_language_instruction("如何计算梯度？")
    assert "English" in _answer_language_instruction("How does it work?")
