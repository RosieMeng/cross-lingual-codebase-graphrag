"""Production qualified graph and graph-to-context regressions."""
from pathlib import Path
import sys

from langchain_core.documents import Document

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reposage.chunkers.python_chunker import PythonASTChunker
from reposage.ingestion.source_graph import build_source_graph, write_graph
from reposage.query.evidence import chunk_identity, evidence_from_docs
from reposage.query.rag_engine import CodeRAGEngine
from reposage.query.source_graph import enrich, load_graph, requested_direction


def fixture(tmp_path):
    sources = {
        "a.py": "def helper():\n    value = 1\n    value += 1\n    value += 1\n    return value\n\ndef caller():\n    value = helper()\n    value += 1\n    value += 1\n    return value\n\nclass A:\n    def short(self):\n        return 1\n\n    def run(self):\n        value = self.short()\n        value += 1\n        value += 1\n        return value\n",
        "b.py": "from a import helper\n\ndef cross():\n    value = helper()\n    value += 1\n    value += 1\n    return value\n\ndef caller():\n    value = 3\n    value += 1\n    value += 1\n    return value\n\ndef unknown(obj):\n    value = obj.missing()\n    value += 1\n    value += 1\n    return value\n",
    }
    docs = []
    for name, source in sources.items():
        path = tmp_path / name
        path.write_text(source, encoding="utf-8")
        for chunk in PythonASTChunker().chunk_file(str(path)):
            docs.append(Document(page_content=chunk.content, metadata={"repo": "fixture", "file": name, "name": chunk.name,
                "type": chunk.chunk_type, "line_start": chunk.line_start, "line_end": chunk.line_end, "language": "python"}))
    return docs, build_source_graph(tmp_path, docs, "fixture")


def test_graph_qualified_symbols_calls_and_short_function_mapping(tmp_path):
    docs, graph = fixture(tmp_path)
    nodes = {(n["file"], n["qualified"]): n for n in graph["nodes"]}
    assert nodes[("a.py", "caller")]["id"] != nodes[("b.py", "caller")]["id"]
    assert nodes[("a.py", "A.short")]["chunk_mapping"] == "enclosing"
    assert nodes[("a.py", "A.short")]["chunk_id"] == nodes[("a.py", "A")]["chunk_id"]
    pairs = {(e["source"], e["target"]) for e in graph["edges"]}
    assert (nodes[("a.py", "caller")]["id"], nodes[("a.py", "helper")]["id"]) in pairs
    assert (nodes[("b.py", "cross")]["id"], nodes[("a.py", "helper")]["id"]) in pairs
    assert (nodes[("a.py", "A.run")]["id"], nodes[("a.py", "A.short")]["id"]) in pairs
    assert (nodes[("a.py", "A")]["id"], nodes[("a.py", "A.short")]["id"]) not in pairs
    assert any(c["expression"] == "obj.missing" for c in graph["unresolved"])
    assert all(n["chunk_id"] in {chunk_identity(d, "fixture") for d in docs} for n in graph["nodes"] if n["chunk_id"])


def test_direction_dedup_budget_and_source_context(tmp_path):
    docs, graph = fixture(tmp_path)
    helper = next(d for d in docs if d.metadata["file"] == "a.py" and d.metadata["name"] == "helper")
    caller = next(d for d in docs if d.metadata["file"] == "a.py" and d.metadata["name"] == "caller")
    callers = enrich(graph, [helper], docs, "fixture", "Who calls helper?")
    assert {(d.metadata["file"], d.metadata["name"]) for d in callers} == {("a.py", "caller"), ("b.py", "cross")}
    assert enrich(graph, [helper], docs, "fixture", "What does helper call?") == []
    assert requested_direction("Qui appelle helper ?") == "callers"
    assert requested_direction("Quelles fonctions appellent helper ?") == "callers"
    assert requested_direction("Who does caller call?") == "callees"
    assert requested_direction("谁调用 helper？") == "callers"
    assert requested_direction("Explain helper") == "both"
    assert not enrich(graph, [helper], docs, "fixture", "Who calls A.run?")
    assert len(enrich(graph, [helper], docs, "fixture", "Explain helper", max_chunks=1)) == 1
    assert not enrich(graph, [helper], docs, "fixture", "Who calls helper?", max_chars=1)
    assert not enrich(graph, [helper, caller], docs, "fixture", "What does caller call?")
    engine = CodeRAGEngine.__new__(CodeRAGEngine)
    engine.repo_name = "fixture"
    evidence = evidence_from_docs([helper] + callers, "fixture")
    context = engine._format_context([helper] + callers)
    assert all(f"[{e['citation_id']}] {e['file']}:{e['line_start']}" in context for e in evidence)
    assert all(e["retrieval_stage"] == "graph" for e in evidence[1:])
    assert all(d.metadata["type"] == "function" for d in docs if d.metadata["name"] == "caller")


def test_traversal_prefers_higher_ranked_seed_before_budget_cutoff(tmp_path):
    docs, graph = fixture(tmp_path)
    helper = next(d for d in docs if d.metadata["file"] == "a.py" and d.metadata["name"] == "helper")
    run = next(d for d in docs if d.metadata["file"] == "a.py" and d.metadata["name"] == "run")
    chosen = enrich(graph, [run, helper], docs, "fixture", "Explain both", max_chunks=1)
    assert [(d.metadata["file"], d.metadata["name"]) for d in chosen] == [("a.py", "A")]
    directed = enrich(graph, [run, helper], docs, "fixture", "What does A.run call?")
    assert [(d.metadata["file"], d.metadata["name"]) for d in directed] == [("a.py", "A")]


def test_unmapped_and_stale_chunks_are_not_used(tmp_path):
    docs, graph = fixture(tmp_path)
    seed = next(d for d in docs if d.metadata["file"] == "a.py" and d.metadata["name"] == "helper")
    changed = [d.model_copy(deep=True) for d in docs]
    for d in changed:
        if d.metadata["file"] == "b.py" and d.metadata["name"] == "cross": d.page_content += "# changed"
    result = enrich(graph, [seed], changed, "fixture", "Who calls helper?")
    assert [(d.metadata["file"], d.metadata["name"]) for d in result] == [("a.py", "caller")]
    path = tmp_path / "graph.json"
    write_graph(graph, path)
    assert load_graph(path, "fixture")["schema"] == graph["schema"]


def test_module_rebinding_does_not_become_a_guessed_import_edge(tmp_path):
    (tmp_path / "a.py").write_text("def helper():\n    return 1\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("from a import helper\nhelper = other\ndef caller():\n    helper()\n", encoding="utf-8")
    graph = build_source_graph(tmp_path, [], "fixture")
    caller = next(n for n in graph["nodes"] if n["qualified"] == "caller")
    assert any(c["source"] == caller["id"] and c["expression"] == "helper" for c in graph["unresolved"])
    assert not any(c["source"] == caller["id"] for c in graph["edges"])


def test_reassigned_self_does_not_become_a_guessed_method_edge(tmp_path):
    (tmp_path / "a.py").write_text("class A:\n    def run(self, other):\n        self = other\n        self.work()\n    def work(self):\n        pass\n", encoding="utf-8")
    graph = build_source_graph(tmp_path, [], "fixture")
    caller = next(n for n in graph["nodes"] if n["qualified"] == "A.run")
    assert any(c["source"] == caller["id"] and c["expression"] == "self.work" for c in graph["unresolved"])
    assert not any(c["source"] == caller["id"] for c in graph["edges"])


def test_local_import_shadowing_is_not_resolved_to_a_module_function(tmp_path):
    (tmp_path / "a.py").write_text("def helper():\n    pass\ndef caller():\n    from external import helper\n    helper()\n", encoding="utf-8")
    graph = build_source_graph(tmp_path, [], "fixture")
    caller = next(n for n in graph["nodes"] if n["qualified"] == "caller")
    assert any(c["source"] == caller["id"] and c["expression"] == "helper" for c in graph["unresolved"])
    assert not any(c["source"] == caller["id"] for c in graph["edges"])


def test_real_rag_path_sends_graph_source_to_generation(tmp_path):
    from types import SimpleNamespace
    docs, graph = fixture(tmp_path)
    helper = next(d for d in docs if d.metadata["file"] == "a.py" and d.metadata["name"] == "helper")
    retriever = SimpleNamespace(docs=docs, k=1, invoke=lambda _: [helper])
    captured = []
    engine = CodeRAGEngine.__new__(CodeRAGEngine)
    engine.repo_name = "fixture"
    engine.vectorstore = SimpleNamespace(as_retriever=lambda **_: retriever)
    engine.bm25_retriever = retriever
    engine.query_transformer = SimpleNamespace(expand_query=lambda q: [q])
    engine.graph = None
    engine.source_graph = graph
    engine.llm = SimpleNamespace(invoke=lambda messages: (captured.extend(messages), SimpleNamespace(content="Found [S2]", response_metadata={}))[1])
    result = engine.query("Who calls helper?", k=1)
    graph_evidence = [e for e in result["evidence"] if e["retrieval_stage"] == "graph"]
    assert {(e["file"], e["symbol"]) for e in graph_evidence} == {("a.py", "caller"), ("b.py", "cross")}
    assert all(any(d.metadata["file"] == e["file"] and d.metadata["line_start"] == e["line_start"]
                   and d.metadata["line_end"] == e["line_end"] for d in docs) for e in graph_evidence)
    assert all(f"[{e['citation_id']}] {e['file']}:{e['line_start']}" in captured[1].content for e in graph_evidence)
    assert all(f"[lines {e['line_start']}-{e['line_end']}]" in captured[1].content for e in graph_evidence)
    assert result["citation_audit"]["status"] == "matched"


def test_missing_explicit_graph_does_not_fall_back_to_legacy(tmp_path):
    import networkx as nx
    from types import SimpleNamespace
    docs, _ = fixture(tmp_path)
    helper = next(d for d in docs if d.metadata["file"] == "a.py" and d.metadata["name"] == "helper")
    retriever = SimpleNamespace(docs=docs, k=1, invoke=lambda _: [helper])
    engine = CodeRAGEngine.__new__(CodeRAGEngine)
    engine.repo_name = "fixture"
    engine.vectorstore = SimpleNamespace(as_retriever=lambda **_: retriever)
    engine.bm25_retriever = retriever
    engine.query_transformer = SimpleNamespace(expand_query=lambda q: [q])
    engine.graph = nx.DiGraph([("helper", "caller")])
    engine.source_graph = None
    engine.disable_legacy_graph = True
    result, _ = engine._retrieve_and_fuse("helper", 1)
    assert result == [helper]
