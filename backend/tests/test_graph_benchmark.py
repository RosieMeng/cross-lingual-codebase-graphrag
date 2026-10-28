"""Tests for source-derived facts, ambiguity and question validation."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmarks.graph_ablation.source_benchmark import analyze, make_dataset, question, validate_question


def test_ast_facts_are_qualified_and_resolve_only_static_imports(tmp_path):
    (tmp_path / "a.py").write_text("class A:\n    def run(self):\n        return 1\n\ndef helper():\n    return 1\n\ndef caller():\n    return helper()\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("from a import helper\n\nclass B:\n    def run(self):\n        return helper()\n\ndef unknown(x):\n    return x.run()\n", encoding="utf-8")
    symbols, calls = analyze(tmp_path)
    assert {(s.file, s.qualified) for s in symbols if s.name == "run"} == {("a.py", "A.run"), ("b.py", "B.run")}
    local = next(c for c in calls if c.source.qualified == "caller")
    assert (local.target.file, local.target.qualified, local.resolution) == ("a.py", "helper", "same_module_definition")
    cross = next(c for c in calls if c.source.qualified == "B.run")
    assert (cross.target.file, cross.target.qualified, cross.resolution) == ("a.py", "helper", "explicit_from_import")
    dynamic = next(c for c in calls if c.source.qualified == "unknown")
    assert dynamic.target is None
    data = make_dataset(tmp_path)
    assert {t["category"] for t in data["tasks"]} == {"exact_definition", "same_named_definition", "local_caller_callee", "cross_file_caller_callee", "unresolved_call"}
    assert len(data["queries"]) == 4 * len(data["tasks"])
    task = data["tasks"][0]
    text = question(task, "zh")
    assert validate_question(task, "zh", text)
    assert not validate_question(task, "zh", text + " invented")


def test_duplicate_qualified_definitions_do_not_resolve_arbitrarily(tmp_path):
    (tmp_path / "m.py").write_text("def f(): pass\ndef f(): pass\ndef caller():\n    f()\n", encoding="utf-8")
    _, calls = analyze(tmp_path)
    call = next(c for c in calls if c.source.qualified == "caller")
    assert call.target is None


def test_reassigned_nested_function_is_not_resolved(tmp_path):
    (tmp_path / "m.py").write_text("def caller(other):\n    def f():\n        pass\n    f = other\n    f()\n", encoding="utf-8")
    _, calls = analyze(tmp_path)
    assert next(c for c in calls if c.source.qualified == "caller").target is None


def test_graph_trace_distinguishes_collisions_and_missing_chunks(tmp_path):
    import networkx as nx
    from types import SimpleNamespace
    from langchain_core.documents import Document
    from benchmarks.graph_ablation.run_ablation import trace_graph
    (tmp_path / "a.py").write_text("def f():\n    return 1\n\ndef caller():\n    return f()\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("def f():\n    return 2\n", encoding="utf-8")
    symbols, facts = analyze(tmp_path)
    caller = Document(page_content="caller", metadata={"file": "a.py", "name": "caller", "line_start": 4, "line_end": 5})
    wrong = Document(page_content="f", metadata={"file": "b.py", "name": "f", "line_start": 1, "line_end": 2})
    graph = nx.DiGraph()
    graph.add_edge("caller", "f")
    graph.add_edge("caller", "missing")
    engine = SimpleNamespace(graph=graph, bm25_retriever=SimpleNamespace(docs=[caller, wrong]))
    trace = trace_graph(engine, [caller], symbols, facts)[0]
    assert trace["neighbors"][0]["relationship_status"] == "conflicting_resolved_target"
    assert trace["neighbors"][0]["resolved_chunk"]["file"] == "b.py"
    assert trace["neighbors"][1]["relationship_status"] == "no_indexed_chunk"


def test_experimental_graph_preserves_qualified_ids_and_unresolved_calls(tmp_path):
    from benchmarks.graph_ablation.qualified_graph import build
    (tmp_path / "a.py").write_text("class A:\n    def run(self):\n        self.work()\n    def work(self):\n        return 1\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("class B:\n    def run(self, x):\n        x.work()\n", encoding="utf-8")
    graph = build(tmp_path)
    ids = {n["id"] for n in graph["nodes"]}
    assert "a.py::A.run@2-3" in ids and "b.py::B.run@2-3" in ids
    assert any(e["source"] == "a.py::A.run@2-3" and e["target"] == "a.py::A.work@4-5" for e in graph["edges"])
    assert any(e["source"] == "b.py::B.run@2-3" and e["expression"] == "x.work" for e in graph["unresolved_calls"])
    assert graph["enabled_in_production"] is False
