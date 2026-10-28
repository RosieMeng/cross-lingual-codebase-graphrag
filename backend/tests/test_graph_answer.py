"""Evidence-first graph answers do not promote unresolved calls to facts."""
import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reposage.query.evidence import audit_citations
from reposage.query.graph_answer import answer_language, graph_answer


def facts():
    evidence = [
        {"id": "root", "citation_id": "S1", "file": "micrograd/engine.py", "line_start": 54, "line_end": 70},
        {"id": "callee", "citation_id": "S2", "file": "micrograd/engine.py", "line_start": 59, "line_end": 64},
        {"id": "test", "citation_id": "S3", "file": "test/test_engine.py", "line_start": 28, "line_end": 67},
    ]
    info = {"lookup_status": "matched", "direction": "callees",
            "root": {"qualified": "Value.backward", "file": "micrograd/engine.py", "line_start": 54, "line_end": 70, "chunk_id": "root"},
            "relationships": [{"caller": "Value.backward", "caller_file": "micrograd/engine.py", "caller_chunk_id": "root", "caller_citation_id": "S1",
                "call_line": 65, "expression": "build_topo", "callee": "Value.backward.build_topo", "callee_file": "micrograd/engine.py",
                "callee_line_start": 59, "callee_line_end": 64, "callee_chunk_id": "callee", "callee_citation_id": "S2", "status": "static_reference"}],
            "unresolved": [{"caller": "Value.backward", "file": "micrograd/engine.py", "line": 70, "expression": "v._backward",
                "caller_chunk_id": "root", "caller_citation_id": "S1", "status": "unresolved_target"}],
            "relationships_total": 1, "unresolved_total": 1}
    return info, evidence


@pytest.mark.parametrize("question,marker", [
    ("What does Value.backward call?", "Static reference"),
    ("Qu'appelle Value.backward ?", "Référence statique"),
    ("Value.backward调用了哪些函数？", "静态引用"),
    ("What does Value.backward call dans le graphe ?", "Static reference"),
    ("What does Value.backward call? Answer in French.", "Référence statique"),
    ("What does Value.backward call? 用中文回答。", "静态引用"),
])
def test_multilingual_evidence_first_call_answers(question, marker):
    info, evidence = facts()
    answer = graph_answer(question, info, evidence)
    assert marker in answer
    assert "micrograd/engine.py:65 [S1]" in answer
    assert "micrograd/engine.py:59-64 [S2]" in answer
    assert "v._backward" in answer
    assert audit_citations(answer, evidence)["status"] == "matched"
    assert "computes gradients" not in answer


def test_caller_observation_is_not_a_confirmed_target():
    info, evidence = facts()
    info["direction"] = "callers"
    info["relationships"] = []
    info["relationships_total"] = 0
    info["unresolved"] = [{"caller": "test_more_ops", "file": "test/test_engine.py", "line": 42,
        "expression": "g.backward", "caller_chunk_id": "test", "caller_citation_id": "S3", "status": "unresolved_target"}]
    answer = graph_answer("Who calls Value.backward?", info, evidence)
    assert "Observed call site; target unresolved" in answer
    assert "test/test_engine.py:42 [S3]" in answer
    assert "Static reference" not in answer
    assert "may be incomplete" in answer


def test_mismatched_source_or_unretrieved_target_is_never_cited():
    info, evidence = facts()
    info["unresolved"][0]["line"] = 71
    wrong = copy.deepcopy(evidence)
    wrong[1]["file"] = "wrong.py"
    answer = graph_answer("What does Value.backward call?", info, wrong)
    assert "build_topo" not in answer and "v._backward" not in answer
    assert "not shown" in answer


def test_unsupported_target_abstains_and_non_call_questions_keep_model_path():
    info, evidence = facts()
    info["lookup_status"] = "ambiguous"
    assert "cannot identify" in graph_answer("Who calls `run`?", info, evidence)
    info["direction"] = "both"
    assert graph_answer("Explain Value.backward", info, evidence) is None
    assert answer_language("Qu'appelle Value.backward ?") == "fr"
