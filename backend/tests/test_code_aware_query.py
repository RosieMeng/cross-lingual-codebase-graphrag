import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from reposage.query.code_aware_query import CodeAwareQueryPlanner, query_spans
from reposage.query.multilingual_query import MultilingualQueryPlanner, PlannerConfig
from benchmarks.cross_lingual_code_qa.controlled_eval import (
    ARMS, DATASET, coverage, evaluate, fingerprint, snapshot, synthetic_engine, validate_dataset,
)


def planner(rewriter):
    return CodeAwareQueryPlanner(MultilingualQueryPlanner(PlannerConfig(enabled=True), rewriter))


def test_three_language_spans_and_fixed_routes_preserve_symbols():
    q = "How does Value.backward calculer les gradients 通过计算图？"
    plan = planner(lambda _: "How does REPOSAGEPROTECTED0TOKEN compute gradients?").plan(q, 5)
    assert plan.variants[0].text == q
    assert plan.variants[1].text == "How does Value.backward compute gradients?"
    assert [(r.retriever, r.variant_index, r.candidate_limit) for r in plan.requests] == [("vector", 1, 5), ("bm25", 0, 5)]
    assert {s.label for s in plan.spans} >= {"en", "fr", "han_script", "identifier"}
    assert all(s.confidence is None for s in plan.spans)
    assert [q[s.start:s.end] for s in plan.spans if s.kind == "identifier"] == ["Value.backward"]


def test_ambiguous_words_not_claimed_as_code_and_paths_preserved():
    q = "How does call work in src/auth.py with authenticate_user?"
    spans = query_spans(q)
    assert not any(q[s.start:s.end] == "call" for s in spans)
    assert {q[s.start:s.end] for s in spans if s.kind in {"path", "identifier"}} == {"src/auth.py", "authenticate_user"}


def test_english_and_fallback_route_raw_for_both_channels():
    en = "How does Value.backward work?"
    plan = planner(lambda _: pytest.fail("English must not invoke rewriter")).plan(en, 3)
    assert [plan.variants[r.variant_index].text for r in plan.requests] == [en, en]
    fr = "Comment fonctionne authenticate_user dans le code ?"
    plan = planner(lambda _: "How does it work?").plan(fr, 3)
    assert plan.status == "rewrite_invalid"
    assert [plan.variants[r.variant_index].text for r in plan.requests] == [fr, fr]


def test_raw_bm25_keeps_exact_whitespace_tokenization():
    q = "Comment `foo bar` utilise src/auth.py et authenticate_user ?"
    plan = planner(lambda _: "How does REPOSAGEPROTECTED0TOKEN use REPOSAGEPROTECTED1TOKEN and REPOSAGEPROTECTED2TOKEN?").plan(q, 4)
    lexical = next(r for r in plan.requests if r.retriever == "bm25")
    assert plan.variants[lexical.variant_index].text.split() == q.split()


def test_controlled_evaluation_fixed_budget_and_rank_traces():
    dataset = json.loads(DATASET.read_text(encoding="utf-8"))
    engine, indexes = synthetic_engine()
    def rewrite(q):
        marker = "REPOSAGEPROTECTED0TOKEN " if "REPOSAGEPROTECTED0TOKEN" in q else ""
        return "How does " + marker + "compute gradients in the graph?"
    report = evaluate(dataset, engine, indexes, rewrite, 3)
    assert len(report["queries"]) == len(dataset["queries"]) * len(ARMS)
    for row in report["queries"]:
        assert row["execution_status"] == "success"
        assert row["stages"]["rrf"]
        assert row["coverage"]["target_present"] is True
        if row["arm"] != "milestone2_expanded":
            assert sorted(c["retriever"] for c in row["calls"]) == ["bm25", "vector"]
            assert all(r["candidate_limit"] == 3 for r in row["plan"]["requests"])
    row = next(r for r in report["queries"] if r["query_id"] == "free-fr" and r["arm"] == "milestone2_expanded")
    assert len(row["calls"]) == 4
    assert row["llm_calls"] == 1
    assert report["summaries"]["raw"]["paired_with_english"]
    assert len(report["paired_vs_raw"]) == len(dataset["queries"]) * 3


def test_coverage_unknown_and_missing_do_not_become_accuracy_zero():
    gold = [{"file": "a.py", "function": "f", "line_start": 1}, {"file": "a.py", "function": "g", "line_start": 2}]
    assert coverage(gold, {"vector": [], "bm25": []})["target_present"] is False
    assert coverage(gold, {"vector": [], "bm25": None})["target_present"] is None
    assert coverage(gold, {"vector": [gold[1]], "bm25": None})["target_present"] is True


def test_invalid_annotation_provenance_rejected():
    dataset = json.loads(DATASET.read_text(encoding="utf-8"))
    dataset["tasks"][0]["annotation_status"] = "reviewed_maybe"
    with pytest.raises(ValueError): validate_dataset(dataset)


def test_exceptions_are_reported_separately_from_accuracy():
    dataset = json.loads(DATASET.read_text(encoding="utf-8"))
    dataset["queries"] = dataset["queries"][:1]
    engine, indexes = synthetic_engine()
    def fail(query): raise RuntimeError("private")
    engine.bm25_retriever.invoke = fail
    result = evaluate(dataset, engine, indexes, fail, 2)
    for row in result["queries"]:
        assert row["execution_status"] == "partial"
        assert row["metrics"] is None
        assert row["errors"][0]["stage"] == "bm25"


def test_missing_index_is_recorded_and_not_scored():
    dataset = json.loads(DATASET.read_text(encoding="utf-8"))
    dataset["queries"] = dataset["queries"][:1]
    engine, indexes = synthetic_engine()
    indexes["bm25"] = None
    engine.bm25_retriever = None
    report = evaluate(dataset, engine, indexes, lambda _: "unused", 2)
    assert all(row["execution_status"] == "partial" and row["metrics"] is None for row in report["queries"])


def test_snapshot_is_byte_identical_and_does_not_mutate_source(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "chroma.sqlite3").write_bytes(b"vector")
    (source / "bm25_docs.pkl").write_bytes(b"lexical")
    before = fingerprint(source)
    copied = tmp_path / "copy"
    assert snapshot(source, copied) == before == fingerprint(copied)
    (copied / "chroma.sqlite3").write_bytes(b"changed")
    assert fingerprint(source) == before
