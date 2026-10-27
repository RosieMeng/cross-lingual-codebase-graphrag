import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmarks.cross_lingual_code_qa.metrics import (
    aggregate_metrics,
    function_hit_at_k,
    hit_at_k,
    query_metrics,
    reciprocal_rank,
)


def test_hit_at_k_and_reciprocal_rank_use_ranked_evidence():
    results = [
        {"file": "other.py", "function": "target"},
        {"file": "target.py", "function": "target"},
    ]
    gold = {"file": "target.py", "function": "target"}

    assert hit_at_k(results, "file", "target.py", 2) == 1.0
    assert function_hit_at_k(results, gold, 1) == 0.0
    assert function_hit_at_k(results, gold, 2) == 1.0
    assert reciprocal_rank(results, gold) == 0.5


def test_query_metrics_and_language_aggregation():
    gold = {"file": "src/model.py", "function": "forward"}
    row = {
        "language": "fr",
        "metrics": query_metrics([gold], gold, cutoffs=(1, 3)),
    }

    assert row["metrics"] == {
        "file_hit@1": 1.0,
        "function_hit@1": 1.0,
        "file_hit@3": 1.0,
        "function_hit@3": 1.0,
        "mrr": 1.0,
    }
    assert aggregate_metrics([row], "fr")["mrr"] == 1.0
    assert aggregate_metrics([row], "zh") == {"query_count": 0}


@pytest.mark.parametrize("k", [0, -1])
def test_hit_metrics_reject_nonpositive_cutoffs(k):
    with pytest.raises(ValueError):
        hit_at_k([], "file", "x.py", k)


def test_toy_dataset_has_aligned_en_fr_zh_queries_and_is_not_research_validated():
    dataset_path = ROOT / "benchmarks" / "cross_lingual_code_qa" / "dataset_v1.json"
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    tasks = {}
    for query in dataset["queries"]:
        tasks.setdefault(query["task_id"], []).append(query)

    assert dataset["schema_version"] == "1.0.0"
    assert dataset["benchmark_kind"] == "toy_smoke"
    assert dataset["validated_research_benchmark"] is False
    assert len(tasks) == 3
    for task_queries in tasks.values():
        assert {query["language"] for query in task_queries} == {"en", "fr", "zh"}
        assert len({(query["gold"]["file"], query["gold"]["function"]) for query in task_queries}) == 1


@pytest.mark.parametrize("failure,expected", [("none", "complete"), ("unindexable", "complete"), ("bm25_missing", "partial"), ("vector_error", "partial"), ("bm25_error", "partial"), ("query_error", "failed"), ("vector_missing", "failed")])
def test_runner_reports_coverage_and_execution_failures(tmp_path, monkeypatch, failure, expected):
    from types import SimpleNamespace
    from langchain_core.documents import Document
    from benchmarks.cross_lingual_code_qa import run_baseline
    import reposage.query.rag_engine as rag

    gold = {"file": "x.py", "function": "target", "line_start": 1}
    doc = Document(page_content="code", metadata={"file": "x.py", "name": "target", "line_start": 1})
    if failure == "unindexable":
        doc.metadata["name"] = "MLP"
    dataset = {"schema_version": "1", "benchmark_kind": "toy_smoke", "validated_research_benchmark": False, "corpus": {}, "queries": [{"query_id": "q", "task_id": "t", "language": "en", "query": "question", "gold": gold}]}
    dataset_file = tmp_path / "dataset.json"
    dataset_file.write_text(json.dumps(dataset), encoding="utf-8")
    collection = tmp_path / "index" / "sample"
    collection.mkdir(parents=True)
    if failure != "vector_missing":
        (collection / "chroma.sqlite3").touch()

    class Retriever:
        docs = [doc]
        k = 1
        def __init__(self, stage): self.stage = stage
        def invoke(self, query):
            if failure == self.stage + "_error": raise RuntimeError("secret")
            return [doc]

    class Vector:
        _collection = SimpleNamespace(get=lambda **kw: {"metadatas": [doc.metadata]})
        def as_retriever(self, search_kwargs): return Retriever("vector")

    class Engine:
        def __init__(self, **kwargs):
            self.query_transformer = SimpleNamespace()
            self.vectorstore = Vector()
            self.bm25_retriever = None if failure == "bm25_missing" else Retriever("bm25")
        def _retrieve_and_fuse(self, query, k, stage_observer):
            if failure == "query_error": raise RuntimeError("secret")
            vector = []
            bm25 = []
            try: vector = self.vectorstore.as_retriever({"k": k}).invoke(query)
            except RuntimeError: pass
            if self.bm25_retriever:
                try: bm25 = self.bm25_retriever.invoke(query)
                except RuntimeError: pass
            docs = vector or bm25
            stage_observer("rrf", docs)
            return docs, []

    monkeypatch.setattr(rag, "CodeRAGEngine", Engine)
    args = SimpleNamespace(dataset=dataset_file, persist_dir=tmp_path / "index", clone_dir=tmp_path, repo_name="sample", k=10)
    report = run_baseline.run(args)
    assert report["retrieval_status"] == expected
    row = report["queries"][0]
    assert row["execution_status"] == {"complete": "success", "partial": "partial", "failed": "failed"}[expected]
    assert (row["metrics"] is not None) == (expected == "complete" and failure != "unindexable")
    if failure == "unindexable":
        assert row["index_coverage"]["target_present"] is False
        assert row["gold"] == gold
        assert report["coverage"]["unindexable"] == 1
    if failure == "none":
        assert row["index_coverage"]["target_present"] is True
        assert report["coverage"]["eligible_successful_queries"] == 1
