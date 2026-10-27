import copy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmarks.cross_lingual_code_qa.analyze_milestone3 import ARMS, generate
from benchmarks.cross_lingual_code_qa.validate_review_candidates import validate

DATASET = ROOT / "benchmarks/cross_lingual_code_qa/review_candidates_v1.json"


def candidate_data():
    return json.loads(DATASET.read_text(encoding="utf-8"))


def test_candidate_spans_revisions_and_review_states():
    data = candidate_data()
    assert len(data["repositories"]) == 2
    assert len(data["tasks"]) == 5
    assert len(data["queries"]) == 36
    assert data["annotation_status"] == "unverified"
    assert not validate(data)
    assert all(e["coverage"]["status"] == "not_checked" for t in data["tasks"] for e in t["evidence"])


def test_rejects_corrupt_offsets_false_review_and_unproven_coverage():
    data = candidate_data()
    data["queries"][0]["spans"][0]["end"] += 1
    assert any("Invalid span" in e for e in validate(data))
    data = candidate_data()
    data["annotation_status"] = "independently_reviewed"
    assert any("not all accepted" in e for e in validate(data))
    data = candidate_data()
    data["tasks"][0]["evidence"][0]["coverage"]["status"] = "present"
    assert any("missing manifest" in e for e in validate(data))


def test_review_disagreement_needs_adjudication_and_splits_do_not_overlap():
    data = candidate_data()
    review = data["queries"][0]["review"]
    review.update(status="accepted", reviewers=["a", "b", "c"], decisions=[
        {"reviewer": "a", "decision": "accept", "timestamp": "2026-01-01T00:00:00Z", "reason": "sound"},
        {"reviewer": "b", "decision": "reject", "timestamp": "2026-01-01T00:00:00Z", "reason": "ambiguous"},
        {"reviewer": "c", "decision": "accept", "timestamp": "2026-01-02T00:00:00Z", "reason": "resolved"},
    ])
    data["annotation_status"] = "mixed_review"
    assert any("lack adjudication" in e for e in validate(data))
    review["adjudication"] = {"reviewer": "c", "timestamp": "2026-01-02T00:00:00Z", "reason": "resolved"}
    assert not validate(data)
    clone = copy.deepcopy(data["repositories"][0])
    clone.update(repo_id="same_repo_alias", split="test")
    data["repositories"][0]["split"] = "development"
    data["repositories"].append(clone)
    assert any("development and test" in e for e in validate(data))


def test_analysis_compares_all_channels_and_graph_without_index_access():
    gold = {"file": "a.py", "function": "target", "line_start": 1}
    other = {"file": "b.py", "function": "other", "line_start": 2}
    rows = []
    for arm in ARMS:
        rrf = [other, gold] if arm == "retriever_specific" else [gold, other]
        rows.append({"query_id": "q", "category": "en-fr", "identifier": "absent", "query": "test query",
                     "arm": arm, "evidence": [gold], "execution_status": "success", "plan": {"status": "rewritten"},
                     "calls": [{"retriever": "vector", "query": "test query", "ranking": [gold]},
                               {"retriever": "bm25", "query": "test query", "ranking": [other]}],
                     "llm_calls": 1, "llm_call_events": [{"latency_ms": 2.0}], "latency_ms": 3.0,
                     "stages": {"rrf": rrf, "graph_enrichment": [gold]}})
    report = generate({"queries": rows}, "hash")
    assert "below translation-only: q" in report
    assert report.count("V ranking:") == 4
    assert report.count("B ranking:") == 4
    assert report.count("RRF:") == 4
    assert report.count("G:") == 4
    assert "a.py:1 target*" in report
    with pytest.raises(ValueError): generate({"queries": rows + [copy.deepcopy(rows[0])]}, "hash")
