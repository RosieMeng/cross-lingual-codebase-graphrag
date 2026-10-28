"""Checks for frozen benchmark structure and exact-coverage scoring."""
import json
from pathlib import Path
import sys

from langchain_core.documents import Document

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "backend")]
from benchmarks.final_v1.evaluate import result_metrics


def test_final_benchmark_preserves_task_pairing_and_annotation_status():
    data = json.loads((ROOT / "benchmarks/final_v1/benchmark.json").read_text(encoding="utf-8"))
    assert len(data["tasks"]) == 13
    assert not data["independently_human_validated"]
    assert {t["repo"] for t in data["tasks"]} == {"micrograd", "sample_repo"}
    assert {t["type"] for t in data["tasks"]} >= {"definition", "callee", "caller", "behavior", "ambiguous", "no_answer"}
    for task in data["tasks"]:
        queries = [q for q in data["queries"] if q["task_id"] == task["id"]]
        assert {q["language"] for q in queries} == {"en", "fr", "zh", "mixed"}
        assert task["language_judgment"] == "unverified"
        assert bool(task["targets"]) == (task["type"] != "no_answer")


def test_exact_metrics_do_not_count_wrong_ranges_or_no_answer_as_hit():
    target = {"file": "a.py", "function": "run", "line_start": 3, "line_end": 6}
    correct = Document(page_content="a", metadata={"file": "a.py", "name": "run", "line_start": 3, "line_end": 6})
    wrong = Document(page_content="a", metadata={"file": "b.py", "name": "run", "line_start": 3, "line_end": 6})
    result = result_metrics([wrong, correct], [target])
    assert result == {"recall": 1.0, "all_targets": True, "hit": True, "reciprocal_rank": 0.5}
    assert result_metrics([wrong], [target])["recall"] == 0
    assert result_metrics([correct], []) is None


def test_crossfile_supplement_is_explicit_and_separate():
    data = json.loads((ROOT / "benchmarks/final_v1/crossfile_supplement.json").read_text(encoding="utf-8"))
    assert "after main results" in data["selection_note"]
    assert len(data["tasks"]) == 1
    task = data["tasks"][0]
    assert task["type"] == "cross_file"
    assert task["targets"][0]["file"] != task["targets"][1]["file"]
    assert task["relationship"]["resolution"] == "explicit_from_import"
    assert {q["language"] for q in data["queries"]} == {"en", "fr", "zh", "mixed"}
