"""Freeze one supplemental task omitted from the first frozen dataset."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmarks.graph_ablation.source_benchmark import analyze

MAIN = ROOT / "benchmarks/final_v1/benchmark.json"
OUT = ROOT / "benchmarks/final_v1/crossfile_supplement.json"


def main():
    if OUT.exists(): raise FileExistsError(OUT)
    data = json.loads(MAIN.read_text(encoding="utf-8"))
    symbols, calls = analyze(ROOT / "backend/tmp/reposage/micrograd")
    caller = next(s for s in symbols if s.file == "test/test_engine.py" and s.qualified == "test_sanity_check")
    callee = next(s for s in symbols if s.file == "micrograd/engine.py" and s.qualified == "Value")
    call = next(c for c in calls if c.source == caller and c.target == callee and c.resolution == "explicit_from_import")
    def target(s): return {"file": s.file, "qualified": s.qualified, "function": s.name, "line_start": s.line_start, "line_end": s.line_end}
    task = {"id": "m_crossfile", "repo": "micrograd", "type": "cross_file", "identifier": "present",
            "targets": [target(caller), target(callee)], "relationship": {"caller": caller.qualified, "callee": callee.qualified,
            "call_line": call.line, "resolution": call.resolution}, "fact_provenance": "Python AST explicit import and call (pre-existing Milestone 6A case)",
            "language_judgment": "unverified", "human_reviewed": False}
    questions = ["Find the call from `test_sanity_check` in `test/test_engine.py` to `Value` in `micrograd/engine.py`.",
                 "Trouvez l'appel de `test_sanity_check` dans `test/test_engine.py` vers `Value` dans `micrograd/engine.py`.",
                 "查找 `test/test_engine.py` 中 `test_sanity_check` 对 `micrograd/engine.py` 中 `Value` 的调用。",
                 "Find l'appel de `test_sanity_check` dans `test/test_engine.py` vers `Value` dans `micrograd/engine.py`."]
    data.update({"version": "final-v1-crossfile-supplement-1", "selection_note": "Frozen after main results because the requested cross-file category was omitted; pre-existing Milestone 6A case; report separately.",
                 "tasks": [task], "queries": [{"id": f"m_crossfile-{lang}", "task_id": "m_crossfile", "language": lang, "question": q}
                 for lang, q in zip(("en", "fr", "zh", "mixed"), questions)]})
    with OUT.open("x", encoding="utf-8") as f: json.dump(data, f, ensure_ascii=False, indent=2)
    print("frozen supplemental cross-file task", call.line)


if __name__ == "__main__": main()
