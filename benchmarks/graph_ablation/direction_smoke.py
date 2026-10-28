"""Compare real caller/callee requests using a verified temporary index copy."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT))
from benchmarks.cross_lingual_code_qa.controlled_eval import fingerprint, snapshot
from benchmarks.graph_ablation.run_ablation import record
from reposage.query.rag_engine import CodeRAGEngine
from reposage.query.source_graph import requested_direction


def worker(index, sidecars, output):
    os.environ["REPOSAGE_SOURCE_GRAPH_DIR"] = sidecars
    engine = CodeRAGEngine("micrograd", persist_dir=str(Path(index).parent), ollama_model="qwen3:1.7b", ollama_embed_model="nomic-embed-text")
    if not engine.source_graph: raise RuntimeError("No v2 graph loaded")
    engine.query_transformer.expand_query = lambda q: [q]
    rows = []
    for question in ("Who calls Value.backward?", "What does Value.backward call?"):
        stages = {}
        docs, _ = engine._retrieve_and_fuse(question, 5, stage_observer=lambda stage, value: stages.update({stage: list(value) if isinstance(value, list) else value}))
        rows.append({"query": question, "direction": requested_direction(question),
                     "seeds": [record(d) for d in stages.get("rrf", [])],
                     "additions": [record(d) for d in stages.get("graph_enrichment", [])],
                     "errors": stages.get("error", []), "final": [record(d) for d in docs]})
    Path(output).write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--sidecars", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    index, output = args.index.resolve(), args.output.resolve()
    if output.exists() or output.is_relative_to(index) or output.is_relative_to((ROOT / "backend/chroma_db").resolve()): raise ValueError("Unsafe output")
    with tempfile.TemporaryDirectory(prefix="graph-direction-") as temp:
        copied, result = Path(temp) / "micrograd", Path(temp) / "result.json"
        hashes = snapshot(index, copied)
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", str(copied), str(args.sidecars.resolve()), str(result)], cwd=ROOT, check=True)
        data = {"queries": json.loads(result.read_text(encoding="utf-8")), "source_unchanged": fingerprint(index) == hashes, "source_hashes": hashes}
    with output.open("x", encoding="utf-8") as f: json.dump(data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker": worker(*sys.argv[2:])
    else: main()
