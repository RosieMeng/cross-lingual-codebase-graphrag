"""Real retrieval traces for symbol grounding on a verified index copy."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT))
from benchmarks.cross_lingual_code_qa.controlled_eval import fingerprint, snapshot
from benchmarks.graph_ablation.run_ablation import TraceRetriever, TraceVector, record
from reposage.query.evidence import evidence_from_docs
from reposage.query.rag_engine import CodeRAGEngine
from reposage.query.source_graph import graph_context

QUERIES = [
    ("en-caller", "Who calls `Value.backward`?"),
    ("en-callee", "What does `Value.backward` call?"),
    ("fr-caller", "Qui appelle `Value.backward` ?"),
    ("fr-callee", "Qu'appelle `Value.backward` ?"),
    ("zh-caller", "谁调用Value.backward？"),
    ("zh-callee", "Value.backward调用了哪些函数？"),
    ("mixed-caller", "Who calls `Value.backward` dans le code？"),
    ("mixed-callee", "What does `Value.backward` call 在计算图中？"),
    ("en-no-id", "Where does the code traverse the graph to compute gradients?"),
    ("zh-no-id", "哪里实现了通过计算图传播梯度？"),
]


def worker(index, sidecars, output):
    os.environ["REPOSAGE_SOURCE_GRAPH_DIR"] = sidecars
    engine = CodeRAGEngine("micrograd", persist_dir=str(Path(index).parent), ollama_model="qwen3:1.7b", ollama_embed_model="nomic-embed-text")
    if not engine.source_graph or not engine.bm25_retriever: raise RuntimeError("Missing qualified graph or BM25")
    engine.query_transformer.expand_query = lambda q: [q]
    calls = []
    engine.vectorstore = TraceVector(engine.vectorstore, calls)
    engine.bm25_retriever = TraceRetriever(engine.bm25_retriever, "bm25", calls)
    rows = []
    for label, question in QUERIES:
        stages, info = {}, {}
        def observe(stage, value): stages[stage] = list(value) if isinstance(value, list) else value
        offset = len(calls)
        start = time.perf_counter()
        docs, _ = engine._retrieve_and_fuse(question, 5, stage_observer=observe, graph_observer=info.update)
        elapsed = round((time.perf_counter() - start) * 1000, 3)
        seeds = stages.get("rrf", [])
        evidence = evidence_from_docs(docs, "micrograd")
        context = engine._format_context(docs) + graph_context(info, evidence)
        has_target = lambda ds: any(d.metadata.get("file") == "micrograd/engine.py" and d.metadata.get("name") == "backward" and d.metadata.get("line_start") == 54 for d in ds)
        rows.append({"id": label, "question": question, "calls": calls[offset:], "rrf_seeds": [record(d) for d in seeds],
                     "graph_additions": [record(d) for d in stages.get("graph_enrichment", [])], "graph": info,
                     "evidence": evidence, "target_off": has_target(seeds), "target_on": has_target(docs),
                     "target_in_context": any(e["file"] == "micrograd/engine.py" and e["symbol"] == "backward" and f"[{e['citation_id']}] micrograd/engine.py:54" in context for e in evidence),
                     "context_characters": len(context), "latency_ms": elapsed,
                     "errors": [c["error"] for c in calls[offset:] if "error" in c] + ([stages["error"]] if "error" in stages else [])})
        print(label, info.get("lookup_status"), info.get("direction"), "target", rows[-1]["target_off"], rows[-1]["target_on"], flush=True)
    Path(output).write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--sidecars", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    index, output = args.index.resolve(), args.output.resolve()
    if output.exists() or output.is_relative_to(index) or output.is_relative_to((ROOT / "backend/chroma_db").resolve()): raise ValueError("Unsafe output")
    with tempfile.TemporaryDirectory(prefix="symbol-grounding-") as temp:
        copied, result = Path(temp) / "micrograd", Path(temp) / "result.json"
        hashes = snapshot(index, copied)
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", str(copied), str(args.sidecars.resolve()), str(result)], cwd=ROOT, check=True)
        data = {"description": "Unreviewed multilingual retrieval smoke; graph off uses identical RRF seeds", "queries": json.loads(result.read_text(encoding="utf-8")),
                "source_unchanged": fingerprint(index) == hashes, "source_hashes": hashes}
    with output.open("x", encoding="utf-8") as f: json.dump(data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker": worker(*sys.argv[2:])
    else: main()
