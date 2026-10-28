"""Ablate the production v2 graph on identical RRF seeds in a verified copy."""
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
from benchmarks.graph_ablation.run_ablation import TraceRetriever, TraceVector, matches, record, token_count
from benchmarks.graph_ablation.source_benchmark import make_dataset
from reposage.query.evidence import chunk_identity, evidence_from_docs
from reposage.query.rag_engine import CodeRAGEngine
from reposage.query.source_graph import requested_direction


def worker(index, dataset, source, graph_dir, output, task_filter, measure_tokens):
    import httpx
    data = json.loads(Path(dataset).read_text(encoding="utf-8"))
    if make_dataset(Path(source)) != data:
        raise ValueError("Dataset no longer matches source or generator")
    os.environ["REPOSAGE_SOURCE_GRAPH_DIR"] = graph_dir
    engine = CodeRAGEngine("micrograd", persist_dir=str(Path(index).parent), ollama_model="qwen3:1.7b", ollama_embed_model="nomic-embed-text")
    if not engine.source_graph or not engine.bm25_retriever:
        raise RuntimeError("v2 graph or BM25 unavailable")
    engine.query_transformer.expand_query = lambda q: [q]
    calls = []
    engine.vectorstore = TraceVector(engine.vectorstore, calls)
    engine.bm25_retriever = TraceRetriever(engine.bm25_retriever, "bm25", calls)
    vector_metadata = engine.vectorstore.inner._collection.get(include=["metadatas"])["metadatas"]
    tasks = {t["id"]: t for t in data["tasks"]}
    if task_filter and task_filter not in tasks:
        raise ValueError("Unknown task filter")
    graph = engine.source_graph
    node_map = {n["id"]: n for n in graph["nodes"]}
    rows = []
    with httpx.Client(timeout=120) as client:
        for q in data["queries"]:
            if task_filter and q["task_id"] != task_filter:
                continue
            task = tasks[q["task_id"]]
            stages, times, errors = {}, {}, []
            def observe(stage, docs):
                times[stage] = time.perf_counter()
                if stage == "error": errors.append(docs)
                else: stages[stage] = list(docs)
            offset = len(calls)
            start = time.perf_counter()
            docs, _ = engine._retrieve_and_fuse(q["question"], 5, stage_observer=observe)
            elapsed = (time.perf_counter() - start) * 1000
            seed, extra = stages.get("rrf", []), stages.get("graph_enrichment", [])
            off_evidence, on_evidence = evidence_from_docs(seed, "micrograd"), evidence_from_docs(docs, "micrograd")
            off_ids = {e["id"] for e in off_evidence}
            new = [e for e in on_evidence if e["id"] not in off_ids]
            seed_ids = {chunk_identity(d, "micrograd") for d in seed}
            seed_nodes = [n for n in graph["nodes"] if n["chunk_mapping"] == "exact" and n["chunk_id"] in seed_ids]
            selected = {n["id"] for n in seed_nodes}
            direction = requested_direction(q["question"])
            traced_edges = [e for e in graph["edges"] if (e["source"] in selected and direction in ("both", "callees")) or (e["target"] in selected and direction in ("both", "callers"))]
            target_coverage = [{"target": t, "bm25_exact": any(matches(d, t) for d in engine.bm25_retriever.docs),
                "vector_exact": any(m.get("file") == t["file"] and m.get("name") == t["qualified"].rsplit(".", 1)[-1] and m.get("line_start") == t["line_start"] and m.get("line_end") == t["line_end"] for m in vector_metadata),
                "off": any(matches(d, t) for d in seed), "on": any(matches(d, t) for d in docs)} for t in task["targets"]]
            off_context, on_context = engine._format_context(seed), engine._format_context(docs)
            tokens = {"status": "not_measured", "delta": None}
            if measure_tokens:
                try:
                    off = token_count(client, q["question"], off_context)
                    on = token_count(client, q["question"], on_context)
                    tokens = {"status": "measured_ollama_prompt_eval_count", "off": off, "on": on, "delta": on - off}
                except Exception as exc:
                    tokens = {"status": "failed", "delta": None, "error_type": type(exc).__name__}
            rows.append({"task_id": q["task_id"], "category": task["category"], "language": q["language"], "query": q["question"],
                "calls": calls[offset:], "rrf_seeds": [record(d) for d in seed], "graph_additions": [record(d) for d in extra],
                "direction": direction, "seed_nodes": seed_nodes, "edges": [{**e, "source_node": node_map[e["source"]], "target_node": node_map[e["target"]]} for e in traced_edges],
                "unresolved_seed_calls": [e for e in graph["unresolved"] if e["source"] in selected],
                "target_coverage": target_coverage, "evidence_off": off_evidence, "evidence_on": on_evidence,
                "new_evidence_ids": [e["id"] for e in new],
                "new_evidence_in_context": {e["citation_id"]: f"[{e['citation_id']}] {e['file']}:{e['line_start']}" in on_context for e in new},
                "context_chars_off": len(off_context), "context_chars_on": len(on_context), "prompt_tokens": tokens,
                "retrieval_ms_including_graph": round(elapsed, 3),
                "graph_stage_ms": round((times["graph_enrichment"] - times["rrf"]) * 1000, 3) if "graph_enrichment" in times else None,
                "errors": errors + [c["error"] for c in calls[offset:] if "error" in c]})
            print(q["task_id"], q["language"], "new", len(new), "errors", len(rows[-1]["errors"]), flush=True)
    Path(output).write_text(json.dumps({"queries": rows}, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-id")
    parser.add_argument("--skip-tokens", action="store_true")
    args = parser.parse_args()
    source, output = args.index.resolve(), args.output.resolve()
    if output.exists() or output.is_relative_to(source) or output.is_relative_to((ROOT / "backend/chroma_db").resolve()):
        raise ValueError("Output must be new and outside index data")
    with tempfile.TemporaryDirectory(prefix="graph-v2-ablation-") as temp:
        copied, result = Path(temp) / "micrograd", Path(temp) / "result.json"
        hashes = snapshot(source, copied)
        cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", str(copied), str(args.dataset.resolve()), str(args.source.resolve()), str(args.graph_dir.resolve()), str(result), args.task_id or "", str(int(not args.skip_tokens))]
        subprocess.run(cmd, cwd=ROOT, check=True)
        data = json.loads(result.read_text(encoding="utf-8"))
        data.update({"source_hashes": hashes, "source_unchanged": fingerprint(source) == hashes,
                     "snapshot_changed": fingerprint(copied) != hashes, "configuration": {"k": 5, "query_expansion": False,
                     "embedding_model": "nomic-embed-text", "graph": "source_graph_v2", "task_filter": args.task_id,
                     "prompt_probe": "disabled" if args.skip_tokens else "qwen3:1.7b, one output token"}})
    with output.open("x", encoding="utf-8") as f: json.dump(data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker": worker(*sys.argv[2:7], sys.argv[7] or None, bool(int(sys.argv[8])))
    else: main()
