"""Query one verified index snapshot once, and compare its RRF seeds to graph output."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT))
from benchmarks.cross_lingual_code_qa.controlled_eval import fingerprint, snapshot
from benchmarks.graph_ablation.source_benchmark import analyze, make_dataset
from reposage.query.evidence import chunk_identity, evidence_from_docs
from reposage.query.rag_engine import CodeRAGEngine, QUERY_PROMPT, SYSTEM_PROMPT, _answer_language_instruction


def record(doc):
    m = doc.metadata
    return {"id": chunk_identity(doc, "micrograd"), "file": m.get("file"), "name": m.get("name"),
            "line_start": m.get("line_start"), "line_end": m.get("line_end"), "type": m.get("type")}


def matches(doc, target):
    m = doc.metadata
    return (m.get("file") == target["file"] and m.get("name") == target["qualified"].rsplit(".", 1)[-1]
            and m.get("line_start") == target["line_start"] and m.get("line_end") == target["line_end"])


class TraceRetriever:
    def __init__(self, inner, kind, calls): self.inner, self.kind, self.calls = inner, kind, calls
    @property
    def docs(self): return self.inner.docs
    @property
    def k(self): return self.inner.k
    @k.setter
    def k(self, value): self.inner.k = value
    def invoke(self, query):
        started = time.perf_counter()
        event = {"channel": self.kind, "query": query}
        try:
            docs = self.inner.invoke(query)
            event["candidates"] = [record(d) for d in docs]
            return docs
        except Exception as exc:
            event["error"] = type(exc).__name__
            raise
        finally:
            event["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
            self.calls.append(event)


class TraceVector:
    def __init__(self, inner, calls): self.inner, self.calls = inner, calls
    def as_retriever(self, search_kwargs):
        return TraceRetriever(self.inner.as_retriever(search_kwargs=search_kwargs), "vector", self.calls)


def source_for_doc(doc, symbols):
    matches_ = [s for s in symbols if matches(doc, asdict(s))]
    return matches_[0] if len(matches_) == 1 else None


def edge_status(caller, callee, facts):
    if caller is None or callee is None:
        return "unresolved_source_identity"
    direct = [c for c in facts if c.source == caller and c.target == callee]
    if direct:
        return "statically_supported"
    conflicting = [c for c in facts if c.source == caller and c.target is not None
                   and c.expression.rsplit(".", 1)[-1] == callee.name and c.target != callee]
    return "conflicting_resolved_target" if conflicting else "not_established"


def trace_graph(engine, seeds, symbols, facts):
    graph = engine.graph
    lookup = {d.metadata.get("name"): d for d in engine.bm25_retriever.docs if d.metadata.get("name")}
    traces = []
    for seed in seeds:
        name = seed.metadata.get("name")
        seed_source = source_for_doc(seed, symbols)
        matches_ = [asdict(s) for s in symbols if s.name == name]
        item = {"seed": record(seed), "source_symbol": asdict(seed_source) if seed_source else None,
                "graph_node": name if graph and graph.has_node(name) else None,
                "same_name_source_symbols": matches_, "neighbors": []}
        if item["graph_node"]:
            neighbors = [(n, "caller") for n in graph.predecessors(name)] + [(n, "callee") for n in graph.successors(name)]
            for i, (neighbor, direction) in enumerate(neighbors):
                doc = lookup.get(neighbor)
                neighbor_source = source_for_doc(doc, symbols) if doc else None
                caller, callee = (neighbor_source, seed_source) if direction == "caller" else (seed_source, neighbor_source)
                item["neighbors"].append({"name": neighbor, "direction": direction, "within_first_three": i < 3,
                    "edge": [neighbor, name] if direction == "caller" else [name, neighbor],
                    "resolved_chunk": record(doc) if doc else None,
                    "relationship_status": edge_status(caller, callee, facts) if doc else "no_indexed_chunk"})
        traces.append(item)
    return traces


def token_count(client, question, context):
    r = client.post("http://127.0.0.1:11434/api/chat", json={"model": "qwen3:1.7b", "stream": False, "think": False,
        "options": {"num_predict": 1, "temperature": 0}, "messages": [
            {"role": "system", "content": SYSTEM_PROMPT.format(repo_name="micrograd") + "\n" + _answer_language_instruction(question)},
            {"role": "user", "content": QUERY_PROMPT.format(repo_name="micrograd", context=context, question=question)}]})
    r.raise_for_status()
    return r.json().get("prompt_eval_count")


def worker(index, dataset, source, output, measure_tokens, task_filter=None):
    import httpx
    data = json.loads(Path(dataset).read_text(encoding="utf-8"))
    fresh = make_dataset(Path(source))
    for key in ("source_revision", "source_sha256", "facts"):
        if data[key] != fresh[key]:
            raise ValueError(f"Dataset/source mismatch: {key}")
    fresh_tasks = {t["id"]: t for t in fresh["tasks"]}
    fresh_queries = {(q["task_id"], q["language"]): q for q in fresh["queries"]}
    if len({t["id"] for t in data["tasks"]}) != len(data["tasks"]) or any(fresh_tasks.get(t["id"]) != t for t in data["tasks"]):
        raise ValueError("Dataset/source mismatch: tasks")
    if len({(q["task_id"], q["language"]) for q in data["queries"]}) != len(data["queries"]) or any(fresh_queries.get((q["task_id"], q["language"])) != q for q in data["queries"]):
        raise ValueError("Dataset/source mismatch: queries")
    symbols, facts = analyze(Path(source))
    engine = CodeRAGEngine("micrograd", persist_dir=str(Path(index).parent), ollama_model="qwen3:1.7b", ollama_embed_model="nomic-embed-text")
    if engine.graph is None or engine.bm25_retriever is None:
        raise RuntimeError("Required graph or BM25 index unavailable")
    engine.query_transformer.expand_query = lambda text: [text]
    calls = []
    engine.vectorstore = TraceVector(engine.vectorstore, calls)
    engine.bm25_retriever = TraceRetriever(engine.bm25_retriever, "bm25", calls)
    vector_metadata = engine.vectorstore.inner._collection.get(include=["metadatas"])["metadatas"]
    tasks = {t["id"]: t for t in data["tasks"]}
    if task_filter and task_filter not in tasks:
        raise ValueError("Unknown task filter")
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
            started = time.perf_counter()
            docs, _ = engine._retrieve_and_fuse(q["question"], 5, stage_observer=observe)
            total_ms = (time.perf_counter() - started) * 1000
            seed = stages.get("rrf", [])
            added = stages.get("graph_enrichment", [])
            ctx_off, ctx_on = engine._format_context(seed), engine._format_context(docs)
            ev_off, ev_on = evidence_from_docs(seed, "micrograd"), evidence_from_docs(docs, "micrograd")
            off_ids = {e["id"] for e in ev_off}
            new_ev = [e for e in ev_on if e["id"] not in off_ids]
            target_coverage = [{"target": target,
                "bm25_exact": any(matches(d, target) for d in engine.bm25_retriever.docs),
                "vector_exact": any(m.get("file") == target["file"] and m.get("name") == target["qualified"].rsplit(".", 1)[-1]
                                    and m.get("line_start") == target["line_start"] and m.get("line_end") == target["line_end"] for m in vector_metadata),
                "off": any(matches(d, target) for d in seed), "on": any(matches(d, target) for d in docs)} for target in task["targets"]]
            tokens = {"status": "not_measured", "off": None, "on": None, "delta": None}
            if measure_tokens and new_ev:
                try:
                    off = token_count(client, q["question"], ctx_off)
                    on = token_count(client, q["question"], ctx_on)
                    tokens = {"status": "measured_ollama_prompt_eval_count", "off": off, "on": on, "delta": on - off}
                except Exception as exc:
                    tokens = {"status": "failed", "error_type": type(exc).__name__, "off": None, "on": None, "delta": None}
            row = {"task_id": task["id"], "category": task["category"], "language": q["language"], "query": q["question"],
                   "calls": calls[offset:], "rrf_seeds": [record(d) for d in seed], "graph_additions": [record(d) for d in added],
                   "graph_trace": trace_graph(engine, seed, symbols, facts), "target_coverage": target_coverage,
                   "evidence_off": ev_off, "evidence_on": ev_on, "new_evidence_ids": [e["id"] for e in new_ev],
                   "new_evidence_in_context": {e["citation_id"]: f"[{e['citation_id']}] {e['file']}:{e['line_start']}" in ctx_on for e in new_ev},
                   "context_chars_off": len(ctx_off), "context_chars_on": len(ctx_on), "prompt_tokens": tokens,
                   "retrieval_ms_including_graph": round(total_ms, 3),
                   "graph_stage_ms": round((times["graph_enrichment"] - times["rrf"]) * 1000, 3) if "graph_enrichment" in times and "rrf" in times else None,
                   "errors": errors + [c["error"] for c in calls[offset:] if "error" in c]}
            rows.append(row)
            print(task["id"], q["language"], "graph", len(new_ev), "errors", len(row["errors"]), flush=True)
    Path(output).write_text(json.dumps({"queries": rows}, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-index", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt-tokens", action="store_true")
    parser.add_argument("--task-id", help="Optional exact task ID; report records this filter")
    args = parser.parse_args()
    source, output = args.source_index.resolve(), args.output.resolve()
    if output.exists() or output.is_relative_to(source) or output.is_relative_to((ROOT / "backend/chroma_db").resolve()):
        raise ValueError("Output must be new and outside index data")
    with tempfile.TemporaryDirectory(prefix="graph-ablation-") as temp:
        copied, result = Path(temp) / "micrograd", Path(temp) / "result.json"
        hashes = snapshot(source, copied)
        cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", str(copied), str(args.dataset.resolve()), str(args.source.resolve()), str(result), str(int(args.prompt_tokens)), args.task_id or ""]
        subprocess.run(cmd, cwd=ROOT, check=True)
        data = json.loads(result.read_text(encoding="utf-8"))
        data.update({"source_hashes": hashes, "source_unchanged": fingerprint(source) == hashes,
                     "index_copy_changed": fingerprint(copied) != hashes,
                     "configuration": {"k": 5, "query_expansion": False, "embedding_model": "nomic-embed-text", "task_filter": args.task_id, "answer_generation": "only one-token prompt-count probes when enabled"}})
    with output.open("x", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        worker(*sys.argv[2:6], bool(int(sys.argv[6])), sys.argv[7] or None)
    else:
        main()
