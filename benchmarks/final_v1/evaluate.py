"""Evaluate frozen tasks on verified copies of existing indexes, never originals."""
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "backend"), str(ROOT)]
from benchmarks.cross_lingual_code_qa.controlled_eval import fingerprint, snapshot, plan_for
from benchmarks.graph_ablation.run_ablation import TraceRetriever, TraceVector, record
from reposage.ingestion.source_graph import build_source_graph
from reposage.query.evidence import audit_citations, evidence_from_docs
from reposage.query.graph_answer import graph_answer
from reposage.query.multilingual_query import MultilingualQueryPlanner, PlannerConfig, ollama_rewriter
from reposage.query.rag_engine import CodeRAGEngine, SYSTEM_PROMPT, QUERY_PROMPT, _answer_language_instruction
from reposage.query.source_graph import SymbolLookup, enrich, graph_context, load_graph
from langchain_core.messages import HumanMessage, SystemMessage

DATA = ROOT / "benchmarks/final_v1/benchmark.json"
SOURCES = {"micrograd": ROOT / "backend/tmp/reposage/micrograd", "sample_repo": ROOT / "backend/tests/fixtures/sample_repo"}
ANSWER_TASKS = {"m_callee", "m_caller", "s_callee", "m_noanswer", "s_noanswer", "s_hash"}


def matches(doc, target):
    m = doc.metadata
    return m.get("file") == target["file"] and m.get("name") == target["function"] and m.get("line_start") == target["line_start"] and m.get("line_end") == target["line_end"]


def result_metrics(docs, targets):
    if not targets: return None
    hits = [any(matches(doc, target) for doc in docs) for target in targets]
    first = next((i for i, doc in enumerate(docs, 1) if any(matches(doc, target) for target in targets)), None)
    return {"recall": sum(hits) / len(hits), "all_targets": all(hits), "hit": bool(first), "reciprocal_rank": 1 / first if first else 0.0}


def worker(repo, index, dataset_file, output, micro_graph):
    data = json.loads(Path(dataset_file).read_text(encoding="utf-8"))
    actual = {f.relative_to(SOURCES[repo]).as_posix(): hashlib.sha256(f.read_bytes()).hexdigest()
              for f in sorted(SOURCES[repo].rglob("*.py")) if "__pycache__" not in f.parts and f.is_file()}
    if actual != data["repositories"][repo]["source_sha256"]: raise ValueError("Source revision mismatch")
    engine = CodeRAGEngine(repo, persist_dir=str(Path(index).parent), ollama_model="qwen3:1.7b", ollama_embed_model="nomic-embed-text")
    engine.llm.num_predict = 256
    if not engine.bm25_retriever: raise RuntimeError("BM25 unavailable")
    graph = (load_graph(micro_graph, repo) if repo == "micrograd"
             else build_source_graph(SOURCES[repo], engine.bm25_retriever.docs, repo))
    calls = []
    engine.vectorstore = TraceVector(engine.vectorstore, calls)
    engine.bm25_retriever = TraceRetriever(engine.bm25_retriever, "bm25", calls)
    engine.graph, engine.source_graph = None, None
    vector_meta = engine.vectorstore.inner._collection.get(include=["metadatas"])["metadatas"]
    tasks = {x["id"]: x for x in data["tasks"] if x["repo"] == repo}
    config = PlannerConfig(enabled=True, timeout_seconds=30)
    rewriter = ollama_rewriter(config, "http://127.0.0.1:11434", "qwen3:1.7b")
    rows = []
    for query in data["queries"]:
        if query["task_id"] not in tasks: continue
        task = tasks[query["task_id"]]
        targets = task["targets"]
        coverage = [{"target": t, "bm25": any(matches(d, t) for d in engine.bm25_retriever.docs),
                     "vector": any(m.get("file") == t["file"] and m.get("name") == t["function"] and m.get("line_start") == t["line_start"] and m.get("line_end") == t["line_end"] for m in vector_meta)} for t in targets]
        retrieval = {}
        saved_seeds = []
        for arm in ("raw", "translation_only"):
            llm_count = []
            def counted_rewriter(text):
                llm_count.append(1)
                return rewriter(text)
            planner = MultilingualQueryPlanner(config, counted_rewriter)
            start = time.perf_counter()
            plan = plan_for(arm, query["question"], planner, 5)
            begin = len(calls)
            stages = {}
            def observer(stage, value): stages[stage] = list(value) if isinstance(value, list) else value
            error = None
            try: seeds, _ = engine._retrieve_and_fuse(query["question"], 5, stage_observer=observer, retrieval_plan=plan)
            except Exception as exc: seeds, error = [], type(exc).__name__
            per_calls = calls[begin:]
            errors = [c["error"] for c in per_calls if "error" in c]
            if error: errors.append(error)
            if len(per_calls) != 2: errors.append("MissingRetrieverCall")
            rankings = {c["channel"]: c.get("candidates", []) for c in per_calls}
            retrieval[arm] = {"plan": asdict(plan), "calls": per_calls, "rankings": rankings, "rrf": [record(x) for x in seeds],
                              "metrics": result_metrics(seeds, targets) if not errors else None, "errors": errors,
                              "llm_calls": len(llm_count), "latency_ms": round((time.perf_counter() - start) * 1000, 3)}
            if arm == "raw": saved_seeds = list(seeds)
        # Identical raw seeds and same graph budget for both graph variants.
        with patch("reposage.query.source_graph.resolve_symbol", return_value=SymbolLookup("not_applicable")):
            qualified = enrich(graph, saved_seeds, engine.bm25_retriever.docs, repo, query["question"])
        info = {}
        grounded = enrich(graph, saved_seeds, engine.bm25_retriever.docs, repo, query["question"], diagnostics=info)
        graphs = {"off": saved_seeds, "qualified": saved_seeds + qualified, "grounded": saved_seeds + grounded}
        graph_results = {name: {"ranking": [record(x) for x in values], "metrics": result_metrics(values, targets),
                         "context_characters": len(engine._format_context(values))} for name, values in graphs.items()}
        answer_data = None
        if task["id"] in ANSWER_TASKS:
            evidence = evidence_from_docs(graphs["grounded"], repo)
            context = engine._format_context(graphs["grounded"]) + graph_context(info, evidence)
            start = time.perf_counter()
            formatted = graph_answer(query["question"], info, evidence)
            format_ms = round((time.perf_counter() - start) * 1000, 3)
            messages = [SystemMessage(content=SYSTEM_PROMPT.format(repo_name=repo) + "\n" + _answer_language_instruction(query["question"])),
                        HumanMessage(content=QUERY_PROMPT.format(repo_name=repo, context=context, question=query["question"]))]
            start = time.perf_counter()
            model_error = None
            try:
                model = engine.llm.invoke(messages, think=False)
                answer = model.content
                truncated = model.response_metadata.get("done_reason") == "length"
            except Exception as exc:
                answer, truncated, model_error = "", None, type(exc).__name__
            answer_data = {"evidence": evidence, "graph": info, "model": {"answer": answer, "citation_audit": audit_citations(answer, evidence),
                "truncated": truncated, "error": model_error, "latency_ms": round((time.perf_counter() - start) * 1000, 3)},
                "evidence_first": {"answer": formatted, "citation_audit": audit_citations(formatted or "", evidence), "applicable": formatted is not None, "latency_ms": format_ms},
                "factual_judgments": "unverified; inspect source and output", "language_compliance": "unverified"}
        rows.append({"query_id": query["id"], "task_id": task["id"], "task_type": task["type"], "language": query["language"], "question": query["question"],
                     "coverage": coverage, "retrieval": retrieval, "graph": graph_results, "grounding": info, "answer_evaluation": answer_data})
        print(repo, query["id"], "raw_errors", len(retrieval["raw"]["errors"]), flush=True)
    Path(output).write_text(json.dumps({"repo": repo, "queries": rows}, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--micro-index", type=Path, required=True)
    parser.add_argument("--sample-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, default=DATA)
    parser.add_argument("--micro-graph", type=Path, default=ROOT / "validation/m6b_sidecars/micrograd.json")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or any(output.is_relative_to(p.resolve()) for p in (args.micro_index, args.sample_index, ROOT / "backend/chroma_db")):
        raise ValueError("Output must be new and outside indexes")
    benchmark = args.benchmark.resolve()
    datasets = json.loads(benchmark.read_text(encoding="utf-8"))
    data = {"benchmark_sha256": hashlib.sha256(benchmark.read_bytes()).hexdigest(), "benchmark_version": datasets["version"],
            "configuration": {"k": 5, "embedding_query_model": "nomic-embed-text", "rewrite_model": "qwen3:1.7b",
             "model_answer": "qwen3:1.7b", "answer_tokens": 256, "original_index_embedding_model": "not_recorded",
             "graph_arms": "identical RRF seeds; qualified arm disables exact symbol grounding in-memory only"}, "indexes": {}, "queries": []}
    with tempfile.TemporaryDirectory(prefix="final-v1-") as temp:
        for repo, source in (("micrograd", args.micro_index), ("sample_repo", args.sample_index)):
            source = source.resolve()
            copied, result = Path(temp) / repo, Path(temp) / (repo + "_results.json")
            hashes = snapshot(source, copied)
            subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", repo, str(copied), str(benchmark), str(result), str(args.micro_graph.resolve())], cwd=ROOT, check=True)
            data["queries"].extend(json.loads(result.read_text(encoding="utf-8"))["queries"])
            data["indexes"][repo] = {"source_hashes": hashes, "source_unchanged": fingerprint(source) == hashes, "copy_changed": fingerprint(copied) != hashes}
    with output.open("x", encoding="utf-8") as f: json.dump(data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker": worker(*sys.argv[2:])
    else: main()
