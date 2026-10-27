"""Controlled retrieval comparison. Default is a deterministic synthetic smoke.

Live mode always copies the source index into an isolated temporary snapshot.
It never opens the source index through Chroma.
"""
from __future__ import annotations
import argparse
from collections import Counter
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
from reposage.query.code_aware_query import CodeAwareQueryPlanner
from reposage.query.multilingual_query import MultilingualQueryPlanner, PlannerConfig, RetrievalRequest, ollama_rewriter
from reposage.query.rag_engine import CodeRAGEngine
from langchain_core.documents import Document

DATASET = Path(__file__).with_name("dataset_v2_candidates.json")
ARMS = ("raw", "translation_only", "milestone2_expanded", "retriever_specific")


def fingerprint(path):
    result = {}
    for f in path.rglob("*"):
        if f.is_symlink():
            raise ValueError("Snapshot input contains a symlink")
        if f.is_file():
            result[f.relative_to(path).as_posix()] = hashlib.sha256(f.read_bytes()).hexdigest()
    return result


def snapshot(source, destination):
    """Copy and verify source bytes without opening its persistent database."""
    before = fingerprint(source)
    if "chroma.sqlite3" not in before or "bm25_docs.pkl" not in before:
        raise ValueError("Source index lacks required artifacts")
    shutil.copytree(source, destination, copy_function=shutil.copy2)
    if before != fingerprint(source) or before != fingerprint(destination):
        raise ValueError("Index changed while snapshot was created")
    return before


def record(doc):
    m = doc.metadata
    return {"file": m.get("file"), "function": m.get("name"), "line_start": m.get("line_start"), "type": m.get("type")}


def matches(record_, gold):
    return all(record_.get(k) == gold[k] for k in ("file", "function", "line_start") if k in gold)


def coverage(gold, indexes):
    checks = {name: None if docs is None else [any(matches(d, g) for d in docs) for g in gold]
              for name, docs in indexes.items()}
    present = [any(v is not None and v[i] for v in checks.values()) for i in range(len(gold))]
    known = all(v is not None for v in checks.values())
    return {"by_component": checks, "evidence_present": present,
            "target_present": True if any(present) else False if known else None}


def metrics(ranking, gold, k):
    positions = [i for i, doc in enumerate(ranking[:k], 1) if any(matches(doc, g) for g in gold)]
    return {"hit_at_k": float(bool(positions)), "mrr_at_k": 1 / positions[0] if positions else 0.0}


class TraceRetriever:
    def __init__(self, inner, stage, calls): self.inner, self.stage, self.calls = inner, stage, calls
    @property
    def docs(self): return self.inner.docs
    @property
    def k(self): return self.inner.k
    @k.setter
    def k(self, value): self.inner.k = value
    def invoke(self, query):
        event = {"retriever": self.stage, "query": query}
        t = time.perf_counter()
        try:
            docs = self.inner.invoke(query)
            event["ranking"] = [record(d) for d in docs]
            return docs
        except Exception as error:
            event["error_type"] = type(error).__name__
            raise
        finally:
            event["latency_ms"] = round((time.perf_counter() - t) * 1000, 3)
            self.calls.append(event)


class TraceVector:
    def __init__(self, inner, calls): self.inner, self.calls = inner, calls
    def as_retriever(self, search_kwargs):
        return TraceRetriever(self.inner.as_retriever(search_kwargs=search_kwargs), "vector", self.calls)


def plan_for(arm, question, planner, k):
    if arm == "raw":
        disabled = MultilingualQueryPlanner(PlannerConfig()).plan(question)
        return replace(disabled, requests=(RetrievalRequest("vector", 0, k), RetrievalRequest("bm25", 0, k)))
    if arm == "retriever_specific":
        return CodeAwareQueryPlanner(planner).plan(question, k)
    plan = planner.plan(question)
    if arm == "translation_only":
        selected = next((i for i, v in enumerate(plan.variants) if v.provenance == "english_rewrite"), 0)
        requests = (RetrievalRequest("vector", selected, k), RetrievalRequest("bm25", selected, k))
    elif arm == "milestone2_expanded":
        requests = tuple(RetrievalRequest(stage, i, k) for stage in ("vector", "bm25") for i in range(len(plan.variants)))
    else:
        raise ValueError("Unknown arm")
    return replace(plan, requests=requests)


def summarize(rows):
    eligible = [r for r in rows if r["metrics"] is not None]
    return {"queries": len(rows), "eligible": len(eligible),
            "covered": sum(r["coverage"]["target_present"] is True for r in rows),
            "unindexable": sum(r["coverage"]["target_present"] is False for r in rows),
            "coverage_unknown": sum(r["coverage"]["target_present"] is None for r in rows),
            "success": sum(r["execution_status"] == "success" for r in rows),
            "partial": sum(r["execution_status"] == "partial" for r in rows),
            "failed": sum(r["execution_status"] == "failed" for r in rows),
            "abstentions_or_fallbacks": sum(r["plan"]["status"] in {"abstained", "rewrite_failed", "rewrite_invalid", "rewrite_not_english", "rewriter_unavailable"} for r in rows),
            "mean_vector_calls": sum(sum(c["retriever"] == "vector" for c in r["calls"]) for r in rows) / len(rows) if rows else None,
            "mean_bm25_calls": sum(sum(c["retriever"] == "bm25" for c in r["calls"]) for r in rows) / len(rows) if rows else None,
            "mean_llm_calls": sum(r["llm_calls"] for r in rows) / len(rows) if rows else None,
            "mean_latency_ms": sum(r["latency_ms"] for r in rows) / len(rows) if rows else None,
            "hit_at_k": sum(r["metrics"]["hit_at_k"] for r in eligible) / len(eligible) if eligible else None,
            "mrr_at_k": sum(r["metrics"]["mrr_at_k"] for r in eligible) / len(eligible) if eligible else None}


def evaluate(dataset, engine, indexes, rewriter, k=5):
    if k < 1: raise ValueError("k must be positive")
    validate_dataset(dataset)
    tasks = {task["task_id"]: task for task in dataset["tasks"]}
    rows = []
    calls = []
    engine.vectorstore = TraceVector(engine.vectorstore, calls)
    if engine.bm25_retriever is not None:
        engine.bm25_retriever = TraceRetriever(engine.bm25_retriever, "bm25", calls)
    for item in dataset["queries"]:
        gold = tasks[item["task_id"]]["evidence"]
        target_coverage = coverage(gold, indexes)
        for arm in ARMS:
            llm_calls = []
            def counted_rewriter(query):
                event = {}
                start_call = time.perf_counter()
                try:
                    return rewriter(query)
                except Exception as error:
                    event["error_type"] = type(error).__name__
                    raise
                finally:
                    event["latency_ms"] = round((time.perf_counter() - start_call) * 1000, 3)
                    llm_calls.append(event)
            planner = MultilingualQueryPlanner(PlannerConfig(enabled=True, timeout_seconds=30), counted_rewriter)
            start = time.perf_counter()
            plan = plan_for(arm, item["query"], planner, k)
            trace_start = len(calls)
            stages = {}
            errors = []
            def observe(stage, docs):
                if stage == "error": errors.append(docs)
                else: stages[stage] = [record(doc) for doc in docs]
            if indexes.get("vector") is None: errors.append({"stage": "vector_index", "error_type": "Unavailable"})
            if indexes.get("bm25") is None: errors.append({"stage": "bm25_index", "error_type": "Unavailable"})
            try:
                engine._retrieve_and_fuse(item["query"], k, stage_observer=observe, retrieval_plan=plan)
            except Exception as error:
                errors.append({"stage": "retrieval", "error_type": type(error).__name__})
            query_calls = calls[trace_start:]
            errors.extend({"stage": c["retriever"], "error_type": c["error_type"]} for c in query_calls if "error_type" in c)
            requested = Counter(r.retriever for r in plan.requests)
            observed = Counter(c["retriever"] for c in query_calls)
            for stage in ("vector", "bm25"):
                if observed[stage] != requested[stage]:
                    errors.append({"stage": stage, "error_type": "MissingExecution"})
            success_stages = {c["retriever"] for c in query_calls if "ranking" in c}
            status = "success" if not errors and success_stages == {"vector", "bm25"} else "partial" if success_stages else "failed"
            # Failed / partial runs and unknown or absent coverage are never accuracy zeros.
            quality = metrics(stages.get("rrf", []), gold, k) if status == "success" and target_coverage["target_present"] is True else None
            rows.append({"query_id": item["query_id"], "task_id": item["task_id"], "pair_id": item["pair_id"], "category": item["category"],
                         "identifier": item["identifier"], "query": item["query"], "arm": arm,
                         "annotation_status": tasks[item["task_id"]]["annotation_status"], "evidence": gold,
                         "coverage": target_coverage, "plan": asdict(plan), "execution_status": status,
                         "errors": errors, "calls": query_calls, "llm_calls": len(llm_calls), "llm_call_events": llm_calls, "stages": stages,
                         "latency_ms": round((time.perf_counter() - start) * 1000, 3), "metrics": quality})
    result = {}
    for arm in ARMS:
        selected = [r for r in rows if r["arm"] == arm]
        pairs = []
        for r in selected:
            if r["category"] == "en": continue
            base = next((b for b in selected if b["pair_id"] == r["pair_id"] and b["category"] == "en"), None)
            if base:
                pairs.append({"query_id": r["query_id"], "english_query_id": base["query_id"], "task_id": r["task_id"], "category": r["category"],
                              "hit_delta": r["metrics"]["hit_at_k"] - base["metrics"]["hit_at_k"] if r["metrics"] and base["metrics"] else None,
                              "mrr_delta": r["metrics"]["mrr_at_k"] - base["metrics"]["mrr_at_k"] if r["metrics"] and base["metrics"] else None})
        result[arm] = {"overall": summarize(selected),
                       "by_category": {x: summarize([r for r in selected if r["category"] == x]) for x in sorted({r["category"] for r in selected})},
                       "by_language_including_mixed": {x: summarize([r for r in selected if x in r["category"].split("-")]) for x in ("en", "fr", "zh")},
                       "by_annotation_status": {x: summarize([r for r in selected if r["annotation_status"] == x]) for x in sorted({r["annotation_status"] for r in selected})},
                       "by_task": {x: summarize([r for r in selected if r["task_id"] == x]) for x in sorted({r["task_id"] for r in selected})},
                       "paired_with_english": pairs}
    paired_arms = []
    for raw in (r for r in rows if r["arm"] == "raw"):
        for other in (r for r in rows if r["query_id"] == raw["query_id"] and r["arm"] != "raw"):
            paired_arms.append({"query_id": raw["query_id"], "pair_id": raw["pair_id"], "task_id": raw["task_id"],
                                "category": raw["category"], "arm": other["arm"],
                                "hit_delta": other["metrics"]["hit_at_k"] - raw["metrics"]["hit_at_k"] if other["metrics"] and raw["metrics"] else None,
                                "mrr_delta": other["metrics"]["mrr_at_k"] - raw["metrics"]["mrr_at_k"] if other["metrics"] and raw["metrics"] else None})
    return {"summaries": result, "paired_vs_raw": paired_arms, "queries": rows}


def validate_dataset(dataset):
    """Reject malformed or provenance-free judgments before scoring."""
    tasks = dataset["tasks"]
    task_ids = [t["task_id"] for t in tasks]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("Duplicate task IDs")
    allowed = {"independently_reviewed", "generated_unverified_candidate"}
    for task in tasks:
        if task.get("annotation_status") not in allowed or not task.get("evidence"):
            raise ValueError("Missing evidence or annotation provenance")
        if task["annotation_status"] == "independently_reviewed" and not task.get("reviewer"):
            raise ValueError("Reviewed annotations require reviewer provenance")
        for item in task["evidence"]:
            if not isinstance(item.get("file"), str) or not isinstance(item.get("function"), str) or not isinstance(item.get("line_start"), int):
                raise ValueError("Evidence requires exact file, function and line")
    query_ids = []
    seen_pairs = set()
    pair_tasks = {}
    for query in dataset["queries"]:
        if query["task_id"] not in task_ids or not query.get("pair_id") or query["category"] not in {"en", "fr", "zh", "en-fr", "en-zh", "fr-zh", "en-fr-zh"} or query["identifier"] not in {"present", "absent", "ambiguous"} or not query["query"].strip():
            raise ValueError("Invalid query annotation")
        pair_key = (query["pair_id"], query["category"])
        if pair_key in seen_pairs: raise ValueError("Duplicate category within pair")
        seen_pairs.add(pair_key)
        pair_description = (query["task_id"], query["identifier"])
        if query["pair_id"] in pair_tasks and pair_tasks[query["pair_id"]] != pair_description:
            raise ValueError("Inconsistent task pair")
        pair_tasks[query["pair_id"]] = pair_description
        query_ids.append(query["query_id"])
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("Duplicate query IDs")


class SyntheticRetriever:
    def __init__(self, docs): self.docs, self.k = docs, 5
    def invoke(self, query):
        # Predictable plumbing-only ordering; this is not a retrieval model.
        return sorted(self.docs, key=lambda d: -(len(set(query.lower().split()) & set(d.page_content.lower().split()))))[:self.k]


class SyntheticVector:
    def __init__(self, docs): self.docs = docs
    def as_retriever(self, search_kwargs):
        retriever = SyntheticRetriever(self.docs)
        retriever.k = search_kwargs["k"]
        return retriever


def synthetic_engine():
    docs = [Document(page_content=text, metadata={"file": file, "name": name, "line_start": line, "type": typ}) for file, name, line, typ, text in [
        ("micrograd/engine.py", "backward", 54, "function", "backward graph gradients calculate"),
        ("micrograd/engine.py", "Value", 2, "class", "Value backward gradients graph"),
        ("micrograd/nn.py", "MLP", 45, "class", "MLP layers input call"),
        ("micrograd/nn.py", "Layer", 30, "class", "Layer input call"),
    ]]
    engine = CodeRAGEngine.__new__(CodeRAGEngine)
    engine.vectorstore, engine.bm25_retriever = SyntheticVector(docs), SyntheticRetriever(docs)
    engine.graph = None
    return engine, {"vector": [record(d) for d in docs], "bm25": [record(d) for d in docs]}


def revision(path):
    try:
        top = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], cwd=path, text=True, stderr=subprocess.DEVNULL).strip()
        if Path(top).resolve() != Path(path).resolve(): return None
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError): return None


def dirty(path):
    if revision(path) is None: return None
    try:
        return bool(subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all"], cwd=path, text=True, stderr=subprocess.DEVNULL).strip())
    except (OSError, subprocess.CalledProcessError): return None


def _snapshot_worker(index_dir, dataset_path, result_path, url, model, embedding, k):
    """Run in a child so Windows releases Chroma file handles on exit."""
    index_dir = Path(index_dir).resolve()
    original = (ROOT / "backend" / "chroma_db").resolve()
    if index_dir.is_relative_to(original):
        raise ValueError("Worker cannot open the original index")
    dataset = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
    engine = CodeRAGEngine("micrograd", persist_dir=str(index_dir.parent), ollama_base_url=url,
                           ollama_model=model, ollama_embed_model=embedding)
    try:
        metadata = engine.vectorstore._collection.get(include=["metadatas"])["metadatas"]
        vector = [{"file": m.get("file"), "function": m.get("name"), "line_start": m.get("line_start")} for m in metadata] or None
    except Exception: vector = None
    bm25 = [record(d) for d in engine.bm25_retriever.docs] if engine.bm25_retriever and engine.bm25_retriever.docs else None
    rewriter = ollama_rewriter(PlannerConfig(enabled=True, timeout_seconds=30), url, model)
    result = evaluate(dataset, engine, {"vector": vector, "bm25": bm25}, rewriter, int(k))
    Path(result_path).write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--live-index-source", type=Path, help="Source directory copied to an isolated temporary index")
    parser.add_argument("--source-checkout", type=Path, default=ROOT / "backend" / "tmp" / "reposage" / "micrograd")
    parser.add_argument("--model", default="qwen3:1.7b")
    parser.add_argument("--embedding-model", default="nomic-embed-text")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    args = parser.parse_args()
    if args.output.resolve().is_relative_to((ROOT / "backend" / "chroma_db").resolve()):
        raise ValueError("Output cannot be inside the existing index directory")
    if args.output.exists():
        raise FileExistsError("Output already exists; choose a new report path")
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    if args.live_index_source:
        if args.output.resolve().is_relative_to(args.live_index_source.resolve()):
            raise ValueError("Output cannot be inside the source index")
        with tempfile.TemporaryDirectory(prefix="reposage-eval-") as temp:
            source = args.live_index_source.resolve()
            copied = Path(temp) / "micrograd"
            source_hashes = snapshot(source, copied)
            child_result = Path(temp) / "result.json"
            worker = "import sys; from benchmarks.cross_lingual_code_qa.controlled_eval import _snapshot_worker; _snapshot_worker(*sys.argv[1:])"
            subprocess.run([sys.executable, "-c", worker, str(copied), str(args.dataset.resolve()), str(child_result),
                            args.ollama_url, args.model, args.embedding_model, str(args.k)], cwd=ROOT, check=True)
            result = json.loads(child_result.read_text(encoding="utf-8"))
            identity = {"kind": "verified_isolated_snapshot", "source_hashes": source_hashes,
                        "source_unchanged": fingerprint(source) == source_hashes,
                        "snapshot_changed_during_retrieval": fingerprint(copied) != source_hashes,
                        "indexed_embedding_model": "unknown_not_recorded_in_index", "query_embedding_model": args.embedding_model}
    else:
        engine, indexes = synthetic_engine()
        def fake_rewriter(masked):
            marker = "REPOSAGEPROTECTED0TOKEN " if "REPOSAGEPROTECTED0TOKEN" in masked else ""
            return "How does " + marker + "compute gradients in the graph?"
        result = evaluate(dataset, engine, indexes, fake_rewriter, args.k)
        identity = {"kind": "synthetic_fixture_no_real_index", "indexed_embedding_model": None, "query_embedding_model": None}
    result.update({"schema_version": "1.0.0", "dataset_kind": dataset["benchmark_kind"],
                   "validated_research_benchmark": dataset["validated_research_benchmark"],
                   "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
                   "repository_revision": revision(args.source_checkout), "repository_dirty": dirty(args.source_checkout),
                   "expected_repository_revision": dataset.get("corpus", {}).get("expected_source_revision"),
                   "code_revision": revision(ROOT), "code_dirty": dirty(ROOT),
                   "implementation_sha256": {str(p.relative_to(ROOT)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest() for p in (
                       ROOT / "backend/reposage/query/multilingual_query.py", ROOT / "backend/reposage/query/code_aware_query.py",
                       ROOT / "backend/reposage/query/rag_engine.py", Path(__file__))},
                   "index": identity, "configuration": {"k": args.k, "fixed_budget_arms": ["raw", "translation_only", "retriever_specific"],
                   "expanded_budget_arm": "milestone2_expanded", "rewrite_model": args.model if args.live_index_source else "deterministic_fake",
                   "rewrite_timeout_seconds": 30, "rewrite_max_tokens": 160, "rewrite_temperature": 0,
                   "answer_generation": False, "python_version": sys.version.split()[0]}})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        output.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"index": identity["kind"], "summaries": {arm: result["summaries"][arm]["overall"] for arm in ARMS}}, indent=2))


if __name__ == "__main__": main()
