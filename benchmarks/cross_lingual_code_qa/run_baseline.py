"""Run raw-query retrieval against an existing index without answer generation."""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmarks.cross_lingual_code_qa.metrics import aggregate_metrics, query_metrics, _matches_function


BACKEND = ROOT / "backend"
DATASET_PATH = Path(__file__).with_name("dataset_v1.json")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_value(args: list[str], cwd: Path) -> str | None:
    try:
        return subprocess.check_output(args, cwd=cwd, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _document_record(doc) -> dict:
    metadata = doc.metadata
    chunk_type = metadata.get("type")
    return {
        "file": metadata.get("file"),
        "function": metadata.get("name"),
        "type": chunk_type,
        "line_start": metadata.get("line_start"),
        "line_end": metadata.get("line_end"),
    }


class _ObservedRetriever:
    def __init__(self, retriever, stage, traces, errors):
        self._retriever = retriever
        self._stage = stage
        self._traces = traces
        self._errors = errors

    def invoke(self, query):
        try:
            docs = self._retriever.invoke(query)
            self._traces[self._stage].append([_document_record(doc) for doc in docs])
            return docs
        except Exception as error:
            self._errors.append({"stage": self._stage, "error_type": type(error).__name__})
            raise


class _ObservedVectorStore:
    def __init__(self, vectorstore, traces, errors):
        self._vectorstore = vectorstore
        self._traces = traces
        self._errors = errors

    def as_retriever(self, search_kwargs):
        try:
            retriever = self._vectorstore.as_retriever(search_kwargs=search_kwargs)
        except Exception as error:
            self._errors.append({"stage": "vector", "error_type": type(error).__name__})
            raise
        return _ObservedRetriever(retriever, "vector", self._traces, self._errors)


class _ObservedBM25:
    def __init__(self, retriever, traces, errors):
        self._retriever = retriever
        self._traces = traces
        self._errors = errors

    @property
    def docs(self):
        return self._retriever.docs

    @property
    def k(self):
        return self._retriever.k

    @k.setter
    def k(self, value):
        self._retriever.k = value

    def invoke(self, query):
        return _ObservedRetriever(self._retriever, "bm25", self._traces, self._errors).invoke(query)


def _corpus_identity(collection_dir: Path, source_dir: Path, expected_source_revision: str | None) -> dict:
    metadata_path = collection_dir / "metadata.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        metadata = {}

    artifact_hashes = {}
    for name in ("metadata.json", "bm25_docs.pkl", "graph.pkl", "chroma.sqlite3"):
        artifact = collection_dir / name
        if artifact.is_file():
            artifact_hashes[name] = _sha256(artifact)

    source_revision = _git_value(["git", "rev-parse", "HEAD"], source_dir)
    return {
        "repo_name": collection_dir.name,
        "source_revision": source_revision,
        "expected_source_revision": expected_source_revision,
        "source_revision_matches_expected": source_revision == expected_source_revision,
        "chunk_count": metadata.get("chunks"),
        "files_processed": metadata.get("files_processed"),
        "artifact_sha256": artifact_hashes,
    }


def run(args) -> dict:
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    collection_dir = args.persist_dir / args.repo_name
    source_dir = args.clone_dir / args.repo_name
    traces = {"vector": [], "bm25": []}
    errors = []
    rows = []

    sys.path.insert(0, str(BACKEND))
    from reposage.query.rag_engine import CodeRAGEngine

    engine = None
    engine_error = None
    coverage_indexes = {"vector": None, "bm25": None}
    component_errors = []
    try:
        if not (collection_dir / "chroma.sqlite3").is_file():
            raise FileNotFoundError("Vector index missing")
        engine = CodeRAGEngine(
            repo_name=args.repo_name,
            persist_dir=str(args.persist_dir),
            ollama_base_url="http://localhost:11434",
            ollama_model="qwen2.5-coder:7b",
            ollama_embed_model="nomic-embed-text",
        )
        engine.query_transformer.expand_query = lambda query: [query]
        try:
            metadata = engine.vectorstore._collection.get(include=["metadatas"])["metadatas"]
            coverage_indexes["vector"] = [{"file": m.get("file"), "function": m.get("name"), "line_start": m.get("line_start")} for m in metadata]
            if not metadata:
                raise ValueError("Vector index empty")
        except Exception as error:
            coverage_indexes["vector"] = None
            component_errors.append({"stage": "vector_index", "error_type": type(error).__name__})
        if engine.bm25_retriever is not None and engine.bm25_retriever.docs:
            coverage_indexes["bm25"] = [_document_record(doc) for doc in engine.bm25_retriever.docs]
        else:
            component_errors.append({"stage": "bm25_index", "error_type": "UnavailableIndex"})
        engine.vectorstore = _ObservedVectorStore(engine.vectorstore, traces, errors)
        if engine.bm25_retriever is not None:
            engine.bm25_retriever = _ObservedBM25(engine.bm25_retriever, traces, errors)
    except Exception as error:
        engine_error = type(error).__name__

    for item in dataset["queries"]:
        started = time.perf_counter()
        ranked_results = []
        sources = []
        query_errors = []
        vector_rankings = []
        bm25_rankings = []
        stages = {"rrf": [], "graph_enrichment": []}
        target_coverage = {
            name: (None if records is None else any(_matches_function(record, item["gold"]) for record in records))
            for name, records in coverage_indexes.items()
        }
        values = list(target_coverage.values())
        covered = True if any(value is True for value in values) else (False if all(value is False for value in values) else None)
        query_errors.extend(component_errors)
        if engine_error:
            query_errors.append({"stage": "engine_init", "error_type": engine_error})
        if engine is not None:
            trace_start = {stage: len(values) for stage, values in traces.items()}
            error_start = len(errors)
            def observe_stage(stage, values):
                if stage == "error":
                    query_errors.append(values)
                else:
                    stages[stage] = [_document_record(doc) for doc in values]
            try:
                docs, sources = engine._retrieve_and_fuse(item["query"], args.k, stage_observer=observe_stage)
                ranked_results = [_document_record(doc) for doc in docs]
            except Exception as error:
                query_errors.append({"stage": "retrieval", "error_type": type(error).__name__})
            vector_rankings = traces["vector"][trace_start["vector"]:]
            bm25_rankings = traces["bm25"][trace_start["bm25"]:]
            query_errors.extend(errors[error_start:])

        successful_components = sum(bool(x) for x in (vector_rankings, bm25_rankings))
        status = ("success" if not query_errors else "partial" if successful_components else "failed")

        rows.append({
            "query_id": item["query_id"],
            "task_id": item["task_id"],
            "language": item["language"],
            "query": item["query"],
            "gold": item["gold"],
            "index_coverage": {"target_present": covered, "by_component": target_coverage},
            "execution_status": status,
            "vector_rankings": vector_rankings,
            "bm25_rankings": bm25_rankings,
            "ranked_results": ranked_results,
            "stages": stages,
            "sources": sources,
            "metrics": query_metrics(ranked_results, item["gold"]) if status == "success" and covered is True else None,
            "latency_ms": round((time.perf_counter() - started) * 1000, 3),
            "errors": query_errors,
        })

    eligible = [row for row in rows if row["metrics"] is not None]
    languages = sorted({row["language"] for row in eligible})
    language_metrics = {language: aggregate_metrics(eligible, language) for language in languages}
    report_errors = component_errors + errors + ([{"stage": "engine_init", "error_type": engine_error}] if engine_error else [])
    statuses = [row["execution_status"] for row in rows]
    retrieval_status = "complete" if statuses and all(s == "success" for s in statuses) else "failed" if not statuses or all(s == "failed" for s in statuses) else "partial"
    return {
        "report_schema_version": "2.0.0",
        "benchmark": {
            "name": "cross_lingual_code_qa",
            "version": dataset["schema_version"],
            "kind": dataset["benchmark_kind"],
            "validated_research_benchmark": dataset["validated_research_benchmark"],
        },
        "code_revision": {
            "commit": _git_value(["git", "rev-parse", "HEAD"], ROOT),
            "tracked_worktree_dirty": bool(_git_value(["git", "status", "--porcelain"], ROOT)),
        },
        "configuration": {
            "query_embedding_model_id": "nomic-embed-text",
            "indexed_embedding_model_id": "unknown_not_recorded_in_existing_index_metadata",
            "chat_model_id": None,
            "query_mode": "raw_query_only",
            "query_expansion": False,
            "k": args.k,
            "answer_generation": False,
            "configuration_source": "explicit source defaults; .env and process environment not read",
        },
        "indexed_corpus": _corpus_identity(
            collection_dir,
            source_dir,
            dataset["corpus"].get("expected_source_revision"),
        ),
        "retrieval_status": retrieval_status,
        "execution_counts": {status: statuses.count(status) for status in ("success", "partial", "failed")},
        "coverage": {
            "query_count": len(rows),
            "covered": sum(row["index_coverage"]["target_present"] is True for row in rows),
            "unindexable": sum(row["index_coverage"]["target_present"] is False for row in rows),
            "unknown": sum(row["index_coverage"]["target_present"] is None for row in rows),
            "eligible_successful_queries": len(eligible),
        },
        "errors": report_errors,
        "metrics": {
            "overall_micro_average": aggregate_metrics(eligible),
            "language_macro_average": {
                metric: sum(language_metrics[language][metric] for language in languages) / len(languages)
                for metric in language_metrics[languages[0]]
                if metric != "query_count"
            } if languages else {},
            "by_language": language_metrics,
        },
        "queries": rows,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--report", type=Path, default=Path(__file__).with_name("reports") / "baseline_smoke.json")
    parser.add_argument("--persist-dir", type=Path, default=BACKEND / "chroma_db")
    parser.add_argument("--clone-dir", type=Path, default=BACKEND / "tmp" / "reposage")
    parser.add_argument("--repo-name", default="micrograd")
    parser.add_argument("--k", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    report = run(args)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "report": str(args.report),
        "retrieval_status": report["retrieval_status"],
        "query_count": len(report["queries"]),
        "metrics": report["metrics"],
        "errors": report["errors"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
