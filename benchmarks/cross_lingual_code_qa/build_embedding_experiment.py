"""Build an isolated Qwen vector index from already-indexed chunks, never the source index."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT))
from benchmarks.cross_lingual_code_qa.controlled_eval import fingerprint, revision
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
import httpx


def build(source, output, model, base_url):
    source, output = source.resolve(), output.resolve()
    original = (ROOT / "backend/chroma_db").resolve()
    if source.is_relative_to(original) or output.is_relative_to(original) or output == source:
        raise ValueError("Use separate source snapshot and output outside the original indexes")
    if output.exists():
        raise FileExistsError("Experimental output already exists")
    if output.name != "micrograd":
        raise ValueError("Output directory must be named micrograd")
    before = fingerprint(source)
    if not {"chroma.sqlite3", "bm25_docs.pkl", "graph.pkl"}.issubset(before):
        raise ValueError("Source snapshot missing required artifacts")
    start = time.perf_counter()
    with (source / "bm25_docs.pkl").open("rb") as f:
        docs = pickle.load(f)
    sanitized = []
    for doc in docs:
        metadata = dict(doc.metadata)
        if isinstance(metadata.get("calls"), list):
            metadata["calls"] = ",".join(metadata["calls"])
        sanitized.append(Document(page_content=doc.page_content, metadata=metadata))
    embedding = OllamaEmbeddings(model=model, base_url=base_url)
    dimension = len(embedding.embed_query("code retrieval dimension probe"))
    output.mkdir(parents=True)
    for name in ("bm25_docs.pkl", "graph.pkl", "metadata.json"):
        shutil.copy2(source / name, output / name)
    Chroma.from_documents(sanitized, embedding, persist_directory=str(output), collection_name="micrograd")
    tags = httpx.get(base_url + "/api/tags", timeout=10).json().get("models", [])
    model_info = next((m for m in tags if m.get("name") == model), {})
    running = httpx.get(base_url + "/api/ps", timeout=10).json().get("models", [])
    runtime = next((m for m in running if m.get("name") == model), {})
    result = {"status": "complete", "embedding_model": model, "model_digest": model_info.get("digest"),
              "model_size_bytes": model_info.get("size"), "embedding_dimension": dimension,
              "ollama_resident_bytes": runtime.get("size"), "ollama_vram_bytes": runtime.get("size_vram"),
              "repository_revision": revision(ROOT / "backend/tmp/reposage/micrograd"),
              "indexed_chunks": len(docs), "chunk_source": "existing BM25 documents; no rechunking",
              "collection_name": "micrograd", "query_instruction": None,
              "source_index_sha256": before, "source_unchanged": before == fingerprint(source),
              "indexing_latency_seconds": round(time.perf_counter() - start, 3)}
    (output / "experiment_manifest.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-snapshot", type=Path, required=True)
    parser.add_argument("--output-index", type=Path, required=True)
    parser.add_argument("--model", default="qwen3-embedding:0.6b")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    args = parser.parse_args()
    print(json.dumps(build(args.source_snapshot, args.output_index, args.model, args.ollama_url), indent=2))


if __name__ == "__main__": main()
