"""Build a v2 graph beside, never inside, an existing trusted index."""
import argparse
import json
from pathlib import Path
import pickle
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT))
from benchmarks.cross_lingual_code_qa.controlled_eval import fingerprint
from reposage.ingestion.source_graph import build_source_graph, write_graph
from reposage.security import validate_repo_name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = validate_repo_name(args.repo)
    source, index, output = args.source.resolve(), args.index.resolve(), args.output.resolve()
    if output.exists() or any(output.is_relative_to(p) for p in (source, index, (ROOT / "backend/chroma_db").resolve())):
        raise ValueError("Output must be new and outside source and index directories")
    hashes = fingerprint(index)
    if "bm25_docs.pkl" not in hashes or "chroma.sqlite3" not in hashes:
        raise ValueError("Index lacks required artifacts")
    # Only use a trusted local index: pickle can execute code when loaded.
    with (index / "bm25_docs.pkl").open("rb") as f:
        docs = pickle.load(f)
    graph = build_source_graph(source, docs, repo)
    if fingerprint(index) != hashes:
        raise RuntimeError("Index changed during sidecar creation")
    graph["index_fingerprint"] = hashes
    write_graph(graph, output)
    print(json.dumps({"nodes": len(graph["nodes"]), "edges": len(graph["edges"]), "unresolved": len(graph["unresolved"]),
                      "mapped_exact": sum(n["chunk_mapping"] == "exact" for n in graph["nodes"]),
                      "mapped_enclosing": sum(n["chunk_mapping"] == "enclosing" for n in graph["nodes"]),
                      "index_unchanged": fingerprint(index) == hashes}))


if __name__ == "__main__": main()
