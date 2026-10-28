"""Record post-run tool/model identities without exposing machine paths."""
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

import httpx

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "validation/final_v1_manifest.json"


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    output = parser.parse_args().output
    if output.exists(): raise FileExistsError(output)
    tags = httpx.get("http://127.0.0.1:11434/api/tags", timeout=10).json()["models"]
    models = {m["name"]: m["digest"] for m in tags if m["name"] in {"qwen3:1.7b", "nomic-embed-text:latest"}}
    files = ["benchmarks/final_v1/benchmark.json", "benchmarks/final_v1/evaluate.py", "benchmarks/final_v1/summarize.py",
             "backend/reposage/query/rag_engine.py", "backend/reposage/query/source_graph.py", "backend/reposage/query/graph_answer.py",
             "validation/m6b_sidecars/micrograd.json"]
    data = {"note": "Captured after evaluation; evaluator CLI gained benchmark selection after the main run and sidecar selection after the supplement. No production code changed.",
            "code_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "working_tree_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()),
            "python_version": platform.python_version(), "platform": sys.platform, "model_digests_observed_after_run": models,
            "file_sha256_observed_after_run": {file: hashlib.sha256((ROOT / file).read_bytes()).hexdigest() for file in files}}
    with output.open("x", encoding="utf-8") as f: json.dump(data, f, ensure_ascii=False, indent=2)
    print("wrote post-run provenance manifest")


if __name__ == "__main__": main()
