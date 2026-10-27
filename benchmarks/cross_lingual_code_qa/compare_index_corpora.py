"""Compare content/metadata identities from isolated Chroma snapshots."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmarks.cross_lingual_code_qa.controlled_eval import snapshot, fingerprint


def worker(index_path, output):
    import chromadb
    collection = chromadb.PersistentClient(path=index_path).get_collection("micrograd")
    data = collection.get(include=["documents", "metadatas"])
    signatures = []
    for content, metadata in zip(data["documents"], data["metadatas"]):
        identity = {"content": content, **{k: metadata.get(k) for k in ("file", "name", "line_start", "line_end", "type")}}
        signatures.append(hashlib.sha256(json.dumps(identity, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest())
    Path(output).write_text(json.dumps(signatures), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError("Output already exists")
    results = []
    with tempfile.TemporaryDirectory(prefix="reposage-compare-") as temp:
        for i, source in enumerate((args.first, args.second)):
            source = source.resolve()
            copy = Path(temp) / str(i) / "micrograd"
            before = snapshot(source, copy)
            out = Path(temp) / f"{i}.json"
            command = "import sys; from benchmarks.cross_lingual_code_qa.compare_index_corpora import worker; worker(*sys.argv[1:])"
            subprocess.run([sys.executable, "-c", command, str(copy), str(out)], cwd=ROOT, check=True)
            signatures = json.loads(out.read_text(encoding="utf-8"))
            results.append({"count": len(signatures), "signatures": signatures, "source_unchanged": fingerprint(source) == before})
    result = {"identical_content_metadata_multisets": Counter(results[0]["signatures"]) == Counter(results[1]["signatures"]),
              "indexes": results, "note": "content, file, symbol, start/end lines and type; vectors intentionally differ"}
    with args.output.open("x", encoding="utf-8") as target:
        json.dump(result, target, indent=2)
    print("counts", [r["count"] for r in results], "identical", result["identical_content_metadata_multisets"], "sources_unchanged", [r["source_unchanged"] for r in results])


if __name__ == "__main__": main()
