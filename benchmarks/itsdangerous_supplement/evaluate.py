"""Run the existing V1 worker on a verified copy of the separate index."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'backend'))
from benchmarks.cross_lingual_code_qa.controlled_eval import fingerprint, snapshot
from benchmarks.final_v1 import evaluate

def main():
    if len(sys.argv) > 1 and sys.argv[1] == '--worker':
        _, _, source, index, dataset, result = sys.argv
        evaluate.SOURCES['itsdangerous'] = Path(source).resolve()
        evaluate.ANSWER_TASKS = set()
        evaluate.worker('itsdangerous', index, dataset, result, '')
        return
    p = argparse.ArgumentParser(); p.add_argument('--source', type=Path, required=True); p.add_argument('--index', type=Path, required=True); p.add_argument('--output', type=Path, required=True); a = p.parse_args()
    dataset = ROOT / 'benchmarks/itsdangerous_supplement/dataset.json'
    index, output = a.index.resolve(), a.output.resolve()
    if output.exists() or output.is_relative_to(index): raise ValueError('Output must be new and outside the index')
    with tempfile.TemporaryDirectory(prefix='itsdangerous-eval-') as temp:
        copy = Path(temp) / 'itsdangerous'; result = Path(temp) / 'results.json'
        hashes = snapshot(index, copy)
        subprocess.run([sys.executable, __file__, '--worker', str(a.source.resolve()), str(copy), str(dataset), str(result)], cwd=ROOT, check=True)
        data = json.loads(result.read_text(encoding='utf-8'))
        data.update({'dataset_sha256': hashlib.sha256(dataset.read_bytes()).hexdigest(), 'source_index_hashes': hashes, 'source_index_unchanged': fingerprint(index) == hashes, 'embedding': 'nomic-embed-text', 'rewriter': 'qwen3:1.7b', 'k': 5, 'graph': 'identical raw RRF seeds; extra graph context'})
        with output.open('x', encoding='utf-8') as f: json.dump(data, f, ensure_ascii=False, indent=2)

if __name__ == '__main__': main()
