"""Freeze source-derived facts and authored questions for the pinned checkout."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmarks.graph_ablation.source_benchmark import analyze

REVISION = '672971d66a2ef9f85151e53283113f33d642dabd'
SPECS = [
 ('sign', 'definition', [('src/itsdangerous/signer.py', 'Signer.sign')], ["Where is `Signer.sign` defined?", "Où est défini `Signer.sign` ?", "`Signer.sign` 定义在哪里？", "Where est défini `Signer.sign` ?"]),
 ('encoding', 'behavior', [('src/itsdangerous/encoding.py', 'base64_encode')], ["Where is URL-safe base64 encoding implemented?", "Où est implémenté l'encodage base64 compatible avec les URL ?", "哪里实现了适用于 URL 的 base64 编码？", "Where is l'encodage base64 compatible URL implemented?"]),
 ('callee', 'callee', [('src/itsdangerous/signer.py', 'Signer.sign'), ('src/itsdangerous/signer.py', 'Signer.get_signature')], ["What does `Signer.sign` call?", "Qu'appelle `Signer.sign` ?", "`Signer.sign` 调用了什么？", "What does `Signer.sign` appeler ?"]),
 ('caller', 'caller', [('src/itsdangerous/signer.py', 'Signer.validate'), ('src/itsdangerous/signer.py', 'Signer.unsign')], ["Who calls `Signer.unsign`?", "Qui appelle `Signer.unsign` ?", "谁调用了 `Signer.unsign`？", "Who appelle `Signer.unsign` ?"]),
 ('cross', 'cross_module', [('src/itsdangerous/url_safe.py', 'URLSafeSerializerMixin.dump_payload'), ('src/itsdangerous/encoding.py', 'base64_encode')], ["Where does URL-safe serialization use the base64 encoder?", "Où la sérialisation compatible URL utilise-t-elle l'encodeur base64 ?", "URL 安全序列化在哪里使用 base64 编码器？", "Where does la sérialisation URL use the base64 encoder?"]),
 ('ambiguous', 'ambiguous', [('src/itsdangerous/signer.py', 'Signer.sign'), ('src/itsdangerous/timed.py', 'TimestampSigner.sign')], ["Where is `sign` defined?", "Où est défini `sign` ?", "`sign` 定义在哪里？", "Where est défini `sign` ?"]),
 ('absent', 'no_answer', [], ["Where is `Signer.teleport` defined?", "Où est défini `Signer.teleport` ?", "`Signer.teleport` 定义在哪里？", "Where est défini `Signer.teleport` ?"]),
]

def main():
    p = argparse.ArgumentParser(); p.add_argument('--source', type=Path, required=True); p.add_argument('--output', type=Path, required=True); a = p.parse_args()
    source = a.source.resolve()
    if subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip() != REVISION: raise ValueError('Revision mismatch')
    if subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain'], text=True).strip(): raise ValueError('Source checkout is dirty')
    symbols, calls = analyze(source)
    tasks, queries = [], []
    for task_id, kind, targets, questions in SPECS:
        found = []
        for file, name in targets:
            matches = [s for s in symbols if s.file == file and s.qualified == name]
            if len(matches) != 1: raise ValueError(f'Non-unique target: {file} {name}')
            s = matches[0]; found.append({'file': s.file, 'qualified': s.qualified, 'function': s.name, 'line_start': s.line_start, 'line_end': s.line_end})
        relation = None
        if kind in ('callee', 'caller'):
            edges = [c for c in calls if c.target and c.source.file == targets[0][0] and c.source.qualified == targets[0][1] and c.target.file == targets[1][0] and c.target.qualified == targets[1][1]]
            if not edges: raise ValueError('Missing static call')
            relation = {'expression': edges[0].expression, 'line': edges[0].line, 'resolution': edges[0].resolution, 'runtime_verified': False}
        if kind == 'cross_module':
            tree = ast.parse((source / targets[0][0]).read_text(encoding='utf-8'))
            assert any(isinstance(n, ast.ImportFrom) and n.level == 1 and n.module == 'encoding' and any(x.name == 'base64_encode' for x in n.names) for n in tree.body)
            calls_here = [c for c in calls if c.source.file == targets[0][0] and c.source.qualified == targets[0][1] and c.expression == 'base64_encode']
            assert calls_here and calls_here[0].target is None
            relation = {'expression': 'base64_encode', 'line': calls_here[0].line, 'resolution': 'relative import and call checked; graph analyzer unresolved', 'runtime_verified': False}
        if task_id == 'encoding':
            tree = ast.parse((source / targets[0][0]).read_text(encoding='utf-8'))
            fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'base64_encode')
            assert any(isinstance(n, ast.Attribute) and n.attr == 'urlsafe_b64encode' for n in ast.walk(fn))
        if kind == 'no_answer' and any(s.qualified == 'Signer.teleport' for s in symbols): raise ValueError('Target exists')
        tasks.append({'id': task_id, 'repo': 'itsdangerous', 'type': kind, 'targets': found, 'relationship': relation, 'annotation_method': 'Python AST; static facts only', 'language_judgment': 'unverified', 'human_reviewed': False})
        for lang, question in zip(('en', 'fr', 'zh', 'mixed'), questions): queries.append({'id': task_id+'-'+lang, 'task_id': task_id, 'language': lang, 'question': question})
    hashes = {f.relative_to(source).as_posix(): hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(source.rglob('*.py')) if '__pycache__' not in f.parts and '.git' not in f.parts and f.is_file()}
    data = {'version': 'itsdangerous-supplement-v1', 'repositories': {'itsdangerous': {'revision': REVISION, 'upstream': 'https://github.com/pallets/itsdangerous', 'license': 'BSD-3-Clause', 'source_sha256': hashes}}, 'tasks': tasks, 'queries': queries}
    with a.output.open('x', encoding='utf-8') as f: json.dump(data, f, ensure_ascii=False, indent=2)

if __name__ == '__main__': main()
