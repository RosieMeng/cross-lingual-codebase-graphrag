"""Freeze a small source-checked QA set; never modifies source or indexes."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmarks.graph_ablation.source_benchmark import analyze

MICRO = ROOT / "backend/tmp/reposage/micrograd"
FIXTURE = ROOT / "backend/tests/fixtures/sample_repo"
OUTPUT = ROOT / "benchmarks/final_v1/benchmark.json"

# Questions are authored templates; source-derived labels do not validate
# semantic equivalence or naturalness of these translations.
SPECS = [
 ("m_backward", "micrograd", "definition", "present", [("micrograd/engine.py", "Value.backward")], ["Where is `Value.backward` defined?", "Où est définie `Value.backward` ?", "`Value.backward` 定义在哪里？", "Where se trouve `Value.backward` dans le code ?"]),
 ("m_graph", "micrograd", "behavior", "absent", [("micrograd/engine.py", "Value.backward")], ["Where does the code traverse the computation graph to propagate gradients?", "Où le code parcourt-il le graphe de calcul pour propager les gradients ?", "哪里通过遍历计算图传播梯度？", "Where does le code parcourir le graphe pour propager les gradients ?"]),
 ("m_callee", "micrograd", "callee", "present", [("micrograd/engine.py", "Value.backward"), ("micrograd/engine.py", "Value.backward.build_topo")], ["What does `Value.backward` call?", "Qu'appelle `Value.backward` ?", "`Value.backward`调用了哪些函数？", "What does `Value.backward` call dans le graphe ?"]),
 ("m_relu", "micrograd", "behavior", "absent", [("micrograd/engine.py", "Value.relu")], ["Where is the rectified linear activation and its gradient implemented?", "Où sont implémentés l'activation linéaire rectifiée et son gradient ?", "哪里实现了整流线性激活及其梯度？", "Where is l'activation linéaire rectifiée et son gradient implemented?"]),
 ("m_ambiguous", "micrograd", "ambiguous", "ambiguous", [("micrograd/nn.py", "Neuron.__call__"), ("micrograd/nn.py", "Layer.__call__"), ("micrograd/nn.py", "MLP.__call__")], ["Where is `__call__` defined in the neural network module?", "Où `__call__` est-il défini dans le module réseau neuronal ?", "神经网络模块中的`__call__`定义在哪里？", "Where is `__call__` défini dans le module réseau neuronal ?"]),
 ("m_noanswer", "micrograd", "no_answer", "present", [], ["Where is the Python method `Value.teleport` defined?", "Où est définie la méthode Python `Value.teleport` ?", "Python方法`Value.teleport`定义在哪里？", "Where est définie la méthode Python `Value.teleport` ?"]),
 ("s_auth", "sample_repo", "definition", "present", [("auth.py", "authenticate_user")], ["Where is `authenticate_user` defined?", "Où est définie `authenticate_user` ?", "`authenticate_user`定义在哪里？", "Where est définie `authenticate_user` ?"]),
 ("s_callee", "sample_repo", "callee", "present", [("auth.py", "authenticate_user"), ("auth.py", "get_user_from_db")], ["What does `authenticate_user` call?", "Qu'appelle `authenticate_user` ?", "`authenticate_user`调用了哪些函数？", "What does `authenticate_user` call dans le code ?"]),
 ("s_hash", "sample_repo", "behavior", "absent", [("auth.py", "create_password_hash")], ["Where is a password transformed into a SHA-256 hash?", "Où un mot de passe est-il transformé en empreinte SHA-256 ?", "哪里将密码转换为SHA-256哈希？", "Where is le mot de passe transformé en empreinte SHA-256 ?"]),
 ("s_page", "sample_repo", "behavior", "absent", [("utils.py", "paginate")], ["Where are pagination bounds and page counts calculated?", "Où sont calculées les limites et le nombre de pages ?", "哪里计算分页范围和页数？", "Where are les limites et le nombre de pages calculés ?"]),
 ("s_session", "sample_repo", "definition", "present", [("auth.py", "SessionManager.create_session")], ["Where is `SessionManager.create_session` defined?", "Où est définie `SessionManager.create_session` ?", "`SessionManager.create_session`定义在哪里？", "Where est définie `SessionManager.create_session` ?"]),
 ("s_noanswer", "sample_repo", "no_answer", "present", [], ["Where is the Python method `SessionManager.encrypt_session` defined?", "Où est définie la méthode Python `SessionManager.encrypt_session` ?", "Python方法`SessionManager.encrypt_session`定义在哪里？", "Where est définie la méthode Python `SessionManager.encrypt_session` ?"]),
 ("m_caller", "micrograd", "caller", "present", [("test/test_engine.py", "test_sanity_check"), ("test/test_engine.py", "test_more_ops")], ["Who calls `Value.backward`?", "Qui appelle `Value.backward` ?", "谁调用`Value.backward`？", "Who calls `Value.backward` dans les tests ?"]),
]


def revision(path):
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def main():
    if OUTPUT.exists(): raise FileExistsError(OUTPUT)
    corpora = {"micrograd": MICRO, "sample_repo": FIXTURE}
    facts = {name: analyze(path) for name, path in corpora.items()}
    tasks, queries = [], []
    for task_id, repo, kind, identifier, targets, texts in SPECS:
        symbols, calls = facts[repo]
        selected = []
        for file, qualified in targets:
            found = [s for s in symbols if s.file == file and s.qualified == qualified]
            if len(found) != 1: raise ValueError(f"Missing/ambiguous AST symbol: {task_id}")
            s = found[0]
            selected.append({"file": file, "qualified": qualified, "function": s.name, "line_start": s.line_start, "line_end": s.line_end})
        if kind == "callee":
            found = [c for c in calls if c.source.file == targets[0][0] and c.source.qualified == targets[0][1] and c.target and c.target.file == targets[1][0] and c.target.qualified == targets[1][1]]
            if not found: raise ValueError(f"Unverified AST call: {task_id}")
            relationship = {"caller": targets[0][1], "callee": targets[1][1], "call_line": found[0].line, "resolution": found[0].resolution}
        else: relationship = None
        if kind == "caller":
            observations = [c for c in calls if (c.source.file, c.source.qualified) in targets and c.expression.endswith(".backward") and c.target is None]
            if not observations: raise ValueError("No unresolved backward call sites")
            relationship = {"kind": "unresolved_candidate_calls", "sites": [{"file": c.source.file, "caller": c.source.qualified, "line": c.line, "expression": c.expression} for c in observations], "runtime_target_verified": False}
        if kind == "no_answer":
            forbidden = "Value.teleport" if repo == "micrograd" else "SessionManager.encrypt_session"
            if any(s.qualified == forbidden for s in symbols): raise ValueError("No-answer symbol exists")
        # These checks independently verify the narrow behavior descriptions.
        if task_id == "m_relu":
            text = (MICRO / "micrograd/engine.py").read_text(encoding="utf-8")
            tree = ast.parse(text)
            assert any(isinstance(n, ast.FunctionDef) and n.name == "relu" and "out.grad" in ast.unparse(n) and "out.data > 0" in ast.unparse(n) for n in ast.walk(tree))
        if task_id == "s_hash":
            tree = ast.parse((FIXTURE / "auth.py").read_text(encoding="utf-8"))
            assert any(isinstance(n, ast.FunctionDef) and n.name == "create_password_hash" and "hashlib.sha256" in ast.unparse(n) for n in ast.walk(tree))
        if task_id == "s_page":
            tree = ast.parse((FIXTURE / "utils.py").read_text(encoding="utf-8"))
            assert any(isinstance(n, ast.FunctionDef) and n.name == "paginate" and "pages" in ast.unparse(n) and "start" in ast.unparse(n) and "end" in ast.unparse(n) for n in ast.walk(tree))
        tasks.append({"id": task_id, "repo": repo, "type": kind, "identifier": identifier, "targets": selected, "relationship": relationship,
                      "fact_provenance": "Python AST definitions/call sites and explicit structural checks", "language_judgment": "unverified", "human_reviewed": False})
        for language, question in zip(("en", "fr", "zh", "mixed"), texts):
            queries.append({"id": f"{task_id}-{language}", "task_id": task_id, "language": language, "question": question})
    files = {}
    for repo, path in corpora.items():
        files[repo] = {f.relative_to(path).as_posix(): hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(path.rglob("*.py")) if "__pycache__" not in f.parts and f.is_file()}
    data = {"version": "final-v1-candidate-1", "frozen_before_evaluation": True, "independently_human_validated": False,
            "repositories": {"micrograd": {"revision": revision(MICRO), "kind": "upstream repository", "source_sha256": files["micrograd"]},
                "sample_repo": {"revision": revision(ROOT), "kind": "synthetic fixture within project repository, not an independent upstream repository", "source_sha256": files["sample_repo"]}},
            "tasks": tasks, "queries": queries}
    with OUTPUT.open("x", encoding="utf-8") as f: json.dump(data, f, ensure_ascii=False, indent=2)
    print("frozen", len(tasks), "tasks", len(queries), "variants")


if __name__ == "__main__": main()
