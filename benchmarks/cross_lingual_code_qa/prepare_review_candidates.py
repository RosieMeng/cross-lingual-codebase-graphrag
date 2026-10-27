"""Prepare unreviewed multilingual annotation candidates from pinned source files."""
import argparse
import ast
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
MICROGRAD = ROOT / "backend/tmp/reposage/micrograd"
CATEGORIES = ("en", "fr", "zh", "en-fr", "en-zh", "fr-zh", "en-fr-zh")


def git_revision(root, paths):
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    subprocess.run(["git", "diff", "--quiet", "HEAD", "--", *paths], cwd=root, check=True)
    return revision


def line_range(root, path, symbol):
    node = ast.parse((root / path).read_text(encoding="utf-8"))
    for name in symbol.split("."):
        node = next((child for child in node.body if getattr(child, "name", None) == name), None)
        if node is None: raise ValueError(f"Symbol missing: {symbol}")
    return node.lineno, node.end_lineno


def unreviewed():
    return {"status": "unverified", "reviewers": [], "decisions": []}


def make_query(task_id, category, fragments):
    text, spans, symbols = "", [], []
    for language, fragment in fragments:
        if text: text += " "
        start = len(text)
        text += fragment
        spans.append({"start": start, "end": len(text), "text": fragment, "label": language})
        if language == "code": symbols.append(fragment)
    return {"query_id": f"{task_id}-{category}", "task_id": task_id, "pair_id": task_id,
            "category": category, "identifier_presence": "present" if symbols else "absent",
            "text": text, "spans": spans, "exact_symbols": symbols, "review": unreviewed()}


def build():
    fixture_paths = ["backend/tests/fixtures/sample_repo/auth.py", "backend/tests/fixtures/sample_repo/utils.py"]
    micro_paths = ["micrograd/engine.py", "micrograd/nn.py"]
    repositories = [
        {"repo_id": "micrograd", "revision": git_revision(MICROGRAD, micro_paths), "source": "local micrograd checkout", "split": "unassigned"},
        {"repo_id": "reposage", "revision": git_revision(ROOT, fixture_paths), "source": "RepoSage repository test fixtures", "split": "unassigned"},
    ]
    # Manually drafted fragments remain unreviewed; language labels are candidates.
    specs = [
        ("backward", "micrograd", "Find the reverse traversal and gradient propagation", [("micrograd/engine.py", "Value.backward", "answer"), ("micrograd/engine.py", "Value", "context")], [
            [("en", "How does"), ("code", "Value.backward"), ("en", "traverse the graph and propagate gradients?")],
            [("fr", "Comment"), ("code", "Value.backward"), ("fr", "parcourt-il le graphe et propage-t-il les gradients ?")],
            [("code", "Value.backward"), ("zh", "如何遍历计算图并传播梯度？")],
            [("en", "How does"), ("code", "Value.backward"), ("fr", "parcourir le graphe et calculer les gradients ?")],
            [("en", "How does"), ("code", "Value.backward"), ("zh", "遍历计算图并传播梯度？")],
            [("fr", "Comment"), ("code", "Value.backward"), ("zh", "遍历计算图并传播梯度？")],
            [("en", "How does"), ("code", "Value.backward"), ("fr", "propager les gradients"), ("zh", "并遍历计算图？")]]),
        ("mlp_forward", "micrograd", "Find how the MLP sends an input through layers", [("micrograd/nn.py", "MLP.__call__", "answer"), ("micrograd/nn.py", "MLP", "context")], [
            [("en", "How does"), ("code", "MLP.__call__"), ("en", "pass an input through the layers?")],
            [("fr", "Comment"), ("code", "MLP.__call__"), ("fr", "fait-il passer une entrée dans les couches ?")],
            [("code", "MLP.__call__"), ("zh", "如何让输入依次经过各层？")],
            [("en", "How does"), ("code", "MLP.__call__"), ("fr", "faire passer une entrée dans les couches ?")],
            [("en", "How does"), ("code", "MLP.__call__"), ("zh", "让输入依次经过各层？")],
            [("fr", "Comment"), ("code", "MLP.__call__"), ("zh", "让输入依次经过各层？")],
            [("en", "How does"), ("code", "MLP.__call__"), ("fr", "traiter une entrée"), ("zh", "并依次经过各层？")]]),
        ("auth", "reposage", "Find credential lookup and password verification", [("backend/tests/fixtures/sample_repo/auth.py", "authenticate_user", "answer"), ("backend/tests/fixtures/sample_repo/auth.py", "get_user_from_db", "context")], [
            [("en", "How does"), ("code", "authenticate_user"), ("en", "look up a user and verify the password?")],
            [("fr", "Comment"), ("code", "authenticate_user"), ("fr", "cherche-t-il un utilisateur et vérifie-t-il son mot de passe ?")],
            [("code", "authenticate_user"), ("zh", "如何查找用户并验证密码？")],
            [("en", "How does"), ("code", "authenticate_user"), ("fr", "chercher un utilisateur et vérifier son mot de passe ?")],
            [("en", "How does"), ("code", "authenticate_user"), ("zh", "查找用户并验证密码？")],
            [("fr", "Comment"), ("code", "authenticate_user"), ("zh", "查找用户并验证密码？")],
            [("en", "How does"), ("code", "authenticate_user"), ("fr", "chercher l'utilisateur"), ("zh", "并验证密码？")]]),
        ("pagination", "reposage", "Find pagination bounds and returned metadata", [("backend/tests/fixtures/sample_repo/utils.py", "paginate", "answer")], [
            [("en", "Where are page boundaries calculated and page metadata returned?")],
            [("fr", "Où sont calculées les limites d'une page et renvoyées les métadonnées ?")],
            [("zh", "在哪里计算分页边界并返回分页信息？")],
            [("en", "Where are page boundaries calculated"), ("fr", "et les métadonnées renvoyées ?")],
            [("en", "Where are page boundaries calculated"), ("zh", "并返回分页信息？")],
            [("fr", "Où sont calculées les limites d'une page"), ("zh", "并返回分页信息？")],
            [("en", "Where are page boundaries calculated"), ("fr", "et les métadonnées"), ("zh", "在哪里返回？")]]),
        ("relu", "micrograd", "Find the rectified activation and its gradient", [("micrograd/engine.py", "Value.relu", "answer")], [
            [("en", "Where is the rectified activation computed and its gradient propagated?")],
            [("fr", "Où sont calculées l'activation rectifiée et sa dérivée ?")],
            [("zh", "在哪里计算修正线性激活及其梯度？")],
            [("en", "Where is the rectified activation computed"), ("fr", "et sa dérivée propagée ?")],
            [("en", "Where is the rectified activation computed"), ("zh", "并传播梯度？")],
            [("fr", "Où est calculée l'activation rectifiée"), ("zh", "并传播梯度？")],
            [("en", "Where is the rectified activation"), ("fr", "calculée"), ("zh", "并传播梯度？")]])
    ]
    tasks, queries = [], []
    for task_id, repo_id, intent, evidence, examples in specs:
        root = MICROGRAD if repo_id == "micrograd" else ROOT
        records = []
        for i, (path, symbol, role) in enumerate(evidence):
            start, end = line_range(root, path, symbol)
            records.append({"evidence_id": f"{task_id}-{i}", "file": path, "symbol": symbol,
                            "line_start": start, "line_end": end, "role": role,
                            "review": unreviewed(), "coverage": {"status": "not_checked", "index_manifest": None, "matching_chunks": []}})
        tasks.append({"task_id": task_id, "repo_id": repo_id, "intent": intent, "evidence": records, "review": unreviewed()})
        queries.extend(make_query(task_id, category, fragments) for category, fragments in zip(CATEGORIES, examples))
    ambiguous = make_query("mlp_forward", "en", [("en", "Where does call send the input through the layers?")])
    ambiguous.update(query_id="mlp_forward-ambiguous-en", pair_id="mlp_forward-ambiguous", identifier_presence="ambiguous")
    queries.append(ambiguous)
    return {"schema_version": "1.0.0", "dataset_id": "m3_1_review_candidates", "annotation_status": "unverified",
            "repositories": repositories, "indexes": [], "tasks": tasks, "queries": queries}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    with args.output.open("x", encoding="utf-8") as out:
        out.write(json.dumps(build(), ensure_ascii=False, indent=2) + "\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__": main()
