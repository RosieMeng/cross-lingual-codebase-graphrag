"""Conservative Python AST facts and deterministic Codex-authored question templates.

The facts are static syntactic references, not claims about dynamic dispatch.
This module neither opens an index nor executes repository code.
"""
from __future__ import annotations

import ast
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import json
import subprocess


@dataclass(frozen=True)
class Symbol:
    file: str
    qualified: str
    kind: str
    line_start: int
    line_end: int

    @property
    def name(self):
        return self.qualified.rsplit(".", 1)[-1]


@dataclass(frozen=True)
class Call:
    source: Symbol
    expression: str
    line: int
    target: Symbol | None
    resolution: str


def _module(file: str) -> str:
    parts = list(Path(file).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def analyze(root: Path):
    root = root.resolve()
    symbols, trees, nodes = [], {}, {}
    for path in sorted(root.rglob("*.py")):
        if any(p in {".git", ".venv", "venv", "__pycache__", "build", "dist"} for p in path.relative_to(root).parts):
            continue
        if path.is_symlink():
            continue
        file = path.relative_to(root).as_posix()
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=file)
        except (UnicodeError, SyntaxError):
            continue
        trees[file] = tree

        def collect(node, scope=()):
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    qualified = scope + (child.name,)
                    symbol = Symbol(file, ".".join(qualified), "class" if isinstance(child, ast.ClassDef) else "function", child.lineno, child.end_lineno)
                    symbols.append(symbol)
                    nodes[symbol] = child
                    collect(child, qualified)
                else:
                    collect(child, scope)
        collect(tree)
    keys = Counter((s.file, s.qualified) for s in symbols)
    by_key = {(s.file, s.qualified): s for s in symbols if keys[(s.file, s.qualified)] == 1}
    modules = {_module(file): file for file in trees}
    calls = []
    for source in symbols:
        if source.kind != "function":
            continue
        node = nodes[source]
        file_tree = trees[source.file]
        imports = {}
        # Only top-level explicit imports; wildcard, re-exports and conditional imports abstain.
        for item in file_tree.body:
            if isinstance(item, ast.ImportFrom) and item.level == 0 and item.module in modules:
                for alias in item.names:
                    if alias.name != "*":
                        imports[alias.asname or alias.name] = (modules[item.module], alias.name)
            elif isinstance(item, ast.Import):
                for alias in item.names:
                    if alias.name in modules and (alias.asname or "." not in alias.name):
                        imports[alias.asname or alias.name] = (modules[alias.name], None)
        # Any local binding can shadow a module/global name; conservatively abstain.
        bound = {arg.arg for arg in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)}
        if node.args.vararg: bound.add(node.args.vararg.arg)
        if node.args.kwarg: bound.add(node.args.kwarg.arg)
        for child in ast.walk(node):
            if isinstance(child, ast.Name) and isinstance(child.ctx, (ast.Store, ast.Del)):
                bound.add(child.id)
        def walk_body(current):
            for child in ast.iter_child_nodes(current):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
                    continue
                if isinstance(child, ast.Call):
                    yield child
                yield from walk_body(child)
        scope = source.qualified.split(".")
        for call in walk_body(node):
            expression = ast.unparse(call.func)
            target, why = None, "unresolved_dynamic_or_external"
            if isinstance(call.func, ast.Name):
                name = call.func.id
                # Nested definitions are lexical declarations in the current function.
                local = by_key.get((source.file, source.qualified + "." + name)) if name not in bound else None
                module = by_key.get((source.file, name)) if name not in bound else None
                imported = imports.get(name) if name not in bound else None
                if local:
                    target, why = local, "lexical_nested_definition"
                elif module:
                    target, why = module, "same_module_definition"
                elif imported and imported[1]:
                    target = by_key.get(imported)
                    if target: why = "explicit_from_import"
            elif isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Name):
                base = call.func.value.id
                # A self method is statically resolvable only within its declaring class.
                if base == "self" and len(scope) >= 2:
                    parent = by_key.get((source.file, ".".join(scope[:-1])))
                    if parent and parent.kind == "class":
                        target = by_key.get((source.file, parent.qualified + "." + call.func.attr))
                        if target: why = "same_class_self_method"
                imported = imports.get(base) if base not in bound else None
                if target is None and imported and imported[1] is None:
                    target = by_key.get((imported[0], call.func.attr))
                    if target: why = "explicit_module_import"
            calls.append(Call(source, expression, call.lineno, target, why))
    return symbols, calls


TEMPLATES = {
    "definition": {
        "en": "Where is the definition of `{symbol}` in `{file}`?",
        "fr": "Où se trouve la définition de `{symbol}` dans `{file}` ?",
        "zh": "`{file}` 中 `{symbol}` 的定义在哪里？",
        "mixed": "Where se trouve `{symbol}` 的定义 dans `{file}` ?",
    },
    "callee": {
        "en": "Find the call from `{caller}` in `{caller_file}` to `{callee}` in `{callee_file}` and show both definitions.",
        "fr": "Trouvez l'appel de `{caller}` dans `{caller_file}` vers `{callee}` dans `{callee_file}` et montrez les deux définitions.",
        "zh": "查找 `{caller_file}` 中 `{caller}` 对 `{callee_file}` 中 `{callee}` 的调用，并展示两个定义。",
        "mixed": "Find l'appel de `{caller}` dans `{caller_file}` vers `{callee}` dans `{callee_file}`，展示两个定义。",
    },
    "unresolved": {
        "en": "Where does `{caller}` in `{caller_file}` call `{expression}`? Identify the call site without assuming its target.",
        "fr": "Où `{caller}` dans `{caller_file}` appelle-t-il `{expression}` ? Indiquez l'appel sans supposer sa cible.",
        "zh": "`{caller_file}` 中 `{caller}` 在哪里调用 `{expression}`？请指出调用位置，不要假定目标。",
        "mixed": "Where does `{caller}` dans `{caller_file}` call `{expression}`？指出调用位置，不要假定目标。",
    },
}


def question(task, language):
    """Only accepted grammar is a reviewed-by-code template with all fact slots exact."""
    return TEMPLATES[task["kind"]][language].format(**task["slots"])


def validate_question(task, language, text):
    return language in TEMPLATES[task["kind"]] and text == question(task, language)


def make_dataset(root: Path):
    symbols, calls = analyze(root)
    counts = Counter(s.name for s in symbols)
    definitions = [s for s in symbols if s.kind == "function"]
    pairs = [c for c in calls if c.target is not None and c.source != c.target]
    local = [c for c in pairs if c.source.file == c.target.file]
    cross = [c for c in pairs if c.source.file != c.target.file]
    unresolved = [c for c in calls if c.target is None]
    tasks = []
    def add_definition(s, category):
        tasks.append({"id": f"definition-{len(tasks)}", "kind": "definition", "category": category,
                      "slots": {"symbol": s.qualified, "file": s.file}, "targets": [asdict(s)],
                      "annotation_method": "python_ast_definition_range", "label_status": "automatically_derived_unreviewed"})
    def add_call(c, category):
        tasks.append({"id": f"call-{len(tasks)}", "kind": "callee", "category": category,
                      "slots": {"caller": c.source.qualified, "caller_file": c.source.file, "callee": c.target.qualified, "callee_file": c.target.file},
                      "targets": [asdict(c.source), asdict(c.target)], "callsite": {"expression": c.expression, "line": c.line},
                      "annotation_method": c.resolution, "label_status": "automatically_derived_unreviewed"})
    if definitions: add_definition(next((s for s in definitions if counts[s.name] == 1), definitions[0]), "exact_definition")
    duplicate = next((s for s in definitions if len({other.file for other in definitions if other.name == s.name}) > 1), None)
    if duplicate: add_definition(duplicate, "same_named_definition")
    if local: add_call(next((c for c in local if c.target.kind == "function"), local[0]), "local_caller_callee")
    if cross: add_call(cross[0], "cross_file_caller_callee")
    if unresolved:
        c = unresolved[0]
        tasks.append({"id": f"unresolved-{len(tasks)}", "kind": "unresolved", "category": "unresolved_call",
                      "slots": {"caller": c.source.qualified, "caller_file": c.source.file, "expression": c.expression},
                      "targets": [asdict(c.source)], "callsite": {"expression": c.expression, "line": c.line},
                      "annotation_method": "python_ast_callsite_target_unresolved", "label_status": "automatically_derived_unreviewed"})
    # Include a second cross-file example whose definitions meet the current
    # chunker's minimum length; actual index coverage is still checked later.
    long_cross = next((c for c in cross if c != cross[0]
                       and c.source.line_end - c.source.line_start + 1 >= 5
                       and c.target.line_end - c.target.line_start + 1 >= 5), None) if cross else None
    if long_cross: add_call(long_cross, "cross_file_caller_callee_long_definitions")
    queries = [{"task_id": t["id"], "language": lang, "question": question(t, lang), "validation": "exact_template_and_ast_fact_slots"}
               for t in tasks for lang in TEMPLATES[t["kind"]]]
    for q in queries:
        assert validate_question(next(t for t in tasks if t["id"] == q["task_id"]), q["language"], q["question"])
    files = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(root.rglob("*.py")) if p.is_file() and not p.is_symlink()
             and not any(x in {".git", ".venv", "venv", "__pycache__", "build", "dist"} for x in p.relative_to(root).parts)}
    try:
        revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    return {"schema": "graph-ast-smoke-1", "annotation_status": "automatically_derived_unreviewed",
            "question_generation": "Codex-authored multilingual fixed templates; exact template and fact-slot validation",
            "source_revision": revision, "source_sha256": files, "tasks": tasks, "queries": queries,
            "facts": {"symbols": [asdict(s) for s in symbols], "calls": [{**asdict(c), "source": asdict(c.source), "target": asdict(c.target) if c.target else None} for c in calls]}}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    dataset = make_dataset(args.source)
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(dataset['tasks'])} tasks and {len(dataset['queries'])} language variants")
