"""Versioned, conservative Python source graph bound to indexed chunks.

Edges describe statically resolvable syntax, not proven runtime dispatch.
Unresolved expressions remain explicit and are never traversed.
"""
from __future__ import annotations

import ast
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path

SCHEMA = "reposage-source-graph-v2"
SKIP = {"node_modules", ".git", "__pycache__", "dist", "build", ".venv", "venv", ".tox", ".mypy_cache",
        ".pytest_cache", "coverage", ".next", ".nuxt", "vendor", "target", "bin", "obj", ".eggs", "site-packages"}


@dataclass(frozen=True)
class Symbol:
    file: str
    qualified: str
    kind: str
    start: int
    end: int

    @property
    def id(self):
        return json.dumps([self.file, self.qualified, self.start, self.end], ensure_ascii=False, separators=(",", ":"))


def _module(path):
    parts = list(Path(path).with_suffix("").parts)
    if parts[-1] == "__init__": parts.pop()
    return ".".join(parts)


def build_source_graph(root, docs, repo_name):
    """Build a graph without modifying source, chunks or indexes."""
    from ..query.evidence import chunk_identity
    root = Path(root).resolve()
    symbols, nodes, trees, sources = [], {}, {}, {}
    for directory, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in SKIP and not (Path(directory) / d).is_symlink())
        for name in sorted(files):
            path = Path(directory) / name
            if path.suffix != ".py" or path.is_symlink() or path.stat().st_size > 100_000: continue
            file = path.relative_to(root).as_posix()
            try:
                source = path.read_text(encoding="utf-8")
                tree = ast.parse(source, filename=file)
            except (OSError, UnicodeError, SyntaxError):
                continue
            trees[file], sources[file] = tree, source.splitlines()
            def collect(parent, scope=()):
                for child in ast.iter_child_nodes(parent):
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
    unique = {(s.file, s.qualified): s for s in symbols if keys[(s.file, s.qualified)] == 1}
    module_counts = Counter(_module(f) for f in trees)
    modules = {_module(f): f for f in trees if module_counts[_module(f)] == 1}
    entries = []
    for symbol in symbols:
        text = "\n".join(sources[symbol.file][symbol.start - 1:symbol.end])
        candidates = []
        for doc in docs:
            m = doc.metadata
            if m.get("file", "").replace("\\", "/") != symbol.file or text not in doc.page_content:
                continue
            start, end = m.get("line_start"), m.get("line_end")
            if not isinstance(start, int) or not isinstance(end, int): continue
            exact = start == symbol.start and end == symbol.end and m.get("name") == symbol.qualified.rsplit(".", 1)[-1]
            enclosing = m.get("type") == "class" and start <= symbol.start and end >= symbol.end
            if exact or enclosing:
                candidates.append((0 if exact else 1, end - start, chunk_identity(doc, repo_name)))
        candidates.sort()
        best = [c for c in candidates if c[:2] == candidates[0][:2]] if candidates else []
        ids = {c[2] for c in best}
        mapping = best[0][0] if len(ids) == 1 else None
        entries.append({"id": symbol.id, "file": symbol.file, "qualified": symbol.qualified, "kind": symbol.kind,
                        "line_start": symbol.start, "line_end": symbol.end,
                        "chunk_id": next(iter(ids)) if mapping is not None else None,
                        "chunk_mapping": "exact" if mapping == 0 else "enclosing" if mapping == 1 else "unavailable"})
    edges, unresolved = [], []
    for symbol in symbols:
        if symbol.kind != "function": continue
        node = nodes[symbol]
        imports, module_writes, import_counts = {}, set(), Counter()
        for item in trees[symbol.file].body:
            if isinstance(item, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                for child in ast.walk(item):
                    if isinstance(child, ast.Name) and isinstance(child.ctx, (ast.Store, ast.Del)):
                        module_writes.add(child.id)
            if isinstance(item, ast.ImportFrom):
                for alias in item.names:
                    if alias.name != "*":
                        import_counts[alias.asname or alias.name] += 1
                        if item.level == 0 and item.module in modules:
                            imports[alias.asname or alias.name] = (modules[item.module], alias.name)
            elif isinstance(item, ast.Import):
                for alias in item.names:
                    import_counts[alias.asname or alias.name.split(".")[0]] += 1
                    if alias.name in modules and (alias.asname or "." not in alias.name):
                        imports[alias.asname or alias.name] = (modules[alias.name], None)
        imports = {name: value for name, value in imports.items() if name not in module_writes and import_counts[name] == 1 and (symbol.file, name) not in unique}
        bound = {a.arg for a in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)}
        if node.args.vararg: bound.add(node.args.vararg.arg)
        if node.args.kwarg: bound.add(node.args.kwarg.arg)
        rebound = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Name) and isinstance(child.ctx, (ast.Store, ast.Del)):
                bound.add(child.id)
                rebound.add(child.id)
            elif isinstance(child, (ast.Import, ast.ImportFrom)):
                bound.update(alias.asname or alias.name.split(".")[0] for alias in child.names if alias.name != "*")
        def calls(parent):
            for child in ast.iter_child_nodes(parent):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)): continue
                if isinstance(child, ast.Call): yield child
                yield from calls(child)
        scope = symbol.qualified.split(".")
        for call in calls(node):
            target, method = None, None
            if isinstance(call.func, ast.Name):
                name = call.func.id
                if name not in bound and name not in module_writes:
                    target = unique.get((symbol.file, symbol.qualified + "." + name))
                    method = "lexical_nested" if target else None
                    if not target:
                        target = unique.get((symbol.file, name)) if not import_counts[name] else None
                        method = "same_module" if target else None
                    if not target and name in imports and imports[name][1]:
                        target = unique.get(imports[name])
                        method = "explicit_import" if target else None
            elif isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Name):
                base = call.func.value.id
                if base == "self" and base not in rebound and len(scope) >= 2:
                    parent = unique.get((symbol.file, ".".join(scope[:-1])))
                    if parent and parent.kind == "class":
                        target = unique.get((symbol.file, parent.qualified + "." + call.func.attr))
                        method = "same_class_self_static" if target else None
                if not target and base not in bound and base in imports and imports[base][1] is None:
                    target = unique.get((imports[base][0], call.func.attr))
                    method = "explicit_module_import" if target else None
            row = {"source": symbol.id, "line": call.lineno, "expression": ast.unparse(call.func)}
            if target:
                edges.append({**row, "target": target.id, "resolution": method})
            else:
                unresolved.append({**row, "reason": "ambiguous_dynamic_or_external"})
    return {"schema": SCHEMA, "repo": repo_name, "nodes": entries, "edges": edges, "unresolved": unresolved,
            "source_hashes": {f: hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest() for f, lines in sources.items()}}


def write_graph(graph, output):
    with Path(output).open("x", encoding="utf-8") as f:
        json.dump(graph, f, ensure_ascii=False, indent=2)
