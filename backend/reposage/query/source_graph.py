"""Load and traverse versioned source graphs without altering indexed documents."""
from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass
from pathlib import Path

from ..ingestion.source_graph import SCHEMA
from .evidence import chunk_identity

MAX_GRAPH_CHUNKS = 6
MAX_GRAPH_CHARS = 8000


@dataclass(frozen=True)
class SymbolLookup:
    status: str
    node_id: str | None = None


_DOTTED = re.compile(r"(?<![A-Za-z0-9_./\\])([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+)(?![A-Za-z0-9_.])")
_QUOTED_SYMBOL = re.compile(r"`([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)`|(?<![A-Za-z0-9_./\\])([A-Za-z_][A-Za-z0-9_]*)\(\)")


def resolve_symbol(graph, question):
    """Resolve explicit identifiers exactly; abstain on conflicting matches."""
    tokens = set(_DOTTED.findall(question))
    tokens.update(a or b for a, b in _QUOTED_SYMBOL.findall(question))
    nodes = graph["nodes"]
    tokens.difference_update(Path(n["file"]).name for n in nodes)
    if not tokens:
        return SymbolLookup("not_applicable")
    # A full repository-relative path is an optional exact disambiguator.
    files = {n["file"] for n in nodes if re.search(r"(?<![A-Za-z0-9_./\\-])" + re.escape(n["file"]) + r"(?![A-Za-z0-9_./\\-])", question)}
    def matches(n, token):
        return n["qualified"] == token or ("." not in token and n["qualified"].rsplit(".", 1)[-1] == token)
    candidates = [n for n in nodes if any(matches(n, token) for token in tokens) and (not files or n["file"] in files)]
    ids = {n["id"] for n in candidates}
    if not ids:
        return SymbolLookup("not_found")
    # If another explicit symbol is unknown, do not silently choose the one
    # known token as the intent of a multi-symbol question.
    recognized = {token for token in tokens if any(matches(n, token) for n in candidates)}
    if len(ids) != 1 or not tokens.issubset(recognized):
        return SymbolLookup("ambiguous")
    return SymbolLookup("matched", next(iter(ids)))


def requested_direction(question):
    text = question.casefold()
    caller = bool(re.search(r"\b(?:who|which\s+\w+)\s+calls?\b|\bwhere\s+is\b.{0,100}\bcalled\b|\bcallers?\s+of\b|\bqui\s+appelle\b|\bquelles?\s+fonctions?\s+appellent\b|\bappelants?\s+de\b|谁调用|哪些函数调用|调用者", text))
    callee = bool(re.search(r"\b(?:what|who|which\s+functions?)\s+does\b.{0,100}\bcall\b|\bcallees?\s+of\b|\bqu['’]appelle\b|调用了哪些|调用哪些函数|被调用函数", text))
    return "callers" if caller and not callee else "callees" if callee and not caller else "both"


def load_graph(path, repo_name):
    with Path(path).open(encoding="utf-8") as f:
        data = json.load(f)
    if data.get("schema") != SCHEMA or data.get("repo") != repo_name:
        raise ValueError("Incompatible source graph")
    nodes = data.get("nodes")
    edges = data.get("edges")
    if not isinstance(nodes, list) or not isinstance(edges, list):
        raise ValueError("Invalid source graph")
    ids = {n.get("id") for n in nodes if isinstance(n, dict)}
    if len(ids) != len(nodes) or None in ids or any(e.get("source") not in ids or e.get("target") not in ids for e in edges):
        raise ValueError("Invalid graph identities")
    return data


def graph_context(info, evidence):
    """Attach source IDs where call-site chunks are actually in context."""
    ids = {e["id"]: e["citation_id"] for e in evidence}
    if info.get("lookup_status") in {"ambiguous", "not_found", "unavailable_chunk", "unavailable_budget", "unavailable"}:
        return "\n\nExact symbol grounding was inconclusive; do not assume a unique target without evidence."
    lines = []
    for edge in info.get("relationships", []):
        cite = ids.get(edge.get("caller_chunk_id"))
        edge["caller_citation_id"] = cite
        edge["callee_citation_id"] = ids.get(edge.get("callee_chunk_id"))
        location = f"{edge['caller_file']}:{edge['call_line']}"
        lines.append(f"Static reference: {edge['caller']} -> {edge['callee']} at {location}" + (f" [{cite}]" if cite else " (caller source not retrieved)"))
    for call in info.get("unresolved", []):
        cite = ids.get(call.get("caller_chunk_id"))
        call["caller_citation_id"] = cite
        lines.append(f"Unresolved call target: {call['expression']} in {call['file']}:{call['line']}" + (f" [{cite}]" if cite else " (caller source not retrieved)"))
    if not lines:
        return ""
    selected, used = [], 0
    for line in lines:
        if used + len(line) + 1 > 4000:
            continue
        selected.append(line)
        used += len(line) + 1
    info["context_entries_omitted"] = len(lines) - len(selected)
    if not selected:
        return ""
    return "\n\nGraph references are static, not proof of runtime dispatch. Unresolved targets are not confirmed relationships.\n" + "\n".join(selected)


def enrich(graph, seeds, indexed_docs, repo_name, question, max_chunks=MAX_GRAPH_CHUNKS, max_chars=MAX_GRAPH_CHARS, diagnostics=None, *, _seed_fallback=False):
    """Use exact seed identities and only indexed, bounded one-hop evidence."""
    if max_chunks <= 0 or max_chars <= 0:
        return []
    indexed = {chunk_identity(doc, repo_name): doc for doc in indexed_docs}
    node_map = {n["id"]: n for n in graph["nodes"]}
    exact_by_chunk = {}
    for node in graph["nodes"]:
        chunk_id = node.get("chunk_id")
        doc = indexed.get(chunk_id)
        if doc is None: continue
        meta = doc.metadata
        if meta.get("file", "").replace("\\", "/") != node["file"]: continue
        start, end = meta.get("line_start"), meta.get("line_end")
        if not isinstance(start, int) or not isinstance(end, int): continue
        if node["chunk_mapping"] == "exact" and (start, end, meta.get("name")) == (node["line_start"], node["line_end"], node["qualified"].rsplit(".", 1)[-1]):
            exact_by_chunk.setdefault(chunk_id, []).append(node["id"])
    adjacency = {node_id: {"callers": [], "callees": []} for node_id in node_map}
    for edge in graph["edges"]:
        adjacency[edge["source"]]["callees"].append(edge["target"])
        adjacency[edge["target"]]["callers"].append(edge["source"])
    direction = requested_direction(question)
    lookup = SymbolLookup("not_applicable") if _seed_fallback else resolve_symbol(graph, question)
    info = diagnostics if diagnostics is not None else {}
    info.update({"lookup_status": lookup.status, "direction": direction, "root": None, "relationships": [], "unresolved": []})
    def usable(node):
        doc = indexed.get(node.get("chunk_id"))
        if doc is None: return None
        m = doc.metadata
        if (m.get("file") or "").replace("\\", "/") != node["file"]: return None
        start, end = m.get("line_start"), m.get("line_end")
        if not isinstance(start, int) or not isinstance(end, int) or not (start <= node["line_start"] <= node["line_end"] <= end): return None
        if node.get("chunk_mapping") == "exact":
            if (start, end, m.get("name")) != (node["line_start"], node["line_end"], node["qualified"].rsplit(".", 1)[-1]): return None
        elif node.get("chunk_mapping") != "enclosing" or m.get("type") != "class":
            return None
        return doc

    root = node_map.get(lookup.node_id)
    if root:
        root_doc = usable(root)
        if root_doc is None:
            info["lookup_status"] = "unavailable_chunk"
            return enrich(graph, seeds, indexed_docs, repo_name, question, max_chunks, max_chars, _seed_fallback=True)
        info["root"] = {key: root[key] for key in ("id", "file", "qualified", "line_start", "line_end", "chunk_id", "chunk_mapping")}
        for edge in graph["edges"]:
            if ((direction in ("both", "callees") and edge["source"] == root["id"]) or
                    (direction in ("both", "callers") and edge["target"] == root["id"])):
                src, dst = node_map[edge["source"]], node_map[edge["target"]]
                info["relationships"].append({"caller": src["qualified"], "caller_file": src["file"], "call_line": edge["line"],
                    "expression": edge["expression"], "callee": dst["qualified"], "callee_file": dst["file"],
                    "callee_line_start": dst["line_start"], "callee_line_end": dst["line_end"], "callee_chunk_id": dst.get("chunk_id"),
                    "resolution": edge["resolution"], "status": "static_reference", "caller_chunk_id": src.get("chunk_id")})
        root_name = root["qualified"].rsplit(".", 1)[-1]
        for call in graph.get("unresolved", []):
            # Matching the spelling of an unresolved caller is only a hint,
            # never a claim that it actually targets the requested symbol.
            if ((direction in ("both", "callees") and call["source"] == root["id"]) or
                    (direction in ("both", "callers") and call["expression"].rsplit(".", 1)[-1] == root_name)):
                src = node_map[call["source"]]
                info["unresolved"].append({"caller": src["qualified"], "file": src["file"], "line": call["line"],
                    "expression": call["expression"], "status": "unresolved_target", "caller_chunk_id": src.get("chunk_id")})
        info["relationships_total"] = len(info["relationships"])
        info["unresolved_total"] = len(info["unresolved"])
        info["relationships"] = info["relationships"][:20]
        info["unresolved"] = info["unresolved"][:20]
    candidate_priorities = {}
    graph_seeds = [(0, root["id"])] if root else []
    for seed_rank, seed in enumerate(seeds) if root is None else ():
        matches = exact_by_chunk.get(chunk_identity(seed, repo_name), [])
        # A chunk mapping to multiple definitions is not an unambiguous seed.
        if len(matches) != 1: continue
        graph_seeds.append((seed_rank, matches[0]))
    for seed_rank, node_id in graph_seeds:
        for kind in (("callers", "callees") if direction == "both" else (direction,)):
            for neighbor_id in adjacency[node_id][kind]:
                candidate_priorities.setdefault(neighbor_id, seed_rank)
    seen = {chunk_identity(d, repo_name) for d in seeds}
    candidates = sorted((node_map[node_id] for node_id in candidate_priorities),
                        key=lambda n: (candidate_priorities[n["id"]], n.get("chunk_mapping") != "exact", n["file"], n["line_start"], n["qualified"]))
    if root is not None:
        candidates = [root] + [node for node in candidates if node["id"] != root["id"]]
    added, used = [], 0
    for node in candidates:
        identity = node.get("chunk_id")
        doc = usable(node)
        if doc is None or identity in seen: continue
        size = len(doc.page_content)
        if used + size > max_chars: continue
        extra = copy.copy(doc)
        extra.metadata = dict(doc.metadata)
        extra.metadata["type"] = (extra.metadata.get("type") or "") + " (Graph Context)"
        added.append(extra)
        seen.add(identity)
        used += size
        if len(added) >= max_chunks: break
    if root and root["chunk_id"] not in seen:
        info["lookup_status"] = "unavailable_budget"
        info.update({"root": None, "relationships": [], "unresolved": [], "relationships_total": 0, "unresolved_total": 0})
        return enrich(graph, seeds, indexed_docs, repo_name, question, max_chunks, max_chars, _seed_fallback=True)
    return added
