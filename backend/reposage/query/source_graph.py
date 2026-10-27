"""Load and traverse versioned source graphs without altering indexed documents."""
from __future__ import annotations

import copy
import json
import re
from pathlib import Path

from ..ingestion.source_graph import SCHEMA
from .evidence import chunk_identity

MAX_GRAPH_CHUNKS = 6
MAX_GRAPH_CHARS = 8000


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


def enrich(graph, seeds, indexed_docs, repo_name, question, max_chunks=MAX_GRAPH_CHUNKS, max_chars=MAX_GRAPH_CHARS):
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
    # A clear direction plus an explicit qualified symbol must not traverse
    # unrelated RRF seeds when the requested symbol was not retrieved.
    mentioned = set()
    if direction != "both":
        for node in graph["nodes"]:
            name = node["qualified"]
            if "." in name and re.search(r"(?<![\w.])" + re.escape(name) + r"(?![\w.])", question):
                mentioned.add(node["id"])
    candidate_priorities = {}
    for seed_rank, seed in enumerate(seeds):
        matches = exact_by_chunk.get(chunk_identity(seed, repo_name), [])
        # A chunk mapping to multiple definitions is not an unambiguous seed.
        if len(matches) != 1: continue
        if mentioned and matches[0] not in mentioned: continue
        for kind in (("callers", "callees") if direction == "both" else (direction,)):
            for node_id in adjacency[matches[0]][kind]:
                candidate_priorities.setdefault(node_id, seed_rank)
    seen = {chunk_identity(d, repo_name) for d in seeds}
    candidates = sorted((node_map[node_id] for node_id in candidate_priorities),
                        key=lambda n: (candidate_priorities[n["id"]], n.get("chunk_mapping") != "exact", n["file"], n["line_start"], n["qualified"]))
    added, used = [], 0
    for node in candidates:
        identity = node.get("chunk_id")
        doc = indexed.get(identity)
        if doc is None or identity in seen: continue
        m = doc.metadata
        if m.get("file", "").replace("\\", "/") != node["file"]: continue
        start, end = m.get("line_start"), m.get("line_end")
        if not isinstance(start, int) or not isinstance(end, int) or not (start <= node["line_start"] <= node["line_end"] <= end): continue
        if node.get("chunk_mapping") not in {"exact", "enclosing"}: continue
        if node["chunk_mapping"] == "exact" and (start, end, m.get("name")) != (node["line_start"], node["line_end"], node["qualified"].rsplit(".", 1)[-1]): continue
        if node["chunk_mapping"] == "enclosing" and m.get("type") != "class": continue
        size = len(doc.page_content)
        if used + size > max_chars: continue
        extra = copy.copy(doc)
        extra.metadata = dict(doc.metadata)
        extra.metadata["type"] = (extra.metadata.get("type") or "") + " (Graph Context)"
        added.append(extra)
        seen.add(identity)
        used += size
        if len(added) >= max_chunks: break
    return added
