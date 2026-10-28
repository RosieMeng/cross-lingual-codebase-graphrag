"""Experimental source-only graph. Never loaded by production or written to graph.pkl."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmarks.graph_ablation.source_benchmark import analyze


def symbol_id(symbol):
    return f"{symbol.file}::{symbol.qualified}@{symbol.line_start}-{symbol.line_end}"


def build(root: Path):
    symbols, calls = analyze(root)
    nodes = [{"id": symbol_id(s), **asdict(s)} for s in symbols]
    edges, unresolved = [], []
    for c in calls:
        row = {"source": symbol_id(c.source), "expression": c.expression, "line": c.line, "resolution": c.resolution}
        if c.target is None:
            unresolved.append(row)
        else:
            edges.append({**row, "target": symbol_id(c.target)})
    return {"schema": "experimental-qualified-static-graph-v1", "enabled_in_production": False,
            "semantics": "conservative static syntactic references; runtime dispatch not verified",
            "nodes": nodes, "edges": edges, "unresolved_calls": unresolved}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    graph = build(args.source)
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(graph, f, ensure_ascii=False, indent=2)
    print(f"nodes={len(graph['nodes'])} edges={len(graph['edges'])} unresolved={len(graph['unresolved_calls'])}")
