# Qualified source graph

Newly indexed repositories with Python symbols receive `source_graph_v2.json` alongside the existing `graph.pkl`. The query engine prefers the v2 sidecar when present. Existing indexes without a sidecar, and newly indexed repositories without Python symbols, retain their legacy behavior; no query or migration step rewrites their data. In mixed-language repositories v2 does not traverse non-Python edges. An explicitly configured but missing or invalid v2 sidecar disables graph enrichment for that repository rather than silently using the legacy graph.

To experiment with an existing **trusted** index without changing it, create a new directory outside that index and its source checkout, then run from the repository root:

```powershell
New-Item -ItemType Directory validation/my_source_graphs
backend/.venv/Scripts/python.exe benchmarks/graph_ablation/build_sidecar.py --source backend/tmp/reposage/micrograd --index validation/m4a_nomic_snapshot/micrograd --repo micrograd --output validation/my_source_graphs/micrograd.json
$env:REPOSAGE_SOURCE_GRAPH_DIR=(Resolve-Path validation/my_source_graphs).Path
```

Set that process variable before starting the backend. The file name must match the repository name. Unset it to return to automatic sidecar discovery and legacy compatibility. The migration reads BM25 documents via pickle, so do not use untrusted indexes. It does not open Chroma, modify the index, or overwrite an existing output. The source checkout must correspond to the indexed source; each mapped node is checked against matching indexed source text and chunk identity. Edges without usable indexed chunks cannot contribute evidence.

V2 identifies symbols by relative path, lexical qualified name, and source range, within a repository-specific graph. Calls are extracted from method/function bodies without treating enclosing classes as callers. Explicit same-module, nested, imported and `self` method references are conservatively represented as static references; unresolved or dynamic calls are retained without guessed targets. Relative imports, inheritance, alias mutation, runtime dispatch and other dynamic Python behavior are not fully resolved. A static `self` edge is not proof of runtime dispatch. Non-Python files do not have v2 source edges.

Exact indexed chunk identities identify graph seeds. A short, otherwise unindexed symbol may map to its uniquely identified enclosing indexed class only if the exact source text is present; the evidence retains that class chunk's real file and range. No new chunk or line citation is invented. The formatted generation context now shows the complete available indexed line range as well as the source ID. The graph stage uses one hop, prioritizes earlier RRF seeds, prefers exact mappings, deduplicates by chunk identity, and caps additions at six documents and 8,000 source characters. This is a graph-addition budget; it does not truncate or bound the existing retrieval seed context. Recognizable caller/callee questions choose a direction; ambiguous questions use both. The language patterns are conservative heuristics, not a general query planner. RRF, vector/BM25 search and existing feature flags are unchanged.

For a clear caller/callee question that names a qualified symbol in the graph, traversal is restricted to seeds matching that symbol. If retrieval did not select it as a seed, enrichment abstains instead of following unrelated seeds; it does not introduce a new symbol-search ranking path. Bare, ambiguous words are not assumed to be identifiers.
