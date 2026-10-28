# Source-grounded graph ablation

This is an experimental diagnostic, not a production graph or a human-validated benchmark. It reads Python source without executing it, uses the existing query engine on a verified temporary index copy, and never writes `graph.pkl` or an existing index. Use only trusted index artifacts: the existing engine loads pickle files.

Generate a new benchmark from a local checkout (an existing output is never overwritten):

```powershell
backend/.venv/Scripts/python.exe benchmarks/graph_ablation/source_benchmark.py backend/tmp/reposage/micrograd validation/my_graph_benchmark.json
```

Each task stores exact Python AST symbol ranges, source hashes and revision, a static call expression and line when applicable, an annotation method, and unreviewed status. Four deterministic Codex-authored EN/FR/ZH/mixed templates supply paraphrases. Exact template and fact-slot validation rejects any altered generated text; it is not a general semantic validator and does not certify naturalness. Definitions and lexical references are independently derived from source, not supplied by the question author. Static references do not prove runtime dispatch, particularly for `self` or monkey-patched imports. Dynamic calls, external imports and ambiguous duplicate definitions are unresolved. The analyzer intentionally omits wildcard/relative imports, inherited dispatch, complex attributes and non-Python code. It does not claim the whole graph is correct.

Build an isolated candidate schema for inspection:

```powershell
backend/.venv/Scripts/python.exe benchmarks/graph_ablation/qualified_graph.py backend/tmp/reposage/micrograd validation/my_qualified_graph.json
```

The JSON graph is source-only and disabled in production. Nodes include file, lexical qualified name and source range. Resolved syntactic calls link those nodes; unresolved calls remain explicit. It does not reindex, provide a production migration, or silently convert old graph data.

Run the real graph ablation with the existing Nomic-query index snapshot. The driver checks source hashes and task/question data, copies and verifies the snapshot, and queries only that copy. With `--prompt-tokens`, it sends each fixed graph-off/on prompt to local `qwen3:1.7b` with a one-token generation budget to measure Ollama `prompt_eval_count`; those calls are measurement probes, not answer-quality evaluation. Omit the flag to skip them.

```powershell
backend/.venv/Scripts/python.exe benchmarks/graph_ablation/run_ablation.py --source-index validation/m4a_nomic_snapshot/micrograd --dataset validation/my_graph_benchmark.json --source backend/tmp/reposage/micrograd --output validation/my_graph_ablation.json --prompt-tokens
backend/.venv/Scripts/python.exe benchmarks/graph_ablation/summarize.py validation/my_graph_ablation.json validation/my_graph_summary.json
```

Graph-off uses the **same RRF seed documents** observed in the graph-on call; no second retrieval changes seeds. The report stores vector/BM25 candidates, graph seeds, node identities, all neighbors and the first-three cutoff, unresolvable nodes, inferred direction, added documents, evidence IDs, formatted-context inclusion, retrieval and graph-stage timing, and prompt-token counts. Graph relationship classification is a conservative AST cross-check: `statically_supported`, `conflicting_resolved_target`, `not_established`, `unresolved_source_identity`, or `no_indexed_chunk`. `not_established` is not automatically a false edge. Observations across language variants are repeated measurements, not independent tasks. Exact target coverage is checked in both index components; task-level recall excludes tasks with missing exact target chunks.

The default dataset is deliberately small. The current micrograd index excludes functions shorter than five lines, which leaves its selected cross-file caller unindexable. A future multi-repository run should sample more tasks, include independently captured source/index identities, and report static-reference ambiguity separately from demonstrated errors. Preserve the existing benchmark, reports and index snapshots; choose fresh paths for every run.

For the production qualified graph, first build a sidecar outside the source index as described in `backend/reposage/query/SOURCE_GRAPH.md`. Then run the same tasks on one verified copy, comparing the identical RRF seeds with and without the actual v2 additions:

```powershell
backend/.venv/Scripts/python.exe benchmarks/graph_ablation/run_v2_ablation.py --index validation/m4a_nomic_snapshot/micrograd --dataset validation/m6a_benchmark_v3.json --source backend/tmp/reposage/micrograd --graph-dir validation/my_source_graphs --output validation/my_v2_ablation.json --skip-tokens
backend/.venv/Scripts/python.exe benchmarks/graph_ablation/summarize_v2.py validation/my_v2_ablation.json validation/my_v2_summary.json
```

Omit `--skip-tokens` to measure actual prompt-token deltas with local Qwen one-token probes. `--task-id call-5` limits the run to one task; language variants remain correlated. Output paths must be new. The v2 result records exact seeds, edges, unresolved calls, added chunks, target coverage and formatted evidence. Static edge resolution does not validate runtime behavior.
