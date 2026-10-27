# Cross-Lingual Code QA Evaluation

`dataset_v1.json` is a versioned **toy smoke fixture**, not a validated research benchmark. It contains three aligned EN/FR/ZH query tasks over the local micrograd corpus. The gold judgments target repository-relative files, function names, and source start lines.

Run retrieval-only evaluation from the workspace root with the backend Python 3.11 environment:

```powershell
backend/.venv/Scripts/python.exe benchmarks/cross_lingual_code_qa/run_baseline.py
```

The runner reads the existing Chroma/BM25/graph index and source checkout, runs raw-query retrieval with expansion disabled, and writes `reports/baseline_smoke.json`. It does not generate answers, pull models, modify the index, or read `.env`. The report records query/vector/BM25 rankings, fused results, identifiers, per-query latency, metrics, Git revisions, and SHA-256 hashes of available persisted artifacts. Query embedding model is explicit; the model used to build the pre-existing index is unknown because current index metadata does not record it.

The report checks each original gold target against both indexes using file, function and (when present) start line. A target absent from both is marked unindexable for this index; if an index is unavailable and neither confirms the target, coverage is unknown. Coverage counts retain every query. Accuracy aggregates include only covered queries whose vector and BM25 retrieval completed successfully; partial and failed executions have null metrics and remain visible with errors and results. Consequently the accuracy figures describe the eligible subset and must be read alongside coverage, not compared directly with older all-query scores. Vector and BM25 rankings and the RRF and graph stages are recorded separately.

The benchmark intentionally has no acceptance threshold. Its tiny, hand-authored sample is useful for exercising report generation and retrieval plumbing only. A research benchmark requires broader repositories, independently reviewed judgments, repository-disjoint splits, repeat runs, and uncertainty analysis. The existing index does not record the source revision or embedding model used during indexing; matching the checkout revision does not prove index provenance.
