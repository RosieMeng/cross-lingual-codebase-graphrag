# Experimental code-aware retrieval and evaluation

`ENABLE_CODE_AWARE_QUERY_PLANNING=true` enables one vector and one BM25 request per query in Standard RAG, streaming RAG, and Agent semantic search. It takes precedence over Milestone 2 planning if both flags are true. It is disabled by default. The vector request uses a validated English rewrite where available, otherwise a safe contextual or original query; BM25 receives the exact original string and its existing whitespace tokenizer is unchanged. The configured chat model is used for rewriting, and no embedding model is changed. Query plans carry uncalibrated, half-open Unicode code-point spans and explicit request budgets. Han spans identify script, not a language. Unknown words are not presumed to be identifiers. A plan may contain sensitive text: do not log it by default.

The dataset `dataset_v2_candidates.json` is an engineering fixture. All judgments, translated and code-switched examples are generated, unverified candidates. Multiple exact evidence chunks are allowed; a match to any indexed evidence makes a query eligible, and every candidate's component-level coverage remains in the report. `independently_reviewed` judgments require reviewer provenance; they are reported separately from candidates. Existing v1 annotations are not changed. Exact file, function and line metadata are required.

Run the deterministic synthetic plumbing test using a new output path:

```powershell
backend/.venv/Scripts/python.exe benchmarks/cross_lingual_code_qa/controlled_eval.py --output validation/milestone3_synthetic_smoke.json
```

For a live smoke, the tool hashes the source index, copies it to an isolated temporary directory, verifies the copy, and opens only the copy in a child process. It checks the source hashes again after retrieval. Chroma may modify the snapshot; Windows needs the child process to exit before cleanup. Pick a new output path; the tool refuses to overwrite an existing report.

```powershell
backend/.venv/Scripts/python.exe benchmarks/cross_lingual_code_qa/controlled_eval.py --live-index-source backend/chroma_db/micrograd --output validation/milestone3_live_smoke.json
```

Raw, translation-only and retriever-specific arms each have one vector and one BM25 call at the same candidate and final RRF cutoff. Milestone 2 expanded planning is explicitly extra-budget when it produces a rewrite. LLM calls, retriever calls, fallback statuses, timings, channel rankings, RRF and graph stages, coverage, and errors are recorded per query. Accuracy is calculated only for successful, covered queries. Summaries include counts and category, language-membership and task groups, plus explicit pairs against each English query. Language-membership groups overlap for mixed queries. Answer generation is excluded. Synthetic mode uses a fake rewriter and lexical mock retrievers; its numbers have no model-quality meaning.

The existing index does not record its embedding model or source revision, so matching a checkout does not prove index provenance. These tiny candidate fixtures, unbalanced categories, repeated tasks, broad acceptable evidence, model warm-up and sequential run order preclude research claims. Before drawing conclusions, use a larger independently reviewed, repository-disjoint dataset, validated stable index manifests, fixed call budgets, paired statistical analysis and repeated randomized runs. Changing BM25 tokenization, graph construction or existing gold labels is a separate experiment.
