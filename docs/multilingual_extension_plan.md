# Cross-Lingual Codebase GraphRAG Assistant: extension plan

## Goal and boundary

Extend RepoSage so developers can ask codebase questions in English, French, or Chinese while retrieving the same source-code evidence and receiving grounded, cited answers. This plan is intentionally design-only: it does not change application code, dependencies, models, or runtime settings.

The first release should preserve existing English behavior, preserve exact identifiers and paths, and make multilingual retrieval/evaluation observable. English, French, and Simplified Chinese are the initial user-query languages; code files themselves may contain any language supported by the current ingestion pipeline.

## Design principles

1. Keep the raw user query for display, audit, fallback retrieval, and evaluation.
2. Preserve identifiers, quoted strings, paths, URLs, flags, stack traces, and code spans byte-for-byte through language detection and rewriting.
3. Generate a small, traceable set of retrieval variants rather than replacing the query silently.
4. Treat multilingual embeddings as a configurable retrieval capability, not an assumption about every Ollama model.
5. Measure retrieval and citation quality separately from answer fluency.
6. Fall back safely to the original query when language detection, rewriting, or expansion fails.

## Proposed request-to-answer flow

```text
raw query + chat history
        |
        v
language detection + confidence
        |
        v
protect code tokens / identifiers / paths
        |
        v
identifier-preserving rewrite and code-aware expansion
        |
        v
raw + rewritten + expansion variants
        |
        +---- vector retrieval (configured embedding model)
        +---- identifier-aware BM25 retrieval
        |
        v
RRF + existing call-neighborhood enrichment
        |
        v
answer in requested/detected language + structured citations
        |
        v
benchmark and telemetry records
```

## English, French, and Chinese query support

Add a small query-language contract to API requests and UI state:

- `language: "auto" | "en" | "fr" | "zh"` selects automatic detection or a user override.
- `answer_language` defaults to the detected/requested language and can be explicit later if needed.
- The backend returns the effective language, detector confidence, and retrieval variants in a debug/evaluation-safe response field or server log, never by replacing the user-visible raw query.

The UI should begin with an unobtrusive auto-detect label and a manual override. Translate only stable interface text and example prompts; keep repository identifiers, source snippets, paths, and citations unchanged. Use UTF-8 end-to-end and test Chinese punctuation, full-width punctuation, and mixed-language questions.

## Language detection

Introduce an isolated `QueryLanguageDetector` service, called before expansion/retrieval. It should return a language code, confidence, and reason/source. Its input should exclude or mask protected code spans first, because identifiers and source fragments can distort natural-language detection.

Recommended staged approach:

1. Start with deterministic Unicode/script signals and short-text handling: Han characters strongly indicate Chinese; French lexical/diacritic signals can distinguish French from English when enough prose remains.
2. Add an optional lightweight language-identification provider behind the same interface only after its accuracy is evaluated on developer queries.
3. For low confidence, mixed English/French, code-only, or very short queries, label the request `und`/mixed and use the raw query plus bilingual expansion; do not guess a translation.

Record detector confusion by language in the benchmark. A detector failure must never prevent search.

## Identifier-preserving query rewriting

Add a `QueryNormalizer` before the current `QueryTransformer`. It should extract and replace protected spans with reversible placeholders, then restore them after rewriting. Protect at least:

- backtick code spans and fenced snippets;
- identifiers in `snake_case`, `camelCase`, `PascalCase`, dotted names, namespaces, and method calls;
- file paths, extensions, URLs, error messages, HTTP routes, CLI flags, version strings, and quoted literals;
- known symbols from indexed metadata when they appear as standalone query tokens.

For each supported language, create a concise code-search rewrite prompt or deterministic template that asks for retrieval terms in the query language plus an English technical bridge when useful. It must state that placeholders must be reproduced exactly. Restore them and reject an output if any protected token is missing, altered, duplicated unexpectedly, or if the rewrite is empty; the raw query remains the fallback.

Example: the French query `Où validateToken vérifie-t-elle l'expiration ?` should retain `validateToken` exactly while adding concepts such as `token expiration validation` for retrieval. A Chinese question containing `RepoIndexer.index_repo` must preserve that full identifier while adding terms such as `repository indexing` only as optional retrieval variants.

## Code-aware query expansion

Refactor the current optional expansion concept into a language-aware expansion interface, without changing its default-off behavior initially. It should produce a bounded, deduplicated set such as:

1. Raw user query.
2. Identifier-preserving normalized query.
3. A code-concept variant, emphasizing probable API names, control flow, data structures, or error terms.
4. An English bridge variant for French/Chinese queries when the embedding model benefits from it.
5. An exact-token variant containing protected identifiers and path fragments only when present.

Do not generate broad translations that discard the original intent. Cap variants (for example, four or five total), attach their provenance (`raw`, `normalized`, `bridge`, `identifier`), and deduplicate by normalized text. The existing `QueryTransformer` currently parses model output as numbered lines and has no language/identifier safeguards; it is the natural initial integration point.

## Optional multilingual embedding settings

Keep `OLLAMA_EMBED_MODEL` as the compatibility default and add a documented setting layer rather than hard-coding a new model. Candidate settings should include:

| Setting | Purpose |
| --- | --- |
| `EMBEDDING_MODE=default|multilingual` | Makes the choice explicit without changing the current default. |
| `OLLAMA_EMBED_MODEL` | Existing model name; select a model empirically verified to embed English/French/Chinese queries. |
| `QUERY_LANGUAGE` | Default `auto`; supports a deployment-wide default. |
| `ENABLE_CROSS_LINGUAL_REWRITE` | Keeps rollout controlled and reversible. |
| `CROSS_LINGUAL_VARIANT_LIMIT` | Bounds retrieval fan-out and latency. |

Before endorsing a particular Ollama embedding model, verify it is actually available from the configured registry, supports the three target query languages, uses a stable embedding dimension, and has acceptable latency/memory in the deployment. Changing an embedding model requires reindexing every Chroma collection; store embedding model name, dimension if available, index schema version, and build timestamp in collection metadata. Reject or clearly isolate a query when the loaded index metadata does not match the configured embedding model.

BM25 should remain enabled because exact symbols and paths are language-independent. Later add identifier splitting (`RepoIndexer` -> `repo`, `indexer`) and a Unicode-aware analyzer, then benchmark those choices instead of assuming standard tokenization works for Chinese.

## Cross-lingual code-QA benchmark

Create a versioned benchmark under a future `benchmarks/cross_lingual_code_qa/` directory. It should contain frozen, small repositories or the existing test fixture plus purpose-built examples; do not rely only on external GitHub URLs. Each item should include:

- `id`, repository revision/fixture, source language, and query language;
- an English/French/Chinese semantically equivalent question set;
- expected relevant file(s), symbol(s), and acceptable line spans;
- expected graph-neighbor evidence when applicable;
- answer requirements and one or more acceptable citations;
- tags for intent: definition, control flow, call path, bug location, configuration, architecture, or identifier lookup.

Split items by repository, not merely by question, to avoid memorizing code during prompt/model tuning. Include parallel questions, natural native-language questions, code-mixed questions, exact-identifier queries, and queries whose French/Chinese wording has no direct identifier overlap. Add difficult cases for ambiguous names, short queries, Unicode identifiers, and unsupported code-language chunking.

The benchmark runner should persist raw rankings, selected documents, RRF scores/provenance, model/index configuration, answer, citation list, timing, and errors. Separate deterministic retrieval evaluation from optional model-answer evaluation so a missing Ollama model does not prevent all benchmark work.

## Evaluation metrics

Calculate metrics per query language and report macro averages plus English-relative differences.

| Metric | Definition |
| --- | --- |
| File Hit@k | Fraction of questions where at least one expected file appears among the first `k` retrieved chunks/files. Report k=1, 3, 5, and 10. |
| Function Hit@k | Fraction where a chunk with the expected function/class identifier appears among the first `k` results. |
| MRR / nDCG@k | Ranking quality when benchmark items include graded or multiple relevant chunks. |
| Citation correctness | Fraction of returned citations that resolve to the claimed file/line and support the answer claim, measured with human or adjudicated labels. Report citation precision and answer citation coverage separately. |
| Citation recall | Fraction of benchmark-required evidence locations cited by the answer when a citation is expected. |
| Language degradation gap | `metric(en) - metric(language)` for the same parallel task set. A positive number means French/Chinese performs worse. Report intervals and sample counts. |
| Rewrite preservation rate | Fraction of protected identifiers/paths that survive normalization and expansion exactly. |
| Detector accuracy / abstention | Accuracy and low-confidence/`und` rate against labeled query language. |
| Latency | Detection, rewriting, vector retrieval, BM25 retrieval, fusion, graph expansion, and generation latency percentiles. |

Set initial acceptance targets only after collecting a baseline. A practical rollout gate is no statistically meaningful regression for English, no identifier loss, and an explicitly agreed maximum language degradation gap on the parallel benchmark.

## Files likely to be modified later

| File/path | Expected future change |
| --- | --- |
| `backend/main.py` | Add request/response language fields, configuration wiring, and potentially a benchmark/debug endpoint. |
| `backend/reposage/query/query_transformer.py` | Add protected-token handling, multilingual rewrite/expansion variants, provenance, and safe fallback. |
| `backend/reposage/query/rag_engine.py` | Retrieve with multilingual variants, retain variant provenance, support identifier-aware BM25, and return structured retrieval/citation data. |
| `backend/reposage/query/agent_engine.py` | Pass language intent to tools and make agent answers return structured citations. |
| `backend/reposage/ingestion/repo_indexer.py` | Persist index/embedding schema metadata and possibly richer symbol metadata for protection and graph resolution. |
| `backend/reposage/chunkers/python_chunker.py` | Optionally emit qualified names/import metadata for stronger graph disambiguation. |
| `backend/reposage/chunkers/js_chunker.py` | Optionally replace/augment regex extraction with parser-backed or qualified symbol metadata. |
| `backend/requirements.txt` | Only if evaluation shows a language-ID or parsing dependency is necessary; no dependency should be added before that decision. |
| `.env.example` and `README.md` | Document multilingual runtime switches, reindexing requirements, and verified model options. |
| `frontend/src/App.jsx` | Carry language selection/detection state through indexing/query flows. |
| `frontend/src/components/ChatInterface.jsx` | Language selector, request payload, answer-language display, and citation presentation. |
| `frontend/src/components/RepoDashboard.jsx` | Localized suggested prompts and status. |
| `backend/tests/` | Unit/integration tests for detection, preservation, expansion, retrieval, graph behavior, citations, and API contracts. |
| `frontend/src/**` test files | Component and interaction coverage for selection, mixed scripts, and citation rendering. |
| `benchmarks/cross_lingual_code_qa/**` | New benchmark corpus, annotations, runner, reports, and regression thresholds. |

## Recommended implementation order

1. Establish a baseline: freeze representative repositories/fixtures, add retrieval traces, and measure the current English/French/Chinese behavior using raw queries only.
2. Define request/response contracts and add UI/API language override plumbing while retaining `auto` and current English defaults.
3. Implement and test protected-token extraction/restoration. Make preservation failures fall back to the raw query.
4. Add deterministic language detection with confidence/abstention and tests for English, French, Chinese, mixed text, and code-only input.
5. Add language-aware query rewriting and bounded code-aware expansion behind feature flags; benchmark it before enabling by default.
6. Add index metadata/versioning and evaluate one optional multilingual embedding configuration. Reindex a test corpus and compare it against the default embedding model.
7. Improve BM25 tokenization/identifier splitting only when retrieval traces show it addresses an observed gap.
8. Strengthen graph identity with qualified symbols and validate call-neighborhood precision before expanding graph traversal depth.
9. Return structured citations from both direct RAG and the LangGraph agent; add citation correctness tests and review workflow.
10. Make the cross-lingual benchmark a CI-visible regression report with per-language metrics and degradation gates.

## Risks and unknowns

- **Embedding availability:** Ollama model availability, language coverage, embedding dimensions, hardware cost, and registry access are deployment-dependent. A model change invalidates existing Chroma indexes.
- **Network policy:** this cloud environment previously returned HTTP 403 for Ollama binary/model registry requests. Static inspection and non-model tests can proceed, but model evaluation and real indexing need permitted downloads or an existing Ollama deployment.
- **Translation drift:** unconstrained rewriting can change the question. Protected spans, raw-query fallback, and benchmark checks are required safeguards.
- **Chinese lexical search:** default BM25 tokenization may not segment Chinese usefully. Vector retrieval and later analyzer evaluation are especially important for Chinese queries.
- **Identifier ambiguity:** the current graph uses bare names and can connect unrelated functions/classes. Cross-lingual expansion can amplify this unless symbol identity becomes qualified.
- **Chunking coverage:** only Python has AST parsing; JS/TS uses regex and other supported languages use line windows. Retrieval quality can vary by source language independently of query language.
- **Citation grounding:** current citations are selected-chunk metadata, while inline citations are generated text. Neither guarantees that a cited line proves a claim.
- **Agent output:** the LangGraph path returns free-form final text without an authoritative source list, so it needs a citation contract before it is used for benchmarked QA.
- **Evaluation cost:** citation correctness requires source inspection and likely human/adjudicated labels; LLM-only grading should be treated as supplementary evidence.

## Current verification commands

The following remain the minimal non-model checks while the extension is still unimplemented:

```bash
cd /workspace/cross-lingual-codebase-graphrag
.venv/bin/python -m pytest backend/tests -q
.venv/bin/python -m pip check
cd frontend && npm run build
```

When Ollama is reachable and models are present, additionally start the backend, index a small fixture/local repository, issue equivalent English/French/Chinese questions, and record retrieval/citation outputs for the baseline. Do not treat a successful health endpoint alone as proof that the configured Ollama model can embed or answer queries.
