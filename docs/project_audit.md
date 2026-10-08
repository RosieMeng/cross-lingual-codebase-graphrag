# RepoSage project audit

## Scope and current state

RepoSage is a local-first codebase question-answering application. It accepts a GitHub URL (or a `local:///` path), converts selected source files into code chunks, stores vector embeddings in a per-repository Chroma collection, and answers questions through either a streaming RAG path or a LangGraph agent path. Ollama supplies both the chat model and embeddings.

This audit is based on static inspection and the existing unit tests. The local test suite passed with **36 tests**. Ollama was not available in this environment, so indexing, embedding, model generation, and end-to-end retrieval were not exercised.

## File-level map

| Path | Responsibility |
| --- | --- |
| `README.md` | Product overview, local and Docker Compose setup, and the single documented test command. |
| `.env.example` | Ollama URL/model settings, Chroma path, and optional GitHub token. |
| `docker-compose.yml` | Two-service deployment: FastAPI backend on host port 8001 and Vite frontend on 5173. |
| `backend/main.py` | FastAPI application, configuration, request models, REST/SSE endpoints, and component construction. |
| `backend/reposage/ingestion/repo_indexer.py` | Clone/local-path ingestion, file selection, chunk routing, Chroma persistence, BM25 document persistence, and knowledge-graph creation. |
| `backend/reposage/ingestion/docstring_gen.py` | LLM helper for contextual summaries; the current indexer deliberately bypasses it for rapid indexing. |
| `backend/reposage/ingestion/file_watcher.py` | Watchdog-based local-repository synchronization helper; it is not wired into `main.py`. |
| `backend/reposage/chunkers/python_chunker.py` | Python AST chunker and shared `CodeChunk` data structure. |
| `backend/reposage/chunkers/js_chunker.py` | Regex-based JavaScript/TypeScript/JSX/TSX chunker. |
| `backend/reposage/chunkers/generic_chunker.py` | Overlapping line-window fallback for supported non-Python/non-JS file types. |
| `backend/reposage/query/rag_engine.py` | Chroma similarity search, BM25, reciprocal-rank fusion, graph expansion, prompt construction, regular answers, and SSE token streaming. |
| `backend/reposage/query/query_transformer.py` | Optional LLM-based multi-query expansion, controlled by `ENABLE_QUERY_EXPANSION`. |
| `backend/reposage/query/agent_engine.py` | LangGraph tool-using agent over semantic search and full-file reads. |
| `backend/reposage/analysis/repo_summarizer.py` | Cached LLM-generated repository profile used by the dashboard. |
| `backend/tests/test_chunkers.py` | Unit tests for Python, JS, and generic chunkers. |
| `backend/tests/test_retrieval.py` | Unit tests for metadata, context/citation formatting, query-helper logic, and RRF ordering. |
| `frontend/src/App.jsx` | Repository indexing form, repository list, active-repository state, and routing between landing and chat views. |
| `frontend/src/components/ChatInterface.jsx` | Query submission, agent-mode toggle, SSE consumption, messages, and source-chip rendering. |
| `frontend/src/components/RepoDashboard.jsx` | Repository summary and English quick-query suggestions. |
| `frontend/src/components/LandingPage.jsx` | Static landing page. |
| `frontend/src/utils/markdown.js` | Lightweight markdown-to-HTML rendering for model answers and summaries. |

## Backend architecture

`backend/main.py` loads `.env`, creates singleton `RepoIndexer`, `RepoSummarizer`, and `QueryTransformer` instances, and exposes these operations:

- `POST /index` indexes a repository synchronously and reports in-memory status through `GET /index/status/{repo_name}`.
- `POST /query` uses `CodeAgentEngine`, the LangGraph agent path.
- `POST /query/stream` uses `CodeRAGEngine` and emits token, source, and completion events as SSE.
- `GET /repos`, `GET /repos/{name}/summary`, and `DELETE /repos/{name}` manage saved collections.
- `GET /health` reports basic configuration and process health; it does not test the Ollama connection.

The backend is configured through `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, `OLLAMA_EMBED_MODEL`, `CHROMA_PERSIST_DIR`, `CLONE_DIR`, and optionally `GITHUB_TOKEN`. The default model in `main.py` is `qwen2.5-coder:7b`; the sample environment file proposes `qwen3:4b`.

## Frontend architecture

The frontend is a React/Vite single-page interface styled with Tailwind and animated with Framer Motion. `App.jsx` posts a GitHub URL to `/index`, lists indexed repositories, and opens `ChatInterface` for the selected repository. `ChatInterface` sends both normal and streaming requests. Its agent toggle selects `/query` versus `/query/stream`; both send the same repository name, question, `k`, and local chat history.

For streaming responses, the component reads SSE records incrementally, appends token events to the active assistant message, and displays the source list sent at the end. The dashboard requests the summary endpoint and presents fixed English suggested questions. There is no user-language selection, detected-language state, query normalization, or multilingual UI copy today.

## Repository ingestion

`RepoIndexer.index_repo` accepts either a `local:///` directory or a remote Git URL. Remote repositories are shallow-cloned with GitPython into `CLONE_DIR/<repo-name>` and retained so the agent can later read original files. The walker prunes dependency/build directories, rejects binary/generated and lock-like suffixes, skips files larger than 100 KB, and indexes a declared set of code and text extensions.

Chunker routing is extension based:

- Python uses `PythonASTChunker`.
- JavaScript, TypeScript, JSX, and TSX use `JSChunker`.
- Other listed extensions use the generic 80-line, 15-line-overlap chunker.

Each chunk becomes a LangChain `Document` with repository, relative file path, type, identifier, line range, language, and extracted `calls`. The indexer batches 100 documents at a time into a Chroma collection. It also serializes the original documents to `bm25_docs.pkl`, simple collection metadata to `metadata.json`, and a NetworkX directed graph to `graph.pkl`. Re-indexing removes the prior collection directory before rebuilding it.

The `DocstringGenerator` exists but is currently not invoked by this pipeline; the code explicitly labels contextual retrieval as bypassed for fast indexing.

## AST-based and structural chunking

For Python, `PythonASTChunker` parses the source with the standard `ast` module. It emits functions, async functions, and classes as single chunks, preserving their source span and AST-derived identifier. A chunk includes an added searchable header containing the symbol name and docstring. Module docstrings become small module chunks. Functions shorter than five lines are skipped, very large chunks are truncated at 8,000 characters, and invalid Python falls back to one whole-file module chunk.

The same AST walk collects simple calls: bare calls become the callee name and attribute calls become the final attribute name. This is useful but not namespace-safe; identical names in different modules can collide.

JavaScript-family files do not use an AST parser. `JSChunker` recognizes common named functions, arrow functions, function expressions, and classes with regular expressions. It approximates block boundaries with brace matching, captures immediately preceding JSDoc, and heuristically finds calls. It falls back to line windows when no pattern matches. The generic chunker uses file extension as the language label and does not extract symbols or calls.

## Vector retrieval, BM25, and hybrid RRF

During ingestion, `OllamaEmbeddings` embeds each Chroma document using `OLLAMA_EMBED_MODEL`. During retrieval, `CodeRAGEngine` loads that collection and performs `similarity_search` for each query variant. This semantic route is intended to connect natural-language descriptions to code/docstring meaning.

The same saved documents are used to build an in-memory LangChain `BM25Retriever` at query-engine creation. BM25 token matching favors exact terms such as identifier names, paths, API names, and error strings. The existing analyzer is default BM25 tokenization; it has no language-specific segmentation or identifier-aware tokenization.

`QueryTransformer.expand_query` always keeps the original query and, when `ENABLE_QUERY_EXPANSION=true`, asks Ollama for up to four additional search phrasings. Vector and BM25 searches run concurrently across all query variants using a thread pool. Duplicate documents are removed by their content. The result lists are fused using Reciprocal Rank Fusion (RRF): each document receives `1/(60 + rank)` from each result list, then the engine returns the highest-scoring documents up to `k`.

## Call graph and GraphRAG behavior

The indexer makes a NetworkX `DiGraph` where a chunk name is a node and each extracted call creates a directed `caller -> callee` edge. Nodes store only the file path. At retrieval time, the RAG engine looks up each fused result by symbol name, adds up to three predecessor/successor documents from the persisted BM25 documents, and marks those chunks as `Graph Context` for the prompt.

This is a lightweight call-neighborhood expansion, not a fully resolved static call graph. It cannot disambiguate symbols with the same name, infer imports/receivers accurately, resolve dynamic dispatch, or supply graph edges for generic chunks. It nevertheless can surface adjacent code when names are unique and the chunker recognizes calls.

## LangGraph agent path

`CodeAgentEngine` wraps a `CodeRAGEngine` and a tool-bound `ChatOllama` model in a `StateGraph`. The graph cycles from `agent` to `action` while the latest response contains tool calls, then ends when the model returns a normal answer. It has a recursion limit of 10.

The available tools are:

- `semantic_search(query)`: runs the regular RAG query with `k=5` and returns its generated answer plus sources.
- `read_file(file_path)`: reads the retained cloned file. If unavailable, it reconstructs matching chunk content from the BM25 document set.

The system prompt asks the agent to search first and read files if it needs additional context. Agent answers are not streamed and do not return a separately structured citation list; this is an important later design consideration.

## Answers and citations

The direct RAG path condenses a follow-up question with the last six chat messages, retrieves/fuses chunks, formats each chunk as a `file:line (type: name)` header plus a language-fenced code block, and asks the Ollama chat model to answer only from that context. The prompt asks the model to include `file:line` citations in prose. Separately, the API returns a source list derived from each selected document's metadata. The SSE path sends those sources after the final token.

The application therefore has two citation channels: prompt-guided inline citations and system-generated source chips. Neither is currently verified against the generated answer or source text. The agent path can mention sources from its tool outputs, but its response schema does not enforce citations.

## Local operation

Prerequisites are Python 3.11+ (the environment was tested with Python 3.12), Node.js 18+, Git, and an Ollama service with one chat model and `nomic-embed-text` or a chosen embedding model. Create `.env` from the example and set an Ollama URL/model pair reachable from the backend.

Run the backend from `backend/` with a virtual environment and `python main.py`; run the frontend from `frontend/` with `npm install` and `npm run dev`. Docker Compose builds both services and maps backend port 8000 to host 8001. In containers, Ollama must be reachable at the configured host address. The repository README provides the full setup commands.

## Tests present and missing

Existing tests cover Python AST chunk extraction, JS regex chunk extraction, generic line-window behavior, syntax-error fallback, metadata shape, local context formatting, source-string formatting, query-helper expectations, and RRF ordering. The current suite has 36 passing tests and mocks/avoids Ollama.

Important gaps:

- No API tests for indexing, query, streaming, status, deletion, CORS, or error responses.
- No integration test that creates a Chroma collection, builds BM25/graph artifacts, and retrieves through the actual engine.
- No model-backed smoke test for embeddings, generation, query expansion, summaries, or LangGraph tool execution.
- No test for graph construction/traversal, symbol collisions, or cross-file call resolution.
- No frontend unit, component, integration, accessibility, or end-to-end tests.
- No multilingual queries, Unicode identifiers, segmentation, query-rewriting, or multilingual-retrieval tests.
- No benchmark fixtures, relevance judgments, citation-grounding checks, or regression thresholds.

## Minimal verification commands

Run these from the repository root after dependencies are installed:

```bash
.venv/bin/python -m pytest backend/tests -q
.venv/bin/python -m pip check
cd frontend && npm run build
```

For a running-backend smoke check, start the backend and request:

```bash
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8000/repos
```

These checks do not establish model availability or end-to-end question answering. If Ollama installation or model pulls receive HTTP 403 under the cloud network policy, retain the passing static/unit validation and retry those model-dependent checks only after network access is enabled.
