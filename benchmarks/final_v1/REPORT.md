# V1 multilingual Code GraphRAG evaluation

This is an **automated, unreviewed engineering evaluation**, not independently human-validated research. It provides concrete examples of retrieval and graph behavior; it does not establish general performance or factual reliability.

## Dataset and provenance

The main frozen set contains 13 distinct task records and four authored EN/FR/ZH/code-switched variants per task (52 queries). Several tasks share `Value.backward` or other source code, so they are not statistically independent. It covers definitions, identifier-free behavior, local caller/callee evidence, ambiguity and absent Python definitions. Gold paths, qualified symbols, line ranges and static call sites are derived from Python ASTs; targeted behavior descriptions have additional structural checks. The wording, translations, relevance policy and answer judgments remain unverified. Unresolved `.backward` calls are candidate call-site evidence, not verified runtime callers. No-answer labels mean the named definition was absent from the scanned Python ASTs, not proof of absence from all dynamic behavior.

The corpora are upstream micrograd at `7bc720e951fe422b8f8814aa5aa1b64121d26b4c` and the project's **synthetic `sample_repo` fixture**, not an independently sourced second repository, at project revision `6540becbe821ab4b8a550f2e6d3e58a200598852`. The frozen dataset records source file hashes. The fixture's placeholder database lookup is not a working database. The benchmark was frozen before the main evaluation; its SHA-256 is `c47b4791c4c237023ea1981f156389a3186595594fe56b0f9a47f66be2f868a3`.

The initial frozen set omitted a cross-file task. A separate supplemental task, based on the previously source-checked Milestone 6A `test_sanity_check → Value` case, was frozen **after** observing the main results and evaluated separately. It is not folded into the main scores. The main run is based on nine eligible tasks; the supplement adds one separate task, not four independent tasks.

Both source indexes were copied and hash-verified before live queries; both source snapshots remained unchanged. Index identity hashes are in the machine-readable results. The original index does not record its indexing embedding model or source revision, so exact compatibility with the known Nomic query model cannot be independently established. Query embeddings used `nomic-embed-text:latest`; rewriting and answer generation used `qwen3:1.7b`. Model digests and implementation hashes were recorded **after** the run in `validation/final_v1_manifest_final.json`; the evaluator CLI gained benchmark and sidecar selection after the original runs, without production changes. Chroma telemetry emitted errors but no retrieval or model execution errors were reported.

## Retrieval: fixed search budgets

Every fixed-budget arm used one vector and one BM25 search with candidate limit five and RRF cutoff five. Dense and BM25 columns score their individual candidate lists; translation uses a validated English rewrite for both searches when available, otherwise the original. It added 33 LLM rewrite calls across 52 variants: nine accepted rewrites, 24 rejected as not English, six language-detection abstentions, and 13 English originals. Timing and model warm-up were not controlled. These are not equal-cost arms.

Exact evidence coverage was checked in **both** vector and BM25 indexes before conditional quality metrics. Nine of eleven answerable tasks were fully covered (36 variants). The two no-answer tasks do not receive retrieval-recall scores. Three short `__call__` methods and the short `get_user_from_db` target lacked exact standalone chunks, so the ambiguous-method and one callee task are excluded from conditional scores rather than counted as retrieval failures. Graph context may contain enclosing class chunks, but they do not count as exact method chunks.

The numbers below are task-macro MRR@5: average language variants within each eligible task, then average tasks. Multiple acceptable targets count as a hit when the first target occurs; multi-target completeness is measured separately.

| Group | Eligible tasks | Dense | BM25 | Raw hybrid RRF | Translation hybrid |
|---|---:|---:|---:|---:|---:|
| Overall | 9 | 0.587 | 0.275 | 0.516 | 0.495 |
| EN | 9 | 0.596 | 0.481 | 0.578 | 0.578 |
| FR | 9 | 0.620 | 0.259 | 0.581 | 0.526 |
| ZH | 9 | 0.500 | 0.167 | 0.300 | 0.300 |
| Mixed | 9 | 0.630 | 0.194 | 0.606 | 0.578 |

By task type, raw hybrid MRR@5 was 0.617 for three definition tasks, 0.515 for four behavior tasks, 0.300 for one eligible callee task, and 0.438 for one caller-observation task. Ambiguous and no-answer tasks have no eligible retrieval denominator. Results by individual task, corpus and language are in `validation/final_v1_summary_v4.json`. The fixture is easier in this smoke and must not be treated as an independent repository sample.

## Graph contribution and failures

Graph arms use the **identical raw RRF seeds**. The qualified arm disables exact symbol grounding only in the evaluation process; the grounded arm uses current v2 behavior. Graph evidence is appended beyond the five seeds, so compare complete-context coverage, not these values as equal-budget MRR@5.

| Complete context, task macro across nine eligible tasks | Off | Qualified graph | Qualified graph + exact grounding |
|---|---:|---:|---:|
| All required exact chunks present | 0.694 | 0.806 | 0.917 |
| Fraction of required chunks present | 0.736 | 0.819 | 0.931 |

Qualified and grounded contexts added a mean 1.12 and 0.98 documents respectively; mean formatted source context was 4,437 characters off, 5,782 qualified, and 5,027 grounded. These are extra evidence budgets. For the local `backward → build_topo` task, complete coverage across four variants was 0/4 off, 2/4 qualified and 4/4 grounded. For the Chinese identifier-free graph-traversal question, the exact `backward` chunk was absent from all three contexts; its rewrite was rejected. This is a retrieval failure, not an execution failure.

The separate four-variant cross-file supplement had both exact chunks indexed: complete coverage was 1/4 off, 4/4 qualified and 1/4 grounded. The question names both symbols, so current exact grounding abstained as ambiguous and added no graph evidence. The qualified graph helped on this case, while the grounding guard reduced its contribution. This is one post-main supplemental task and must not be presented as a general comparison.

## Answer comparison

Answer generation was run on six selected tasks (24 variants), not the whole benchmark. The comparable subset contains three caller/callee tasks and 12 paired variants with **identical retrieved evidence** for model and evidence-first formatting. `qwen3:1.7b` used 256 output tokens; the evidence-first answer makes no answer-model call. The other three tasks—two no-answer tasks and one behavior task—have model answers only.

| Diagnostic | Model, paired 12 | Evidence-first, paired 12 |
|---|---:|---:|
| At least one valid source ID, no invalid ID | 11/12 | 12/12 |
| No recognized source ID | 1/12 | 0/12 |
| Marked truncated | 2/12 | 0/12 |
| Detected answer-language matches, heuristic | 3/9 | 9/9 |

The language heuristic scores EN/FR/ZH only; mixed questions without an explicit target language are excluded. It is not calibrated or human verified. A valid ID is not factual support or complete claim coverage. The model's 24 responses included eight no-answer variants; an incomplete multilingual keyword heuristic detected abstention wording in 8/8, but this does not establish the rest of each answer was correct. The formatter is not applicable to those no-answer requests.

The paired set includes the four `authenticate_user` callee variants whose `get_user_from_db` target is not separately indexed. The formatter says some call sites are omitted rather than asserting a fully source-backed target; its citation and language-format passes are not evidence that it answered that task completely.

The narrow AST-backed checks found **14/14 emitted, exact caller call-site observations** in evidence-first answers matched source expression, file and line; their runtime target remains unresolved. The four formatted callee variants mentioned the verified `build_topo` call at line 65. The model emitted no caller observations in the exact expression-plus-file-line format checked, so it has no denominator for that check. An unseen backticked dotted-symbol heuristic flagged 0/12 paired model answers and 0/12 formatted answers; it is not a complete hallucination detector and does not validate all symbols or claims.

Concrete unverified answer failures remain: the model's English callee answer described `backward` calling the graph node's own `backward` method, rather than limiting the answer to the shown `build_topo` and unresolved `_backward` call sites. In a no-answer mixed-language response it said both `Value` and `Module` are defined in `micrograd/nn.py`; AST inspection finds `Module` there and `Value` defined in `micrograd/engine.py`. A French no-answer response listed `parameters` among `Value` methods even though its class AST has no such method. These examples show why the zero unseen-dotted-token heuristic and valid citation IDs do not prove factual correctness.

The main 24-answer sample omitted the ambiguous-definition task. A separate follow-up (`validation/final_v1_ambiguous_answers.json`) ran the four already-frozen variants on their identical saved evidence; the evidence-first caller/callee formatter is not applicable. All four model outputs selected one class without acknowledging the other same-named methods, despite the graph reporting ambiguity. All had recognized citations and stopped normally. The French output also identified the `Neuron` class start line as the method location. The three exact method chunks were not indexed, although enclosing class chunks were available. This is a separately reported follow-up, not part of the paired answer counts.

## Reproduction

The commands below require the exact source checkouts, trusted local index snapshots, Ollama models and sidecar used in the run. The evaluator copies snapshots into a temporary directory before opening Chroma and refuses to overwrite its output. Do not use untrusted pickle indexes. `benchmark.json` and the supplement are already frozen; their freeze scripts fail if the output exists. Restore micrograd at the recorded revision under `backend/tmp/reposage/micrograd`, and restore the project fixture at the recorded source hashes. Install the existing backend dependencies and load the recorded Ollama models. A newly rebuilt index is a **new experiment**, not proof of reproducing this index with its unknown embedding provenance.

```powershell
backend/.venv/Scripts/python.exe benchmarks/final_v1/evaluate.py --micro-index validation/m4a_nomic_snapshot/micrograd --sample-index validation/m4b_live_index/sample_repo --micro-graph validation/m6b_sidecars/micrograd.json --output validation/repro_final_v1.json
backend/.venv/Scripts/python.exe benchmarks/final_v1/summarize.py validation/repro_final_v1.json validation/repro_final_v1_summary.json
backend/.venv/Scripts/python.exe benchmarks/final_v1/evaluate.py --micro-index validation/m4a_nomic_snapshot/micrograd --sample-index validation/m4b_live_index/sample_repo --micro-graph validation/m6b_sidecars/micrograd.json --benchmark benchmarks/final_v1/crossfile_supplement.json --output validation/repro_crossfile.json
backend/.venv/Scripts/python.exe benchmarks/final_v1/summarize.py validation/repro_crossfile.json validation/repro_crossfile_summary.json --benchmark benchmarks/final_v1/crossfile_supplement.json
```

The cross-file command was repeated on a fresh snapshot: raw RRF, graph lists and rewrite plans matched in that repeat. This does not establish deterministic model outputs across runs or hardware. To repeat the ambiguous-answer follow-up, use `backend/.venv/Scripts/python.exe benchmarks/final_v1/ambiguous_answers.py` after making `validation/final_v1_results.json` available; its output must not already exist. `SOURCE_GRAPH.md` describes how to build an external sidecar from a corresponding trusted checkout and index. Answers, full traces and index fingerprints are in `validation/final_v1_results.json` and `validation/final_v1_crossfile_results.json`; the summaries and post-run manifest are separate JSON files.

## Commit artifact audit

Proposed evaluation allowlist, after normal review: `benchmarks/final_v1/benchmark.json`, `crossfile_supplement.json`, `freeze_benchmark.py`, `freeze_crossfile.py`, `evaluate.py`, `summarize.py`, `ambiguous_answers.py`, `write_manifest.py`, this report, `backend/tests/test_final_v1_evaluation.py`, and the sanitized `validation/final_v1_summary_v4.json`, `final_v1_crossfile_summary.json`, and `final_v1_manifest_final.json`. The evaluation also imports the existing untracked `benchmarks/graph_ablation/source_benchmark.py` and `run_ablation.py`; include those reviewed dependencies and the necessary production implementation in a separately reviewed commit. The full main trace is approximately 1.54 MB and can contain model-reproduced source text: keep it local unless its contents, licensing and distribution are reviewed; a sanitized release artifact is an alternative.

Exclude `.env` and other local environment files, `backend/tmp/`, `.venv/`, `node_modules/`, all Chroma snapshots and databases under `validation/`, `*.pkl`, `*.bin`, caches, and historical live reports by default. `.gitignore` covers `chroma_db/` and `tmp/`, but does **not** automatically exclude all validation snapshots; do not stage `validation/` wholesale. A targeted text scan for common key/private-key patterns found no matches in the evaluation candidates; it is not a complete secret audit. An existing historical `validation/m5_fixed_context_generation.json` contains an absolute personal Windows path and should not be published unchanged. No binary indexes or `.env` files were inspected for secrets. The local largest cached index file is about 4.2 MB; it is data, not a release artifact. Resolve upstream license and attribution questions before publishing source-derived excerpts or claiming an applicable license.

The highest-value next work is to improve identifier-free target resolution and extend source-based factual and language checks to diverse real repositories. Independent human review can strengthen evidence later, but is not required to run this benchmark. Keep static-versus-runtime uncertainty and exact index coverage visible; do not claim broad retrieval improvement from this small automatically derived set.
