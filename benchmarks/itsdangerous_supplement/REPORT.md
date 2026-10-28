# ItsDangerous independent repository supplement

This is an automated, unreviewed smoke evaluation, not independently human-validated performance. The seven source-derived tasks have four authored EN/FR/ZH/mixed variants each; translations are repeated observations, not independent information needs. The caller task checks one sufficient static caller/callee pair, not an exhaustive caller list. Natural-language relevance and wording are unverified.

Source: [Pallets ItsDangerous](https://github.com/pallets/itsdangerous), revision `672971d66a2ef9f85151e53283113f33d642dabd`, BSD-3-Clause, copyright 2011 Pallets (see its LICENSE.txt). Definitions, ranges and local references are derived from Python AST. Relative import `base64_encode` and its call are checked independently, but the graph analyzer leaves the call unresolved. Static references do not establish runtime dispatch. The no-answer task checks Python definitions only.

The separate index uses the existing ingestion pipeline, Nomic `nomic-embed-text:latest` (Ollama ID `0a109f422b47`, 768 dimensions), 120 chunks from 30 files, no LLM docstring generation, and its own v2 source graph. Query model is the same Nomic model; rewrites use `qwen3:1.7b` (Ollama ID `8f68893c685c`). The evaluator reuses the V1 worker, rebuilds its v2 graph in memory from the same source and indexed documents, and makes one vector and one BM25 request per arm, candidate limit and final RRF cutoff 5. Source and dataset hashes and per-query details are recorded locally. Evaluation opens only a hash-verified temporary index copy; source index hash remained unchanged. The original micrograd index and historic results were not touched.

Full model digests: Nomic `0a109f422b47e3a30ba2b10eca18548e944e8a23073ee3f3e947efcf3c45e59f`; Qwen `8f68893c685c3ddff2aa3fffce2aa60a30bb2da65ca488b61fff134a4d1730e7`. The frozen dataset SHA-256 is `f8d008555f020793eda061d2b417b0d06666aad16717c0651e28dd7d044bd2e4`; regenerating it produced the identical hash. Code revision when indexed was `5ca82fe` plus the supplement scripts in the working tree. Translation made 12 LLM calls across 28 variants.

Exact target chunks were present in both index components for 3/6 answerable tasks (12/24 variants): `encoding`, `caller`, and `cross`. `Signer.sign` spans only four lines and is not an exact indexed chunk; therefore `sign`, `callee`, and `ambiguous` are excluded from conditional metrics, not counted as retrieval failures. The no-answer task has no retrieval recall denominator. Chunk coverage does not imply semantic completeness.

### Coverage and cross-module diagnosis

All three ineligible tasks lack the same exact target, `src/itsdangerous/signer.py:222–225` (`Signer.sign`); their other requested targets, `Signer.get_signature` (215–220) and `TimestampSigner.sign` (45–51), are present in both components. The AST contains `Signer.sign`, but the Python chunker skips definitions shorter than five lines, before documents reach either index. This is not an index failure or an exact-range matching error. V2 maps the method to its enclosing `Signer` class chunk, but that document is 9,007 characters and exceeds the 8,000-character graph addition budget; the two single-symbol English traces report `unavailable_budget`. The ambiguous bare `sign` query correctly abstains. In a future separately evaluated V1.1, consider retaining short definitions as exact chunks in a new index, and independently consider handling oversized mapped evidence without fabricating ranges. Both require testing and explicit migration; this release changes neither.

In each cross-module variant the raw seeds include `base64_encode` (20–25) and the enclosing `URLSafeSerializerMixin` class (15–69), which contains `dump_payload` (55–69), but not the latter's **exact function chunk**. The exact-match metric does not count that enclosing class as the function chunk. No exact-symbol grounding applies to these identifier-free questions. The v2 sidecar records the call to `base64_encode` at `url_safe.py:64` as unresolved and has no edge for it: its resolver accepts absolute explicit imports, while this import is relative (`url_safe.py:8`). Graph expansion adds `want_bytes`, not the missing exact function chunk, in all four variants. Thus the observed 0/4 is an exact-chunk coverage result despite source text being present in an enclosing class, and the absent edge prevents that call from supplying an additional traversal route; it does not prove the source text was absent or that resolving this edge alone would change the outcome.

| Eligible group | N variants | Dense Hit@5 / MRR@5 | BM25 Hit@5 / MRR@5 | Raw hybrid Hit@5 / MRR@5 | Translation hybrid Hit@5 / MRR@5 |
|---|---:|---:|---:|---:|---:|
| Overall (3 tasks) | 12 | 1.000 / 0.792 | 0 / 0 | 1.000 / 0.425 | 1.000 / 0.425 |
| EN | 3 | 1.000 / 0.778 | 0 / 0 | 1.000 / 0.289 | 1.000 / 0.289 |
| FR | 3 | 1.000 / 0.778 | 0 / 0 | 1.000 / 0.400 | 1.000 / 0.400 |
| ZH | 3 | 1.000 / 0.833 | 0 / 0 | 1.000 / 0.444 | 1.000 / 0.444 |
| Mixed | 3 | 1.000 / 0.778 | 0 / 0 | 1.000 / 0.567 | 1.000 / 0.567 |

Hit and reciprocal rank credit the first of the task's required targets, not complete evidence. For all eligible tasks every BM25 query missed its exact targets; dense and both hybrid arms hit at least one. The hybrid ranks are lower than dense in this tiny sample; the observed lexical misses are consistent with that result, but do not prove a general cause. Both hybrid scores were identical on eligible queries: none received an accepted rewrite. Across all 28 variants: one accepted rewrite, ten rejected as not English, one invalid rewrite, seven language abstentions, nine English-original plans; failed/abstained rewrites fall back to raw. The rewrite arm incurred extra calls and is not equal cost.

Graph comparison uses identical raw RRF seeds, with graph context appended beyond the five seeds. All-required exact chunk coverage: caller off 0/4, qualified 4/4, grounded 4/4; cross-module off 0/4, qualified 0/4, grounded 0/4. Mean formatted context characters were respectively 2,472 / 3,191 / 3,191 for caller and 4,181 / 4,704 / 4,704 for cross-module. The graph's extra context is not an equal-budget ranking gain. The cross-module failure is consistent with the unresolved relative import, but this experiment alone does not establish causation. No answer-generation experiment was run; no claim is made about citation or factual answer quality here. There were 0/28 reported raw or translation retrieval execution failures. Chroma emitted telemetry compatibility warnings.

## Reproduce

Use a clean checkout at the pinned revision. The following builds a **new** separate index, never an existing collection. The indexer deletes an existing same-named destination; check the destination first. Install the existing backend requirements and start Ollama with the named models. Set the source path to your checkout.

```powershell
$source = 'C:/path/to/itsdangerous'
if (Test-Path validation/itsdangerous_new_index) { throw 'Choose a new index directory' }
$env:PYTHONPATH = 'backend'
backend/.venv/Scripts/python.exe -c "from reposage.ingestion.repo_indexer import RepoIndexer; import sys; print(RepoIndexer(persist_dir='validation/itsdangerous_new_index', ollama_embed_model='nomic-embed-text').index_repo('local:///'+sys.argv[1]))" $source
backend/.venv/Scripts/python.exe benchmarks/itsdangerous_supplement/freeze.py --source $source --output validation/itsdangerous_refrozen.json
Get-FileHash benchmarks/itsdangerous_supplement/dataset.json -Algorithm SHA256
Get-FileHash validation/itsdangerous_refrozen.json -Algorithm SHA256
backend/.venv/Scripts/python.exe benchmarks/itsdangerous_supplement/evaluate.py --source $source --index validation/itsdangerous_new_index/itsdangerous --output validation/itsdangerous_new_results.json
backend/.venv/Scripts/python.exe benchmarks/itsdangerous_supplement/summarize.py validation/itsdangerous_new_results.json
```

`freeze.py` refuses an existing output and verifies clean revision and static facts. Run the refreeze check before indexing if the source contains ignored generated Python files; those can change the source hash. Index construction and LLM output can vary by model build and environment; compare model digests and configuration, not just names. Local result traces may contain source text and must be reviewed before sharing.
