# Release gate observations

These are local checks, not human-validated benchmark results or a legal clearance.

## Publication refs and files

On 2026-10-28, `git ls-remote --heads --tags origin` advertised only `refs/heads/main` at `5ecc90ef1ff3768b9dfc01f373379e639595207f` and no tags. Its commit matches the locally cached remote main. The local feature branch was not advertised. A targeted scan of blobs reachable from local main, cached remote main, and the feature branch found no matches for the selected private-key, token, personal Windows path, database/index filename or >1 MB checks. This is not a comprehensive secret scan. Local `refs/codex/turn-diffs/*` do contain historical validation/index blobs and a personal path; these are not reachable from the checked branches and must not be published. Never use `--mirror` or push those refs. This inspection does not establish GitHub visibility, permissions or future remote changes.

The current working tree contains many untracked historical traces and indexes, including local paths and generated text; `validation/` is not comprehensively ignored. Use explicit staging allowlists. Do not stage `.env`, venvs, model data, Chroma databases, pickles, indexes or full traces. Targeted scans of the release documentation and supplement found no selected secret or personal-path patterns; review before any publication.

## Dependency and asset review

The npm lockfile reports mostly MIT, Apache and BSD dependencies, plus `caniuse-lite` (CC-BY-4.0), `lightningcss` packages (MPL-2.0) and `minimatch` (BlueOak-1.0.0). The installed backend environment reported 118 distributions; 11 lacked a simple `License` or `License-Expression` metadata field, which is not evidence that they are unlicensed. Follow up in the actual package license files and check distribution obligations. The clean frontend `npm ci` audit reported zero vulnerabilities at that time; this is not a security guarantee.

Tracked `hero.png`, `react.svg`, `vite.svg`, `favicon.svg`, and `icons.svg` were present in the initial import; their independent provenance and permission have not been established. No imports of the three `src/assets` files or `icons.svg` were found in current source, while the favicon is referenced by `frontend/index.html`. The frontend loads Inter and JetBrains Mono from Google Fonts, which also means browser use makes external requests. Audit their terms and privacy implications. The landing page depicts a `reposage index` terminal command and a 12,453-file/18.4-second run as animation; neither is an observed result of this release, and the command is not an implemented CLI. The UI still displays RepoSage branding. Review this copy before a public portfolio release.

`backend/reposage/ingestion/repo_indexer.py` logs local repository paths and remote URLs at INFO level. Do not put credentials into repository URLs; review logging before handling sensitive repositories. The backend's permissive CORS and default host binding warrant review before exposing the service beyond a local machine. These are observed configuration risks, not findings of leaked credentials.

## Demonstration performed

A running backend using the existing Python 3.11 environment, local `qwen3:1.7b` and `nomic-embed-text`, and both experimental planning flags disabled indexed the clean ItsDangerous revision `672971d66a2ef9f85151e53283113f33d642dabd` through `POST /index` into a new isolated directory: 30 files, 120 chunks, v2 sidecar with 144 nodes and 31 edges. Existing indexes were not opened for this demo. Five actual `POST /query/stream` requests completed without reported errors or truncation:

| Query | Observation |
|---|---|
| EN, FR, ZH, and mixed identifier-free questions about URL-safe base64 encoding | All four retrieved `src/itsdangerous/encoding.py:20–25`; evidence IDs used in answers mapped to returned source IDs. The French and Chinese answers were in English. Citation syntax does not verify factual claims; the Chinese response also described `urlsafe_b64decode` as producing encoded strings, which the source does not support. |
| `Who calls Signer.unsign?` (symbol backticked) | Evidence-first answer cited `Signer.unsign` at `src/itsdangerous/signer.py:244–256`, the static `Signer.validate` call at line 263, and an unresolved call site separately. Runtime dispatch was not asserted. |

All returned evidence paths and line ranges existed within the source checkout; this check did not establish every cited claim. A `POST /query` with the same graph question returned `execution_mode=graph_grounded_rag`, `tools_executed=false`, and six genuine evidence records. This tests the graph RAG route, not autonomous Agent tool behavior or every fallback mode.

The separately installed frontend served HTTP 200, and its test suite passed four tests, lint and production build passed. Browser automation had no available browser surfaces; no interactive evidence-panel acceptance is claimed. Manual walkthrough: start backend and frontend per README, index a new repository name, select it, send the four language queries and the backticked graph question, check displayed source IDs and file/line ranges against the checkout, toggle Agent mode and inspect fallback/tool status, and check the warning shown when a citation is missing or unsupported. Verify language and factual content separately.

The system `python` was 3.13.7; a clean venv using it failed to build `chroma-hnswlib` because Microsoft Visual C++ 14.0 or newer was unavailable. A separate clean Python 3.11.17 environment installed `backend/requirements.txt`, passed `pip check`, imported the application and returned HTTP 200 from `/health` through FastAPI's test client. The live indexing and multilingual requests above used the existing 3.11 environment, not the newly installed one. Neither test modified the existing environment.

## Decision

**BLOCKED for public release pending upstream licensing/attribution and third-party asset review.** A browser walkthrough and a full independent secret/history review are also outstanding. The small automatic multilingual tests demonstrate source retrieval and a narrow graph answer, not general multilingual answer reliability.
