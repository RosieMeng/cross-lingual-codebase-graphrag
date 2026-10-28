# Release licensing and provenance

This is a fact-finding record, not a legal opinion. [Repo-Sage](https://github.com/Ganesh-403/Repo-Sage) is the original project. Its README explicitly states MIT and links a LICENSE, but the inspected upstream root does not contain that file. The initial local import matches the upstream README and RAG engine Git blobs; this does not prove the provenance of every file, exact upstream revision, or copyright holder. Existing application architecture and original files are inherited; multilingual planning, source graph v2, grounding, evidence-first answers, tests and evaluations are project extensions. Some extension files modify inherited files, so ownership cannot be inferred solely from paths. Preserve attribution and all notices. The original copyright notice and the provenance of some inherited assets remain uncertain. Contacting the upstream maintainer is optional, not a prerequisite imposed here.

The independent ItsDangerous checkout is [Pallets/ItsDangerous](https://github.com/pallets/itsdangerous) at `672971d66a2ef9f85151e53283113f33d642dabd`; its LICENSE.txt says copyright 2011 Pallets and contains the BSD 3-Clause terms. The checkout and built index are not bundled. Source-derived facts and small excerpts still deserve attribution and review.

The root [LICENSE](../LICENSE) supplies unchanged standard MIT terms for RosieMeng's original contributions, to the extent RosieMeng holds the relevant rights. It does not assert ownership of upstream or third-party material. The upstream README's statement is evidence of its declared MIT license; the missing upstream license file means its original copyright notice is not verified. See [ATTRIBUTION.md](../ATTRIBUTION.md).

Dependencies and frontend packages have separate terms. See [the third-party audit](third_party_audit.md) for installed package notices, unresolved package-file gaps, and inherited assets. The installed Ollama models identify Apache-2.0 in `ollama show`; model distribution and its terms require separate review. The generated build and `node_modules` are not proposed for Git. A complete dependency and historical asset notice audit remains outstanding. The earlier [MIT draft](LICENSE-MIT-DRAFT.txt) remains a historical review template.

## Suggested maintainer issue (draft only; not posted)

**Title:** Clarification of MIT license and copyright notice

Hello, and thank you for publishing Repo-Sage. The README describes the project as MIT-licensed and links to a LICENSE file, but I could not find that file in the repository. Would you confirm the intended license terms and copyright attribution, or add the missing license file? I am preparing a derivative project and want to preserve the correct attribution and terms. Thank you!

## Public release checklist and Git findings

The earlier release assessment identified unresolved dependency/asset and GitHub-hosted material review. A targeted scan of all objects reachable from local refs found a personal path in a historical validation trace and large index/trace blobs; inspection shows these are reachable from local `refs/codex/turn-diffs/...` checkpoint refs, not the named project branch history as checked. Review exactly which refs would be published and conduct an independent history and secret scan before publication. A targeted regex scan is not proof that history is secret-free. Do not publish `refs/codex/*`. Do not rewrite history or delete refs without a separate informed decision.

Do not stage local `.env`, virtual environments, `node_modules`, validation snapshots, Chroma databases, pickles, binary indexes or full generated traces. `.gitignore` does not cover all validation data; use explicit file allowlists. Existing user files and historic reports remain untouched. Verify copyright and third-party content in all candidate artifacts before sharing. Preserve the original remote and repository visibility.
