# Release licensing and provenance: blocked pending confirmation

This is a fact-finding record, not a legal opinion or a grant of rights. [Repo-Sage](https://github.com/Ganesh-403/Repo-Sage) is the original project. Its README states MIT and links a LICENSE, but the inspected upstream root does not contain that file. The initial local import matches the upstream README and RAG engine Git blobs; this does not prove the provenance of every file, exact upstream revision, or copyright holder. Existing application architecture and original files are inherited; multilingual planning, source graph v2, grounding, evidence-first answers, tests and evaluations are project extensions. Some extension files modify inherited files, so ownership cannot be inferred solely from paths. Preserve attribution and all notices. Confirm the original grant, copyright notice and any third-party source or asset provenance with the maintainer before publishing. Do not create a repository-wide LICENSE asserting sole ownership.

The independent ItsDangerous checkout is [Pallets/ItsDangerous](https://github.com/pallets/itsdangerous) at `672971d66a2ef9f85151e53283113f33d642dabd`; its LICENSE.txt says copyright 2011 Pallets and contains the BSD 3-Clause terms. The checkout and built index are not bundled. Source-derived facts and small excerpts still deserve attribution and review.

Dependencies and frontend packages have separate terms. The installed Ollama models identify Apache-2.0 in `ollama show`; model distribution, model licenses, npm transitive dependencies, fonts, and bundled frontend assets have not been exhaustively checked. The generated build and `node_modules` are not proposed for Git. A complete dependency and asset notice audit remains outstanding.

## Suggested maintainer issue (draft only; not posted)

**Title:** Clarification of MIT license and copyright notice

Hello, and thank you for publishing Repo-Sage. The README describes the project as MIT-licensed and links to a LICENSE file, but I could not find that file in the repository. Would you confirm the intended license terms and copyright attribution, or add the missing license file? I am preparing a derivative project and want to preserve the correct attribution and terms. Thank you!

## Public release checklist and Git findings

**Decision: BLOCKED.** Confirm the upstream grant and complete dependency/asset review. A targeted scan of all objects reachable from local refs found a personal path in a historical validation trace and large index/trace blobs; inspection shows these are reachable from local `refs/codex/turn-diffs/...` checkpoint refs, not the named project branch history as checked. Review exactly which refs would be published and conduct an independent history and secret scan before publication. A targeted regex scan is not proof that history is secret-free. Do not publish `refs/codex/*`. Do not rewrite history or delete refs without a separate informed decision.

Do not stage local `.env`, virtual environments, `node_modules`, validation snapshots, Chroma databases, pickles, binary indexes or full generated traces. `.gitignore` does not cover all validation data; use explicit file allowlists. Existing user files and historic reports remain untouched. Verify copyright and third-party content in all candidate artifacts before sharing. Preserve the original remote and repository visibility.

## MIT template for review only

The following standard template is copied from the [Open Source Initiative MIT license](https://opensource.org/license/mit). Replace placeholders only after confirming the relevant holder and year, and apply it only where the holder is entitled to do so. This draft does not license inherited code.

> Copyright <YEAR> <COPYRIGHT HOLDER>
>
> Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
