"""Structured evidence and conservative checks for answer references."""
import hashlib
import json
import re


def chunk_identity(doc, repo_name=None):
    """Stable identity shared by vector, lexical, and graph copies of a chunk."""
    metadata = doc.metadata
    identity = {"repo": metadata.get("repo") or repo_name, "file": (metadata.get("file") or "").replace("\\", "/"),
                "line_start": metadata.get("line_start"), "line_end": metadata.get("line_end"),
                "content_sha256": hashlib.sha256(doc.page_content.encode("utf-8")).hexdigest()}
    return hashlib.sha256(json.dumps(identity, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def evidence_from_docs(docs, repo_name=None):
    evidence, seen = [], set()
    for doc in docs:
        metadata = doc.metadata
        file = metadata.get("file")
        start, end = metadata.get("line_start"), metadata.get("line_end")
        if not isinstance(file, str) or not file or not isinstance(start, int) or isinstance(start, bool) or start < 1:
            continue
        if not isinstance(end, int) or isinstance(end, bool) or end < start:
            end = None
        raw_type = metadata.get("type") or ""
        graph = raw_type.endswith(" (Graph Context)")
        chunk_type = raw_type.removesuffix(" (Graph Context)")
        digest = chunk_identity(doc, repo_name)
        if digest in seen:
            continue
        seen.add(digest)
        evidence.append({"id": digest, "citation_id": f"S{len(evidence) + 1}", "repo": metadata.get("repo") or repo_name, "file": file, "line_start": start, "line_end": end,
                         "symbol": metadata.get("name"), "type": chunk_type,
                         "retrieval_stage": "graph" if graph else "retrieval"})
    return evidence


_CITATION = re.compile(r"(?<![\w/])([\w./\\-]+\.[\w]+):(\d+)(?:-(\d+))?", re.UNICODE)
_SOURCE_MARKER = re.compile(r"\[S[^\]\n]*\]", re.IGNORECASE)


def audit_citations(answer, evidence):
    """Check recognizable file:line references against retrieved line ranges.

    This does not judge whether evidence supports the surrounding claim.
    """
    invalid, cited = [], []
    found = False
    valid_ids = {item.get("citation_id") for item in evidence if item.get("citation_id")}
    for match in _SOURCE_MARKER.finditer(answer):
        found = True
        marker = match.group()
        name = marker[1:-1]
        if name not in valid_ids:
            if marker not in invalid: invalid.append(marker)
        elif name not in cited:
            cited.append(name)
    for match in _CITATION.finditer(answer):
        file = match.group(1).replace("\\", "/")
        start, end = int(match.group(2)), int(match.group(3) or match.group(2))
        valid = any(item["file"].replace("\\", "/") == file and item["line_start"] <= start <= end
                    and end <= (item["line_end"] or item["line_start"]) for item in evidence)
        if not valid and match.group() not in invalid:
            invalid.append(match.group())
    return {"status": "unsupported" if invalid else "matched" if found and cited else "not_detected",
            "unsupported": invalid, "cited": cited, "claim_coverage": "not_assessed", "factual_support": "not_verified"}
