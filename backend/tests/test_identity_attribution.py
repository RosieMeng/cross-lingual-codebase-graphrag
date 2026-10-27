"""Identical source text must retain distinct source identities through fusion."""
from langchain_core.documents import Document

from reposage.query.evidence import audit_citations, evidence_from_docs
from reposage.query.rag_engine import CodeRAGEngine


def test_fused_identical_text_maps_to_both_real_source_locations():
    engine = CodeRAGEngine.__new__(CodeRAGEngine)
    engine.repo_name = "example"
    first = Document(page_content="def run(): pass", metadata={"file": "a.py", "line_start": 1, "line_end": 1})
    second = Document(page_content=first.page_content, metadata={"file": "b.py", "line_start": 9, "line_end": 9})
    fused = engine._reciprocal_rank_fusion([first, second], [first])
    evidence = evidence_from_docs(fused, "example")
    assert [(e["citation_id"], e["file"], e["line_start"]) for e in evidence] == [
        ("S1", "a.py", 1), ("S2", "b.py", 9)
    ]
    assert evidence[0]["id"] != evidence[1]["id"]
    assert audit_citations("See [S1] and [S2].", evidence)["status"] == "matched"
    assert audit_citations("See [S3].", evidence)["status"] == "unsupported"
