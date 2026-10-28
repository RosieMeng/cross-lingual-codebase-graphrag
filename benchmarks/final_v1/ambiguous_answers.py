"""Evaluate the frozen ambiguous task against its saved retrieval contexts."""
import copy
import json
from pathlib import Path
import pickle
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "backend"), str(ROOT)]
from benchmarks.cross_lingual_code_qa.controlled_eval import fingerprint
from reposage.query.evidence import audit_citations, chunk_identity, evidence_from_docs
from reposage.query.graph_answer import graph_answer
from reposage.query.rag_engine import CodeRAGEngine, QUERY_PROMPT, SYSTEM_PROMPT, _answer_language_instruction
from reposage.query.source_graph import enrich, graph_context, load_graph
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

INDEX = ROOT / "validation/m4a_nomic_snapshot/micrograd"
OUT = ROOT / "validation/final_v1_ambiguous_answers.json"


def main():
    if OUT.exists(): raise FileExistsError(OUT)
    hashes = fingerprint(INDEX)
    with (INDEX / "bm25_docs.pkl").open("rb") as f: docs = pickle.load(f)  # trusted index only
    indexed = {chunk_identity(d, "micrograd"): d for d in docs}
    graph = load_graph(ROOT / "validation/m6b_sidecars/micrograd.json", "micrograd")
    source = json.loads((ROOT / "validation/final_v1_results.json").read_text(encoding="utf-8"))
    rows = []
    engine = CodeRAGEngine.__new__(CodeRAGEngine)
    engine.repo_name = "micrograd"
    model = ChatOllama(base_url="http://127.0.0.1:11434", model="qwen3:1.7b", temperature=0, num_predict=256)
    for row in source["queries"]:
        if row["task_id"] != "m_ambiguous": continue
        seeds = [indexed[item["id"]] for item in row["graph"]["off"]["ranking"]]
        info = {}
        extra = enrich(graph, seeds, docs, "micrograd", row["question"], diagnostics=info)
        selected = seeds + extra
        evidence = evidence_from_docs(selected, "micrograd")
        if [e["id"] for e in evidence] != [r["id"] for r in row["graph"]["grounded"]["ranking"]]:
            raise ValueError("Saved graph context differs")
        context = engine._format_context(selected) + graph_context(info, evidence)
        messages = [SystemMessage(content=SYSTEM_PROMPT.format(repo_name="micrograd") + "\n" + _answer_language_instruction(row["question"])),
                    HumanMessage(content=QUERY_PROMPT.format(repo_name="micrograd", context=context, question=row["question"]))]
        start = time.perf_counter()
        error = None
        try:
            response = model.invoke(messages, think=False)
            answer, done = response.content, response.response_metadata.get("done_reason")
        except Exception as exc:
            answer, done, error = "", None, type(exc).__name__
        rows.append({"query_id": row["query_id"], "language": row["language"], "question": row["question"], "evidence": evidence,
                     "graph": copy.deepcopy(info), "model_answer": answer, "citation_audit": audit_citations(answer, evidence),
                     "done_reason": done, "error": error, "latency_ms": round((time.perf_counter() - start) * 1000, 3),
                     "evidence_first": graph_answer(row["question"], info, evidence), "judgment_status": "unverified"})
        print(row["query_id"], "done", done, "error", error, flush=True)
    if fingerprint(INDEX) != hashes: raise RuntimeError("Source index changed")
    with OUT.open("x", encoding="utf-8") as f: json.dump({"note": "Ambiguous definition task; exact chunks absent; non-graph model path; unverified", "source_unchanged": True, "queries": rows}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__": main()
