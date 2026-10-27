"""Generate a read-only diagnostic from a completed Milestone 3 report."""
import argparse
import hashlib
import json
from pathlib import Path

ARMS = ("raw", "translation_only", "milestone2_expanded", "retriever_specific")


def matches(item, gold):
    return all(item.get(key) == gold.get(key) for key in ("file", "function", "line_start"))


def rank(ranking, gold):
    return next((i for i, item in enumerate(ranking, 1) if any(matches(item, g) for g in gold)), None)


def short(item, gold):
    mark = "*" if any(matches(item, g) for g in gold) else ""
    return f"{item.get('file')}:{item.get('line_start')} {item.get('function')}{mark}"


def generate(report, input_sha):
    lines = ["# Milestone 3 retrieval trace analysis", "",
             "Generated from a saved report; no retrieval or index access was performed.",
             f"Input SHA-256: `{input_sha}`", "",
             "Asterisk marks an exact match to **any unverified candidate evidence**. "
             "V and B are individual vector and BM25 call rankings, in the report's completion order; "
             "M2 can contain multiple calls. RRF is pre-graph; G is the appended graph-only list. "
             "The full graph-enriched list is RRF followed by G, not a new ranked fusion. "
             "Call order in this trace is not necessarily variant order.", "",
             "Per-query latency starts before planning and ends after retrieval, graph enrichment and scoring; "
             "it includes rewrite generation when called, plus trace overhead. It excludes engine/index initialization "
             "and answer generation. Per-call timings overlap under parallel retrieval and must not be summed as wall time.", ""]
    grouped = {}
    for row in report["queries"]:
        arms = grouped.setdefault(row["query_id"], {})
        if row["arm"] in arms:
            raise ValueError(f"Duplicate arm for {row['query_id']}")
        arms[row["arm"]] = row
    gaps = []
    for query_id, arms in grouped.items():
        if set(arms) != set(ARMS):
            raise ValueError(f"Missing or duplicate arms for {query_id}")
        baseline = arms["raw"]
        gold = baseline["evidence"]
        lines.extend([f"## {query_id} ({baseline['category']}, identifier: {baseline['identifier']})", "",
                      f"Query: {baseline['query']}", "",
                      "Candidate evidence: " + "; ".join(short(g, gold) for g in gold), ""])
        trans, route = arms["translation_only"], arms["retriever_specific"]
        trank, rrank = rank(trans["stages"].get("rrf", []), gold), rank(route["stages"].get("rrf", []), gold)
        if (trank or 10**9) < (rrank or 10**9):
            gaps.append(query_id)
        for name in ARMS:
            row = arms[name]
            status = row["plan"]["status"]
            stages = row["stages"]
            lines.append(f"**{name}** — status `{row['execution_status']}`, plan `{status}`, "
                         f"RRF first candidate rank {rank(stages.get('rrf', []), gold) or 'not in top k'}, "
                         f"{len(row['calls'])} retriever calls, {row['llm_calls']} rewrite calls, "
                         f"{row['latency_ms']:.3f} ms total; "
                         f"rewrite call latency {sum(c['latency_ms'] for c in row['llm_call_events']):.3f} ms.")
            for call in row["calls"]:
                symbol = "V" if call["retriever"] == "vector" else "B"
                ranking = call.get("ranking", [])
                entries = " → ".join(short(x, gold) for x in ranking) or f"[empty/error: {call.get('error_type', 'none')}]"
                lines.append(f"- {symbol} query: {call['query']}")
                lines.append(f"- {symbol} ranking: {entries}")
            for stage, label in (("rrf", "RRF"), ("graph_enrichment", "G")):
                entries = " → ".join(short(x, gold) for x in stages.get(stage, [])) or "[none]"
                lines.append(f"- {label}: {entries}")
            lines.append("")
    lines[2:2] = [f"Queries where the retriever-specific first candidate ranks below translation-only: "
                  f"{', '.join(gaps) if gaps else 'none'}. These are observations on candidate labels, not validated gold.", ""]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    raw = args.report.read_bytes()
    report = json.loads(raw)
    result = generate(report, hashlib.sha256(raw).hexdigest())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as out:
        out.write(result)
    print(f"Wrote {args.output}")


if __name__ == "__main__": main()
