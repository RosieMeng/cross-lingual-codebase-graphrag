"""Summarize qualified-graph results by source-derived task."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import mean


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    data = json.loads(args.report.read_text(encoding="utf-8"))
    groups = defaultdict(list)
    for row in data["queries"]:
        if not row["errors"]: groups[row["task_id"]].append(row)
    tasks = []
    for task_id, rows in groups.items():
        eligible = all(t["bm25_exact"] and t["vector_exact"] for t in rows[0]["target_coverage"])
        task = {"task_id": task_id, "category": rows[0]["category"], "variants": len(rows), "all_targets_indexed": eligible,
                "graph_only_relevant_variants": sum(any(not x["off"] and x["on"] for x in r["target_coverage"]) for r in rows)}
        for state in ("off", "on"):
            task[f"exact_recall_{state}"] = mean(mean(int(x[state]) for x in r["target_coverage"]) for r in rows) if eligible else None
            task[f"all_targets_{state}"] = mean(all(x[state] for x in r["target_coverage"]) for r in rows) if eligible else None
        tasks.append(task)
    eligible = [t for t in tasks if t["all_targets_indexed"]]
    rows = data["queries"]
    tokens = [r["prompt_tokens"]["delta"] for r in rows if r["prompt_tokens"]["status"] == "measured_ollama_prompt_eval_count"]
    times = [r["graph_stage_ms"] for r in rows if r["graph_stage_ms"] is not None]
    result = {"tasks": tasks, "task_count": len(tasks), "eligible_tasks": len(eligible), "query_variants": len(rows),
              "execution_failures": sum(bool(r["errors"]) for r in rows),
              "macro_exact_recall_off": mean(t["exact_recall_off"] for t in eligible) if eligible else None,
              "macro_exact_recall_on": mean(t["exact_recall_on"] for t in eligible) if eligible else None,
              "graph_only_relevant_variants": sum(t["graph_only_relevant_variants"] for t in tasks),
              "mean_graph_added_chunks": mean(len(r["graph_additions"]) for r in rows) if rows else None,
              "mean_context_character_delta": mean(r["context_chars_on"] - r["context_chars_off"] for r in rows) if rows else None,
              "mean_graph_stage_ms": mean(times) if times else None,
              "prompt_probe_count": len(tokens), "prompt_token_delta_mean": mean(tokens) if tokens else None,
              "all_new_evidence_in_context": all(all(r["new_evidence_in_context"].values()) for r in rows)}
    with args.output.open("x", encoding="utf-8") as f: json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
