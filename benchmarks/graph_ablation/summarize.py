"""Summarize ablation by task, keeping query variants as repeated measurements."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from statistics import mean


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    data = json.loads(args.report.read_text(encoding="utf-8"))
    grouped = defaultdict(list)
    counts, addition_edges = Counter(), Counter()
    all_times, token_deltas = [], []
    for row in data["queries"]:
        if row["errors"]:
            continue
        grouped[row["task_id"]].append(row)
        if row["graph_stage_ms"] is not None: all_times.append(row["graph_stage_ms"])
        if row["prompt_tokens"]["status"] == "measured_ollama_prompt_eval_count": token_deltas.append(row["prompt_tokens"]["delta"])
        added_ids = {item["id"] for item in row["graph_additions"]}
        for trace in row["graph_trace"]:
            for neighbor in trace["neighbors"]:
                if neighbor["within_first_three"]:
                    counts[neighbor["relationship_status"]] += 1
                    if neighbor["resolved_chunk"] and neighbor["resolved_chunk"]["id"] in added_ids:
                        addition_edges[neighbor["relationship_status"]] += 1
    summaries = []
    for task_id, rows in grouped.items():
        t = rows[0]["target_coverage"]
        eligible = all(x["bm25_exact"] and x["vector_exact"] for x in t)
        summary = {"task_id": task_id, "category": rows[0]["category"], "variants": len(rows), "all_targets_indexed": eligible,
                   "target_coverage": t, "eligible_exact_symbol_recall_off": None, "eligible_exact_symbol_recall_on": None,
                   "eligible_all_targets_coverage_off": None, "eligible_all_targets_coverage_on": None,
                   "graph_only_relevant_variants": sum(any(not x["off"] and x["on"] for x in r["target_coverage"]) for r in rows)}
        if eligible:
            for state in ("off", "on"):
                summary[f"eligible_exact_symbol_recall_{state}"] = mean(mean(int(x[state]) for x in r["target_coverage"]) for r in rows)
                summary[f"eligible_all_targets_coverage_{state}"] = mean(all(x[state] for x in r["target_coverage"]) for r in rows)
        summaries.append(summary)
    eligible = [t for t in summaries if t["all_targets_indexed"]]
    result = {"task_count": len(summaries), "query_variant_count": len(data["queries"]),
              "task_level_eligible_count": len(eligible),
              "macro_exact_recall_off": mean(t["eligible_exact_symbol_recall_off"] for t in eligible) if eligible else None,
              "macro_exact_recall_on": mean(t["eligible_exact_symbol_recall_on"] for t in eligible) if eligible else None,
              "graph_only_relevant_variants": sum(t["graph_only_relevant_variants"] for t in summaries),
              "edge_observations_across_query_variants_not_independent": dict(counts),
              "edge_observations_matching_added_chunks_not_independent": dict(addition_edges),
              "graph_stage_ms_mean": mean(all_times) if all_times else None,
              "prompt_token_delta_mean_when_measured": mean(token_deltas) if token_deltas else None,
              "prompt_token_delta_min": min(token_deltas) if token_deltas else None,
              "prompt_token_delta_max": max(token_deltas) if token_deltas else None,
              "prompt_token_measurements": len(token_deltas), "tasks": summaries}
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
