"""Task-weighted metrics and explicitly limited automated answer diagnostics."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import sys
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "backend"), str(ROOT)]
from reposage.query.multilingual_query import detect_language
from benchmarks.graph_ablation.source_benchmark import analyze


def match(doc, target):
    return (doc.get("file") == target["file"] and doc.get("name") == target["function"]
            and doc.get("line_start") == target["line_start"] and doc.get("line_end") == target["line_end"])


def metrics(ranking, targets):
    if not targets: return None
    positions = [i for i, doc in enumerate(ranking, 1) if any(match(doc, target) for target in targets)]
    covered = [any(match(doc, target) for doc in ranking) for target in targets]
    return {"hit": int(bool(positions)), "mrr": 1 / min(positions) if positions else 0, "recall": sum(covered) / len(targets), "all": int(all(covered))}


def classify_language(answer):
    if answer.startswith(("Definition:", "I cannot")): return "en"
    if answer.startswith(("Définition:", "Je ne peux")): return "fr"
    if answer.startswith(("定义:", "无法")): return "zh"
    result = detect_language(answer).language
    return result if result in {"en", "fr", "zh"} else "unknown"


def abstention(answer):
    return bool(re.search(r"\b(?:cannot|can.t|not found|not defined|not implemented|does not exist|no such|not present|unable|insufficient)\b|\b(?:aucun|aucune|introuvable|impossible|n'existe|pas trouvé|pas définie?|ne peut)\b|不存在|没有找到|未找到|未定义|没有定义|无法|不足", answer, re.I))


def summary(rows, tasks):
    eligible = [r for r in rows if r["coverage"] and all(c["vector"] and c["bm25"] for c in r["coverage"]) and not r["retrieval"]["raw"]["errors"]]
    by_task = defaultdict(list)
    for r in eligible: by_task[r["task_id"]].append(r)
    arms = ("dense", "bm25", "rrf", "translation", "graph_off", "qualified", "grounded")
    result = {"query_variants": len(rows), "task_count": len({r['task_id'] for r in rows}), "eligible_tasks": len(by_task),
              "eligible_variants": len(eligible), "raw_failures": sum(bool(r["retrieval"]["raw"]["errors"]) for r in rows),
              "translation_failures": sum(bool(r["retrieval"]["translation_only"]["errors"]) for r in rows), "arms": {}}
    for arm in arms:
        task_means = []
        for task_id, variants in by_task.items():
            vals = []
            for r in variants:
                if arm == "translation" and r["retrieval"]["translation_only"]["errors"]: continue
                if arm in ("dense", "bm25"):
                    rank = r["retrieval"]["raw"]["rankings"].get("vector" if arm == "dense" else "bm25", [])[:5]
                elif arm in ("rrf", "translation"):
                    rank = r["retrieval"]["raw" if arm == "rrf" else "translation_only"]["rrf"][:5]
                else:
                    rank = r["graph"][{"graph_off": "off", "qualified": "qualified", "grounded": "grounded"}[arm]]["ranking"]
                vals.append(metrics(rank, tasks[task_id]["targets"]))
            if vals: task_means.append({key: mean(v[key] for v in vals) for key in ("hit", "mrr", "recall", "all")})
        result["arms"][arm] = {"eligible_tasks": len(task_means), **{key: mean(t[key] for t in task_means) if task_means else None for key in ("hit", "mrr", "recall", "all")}}
    result["mean_translation_llm_calls"] = mean(r["retrieval"]["translation_only"]["llm_calls"] for r in rows) if rows else None
    result["translation_fallbacks"] = sum(r["retrieval"]["translation_only"]["plan"]["status"] in {"rewrite_failed", "rewrite_invalid", "rewrite_not_english", "abstained", "rewriter_unavailable"} for r in rows)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("results", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--benchmark", type=Path, default=ROOT / "benchmarks/final_v1/benchmark.json")
    args = p.parse_args()
    data = json.loads(args.results.read_text(encoding="utf-8"))
    dataset = json.loads(args.benchmark.read_text(encoding="utf-8"))
    tasks = {t["id"]: t for t in dataset["tasks"]}
    rows = data["queries"]
    source_paths = {"micrograd": ROOT / "backend/tmp/reposage/micrograd", "sample_repo": ROOT / "backend/tests/fixtures/sample_repo"}
    source_symbols, source_text = {}, {}
    for repo, path in source_paths.items():
        symbols, _ = analyze(path)
        source_symbols[repo] = {s.qualified for s in symbols}
        source_text[repo] = "\n".join((path / f).read_text(encoding="utf-8") for f in dataset["repositories"][repo]["source_sha256"])
    result = {"overall": summary(rows, tasks), "by_language": {}, "by_task_type": {}, "by_repository": {}, "by_task": {}, "answers": {}}
    for label, key in (("by_language", "language"), ("by_task_type", "task_type")):
        for value in sorted({r[key] for r in rows}): result[label][value] = summary([r for r in rows if r[key] == value], tasks)
    for repo in ("micrograd", "sample_repo"):
        result["by_repository"][repo] = summary([r for r in rows if tasks[r["task_id"]]["repo"] == repo], tasks)
    for task_id in tasks: result["by_task"][task_id] = summary([r for r in rows if r["task_id"] == task_id], tasks)
    answer_rows = [r for r in rows if r["answer_evaluation"] is not None]
    for arm in ("model", "model_paired", "evidence_first"):
        cases = []
        for r in answer_rows:
            if arm == "model_paired" and r["answer_evaluation"]["evidence_first"]["answer"] is None:
                continue
            a = r["answer_evaluation"]["model" if arm == "model_paired" else arm]
            answer = a.get("answer")
            if answer is None: continue
            language = classify_language(answer)
            target = r["language"] if r["language"] != "mixed" else None
            # Exact call-site assertions only; absence is not a factual failure.
            callsite = ("micrograd/engine.py:65" in answer and "build_topo" in answer) if r["task_id"] == "m_callee" else None
            checked_claims = []
            if r["task_id"] == "m_caller":
                sites = tasks[r["task_id"]]["relationship"]["sites"]
                verified = {(s["expression"], s["file"], s["line"]) for s in sites}
                for line in answer.splitlines():
                    expressions = re.findall(r"\b[yg]\.backward\b", line)
                    locations = re.findall(r"(test/test_engine\.py):(\d+)", line)
                    for expression in expressions:
                        for file, number in locations:
                            checked_claims.append({"expression": expression, "file": file, "line": int(number),
                                                   "source_observation_matches": (expression, file, int(number)) in verified,
                                                   "runtime_target_verified": False})
            guessed = []
            # Only unseen dotted identifiers introduced by the answer, not
            # symbols already in the question; heuristic, not a verdict.
            repo = tasks[r["task_id"]]["repo"]
            corpus = source_text[repo]
            for token in re.findall(r"`([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+)`", answer):
                if token.endswith((".py", ".js", ".ts", ".md", ".json")): continue
                if token not in r["question"] and token not in corpus and token not in source_symbols[repo]: guessed.append(token)
            cases.append({"query_id": r["query_id"], "task_id": r["task_id"], "language": r["language"], "citation_status": a["citation_audit"]["status"],
                          "language_detected_heuristic": language, "requested_language": target,
                          "language_matches_heuristic": language == target if target and language != "unknown" else None,
                          "abstention_heuristic": abstention(answer), "truncated": a.get("truncated"), "error": a.get("error"),
                          "exact_build_topo_callsite_mentioned": callsite,
                          "checked_caller_callsite_observations": checked_claims,
                          "unseen_quoted_dotted_symbols_unverified": sorted(set(guessed))})
        scored = [c for c in cases if c["language_matches_heuristic"] is not None]
        no_answer = [c for c in cases if tasks[c["task_id"]]["type"] == "no_answer"]
        def answer_group(group):
            judged = [c for c in group if c["language_matches_heuristic"] is not None]
            return {"cases": len(group), "tasks": len({c["task_id"] for c in group}),
                    "citation_status": dict(Counter(c["citation_status"] for c in group)),
                    "truncated": sum(c["truncated"] is True for c in group),
                    "language_heuristic_denominator": len(judged),
                    "language_heuristic_matches": sum(c["language_matches_heuristic"] for c in judged)}
        result["answers"][arm] = {"cases": len(cases), "task_count": len({c['task_id'] for c in cases}),
            "citation_status": dict(Counter(c["citation_status"] for c in cases)), "truncated": sum(c["truncated"] is True for c in cases),
            "errors": sum(bool(c["error"]) for c in cases), "language_heuristic_denominator": len(scored),
            "language_heuristic_matches": sum(c["language_matches_heuristic"] for c in scored),
            "no_answer_cases": len(no_answer), "no_answer_abstention_detected_heuristic": sum(c["abstention_heuristic"] for c in no_answer),
            "exact_build_topo_callsite_mentioned": sum(c["exact_build_topo_callsite_mentioned"] is True for c in cases),
            "caller_callsite_observations_checked": sum(len(c["checked_caller_callsite_observations"]) for c in cases),
            "caller_callsite_observations_matched_source": sum(x["source_observation_matches"] for c in cases for x in c["checked_caller_callsite_observations"]),
            "unseen_quoted_dotted_symbols_unverified_cases": sum(bool(c["unseen_quoted_dotted_symbols_unverified"]) for c in cases),
            "by_language": {lang: answer_group([c for c in cases if c["language"] == lang]) for lang in sorted({c["language"] for c in cases})},
            "by_task": {task: answer_group([c for c in cases if c["task_id"] == task]) for task in sorted({c["task_id"] for c in cases})},
            "details": cases}
    with args.output.open("x", encoding="utf-8") as f: json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps({"overall": result["overall"], "answers": {k: {x: y for x, y in v.items() if x != "details"} for k, v in result["answers"].items()}}, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
