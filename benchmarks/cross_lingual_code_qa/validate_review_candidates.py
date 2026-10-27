"""Validate cross-record invariants; this does not certify human judgments."""
import argparse
import json
from pathlib import Path, PurePosixPath
import re

try:
    from .prepare_review_candidates import ROOT, MICROGRAD, git_revision, line_range
except ImportError:
    from prepare_review_candidates import ROOT, MICROGRAD, git_revision, line_range


def validate(data, verify_local_source=False):
    errors = []
    def check(condition, message):
        if not condition: errors.append(message)
    check(data.get("schema_version") == "1.0.0", "Invalid schema version")
    check(data.get("annotation_status") in {"unverified", "mixed_review", "independently_reviewed"}, "Invalid dataset review status")
    repositories = data.get("repositories", [])
    repos = {r["repo_id"]: r for r in repositories}
    check(len(repos) == len(repositories), "Duplicate repository ID")
    index_list = data.get("indexes", [])
    indexes = {i["index_id"]: i for i in index_list}
    check(len(indexes) == len(index_list), "Duplicate index ID")
    for repo in repositories:
        check(bool(re.fullmatch(r"[0-9a-f]{40}", repo.get("revision", ""))), f"Invalid revision: {repo['repo_id']}")
        check(repo.get("split") in {"unassigned", "development", "test"}, f"Invalid split: {repo['repo_id']}")
    split_by_revision = {}
    for repo in repositories:
        if repo.get("split") in {"development", "test"}:
            split_by_revision.setdefault(repo.get("revision"), set()).add(repo["split"])
    check(all(len(splits) == 1 for splits in split_by_revision.values()), "Same repository revision occurs in development and test")
    for index in index_list:
        repo = repos.get(index.get("repo_id"))
        check(repo is not None and index.get("revision") == repo.get("revision"), f"Index repository revision mismatch: {index['index_id']}")
        check(bool(index.get("artifact_sha256")) and all(re.fullmatch(r"[0-9a-f]{64}", h) for h in index["artifact_sha256"].values()), f"Invalid index hashes: {index['index_id']}")

    def review(obj, name):
        r = obj.get("review", {})
        status, reviewers, decisions = r.get("status"), r.get("reviewers", []), r.get("decisions", [])
        check(status in {"unverified", "accepted", "rejected", "needs_adjudication"}, f"Invalid review status: {name}")
        if status == "unverified": check(not reviewers and not decisions, f"Unverified record has review claims: {name}")
        if status == "accepted":
            accepted = {d.get("reviewer") for d in decisions if d.get("decision") == "accept" and d.get("timestamp") and d.get("reason")}
            check(len(accepted) >= 2 and accepted.issubset(set(reviewers)), f"Accepted record lacks two independent reviewer decisions: {name}")
            if any(d.get("decision") in {"reject", "revise"} for d in decisions):
                adjudication = r.get("adjudication", {})
                adjudicator = adjudication.get("reviewer")
                check(adjudicator in accepted and adjudication.get("reason") and adjudication.get("timestamp")
                      and not any(d.get("reviewer") == adjudicator and d.get("decision") in {"reject", "revise"} for d in decisions),
                      f"Conflicting decisions lack adjudication: {name}")

    tasks_list = data.get("tasks", [])
    tasks = {t["task_id"]: t for t in tasks_list}
    check(len(tasks) == len(tasks_list), "Duplicate task ID")
    source_paths = {}
    seen_evidence = set()
    for task in tasks_list:
        tid = task["task_id"]
        check(task.get("repo_id") in repos, f"Unknown repository: {tid}")
        review(task, tid)
        check(bool(task.get("evidence")), f"No evidence: {tid}")
        for evidence in task.get("evidence", []):
            eid = evidence["evidence_id"]
            check(eid not in seen_evidence, f"Duplicate evidence ID: {eid}")
            seen_evidence.add(eid)
            path = PurePosixPath(evidence["file"])
            check(not path.is_absolute() and ".." not in path.parts and "\\" not in evidence["file"], f"Unsafe evidence path: {eid}")
            check(1 <= evidence["line_start"] <= evidence["line_end"], f"Invalid evidence range: {eid}")
            check(evidence.get("role") in {"answer", "context"}, f"Invalid evidence role: {eid}")
            review(evidence, eid)
            coverage = evidence["coverage"]
            status, manifest, chunks = coverage.get("status"), coverage.get("index_manifest"), coverage.get("matching_chunks", [])
            check(status in {"not_checked", "present", "absent", "unknown"}, f"Invalid coverage: {eid}")
            if status == "not_checked": check(manifest is None and not chunks, f"Unchecked coverage has results: {eid}")
            if status in {"present", "absent"}:
                check(manifest in indexes, f"Checked coverage missing manifest: {eid}")
                if manifest in indexes:
                    check(indexes[manifest]["repo_id"] == task["repo_id"], f"Coverage belongs to another repository: {eid}")
                check(bool(chunks) == (status == "present"), f"Coverage and matching chunks disagree: {eid}")
            source_paths.setdefault(task["repo_id"], set()).add(evidence["file"])

    seen_queries, seen_pairs, pair_tasks = set(), set(), {}
    for query in data.get("queries", []):
        qid = query["query_id"]
        check(qid not in seen_queries, f"Duplicate query: {qid}")
        seen_queries.add(qid)
        check(query["task_id"] in tasks, f"Unknown task: {qid}")
        pair = (query["pair_id"], query["category"])
        check(pair not in seen_pairs, f"Duplicate pair category: {qid}")
        seen_pairs.add(pair)
        check(pair_tasks.setdefault(query["pair_id"], query["task_id"]) == query["task_id"], f"Pair crosses tasks: {qid}")
        review(query, qid)
        text = query["text"]
        last = 0
        languages, symbols = set(), []
        for span in query["spans"]:
            start, end, label = span["start"], span["end"], span["label"]
            check(0 <= start < end <= len(text) and start >= last and text[start:end] == span["text"], f"Invalid span: {qid}")
            last = end
            check(label in {"en", "fr", "zh", "und", "code"}, f"Invalid span label: {qid}")
            if label in {"en", "fr", "zh"}: languages.add(label)
            if label == "code": symbols.append(span["text"])
        check(languages == set(query["category"].split("-")), f"Category and spans disagree: {qid}")
        check(symbols == query["exact_symbols"], f"Symbol mentions and spans disagree: {qid}")
        check(query["identifier_presence"] in {"present", "absent", "ambiguous"}, f"Invalid identifier status: {qid}")
        if query["identifier_presence"] == "present": check(bool(symbols), f"No identifier mention: {qid}")
        if query["identifier_presence"] == "absent": check(not symbols, f"Unexpected identifier mention: {qid}")

    all_records = tasks_list + [e for task in tasks_list for e in task["evidence"]] + data.get("queries", [])
    if data.get("annotation_status") == "independently_reviewed":
        check(all(record.get("review", {}).get("status") == "accepted" for record in all_records), "Dataset asserts review but records are not all accepted")
    if data.get("annotation_status") == "unverified":
        check(all(record.get("review", {}).get("status") == "unverified" for record in all_records), "Unverified dataset contains reviewed records")
    if verify_local_source:
        roots = {"micrograd": MICROGRAD, "reposage": ROOT}
        for repo_id, paths in source_paths.items():
            try:
                root = roots[repo_id]
                check(git_revision(root, sorted(paths)) == repos[repo_id]["revision"], f"Source revision mismatch: {repo_id}")
                for task in (t for t in tasks_list if t["repo_id"] == repo_id):
                    for e in task["evidence"]:
                        check(line_range(root, e["file"], e["symbol"]) == (e["line_start"], e["line_end"]), f"Source range mismatch: {e['evidence_id']}")
            except Exception as error:
                errors.append(f"Source verification failed: {repo_id}: {type(error).__name__}")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--verify-local-source", action="store_true")
    args = parser.parse_args()
    errors = validate(json.loads(args.dataset.read_text(encoding="utf-8")), args.verify_local_source)
    if errors:
        for error in errors: print(error)
        raise SystemExit(1)
    print("Cross-record validation passed; this is not human review.")


if __name__ == "__main__": main()
