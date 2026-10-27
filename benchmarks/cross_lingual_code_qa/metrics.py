"""Retrieval-only metrics for ranked code evidence."""

from collections.abc import Iterable, Mapping


def hit_at_k(ranked_results: Iterable[Mapping], field: str, relevant: str, k: int) -> float:
    """Return 1 when a relevant field value occurs within the first k results."""
    if k < 1:
        raise ValueError("k must be at least 1")
    return float(any(result.get(field) == relevant for result in list(ranked_results)[:k]))


def reciprocal_rank(ranked_results: Iterable[Mapping], relevant: Mapping) -> float:
    """Return reciprocal rank for the first result matching file and function."""
    for rank, result in enumerate(ranked_results, start=1):
        if _matches_function(result, relevant):
            return 1.0 / rank
    return 0.0


def function_hit_at_k(ranked_results: Iterable[Mapping], relevant: Mapping, k: int) -> float:
    """Return 1 when the gold file and function occur together within k."""
    if k < 1:
        raise ValueError("k must be at least 1")
    return float(any(
        _matches_function(result, relevant)
        for result in list(ranked_results)[:k]
    ))


def _matches_function(result: Mapping, relevant: Mapping) -> bool:
    return (
        result.get("file") == relevant.get("file")
        and result.get("function") == relevant.get("function")
        and (
            "line_start" not in relevant
            or result.get("line_start") == relevant["line_start"]
        )
    )


def query_metrics(ranked_results: list[Mapping], gold: Mapping, cutoffs=(1, 3, 5, 10)) -> dict:
    """Calculate file/function Hit@k and joint-target reciprocal rank."""
    metrics = {}
    for cutoff in cutoffs:
        metrics[f"file_hit@{cutoff}"] = hit_at_k(ranked_results, "file", gold["file"], cutoff)
        metrics[f"function_hit@{cutoff}"] = function_hit_at_k(ranked_results, gold, cutoff)
    metrics["mrr"] = reciprocal_rank(ranked_results, gold)
    return metrics


def aggregate_metrics(rows: list[Mapping], language: str | None = None) -> dict:
    """Macro-average query metrics, optionally restricted to one language."""
    selected = [row for row in rows if language is None or row["language"] == language]
    if not selected:
        return {"query_count": 0}

    metric_names = selected[0]["metrics"].keys()
    result = {"query_count": len(selected)}
    for name in metric_names:
        result[name] = sum(row["metrics"][name] for row in selected) / len(selected)
    return result