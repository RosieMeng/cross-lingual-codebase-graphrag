"""Experimental fixed-budget routing over the existing multilingual planner."""
from dataclasses import replace
import re

from .multilingual_query import (
    MultilingualQueryPlanner, QueryPlan, QuerySpan, RetrievalRequest,
    _EN, _FR, _HAN, _PROTECTED, _WORDS,
)


def query_spans(text: str) -> tuple[QuerySpan, ...]:
    """Annotate only locally supported evidence; offsets refer to original text.

    Han is a script observation, not a claim that a span is Chinese. Unmarked
    words, including ambiguous bare symbols, are intentionally left unlabeled.
    """
    spans = []
    occupied = []
    for match in _PROTECTED.finditer(text):
        value = match.group()
        if value[0] in "`\"'“«":
            kind = "quoted"
        elif "/" in value or "\\" in value or re.search(r"\.[A-Za-z0-9_-]+$", value) and value.split(".")[-1].lower() in {"py", "js", "ts", "tsx", "json", "md", "java", "go", "rs"}:
            kind = "path"
        elif value.isupper() and "_" not in value:
            kind = "technical"
        else:
            kind = "identifier"
        spans.append(QuerySpan(match.start(), match.end(), kind, kind))
        occupied.append((match.start(), match.end()))

    def free(start, end):
        return not any(start < b and end > a for a, b in occupied)

    for match in re.finditer(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\U00020000-\U0002fa1f]+", text):
        if free(match.start(), match.end()):
            spans.append(QuerySpan(match.start(), match.end(), "language", "han_script"))
    for match in _WORDS.finditer(text):
        if not free(match.start(), match.end()) or _HAN.search(match.group()):
            continue
        word = match.group().casefold()
        label = "en" if word in _EN - _FR else "fr" if word in _FR - _EN else None
        if label:
            spans.append(QuerySpan(match.start(), match.end(), "language", label))
    return tuple(sorted(spans, key=lambda span: (span.start, span.end)))


class CodeAwareQueryPlanner:
    """Route raw lexical tokens and a validated semantic variant, one call each."""
    def __init__(self, multilingual: MultilingualQueryPlanner):
        self.multilingual = multilingual

    def plan(self, question: str, candidate_limit: int, contextual_question: str | None = None) -> QueryPlan:
        if candidate_limit < 1:
            raise ValueError("candidate_limit must be positive")
        plan = self.multilingual.plan(question, contextual_question)
        # Prefer the validated English translation; otherwise keep a safe
        # contextual query, then the original. BM25 always sees original bytes
        # and therefore uses the existing whitespace tokenizer unchanged.
        semantic = next((i for i, v in enumerate(plan.variants) if v.provenance == "english_rewrite"), None)
        if semantic is None:
            semantic = next((i for i, v in enumerate(plan.variants) if v.provenance == "contextual"), 0)
        return replace(plan, spans=query_spans(question), requests=(
            RetrievalRequest("vector", semantic, candidate_limit),
            RetrievalRequest("bm25", 0, candidate_limit),
        ))
