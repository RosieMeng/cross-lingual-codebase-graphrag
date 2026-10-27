import logging
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from reposage.query.multilingual_query import (
    MultilingualQueryPlanner, PlannerConfig, detect_language, protected_spans,
)


@pytest.mark.parametrize("query,language", [
    ("How does Value.backward compute gradients?", "en"),
    ("Comment Value.backward calcule les gradients ?", "fr"),
    ("Value.backward 如何计算梯度？", "zh"),
    ("How does authenticate_user work et comment fonctionne la connexion ?", "mixed"),
    ("如何使用 authenticate_user in the code?", "mixed"),
    ("authenticate_user Value.backward src/auth.py", "unknown"),
    ("login", "unknown"), ("如何", "unknown"), ("", "unknown"),
    ("😀\u0301", "unknown"),
])
def test_language_detection(query, language):
    assert detect_language(query).language == language


def test_ordinary_words_are_not_treated_as_identifiers():
    spans = protected_spans("How does authentication work with Value.backward and authenticate_user in src/auth.py?")
    assert spans == ("Value.backward", "authenticate_user", "src/auth.py")
    assert "authentication" not in spans


def test_paths_quotes_repetition_and_french_apostrophes():
    q = "Comment l'authentification dans src/auth et ./README utilise «foo bar» et authenticate_user avec authenticate_user ?"
    assert protected_spans(q) == ("src/auth", "./README", "«foo bar»", "authenticate_user", "authenticate_user")
    planner = MultilingualQueryPlanner(PlannerConfig(enabled=True), lambda _: "How does REPOSAGEPROTECTED3TOKEN use REPOSAGEPROTECTED0TOKEN and REPOSAGEPROTECTED1TOKEN with REPOSAGEPROTECTED2TOKEN and REPOSAGEPROTECTED4TOKEN?")
    plan = planner.plan(q)
    assert plan.status == "rewritten"
    assert plan.variants[1].text.count("authenticate_user") == 2


def test_contextual_variant_that_loses_identifier_is_not_used():
    original = "Comment fonctionne authenticate_user dans le code ?"
    planner = MultilingualQueryPlanner(PlannerConfig(enabled=True), lambda _: "How does REPOSAGEPROTECTED0TOKEN work?")
    plan = planner.plan(original, "How does authentication work?")
    assert [v.provenance for v in plan.variants] == ["original", "english_rewrite"]


def test_rewrite_masks_and_restores_spans_and_preserves_provenance():
    seen = []
    def rewrite(query):
        seen.append(query)
        return "How does REPOSAGEPROTECTED0TOKEN use REPOSAGEPROTECTED1TOKEN in REPOSAGEPROTECTED2TOKEN with REPOSAGEPROTECTED3TOKEN?"
    original = 'Comment Value.backward utilise authenticate_user dans src/auth.py avec `x == 1` ?'
    plan = MultilingualQueryPlanner(PlannerConfig(enabled=True), rewrite).plan(original)
    assert plan.status == "rewritten"
    assert plan.variants[0].text == original
    assert [v.provenance for v in plan.variants] == ["original", "english_rewrite"]
    assert plan.variants[1].language == "en"
    assert all(span not in seen[0] for span in plan.protected_spans)
    assert all(span in plan.variants[1].text for span in plan.protected_spans)


@pytest.mark.parametrize("output", [
    "How does it work?", "How does REPOSAGEPROTECTED0TOKEN and REPOSAGEPROTECTED0TOKEN work?",
    "How does REPOSAGEPROTECTED1TOKEN work?", "How does REPOSAGEPROTECTED0TOK work?",
    "How does REPOSAGEPROTECTED0TOKEN work?\nextra", "",
    "Comment fonctionne REPOSAGEPROTECTED0TOKEN dans le code ?",
    "Comment REPOSAGEPROTECTED0TOKEN calculates the gradients by traversing the computation graph?",
    "How to traverse and compute gradient over a computational graph?",
])
def test_invalid_rewrite_falls_back(output):
    original = "Comment fonctionne authenticate_user dans le code ?"
    plan = MultilingualQueryPlanner(PlannerConfig(enabled=True), lambda _: output).plan(original)
    assert len(plan.variants) == 1
    assert plan.variants[0].text == original
    assert plan.status in {"rewrite_invalid", "rewrite_not_english"}


@pytest.mark.parametrize("error", [TimeoutError, RuntimeError])
def test_rewriter_failure_falls_back_without_logging_content(error, caplog):
    def fail(_): raise error("secret query")
    with caplog.at_level(logging.INFO):
        plan = MultilingualQueryPlanner(PlannerConfig(enabled=True), fail).plan("Comment fonctionne authenticate_user dans le code ?")
    assert plan.status == "rewrite_failed"
    assert len(plan.variants) == 1
    assert "secret" not in caplog.text


def test_abstention_disable_limits_and_deduplication():
    def forbidden(_): raise AssertionError("must not call rewriter")
    assert MultilingualQueryPlanner(PlannerConfig(), forbidden).plan("Comment fonctionne le code ?").status == "disabled"
    assert MultilingualQueryPlanner(PlannerConfig(enabled=True), forbidden).plan("authenticate_user").status == "abstained"
    assert MultilingualQueryPlanner(PlannerConfig(enabled=True), forbidden).plan("").status == "empty"
    assert MultilingualQueryPlanner(PlannerConfig(enabled=True, max_input_chars=5), forbidden).plan("Comment fonctionne le code ?").status == "input_too_long"
    planner = MultilingualQueryPlanner(PlannerConfig(enabled=True), lambda _: "How does it work?")
    plan = planner.plan("Comment fonctionne le code ?", "How does it work?")
    assert plan.status == "rewrite_duplicate"
    assert [v.provenance for v in plan.variants] == ["original", "contextual"]


def test_unicode_and_windows_paths_preserved():
    q = '如何在 src/模块.py 和 C:\\repo\\auth.py 中调用 authenticate_user？'
    planner = MultilingualQueryPlanner(PlannerConfig(enabled=True), lambda _: "How does REPOSAGEPROTECTED2TOKEN work in REPOSAGEPROTECTED0TOKEN and REPOSAGEPROTECTED1TOKEN?")
    plan = planner.plan(q)
    assert plan.status == "rewritten"
    assert plan.protected_spans == ("src/模块.py", "C:\\repo\\auth.py", "authenticate_user")
    assert all(s in plan.variants[1].text for s in plan.protected_spans)


def test_rewriter_timeout_is_configured(monkeypatch):
    import langchain_ollama
    from types import SimpleNamespace
    from reposage.query.multilingual_query import ollama_rewriter
    calls = {}
    class Fake:
        def __init__(self, **kwargs): calls.update(kwargs)
        def invoke(self, prompt, **kwargs):
            calls["invoke_kwargs"] = kwargs
            return SimpleNamespace(content="result")
    monkeypatch.setattr(langchain_ollama, "ChatOllama", Fake)
    rewrite = ollama_rewriter(PlannerConfig(enabled=True, timeout_seconds=3, max_output_tokens=80), "http://localhost", "model")
    assert rewrite("text") == "result"
    assert calls["client_kwargs"]["timeout"] == 3
    assert calls["num_predict"] == 80
    assert calls["invoke_kwargs"]["think"] is False
