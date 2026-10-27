import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from reposage.query.agent_engine import CodeAgentEngine
from reposage.ingestion.repo_indexer import RepoIndexer
from reposage.security import normalize_repo_relative_path, resolve_repo_file, validate_repo_name


@pytest.mark.parametrize("repo_name", ["../outside", "..", "folder/name", "folder\\name", "CON", "repo."])
def test_rejects_unsafe_repository_names(repo_name):
    with pytest.raises(ValueError):
        validate_repo_name(repo_name)


def test_indexer_rejects_invalid_repository_name_before_cloning():
    indexer = RepoIndexer.__new__(RepoIndexer)

    with pytest.raises(ValueError, match="Invalid repository name"):
        indexer.index_repo("https://github.com/example/../")


@pytest.mark.parametrize(
    "file_path",
    [
        "../secret.txt",
        "..\\secret.txt",
        "src/../../secret.txt",
        "C:\\secret.txt",
        "\\\\server\\share\\secret.txt",
        "/etc/passwd",
    ],
)
def test_rejects_absolute_and_traversal_file_paths(tmp_path, file_path):
    with pytest.raises(ValueError):
        resolve_repo_file(tmp_path, "sample", file_path)


def test_reads_existing_file_inside_repository(tmp_path):
    repo = tmp_path / "sample"
    source = repo / "src" / "module.py"
    source.parent.mkdir(parents=True)
    source.write_text("value = 42\n", encoding="utf-8")

    engine = CodeAgentEngine.__new__(CodeAgentEngine)
    engine.repo_name = "sample"
    engine.clone_dir = str(tmp_path)
    engine.rag_engine = type("RagStub", (), {"bm25_retriever": None})()

    assert engine._read_file("src/module.py") == "value = 42\n"


@pytest.mark.parametrize("file_path", ["../secret.txt", "..\\secret.txt"])
def test_agent_read_file_rejects_traversal(tmp_path, file_path):
    outside_file = tmp_path.parent / "secret.txt"
    outside_file.write_text("must not be returned", encoding="utf-8")

    engine = CodeAgentEngine.__new__(CodeAgentEngine)
    engine.repo_name = "sample"
    engine.clone_dir = str(tmp_path)
    engine.rag_engine = type("RagStub", (), {"bm25_retriever": None})()

    result = engine._read_file(file_path)

    assert result == "Error: Invalid file path."
    assert "must not be returned" not in result


def test_normalizes_windows_separators_for_repository_relative_paths():
    assert normalize_repo_relative_path(r"src\module.py") == "src/module.py"


def test_symlink_escape_is_rejected(tmp_path):
    repo = tmp_path / "sample"
    repo.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    try:
        (repo / "link.txt").symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("File symlinks unavailable")
    with pytest.raises(ValueError):
        resolve_repo_file(tmp_path, "sample", "link.txt")


def test_repository_symlink_escape_is_rejected(tmp_path):
    clones = tmp_path / "clones"
    clones.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        (clones / "sample").symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("Directory symlinks unavailable")
    with pytest.raises(ValueError):
        resolve_repo_file(clones, "sample", "file.py")


def test_agent_logs_do_not_include_content_or_tool_arguments(monkeypatch, caplog):
    import logging
    from langchain_core.messages import AIMessage
    import reposage.query.agent_engine as module

    secret = "private_prompt_and_path"
    class Model:
        def __init__(self, **kwargs): self.calls = 0
        def bind_tools(self, tools): return self
        def invoke(self, messages):
            self.calls += 1
            if self.calls == 1:
                return AIMessage(content=secret, tool_calls=[{"name": "read_file", "args": {"file_path": secret}, "id": "1"}])
            return AIMessage(content=secret)
    class Rag:
        def __init__(self, **kwargs): self.bm25_retriever = None
    monkeypatch.setattr(module, "ChatOllama", Model)
    monkeypatch.setattr(module, "CodeRAGEngine", Rag)
    with caplog.at_level(logging.INFO):
        engine = module.CodeAgentEngine("sample", clone_dir="missing")
        engine.query(secret)
    assert secret not in caplog.text
    assert "tool=read_file" in caplog.text
    assert "latency_ms=" in caplog.text
