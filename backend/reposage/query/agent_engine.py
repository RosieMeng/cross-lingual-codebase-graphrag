import os
import logging
import time
from typing import TypedDict, Annotated, Sequence, Optional
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from .rag_engine import CodeRAGEngine
from ..security import normalize_repo_relative_path, resolve_repo_file, validate_repo_name

logger = logging.getLogger(__name__)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    repo_name: str
    
class CodeAgentEngine:
    """Agentic query engine using LangGraph and tool calling."""
    def __init__(
        self,
        repo_name: str,
        persist_dir: str = "./chroma_db",
        clone_dir: str = "./tmp/reposage",
        ollama_base_url: str = "http://localhost:11434",
        ollama_model: str = "qwen2.5-coder:7b", # Best for tools
        ollama_embed_model: str = "nomic-embed-text",
    ):
        self.repo_name = validate_repo_name(repo_name)
        self.persist_dir = persist_dir
        self.clone_dir = clone_dir
        
        self.rag_engine = CodeRAGEngine(
            repo_name=repo_name,
            persist_dir=persist_dir,
            ollama_base_url=ollama_base_url,
            ollama_model=ollama_model,
            ollama_embed_model=ollama_embed_model,
        )
        
        self.llm = ChatOllama(
            base_url=ollama_base_url,
            model=ollama_model,
            temperature=0,
        )
        
        # Define Tools
        @tool
        def semantic_search(query: str) -> str:
            """Search the codebase for code snippets relevant to the query. Returns matching files and code."""
            result = self.rag_engine.query(query, k=5)
            return result["answer"] + "\n\nSources: " + ", ".join(result["sources"])

        @tool
        def read_file(file_path: str) -> str:
            """Read a repository-relative file without leaving the repository."""
            return self._read_file(file_path)

        self.tools = [semantic_search, read_file]
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        
        # Build LangGraph
        workflow = StateGraph(AgentState)
        
        def call_model(state: AgentState):
            messages = state["messages"]

            started = time.perf_counter()
            try:
                response = self.llm_with_tools.invoke(messages)
            except Exception as e:
                logger.warning("Agent model failed: error_type=%s latency_ms=%.1f", type(e).__name__, (time.perf_counter() - started) * 1000)
                raise
            logger.info("Agent model completed: tool_count=%d latency_ms=%.1f", len(response.tool_calls), (time.perf_counter() - started) * 1000)

            return {"messages": [response]}
            
        def call_tools(state: AgentState):
            messages = state["messages"]
            last_message = messages[-1]
            
            tool_messages = []
            for tool_call in last_message.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                
                safe_name = tool_name if tool_name in {"semantic_search", "read_file"} else "unknown"
                started = time.perf_counter()
                try:
                    if tool_name == "semantic_search":
                        result = semantic_search.invoke(tool_args)
                    elif tool_name == "read_file":
                        result = read_file.invoke(tool_args)
                    else:
                        result = "Unknown tool"
                except Exception as e:
                    logger.warning("Agent tool failed: tool=%s error_type=%s latency_ms=%.1f", safe_name, type(e).__name__, (time.perf_counter() - started) * 1000)
                    raise
                logger.info("Agent tool completed: tool=%s latency_ms=%.1f", safe_name, (time.perf_counter() - started) * 1000)
                    
                tool_messages.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))
                
            return {"messages": tool_messages}
            
        def should_continue(state: AgentState) -> str:
            messages = state["messages"]
            last_message = messages[-1]
            if not getattr(last_message, "tool_calls", None):
                return "end"
            return "continue"
            
        workflow.add_node("agent", call_model)
        workflow.add_node("action", call_tools)
        
        workflow.set_entry_point("agent")
        workflow.add_conditional_edges("agent", should_continue, {"continue": "action", "end": END})
        workflow.add_edge("action", "agent")
        
        self.app = workflow.compile()

    def _read_file(self, file_path: str) -> str:
        try:
            normalized_target = normalize_repo_relative_path(file_path)
            full_path = resolve_repo_file(self.clone_dir, self.repo_name, file_path)
        except (OSError, ValueError):
            return "Error: Invalid file path."

        if full_path.is_file():
            try:
                with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                    return f.read()
            except Exception as e:
                return f"Error reading file: {e}"

        if self.rag_engine.bm25_retriever and self.rag_engine.bm25_retriever.docs:
            file_chunks = [
                doc for doc in self.rag_engine.bm25_retriever.docs
                if doc.metadata.get("file", "").replace("\\", "/") == normalized_target
            ]
            if file_chunks:
                file_chunks.sort(key=lambda d: d.metadata.get("line_start", 0))
                reconstructed_content = []
                for doc in file_chunks:
                    content = doc.page_content
                    if content.startswith("// Context: ") or content.startswith("# Context: "):
                        parts = content.split("\n\n", 1)
                        if len(parts) > 1:
                            content = parts[1]
                    reconstructed_content.append(content)
                return "\n\n".join(reconstructed_content)

        return f"Error: Could not read file {file_path}. It might not exist."
        
    def query(self, question: str, k: int = 6, chat_history: Optional[list[dict]] = None) -> dict:
        system_msg = SystemMessage(content=f"You are an autonomous code agent. You have access to tools to search and read the {self.repo_name} codebase. Always use `semantic_search` first. If you need more context, use `read_file`.")
        messages = [system_msg]
        
        if chat_history:
            for msg in chat_history[-6:]:
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                else:
                    messages.append(AIMessage(content=msg["content"]))
                    
        messages.append(HumanMessage(content=question))
        
        try:
            result = self.app.invoke(
                {"messages": messages, "repo_name": self.repo_name},
                config={"recursion_limit": 10}
            )
            final_message = result["messages"][-1].content
            if not final_message or not final_message.strip():
                raise ValueError("Agent returned empty response")
                
            return {
                "answer": final_message,
                "sources": ["Agent Tool Execution"],
                "chunks_used": 0
            }
        except Exception as e:
            logger.warning("Agent execution failed: error_type=%s; falling back to standard RAG", type(e).__name__)
            return self.rag_engine.query(
                question=question,
                k=k,
                chat_history=chat_history
            )
