"""
Tools Module - Agent Tool Definitions

Implements:
  - Tool Use / Function Calling: The ability of the LLM to understand
    and execute external APIs, SQL databases, or web search functions.
  - Web Search Tool
  - SQL Database Query Tool
  - File System Tool
  - Calculator Tool
  - Document Ingestion Tool
"""

import os
import json
import sqlite3
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Base Tool
# ─────────────────────────────────────────────

class BaseTool:
    """
    Base class for all agent tools.
    Tool Use / Function Calling pattern.
    """
    
    name: str = "base_tool"
    description: str = "Base tool"
    requires_hitl: bool = False  # Whether this tool needs Human-in-the-Loop approval
    
    def execute(self, **kwargs) -> Dict:
        """Execute the tool and return results."""
        raise NotImplementedError
    
    def get_schema(self) -> Dict:
        """Get the tool schema for function calling."""
        return {
            "name": self.name,
            "description": self.description,
            "requires_hitl": self.requires_hitl,
        }


# ─────────────────────────────────────────────
# Web Search Tool
# ─────────────────────────────────────────────

class WebSearchTool(BaseTool):
    """
    Web Search Tool:
    Simulates web search functionality.
    In production, integrate with DuckDuckGo, SerpAPI, or Tavily.
    """
    
    name = "web_search"
    description = "Search the web for information. Input: search query string."
    requires_hitl = False
    
    def execute(self, query: str = "", **kwargs) -> Dict:
        """Execute web search."""
        try:
            # Try using DuckDuckGo search if available
            try:
                from duckduckgo_search import DDGS
                with DDGS() as ddgs:
                    results = list(ddgs.text(query, max_results=5))
                return {
                    "status": "success",
                    "tool": self.name,
                    "query": query,
                    "results": [
                        {"title": r.get("title", ""), "snippet": r.get("body", ""), "url": r.get("href", "")}
                        for r in results
                    ],
                }
            except ImportError:
                # Fallback: return a structured message about the search
                return {
                    "status": "success",
                    "tool": self.name,
                    "query": query,
                    "results": [{
                        "title": f"Search results for: {query}",
                        "snippet": f"Web search executed for '{query}'. Install duckduckgo-search for live results.",
                        "url": "",
                    }],
                    "note": "Using fallback. Install duckduckgo-search for live results.",
                }
        except Exception as e:
            return {"status": "error", "tool": self.name, "error": str(e)}


# ─────────────────────────────────────────────
# SQL Database Query Tool
# ─────────────────────────────────────────────

class SQLQueryTool(BaseTool):
    """
    SQL Database Query Tool:
    Execute SQL queries against a SQLite database.
    Requires Human-in-the-Loop approval for safety.
    """
    
    name = "sql_query"
    description = "Execute SQL queries against a database. Input: SQL query string."
    requires_hitl = True  # Critical action
    
    def __init__(self, db_path: str = "data/tool_database.db"):
        self.db_path = db_path
        self._ensure_db()
    
    def _ensure_db(self):
        """Create a sample database if it doesn't exist."""
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        if not os.path.exists(self.db_path):
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create a sample table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY,
                    title TEXT,
                    category TEXT,
                    content TEXT,
                    created_at TEXT
                )
            """)
            
            # Insert sample data
            sample_data = [
                (1, "RAG Architecture Guide", "technical", "Comprehensive guide to RAG systems.", "2024-01-15"),
                (2, "Agent Safety Protocols", "security", "Safety guidelines for autonomous agents.", "2024-02-20"),
                (3, "Vector DB Benchmarks", "technical", "Performance benchmarks for vector databases.", "2024-03-10"),
            ]
            cursor.executemany(
                "INSERT OR IGNORE INTO documents VALUES (?, ?, ?, ?, ?)",
                sample_data
            )
            
            conn.commit()
            conn.close()
    
    def execute(self, query: str = "", approved: bool = False, **kwargs) -> Dict:
        """Execute SQL query."""
        try:
            query_upper = query.upper().strip()
            if not approved:
                return {
                    "status": "blocked",
                    "tool": self.name,
                    "error": "SQL execution requires Human-in-the-Loop approval.",
                }
            if not query_upper.startswith("SELECT"):
                return {
                    "status": "blocked",
                    "tool": self.name,
                    "error": "Only SELECT queries are allowed by default.",
                }
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(query)
            
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            rows = cursor.fetchall()
            conn.close()
            return {
                "status": "success",
                "tool": self.name,
                "query": query,
                "columns": columns,
                "rows": rows,
                "row_count": len(rows),
            }
        except Exception as e:
            return {"status": "error", "tool": self.name, "error": str(e)}


# ─────────────────────────────────────────────
# File Operations Tool
# ─────────────────────────────────────────────

class FileOperationsTool(BaseTool):
    """
    File System Tool:
    Read and list files from the document directory.
    Write operations require HITL approval.
    """
    
    name = "file_operations"
    description = "Read files and list directory contents. Input: file path or directory path."
    requires_hitl = False  # Read is safe; write would require HITL
    
    def __init__(self, base_dir: str = "data/documents"):
        self.base_dir = os.path.abspath(base_dir)
        os.makedirs(base_dir, exist_ok=True)
    
    def execute(self, action: str = "list", path: str = "", **kwargs) -> Dict:
        """Execute file operation."""
        try:
            if action == "list":
                return self._list_directory(path or self.base_dir)
            elif action == "read":
                return self._read_file(path)
            else:
                return {"status": "error", "tool": self.name, "error": f"Unknown action: {action}"}
        except Exception as e:
            return {"status": "error", "tool": self.name, "error": str(e)}
    
    def _list_directory(self, directory: str) -> Dict:
        """List directory contents."""
        directory = self._resolve_path(directory)
        if not os.path.exists(directory):
            return {"status": "error", "tool": self.name, "error": f"Directory not found: {directory}"}
        
        files = []
        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)
            files.append({
                "name": item,
                "is_directory": os.path.isdir(item_path),
                "size": os.path.getsize(item_path) if os.path.isfile(item_path) else 0,
            })
        
        return {
            "status": "success",
            "tool": self.name,
            "directory": directory,
            "files": files,
            "count": len(files),
        }
    
    def _read_file(self, file_path: str) -> Dict:
        """Read file contents."""
        file_path = self._resolve_path(file_path)
        if not os.path.exists(file_path):
            return {"status": "error", "tool": self.name, "error": f"File not found: {file_path}"}
        
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read(10000)  # Limit to 10k chars
        
        return {
            "status": "success",
            "tool": self.name,
            "file_path": file_path,
            "content": content,
            "size": os.path.getsize(file_path),
        }

    def _resolve_path(self, path: str) -> str:
        if not path:
            candidate = self.base_dir
        elif os.path.isabs(path):
            candidate = os.path.abspath(path)
        else:
            candidate = os.path.abspath(os.path.join(self.base_dir, path))
        base = os.path.abspath(self.base_dir)
        if os.path.commonpath([candidate, base]) != base:
            raise PermissionError(f"Path outside allowed document directory: {path}")
        return candidate


# ─────────────────────────────────────────────
# Calculator Tool
# ─────────────────────────────────────────────

class CalculatorTool(BaseTool):
    """
    Calculator Tool:
    Perform mathematical calculations.
    """
    
    name = "calculator"
    description = "Perform mathematical calculations. Input: mathematical expression string."
    requires_hitl = False
    
    def execute(self, expression: str = "", **kwargs) -> Dict:
        """Evaluate a mathematical expression."""
        try:
            # Safe evaluation (only allow math operations)
            allowed_chars = set("0123456789+-*/.() ")
            if not all(c in allowed_chars for c in expression):
                return {
                    "status": "error",
                    "tool": self.name,
                    "error": "Expression contains disallowed characters.",
                }
            
            result = eval(expression, {"__builtins__": {}}, {})  # Safe due to char validation
            return {
                "status": "success",
                "tool": self.name,
                "expression": expression,
                "result": result,
            }
        except Exception as e:
            return {"status": "error", "tool": self.name, "error": str(e)}


# ─────────────────────────────────────────────
# Document Retrieval Tool (connects to RAG)
# ─────────────────────────────────────────────

class DocumentRetrievalTool(BaseTool):
    """
    Document Retrieval Tool:
    Searches the vector database for relevant documents.
    Integrates with the Hybrid Search engine.
    """
    
    name = "document_search"
    description = "Search indexed documents using hybrid search (keyword + semantic). Input: search query."
    requires_hitl = False
    
    def __init__(self, hybrid_search=None, context_manager=None):
        self.hybrid_search = hybrid_search
        self.context_manager = context_manager
    
    def execute(self, query: str = "", top_k: int = 5, **kwargs) -> Dict:
        """Search documents."""
        if self.hybrid_search is None:
            return {
                "status": "error",
                "tool": self.name,
                "error": "Hybrid search engine not initialized.",
            }
        
        try:
            results = self.hybrid_search.search(query, top_k=top_k)
            
            formatted_results = []
            for chunk, score in results:
                formatted_results.append({
                    "content": chunk.content[:500],
                    "score": score,
                    "source": chunk.source_file,
                    "chunk_id": chunk.chunk_id,
                })
            
            # Compress context if manager available
            context = ""
            if self.context_manager:
                context = self.context_manager.compress_context(results, query)
            
            return {
                "status": "success",
                "tool": self.name,
                "query": query,
                "results": formatted_results,
                "result_count": len(formatted_results),
                "compressed_context": context,
            }
        except Exception as e:
            return {"status": "error", "tool": self.name, "error": str(e)}


# ─────────────────────────────────────────────
# Tool Registry
# ─────────────────────────────────────────────

class ToolRegistry:
    """
    Tool Use / Function Calling:
    Registry of all available tools that the agent can use.
    Supports dynamic tool registration and schema generation.
    """
    
    def __init__(self):
        self.tools: Dict[str, BaseTool] = {}
    
    def register(self, tool: BaseTool):
        """Register a tool."""
        self.tools[tool.name] = tool
        logger.info(f"ToolRegistry: Registered tool '{tool.name}'")
    
    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Get a tool by name."""
        return self.tools.get(name)
    
    def execute_tool(self, name: str, approved: bool = False, **kwargs) -> Dict:
        """Execute a tool by name."""
        tool = self.get_tool(name)
        if not tool:
            return {"status": "error", "error": f"Tool '{name}' not found."}
        if tool.requires_hitl and not approved:
            return {
                "status": "blocked",
                "tool": name,
                "error": f"Tool '{name}' requires Human-in-the-Loop approval.",
            }
        return tool.execute(approved=approved, **kwargs)
    
    def get_all_schemas(self) -> List[Dict]:
        """Get schemas for all registered tools (for function calling)."""
        return [tool.get_schema() for tool in self.tools.values()]
    
    def get_tools_description(self) -> str:
        """Get a formatted description of all tools (for system prompt)."""
        parts = ["Available tools:"]
        for tool in self.tools.values():
            hitl_marker = " [REQUIRES APPROVAL]" if tool.requires_hitl else ""
            parts.append(f"- {tool.name}: {tool.description}{hitl_marker}")
        return "\n".join(parts)


# ─────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────

def create_tool_registry(hybrid_search=None, context_manager=None) -> ToolRegistry:
    """Create the tool registry with all default tools."""
    registry = ToolRegistry()
    
    registry.register(WebSearchTool())
    registry.register(SQLQueryTool())
    registry.register(FileOperationsTool())
    registry.register(CalculatorTool())
    
    if hybrid_search:
        registry.register(DocumentRetrievalTool(hybrid_search, context_manager))
    
    return registry
