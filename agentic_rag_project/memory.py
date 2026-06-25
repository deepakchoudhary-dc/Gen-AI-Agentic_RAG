"""
Advanced State Management & Agent Memory Module

Implements:
  - Stateful Agents (memory across conversation turns)
  - Directed Acyclic Graph (DAG) state management
  - Cyclic Graphs (loops for self-correction)
  - Short-Term Memory (current conversation)
  - Long-Term Memory (user profiles across sessions)
  - Persistence Layer (SQLite/JSON)
  - Time-Travel / State Rewinding (debugging)
"""

import os
import json
import time
import copy
import sqlite3
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Short-Term Memory (Current Conversation)
# ─────────────────────────────────────────────

@dataclass
class ConversationTurn:
    """A single turn in the conversation."""
    role: str                           # user, assistant, system
    content: str
    timestamp: float = 0.0
    metadata: Dict = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.time()


class ShortTermMemory:
    """
    Short-Term Memory:
    Stores the current conversation thread.
    Maintains context across multiple turns within a single session.
    """
    
    def __init__(self, max_turns: int = 50):
        self.max_turns = max_turns
        self.conversation: List[ConversationTurn] = []
        self.session_id: str = ""
        self.variables: Dict[str, Any] = {}  # Track variables across turns
    
    def add_turn(self, role: str, content: str, metadata: Dict = None):
        """Add a conversation turn."""
        turn = ConversationTurn(
            role=role,
            content=content,
            metadata=metadata or {},
        )
        self.conversation.append(turn)
        
        # Trim if exceeding max turns
        if len(self.conversation) > self.max_turns:
            self.conversation = self.conversation[-self.max_turns:]
    
    def get_context(self, last_n: int = 10) -> List[Dict]:
        """Get recent conversation context."""
        recent = self.conversation[-last_n:]
        return [
            {"role": t.role, "content": t.content, "timestamp": t.timestamp}
            for t in recent
        ]
    
    def get_formatted_history(self, last_n: int = 10) -> str:
        """Get formatted conversation history for prompt injection."""
        turns = self.get_context(last_n)
        parts = []
        for turn in turns:
            parts.append(f"{turn['role'].upper()}: {turn['content']}")
        return "\n".join(parts)
    
    def set_variable(self, key: str, value: Any):
        """Track a variable across conversation turns."""
        self.variables[key] = value
    
    def get_variable(self, key: str, default: Any = None) -> Any:
        """Get a tracked variable."""
        return self.variables.get(key, default)
    
    def clear(self):
        """Clear the conversation."""
        self.conversation = []
        self.variables = {}
    
    def to_dict(self) -> Dict:
        """Serialize to dict."""
        return {
            "session_id": self.session_id,
            "conversation": [
                {"role": t.role, "content": t.content, "timestamp": t.timestamp, "metadata": t.metadata}
                for t in self.conversation
            ],
            "variables": self.variables,
        }


# ─────────────────────────────────────────────
# Long-Term Memory (User Profiles Across Sessions)
# ─────────────────────────────────────────────

class LongTermMemory:
    """
    Long-Term Memory:
    Stores user preferences and profiles across different sessions.
    Persists between application restarts.
    """
    
    def __init__(self, persist_path: str = "data/long_term_memory.json"):
        self.persist_path = persist_path
        self.user_profiles: Dict[str, Dict] = {}
        self.global_facts: List[Dict] = []
        self.interaction_history: Dict[str, List[Dict]] = {}
        self._load()
    
    def _load(self):
        """Load long-term memory from disk."""
        if os.path.exists(self.persist_path):
            try:
                with open(self.persist_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.user_profiles = data.get("user_profiles", {})
                self.global_facts = data.get("global_facts", [])
                self.interaction_history = data.get("interaction_history", {})
                logger.info(f"Long-Term Memory: Loaded {len(self.user_profiles)} profiles")
            except Exception as e:
                logger.error(f"Error loading long-term memory: {e}")
    
    def save(self):
        """Persist long-term memory to disk."""
        os.makedirs(os.path.dirname(self.persist_path) or ".", exist_ok=True)
        data = {
            "user_profiles": self.user_profiles,
            "global_facts": self.global_facts,
            "interaction_history": self.interaction_history,
            "last_saved": datetime.now().isoformat(),
        }
        with open(self.persist_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    
    def update_user_profile(self, user_id: str, key: str, value: Any):
        """Update a user's preference or profile data."""
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = {"created": datetime.now().isoformat()}
        self.user_profiles[user_id][key] = value
        self.save()
    
    def get_user_profile(self, user_id: str) -> Dict:
        """Get a user's profile."""
        return self.user_profiles.get(user_id, {})
    
    def add_fact(self, fact: str, source: str = "", confidence: float = 1.0):
        """Add a global fact to long-term memory."""
        self.global_facts.append({
            "fact": fact,
            "source": source,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat(),
        })
        self.save()
    
    def record_interaction(self, user_id: str, query: str, response: str,
                            scores: Dict = None):
        """Record an interaction for future learning."""
        if user_id not in self.interaction_history:
            self.interaction_history[user_id] = []
        
        self.interaction_history[user_id].append({
            "query": query,
            "response": response[:500],  # Truncate for storage
            "scores": scores or {},
            "timestamp": datetime.now().isoformat(),
        })
        
        # Keep last 100 interactions per user
        if len(self.interaction_history[user_id]) > 100:
            self.interaction_history[user_id] = self.interaction_history[user_id][-100:]
        
        self.save()


# ─────────────────────────────────────────────
# Persistence Layer (State Checkpointing)
# ─────────────────────────────────────────────

class PersistenceLayer:
    """
    Persistence Layer:
    Saves agent states so a conversation can be resumed later.
    Uses SQLite for local persistence.
    
    Supports Time-Travel / State Rewinding for debugging.
    """
    
    def __init__(self, db_path: str = "data/agent_state.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """Initialize the SQLite database."""
        try:
            conn = self._connect()
            self._create_schema(conn)
        except sqlite3.OperationalError as exc:
            if not self._recover_sqlite_startup_failure(exc):
                raise
            conn = self._connect()
            self._create_schema(conn)
        logger.info(f"Persistence Layer: Initialized at {self.db_path}")

    def _connect(self):
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.execute("PRAGMA journal_mode=MEMORY")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _create_schema(self, conn):
        """Create persistence tables and indexes."""
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS agent_states (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id TEXT NOT NULL,
                step_index INTEGER NOT NULL,
                state_json TEXT NOT NULL,
                node_name TEXT,
                timestamp REAL NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversation_sessions (
                session_id TEXT PRIMARY KEY,
                thread_id TEXT NOT NULL,
                user_id TEXT,
                conversation_json TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_states_thread ON agent_states(thread_id)
        """)
        
        conn.commit()
        conn.close()

    def _recover_sqlite_startup_failure(self, exc: sqlite3.OperationalError) -> bool:
        """
        Recover from stale local SQLite artifacts.

        This is intentionally narrow: it handles startup failures caused by
        half-created local files or orphaned journal/WAL files, which are common
        after interrupted test runs on Windows.
        """
        message = str(exc).lower()
        recoverable = any(
            marker in message
            for marker in ["disk i/o error", "database disk image is malformed", "file is not a database"]
        )
        if not recoverable:
            return False

        stamp = int(time.time() * 1000)
        for suffix in ["", "-journal", "-wal", "-shm"]:
            path = f"{self.db_path}{suffix}"
            if not os.path.exists(path):
                continue
            backup = f"{path}.corrupt-{stamp}"
            try:
                os.replace(path, backup)
                logger.warning(f"Recovered stale SQLite artifact: {path} -> {backup}")
            except OSError as backup_error:
                fallback = f"{self.db_path}.recovered-{stamp}.db"
                logger.warning(
                    "Could not rename stale SQLite artifact %s: %s. "
                    "Using fallback database %s.",
                    path,
                    backup_error,
                    fallback,
                )
                self.db_path = fallback
                return True
        return True
    
    def save_state(self, thread_id: str, step_index: int,
                    state: Dict, node_name: str = ""):
        """Save an agent state checkpoint."""
        conn = self._connect()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO agent_states (thread_id, step_index, state_json, node_name, timestamp, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            thread_id, step_index,
            json.dumps(state, default=str),
            node_name,
            time.time(),
            datetime.now().isoformat(),
        ))
        
        conn.commit()
        conn.close()
        logger.debug(f"State saved: thread={thread_id}, step={step_index}, node={node_name}")
    
    def get_state(self, thread_id: str, step_index: int = -1) -> Optional[Dict]:
        """
        Retrieve a state checkpoint.
        If step_index is -1, returns the latest state.
        """
        conn = self._connect()
        cursor = conn.cursor()
        
        if step_index == -1:
            cursor.execute("""
                SELECT state_json FROM agent_states
                WHERE thread_id = ?
                ORDER BY step_index DESC
                LIMIT 1
            """, (thread_id,))
        else:
            cursor.execute("""
                SELECT state_json FROM agent_states
                WHERE thread_id = ? AND step_index = ?
            """, (thread_id, step_index))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return json.loads(row[0])
        return None
    
    def time_travel(self, thread_id: str, target_step: int) -> Optional[Dict]:
        """
        Time-Travel / State Rewinding:
        Reset an agent's state back to a specific step for debugging.
        """
        state = self.get_state(thread_id, target_step)
        if state:
            logger.info(f"Time-Travel: Rewound thread '{thread_id}' to step {target_step}")
            # Delete all states after the target step
            conn = self._connect()
            cursor = conn.cursor()
            cursor.execute("""
                DELETE FROM agent_states
                WHERE thread_id = ? AND step_index > ?
            """, (thread_id, target_step))
            conn.commit()
            conn.close()
        else:
            logger.warning(f"Time-Travel: No state found for thread '{thread_id}' at step {target_step}")
        
        return state
    
    def list_checkpoints(self, thread_id: str) -> List[Dict]:
        """List all checkpoints for a thread (for time-travel UI)."""
        conn = self._connect()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT step_index, node_name, timestamp, created_at
            FROM agent_states
            WHERE thread_id = ?
            ORDER BY step_index
        """, (thread_id,))
        
        checkpoints = []
        for row in cursor.fetchall():
            checkpoints.append({
                "step_index": row[0],
                "node_name": row[1],
                "timestamp": row[2],
                "created_at": row[3],
            })
        
        conn.close()
        return checkpoints
    
    def save_session(self, session_id: str, thread_id: str,
                      user_id: str, conversation: Dict):
        """Save a conversation session."""
        conn = self._connect()
        cursor = conn.cursor()
        now = datetime.now().isoformat()
        
        cursor.execute("""
            INSERT OR REPLACE INTO conversation_sessions
            (session_id, thread_id, user_id, conversation_json, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            session_id, thread_id, user_id,
            json.dumps(conversation, default=str),
            now, now,
        ))
        
        conn.commit()
        conn.close()
    
    def load_session(self, session_id: str) -> Optional[Dict]:
        """Load a conversation session."""
        conn = self._connect()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT conversation_json FROM conversation_sessions
            WHERE session_id = ?
        """, (session_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return json.loads(row[0])
        return None


# ─────────────────────────────────────────────
# Stateful Agent Memory Manager
# ─────────────────────────────────────────────

class AgentMemoryManager:
    """
    Stateful Agents:
    Unified memory manager that coordinates short-term, long-term,
    and persistent memory for the agent.
    
    Maintains memory across multiple turns, tracks variables,
    and remembers past execution failures.
    """
    
    def __init__(self, persist_dir: str = "data"):
        self.short_term = ShortTermMemory()
        self.long_term = LongTermMemory(
            persist_path=os.path.join(persist_dir, "long_term_memory.json")
        )
        self.persistence = PersistenceLayer(
            db_path=os.path.join(persist_dir, "agent_state.db")
        )
        self.current_thread_id: str = ""
        self.current_step: int = 0
        self.execution_failures: List[Dict] = []
    
    def start_session(self, thread_id: str, user_id: str = "default"):
        """Start or resume a conversation session."""
        self.current_thread_id = thread_id
        self.short_term.session_id = thread_id
        
        # Try to resume previous session
        saved_session = self.persistence.load_session(thread_id)
        if saved_session:
            # Restore short-term memory
            for turn in saved_session.get("conversation", []):
                self.short_term.conversation.append(ConversationTurn(
                    role=turn["role"],
                    content=turn["content"],
                    timestamp=turn.get("timestamp", 0),
                ))
            logger.info(f"Resumed session: {thread_id}")
        else:
            self.short_term.clear()
            logger.info(f"New session: {thread_id}")
    
    def record_turn(self, role: str, content: str, user_id: str = "default"):
        """Record a conversation turn in both short and long term memory."""
        self.short_term.add_turn(role, content)
        
        # Save to persistence layer
        self.persistence.save_session(
            self.current_thread_id,
            self.current_thread_id,
            user_id,
            self.short_term.to_dict(),
        )
    
    def save_agent_state(self, state: Dict, node_name: str = ""):
        """Save the current agent state for time-travel."""
        self.persistence.save_state(
            self.current_thread_id,
            self.current_step,
            state,
            node_name,
        )
        self.current_step += 1
    
    def record_failure(self, step_name: str, error: str):
        """Remember past execution failures for self-correction."""
        self.execution_failures.append({
            "step_name": step_name,
            "error": error,
            "timestamp": time.time(),
            "thread_id": self.current_thread_id,
        })
    
    def get_failure_context(self) -> str:
        """Get formatted failure history for self-reflection prompts."""
        if not self.execution_failures:
            return ""
        
        recent = self.execution_failures[-5:]
        parts = ["Previous failures to consider:"]
        for f in recent:
            parts.append(f"- Step '{f['step_name']}': {f['error']}")
        return "\n".join(parts)


# ─────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────

def create_memory_manager(config=None) -> AgentMemoryManager:
    """Create the full memory management stack."""
    from config import RAGConfig
    cfg = config or RAGConfig()
    
    return AgentMemoryManager(persist_dir=cfg.data_dir)
