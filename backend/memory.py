"""
Chapter 7 content: the 4 Memory types, plus save/load to disk.

    Conversation
        |
      Memory
        |
      Prompt  ->  LLM

Each memory type below exposes the same two methods so the pipeline layer
can treat them interchangeably:
    - get_history_text() -> str   (what goes into the prompt's {history})
    - save_turn(user_msg, ai_msg) (record a new exchange)
"""

import json
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Tuple

from backend.config import MEMORY_STORE_DIR, BUFFER_WINDOW_K
from backend.llm import get_llm


class BaseMemory:
    name = "base"

    def get_history_text(self) -> str:
        raise NotImplementedError

    def save_turn(self, user_msg: str, ai_msg: str) -> None:
        raise NotImplementedError

    def save_to_disk(self, session_id: str) -> None:
        raise NotImplementedError

    def load_from_disk(self, session_id: str) -> None:
        raise NotImplementedError

    def _path(self, session_id: str) -> Path:
        return Path(MEMORY_STORE_DIR) / f"{session_id}_{self.name}.json"


class BufferMemory(BaseMemory):
    """Keeps the ENTIRE conversation, verbatim. Simple, but grows unbounded."""

    name = "buffer"

    def __init__(self):
        self.turns: List[Tuple[str, str]] = []

    def get_history_text(self) -> str:
        if not self.turns:
            return "(no previous conversation)"
        return "\n".join(f"User: {u}\nAI: {a}" for u, a in self.turns)

    def save_turn(self, user_msg: str, ai_msg: str) -> None:
        self.turns.append((user_msg, ai_msg))

    def save_to_disk(self, session_id: str) -> None:
        self._path(session_id).write_text(json.dumps(self.turns, ensure_ascii=False))

    def load_from_disk(self, session_id: str) -> None:
        path = self._path(session_id)
        if path.exists():
            self.turns = [tuple(t) for t in json.loads(path.read_text())]


class BufferWindowMemory(BaseMemory):
    """Keeps only the last K exchanges, so context stays bounded."""

    name = "buffer_window"

    def __init__(self, k: int = BUFFER_WINDOW_K):
        self.k = k
        self.turns: Deque[Tuple[str, str]] = deque(maxlen=k)

    def get_history_text(self) -> str:
        if not self.turns:
            return "(no previous conversation)"
        return "\n".join(f"User: {u}\nAI: {a}" for u, a in self.turns)

    def save_turn(self, user_msg: str, ai_msg: str) -> None:
        self.turns.append((user_msg, ai_msg))

    def save_to_disk(self, session_id: str) -> None:
        self._path(session_id).write_text(
            json.dumps(list(self.turns), ensure_ascii=False)
        )

    def load_from_disk(self, session_id: str) -> None:
        path = self._path(session_id)
        if path.exists():
            data = json.loads(path.read_text())
            self.turns = deque((tuple(t) for t in data), maxlen=self.k)


class SummaryMemory(BaseMemory):
    """
    Instead of storing every message, keeps a running LLM-generated summary.
    Ideal for long conversations where verbatim history would blow the
    context window.
    """

    name = "summary"

    def __init__(self):
        self.summary: str = ""

    def get_history_text(self) -> str:
        return self.summary if self.summary else "(no previous conversation)"

    def save_turn(self, user_msg: str, ai_msg: str) -> None:
        llm = get_llm()
        prompt = (
            "Update the running summary of this tutoring conversation with "
            "the new exchange. Keep it concise (max 5 sentences).\n\n"
            f"Current summary:\n{self.summary or '(empty)'}\n\n"
            f"New exchange:\nUser: {user_msg}\nAI: {ai_msg}\n\n"
            "Updated summary:"
        )
        self.summary = llm.invoke(prompt).strip()

    def save_to_disk(self, session_id: str) -> None:
        self._path(session_id).write_text(json.dumps({"summary": self.summary}))

    def load_from_disk(self, session_id: str) -> None:
        path = self._path(session_id)
        if path.exists():
            self.summary = json.loads(path.read_text()).get("summary", "")


class EntityMemory(BaseMemory):
    """
    Tracks named entities (concepts the student asked about) and short notes
    about them, e.g. {"Attention": "student found this difficult"}.
    Useful for a study assistant: "explain the thing I was confused about".
    """

    name = "entity"

    def __init__(self):
        self.entities: Dict[str, str] = {}

    def get_history_text(self) -> str:
        if not self.entities:
            return "(no tracked entities yet)"
        return "\n".join(f"- {ent}: {note}" for ent, note in self.entities.items())

    def save_turn(self, user_msg: str, ai_msg: str) -> None:
        llm = get_llm()
        prompt = (
            "Extract the main academic concept (entity) discussed in this "
            "exchange and a short note about it (e.g. 'explained', "
            "'student found this difficult'). Respond in the exact format "
            "'Entity: <name> | Note: <note>'. If no clear concept, respond "
            "'Entity: none | Note: none'.\n\n"
            f"User: {user_msg}\nAI: {ai_msg}"
        )
        result = llm.invoke(prompt).strip()
        try:
            entity_part, note_part = result.split("|")
            entity = entity_part.split(":", 1)[1].strip()
            note = note_part.split(":", 1)[1].strip()
            if entity.lower() != "none" and entity:
                self.entities[entity] = note
        except (ValueError, IndexError):
            pass  # extraction failed silently; conversation continues fine

    def save_to_disk(self, session_id: str) -> None:
        self._path(session_id).write_text(json.dumps(self.entities, ensure_ascii=False))

    def load_from_disk(self, session_id: str) -> None:
        path = self._path(session_id)
        if path.exists():
            self.entities = json.loads(path.read_text())


def get_memory(strategy: str) -> BaseMemory:
    """strategy: 'buffer' | 'buffer-window' | 'summary' | 'entity'"""
    strategy = strategy.lower().strip()
    mapping = {
        "buffer": BufferMemory,
        "buffer-window": BufferWindowMemory,
        "summary": SummaryMemory,
        "entity": EntityMemory,
    }
    if strategy not in mapping:
        raise ValueError(f"Unknown memory strategy: {strategy}")
    return mapping[strategy]()
