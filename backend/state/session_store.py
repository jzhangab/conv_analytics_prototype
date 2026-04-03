"""
In-memory session store. Each Flask request looks up or creates a ConversationState
by session_id. Sessions expire after a configurable timeout.
"""
from __future__ import annotations

import threading
from datetime import datetime, timedelta

from backend.state.conversation_state import ConversationState


class SessionStore:
    def __init__(self, timeout_minutes: int = 30):
        self._sessions: dict[str, ConversationState] = {}
        self._lock = threading.Lock()
        self._timeout = timedelta(minutes=timeout_minutes)

    def get_or_create(self, session_id: str) -> ConversationState:
        with self._lock:
            self._evict_expired()
            if session_id not in self._sessions:
                self._sessions[session_id] = ConversationState(session_id)
            return self._sessions[session_id]

    def get(self, session_id: str) -> ConversationState | None:
        with self._lock:
            self._evict_expired()
            return self._sessions.get(session_id)

    def delete(self, session_id: str):
        with self._lock:
            self._sessions.pop(session_id, None)

    def _evict_expired(self):
        now = datetime.utcnow()
        expired = [
            sid for sid, state in self._sessions.items()
            if (now - state.last_activity) > self._timeout
        ]
        for sid in expired:
            del self._sessions[sid]
