"""
Conversation state machine and data models for a single user session.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional


class FSMState(str, Enum):
    IDLE = "idle"
    INTENT_CLASSIFICATION = "intent_classification"
    CLARIFICATION_REQUEST = "clarification_request"
    PARAMETER_GATHERING = "parameter_gathering"
    CONFIRMATION_PENDING = "confirmation_pending"
    SKILL_EXECUTION = "skill_execution"
    RESULT_PRESENTATION = "result_presentation"


@dataclass
class Message:
    role: str          # "user" | "assistant" | "system"
    content: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: dict = field(default_factory=dict)


@dataclass
class ConfirmationRequest:
    skill_id: str
    parameter_snapshot: dict
    summary_text: str           # Human-readable confirmation prompt shown to user


@dataclass
class SkillResult:
    result_id: str
    skill_id: str
    parameters_used: dict
    text_response: str
    table_data: Optional[list] = None       # List of dicts for table display
    table_columns: Optional[list] = None    # Ordered column names
    chart_json: Optional[dict] = None       # Bokeh JSON for forecasting charts
    timestamp: datetime = field(default_factory=datetime.utcnow)


class ConversationState:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.created_at = datetime.utcnow()
        self.last_activity = datetime.utcnow()

        self.messages: list[Message] = []
        self.fsm_state: FSMState = FSMState.IDLE
        self.active_skill: Optional[str] = None

        # Parameters collected per skill across turns
        # Shape: {skill_id: {param_name: value}}
        self.collected_parameters: dict[str, dict] = {}

        self.pending_confirmation: Optional[ConfirmationRequest] = None

        # File uploads stored as parsed DataFrames (in-memory)
        # Shape: {file_key: {"filename": str, "data": list_of_dicts, "columns": list}}
        self.uploaded_files: dict[str, dict] = {}

        # Completed skill results available for export
        self.prior_results: list[SkillResult] = []

    # ------------------------------------------------------------------
    # Message history helpers
    # ------------------------------------------------------------------

    def add_message(self, role: str, content: str, metadata: dict = None):
        self.messages.append(Message(role=role, content=content, metadata=metadata or {}))
        self.last_activity = datetime.utcnow()

    def get_recent_messages(self, n: int = 10) -> list[dict]:
        """Return the last n messages as plain dicts suitable for LLM calls."""
        recent = self.messages[-n:] if len(self.messages) > n else self.messages
        return [{"role": m.role, "content": m.content} for m in recent]

    # ------------------------------------------------------------------
    # Parameter helpers
    # ------------------------------------------------------------------

    def set_param(self, skill_id: str, param_name: str, value: Any):
        if skill_id not in self.collected_parameters:
            self.collected_parameters[skill_id] = {}
        self.collected_parameters[skill_id][param_name] = value

    def get_params(self, skill_id: str) -> dict:
        return self.collected_parameters.get(skill_id, {})

    def merge_params(self, skill_id: str, new_params: dict):
        """Merge new_params into existing, skipping None values."""
        existing = self.get_params(skill_id)
        for k, v in new_params.items():
            if v is not None:
                existing[k] = v
        self.collected_parameters[skill_id] = existing

    # ------------------------------------------------------------------
    # Result helpers
    # ------------------------------------------------------------------

    def add_result(self, result: SkillResult):
        self.prior_results.append(result)

    def get_result_by_id(self, result_id: str) -> Optional[SkillResult]:
        for r in self.prior_results:
            if r.result_id == result_id:
                return r
        return None

    # ------------------------------------------------------------------
    # Shared parameter inheritance
    # ------------------------------------------------------------------

    def get_shared_params(self, skill_id: str, shared_keys: list[str]) -> dict:
        """
        Look across all skills' collected parameters for values of shared_keys
        that are not yet set for skill_id. Used so e.g. forecasting can inherit
        indication/age_group/phase from a prior benchmarking run.
        """
        own = self.get_params(skill_id)
        result = {}
        for key in shared_keys:
            if own.get(key) is not None:
                result[key] = own[key]
                continue
            # Search other skills
            for other_skill, params in self.collected_parameters.items():
                if other_skill != skill_id and params.get(key) is not None:
                    result[key] = params[key]
                    break
        return result
