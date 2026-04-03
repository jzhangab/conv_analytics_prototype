"""
Abstract base class that all subagents must implement.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from backend.state.conversation_state import ConversationState, SkillResult


@dataclass
class AgentResult:
    success: bool
    text_response: str
    table_data: Optional[list] = None       # List of dicts for table display
    table_columns: Optional[list] = None
    chart_json: Optional[dict] = None       # Bokeh JSON (enrollment forecasting only)
    error_message: Optional[str] = None


class BaseAgent(ABC):
    skill_id: str
    display_name: str
    description: str

    @abstractmethod
    def run(self, params: dict, state: ConversationState) -> AgentResult:
        """Execute the skill with the provided validated parameters."""

    def build_skill_result(self, result_id: str, params: dict, agent_result: AgentResult) -> SkillResult:
        return SkillResult(
            result_id=result_id,
            skill_id=self.skill_id,
            parameters_used=params,
            text_response=agent_result.text_response,
            table_data=agent_result.table_data,
            table_columns=agent_result.table_columns,
            chart_json=agent_result.chart_json,
        )
