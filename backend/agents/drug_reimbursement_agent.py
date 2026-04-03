"""
Drug Reimbursement Assessment SubAgent.
Assesses reimbursement likelihood and HTA requirements by country.
Countries are always provided by the user — no defaults.
"""
import logging

from backend.agents.base_agent import AgentResult, BaseAgent
from backend.llm.llm_client import LLMClient
from backend.llm.prompt_templates import (DRUG_REIMBURSEMENT_SYSTEM,
                                           DRUG_REIMBURSEMENT_USER)
from backend.llm.response_parser import parse_reimbursement_response
from backend.state.conversation_state import ConversationState
from backend.utils.formatters import format_reimbursement_table

logger = logging.getLogger(__name__)


class DrugReimbursementAgent(BaseAgent):
    skill_id = "drug_reimbursement"
    display_name = "Drug Reimbursement Assessment"
    description = "Assesses drug reimbursement outlook by country for a given indication and phase."

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def run(self, params: dict, state: ConversationState) -> AgentResult:
        indication = params["indication"]
        age_group = params["age_group"]
        phase = params["phase"]
        countries = params["countries"]   # list of strings

        if not countries:
            return AgentResult(
                success=False,
                text_response="",
                error_message="No countries specified. Please provide a list of countries to assess."
            )

        countries_str = ", ".join(countries) if isinstance(countries, list) else str(countries)

        messages = [
            {"role": "system", "content": DRUG_REIMBURSEMENT_SYSTEM},
            {"role": "user", "content": DRUG_REIMBURSEMENT_USER.format(
                indication=indication,
                age_group=age_group,
                phase=phase,
                countries=countries_str,
            )},
        ]

        try:
            raw = self.llm.complete_json(messages, temperature=self.llm.temp_agents)
            data = parse_reimbursement_response(raw)
        except Exception as e:
            logger.error("Drug reimbursement LLM call failed: %s", e)
            return AgentResult(
                success=False,
                text_response="",
                error_message=f"Error during reimbursement assessment: {e}"
            )

        assessments = data.get("country_assessments", [])
        table = format_reimbursement_table(assessments)
        disclaimer = data.get("disclaimer", "")

        response_text = (
            f"**Drug Reimbursement Assessment: {indication} — {phase} — {age_group.capitalize()}**\n\n"
            + data.get("overall_summary", "")
            + (f"\n\n*{disclaimer}*" if disclaimer else "")
        )

        return AgentResult(
            success=True,
            text_response=response_text,
            table_data=[dict(zip(table["columns"], row)) for row in table["rows"]],
            table_columns=table["columns"],
        )
