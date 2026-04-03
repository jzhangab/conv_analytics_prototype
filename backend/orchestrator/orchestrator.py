"""
Main Orchestrator — drives the FSM, coordinates all components.

Per-request flow:
  1. Retrieve or create session state.
  2. Add user message to history.
  3. Depending on FSM state, either:
     a. Classify intent (IDLE → INTENT_CLASSIFICATION)
     b. Gather missing parameters (PARAMETER_GATHERING)
     c. Handle confirmation reply (CONFIRMATION_PENDING)
     d. Execute confirmed skill (SKILL_EXECUTION)
  4. Build and return a ChatResponse dict for the Flask layer.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime

from backend.agents.site_list_merger_agent import parse_uploaded_file
from backend.llm.llm_client import LLMClient
from backend.llm.prompt_templates import CLARIFICATION_MESSAGE
from backend.orchestrator.confirmation_manager import (build_confirmation_prompt,
                                                        parse_confirmation_reply)
from backend.orchestrator.intent_classifier import classify_intent
from backend.orchestrator.parameter_extractor import extract_parameters
from backend.orchestrator.router import Router
from backend.state.conversation_state import (ConversationState, FSMState,
                                               SkillResult)
from backend.state.parameter_schema import SkillSchema, load_schemas
from backend.state.session_store import SessionStore

logger = logging.getLogger(__name__)

SHARED_PARAM_KEYS = ["indication", "age_group", "phase"]


class Orchestrator:
    def __init__(self, session_store: SessionStore, config: dict = None):
        if config is None:
            import yaml
            from pathlib import Path
            cfg_path = Path(__file__).parent.parent.parent / "config" / "llm_config.yaml"
            with open(cfg_path) as f:
                config = yaml.safe_load(f)

        self.session_store = session_store
        self.llm = LLMClient(config)
        self.router = Router(self.llm)
        self.schemas: dict[str, SkillSchema] = load_schemas()
        self.context_turns = config["llm_mesh"].get("context_window_turns", 10)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_message(self, session_id: str, user_message: str) -> dict:
        state = self.session_store.get_or_create(session_id)
        state.add_message("user", user_message)
        history = state.get_recent_messages(self.context_turns)

        response = self._route_fsm(state, user_message, history)
        state.add_message("assistant", response.get("message", ""))
        return response

    def handle_file_upload(self, session_id: str, file_key: str, file_storage) -> dict:
        state = self.session_store.get_or_create(session_id)
        try:
            file_info = parse_uploaded_file(file_storage)
            state.uploaded_files[file_key] = file_info
            label = "CRO" if file_key == "cro_file" else "Sponsor"
            msg = (
                f"{label} site list uploaded: **{file_info['filename']}** "
                f"({len(file_info['data'])} rows, columns: {', '.join(file_info['columns'])})."
            )
            state.add_message("assistant", msg)

            # Auto-detect Site Merger intent if not already set
            if state.active_skill is None or state.active_skill == "site_list_merger":
                state.active_skill = "site_list_merger"
                state.fsm_state = FSMState.PARAMETER_GATHERING

            return self._build_response(message=msg, state=state)
        except ValueError as e:
            return self._build_error(str(e))

    def handle_confirmation(
        self,
        session_id: str,
        confirmed: bool,
        edit_params: dict = None,
    ) -> dict:
        state = self.session_store.get_or_create(session_id)
        if state.fsm_state != FSMState.CONFIRMATION_PENDING:
            return self._build_error("No pending confirmation.")

        if not confirmed:
            state.fsm_state = FSMState.IDLE
            state.pending_confirmation = None
            state.active_skill = None
            msg = "Understood. I've cancelled that request. What else can I help with?"
            state.add_message("assistant", msg)
            return self._build_response(message=msg, state=state)

        if edit_params:
            state.merge_params(state.active_skill, edit_params)
            return self._check_and_confirm(state)

        return self._execute_skill(state)

    def export_to_dataset(self, session_id: str, result_id: str, dataset_name: str) -> dict:
        state = self.session_store.get_or_create(session_id)
        result = state.get_result_by_id(result_id)
        if result is None:
            return self._build_error(f"Result '{result_id}' not found.")
        if not result.table_data:
            return self._build_error("This result has no tabular data to export.")

        try:
            import dataiku
            import pandas as pd
            ds = dataiku.Dataset(dataset_name)
            df = pd.DataFrame(result.table_data)
            ds.write_with_schema(df)
            msg = f"Results exported to Dataiku dataset **{dataset_name}** ({len(df)} rows)."
        except Exception as e:
            logger.error("Export failed: %s", e)
            return self._build_error(f"Export failed: {e}")

        state.add_message("assistant", msg)
        return self._build_response(message=msg, state=state)

    # ------------------------------------------------------------------
    # FSM routing
    # ------------------------------------------------------------------

    def _route_fsm(self, state: ConversationState, user_message: str, history: list) -> dict:
        fsm = state.fsm_state

        # Confirmation pending — treat user message as confirmation reply
        if fsm == FSMState.CONFIRMATION_PENDING:
            reply = parse_confirmation_reply(user_message)
            if reply == "yes":
                return self._execute_skill(state)
            elif reply == "no":
                state.fsm_state = FSMState.IDLE
                state.pending_confirmation = None
                state.active_skill = None
                msg = "Understood. I've cancelled that request. What else can I help with?"
                return self._build_response(message=msg, state=state)
            else:
                # "edit" — re-extract parameters from this message
                extracted = extract_parameters(
                    self.llm, self.schemas[state.active_skill], user_message, history
                )
                state.merge_params(state.active_skill, extracted)
                return self._check_and_confirm(state)

        # Parameter gathering — try to extract more params from this message
        if fsm == FSMState.PARAMETER_GATHERING and state.active_skill:
            extracted = extract_parameters(
                self.llm, self.schemas[state.active_skill], user_message, history
            )
            state.merge_params(state.active_skill, extracted)
            return self._check_and_confirm(state)

        # Idle or clarification — classify intent first
        intent, confidence, reasoning = classify_intent(self.llm, user_message, history)

        if intent is None:
            # Check if user is picking a skill by number
            intent = self._parse_skill_selection(user_message)

        if intent is None:
            state.fsm_state = FSMState.CLARIFICATION_REQUEST
            msg = CLARIFICATION_MESSAGE
            return self._build_response(message=msg, state=state)

        # Intent recognized — set active skill and extract params
        state.active_skill = intent
        state.fsm_state = FSMState.PARAMETER_GATHERING

        # Inherit shared params from prior skill runs
        inherited = state.get_shared_params(intent, SHARED_PARAM_KEYS)
        if inherited:
            state.merge_params(intent, inherited)

        extracted = extract_parameters(
            self.llm, self.schemas[intent], user_message, history
        )
        state.merge_params(intent, extracted)

        return self._check_and_confirm(state)

    # ------------------------------------------------------------------
    # Parameter completeness check → confirmation or gather more
    # ------------------------------------------------------------------

    def _check_and_confirm(self, state: ConversationState) -> dict:
        schema = self.schemas[state.active_skill]
        params = state.get_params(state.active_skill)
        missing = schema.get_missing_required(params)

        # Files are handled by upload endpoint; check them separately
        missing_files = [
            p for p in missing if p.data_type == "file"
            and state.uploaded_files.get(p.name) is None
        ]
        missing_non_files = [p for p in missing if p.data_type != "file"]

        if missing_files or missing_non_files:
            return self._ask_for_missing(state, missing_files, missing_non_files)

        # All required params present — build confirmation
        inherited = state.get_shared_params(state.active_skill, SHARED_PARAM_KEYS)
        confirmation = build_confirmation_prompt(schema, params, inherited_params=inherited)
        state.pending_confirmation = confirmation
        state.fsm_state = FSMState.CONFIRMATION_PENDING

        return self._build_response(message=confirmation.summary_text, state=state)

    def _ask_for_missing(
        self,
        state: ConversationState,
        missing_files: list,
        missing_non_files: list,
    ) -> dict:
        questions = []
        if missing_files:
            for p in missing_files:
                questions.append(f"Please upload the **{p.label}** (CSV or Excel).")
        if missing_non_files:
            for p in missing_non_files:
                q = f"Please provide the **{p.label}**"
                if p.choices:
                    q += f" (options: {', '.join(p.choices)})"
                q += f". _{p.description}_"
                questions.append(q)

        msg = "\n".join(questions)
        return self._build_response(message=msg, state=state)

    # ------------------------------------------------------------------
    # Skill execution
    # ------------------------------------------------------------------

    def _execute_skill(self, state: ConversationState) -> dict:
        state.fsm_state = FSMState.SKILL_EXECUTION
        skill_id = state.active_skill
        params = state.get_params(skill_id)
        agent = self.router.get_agent(skill_id)

        if agent is None:
            state.fsm_state = FSMState.IDLE
            return self._build_error(f"No agent found for skill '{skill_id}'.")

        try:
            agent_result = agent.run(params, state)
        except Exception as e:
            logger.error("Agent execution error (%s): %s", skill_id, e)
            state.fsm_state = FSMState.IDLE
            return self._build_error(f"An error occurred while running {agent.display_name}: {e}")

        if not agent_result.success:
            state.fsm_state = FSMState.IDLE
            return self._build_error(agent_result.error_message or "Skill execution failed.")

        result_id = str(uuid.uuid4())
        skill_result = SkillResult(
            result_id=result_id,
            skill_id=skill_id,
            parameters_used=params,
            text_response=agent_result.text_response,
            table_data=agent_result.table_data,
            table_columns=agent_result.table_columns,
            chart_json=agent_result.chart_json,
        )
        state.add_result(skill_result)

        # Reset for next interaction
        state.fsm_state = FSMState.IDLE
        state.active_skill = None
        state.pending_confirmation = None

        return self._build_response(
            message=agent_result.text_response,
            state=state,
            table_data=agent_result.table_data,
            table_columns=agent_result.table_columns,
            chart_json=agent_result.chart_json,
            result_id=result_id,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _parse_skill_selection(self, message: str) -> str | None:
        """Handle numbered skill selection from clarification menu."""
        skill_by_number = {
            "1": "site_list_merger",
            "2": "trial_benchmarking",
            "3": "drug_reimbursement",
            "4": "enrollment_forecasting",
        }
        stripped = message.strip().rstrip(".,")
        return skill_by_number.get(stripped)

    def _build_response(
        self,
        message: str,
        state: ConversationState,
        table_data: list = None,
        table_columns: list = None,
        chart_json: dict = None,
        result_id: str = None,
    ) -> dict:
        return {
            "message": message,
            "fsm_state": state.fsm_state.value,
            "active_skill": state.active_skill,
            "table_data": table_data,
            "table_columns": table_columns,
            "chart_json": chart_json,
            "result_id": result_id,
            "uploaded_files": {
                k: {"filename": v["filename"], "rows": len(v["data"])}
                for k, v in state.uploaded_files.items()
            },
        }

    def _build_error(self, error_message: str) -> dict:
        return {
            "message": f"Sorry, something went wrong: {error_message}",
            "fsm_state": FSMState.IDLE.value,
            "active_skill": None,
            "table_data": None,
            "table_columns": None,
            "chart_json": None,
            "result_id": None,
            "uploaded_files": {},
            "error": error_message,
        }
