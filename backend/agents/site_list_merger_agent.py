"""
Site List Merger SubAgent.
Accepts two uploaded DataFrames (CRO and sponsor) and merges them via LLM.
"""
import io
import json
import logging
import uuid

import pandas as pd

from backend.agents.base_agent import AgentResult, BaseAgent
from backend.llm.llm_client import LLMClient
from backend.llm.prompt_templates import SITE_MERGER_SYSTEM, SITE_MERGER_USER
from backend.llm.response_parser import parse_site_merger_response
from backend.state.conversation_state import ConversationState
from backend.utils.formatters import dict_list_to_table, format_merger_summary

logger = logging.getLogger(__name__)

MAX_ROWS_FOR_LLM = 300   # Safety cap — very large lists are chunked or truncated


def _df_to_text(df: pd.DataFrame, max_rows: int = MAX_ROWS_FOR_LLM) -> str:
    """Convert a DataFrame to a compact CSV string for the LLM prompt."""
    if len(df) > max_rows:
        df = df.head(max_rows)
        suffix = f"\n[Truncated to {max_rows} rows for processing]"
    else:
        suffix = ""
    return df.to_csv(index=False) + suffix


class SiteListMergerAgent(BaseAgent):
    skill_id = "site_list_merger"
    display_name = "Clinical Site List Merger"
    description = "Merges CRO and sponsor site lists into a reconciled, deduplicated list."

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def run(self, params: dict, state: ConversationState) -> AgentResult:
        cro_file_info = state.uploaded_files.get("cro_file")
        sponsor_file_info = state.uploaded_files.get("sponsor_file")

        if not cro_file_info or not sponsor_file_info:
            missing = []
            if not cro_file_info:
                missing.append("CRO site list file")
            if not sponsor_file_info:
                missing.append("sponsor site list file")
            return AgentResult(
                success=False,
                text_response="",
                error_message=f"Missing uploaded file(s): {', '.join(missing)}."
            )

        cro_df = pd.DataFrame(cro_file_info["data"])
        sponsor_df = pd.DataFrame(sponsor_file_info["data"])
        merge_strategy = params.get("merge_strategy", "flag_conflicts")

        cro_text = _df_to_text(cro_df)
        sponsor_text = _df_to_text(sponsor_df)

        messages = [
            {"role": "system", "content": SITE_MERGER_SYSTEM},
            {"role": "user", "content": SITE_MERGER_USER.format(
                cro_data=cro_text,
                sponsor_data=sponsor_text,
                merge_strategy=merge_strategy,
            )},
        ]

        try:
            raw = self.llm.complete_json(messages, temperature=self.llm.temp_deterministic)
            merged_sites, summary = parse_site_merger_response(raw)
        except Exception as e:
            logger.error("Site merger LLM call failed: %s", e)
            return AgentResult(
                success=False,
                text_response="",
                error_message=f"Error during site list merging: {e}"
            )

        summary_text = format_merger_summary(summary)
        table = dict_list_to_table(merged_sites)

        return AgentResult(
            success=True,
            text_response=summary_text,
            table_data=merged_sites,
            table_columns=table["columns"],
        )


def parse_uploaded_file(file_storage) -> dict:
    """
    Parse a Werkzeug FileStorage object (CSV or Excel) into a dict with
    keys: filename, data (list of dicts), columns (list of str).
    Raises ValueError on unsupported format or parse error.
    """
    filename = file_storage.filename or ""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    try:
        if ext == "csv":
            df = pd.read_csv(io.BytesIO(file_storage.read()))
        elif ext in ("xlsx", "xls"):
            df = pd.read_excel(io.BytesIO(file_storage.read()))
        else:
            raise ValueError(f"Unsupported file type: .{ext}. Please upload CSV or Excel files.")
    except Exception as e:
        raise ValueError(f"Could not parse file '{filename}': {e}")

    # Normalize column names
    df.columns = [str(c).strip() for c in df.columns]
    df = df.where(pd.notna(df), None)

    return {
        "filename": filename,
        "data": df.to_dict(orient="records"),
        "columns": list(df.columns),
    }
