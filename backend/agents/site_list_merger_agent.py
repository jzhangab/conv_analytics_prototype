"""
Site List Matching SubAgent.
Accepts an uploaded site list and semantically matches each row against the
CTMS master site database (data/CTMS_SITES.csv) via LLM.
"""
import io
import logging
from pathlib import Path

import pandas as pd

from backend.agents.base_agent import AgentResult, BaseAgent
from backend.llm.llm_client import LLMClient
from backend.llm.prompt_templates import SITE_MATCHING_SYSTEM, SITE_MATCHING_USER
from backend.llm.response_parser import parse_site_matching_response
from backend.state.conversation_state import ConversationState

logger = logging.getLogger(__name__)

MAX_UPLOADED_ROWS = 200   # Safety cap for uploaded file rows sent to LLM

_CTMS_FILE = Path(__file__).parent.parent.parent / "data" / "CTMS_SITES.csv"

# CTMS columns relevant for matching (keep prompt compact)
_CTMS_MATCH_COLS = ["site_id", "site_name", "country", "city", "pi_name"]


def _df_to_csv_text(df: pd.DataFrame, max_rows: int = MAX_UPLOADED_ROWS) -> str:
    if len(df) > max_rows:
        df = df.head(max_rows)
        suffix = f"\n[Truncated to {max_rows} rows]"
    else:
        suffix = ""
    return df.to_csv(index=True) + suffix


class SiteListMatchingAgent(BaseAgent):
    skill_id = "site_list_matching"
    display_name = "Clinical Site List Matching"
    description = "Semantically matches an uploaded site list against the CTMS master site database."

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def run(self, params: dict, state: ConversationState) -> AgentResult:
        file_info = state.uploaded_files.get("site_file")
        if not file_info:
            return AgentResult(
                success=False,
                text_response="",
                error_message="Missing uploaded file: Site List File."
            )

        # Load CTMS master list
        if not _CTMS_FILE.exists():
            return AgentResult(
                success=False,
                text_response="",
                error_message=f"CTMS site file not found at {_CTMS_FILE}."
            )
        try:
            ctms_df = pd.read_csv(_CTMS_FILE)
            ctms_df.columns = [str(c).strip() for c in ctms_df.columns]
        except Exception as e:
            return AgentResult(
                success=False,
                text_response="",
                error_message=f"Could not load CTMS site file: {e}"
            )

        uploaded_df = pd.DataFrame(file_info["data"])
        n_uploaded = len(uploaded_df)

        # Compact CTMS representation for matching prompt
        ctms_match_cols = [c for c in _CTMS_MATCH_COLS if c in ctms_df.columns]
        ctms_text = ctms_df[ctms_match_cols].to_csv(index=False)
        uploaded_text = _df_to_csv_text(uploaded_df)

        messages = [
            {"role": "system", "content": SITE_MATCHING_SYSTEM},
            {"role": "user", "content": SITE_MATCHING_USER.format(
                uploaded_data=uploaded_text,
                ctms_data=ctms_text,
                n_uploaded=min(n_uploaded, MAX_UPLOADED_ROWS),
            )},
        ]

        try:
            raw = self.llm.complete_json(messages, temperature=self.llm.temp_deterministic)
            matches, unmatched_indices, summary = parse_site_matching_response(raw)
        except Exception as e:
            logger.error("Site matching LLM call failed: %s", e)
            return AgentResult(
                success=False,
                text_response="",
                error_message=f"Error during site matching: {e}"
            )

        n_matched = summary.get("matched", len(matches))
        n_unmatched = summary.get("unmatched", n_uploaded - n_matched)
        match_rate = round(n_matched / max(n_uploaded, 1) * 100, 1)

        summary_text = (
            f"**Site List Matching Results**\n\n"
            f"Uploaded **{n_uploaded}** sites and compared against **{len(ctms_df)}** CTMS sites.\n\n"
            f"- Matched: **{n_matched}** sites ({match_rate}%)\n"
            f"- Unmatched: **{n_unmatched}** sites\n\n"
            f"{summary.get('notes', '')}"
        ).strip()

        # Build result table: one row per uploaded site
        table_data = []
        matched_by_index = {m["uploaded_index"]: m for m in matches}
        rows_to_show = min(n_uploaded, MAX_UPLOADED_ROWS)

        for i in range(rows_to_show):
            uploaded_row = uploaded_df.iloc[i]
            # Best single-column identifier from the uploaded row
            identifier = (
                uploaded_row.get("site_name")
                or uploaded_row.get("name")
                or uploaded_row.get("Site Name")
                or uploaded_row.get("Site")
                or str(uploaded_row.iloc[0])
            )
            match = matched_by_index.get(i)
            if match:
                table_data.append({
                    "Row": i + 1,
                    "Uploaded Site": identifier,
                    "Match Status": "Matched",
                    "CTMS Site ID": match.get("ctms_site_id", ""),
                    "CTMS Site Name": match.get("ctms_site_name", ""),
                    "Confidence": match.get("match_confidence", ""),
                    "Match Basis": match.get("match_basis", ""),
                })
            else:
                table_data.append({
                    "Row": i + 1,
                    "Uploaded Site": identifier,
                    "Match Status": "Not matched",
                    "CTMS Site ID": "",
                    "CTMS Site Name": "",
                    "Confidence": "",
                    "Match Basis": "",
                })

        table_columns = [
            "Row", "Uploaded Site", "Match Status",
            "CTMS Site ID", "CTMS Site Name", "Confidence", "Match Basis",
        ]

        return AgentResult(
            success=True,
            text_response=summary_text,
            table_data=table_data,
            table_columns=table_columns,
        )


def parse_uploaded_file(file_storage) -> dict:
    """
    Parse a file (CSV or Excel) into a dict with keys: filename, data, columns.
    Accepts Werkzeug FileStorage objects or any object with .filename and .read().
    Falls back to content-based detection when the extension is missing or unknown.
    Raises ValueError on unsupported format or parse error.
    """
    filename = file_storage.filename or ""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    raw = file_storage.read()
    buf = io.BytesIO(raw)

    try:
        if ext == "csv":
            df = pd.read_csv(buf)
        elif ext in ("xlsx", "xls"):
            df = pd.read_excel(buf)
        else:
            try:
                df = pd.read_csv(io.BytesIO(raw))
            except Exception:
                try:
                    df = pd.read_excel(io.BytesIO(raw))
                except Exception:
                    raise ValueError(
                        f"Could not parse file '{filename or 'upload'}' as CSV or Excel. "
                        "Please upload a .csv, .xlsx, or .xls file."
                    )
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Could not parse file '{filename}': {e}")

    df.columns = [str(c).strip() for c in df.columns]
    df = df.where(pd.notna(df), None)

    return {
        "filename": filename,
        "data": df.to_dict(orient="records"),
        "columns": list(df.columns),
    }
