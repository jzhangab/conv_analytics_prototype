"""
Site List Matching SubAgent.
Accepts an uploaded site list and semantically matches each row against the
CTMS master site database (Dataiku dataset CTMS_DATASET) via LLM.
"""
import io
import logging

import pandas as pd

from backend.agents.base_agent import AgentResult, BaseAgent
from backend.llm.llm_client import LLMClient
from backend.llm.prompt_templates import SITE_MATCHING_SYSTEM, SITE_MATCHING_USER
from backend.llm.response_parser import parse_site_matching_response
from backend.state.conversation_state import ConversationState

logger = logging.getLogger(__name__)

BATCH_SIZE = 50   # Uploaded rows per LLM call (keeps prompt within context limits)

DEFAULT_CTMS_DATASET = "CTMS_DATASET"

# CTMS columns relevant for matching (keep prompt compact)
_CTMS_MATCH_COLS = ["site_id", "site_name", "country", "city", "pi_name"]


def _get_site_identifier(row):
    """Best single-column identifier from an uploaded row."""
    return str(
        row.get("site_name")
        or row.get("name")
        or row.get("Site Name")
        or row.get("Site")
        or row.iloc[0]
    )


class SiteListMatchingAgent(BaseAgent):
    skill_id = "site_list_matching"
    display_name = "Clinical Site List Matching"
    description = "Semantically matches an uploaded site list against the CTMS master site database."

    def __init__(self, llm_client: LLMClient, dataset_name: str = DEFAULT_CTMS_DATASET):
        self.llm = llm_client
        self.dataset_name = dataset_name

    def _load_ctms_df(self):
        """
        Returns (df, error_string). error_string is None on success.
        Loads from the Dataiku CTMS_DATASET dataset.
        """
        try:
            import dataiku
            df = dataiku.Dataset(self.dataset_name).get_dataframe()
            df.columns = [str(c).strip() for c in df.columns]
            logger.info("Loaded %d rows from Dataiku dataset '%s'.", len(df), self.dataset_name)
            return df, None
        except ImportError:
            return None, "Dataiku SDK not available — cannot load CTMS dataset."
        except Exception as e:
            return None, f"Could not read Dataiku dataset '{self.dataset_name}': {e}"

    def _match_batch(self, batch_df: pd.DataFrame, batch_offset: int,
                     ctms_text: str) -> list[dict]:
        """Run LLM matching on a single batch. Returns list of match dicts
        with uploaded_index adjusted to the global offset."""
        batch_csv = batch_df.to_csv(index=True)
        n_batch = len(batch_df)

        messages = [
            {"role": "system", "content": SITE_MATCHING_SYSTEM},
            {"role": "user", "content": SITE_MATCHING_USER.format(
                uploaded_data=batch_csv,
                ctms_data=ctms_text,
                n_uploaded=n_batch,
            )},
        ]

        raw = self.llm.complete_json(messages, temperature=self.llm.temp_deterministic)
        matches, _unmatched, _summary = parse_site_matching_response(raw)

        # Shift indices to global position
        for m in matches:
            m["uploaded_index"] = m["uploaded_index"] + batch_offset

        return matches

    def run(self, params: dict, state: ConversationState) -> AgentResult:
        file_info = state.uploaded_files.get("site_file")
        if not file_info:
            return AgentResult(
                success=False,
                text_response="",
                error_message="Missing uploaded file: Site List File."
            )

        # Load CTMS master list
        ctms_df, load_err = self._load_ctms_df()
        if load_err:
            return AgentResult(
                success=False,
                text_response="",
                error_message=load_err,
            )

        uploaded_df = pd.DataFrame(file_info["data"])
        n_uploaded = len(uploaded_df)

        # Compact CTMS representation for matching prompt
        ctms_match_cols = [c for c in _CTMS_MATCH_COLS if c in ctms_df.columns]
        ctms_text = ctms_df[ctms_match_cols].to_csv(index=False)

        # Process uploaded rows in batches
        all_matches: list[dict] = []
        for start in range(0, n_uploaded, BATCH_SIZE):
            batch_df = uploaded_df.iloc[start:start + BATCH_SIZE]
            try:
                batch_matches = self._match_batch(batch_df, start, ctms_text)
                all_matches.extend(batch_matches)
            except Exception as e:
                logger.error("Site matching batch %d-%d failed: %s",
                             start, start + len(batch_df), e)

        n_matched = len(all_matches)
        n_unmatched = n_uploaded - n_matched
        match_rate = round(n_matched / max(n_uploaded, 1) * 100, 1)

        summary_text = (
            f"**Site List Matching Results**\n\n"
            f"Uploaded **{n_uploaded}** sites and compared against **{len(ctms_df)}** CTMS sites.\n\n"
            f"- Matched: **{n_matched}** sites ({match_rate}%)\n"
            f"- Unmatched: **{n_unmatched}** sites\n"
        ).strip()

        # Build result table: one row per uploaded site
        matched_by_index = {m["uploaded_index"]: m for m in all_matches}
        table_data = []

        for i in range(n_uploaded):
            uploaded_row = uploaded_df.iloc[i]
            identifier = _get_site_identifier(uploaded_row)
            match = matched_by_index.get(i)
            table_data.append({
                "Row": i + 1,
                "Uploaded Site Name": identifier,
                "Match Status": "Matched" if match else "Not matched",
                "CTMS Site ID": match.get("ctms_site_id", "") if match else "",
                "CTMS Site Name": match.get("ctms_site_name", "") if match else "",
                "Confidence": match.get("match_confidence", "") if match else "",
                "Match Basis": match.get("match_basis", "") if match else "",
            })

        table_columns = [
            "Row", "Uploaded Site Name", "CTMS Site Name",
            "Match Status", "CTMS Site ID", "Confidence", "Match Basis",
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
