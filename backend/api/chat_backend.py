"""
ChatBackend — the single entry point for all backend logic.

Instantiate once at app startup.  Call process() for every user interaction.
Transport-agnostic: works identically from a Flask route, a FastAPI handler,
a test, or a plain Python script.

Usage
-----
    from backend.api.chat_backend import ChatBackend
    from backend.api.models import ChatRequest

    backend = ChatBackend()                         # init once

    req = ChatRequest(
        session_id="abc-123",
        action="message",
        message="benchmark KRAS G12C adults phase 3",
    )
    resp = backend.process(req)                     # call per interaction
    print(resp.message)
    print(resp.table_data)

Dependency injection
--------------------
Pass llm_client, session_store, and/or snowflake_client to override the
defaults.  Useful in tests and when the host app already owns these objects:

    backend = ChatBackend(
        session_store=my_existing_store,
        snowflake_client=my_existing_sf_client,
    )
"""
from __future__ import annotations

import csv
import io
import logging
import os
import traceback
import uuid
from pathlib import Path
from typing import Optional

import yaml

from backend.api.models import ChatRequest, ChatResponse, DownloadableFile, UploadedFile

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "llm_config.yaml"


class ChatBackend:

    def __init__(
        self,
        config_path: str | Path = _DEFAULT_CONFIG_PATH,
        llm_client=None,
        session_store=None,
        snowflake_client=None,
    ):
        self._config = _load_yaml(config_path)
        self._sf = snowflake_client or _try_build_snowflake_client()
        self._session_store = session_store or _build_session_store()
        # Orchestrator owns the LLM client; pass config so it reads the right
        # model and temperatures.  If the host app needs to inject a custom LLM
        # client, extend Orchestrator.__init__ to accept one directly.
        self._orchestrator = _build_orchestrator(self._session_store, self._config)
        logger.info("ChatBackend initialized.")

    # ── Public entry point ─────────────────────────────────────────────────

    def process(self, req: ChatRequest) -> ChatResponse:
        """
        Process any user interaction and return a structured ChatResponse.
        This is the only method the transport layer needs to call.
        """
        try:
            if not req.session_id:
                req.session_id = str(uuid.uuid4())

            if req.action == "message":
                return self._handle_message(req)
            if req.action == "confirm":
                return self._handle_confirm(req)
            if req.action == "upload":
                return self._handle_upload(req)
            if req.action == "export":
                return self._handle_export(req)

            return _error_response(req, f"Unknown action: {req.action!r}")

        except Exception:
            tb = traceback.format_exc()
            logger.error("Unhandled error in ChatBackend.process:\n%s", tb)
            return _error_response(req, tb)

    # ── Action handlers ────────────────────────────────────────────────────

    def _handle_message(self, req: ChatRequest) -> ChatResponse:
        if not req.message or not req.message.strip():
            return _error_response(req, "message field is required for action='message'")
        raw = self._orchestrator.process_message(req.session_id, req.message.strip())
        return _build_response(req, raw)

    def _handle_confirm(self, req: ChatRequest) -> ChatResponse:
        if req.confirmed is None:
            return _error_response(req, "confirmed field is required for action='confirm'")
        raw = self._orchestrator.handle_confirmation(
            req.session_id, req.confirmed, req.edit_params
        )
        return _build_response(req, raw)

    def _handle_upload(self, req: ChatRequest) -> ChatResponse:
        if not req.files:
            return _error_response(req, "files list is required for action='upload'")

        # Upload each file; surface the last result (usually only one file)
        last_raw: dict = {}
        for uf in req.files:
            last_raw = self._orchestrator.handle_file_upload(
                req.session_id, uf.file_key, uf   # uf is duck-type compatible
            )

        if last_raw.get("error"):
            return _error_response(req, last_raw["error"])

        return ChatResponse(
            session_id=req.session_id,
            action=req.action,
            success=True,
            message=last_raw.get("message"),
            fsm_state=last_raw.get("fsm_state"),
            active_skill=last_raw.get("active_skill"),
            uploaded_file_metadata=last_raw.get("uploaded_files"),
        )

    def _handle_export(self, req: ChatRequest) -> ChatResponse:
        if not req.result_id or not req.export_destination:
            return _error_response(
                req, "result_id and export_destination are required for action='export'"
            )
        raw = self._orchestrator.export_to_dataset(
            req.session_id, req.result_id, req.export_destination
        )
        if raw.get("error"):
            return _error_response(req, raw["error"])
        return ChatResponse(
            session_id=req.session_id,
            action=req.action,
            success=True,
            message=raw.get("message"),
        )


# ── Module-level helpers (no class state needed) ───────────────────────────

def _build_response(req: ChatRequest, raw: dict) -> ChatResponse:
    """Convert an orchestrator dict into a typed ChatResponse."""
    chart_json = _serialize_chart(raw.get("chart_json"))
    table_data = raw.get("table_data")
    table_columns = raw.get("table_columns")
    skill_id = raw.get("skill_id")
    has_error = bool(raw.get("error"))

    downloads: list[DownloadableFile] = []
    if table_data and table_columns:
        downloads.append(_table_to_csv(table_data, table_columns, skill_id or "results"))

    return ChatResponse(
        session_id=req.session_id,
        action=req.action,
        success=not has_error,
        message=raw.get("message"),
        fsm_state=raw.get("fsm_state"),
        active_skill=raw.get("active_skill"),
        skill_id=skill_id,
        result_id=raw.get("result_id"),
        table_data=table_data,
        table_columns=table_columns,
        chart_json=chart_json,
        downloadable_files=downloads,
        uploaded_file_metadata=raw.get("uploaded_files"),
        error=raw.get("error"),
    )


def _error_response(req: ChatRequest, detail: str) -> ChatResponse:
    return ChatResponse(
        session_id=req.session_id,
        action=req.action,
        success=False,
        error=detail,
    )


def _serialize_chart(chart) -> Optional[dict]:
    """Serialize a Bokeh Figure to a JSON-safe dict, or pass through if already a dict."""
    if chart is None:
        return None
    if isinstance(chart, dict):
        return chart
    try:
        from bokeh.embed import json_item
        return json_item(chart, "chart")
    except Exception:
        logger.warning("Could not serialize Bokeh chart.")
        return None


def _table_to_csv(
    table_data: list[dict],
    table_columns: list[str],
    skill_id: str,
) -> DownloadableFile:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=table_columns, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(table_data)
    filename = f"{skill_id.replace(' ', '_')}_results.csv"
    return DownloadableFile.from_bytes(
        filename=filename,
        data=buf.getvalue().encode("utf-8"),
        content_type="text/csv",
        description=f"{skill_id.replace('_', ' ').title()} Results (CSV)",
    )


def _load_yaml(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _try_build_snowflake_client():
    required = [
        "SNOWFLAKE_ACCOUNT", "SNOWFLAKE_USER", "SNOWFLAKE_PASSWORD",
        "SNOWFLAKE_WAREHOUSE", "SNOWFLAKE_DATABASE", "SNOWFLAKE_SCHEMA",
    ]
    if not all(os.environ.get(k) for k in required):
        logger.warning(
            "Snowflake env vars not set — data-dependent agents will use local CSV fallback."
        )
        return None
    try:
        from backend.data.snowflake_client import SnowflakeClient
        return SnowflakeClient()
    except ImportError:
        logger.warning("snowflake-connector-python not installed; using CSV fallback.")
        return None


def _build_session_store():
    from backend.state.session_store import SessionStore
    return SessionStore(timeout_minutes=30)


def _build_orchestrator(session_store, config: dict):
    from backend.orchestrator.orchestrator import Orchestrator
    return Orchestrator(session_store, config=config)
