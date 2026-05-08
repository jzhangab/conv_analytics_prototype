"""
Entry-point and exit-point contracts for the chat backend.

ChatRequest  — everything the backend can receive from any caller
ChatResponse — everything the backend can produce for any caller

Neither class has any dependency on Flask, Dataiku, or any other transport.
"""
from __future__ import annotations

import base64
import csv
import io
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class UploadedFile:
    """
    Framework-agnostic file upload.  The Flask adapter converts Werkzeug
    FileStorage objects into this; tests can construct it directly from bytes.
    """
    file_key: str       # "site_file"
    filename: str
    data: bytes
    content_type: str = "application/octet-stream"

    # Duck-type compatibility with Werkzeug FileStorage so existing agent
    # code (parse_uploaded_file) can use this directly.
    def read(self) -> bytes:
        return self.data

    def stream(self) -> io.BytesIO:
        return io.BytesIO(self.data)


@dataclass
class DownloadableFile:
    """
    A file the frontend can offer as a download button.
    Content is base64-encoded so it survives JSON serialization.
    """
    filename: str
    content_b64: str        # base64-encoded bytes
    content_type: str       # e.g. "text/csv", "application/json"
    description: str        # human label, e.g. "Trial Benchmarking Results (CSV)"

    @classmethod
    def from_bytes(
        cls,
        filename: str,
        data: bytes,
        content_type: str,
        description: str,
    ) -> DownloadableFile:
        return cls(
            filename=filename,
            content_b64=base64.b64encode(data).decode("utf-8"),
            content_type=content_type,
            description=description,
        )

    def decode(self) -> bytes:
        return base64.b64decode(self.content_b64)

    def to_dict(self) -> dict:
        return {
            "filename": self.filename,
            "content_b64": self.content_b64,
            "content_type": self.content_type,
            "description": self.description,
        }


@dataclass
class ChatRequest:
    """
    Single entry-point contract.

    Every user interaction — typed message, file upload, confirmation reply,
    or export request — is expressed as a ChatRequest and passed to
    ChatBackend.process().

    action values:
      "message"  — user typed text into the chat input
      "confirm"  — structured yes/no/edit reply from a confirmation widget
      "upload"   — one or more files attached by the user
      "export"   — persist a prior skill result to external storage
    """
    session_id: str
    action: str                                 # "message" | "confirm" | "upload" | "export"

    # action="message"
    message: Optional[str] = None

    # action="confirm"
    confirmed: Optional[bool] = None            # True=yes, False=no
    edit_params: Optional[dict] = None          # param overrides when the user edits

    # action="upload"
    files: list[UploadedFile] = field(default_factory=list)

    # action="export"
    result_id: Optional[str] = None
    export_destination: Optional[str] = None    # Snowflake table name or dataset name

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "action": self.action,
            "message": self.message,
            "confirmed": self.confirmed,
            "edit_params": self.edit_params,
            "files": [{"file_key": f.file_key, "filename": f.filename} for f in self.files],
            "result_id": self.result_id,
            "export_destination": self.export_destination,
        }


@dataclass
class ChatResponse:
    """
    Single exit-point contract.

    Everything the backend produces for a given ChatRequest is packed here.
    The React frontend inspects each field independently:

      message           → render in the assistant chat bubble (markdown)
      fsm_state         → drive UI mode:
                            "idle"                 — normal input
                            "parameter_gathering"  — bot is asking follow-ups
                            "confirmation_pending" — show yes/no/edit widget
                            "analysis_planning"    — show approve/revise widget
      table_data        → render a table below the message
      table_columns     → column headers for the table
      chart_json        → Bokeh JSON item — pass to Bokeh.embed.embed_item()
      downloadable_files → render a download button per file
      uploaded_file_metadata → confirm what was parsed after an upload
      error             → show an error banner (non-null means something failed)
    """
    session_id: str
    action: str
    success: bool

    # Conversational output
    message: Optional[str] = None
    fsm_state: Optional[str] = None
    active_skill: Optional[str] = None
    skill_id: Optional[str] = None
    result_id: Optional[str] = None

    # Structured data output
    table_data: Optional[list[dict]] = None
    table_columns: Optional[list[str]] = None
    chart_json: Optional[dict] = None

    # File output — auto-populated when table_data is present
    downloadable_files: list[DownloadableFile] = field(default_factory=list)

    # Upload acknowledgement
    uploaded_file_metadata: Optional[dict] = None

    # Error (non-null means the request failed)
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "action": self.action,
            "success": self.success,
            "message": self.message,
            "fsm_state": self.fsm_state,
            "active_skill": self.active_skill,
            "skill_id": self.skill_id,
            "result_id": self.result_id,
            "table_data": self.table_data,
            "table_columns": self.table_columns,
            "chart_json": self.chart_json,
            "downloadable_files": [f.to_dict() for f in self.downloadable_files],
            "uploaded_file_metadata": self.uploaded_file_metadata,
            "error": self.error,
        }
