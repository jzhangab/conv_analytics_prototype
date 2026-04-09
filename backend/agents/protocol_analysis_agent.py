"""
Protocol Analysis SubAgent.

Workflow:
  1. Extract full text from the uploaded file (PDF / DOCX / TXT)
  2. Hard-cap at 3,000,000 characters (well within GPT-4.1's 1M-token window)
  3. Send the full text to the analysis LLM in a single call
"""
from __future__ import annotations

import io
import logging

from backend.agents.base_agent import AgentResult, BaseAgent
from backend.llm.llm_client import LLMClient
from backend.llm.prompt_templates import (
    PROTOCOL_ANALYSIS_SYSTEM,
    PROTOCOL_ANALYSIS_USER,
)
from backend.llm.web_search import WebSearchClient
from backend.state.conversation_state import ConversationState

logger = logging.getLogger(__name__)

MAX_CHARS = 800_000  # hard ceiling — ~200k tokens at 4 chars/token
SEVERITY_ORDER = {"critical": 0, "major": 1, "minor": 2, "suggestion": 3}


class ProtocolAnalysisAgent(BaseAgent):
    skill_id     = "protocol_analysis"
    display_name = "Clinical Trial Protocol Analysis"
    description  = "Analyses an uploaded protocol document for study design improvements."

    def __init__(self, llm_client: LLMClient, web_search: WebSearchClient | None = None):
        self.llm = llm_client
        self.web_search = web_search

    # ------------------------------------------------------------------
    # Trace helper
    # ------------------------------------------------------------------

    def _trace(self, msg: str) -> None:
        if not hasattr(self.llm, "call_log"):
            self.llm.call_log = []
        self.llm.call_log.append({
            "messages": [],
            "response": msg,
            "synthetic": True,
            "label": "Protocol Analysis",
        })

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def run(self, params: dict, state: ConversationState) -> AgentResult:
        file_info = state.uploaded_files.get("protocol_file")
        if not file_info:
            return AgentResult(
                success=False, text_response="",
                error_message="No protocol file found. Please upload a PDF, DOCX, or TXT protocol first.",
            )

        filename = file_info["filename"]
        fmt      = file_info.get("format")

        # Backward-compat: old format had "text" key and no "format"
        if fmt is None:
            fmt = "pdf" if "pages" in file_info else "txt"

        self._trace(f"Preparing '{filename}' for analysis (format: {fmt})")

        # Assemble full text
        if fmt == "pdf":
            pages = file_info.get("pages", [])
            if not pages:
                return AgentResult(
                    success=False, text_response="",
                    error_message="No page text could be extracted from the uploaded PDF.",
                )
            full_text = "\n\n".join(p for p in pages if p.strip())
            self._trace(
                f"PDF extracted: {len(pages)} pages, {len(full_text):,} characters total."
            )
        else:
            full_text = (
                file_info.get("full_text")
                or file_info.get("text", "")
            )
            self._trace(f"Text extracted: {len(full_text):,} characters.")

        if not full_text.strip():
            return AgentResult(
                success=False, text_response="",
                error_message="Could not extract any text from the uploaded file.",
            )

        # Apply hard cap
        if len(full_text) > MAX_CHARS:
            full_text = full_text[:MAX_CHARS]
            self._trace(
                f"Document truncated to {MAX_CHARS:,} characters (hard cap). "
                "This covers the vast majority of any clinical protocol."
            )
        else:
            self._trace(
                f"Full document fits within limit ({len(full_text):,} / {MAX_CHARS:,} chars). "
                "Sending complete text for analysis."
            )

        # Web search for relevant regulatory guidance
        web_context = ""
        if self.web_search:
            # Try to extract an indication-like term from the protocol text for a targeted search
            search_hint = filename.replace("_", " ").rsplit(".", 1)[0]
            raw = self.web_search.search_for_skill(
                "protocol_analysis",
                {"indication": search_hint},
            )
            if raw:
                web_context = f"\n\nSupplementary web search results (regulatory guidance):\n{raw}"
                self._trace("Web search context added to analysis prompt.")

        # Single analysis call
        self._trace("Running protocol design analysis...")
        messages = [
            {"role": "system", "content": PROTOCOL_ANALYSIS_SYSTEM},
            {"role": "user",   "content": PROTOCOL_ANALYSIS_USER.format(
                filename=filename,
                protocol_text=full_text + web_context,
            )},
        ]
        try:
            data = self.llm.complete_json(messages, temperature=self.llm.temp_agents)
        except Exception as e:
            logger.error("Protocol analysis failed: %s", e)
            return AgentResult(
                success=False, text_response="",
                error_message=f"Protocol analysis failed: {e}",
            )

        self._trace("Analysis complete.")
        return self._format_result(data, filename)

    # ------------------------------------------------------------------
    # Format output
    # ------------------------------------------------------------------

    def _format_result(self, data: dict, filename: str) -> AgentResult:
        rating   = data.get("overall_rating", "").replace("_", " ").title()
        findings = sorted(
            data.get("findings", []),
            key=lambda f: SEVERITY_ORDER.get(f.get("severity", "suggestion"), 3),
        )
        lines = [
            f"## Protocol Analysis: {filename}",
            f"**Overall Rating:** {rating}",
            "",
            "### Executive Summary",
            data.get("executive_summary", ""),
        ]
        if data.get("strengths"):
            lines += ["", "### Strengths", *[f"- {s}" for s in data["strengths"]]]
        if data.get("critical_concerns"):
            lines += ["", "### Critical Concerns", *[f"- {c}" for c in data["critical_concerns"]]]
        if data.get("section_assessments"):
            lines += ["", "### Section Assessments"]
            for k, v in data["section_assessments"].items():
                if v:
                    lines.append(f"**{k.replace('_', ' ').title()}:** {v}")

        table_data = [
            {
                "#":              i + 1,
                "Section":        f.get("category", ""),
                "Finding":        f.get("finding", ""),
                "Severity":       f.get("severity", "").title(),
                "Recommendation": f.get("recommendation", ""),
            }
            for i, f in enumerate(findings)
        ]
        return AgentResult(
            success=True,
            text_response="\n".join(lines),
            table_data=table_data or None,
            table_columns=["#", "Section", "Finding", "Severity", "Recommendation"] if table_data else None,
        )


# ---------------------------------------------------------------------------
# File parsing helpers (called by orchestrator.handle_file_upload)
# ---------------------------------------------------------------------------

def parse_protocol_file(file_storage) -> dict:
    """
    PDF  → {"filename", "format":"pdf", "pages":[str], "total_pages":int}
    DOCX → {"filename", "format":"docx", "full_text":str}
    TXT  → {"filename", "format":"txt",  "full_text":str}
    """
    filename = getattr(file_storage, "filename", "") or ""
    ext      = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    raw      = file_storage.read()

    if ext == "pdf":
        return _parse_pdf(raw, filename)
    if ext == "docx":
        return _parse_docx(raw, filename)
    if ext == "doc":
        raise ValueError(
            "Legacy .doc files are not supported. Please save as .docx, .pdf, or .txt."
        )
    return _parse_text(raw, filename)


def _parse_pdf(raw: bytes, filename: str) -> dict:
    try:
        import pypdf
    except ImportError:
        raise ValueError("pypdf is required to read PDF files. Install with: pip install pypdf")

    try:
        reader = pypdf.PdfReader(io.BytesIO(raw))
    except Exception as e:
        raise ValueError(f"Failed to open PDF '{filename}': {e}")

    pages = [page.extract_text() or "" for page in reader.pages]
    if not any(p.strip() for p in pages):
        raise ValueError(
            f"No text could be extracted from '{filename}'. "
            "The PDF may be scanned/image-based. Please provide a text-selectable PDF."
        )
    return {
        "filename":    filename,
        "format":      "pdf",
        "pages":       pages,
        "total_pages": len(pages),
    }


def _parse_docx(raw: bytes, filename: str) -> dict:
    try:
        from docx import Document
    except ImportError:
        raise ValueError(
            "python-docx is required to read DOCX files. Install with: pip install python-docx"
        )
    try:
        doc   = Document(io.BytesIO(raw))
        text  = "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
        if not text:
            raise ValueError(f"No text extracted from '{filename}'.")
        return {"filename": filename, "format": "docx", "full_text": text}
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Failed to read DOCX '{filename}': {e}")


def _parse_text(raw: bytes, filename: str) -> dict:
    for enc in ("utf-8", "latin-1"):
        try:
            text = raw.decode(enc)
            if text.strip():
                return {"filename": filename, "format": "txt", "full_text": text}
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not read '{filename}' as text. Supported formats: PDF, DOCX, TXT.")
