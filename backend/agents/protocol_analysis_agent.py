"""
Protocol Analysis SubAgent — map-reduce chunk-and-summarize approach.

Workflow for any format (PDF / DOCX / TXT):
  1. Split the document into overlapping chunks
  2. LLM extracts and preserves all clinically meaningful content from each chunk
  3. Chunk extractions are concatenated into a single condensed document
  4. Single comprehensive protocol design analysis is run on the condensed document
"""
from __future__ import annotations

import io
import logging
import re

from backend.agents.base_agent import AgentResult, BaseAgent
from backend.llm.llm_client import LLMClient
from backend.llm.prompt_templates import (
    PROTOCOL_ANALYSIS_SYSTEM,
    PROTOCOL_ANALYSIS_USER,
    PROTOCOL_CHUNK_SYSTEM,
    PROTOCOL_CHUNK_USER,
)
from backend.state.conversation_state import ConversationState

logger = logging.getLogger(__name__)

# Chunking parameters
CHUNK_PAGES   = 8       # PDF pages per chunk
OVERLAP_PAGES = 1       # pages of overlap between adjacent PDF chunks
CHUNK_CHARS   = 20_000  # chars per chunk for text documents
OVERLAP_CHARS = 1_000   # chars of overlap between adjacent text chunks

# Output limits
MAX_CHUNK_EXTRACTION = 4_000   # soft char cap per chunk extraction (guidance to LLM)
MAX_COMBINED         = 90_000  # hard cap on combined extractions sent to final analysis

SEVERITY_ORDER = {"critical": 0, "major": 1, "minor": 2, "suggestion": 3}


class ProtocolAnalysisAgent(BaseAgent):
    skill_id     = "protocol_analysis"
    display_name = "Clinical Trial Protocol Analysis"
    description  = "Analyses an uploaded protocol document for study design improvements."

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    # ------------------------------------------------------------------
    # Trace helper
    # ------------------------------------------------------------------

    def _trace(self, msg: str) -> None:
        """Append a synthetic progress entry visible in the LLM trace pane."""
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

        # Backward-compat: old parse_protocol_file returned {"text":...} with no "format" key
        if fmt is None:
            fmt = "pdf" if "pages" in file_info else "txt"

        self._trace(f"Starting analysis of '{filename}' (format: {fmt})")

        if fmt == "pdf":
            return self._run_pdf(file_info, filename)
        return self._run_text(file_info, filename)

    # ------------------------------------------------------------------
    # Route to map-reduce
    # ------------------------------------------------------------------

    def _run_pdf(self, file_info: dict, filename: str) -> AgentResult:
        pages = file_info.get("pages", [])
        if not pages:
            return AgentResult(
                success=False, text_response="",
                error_message="No page text could be extracted from the uploaded PDF.",
            )
        # Build overlapping page chunks
        chunks: list[tuple[int, int, str]] = []  # (start_page_label, end_page_label, text)
        i = 0
        while i < len(pages):
            end = min(i + CHUNK_PAGES, len(pages))
            chunk_text = "\n\n".join(p for p in pages[i:end] if p.strip())
            if chunk_text.strip():
                chunks.append((i + 1, end, chunk_text))
            if end >= len(pages):
                break
            i += CHUNK_PAGES - OVERLAP_PAGES

        total_chars = sum(len(p) for p in pages)
        self._trace(
            f"Document: {len(pages)} pages, ~{total_chars:,} chars "
            f"→ split into {len(chunks)} chunks of ~{CHUNK_PAGES} pages."
        )
        return self._map_reduce(chunks, filename)

    def _run_text(self, file_info: dict, filename: str) -> AgentResult:
        # Support old format (has "text" key) and new format (has "full_text" key)
        text = file_info.get("full_text") or file_info.get("text", "")
        if not text.strip():
            # Last resort: stitch pages if present
            pages = file_info.get("pages", [])
            text = "\n\n".join(p for p in pages if p.strip())
        if not text.strip():
            return AgentResult(
                success=False, text_response="",
                error_message="Could not extract text from the uploaded file.",
            )

        # Build overlapping text chunks
        chunks: list[tuple[int, int, str]] = []
        i = 0
        chunk_num = 1
        while i < len(text):
            end = min(i + CHUNK_CHARS, len(text))
            chunk_text = text[i:end]
            if chunk_text.strip():
                chunks.append((chunk_num, chunk_num, chunk_text))
                chunk_num += 1
            if end >= len(text):
                break
            i += CHUNK_CHARS - OVERLAP_CHARS

        self._trace(
            f"Document: {len(text):,} chars "
            f"→ split into {len(chunks)} text chunks of ~{CHUNK_CHARS:,} chars."
        )
        return self._map_reduce(chunks, filename)

    # ------------------------------------------------------------------
    # Map phase: extract each chunk
    # ------------------------------------------------------------------

    def _map_reduce(self, chunks: list[tuple[int, int, str]], filename: str) -> AgentResult:
        total = len(chunks)
        if total == 0:
            return AgentResult(
                success=False, text_response="",
                error_message="The document appears to be empty after parsing.",
            )

        self._trace(f"Phase 1 of 2 — Extracting key content from {total} chunk(s)...")

        extractions: list[str] = []
        for idx, (start_label, end_label, chunk_text) in enumerate(chunks):
            label_str = (
                f"pages {start_label}–{end_label}"
                if start_label != end_label
                else f"chunk {start_label}"
            )
            self._trace(f"  Extracting {label_str} ({len(chunk_text):,} chars)...")
            extraction = self._extract_chunk(chunk_text, filename, idx + 1, total, label_str)
            if extraction:
                extractions.append(extraction)
                self._trace(f"  Done — {len(extraction):,} chars extracted.")
            else:
                self._trace(f"  Extraction failed for {label_str} — skipping.")

        if not extractions:
            return AgentResult(
                success=False, text_response="",
                error_message="Could not extract content from any part of the document.",
            )

        # Combine extractions
        combined = "\n\n" + ("=" * 60) + "\n\n".join(extractions)
        original_len = len(combined)

        if len(combined) > MAX_COMBINED:
            # Trim each extraction proportionally to fit within the cap
            target_per = MAX_COMBINED // len(extractions)
            trimmed = [e[:target_per] for e in extractions]
            combined = "\n\n" + ("=" * 60) + "\n\n".join(trimmed)
            self._trace(
                f"Combined extractions trimmed from {original_len:,} → {len(combined):,} chars "
                f"to fit context window."
            )
        else:
            self._trace(
                f"All {len(extractions)} chunk(s) extracted. "
                f"Combined document: {len(combined):,} chars."
            )

        # Reduce phase: full analysis on combined document
        self._trace("Phase 2 of 2 — Running full protocol design analysis...")
        return self._run_final_analysis(combined, filename)

    def _extract_chunk(
        self, chunk_text: str, filename: str, chunk_num: int, total_chunks: int, label_str: str
    ) -> str | None:
        messages = [
            {"role": "system", "content": PROTOCOL_CHUNK_SYSTEM},
            {"role": "user",   "content": PROTOCOL_CHUNK_USER.format(
                filename=filename,
                chunk_num=chunk_num,
                total_chunks=total_chunks,
                label_str=label_str,
                chunk_text=chunk_text,
                max_chars=MAX_CHUNK_EXTRACTION,
            )},
        ]
        try:
            return self.llm.complete(messages, temperature=self.llm.temp_deterministic)
        except Exception as e:
            logger.error("Chunk %d extraction failed: %s", chunk_num, e)
            return None

    # ------------------------------------------------------------------
    # Reduce phase: final analysis
    # ------------------------------------------------------------------

    def _run_final_analysis(self, combined_text: str, filename: str) -> AgentResult:
        messages = [
            {"role": "system", "content": PROTOCOL_ANALYSIS_SYSTEM},
            {"role": "user",   "content": PROTOCOL_ANALYSIS_USER.format(
                filename=filename,
                protocol_text=combined_text,
            )},
        ]
        try:
            data = self.llm.complete_json(messages, temperature=self.llm.temp_agents)
        except Exception as e:
            logger.error("Final protocol analysis failed: %s", e)
            return AgentResult(
                success=False, text_response="",
                error_message=f"Protocol analysis failed: {e}",
            )
        self._trace("Analysis complete.")
        return self._format_result(data, filename)

    # ------------------------------------------------------------------
    # Format final output
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
    Parse a protocol file and return a format-specific dict.

    PDF  → {"filename", "format":"pdf", "pages":[str], "total_pages":int}
    DOCX → {"filename", "format":"docx", "full_text":str}
    TXT  → {"filename", "format":"txt",  "full_text":str}

    Raises ValueError for unsupported formats or extraction failures.
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
        doc        = Document(io.BytesIO(raw))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        text       = "\n\n".join(paragraphs)
        if not text:
            raise ValueError(f"No text extracted from '{filename}'.")
        return {"filename": filename, "format": "docx", "full_text": text}
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Failed to read DOCX '{filename}': {e}")


def _parse_text(raw: bytes, filename: str) -> dict:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        try:
            text = raw.decode("latin-1")
        except Exception:
            raise ValueError(
                f"Could not read '{filename}' as text. Supported formats: PDF, DOCX, TXT."
            )
    if not text.strip():
        raise ValueError(f"File '{filename}' appears to be empty.")
    return {"filename": filename, "format": "txt", "full_text": text}
