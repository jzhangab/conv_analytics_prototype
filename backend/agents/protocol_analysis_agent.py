"""
Protocol Analysis SubAgent.

Workflow for short documents (≤ SINGLE_CALL_CHARS):
  1. Extract full text from the uploaded file (PDF / DOCX / TXT)
  2. Send the full text to the analysis LLM in a single call

Workflow for long documents (> SINGLE_CALL_CHARS) — Map-Reduce:
  Map:    Split into CHUNK_CHARS-sized chunks and extract structured content
          from each chunk independently (each call well within token limit).
  Reduce: Concatenate the structured extractions and run the final analysis
          on the compact, information-dense combined extraction.
          If the combined extraction still exceeds SYNTHESIS_MAX_CHARS, do
          a second compression pass before synthesizing.

Token budget (at ~3 chars/token for clinical text):
  SINGLE_CALL_CHARS = 600_000  →  ~200k tokens  (safe single call)
  CHUNK_CHARS       = 200_000  →  ~67k tokens   (per extraction call)
  SYNTHESIS_MAX_CHARS = 650_000 → ~217k tokens  (safe synthesis call)
"""
from __future__ import annotations

import io
import logging

from backend.agents.base_agent import AgentResult, BaseAgent
from backend.llm.llm_client import LLMClient
from backend.llm.prompt_templates import (
    PROTOCOL_ANALYSIS_SYSTEM,
    PROTOCOL_ANALYSIS_USER,
    PROTOCOL_CHUNK_SYSTEM,
    PROTOCOL_CHUNK_USER,
    PROTOCOL_SYNTHESIS_USER,
)
from backend.llm.web_search import WebSearchClient
from backend.state.conversation_state import ConversationState

logger = logging.getLogger(__name__)

SINGLE_CALL_CHARS   = 600_000   # use single-call path below this threshold
CHUNK_CHARS         = 200_000   # chars per extraction chunk in the map phase
SYNTHESIS_MAX_CHARS = 650_000   # max combined extraction for synthesis; compress if larger
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
            "messages": [], "response": msg,
            "synthetic": True, "label": "Protocol Analysis",
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
            self._trace(f"PDF extracted: {len(pages)} pages, {len(full_text):,} chars total.")
        else:
            full_text = file_info.get("full_text") or file_info.get("text", "")
            self._trace(f"Text extracted: {len(full_text):,} chars.")

        if not full_text.strip():
            return AgentResult(
                success=False, text_response="",
                error_message="Could not extract any text from the uploaded file.",
            )

        # Web search for regulatory context (indication-hint from filename)
        web_context = ""
        if self.web_search:
            search_hint = filename.replace("_", " ").rsplit(".", 1)[0]
            raw = self.web_search.search_for_skill("protocol_analysis", {"indication": search_hint})
            if raw:
                web_context = f"\n\nSupplementary web search results (regulatory guidance):\n{raw}"
                self._trace("Web search context added.")

        # Route to single-call or map-reduce based on document length
        if len(full_text) <= SINGLE_CALL_CHARS:
            self._trace(
                f"Document fits in single call ({len(full_text):,} / {SINGLE_CALL_CHARS:,} chars). "
                "Running direct analysis."
            )
            return self._run_single_call(full_text + web_context, filename)
        else:
            self._trace(
                f"Document too long for single call ({len(full_text):,} chars > {SINGLE_CALL_CHARS:,}). "
                "Switching to map-reduce analysis."
            )
            return self._run_chunked_analysis(full_text, web_context, filename)

    # ------------------------------------------------------------------
    # Single-call path (short documents)
    # ------------------------------------------------------------------

    def _run_single_call(self, protocol_text: str, filename: str) -> AgentResult:
        messages = [
            {"role": "system", "content": PROTOCOL_ANALYSIS_SYSTEM},
            {"role": "user",   "content": PROTOCOL_ANALYSIS_USER.format(
                filename=filename, protocol_text=protocol_text,
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
        self._trace("Single-call analysis complete.")
        return self._format_result(data, filename)

    # ------------------------------------------------------------------
    # Map-Reduce path (long documents)
    # ------------------------------------------------------------------

    def _run_chunked_analysis(
        self, full_text: str, web_context: str, filename: str
    ) -> AgentResult:
        # ── Map: extract each chunk ──────────────────────────────────────
        chunks = self._split_into_chunks(full_text, CHUNK_CHARS)
        self._trace(
            f"Split into {len(chunks)} chunk(s) of ~{CHUNK_CHARS:,} chars each. "
            "Running extraction pass (Map phase)..."
        )

        extractions: list[str] = []
        for i, chunk in enumerate(chunks):
            extraction = self._extract_chunk(chunk, i + 1, len(chunks), filename)
            extractions.append(extraction)
            self._trace(
                f"  Chunk {i+1}/{len(chunks)}: {len(chunk):,} chars → "
                f"{len(extraction):,} chars extracted."
            )

        # ── Reduce: synthesize all extractions ──────────────────────────
        combined = "\n\n---\n\n".join(extractions)
        self._trace(
            f"Map phase complete. Combined extraction: {len(combined):,} chars "
            f"(from {len(full_text):,} original chars, "
            f"{100 * len(combined) // len(full_text)}% of original)."
        )

        # Second compression pass if combined extractions are still too large
        if len(combined) > SYNTHESIS_MAX_CHARS:
            self._trace(
                f"Combined extraction ({len(combined):,} chars) exceeds synthesis limit "
                f"({SYNTHESIS_MAX_CHARS:,}). Running second compression pass..."
            )
            combined = self._compress_extractions(combined, filename)
            self._trace(
                f"Second compression complete: {len(combined):,} chars."
            )

        if web_context:
            combined += web_context

        self._trace("Running synthesis (Reduce phase)...")
        messages = [
            {"role": "system", "content": PROTOCOL_ANALYSIS_SYSTEM},
            {"role": "user",   "content": PROTOCOL_SYNTHESIS_USER.format(
                filename=filename,
                num_chunks=len(chunks),
                protocol_text=combined,
            )},
        ]
        try:
            data = self.llm.complete_json(messages, temperature=self.llm.temp_agents)
        except Exception as e:
            logger.error("Protocol synthesis failed: %s", e)
            return AgentResult(
                success=False, text_response="",
                error_message=f"Protocol synthesis failed: {e}",
            )

        self._trace("Map-reduce analysis complete.")
        return self._format_result(data, filename)

    # ------------------------------------------------------------------
    # Chunk extraction (one LLM call per chunk)
    # ------------------------------------------------------------------

    def _extract_chunk(
        self, chunk_text: str, chunk_num: int, total_chunks: int, filename: str
    ) -> str:
        pct_start = int(100 * (chunk_num - 1) / total_chunks)
        pct_end   = int(100 * chunk_num / total_chunks)
        label_str = f"{pct_start}%–{pct_end}% of document"

        messages = [
            {"role": "system", "content": PROTOCOL_CHUNK_SYSTEM},
            {"role": "user",   "content": PROTOCOL_CHUNK_USER.format(
                filename=filename,
                chunk_num=chunk_num,
                total_chunks=total_chunks,
                label_str=label_str,
                max_chars=len(chunk_text),
                chunk_text=chunk_text,
            )},
        ]
        try:
            return self.llm.complete(messages, temperature=self.llm.temp_deterministic)
        except Exception as e:
            logger.warning("Chunk %d/%d extraction failed: %s — using raw chunk", chunk_num, total_chunks, e)
            return chunk_text[:SYNTHESIS_MAX_CHARS // total_chunks]

    # ------------------------------------------------------------------
    # Second-pass compression (only for very large docs)
    # ------------------------------------------------------------------

    def _compress_extractions(self, combined: str, filename: str) -> str:
        """
        Re-chunk and re-extract the combined extractions to reduce size further.
        Used only when the first-pass combined output exceeds SYNTHESIS_MAX_CHARS.
        """
        second_chunks = self._split_into_chunks(combined, CHUNK_CHARS)
        self._trace(
            f"Second-pass compression: re-extracting {len(second_chunks)} chunk(s) "
            "from combined first-pass extractions."
        )
        second_extractions = []
        for i, chunk in enumerate(second_chunks):
            extraction = self._extract_chunk(chunk, i + 1, len(second_chunks), filename)
            second_extractions.append(extraction)

        result = "\n\n---\n\n".join(second_extractions)
        if len(result) > SYNTHESIS_MAX_CHARS:
            # Hard truncation as last resort — prefer beginning of doc (objectives/design)
            result = result[:SYNTHESIS_MAX_CHARS]
            self._trace(
                f"Hard truncation applied after second compression pass "
                f"(result still exceeded {SYNTHESIS_MAX_CHARS:,} chars)."
            )
        return result

    # ------------------------------------------------------------------
    # Chunk splitter
    # ------------------------------------------------------------------

    @staticmethod
    def _split_into_chunks(text: str, chunk_size: int) -> list[str]:
        """
        Split text into chunks of up to chunk_size chars, breaking at
        natural boundaries (triple-newline > double-newline > newline).
        """
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            if end < len(text):
                # Search for the best break point in the last 10% of the chunk
                search_from = end - chunk_size // 10
                for sep in ("\n\n\n", "\n\n", "\n"):
                    pos = text.rfind(sep, search_from, end)
                    if pos > search_from:
                        end = pos + len(sep)
                        break
            chunks.append(text[start:end])
            start = end
        return chunks

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
