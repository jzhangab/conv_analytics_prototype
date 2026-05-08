"""
Protocol Analysis SubAgent — TOC-based section extraction.

Workflow:
  1. Scan the first TOC_SCAN_PAGES pages (PDF) or TOC_SCAN_CHARS chars (DOCX/TXT)
     and call the LLM to locate the Table of Contents and map each analysis
     dimension to a section title and start page.
  2. For each dimension, extract only the pages (PDF) or text span (DOCX/TXT)
     belonging to that section — no more, no less.
  3. Call a dimension-specific expert LLM on each section independently.
     Each call is a few thousand to tens-of-thousands of tokens — well within limits.
  4. Synthesize findings: merge all section results programmatically, then make
     one lightweight LLM call (section assessments only, no protocol text) to
     produce the executive summary and overall rating.

Fallback: if no TOC is found, fall back to a single-call analysis on as much
text as fits within FALLBACK_CHARS.
"""
from __future__ import annotations

import io
import logging

from backend.agents.base_agent import AgentResult, BaseAgent
from backend.llm.llm_client import LLMClient
from backend.llm.prompt_templates import (
    PROTOCOL_ANALYSIS_SYSTEM,
    PROTOCOL_ANALYSIS_USER,
    PROTOCOL_DESIGN_SYSTEM,
    PROTOCOL_DESIGN_USER,
    PROTOCOL_OBJECTIVES_SYSTEM,
    PROTOCOL_OBJECTIVES_USER,
    PROTOCOL_OPERATIONAL_SYSTEM,
    PROTOCOL_OPERATIONAL_USER,
    PROTOCOL_POPULATION_SYSTEM,
    PROTOCOL_POPULATION_USER,
    PROTOCOL_SAFETY_SYSTEM,
    PROTOCOL_SAFETY_USER,
    PROTOCOL_SECTION_SYNTHESIS_SYSTEM,
    PROTOCOL_SECTION_SYNTHESIS_USER,
    PROTOCOL_STATISTICAL_SYSTEM,
    PROTOCOL_STATISTICAL_USER,
    PROTOCOL_TOC_SYSTEM,
    PROTOCOL_TOC_USER,
)
from backend.llm.web_search import WebSearchClient
from backend.state.conversation_state import ConversationState

logger = logging.getLogger(__name__)

TOC_SCAN_PAGES  = 15         # PDF pages to scan for Table of Contents
TOC_SCAN_CHARS  = 30_000     # chars to scan for TOC in DOCX/TXT
MAX_SECTION_CHARS = 500_000  # hard cap per section analysis call (~167k tokens at 3 ch/tok)
FALLBACK_CHARS  = 600_000    # max chars for single-call fallback analysis

SEVERITY_ORDER = {"critical": 0, "major": 1, "minor": 2, "suggestion": 3}

DIMENSION_LABELS = {
    "objectives_and_endpoints": "Endpoints & Estimands",
    "trial_design":             "Study Design",
    "trial_population":         "Eligibility Criteria",
    "statistical_approach":     "Statistical Approach",
    "safety_monitoring":        "Safety Monitoring",
    "operational_feasibility":  "Operational Feasibility",
}

# Ordered list of dimensions — controls analysis sequence and display order
ANALYSIS_DIMENSIONS = list(DIMENSION_LABELS.keys())

# Dimension → (system_prompt, user_template) — populated once at module load
_SECTION_PROMPTS: dict[str, tuple[str, str]] = {
    "objectives_and_endpoints": (PROTOCOL_OBJECTIVES_SYSTEM,  PROTOCOL_OBJECTIVES_USER),
    "trial_design":             (PROTOCOL_DESIGN_SYSTEM,       PROTOCOL_DESIGN_USER),
    "trial_population":         (PROTOCOL_POPULATION_SYSTEM,   PROTOCOL_POPULATION_USER),
    "statistical_approach":     (PROTOCOL_STATISTICAL_SYSTEM,  PROTOCOL_STATISTICAL_USER),
    "safety_monitoring":        (PROTOCOL_SAFETY_SYSTEM,       PROTOCOL_SAFETY_USER),
    "operational_feasibility":  (PROTOCOL_OPERATIONAL_SYSTEM,  PROTOCOL_OPERATIONAL_USER),
}


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

        self._trace(f"Loaded '{filename}' (format: {fmt})")

        # Web search for regulatory context
        web_context = ""
        if self.web_search:
            hint = filename.replace("_", " ").rsplit(".", 1)[0]
            raw = self.web_search.search_for_skill("protocol_analysis", {"indication": hint})
            if raw:
                web_context = f"\n\nSupplementary regulatory guidance from web search:\n{raw}"
                self._trace("Web search context added.")

        # ── Step 1: Extract TOC ─────────────────────────────────────────
        toc_input = self._get_toc_input(file_info, fmt)
        self._trace(
            f"Scanning first {TOC_SCAN_PAGES} pages (PDF) / {TOC_SCAN_CHARS:,} chars (text) for Table of Contents..."
        )
        toc = self._call_toc_llm(toc_input, filename)

        if toc and toc.get("found"):
            mapped = [
                dim for dim in ANALYSIS_DIMENSIONS
                if toc.get("sections", {}).get(dim, {}).get("section_title")
            ]
            self._trace(
                f"TOC found. Mapped {len(mapped)}/{len(ANALYSIS_DIMENSIONS)} analysis dimensions: "
                + ", ".join(mapped)
            )
            return self._run_toc_analysis(file_info, fmt, toc, filename, web_context)
        else:
            self._trace(
                "No Table of Contents found in first pages. "
                "Falling back to full-text single-call analysis."
            )
            full_text = self._assemble_full_text(file_info, fmt)
            if len(full_text) > FALLBACK_CHARS:
                self._trace(
                    f"Document truncated to {FALLBACK_CHARS:,} chars for fallback analysis "
                    f"(was {len(full_text):,} chars)."
                )
                full_text = full_text[:FALLBACK_CHARS]
            return self._run_single_call(full_text + web_context, filename)

    # ------------------------------------------------------------------
    # TOC extraction
    # ------------------------------------------------------------------

    def _get_toc_input(self, file_info: dict, fmt: str) -> str:
        """Return the text to send to the TOC LLM (first pages / chars)."""
        if fmt == "pdf":
            pages = file_info.get("pages", [])
            toc_pages = pages[:TOC_SCAN_PAGES]
            return "\n\n--- Page Break ---\n\n".join(
                f"[PDF page {i + 1}]\n{p}" for i, p in enumerate(toc_pages) if p.strip()
            )
        else:
            full_text = file_info.get("full_text") or file_info.get("text", "")
            return full_text[:TOC_SCAN_CHARS]

    def _call_toc_llm(self, toc_input: str, filename: str) -> dict | None:
        messages = [
            {"role": "system", "content": PROTOCOL_TOC_SYSTEM},
            {"role": "user",   "content": PROTOCOL_TOC_USER.format(toc_pages_text=toc_input)},
        ]
        try:
            result = self.llm.complete_json(messages, temperature=self.llm.temp_deterministic)
            return result if isinstance(result, dict) else None
        except Exception as e:
            logger.warning("TOC LLM call failed: %s", e)
            return None

    # ------------------------------------------------------------------
    # TOC-based analysis (main path)
    # ------------------------------------------------------------------

    def _run_toc_analysis(
        self, file_info: dict, fmt: str, toc: dict, filename: str, web_context: str
    ) -> AgentResult:
        sections_map = toc.get("sections", {})
        section_results: dict[str, dict] = {}

        for dimension in ANALYSIS_DIMENSIONS:
            section_info = sections_map.get(dimension) or {}
            section_title = section_info.get("section_title")

            if not section_title:
                self._trace(f"  [{dimension}] Not mapped in TOC — skipping.")
                continue

            # Extract the section text for this dimension
            if fmt == "pdf":
                section_text = self._extract_section_pages(
                    file_info["pages"], toc, dimension
                )
            else:
                full_text = file_info.get("full_text") or file_info.get("text", "")
                section_text = self._extract_section_from_text(full_text, toc, dimension)

            if not section_text:
                self._trace(
                    f"  [{dimension}] Could not extract text for '{section_title}' — skipping."
                )
                continue

            self._trace(
                f"  [{dimension}] Analyzing '{section_title}' "
                f"({len(section_text):,} chars)..."
            )
            result = self._analyze_section(section_text, dimension, section_title, filename)
            if result:
                n = len(result.get("findings", []))
                self._trace(f"  [{dimension}] → {n} finding(s).")
                section_results[dimension] = result
            else:
                self._trace(f"  [{dimension}] Analysis call failed — skipping.")

        if not section_results:
            self._trace(
                "No sections could be extracted or analyzed. "
                "Falling back to full-text single-call analysis."
            )
            full_text = self._assemble_full_text(file_info, fmt)
            return self._run_single_call(full_text[:FALLBACK_CHARS] + web_context, filename)

        return self._format_from_sections(section_results, filename)

    # ------------------------------------------------------------------
    # Section extraction — PDF (page-list based)
    # ------------------------------------------------------------------

    def _extract_section_pages(
        self, pages: list[str], toc: dict, dimension: str
    ) -> str | None:
        section_info = (toc.get("sections") or {}).get(dimension) or {}
        start_page = section_info.get("protocol_page")
        if not start_page:
            return None

        page_offset = toc.get("page_offset", 0)
        all_sections_sorted = sorted(
            [s for s in toc.get("all_sections", []) if s.get("protocol_page")],
            key=lambda s: s["protocol_page"],
        )

        # End page = next section start in the full TOC (after our section's start)
        end_page = None
        for s in all_sections_sorted:
            if s["protocol_page"] > start_page:
                end_page = s["protocol_page"]
                break

        start_idx = max(0, (start_page - 1) + page_offset)
        end_idx   = min(
            len(pages),
            ((end_page - 1) + page_offset) if end_page is not None else len(pages),
        )

        if start_idx >= len(pages):
            self._trace(
                f"    Page index {start_idx} out of range "
                f"(PDF has {len(pages)} pages, offset={page_offset}). "
                "Trying without offset."
            )
            start_idx = max(0, start_page - 1)
            end_idx   = min(len(pages), (end_page - 1) if end_page else len(pages))

        section_pages = [p for p in pages[start_idx:end_idx] if p.strip()]
        if not section_pages:
            return None

        text = "\n\n".join(section_pages)
        if len(text) > MAX_SECTION_CHARS:
            text = text[:MAX_SECTION_CHARS]
            self._trace(f"    Truncated to {MAX_SECTION_CHARS:,} chars.")
        return text

    # ------------------------------------------------------------------
    # Section extraction — DOCX/TXT (heading-search based)
    # ------------------------------------------------------------------

    def _extract_section_from_text(
        self, full_text: str, toc: dict, dimension: str
    ) -> str | None:
        section_info = (toc.get("sections") or {}).get(dimension) or {}
        title = section_info.get("section_title", "")
        if not title:
            return None

        start_pos = self._find_heading_in_text(full_text, title)
        if start_pos < 0:
            return None

        # Find end: next section title that appears after our section starts
        all_sections_sorted = sorted(
            [s for s in toc.get("all_sections", []) if s.get("title") and s.get("protocol_page")],
            key=lambda s: s["protocol_page"],
        )
        my_page = section_info.get("protocol_page", 0)

        end_pos = len(full_text)
        for s in all_sections_sorted:
            if s["protocol_page"] <= my_page:
                continue
            candidate = self._find_heading_in_text(full_text, s["title"], after=start_pos + len(title))
            if candidate > start_pos:
                end_pos = candidate
                break

        text = full_text[start_pos:end_pos]
        if len(text) > MAX_SECTION_CHARS:
            text = text[:MAX_SECTION_CHARS]
        return text if text.strip() else None

    @staticmethod
    def _find_heading_in_text(text: str, heading: str, after: int = 0) -> int:
        """
        Case-insensitive search for `heading` near a line boundary.
        Accepts up to 20 characters of prefix on the line (e.g., "3.2 " or "Section ").
        """
        lower_text    = text.lower()
        lower_heading = heading.lower().strip()
        pos = after
        while pos < len(text):
            idx = lower_text.find(lower_heading, pos)
            if idx < 0:
                return -1
            line_start = text.rfind("\n", 0, idx) + 1
            if idx - line_start <= 20:
                return idx
            pos = idx + 1
        return -1

    # ------------------------------------------------------------------
    # Section analysis (one call per dimension)
    # ------------------------------------------------------------------

    def _analyze_section(
        self, section_text: str, dimension: str, section_title: str, filename: str
    ) -> dict | None:
        system_prompt, user_template = _SECTION_PROMPTS.get(dimension, (None, None))
        if not system_prompt:
            return None
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_template.format(
                filename=filename,
                section_label=section_title,
                section_text=section_text,
            )},
        ]
        try:
            return self.llm.complete_json(messages, temperature=self.llm.temp_agents)
        except Exception as e:
            logger.warning("Section analysis failed for %s: %s", dimension, e)
            return None

    # ------------------------------------------------------------------
    # Synthesis: merge section results → AgentResult
    # ------------------------------------------------------------------

    def _format_from_sections(
        self, section_results: dict[str, dict], filename: str
    ) -> AgentResult:
        all_findings: list[dict] = []
        all_strengths: list[str] = []
        section_assessments: dict[str, str] = {}
        severity_counts: dict[str, int] = {"critical": 0, "major": 0, "minor": 0, "suggestion": 0}

        for dimension in ANALYSIS_DIMENSIONS:
            result = section_results.get(dimension)
            if not result:
                continue

            category  = DIMENSION_LABELS.get(dimension, dimension.replace("_", " ").title())
            assessment = result.get("assessment", "")
            if assessment:
                section_assessments[dimension] = assessment

            for strength in result.get("strengths", []):
                all_strengths.append(strength)

            for f in result.get("findings", []):
                sev = f.get("severity", "suggestion")
                severity_counts[sev] = severity_counts.get(sev, 0) + 1
                all_findings.append({
                    "category":       category,
                    "finding":        f.get("finding", ""),
                    "severity":       sev,
                    "recommendation": f.get("recommendation", ""),
                })

        all_findings.sort(key=lambda f: SEVERITY_ORDER.get(f["severity"], 3))
        critical_concerns = [f["finding"] for f in all_findings if f["severity"] == "critical"]

        exec_summary, overall_rating = self._synthesize_summary(
            section_assessments, severity_counts, filename
        )

        # Build text response
        lines = [
            f"## Protocol Analysis: {filename}",
            f"**Overall Rating:** {overall_rating.replace('_', ' ').title()}",
            "",
            "### Executive Summary",
            exec_summary,
        ]
        if all_strengths:
            lines += ["", "### Strengths", *[f"- {s}" for s in all_strengths[:6]]]
        if critical_concerns:
            lines += ["", "### Critical Concerns", *[f"- {c}" for c in critical_concerns]]
        if section_assessments:
            lines += ["", "### Section Assessments"]
            for dim, assessment in section_assessments.items():
                label = DIMENSION_LABELS.get(dim, dim.replace("_", " ").title())
                lines.append(f"**{label}:** {assessment}")

        table_data = [
            {
                "#":              i + 1,
                "Section":        f["category"],
                "Finding":        f["finding"],
                "Severity":       f["severity"].title(),
                "Recommendation": f["recommendation"],
            }
            for i, f in enumerate(all_findings)
        ]
        return AgentResult(
            success=True,
            text_response="\n".join(lines),
            table_data=table_data or None,
            table_columns=(
                ["#", "Section", "Finding", "Severity", "Recommendation"]
                if table_data else None
            ),
        )

    def _synthesize_summary(
        self,
        section_assessments: dict[str, str],
        severity_counts: dict[str, int],
        filename: str,
    ) -> tuple[str, str]:
        """
        One-shot lightweight LLM call: sends only section summaries (not protocol text)
        to produce executive_summary and overall_rating.
        """
        assessments_text = "\n".join(
            f"- {DIMENSION_LABELS.get(dim, dim)}: {assessment}"
            for dim, assessment in section_assessments.items()
        )
        counts_text = ", ".join(
            f"{sev}: {cnt}" for sev, cnt in severity_counts.items() if cnt > 0
        )
        messages = [
            {"role": "system", "content": PROTOCOL_SECTION_SYNTHESIS_SYSTEM},
            {"role": "user",   "content": PROTOCOL_SECTION_SYNTHESIS_USER.format(
                filename=filename,
                section_assessments=assessments_text,
                severity_counts=counts_text,
            )},
        ]
        try:
            result = self.llm.complete_json(messages, temperature=self.llm.temp_agents)
            return (
                result.get("executive_summary", "Protocol review completed."),
                result.get("overall_rating", "adequate"),
            )
        except Exception as e:
            logger.warning("Summary synthesis failed: %s", e)
            # Deterministic fallback from severity counts
            if severity_counts.get("critical", 0) > 0:
                rating = "significant_concerns"
            elif severity_counts.get("major", 0) >= 3:
                rating = "needs_improvement"
            elif severity_counts.get("major", 0) >= 1:
                rating = "adequate"
            else:
                rating = "strong"
            return "Protocol review completed.", rating

    # ------------------------------------------------------------------
    # Single-call fallback (short documents or no TOC found)
    # ------------------------------------------------------------------

    def _run_single_call(self, protocol_text: str, filename: str) -> AgentResult:
        self._trace(f"Running single-call analysis ({len(protocol_text):,} chars)...")
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
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _assemble_full_text(file_info: dict, fmt: str) -> str:
        if fmt == "pdf":
            pages = file_info.get("pages", [])
            return "\n\n".join(p for p in pages if p.strip())
        return file_info.get("full_text") or file_info.get("text", "")

    def _format_result(self, data: dict, filename: str) -> AgentResult:
        """Format single-call analysis result (unchanged from original)."""
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
            table_columns=(
                ["#", "Section", "Finding", "Severity", "Recommendation"]
                if table_data else None
            ),
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
        doc  = Document(io.BytesIO(raw))
        text = "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
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
