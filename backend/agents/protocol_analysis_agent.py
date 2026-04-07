"""
Protocol Analysis SubAgent — TOC-driven, per-section LLM analysis.

PDF workflow:
  1. Extract text from first 10 PDF pages → LLM locates Table of Contents
  2. Parse TOC to identify start pages for: Objectives/Endpoints, Trial Design, Trial Population
  3. Build a protocol-page → PDF-index map (via page labels or text scanning)
  4. Extract each section's text using page boundaries from the TOC
  5. Run a focused LLM analysis on each section independently
  6. Combine all findings into one structured result

DOCX / TXT fallback: full-text single-call analysis using the general prompt.
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
    PROTOCOL_DESIGN_SYSTEM,
    PROTOCOL_DESIGN_USER,
    PROTOCOL_OBJECTIVES_SYSTEM,
    PROTOCOL_OBJECTIVES_USER,
    PROTOCOL_POPULATION_SYSTEM,
    PROTOCOL_POPULATION_USER,
    PROTOCOL_TOC_SYSTEM,
    PROTOCOL_TOC_USER,
)
from backend.state.conversation_state import ConversationState

logger = logging.getLogger(__name__)

MAX_SECTION_CHARS = 20_000   # per-section cap sent to LLM (~5 k tokens)
MAX_FULL_TEXT_CHARS = 40_000 # fallback cap for DOCX / TXT

SEVERITY_ORDER = {"critical": 0, "major": 1, "minor": 2, "suggestion": 3}

SECTION_LABELS = {
    "objectives_and_endpoints": "Objectives & Endpoints",
    "trial_design":             "Trial Design",
    "trial_population":         "Trial Population",
}

SECTION_PROMPTS = {
    "objectives_and_endpoints": (PROTOCOL_OBJECTIVES_SYSTEM, PROTOCOL_OBJECTIVES_USER),
    "trial_design":             (PROTOCOL_DESIGN_SYSTEM,      PROTOCOL_DESIGN_USER),
    "trial_population":         (PROTOCOL_POPULATION_SYSTEM,  PROTOCOL_POPULATION_USER),
}

TARGET_SECTIONS = ["objectives_and_endpoints", "trial_design", "trial_population"]


class ProtocolAnalysisAgent(BaseAgent):
    skill_id     = "protocol_analysis"
    display_name = "Clinical Trial Protocol Analysis"
    description  = "Analyses an uploaded protocol document for study design improvements."

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

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

        fmt      = file_info.get("format", "txt")
        filename = file_info["filename"]

        if fmt == "pdf":
            return self._run_pdf(file_info, filename)
        return self._run_full_text(file_info, filename)

    # ------------------------------------------------------------------
    # PDF workflow
    # ------------------------------------------------------------------

    def _run_pdf(self, file_info: dict, filename: str) -> AgentResult:
        # Step 1: locate Table of Contents
        logger.info("Searching for TOC in first 10 pages of %s", filename)
        toc_result = self._find_toc(file_info)

        if not toc_result or not toc_result.get("found"):
            notes = (toc_result or {}).get("notes", "")
            return AgentResult(
                success=False, text_response="",
                error_message=(
                    f"Could not find a Table of Contents in the first 10 pages of '{filename}'. "
                    + (f"Note: {notes}. " if notes else "")
                    + "Please ensure the protocol has a TOC within its first 10 pages."
                ),
            )

        sections_info  = toc_result.get("sections", {})
        found_sections = [
            k for k in TARGET_SECTIONS
            if isinstance((sections_info.get(k) or {}).get("protocol_page"), int)
        ]

        if not found_sections:
            return AgentResult(
                success=False, text_response="",
                error_message=(
                    "A TOC was found but none of the target sections (Objectives/Endpoints, "
                    "Trial Design, Trial Population) could be located within it."
                ),
            )

        # Step 2: extract section text
        logger.info("Extracting sections from %s: %s", filename, found_sections)
        sections = self._extract_sections(file_info, toc_result)

        # Step 3: per-section LLM analysis
        section_analyses: dict[str, dict] = {}
        for key in TARGET_SECTIONS:
            text = sections.get(key)
            if not text:
                logger.info("Section '%s' not extracted — skipping", key)
                continue
            logger.info("Analysing '%s' (%d chars)", key, len(text))
            analysis = self._analyze_section(key, text, filename)
            if analysis:
                section_analyses[key] = analysis

        if not section_analyses:
            return AgentResult(
                success=False, text_response="",
                error_message="Could not extract or analyse any of the target protocol sections.",
            )

        return self._build_combined_result(section_analyses, filename, toc_result)

    # ── TOC extraction ────────────────────────────────────────────────

    def _find_toc(self, file_info: dict) -> dict | None:
        pages = file_info.get("pages", [])
        toc_pages = pages[:10]
        if not toc_pages:
            return None

        toc_text = "\n\n".join(
            f"=== PDF Scan Page {i + 1} ===\n{text}"
            for i, text in enumerate(toc_pages)
            if text.strip()
        )
        messages = [
            {"role": "system", "content": PROTOCOL_TOC_SYSTEM},
            {"role": "user",   "content": PROTOCOL_TOC_USER.format(toc_pages_text=toc_text)},
        ]
        try:
            return self.llm.complete_json(messages, temperature=self.llm.temp_deterministic)
        except Exception as e:
            logger.error("TOC extraction failed: %s", e)
            return None

    # ── Section text extraction ───────────────────────────────────────

    def _extract_sections(self, file_info: dict, toc_result: dict) -> dict[str, str | None]:
        sections_info = toc_result.get("sections", {})
        all_sections  = toc_result.get("all_sections", [])
        pages         = file_info.get("pages", [])
        page_index    = file_info.get("page_index", {})
        total         = len(pages)

        # Sort every TOC entry by page number for boundary detection
        ordered = sorted(
            [s for s in all_sections if isinstance(s.get("protocol_page"), int)],
            key=lambda s: s["protocol_page"],
        )

        result: dict[str, str | None] = {}
        for key in TARGET_SECTIONS:
            info    = (sections_info.get(key) or {})
            start_p = info.get("protocol_page")
            if not isinstance(start_p, int):
                result[key] = None
                continue

            # End page = first TOC entry whose page number > start_p
            end_p: int | None = None
            for s in ordered:
                if s["protocol_page"] > start_p:
                    end_p = s["protocol_page"]
                    break

            start_idx = self._protocol_page_to_idx(start_p, page_index, total)
            if start_idx is None:
                logger.warning("Cannot map protocol page %d to PDF index", start_p)
                result[key] = None
                continue

            if end_p is not None:
                end_idx = self._protocol_page_to_idx(end_p, page_index, total)
                if end_idx is None:
                    end_idx = min(start_idx + 25, total)
            else:
                end_idx = min(start_idx + 25, total)

            text = "\n\n".join(p for p in pages[start_idx:end_idx] if p.strip())
            if len(text) > MAX_SECTION_CHARS:
                text = (
                    text[:MAX_SECTION_CHARS]
                    + f"\n\n[Truncated — showing first {MAX_SECTION_CHARS:,} chars of this section]"
                )
            result[key] = text or None

        return result

    @staticmethod
    def _protocol_page_to_idx(protocol_page: int, page_index: dict, total: int) -> int | None:
        """Map a protocol page number to a 0-based PDF index, tolerating small offsets."""
        idx = page_index.get(protocol_page)
        if idx is not None:
            return min(idx, total - 1)
        for delta in (1, -1, 2, -2, 3):
            idx = page_index.get(protocol_page + delta)
            if idx is not None:
                return min(idx, total - 1)
        return None

    # ── Per-section analysis ─────────────────────────────────────────

    def _analyze_section(self, section_key: str, text: str, filename: str) -> dict | None:
        system_prompt, user_prompt = SECTION_PROMPTS[section_key]
        label = SECTION_LABELS[section_key]
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt.format(
                filename=filename,
                section_label=label,
                section_text=text,
            )},
        ]
        try:
            return self.llm.complete_json(messages, temperature=self.llm.temp_agents)
        except Exception as e:
            logger.error("Section analysis failed for '%s': %s", section_key, e)
            return None

    # ── Combine results ───────────────────────────────────────────────

    def _build_combined_result(
        self,
        section_analyses: dict[str, dict],
        filename: str,
        toc_result: dict,
    ) -> AgentResult:
        all_findings: list[dict] = []
        section_assessments: dict[str, str] = {}
        all_strengths: list[str] = []

        for key, analysis in section_analyses.items():
            label      = SECTION_LABELS.get(key, key)
            assessment = analysis.get("assessment", "")
            if assessment:
                section_assessments[label] = assessment
            for s in analysis.get("strengths", []):
                all_strengths.append(f"[{label}] {s}")
            for f in analysis.get("findings", []):
                all_findings.append({**f, "section": label})

        # Sort by severity
        all_findings.sort(key=lambda f: SEVERITY_ORDER.get(f.get("severity", "suggestion"), 3))

        # Overall rating from worst severity present
        if any(f.get("severity") == "critical" for f in all_findings):
            overall_rating = "Significant Concerns"
        elif any(f.get("severity") == "major" for f in all_findings):
            overall_rating = "Needs Improvement"
        elif all_findings:
            overall_rating = "Adequate"
        else:
            overall_rating = "Strong"

        critical_concerns = [
            f"[{f['section']}] {f['finding']}"
            for f in all_findings if f.get("severity") == "critical"
        ]

        sections_label = ", ".join(
            SECTION_LABELS[k] for k in section_analyses if k in SECTION_LABELS
        )

        lines = [
            f"## Protocol Analysis: {filename}",
            f"**Overall Rating:** {overall_rating}",
            f"**Sections Analysed:** {sections_label}",
            "",
        ]
        if all_strengths:
            lines += ["### Strengths", *[f"- {s}" for s in all_strengths], ""]
        if critical_concerns:
            lines += ["### Critical Concerns", *[f"- {c}" for c in critical_concerns], ""]
        if section_assessments:
            lines += ["### Section Assessments"]
            for label, assessment in section_assessments.items():
                lines.append(f"**{label}:** {assessment}")
            lines.append("")

        table_data = [
            {
                "#":              i + 1,
                "Section":        f.get("section", ""),
                "Finding":        f.get("finding", ""),
                "Severity":       f.get("severity", "").title(),
                "Recommendation": f.get("recommendation", ""),
            }
            for i, f in enumerate(all_findings)
        ]
        table_columns = ["#", "Section", "Finding", "Severity", "Recommendation"]

        return AgentResult(
            success=True,
            text_response="\n".join(lines),
            table_data=table_data if table_data else None,
            table_columns=table_columns if table_data else None,
        )

    # ------------------------------------------------------------------
    # DOCX / TXT fallback — full-text single LLM call
    # ------------------------------------------------------------------

    def _run_full_text(self, file_info: dict, filename: str) -> AgentResult:
        text = file_info.get("full_text", "")
        if not text:
            return AgentResult(
                success=False, text_response="",
                error_message="Could not extract text from the uploaded file.",
            )
        if len(text) > MAX_FULL_TEXT_CHARS:
            text = text[:MAX_FULL_TEXT_CHARS] + f"\n\n[Truncated to {MAX_FULL_TEXT_CHARS:,} chars]"

        messages = [
            {"role": "system", "content": PROTOCOL_ANALYSIS_SYSTEM},
            {"role": "user",   "content": PROTOCOL_ANALYSIS_USER.format(
                filename=filename, protocol_text=text,
            )},
        ]
        try:
            data = self.llm.complete_json(messages, temperature=self.llm.temp_agents)
        except Exception as e:
            logger.error("Full-text protocol analysis failed: %s", e)
            return AgentResult(
                success=False, text_response="",
                error_message=f"Protocol analysis failed: {e}",
            )
        return self._format_full_text_result(data, filename)

    def _format_full_text_result(self, data: dict, filename: str) -> AgentResult:
        rating   = data.get("overall_rating", "").replace("_", " ").title()
        findings = sorted(
            data.get("findings", []),
            key=lambda f: SEVERITY_ORDER.get(f.get("severity", "suggestion"), 3),
        )
        lines = [
            f"## Protocol Analysis: {filename}",
            f"**Overall Rating:** {rating}",
            "", "### Executive Summary", data.get("executive_summary", ""),
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

    PDF  → {"filename", "format":"pdf", "pages":[str], "page_index":{int:int}, "total_pages":int}
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

    page_index = _build_page_index(reader, pages)
    return {
        "filename":    filename,
        "format":      "pdf",
        "pages":       pages,
        "page_index":  page_index,
        "total_pages": len(pages),
    }


def _build_page_index(reader, pages: list[str]) -> dict[int, int]:
    """
    Build mapping: protocol page number (int) → 0-based PDF page index.

    Priority:
    1. PDF page labels embedded in the file (most reliable)
    2. Scan each page for a standalone integer at top or bottom
    3. Identity map (1-indexed, no front matter assumed)
    """
    # Method 1: PDF page labels
    try:
        labels = list(reader.page_labels)
        mapping: dict[int, int] = {}
        for i, label in enumerate(labels):
            try:
                num = int(label)
                if num not in mapping:
                    mapping[num] = i
            except (ValueError, TypeError):
                pass
        if mapping:
            return mapping
    except Exception:
        pass

    # Method 2: Scan pages for standalone integers in first/last lines
    mapping = {}
    for i, text in enumerate(pages):
        lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
        if not lines:
            continue
        for candidate in lines[:3] + lines[-3:]:
            if re.match(r"^\d+$", candidate):
                num = int(candidate)
                if num not in mapping:
                    mapping[num] = i
                break

    if mapping:
        return mapping

    # Method 3: Identity (1-indexed, assume no front matter)
    return {i + 1: i for i in range(len(pages))}


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
