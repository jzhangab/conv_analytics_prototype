"""
Protocol Analysis SubAgent.
Accepts an uploaded clinical trial protocol (PDF, DOCX, or TXT) and uses the LLM
to identify study design weaknesses and produce prioritised recommendations.
"""
from __future__ import annotations

import io
import logging

from backend.agents.base_agent import AgentResult, BaseAgent
from backend.llm.llm_client import LLMClient
from backend.llm.prompt_templates import PROTOCOL_ANALYSIS_SYSTEM, PROTOCOL_ANALYSIS_USER
from backend.state.conversation_state import ConversationState

logger = logging.getLogger(__name__)

# ~40 000 chars ≈ 10 000 tokens — generous enough for a full protocol
MAX_PROTOCOL_CHARS = 40_000

SEVERITY_ORDER = {"critical": 0, "major": 1, "minor": 2, "suggestion": 3}


class ProtocolAnalysisAgent(BaseAgent):
    skill_id = "protocol_analysis"
    display_name = "Clinical Trial Protocol Analysis"
    description = "Analyses an uploaded protocol document for study design improvements."

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def run(self, params: dict, state: ConversationState) -> AgentResult:
        file_info = state.uploaded_files.get("protocol_file")
        if not file_info:
            return AgentResult(
                success=False,
                text_response="",
                error_message="No protocol file found. Please upload a PDF, DOCX, or TXT protocol first.",
            )

        protocol_text = file_info["text"]
        filename = file_info["filename"]
        was_truncated = file_info.get("was_truncated", False)
        char_count = file_info.get("char_count", len(protocol_text))

        truncation_note = (
            f"\n\n*Note: The protocol was truncated to {MAX_PROTOCOL_CHARS:,} characters "
            f"(full document: {char_count:,} characters). Analysis covers the portion provided.*"
            if was_truncated else ""
        )

        messages = [
            {"role": "system", "content": PROTOCOL_ANALYSIS_SYSTEM},
            {"role": "user", "content": PROTOCOL_ANALYSIS_USER.format(
                filename=filename,
                protocol_text=protocol_text,
            )},
        ]

        try:
            data = self.llm.complete_json(messages, temperature=self.llm.temp_agents)
        except Exception as e:
            logger.error("Protocol analysis LLM call failed: %s", e)
            return AgentResult(
                success=False,
                text_response="",
                error_message=f"Protocol analysis failed: {e}",
            )

        # ── Build text response ───────────────────────────────────────────────
        rating = data.get("overall_rating", "").replace("_", " ").title()
        exec_summary = data.get("executive_summary", "")
        strengths = data.get("strengths", [])
        critical_concerns = data.get("critical_concerns", [])
        findings = data.get("findings", [])
        section_assessments = data.get("section_assessments", {})

        lines = [
            f"## Protocol Analysis: {filename}",
            f"**Overall Rating:** {rating}{truncation_note}",
            "",
            f"### Executive Summary",
            exec_summary,
        ]

        if strengths:
            lines += ["", "### Strengths"]
            lines += [f"- {s}" for s in strengths]

        if critical_concerns:
            lines += ["", "### Critical Concerns"]
            lines += [f"- {c}" for c in critical_concerns]

        if section_assessments:
            lines += ["", "### Section Assessments"]
            section_labels = {
                "study_design": "Study Design",
                "endpoints_and_estimands": "Endpoints & Estimands",
                "inclusion_exclusion": "Inclusion/Exclusion Criteria",
                "statistical_approach": "Statistical Approach",
                "operational_feasibility": "Operational Feasibility",
                "safety_monitoring": "Safety Monitoring",
                "regulatory_alignment": "Regulatory Alignment",
            }
            for key, label in section_labels.items():
                text = section_assessments.get(key, "")
                if text:
                    lines += [f"**{label}:** {text}"]

        text_response = "\n".join(lines)

        # ── Build findings table ──────────────────────────────────────────────
        sorted_findings = sorted(
            findings,
            key=lambda f: SEVERITY_ORDER.get(f.get("severity", "suggestion"), 3),
        )

        table_data = [
            {
                "#": i + 1,
                "Category": f.get("category", ""),
                "Finding": f.get("finding", ""),
                "Severity": f.get("severity", "").title(),
                "Recommendation": f.get("recommendation", ""),
            }
            for i, f in enumerate(sorted_findings)
        ]
        table_columns = ["#", "Category", "Finding", "Severity", "Recommendation"]

        return AgentResult(
            success=True,
            text_response=text_response,
            table_data=table_data if table_data else None,
            table_columns=table_columns if table_data else None,
        )


def parse_protocol_file(file_storage) -> dict:
    """
    Extract text from a protocol file (PDF, DOCX, or TXT).
    Returns {"filename": str, "text": str, "char_count": int, "was_truncated": bool}.
    Raises ValueError for unsupported formats or extraction failures.
    """
    filename = getattr(file_storage, "filename", "") or ""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    raw = file_storage.read()

    text = _extract_text(raw, ext, filename)

    char_count = len(text)
    was_truncated = char_count > MAX_PROTOCOL_CHARS
    if was_truncated:
        text = text[:MAX_PROTOCOL_CHARS]

    return {
        "filename": filename,
        "text": text,
        "char_count": char_count,
        "was_truncated": was_truncated,
    }


def _extract_text(raw: bytes, ext: str, filename: str) -> str:
    """Dispatch to the correct extractor based on file extension."""
    if ext == "pdf":
        return _extract_pdf(raw, filename)
    if ext in ("docx",):
        return _extract_docx(raw, filename)
    if ext in ("doc",):
        raise ValueError(
            "Legacy .doc files are not supported. Please save as .docx, .pdf, or .txt."
        )
    # TXT or unknown — try UTF-8 then latin-1
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        try:
            return raw.decode("latin-1")
        except Exception:
            raise ValueError(
                f"Could not read '{filename}' as text. Supported formats: PDF, DOCX, TXT."
            )


def _extract_pdf(raw: bytes, filename: str) -> str:
    try:
        import pypdf
    except ImportError:
        raise ValueError(
            "pypdf is required to read PDF files. Install it with: pip install pypdf"
        )
    try:
        reader = pypdf.PdfReader(io.BytesIO(raw))
        pages = [page.extract_text() or "" for page in reader.pages]
        text = "\n\n".join(pages).strip()
        if not text:
            raise ValueError(
                f"No text could be extracted from '{filename}'. "
                "The PDF may be scanned/image-based. Please provide a text-selectable PDF."
            )
        return text
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Failed to read PDF '{filename}': {e}")


def _extract_docx(raw: bytes, filename: str) -> str:
    try:
        from docx import Document
    except ImportError:
        raise ValueError(
            "python-docx is required to read DOCX files. Install it with: pip install python-docx"
        )
    try:
        doc = Document(io.BytesIO(raw))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        text = "\n\n".join(paragraphs).strip()
        if not text:
            raise ValueError(
                f"No text could be extracted from '{filename}'. The document may be empty."
            )
        return text
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Failed to read DOCX '{filename}': {e}")
