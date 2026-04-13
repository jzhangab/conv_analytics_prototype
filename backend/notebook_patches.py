"""
Notebook-specific patches applied on top of the standard backend.

Call ``apply_patches(orchestrator)`` from the Dataiku notebook to:

* Add call logging to LLMClient.complete
* Register the inline site-list-matching agent (bypasses workdir cache)
* Install the JSON-repair _parse_json override
* Shorten the benchmarking system prompt
* Raise intent-classification confidence thresholds
* Add the general-knowledge fallback when no skill matches
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd

import backend
import backend.llm.prompt_templates as _pt
import backend.orchestrator.intent_classifier as _ic
from backend.agents.base_agent import AgentResult
from backend.llm.llm_client import LLMClient
from backend.orchestrator.intent_classifier import classify_intent
from backend.orchestrator.parameter_extractor import extract_parameters
from backend.state.conversation_state import FSMState
from backend.state.parameter_schema import load_schemas

logger = logging.getLogger(__name__)


# =========================================================================
# 1. LLMClient.complete — add call_log for the trace pane
# =========================================================================

def _patched_complete(self, messages, temperature=None):
    if not hasattr(self, "call_log"):
        self.call_log = []
    try:
        api_client = self._get_dataiku_client()
        project_key = __import__("dataiku").default_project_key()
        project = api_client.get_project(project_key)
        llm = project.get_llm(self.connection_id)
        completion = llm.new_completion()
        for msg in messages:
            completion.with_message(msg["content"], msg["role"])
        try:
            completion.with_max_output_tokens(self.max_tokens)
        except Exception:
            pass  # older Dataiku SDK versions may not support this
        resp = completion.execute()
        self.call_log.append({"messages": messages, "response": resp.text})
        return resp.text
    except Exception as e:
        logger.error("LLM Mesh call failed: %s", e)
        self.call_log.append({"messages": messages, "response": f"ERROR: {e}", "error": True})
        raise


# =========================================================================
# 2. Inline site-list-matching agent (workdir-independent)
# =========================================================================

_SITE_MATCHING_SYSTEM = """\
You are an expert clinical operations data specialist.
Your task is to semantically match an uploaded list of clinical trial sites against a master CTMS site database.

For each row in the uploaded file, determine whether it refers to the same real-world clinical site as any entry in the CTMS database.
Use semantic matching — consider site name variations, abbreviations, alternate spellings, country codes vs. full names, and PI name formatting differences.
A match requires confident identification of the same physical site; do not match on superficial similarity alone.

Return a JSON object:
{
  "matches": [
    {
      "uploaded_index": <int, 0-based row index in the uploaded file>,
      "uploaded_identifier": "<best identifying string from the uploaded row>",
      "ctms_site_id": "<site_id from CTMS, e.g. SP-001>",
      "ctms_site_name": "<site_name from CTMS>",
      "match_confidence": "high|medium|low",
      "match_basis": "<brief explanation of why this is a match>"
    }
  ],
  "unmatched_indices": [<list of 0-based row indices with no match>],
  "summary": {
    "total_uploaded": <int>,
    "matched": <int>,
    "unmatched": <int>,
    "notes": "<any overall observations about match quality or ambiguities>"
  }
}

Rules:
- Only include a row in "matches" if you are reasonably confident it corresponds to a CTMS site.
- Each uploaded row may match at most one CTMS site.
- Each CTMS site may match at most one uploaded row.
- All uploaded row indices not in "matches" must appear in "unmatched_indices".
- Return ONLY the JSON object, no markdown fences, no other text."""

_SITE_MATCHING_USER = """\
Uploaded site list ({n_uploaded} rows, CSV with index):
{uploaded_data}

CTMS master site database (CSV):
{ctms_data}

Match each uploaded row to the most appropriate CTMS site, or mark it as unmatched."""


def _find_ctms():
    candidates = [
        Path(os.getcwd()) / "data" / "CTMS_SITES.csv",
        Path(backend.__file__).parent.parent / "data" / "CTMS_SITES.csv",
        Path(backend.__file__).parent.parent.parent / "data" / "CTMS_SITES.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


class SiteListMatchingAgent:
    skill_id = "site_list_matching"
    display_name = "Clinical Site List Matching"

    def __init__(self, llm):
        self.llm = llm

    def run(self, params, state):
        MAX_ROWS = 200

        file_info = state.uploaded_files.get("site_file")
        if not file_info:
            return AgentResult(success=False, text_response="",
                               error_message="Missing uploaded file: Site List File.")

        ctms_path = _find_ctms()
        if not ctms_path:
            return AgentResult(success=False, text_response="",
                               error_message="CTMS_SITES.csv not found.")
        ctms_df = pd.read_csv(ctms_path)
        ctms_cols = [c for c in ["site_id", "site_name", "country", "city", "pi_name"]
                     if c in ctms_df.columns]
        ctms_text = ctms_df[ctms_cols].to_csv(index=False)

        uploaded_df = pd.DataFrame(file_info["data"])
        n_uploaded = len(uploaded_df)
        truncated = uploaded_df.head(MAX_ROWS).to_csv(index=True)
        if n_uploaded > MAX_ROWS:
            truncated += f"\n[Truncated to {MAX_ROWS} rows]"

        messages = [
            {"role": "system", "content": _SITE_MATCHING_SYSTEM},
            {"role": "user", "content": _SITE_MATCHING_USER.format(
                uploaded_data=truncated,
                ctms_data=ctms_text,
                n_uploaded=min(n_uploaded, MAX_ROWS),
            )},
        ]
        try:
            raw = self.llm.complete_json(messages, temperature=self.llm.temp_deterministic)
            matches = raw.get("matches", [])
            summary = raw.get("summary", {})
        except Exception as e:
            return AgentResult(success=False, text_response="",
                               error_message=f"Error during site matching: {e}")

        n_matched = summary.get("matched", len(matches))
        n_unmatched = summary.get("unmatched", n_uploaded - n_matched)
        match_rate = round(n_matched / max(n_uploaded, 1) * 100, 1)

        summary_text = (
            f"**Site List Matching Results**\n\n"
            f"Uploaded **{n_uploaded}** sites — compared against **{len(ctms_df)}** CTMS sites.\n\n"
            f"- Matched: **{n_matched}** ({match_rate}%)\n"
            f"- Unmatched: **{n_unmatched}**\n\n"
            f"{summary.get('notes', '')}".strip()
        )

        matched_by_idx = {m["uploaded_index"]: m for m in matches}
        table_data = []
        for i in range(min(n_uploaded, MAX_ROWS)):
            row = uploaded_df.iloc[i]
            ident = str(next((row.get(k) for k in
                              ["site_name", "name", "Site Name", "Site"] if row.get(k)), row.iloc[0]))
            m = matched_by_idx.get(i)
            table_data.append({
                "Row": i + 1,
                "Uploaded Site": ident,
                "Match Status": "Matched" if m else "Not matched",
                "CTMS Site ID": m.get("ctms_site_id", "") if m else "",
                "CTMS Site Name": m.get("ctms_site_name", "") if m else "",
                "Confidence": m.get("match_confidence", "") if m else "",
                "Match Basis": m.get("match_basis", "") if m else "",
            })

        return AgentResult(
            success=True,
            text_response=summary_text,
            table_data=table_data,
            table_columns=["Row", "Uploaded Site", "Match Status",
                           "CTMS Site ID", "CTMS Site Name", "Confidence", "Match Basis"],
        )


# =========================================================================
# 3. Shortened benchmarking system prompt
# =========================================================================

_SHORT_BENCHMARKING_SYSTEM = """\
You are an expert clinical development strategist. You will be given aggregated benchmark statistics from a Citeline trial database. Use these as the primary source of truth for numeric metrics; do not contradict them.

Return a JSON object — keep all string values concise (1 sentence each, max 2 sentences for benchmark_summary):
{
  "benchmark_summary": "<1-2 sentence summary of the data>",
  "key_metrics": {
    "median_enrollment_rate_patients_per_site_per_month": <float>,
    "median_dropout_rate_percent": <float>,
    "typical_duration_months": <int>,
    "typical_site_count_range": "<e.g. 50-150>",
    "typical_screen_failure_rate_percent": <float>
  },
  "notable_patterns": ["<max 3 short bullets>"],
  "key_challenges": ["<max 3 short bullets>"],
  "data_source": "<one sentence>",
  "caveats": "<one sentence>"
}

Return ONLY the JSON object, no markdown fences, no other text."""


# =========================================================================
# 4. General-knowledge fallback prompts
# =========================================================================

GENERAL_KNOWLEDGE_SYSTEM = """\
You are a knowledgeable clinical R&D assistant embedded in an analytics chatbot.
The user asked a question that does not match any of the chatbot's built-in analytical skills.
You have access to real-time web search — search results are included below when available. Use them together with your own knowledge to give the best possible answer.

Guidelines:
- Be concise and informative. Use markdown formatting for readability.
- You have web search capability. When web search results are included, incorporate and cite them to support your answer.
- Combine web search data with your own clinical R&D knowledge to provide a comprehensive response.
- If you are not confident in the answer, say so clearly.
- Do NOT pretend you ran an analytical tool or generated data — you are answering from general knowledge and web search only.
- At the end of your answer, briefly remind the user what analytical skills are available if their question could benefit from one."""

GENERAL_KNOWLEDGE_USER = """\
Conversation history:
{history}
{web_context}
---
User question: {user_message}

Answer the question using your general knowledge and any web search context above."""

SKILLS_REMINDER = """

---
*If your question relates to one of my analytical capabilities, I can do a deeper data-driven analysis:*
1. *Clinical Site List Matching*
2. *Trial Benchmarking*
3. *Drug Reimbursement Assessment*
4. *Enrollment & Site Activation Forecasting*
5. *Protocol Analysis*
6. *Country Ranking by Trial Experience*"""


def _handle_general_question(self, state, user_message, history):
    """Answer a general question using web search + LLM knowledge."""
    state.fsm_state = FSMState.IDLE

    history_text = "\n".join(
        f"{m['role'].capitalize()}: {m['content']}" for m in history[:-1]
    ) or "(no prior turns)"

    web_context = self.web_search.search(user_message)
    web_block = (
        f"\n---\nWeb search results:\n{web_context}\n"
        if web_context else ""
    )

    messages = [
        {"role": "system", "content": GENERAL_KNOWLEDGE_SYSTEM},
        {"role": "user", "content": GENERAL_KNOWLEDGE_USER.format(
            history=history_text,
            web_context=web_block,
            user_message=user_message,
        )},
    ]

    try:
        answer = self.llm.complete(messages, temperature=self.llm.temp_agents)
    except Exception as e:
        logger.error("General knowledge LLM call failed: %s", e)
        from backend.llm.prompt_templates import CLARIFICATION_MESSAGE
        return self._build_response(message=CLARIFICATION_MESSAGE, state=state)

    if self.llm.call_log:
        self.llm.call_log[-1]["label"] = "General Knowledge"

    answer = answer.strip() + SKILLS_REMINDER
    return self._build_response(message=answer, state=state)


# =========================================================================
# 5. Patched _route_fsm — general-knowledge fallback replaces clarification
# =========================================================================

def _patched_route_fsm(self, state, user_message, history):
    from backend.orchestrator.orchestrator import SHARED_PARAM_KEYS

    fsm = state.fsm_state

    # Let the original handle all non-IDLE states
    if fsm not in (FSMState.IDLE, FSMState.CLARIFICATION_REQUEST):
        return _patched_route_fsm._original(self, state, user_message, history)

    intent, confidence, reasoning = classify_intent(self.llm, user_message, history)

    if intent is None:
        intent = self._parse_skill_selection(user_message)

    if intent is None and state.prior_results:
        return self._generate_plan(state, user_message, history)

    if intent is None:
        return self._handle_general_question(state, user_message, history)

    if intent == "data_reasoning":
        if state.prior_results:
            return self._generate_plan(state, user_message, history)
        return self._handle_reasoning(state, user_message, history)

    state.active_skill = intent
    state.fsm_state = FSMState.PARAMETER_GATHERING

    inherited = state.get_shared_params(intent, SHARED_PARAM_KEYS)
    if inherited:
        state.merge_params(intent, inherited)

    extracted = extract_parameters(
        self.llm, self.schemas[intent], user_message, history
    )
    state.merge_params(intent, extracted)

    return self._check_and_confirm(state)


# =========================================================================
# Public entry point
# =========================================================================

def apply_patches(orchestrator):
    """Apply all notebook-specific patches to *orchestrator* (and its classes)."""
    from backend.orchestrator.orchestrator import Orchestrator

    # 1. LLMClient.complete with call logging
    LLMClient.complete = _patched_complete
    orchestrator.llm.call_log = []

    # 2. Site-list-matching agent (register under both legacy IDs)
    agent = SiteListMatchingAgent(orchestrator.llm)
    orchestrator.router._registry["site_list_matching"] = agent
    orchestrator.router._registry["site_list_merger"] = agent
    _schema = (orchestrator.schemas.get("site_list_matching")
               or orchestrator.schemas.get("site_list_merger"))
    if _schema:
        orchestrator.schemas["site_list_matching"] = _schema
        orchestrator.schemas["site_list_merger"] = _schema

    # Reload schemas from project root (bypasses Dataiku workdir cache)
    skills_cfg = Path(os.getcwd()) / "config" / "skills_config.yaml"
    if not skills_cfg.exists():
        skills_cfg = Path(backend.__file__).parent.parent / "config" / "skills_config.yaml"
    orchestrator.schemas = load_schemas(str(skills_cfg))

    # 3. Shortened benchmarking prompt
    _pt.TRIAL_BENCHMARKING_SYSTEM = _SHORT_BENCHMARKING_SYSTEM

    # 4. Raise confidence thresholds
    _ic.CONFIDENCE_THRESHOLD = 0.85
    _ic.DATA_REASONING_THRESHOLD = 0.75

    # 5. General-knowledge fallback
    Orchestrator._handle_general_question = _handle_general_question
    _patched_route_fsm._original = Orchestrator._route_fsm
    Orchestrator._route_fsm = _patched_route_fsm

    # Summary
    print(f"Router skills:  {list(orchestrator.router._registry.keys())}")
    ws = orchestrator.web_search
    print(f"Web search:     {'enabled' if ws.enabled else 'disabled'}")
    print(f"Schemas loaded: {list(orchestrator.schemas.keys())}")
    print(f"Intent thresholds: skill={_ic.CONFIDENCE_THRESHOLD}, "
          f"data_reasoning={_ic.DATA_REASONING_THRESHOLD}")
    print(f"LLM connection: {orchestrator.llm.connection_id}")
    print("All notebook patches applied.")
