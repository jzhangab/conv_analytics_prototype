"""
Panel-based chat UI for the Clinical Analytics notebook.

Call ``build_app(orchestrator, session_store)`` to get a servable Panel app.
"""
from __future__ import annotations

import io
import re
import uuid
from datetime import datetime

import pandas as pd
import panel as pn
from panel.chat import ChatInterface

from backend.agents.site_list_merger_agent import parse_uploaded_file
from backend.state.conversation_state import FSMState


# =========================================================================
# CSS / JS — widen chat message bubbles
# =========================================================================

_CHAT_WIDTH_HTML = """\
<style id="chat-width-override">
  [class*="chat"], [class*="Chat"],
  .bk-panel-models-chat-ChatInterface,
  .bk-panel-models-chat-ChatFeed,
  .bk-panel-models-chat-ChatMessage {
    max-width: 100% !important;
    width:     100% !important;
    box-sizing: border-box !important;
    flex-shrink: 0 !important;
  }
  [class*="chat"] .bk,
  [class*="chat"] .bk-root,
  [class*="chat"] .bk-Column,
  [class*="chat"] .bk-Row,
  [class*="chat"] > div,
  [class*="chat"] > div > div {
    max-width: 100% !important;
    width:     100% !important;
    box-sizing: border-box !important;
  }
  [class*="chat"] .bk-panel-models-markup-Markdown,
  [class*="chat"] .markdown,
  [class*="chat"] .pn-ChatMessage__bubble {
    max-width: 100% !important;
    width:     100% !important;
  }
</style>
<script>
(function () {
  var SELECTORS = [
    '[class*="chat"]', '[class*="Chat"]',
    '.bk-panel-models-chat-ChatInterface',
    '.bk-panel-models-chat-ChatFeed',
    '.bk-panel-models-chat-ChatMessage',
  ];
  function forceWide() {
    SELECTORS.forEach(function (sel) {
      document.querySelectorAll(sel).forEach(function (el) {
        el.style.setProperty('max-width', '100%', 'important');
        el.style.setProperty('width',     '100%', 'important');
        el.style.setProperty('box-sizing','border-box','important');
        Array.prototype.forEach.call(el.children, function (child) {
          child.style.setProperty('max-width', '100%', 'important');
          child.style.setProperty('width',     '100%', 'important');
        });
      });
    });
  }
  forceWide();
  var obs = new MutationObserver(forceWide);
  obs.observe(document.body, {
    childList: true, subtree: true,
    attributes: true, attributeFilter: ['style', 'class'],
  });
})();
</script>"""


# =========================================================================
# LLM trace log
# =========================================================================

def _infer_call_label(messages):
    system = next((m["content"] for m in messages if m["role"] == "system"), "").lower()
    if system.startswith("[citeline semantic mapping]"):  return "Citeline Semantic Mapping"
    if system.startswith("[citeline filter result]"):     return "Citeline Filter Result"
    if "senior clinical r&d strategist" in system or "data_reasoning" in system:
        return "Data Reasoning"
    if "senior clinical research expert" in system or "protocol_analysis" in system or "gcp" in system:
        return "Protocol Analysis"
    if "routing assistant" in system or ("intent" in system and "skill" in system):
        return "Intent Classification"
    if "parameter extraction" in system: return "Parameter Extraction"
    if "site list" in system or "ctms" in system: return "Site List Matching Agent"
    if "benchmarking" in system or "benchmark" in system: return "Trial Benchmarking Agent"
    if "reimbursement" in system or "hta" in system: return "Drug Reimbursement Agent"
    if "country ranking" in system or "country feasibility" in system: return "Country Ranking Agent"
    if "enrollment" in system and ("pessimistic" in system or "scenario" in system):
        return "Enrollment Params Estimation"
    if "narrative" in system or ("enrollment" in system and "interpret" in system):
        return "Enrollment Narrative"
    if system.startswith("["):  return system.strip("[]").title()
    return "LLM Call"


def _make_trace_updater(orchestrator, trace_content):
    """Return a callback that refreshes the trace pane from the LLM call log."""
    def _update_trace_log():
        log = getattr(orchestrator.llm, "call_log", [])
        conn_id = getattr(orchestrator.llm, "connection_id", "unknown")
        if not log:
            trace_content.object = (
                f"Connection ID in use: {conn_id}\n\n"
                "No LLM calls recorded yet.\n"
                "If this stays empty after sending a message, the LLM call is failing\n"
                "before it can be logged — check Cell 3 output for import errors."
            )
            return
        entries = []
        for i, entry in enumerate(log, 1):
            msgs = entry.get("messages", [])
            resp_text = entry.get("response", "")
            is_error = entry.get("error", False)
            is_synth = entry.get("synthetic", False)
            label = entry.get("label") or _infer_call_label(msgs)
            if is_synth:
                lines = [f"=== {i}: {label} ==="]
                lines.append(resp_text)
            elif is_error:
                lines = [f"!!! ERROR Call {i}: {label} (conn={conn_id}) !!!"]
                for msg in msgs:
                    role = msg.get("role", "user").upper()
                    content = msg.get("content", "")[:600]
                    lines.append(f"[{role}]\n{content}")
                lines.append(f"[RESPONSE]\n{resp_text[:800]}")
            else:
                lines = [f"--- Call {i}: {label} (conn={conn_id}) ---"]
                for msg in msgs:
                    role = msg.get("role", "user").upper()
                    content = msg.get("content", "")[:600]
                    content += ("...[truncated]" if len(msg.get("content", "")) > 600 else "")
                    lines.append(f"[{role}]\n{content}")
                resp_trunc = resp_text[:800] + ("...[truncated]" if len(resp_text) > 800 else "")
                lines.append(f"[RESPONSE]\n{resp_trunc}")
            entries.append("\n".join(lines))
        trace_content.object = "\n\n".join(entries)
    return _update_trace_log


# =========================================================================
# Response → Panel widgets
# =========================================================================

def _make_response_renderer(orchestrator, session_id, chat, maybe_show_export):
    """Return a function that converts a response dict to Panel widgets."""

    def _response_to_panel(resp):
        items = []

        if resp.get("message"):
            items.append(pn.pane.Markdown(resp["message"], sizing_mode="stretch_width"))

        if resp.get("table_data"):
            rows = resp["table_data"]
            columns = resp.get("table_columns")
            df = pd.DataFrame(rows, columns=columns) if columns else pd.DataFrame(rows)
            items.append(pn.pane.DataFrame(
                df, sizing_mode="stretch_width", max_rows=200,
                styles={"font-size": "12px"},
            ))

        if resp.get("chart_json"):
            _chart = resp["chart_json"]
            if isinstance(_chart, dict):
                items.append(pn.pane.Markdown("*Chart unavailable.*"))
            else:
                items.append(pn.pane.Bokeh(_chart, sizing_mode="stretch_width"))

        if resp.get("fsm_state") == "confirmation_pending":
            yes_btn = pn.widgets.Button(
                name="\u2713  Yes, proceed", button_type="success",
                width=160, margin=(8, 6, 4, 0),
            )
            no_btn = pn.widgets.Button(
                name="\u2717  Cancel", button_type="danger",
                width=110, margin=(8, 0, 4, 0),
            )

            def _confirm(event):
                yes_btn.disabled = True
                no_btn.disabled = True
                r = orchestrator.handle_confirmation(session_id, confirmed=True)
                chat.send(_response_to_panel(r), user="Assistant", respond=False)
                maybe_show_export(r)

            def _cancel(event):
                yes_btn.disabled = True
                no_btn.disabled = True
                r = orchestrator.handle_confirmation(session_id, confirmed=False)
                chat.send(_response_to_panel(r), user="Assistant", respond=False)

            yes_btn.on_click(_confirm)
            no_btn.on_click(_cancel)
            items.append(pn.Row(yes_btn, no_btn, margin=(4, 0, 0, 0)))

        if resp.get("result_id"):
            maybe_show_export(resp)

        return pn.Column(*items, sizing_mode="stretch_width") if items else pn.pane.Markdown("...")

    return _response_to_panel


# =========================================================================
# Protocol PDF generation
# =========================================================================

def generate_protocol_pdf(result):
    """Build a formatted PDF from a protocol analysis result dict."""
    if result is None:
        return io.BytesIO(b"No protocol analysis result available.")
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.lib import colors
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable,
        )

        buf = io.BytesIO()
        fn = result.get("filename", "protocol")
        doc = SimpleDocTemplate(
            buf, pagesize=A4,
            leftMargin=2 * cm, rightMargin=2 * cm,
            topMargin=2.5 * cm, bottomMargin=2 * cm,
            title=f"Protocol Analysis — {fn}",
        )

        base = getSampleStyleSheet()
        title_s = ParagraphStyle("PATitle", parent=base["Title"],
                                 fontSize=18, textColor=colors.HexColor("#1a1a2e"),
                                 spaceAfter=4)
        sub_s = ParagraphStyle("PASub", parent=base["Normal"],
                               fontSize=10, textColor=colors.HexColor("#555"),
                               spaceAfter=12)
        h1_s = ParagraphStyle("PAH1", parent=base["Heading1"],
                              fontSize=13, textColor=colors.HexColor("#1a1a2e"),
                              spaceBefore=14, spaceAfter=6)
        h2_s = ParagraphStyle("PAH2", parent=base["Heading2"],
                              fontSize=11, textColor=colors.HexColor("#333"),
                              spaceBefore=10, spaceAfter=4)
        body_s = ParagraphStyle("PABody", parent=base["Normal"],
                                fontSize=10, leading=14, spaceAfter=3)
        bullet_s = ParagraphStyle("PABullet", parent=base["Normal"],
                                  fontSize=10, leading=14, spaceAfter=2,
                                  leftIndent=14, bulletIndent=4)

        def _md(text):
            return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)

        elements = []
        elements.append(Paragraph("Clinical Trial Protocol Analysis Report", title_s))
        elements.append(Paragraph(
            f"Protocol: <b>{fn}</b> &nbsp;&nbsp;&nbsp; "
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            sub_s,
        ))
        elements.append(HRFlowable(width="100%", thickness=2,
                                   color=colors.HexColor("#1a1a2e"), spaceAfter=10))

        for line in result.get("text", "").splitlines():
            s = line.strip()
            if not s:
                elements.append(Spacer(1, 4))
            elif s.startswith("## "):
                elements.append(Paragraph(_md(s[3:]), h1_s))
            elif s.startswith("### "):
                elements.append(Paragraph(_md(s[4:]), h2_s))
            elif s.startswith("- "):
                elements.append(Paragraph(f"\u2022 {_md(s[2:])}", bullet_s))
            else:
                elements.append(Paragraph(_md(s), body_s))

        tdata = result.get("table_data")
        tcols = result.get("table_columns")
        if tdata and tcols:
            elements.append(Spacer(1, 12))
            elements.append(Paragraph("Findings", h1_s))
            elements.append(HRFlowable(width="100%", thickness=1,
                                       color=colors.HexColor("#ccc"), spaceAfter=6))

            SEV_COLORS = {
                "Critical": colors.HexColor("#ffe0e0"),
                "Major": colors.HexColor("#fff3cd"),
                "Minor": colors.HexColor("#fffde0"),
                "Suggestion": colors.HexColor("#e8f5e9"),
            }
            PRESET = {"#": 0.05, "Section": 0.14, "Severity": 0.10,
                      "Finding": 0.33, "Recommendation": 0.33}
            total_w = 17 * cm
            col_ws = [PRESET.get(c, 1 / len(tcols)) * total_w for c in tcols]
            factor = total_w / sum(col_ws)
            col_ws = [w * factor for w in col_ws]

            hdr = [Paragraph(f"<b>{c}</b>", body_s) for c in tcols]
            rows = [hdr]
            row_bg_styles = []
            for ri, row in enumerate(tdata, 1):
                sev = str(row.get("Severity", "")).title()
                cells = [Paragraph(_md(str(row.get(c, ""))), body_s) for c in tcols]
                rows.append(cells)
                bg = SEV_COLORS.get(sev)
                if bg:
                    row_bg_styles.append(("BACKGROUND", (0, ri), (-1, ri), bg))

            tbl = Table(rows, colWidths=col_ws, repeatRows=1)
            tbl.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a1a2e")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#ccc")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ] + row_bg_styles))
            elements.append(tbl)

        doc.build(elements)
        buf.seek(0)
        return buf

    except ImportError:
        txt = "Protocol Analysis Report\n" + "=" * 40 + "\n\n"
        txt += result.get("text", "")
        return io.BytesIO(txt.encode("utf-8"))
    except Exception as e:
        return io.BytesIO(f"PDF generation error: {e}".encode())


# =========================================================================
# Fake file-storage adapter (Panel FileInput → parse_uploaded_file)
# =========================================================================

class _FakeFileStorage:
    def __init__(self, filename, data):
        self.filename = filename
        self._data = data

    def read(self):
        return self._data


# =========================================================================
# build_app — the single public entry point
# =========================================================================

def build_app(orchestrator, session_store):
    """Construct and return the full Panel app layout (servable)."""
    session_id = str(uuid.uuid4())

    # -- Widgets: CSS injection -------------------------------------------------
    chat_width_css = pn.pane.HTML(_CHAT_WIDTH_HTML, width=0, height=0, margin=0,
                                  sizing_mode="fixed")

    # -- Widgets: trace pane ----------------------------------------------------
    trace_content = pn.pane.Markdown(
        "*No LLM calls yet — send a message to see the trace.*",
        sizing_mode="stretch_width",
        styles={"font-size": "11px", "white-space": "pre-wrap",
                "font-family": "monospace", "color": "#222"},
    )
    update_trace_log = _make_trace_updater(orchestrator, trace_content)

    # -- State: export / protocol result ----------------------------------------
    _state = {"result_id": None, "table_data": None, "table_columns": None,
              "protocol_result": None}

    # -- Widgets: export bar ----------------------------------------------------
    export_input = pn.widgets.TextInput(placeholder="Dataiku dataset name\u2026", width=260)
    export_button = pn.widgets.Button(name="\u2b07  Export to Dataiku Dataset",
                                      button_type="primary", width=220)
    export_status = pn.pane.Markdown("", width=400)
    export_row = pn.Column(
        pn.pane.Markdown("**Export last result:**", margin=(0, 0, 4, 0)),
        pn.Row(export_input, export_button, export_status),
        visible=False,
        styles={"background": "#f0f4ff", "padding": "8px", "border-radius": "6px"},
    )

    # -- Widgets: protocol PDF download -----------------------------------------
    proto_pdf_btn = pn.widgets.FileDownload(
        label="\u2b07  Download Analysis PDF",
        button_type="primary",
        filename="protocol_analysis.pdf",
        callback=lambda: generate_protocol_pdf(_state["protocol_result"]),
        embed=False, width=230, margin=(0, 0, 0, 0),
    )
    proto_pdf_row = pn.Row(
        pn.pane.Markdown("\U0001f4c4 **Protocol Analysis ready:**",
                         margin=(6, 10, 0, 0),
                         styles={"font-size": "13px", "white-space": "nowrap"}),
        proto_pdf_btn,
        visible=False, sizing_mode="stretch_width",
        styles={"background": "#e8f5e9", "padding": "8px 12px",
                "border-bottom": "1px solid #c8e6c9"},
        align="center",
    )

    def _maybe_show_export(resp):
        if resp.get("result_id"):
            _state["result_id"] = resp["result_id"]
        if resp.get("table_data"):
            _state["table_data"] = resp["table_data"]
            _state["table_columns"] = resp.get("table_columns")
            export_row.visible = True
        if resp.get("skill_id") == "protocol_analysis":
            st = session_store.get_or_create(session_id)
            file_info = st.uploaded_files.get("protocol_file", {})
            _state["protocol_result"] = {
                "text": resp.get("message", ""),
                "table_data": resp.get("table_data"),
                "table_columns": resp.get("table_columns"),
                "filename": file_info.get("filename", "protocol"),
            }
            proto_pdf_btn.filename = (
                file_info.get("filename", "protocol").rsplit(".", 1)[0] + "_analysis.pdf"
            )
            proto_pdf_row.visible = True

    def _on_export(event):
        dataset_name = export_input.value.strip()
        if not dataset_name:
            export_status.object = "\u26a0 Enter a dataset name first."
            return
        if not _state["table_data"]:
            export_status.object = "\u26a0 No table result to export yet."
            return
        try:
            import dataiku
            ds = dataiku.Dataset(dataset_name)
            df = (pd.DataFrame(_state["table_data"], columns=_state["table_columns"])
                  if _state["table_columns"] else pd.DataFrame(_state["table_data"]))
            ds.write_with_schema(df)
            export_status.object = f"\u2713 Exported {len(df)} rows to **{dataset_name}**."
        except Exception as exc:
            export_status.object = f"\u26a0 Export failed: {exc}"

    export_button.on_click(_on_export)

    # -- Chat interface ---------------------------------------------------------
    # Forward-declare so _response_to_panel can reference it
    chat = ChatInterface(
        callback=None,  # set below
        user="You", avatar="\U0001f464", callback_user="Assistant",
        show_rerun=False, show_undo=False, show_copy_icon=False,
        placeholder_text="Describe what you need\u2026",
        sizing_mode="stretch_width", height=720,
        styles={"border": "1px solid #e0e0e0", "border-radius": "8px"},
    )

    _response_to_panel = _make_response_renderer(
        orchestrator, session_id, chat, _maybe_show_export,
    )

    def chat_callback(contents, user, instance):
        live_cb = pn.state.add_periodic_callback(update_trace_log, period=2000)
        try:
            resp = orchestrator.process_message(session_id, contents)
        except Exception as exc:
            import traceback
            tb = traceback.format_exc()
            trace_content.object = f"EXCEPTION in process_message:\n{tb}"
            return pn.pane.Markdown(f"**Error:** `{exc}`  \nSee trace pane for full traceback.")
        finally:
            live_cb.stop()
        _maybe_show_export(resp)
        update_trace_log()
        return _response_to_panel(resp)

    chat.callback = chat_callback

    # -- Welcome message --------------------------------------------------------
    chat.send(
        pn.pane.Markdown(
            "Hello! I'm your **Clinical Analytics Assistant**. I can help you with:\n\n"
            "1. **Site List Matching** \u2014 Upload a site list and match it against the "
            "CTMS master database  \n"
            "2. **Trial Benchmarking** \u2014 Benchmark trials by indication, age group, "
            "and phase  \n"
            "3. **Drug Reimbursement** \u2014 Assess reimbursement outlook by country  \n"
            "4. **Enrollment Forecasting** \u2014 Generate pessimistic / moderate / optimistic "
            "enrollment curves  \n"
            "5. **Protocol Analysis** \u2014 Upload a protocol (PDF, DOCX, TXT) for a study "
            "design review  \n"
            "6. **Country Ranking** \u2014 Rank countries by trial experience for a given "
            "indication  \n\n"
            "After running any tool, you can ask me follow-up questions like:\n"
            "- *Based on this enrollment forecast, what is the best study design?*  \n"
            "- *What are the key risks given these benchmarks?*  \n"
            "- *Which country has the best reimbursement outlook and why?*  \n\n"
            "What would you like to do?"
        ),
        user="Assistant", respond=False,
    )

    # -- File upload: CRO site list ---------------------------------------------
    upload_cro = pn.widgets.FileInput(accept=".csv,.xlsx,.xls", width=260)
    upload_status = pn.pane.Markdown(
        "", sizing_mode="stretch_width",
        styles={"font-size": "12px", "color": "#444"},
    )

    def _on_cro_upload(event):
        if upload_cro.value is None:
            return
        filename = upload_cro.filename or "upload"
        try:
            parsed = parse_uploaded_file(_FakeFileStorage(filename, upload_cro.value))
            state = session_store.get_or_create(session_id)
            state.uploaded_files["site_file"] = parsed
            state.active_skill = "site_list_matching"
            state.fsm_state = FSMState.PARAMETER_GATHERING
            upload_status.object = (
                f"\u2705 **CRO file loaded:** {filename} \u2014 {len(parsed['data'])} rows, "
                f"columns: {', '.join(parsed['columns'])}"
            )
            chat.send(
                f"Site list uploaded: **{filename}** ({len(parsed['data'])} rows). "
                "Type \"match\" or \"proceed\" to run matching against the CTMS database.",
                user="Assistant", respond=False,
            )
        except ValueError as e:
            upload_status.object = f"\u26a0 **Upload error:** {e}"

    upload_cro.param.watch(_on_cro_upload, "value")

    # -- File upload: protocol --------------------------------------------------
    protocol_upload_widget = pn.widgets.FileInput(accept=".pdf,.docx,.txt", width=260)
    protocol_upload_status = pn.pane.Markdown(
        "", sizing_mode="stretch_width",
        styles={"font-size": "12px", "color": "#444"},
    )

    def _on_protocol_upload(event):
        if protocol_upload_widget.value is None:
            return
        filename = protocol_upload_widget.filename or "protocol"
        try:
            resp = orchestrator.handle_file_upload(
                session_id, "protocol_file",
                _FakeFileStorage(filename, protocol_upload_widget.value),
            )
            if resp.get("error"):
                protocol_upload_status.object = f"\u26a0 **Upload error:** {resp['error']}"
            else:
                file_info = (
                    session_store.get_or_create(session_id)
                    .uploaded_files.get("protocol_file", {})
                )
                fmt = file_info.get("format", "txt")
                if fmt == "pdf":
                    size_desc = f"{file_info.get('total_pages', '?')} pages"
                else:
                    size_desc = f"{len(file_info.get('full_text', '')):,} characters"
                protocol_upload_status.object = f"\u2705 **{filename}** \u2014 {size_desc}"
                chat.send(
                    resp.get("message", f"Protocol uploaded: **{filename}**."),
                    user="Assistant", respond=False,
                )
        except Exception as e:
            protocol_upload_status.object = f"\u26a0 **Upload error:** {e}"

    protocol_upload_widget.param.watch(_on_protocol_upload, "value")

    # -- Upload bars ------------------------------------------------------------
    upload_bar = pn.Row(
        pn.pane.Markdown(
            "\U0001f4c2 **CRO Site List** *(for Site List Merger)*",
            margin=(6, 10, 0, 0),
            styles={"font-size": "13px", "white-space": "nowrap"},
        ),
        upload_cro, upload_status,
        sizing_mode="stretch_width",
        styles={"background": "#f9f9f9", "padding": "8px 12px",
                "border-bottom": "1px solid #e0e0e0"},
        align="center",
    )
    protocol_upload_bar = pn.Row(
        pn.pane.Markdown(
            "\U0001f4c4 **Protocol** *(for Protocol Analysis)*",
            margin=(6, 10, 0, 0),
            styles={"font-size": "13px", "white-space": "nowrap"},
        ),
        protocol_upload_widget, protocol_upload_status,
        sizing_mode="stretch_width",
        styles={"background": "#f9f9f9", "padding": "8px 12px",
                "border-bottom": "1px solid #e0e0e0"},
        align="center",
    )

    # -- Layout -----------------------------------------------------------------
    app = pn.Column(
        pn.pane.Markdown(
            "# Clinical Analytics Assistant\n*Drug R&D | Dataiku Notebook*",
            styles={"background": "#1a1a2e", "color": "white",
                    "padding": "14px 18px", "border-radius": "10px 10px 0 0",
                    "margin": "0"},
        ),
        upload_bar,
        protocol_upload_bar,
        chat_width_css,
        chat,
        export_row,
        proto_pdf_row,
        pn.Column(
            pn.pane.Markdown("### LLM Call Trace", margin=(0, 0, 4, 0),
                             styles={"color": "#222"}),
            trace_content,
            sizing_mode="stretch_width", height=320, scroll=True,
            styles={"background": "#f7f7f7", "padding": "10px",
                    "border": "1px solid #ddd", "border-radius": "6px"},
        ),
        sizing_mode="stretch_width",
        styles={"border": "1px solid #ccc", "border-radius": "10px",
                "overflow": "hidden"},
    )

    return app
